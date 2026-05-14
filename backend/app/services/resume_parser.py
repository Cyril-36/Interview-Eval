"""
Resume parsing: PDF → plain text → extracted skill keyphrases.

Skill extraction reuses KeyBERT (already loaded in the model registry) and
filters against a curated tech-and-soft-skill vocabulary so that interviewer
question prompts get high-signal context, not arbitrary resume noise.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings
from app.models_loader import get_registry

logger = logging.getLogger(__name__)


class ResumeParseError(ValueError):
    """Raised when a resume file cannot be parsed into usable text."""


_WHITESPACE_RUN = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


# Domain vocabulary used as a soft filter on KeyBERT output. Lowercase.
# Not exhaustive — KeyBERT can still return useful phrases outside this set,
# which we keep when they look like skill-shaped tokens.
_SKILL_VOCAB = (
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "bash", "shell", "lua", "perl", "haskell", "elixir", "objective-c",
    # Frontend
    "react", "angular", "vue", "svelte", "next.js", "nextjs", "nuxt", "redux",
    "html", "css", "sass", "tailwind", "tailwindcss", "bootstrap", "webpack",
    "vite", "jest", "cypress", "playwright",
    # Backend / frameworks
    "fastapi", "flask", "django", "express", "nestjs", "spring", "spring boot",
    "rails", "laravel", "graphql", "rest", "grpc", "websocket",
    # Data / ML
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "scipy", "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake",
    "databricks", "huggingface", "transformers", "langchain", "llamaindex",
    "rag", "nlp", "computer vision", "deep learning", "machine learning",
    "reinforcement learning", "mlops",
    # Cloud / DevOps
    "aws", "gcp", "azure", "kubernetes", "docker", "terraform", "ansible",
    "jenkins", "github actions", "circleci", "argocd", "helm", "prometheus",
    "grafana", "datadog", "elk", "splunk",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "neo4j", "clickhouse",
    # Concepts
    "microservices", "distributed systems", "system design", "ci/cd",
    "testing", "tdd", "agile", "scrum", "code review", "design patterns",
    "data structures", "algorithms", "concurrency", "multithreading",
    "asynchronous", "event-driven", "api design", "security", "oauth",
    "authentication", "authorization", "encryption",
    # Soft / behavioral
    "leadership", "mentoring", "collaboration", "communication",
    "problem solving", "project management", "stakeholder management",
    "cross-functional",
)


@dataclass
class ParsedResume:
    text: str
    skills: list[str]
    page_count: int
    truncated: bool

    def summary(self, max_chars: int = 600) -> str:
        """First-paragraph-ish summary suitable for prompt injection."""
        snippet = self.text.strip()[:max_chars]
        # Cut at the last sentence boundary for cleanliness.
        last_period = snippet.rfind(".")
        if last_period > max_chars // 2:
            snippet = snippet[: last_period + 1]
        return snippet


def parse_pdf(
    file_bytes: bytes,
    max_bytes: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> ParsedResume:
    """Extract clean text + skill keyphrases from a PDF blob."""
    settings = get_settings()
    max_bytes = max_bytes if max_bytes is not None else settings.RESUME_MAX_BYTES
    max_chars = max_chars if max_chars is not None else settings.RESUME_MAX_TEXT_CHARS

    if not file_bytes:
        raise ResumeParseError("Empty file")
    if len(file_bytes) > max_bytes:
        raise ResumeParseError(
            f"File too large ({len(file_bytes)} bytes, limit {max_bytes})"
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except (PdfReadError, OSError, ValueError) as e:
        raise ResumeParseError(f"Could not read PDF: {e}") from e

    if getattr(reader, "is_encrypted", False):
        raise ResumeParseError("Encrypted PDFs are not supported")

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:  # pypdf can raise a variety of errors per page
            logger.warning(f"Skipping unreadable page: {e}")
            pages_text.append("")

    raw_text = "\n\n".join(pages_text)
    cleaned = _clean_text(raw_text)
    if not cleaned.strip():
        raise ResumeParseError("No extractable text found in PDF")

    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = cleaned[:max_chars]

    skills = extract_skills(cleaned, top_n=settings.RESUME_SKILL_COUNT)
    return ParsedResume(
        text=cleaned,
        skills=skills,
        page_count=len(reader.pages),
        truncated=truncated,
    )


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    # Strip empty lines on each line
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def extract_skills(text: str, top_n: int = 12) -> list[str]:
    """
    Return ranked skill phrases. Prefers vocabulary matches (high precision),
    backfills with KeyBERT keyphrases when needed (higher recall).
    """
    if not text.strip():
        return []

    lower = text.lower()
    vocab_hits: list[str] = []
    seen: set[str] = set()
    for skill in _SKILL_VOCAB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower) and skill not in seen:
            vocab_hits.append(skill)
            seen.add(skill)

    if len(vocab_hits) >= top_n:
        return vocab_hits[:top_n]

    # Backfill with KeyBERT keyphrases the vocabulary missed.
    try:
        keybert = get_registry().keybert
        keyphrases = keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n * 2,
        )
    except Exception as e:
        logger.warning(f"KeyBERT skill extraction failed: {e}")
        return vocab_hits[:top_n]

    for phrase, _score in keyphrases:
        normalized = phrase.lower().strip()
        if not normalized or normalized in seen:
            continue
        if len(normalized) < 3 or len(normalized) > 40:
            continue
        # Skip phrases that look like names / generic verbs.
        if not re.search(r"[a-z]", normalized):
            continue
        vocab_hits.append(normalized)
        seen.add(normalized)
        if len(vocab_hits) >= top_n:
            break

    return vocab_hits[:top_n]
