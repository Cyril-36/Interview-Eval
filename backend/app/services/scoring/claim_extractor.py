import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)

CLAUSE_SPLIT_RE = re.compile(
    r"\s*;\s*|\s*:\s*|,\s+(?=(?:and|but|while|because|so)\b)",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
EXAMPLE_MARKER_RE = re.compile(
    r"\b(?:such as|including|for example|for instance|e\.g\.|like)\b",
    re.IGNORECASE,
)
CORE = "core"
OPTIONAL = "optional"
OPTIONAL_IMPORTANCE_VALUES = {
    "example",
    "examples",
    "optional",
    "supporting",
    "nice_to_have",
    "nice-to-have",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with",
}

LLM_CLAIM_PROMPT = """Extract up to {max_claims} atomic claims from the ideal answer.

Question: {question}

Ideal Answer:
{ideal_answer}

Requirements:
- Return short standalone claims, each describing one key fact, concept, or action.
- Keep claims concise and specific.
- Do not repeat the same idea in different wording.
- Ignore filler, framing, and motivational phrases.
- Mark claims as "core" when the candidate should cover the idea.
- Mark claims as "optional" when the claim is only an example, named tool,
  database, library, framework, or implementation option for a broader idea.
- Do not make examples mandatory. If the answer says "use a database like
  MongoDB or Cassandra", make "use a database" core and the specific named
  databases optional.

Respond with ONLY a JSON array of objects. Example:
[
  {{"claim": "Use a database to store user interactions", "importance": "core"}},
  {{"claim": "MongoDB or Cassandra are possible database examples", "importance": "optional"}}
]
"""


@dataclass(frozen=True)
class ExtractedClaim:
    text: str
    importance: str = CORE


def _signature(text: str) -> tuple[str, ...]:
    tokens = [
        token.lower()
        for token in WORD_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def _coerce_importance(value: object) -> str:
    if isinstance(value, str) and value.strip().lower() in OPTIONAL_IMPORTANCE_VALUES:
        return OPTIONAL
    return CORE


def _clean_claim_text(text: str) -> str:
    return text.strip(" \n\t-.,;:")


def _split_example_claim(text: str) -> list[ExtractedClaim]:
    cleaned = _clean_claim_text(text)
    if not cleaned:
        return []

    marker = EXAMPLE_MARKER_RE.search(cleaned)
    if not marker:
        return [ExtractedClaim(cleaned, CORE)]

    prefix = _clean_claim_text(cleaned[: marker.start()])
    suffix = _clean_claim_text(cleaned[marker.end():])
    claims: list[ExtractedClaim] = []

    if len(prefix.split()) >= 5:
        claims.append(ExtractedClaim(prefix, CORE))

    if suffix:
        optional_text = f"Examples include {suffix}"
        if len(optional_text.split()) >= 3:
            claims.append(ExtractedClaim(optional_text, OPTIONAL))

    if claims:
        return claims
    return [ExtractedClaim(cleaned, OPTIONAL)]


def _dedupe_claim_specs(parts: list[ExtractedClaim], max_claims: int) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    signatures: list[set[str]] = []

    for part in parts:
        signature = set(_signature(part.text))
        if len(signature) < 2:
            continue

        duplicate = False
        for existing in signatures:
            overlap = len(signature & existing) / max(len(signature | existing), 1)
            if overlap >= 0.8:
                duplicate = True
                break
        if duplicate:
            continue

        signatures.append(signature)
        claims.append(part)

        if len(claims) >= max_claims:
            break

    return claims


def _dedupe_claims(parts: list[str], max_claims: int) -> list[str]:
    specs = [ExtractedClaim(_clean_claim_text(part), CORE) for part in parts]
    return [claim.text for claim in _dedupe_claim_specs(specs, max_claims)]


def _regex_extract_claim_specs(
    ideal_answer: str, max_claims: int = 6,
) -> list[ExtractedClaim]:
    text = ideal_answer.strip()
    if not text:
        return []

    raw_specs: list[ExtractedClaim] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip(" \n\t-")
        if not sentence:
            continue
        for part in CLAUSE_SPLIT_RE.split(sentence):
            cleaned = part.strip(" ,.;:-")
            if len(cleaned.split()) >= 4:
                raw_specs.extend(_split_example_claim(cleaned))

    if not raw_specs:
        raw_specs = [ExtractedClaim(text, CORE)]

    claims = _dedupe_claim_specs(raw_specs, max_claims)
    return claims or [ExtractedClaim(text, CORE)]


def _regex_extract_claims(ideal_answer: str, max_claims: int = 6) -> list[str]:
    return [
        claim.text
        for claim in _regex_extract_claim_specs(ideal_answer, max_claims=max_claims)
    ]


def _coerce_claim_specs(data: object, max_claims: int) -> list[ExtractedClaim]:
    if not isinstance(data, list):
        return []

    specs: list[ExtractedClaim] = []
    for item in data:
        if isinstance(item, ExtractedClaim):
            specs.append(item)
            continue

        if isinstance(item, str):
            claim = _clean_claim_text(item)
            if len(claim.split()) >= 3:
                specs.append(ExtractedClaim(claim, CORE))
            continue

        if not isinstance(item, dict):
            continue
        raw_claim = item.get("claim") or item.get("text")
        if not isinstance(raw_claim, str):
            continue
        claim = _clean_claim_text(raw_claim)
        if len(claim.split()) >= 3:
            specs.append(
                ExtractedClaim(
                    claim,
                    _coerce_importance(item.get("importance") or item.get("type")),
                )
            )

    return _dedupe_claim_specs(specs, max_claims)


def _parse_llm_claims(response_text: str, max_claims: int) -> list[ExtractedClaim]:
    text = response_text.strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_ARRAY_RE.search(text)
        if not match:
            lines = [
                line.strip(" -*0123456789.()\t")
                for line in text.splitlines()
                if line.strip()
            ]
            return _dedupe_claim_specs(
                [ExtractedClaim(line, CORE) for line in lines],
                max_claims,
            )
        data = json.loads(match.group(0))

    return _coerce_claim_specs(data, max_claims)


@lru_cache(maxsize=512)
def _extract_claims_with_llm(
    question: str,
    ideal_answer: str,
    max_claims: int,
    model: str,
) -> tuple[ExtractedClaim, ...]:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        return ()

    prompt = LLM_CLAIM_PROMPT.format(
        question=question or "N/A",
        ideal_answer=ideal_answer,
        max_claims=max_claims,
    )
    client = Groq(api_key=settings.GROQ_API_KEY, timeout=15)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    return tuple(_parse_llm_claims(content, max_claims))


def extract_claim_specs(
    ideal_answer: str,
    question: str = "",
    max_claims: int = 6,
) -> list[ExtractedClaim]:
    """
    Break an ideal answer into short atomic claims with importance metadata.

    Core claims are required for coverage; optional claims are examples or
    named implementation choices that can add credit but should not be listed
    as missing concepts when absent.
    """
    settings = get_settings()
    fallback = _regex_extract_claim_specs(ideal_answer, max_claims=max_claims)

    if settings.CLAIM_EXTRACTION_MODE != "llm":
        return fallback
    if not settings.GROQ_API_KEY:
        return fallback

    try:
        claims = _coerce_claim_specs(
            list(
                _extract_claims_with_llm(
                    question.strip(),
                    ideal_answer.strip(),
                    max_claims,
                    settings.CLAIM_EXTRACTION_MODEL,
                )
            ),
            max_claims,
        )
        return claims or fallback
    except Exception as exc:
        logger.warning("LLM claim extraction failed, falling back to regex: %s", exc)
        return fallback


def extract_claims(
    ideal_answer: str,
    question: str = "",
    max_claims: int = 6,
) -> list[str]:
    """
    Backward-compatible wrapper that returns claim text only.
    """
    return [
        claim.text
        for claim in extract_claim_specs(
            ideal_answer,
            question=question,
            max_claims=max_claims,
        )
    ]
