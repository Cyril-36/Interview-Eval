import re
import json
import logging
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
- Prefer claims that would help evaluate whether a candidate covered the important points.

Respond with ONLY a JSON array of strings. Example:
["Claim one", "Claim two"]
"""


def _signature(text: str) -> tuple[str, ...]:
    tokens = [
        token.lower()
        for token in WORD_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def _dedupe_claims(parts: list[str], max_claims: int) -> list[str]:
    claims: list[str] = []
    signatures: list[set[str]] = []

    for part in parts:
        signature = set(_signature(part))
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


def _regex_extract_claims(ideal_answer: str, max_claims: int = 6) -> list[str]:
    text = ideal_answer.strip()
    if not text:
        return []

    raw_parts: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip(" \n\t-")
        if not sentence:
            continue
        for part in CLAUSE_SPLIT_RE.split(sentence):
            cleaned = part.strip(" ,.;:-")
            if len(cleaned.split()) >= 4:
                raw_parts.append(cleaned)

    if not raw_parts:
        raw_parts = [text]

    claims = _dedupe_claims(raw_parts, max_claims)
    return claims or [text]


def _parse_llm_claims(response_text: str, max_claims: int) -> list[str]:
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
            return _dedupe_claims(lines, max_claims)
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        return []

    cleaned = []
    for item in data:
        if not isinstance(item, str):
            continue
        claim = item.strip(" \n\t-")
        if len(claim.split()) >= 3:
            cleaned.append(claim)
    return _dedupe_claims(cleaned, max_claims)


@lru_cache(maxsize=512)
def _extract_claims_with_llm(
    question: str,
    ideal_answer: str,
    max_claims: int,
    model: str,
) -> tuple[str, ...]:
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


def extract_claims(
    ideal_answer: str,
    question: str = "",
    max_claims: int = 6,
) -> list[str]:
    """
    Break an ideal answer into short atomic claims that can be matched
    independently against the candidate answer.
    """
    settings = get_settings()
    fallback = _regex_extract_claims(ideal_answer, max_claims=max_claims)

    if settings.CLAIM_EXTRACTION_MODE != "llm":
        return fallback
    if not settings.GROQ_API_KEY:
        return fallback

    try:
        claims = list(
            _extract_claims_with_llm(
                question.strip(),
                ideal_answer.strip(),
                max_claims,
                settings.CLAIM_EXTRACTION_MODEL,
            )
        )
        return claims or fallback
    except Exception as exc:
        logger.warning("LLM claim extraction failed, falling back to regex: %s", exc)
        return fallback
