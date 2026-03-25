import asyncio
import json
import logging
from dataclasses import dataclass
from groq import Groq
from app.config import get_settings
from app.services.scoring.executor import executor as _executor

logger = logging.getLogger(__name__)

RUBRIC_FIELDS = ("correctness", "completeness", "clarity", "depth")
RUBRIC_WEIGHTS = {
    "correctness": 0.40,
    "completeness": 0.40,
    "clarity": 0.10,
    "depth": 0.10,
}


@dataclass
class LLMJudgeResult:
    normalized_score: float
    reason: str
    correctness: float
    completeness: float
    clarity: float
    depth: float
    is_fallback: bool = False


LLM_SCORING_PROMPT = """You are an expert interview evaluator. Score the candidate's answer \
compared to the ideal answer on a scale of 0 to 100.

Question: {question}

Ideal Answer: {ideal_answer}

Candidate's Answer: {candidate_answer}

Evaluate based on:
1. Correctness — Are the facts and concepts accurate?
2. Completeness — Does it cover the key points from the ideal answer?
3. Clarity — Is it well-structured and easy to understand?
4. Depth — Does it show genuine understanding, not just surface-level recall?

Respond with ONLY a JSON object in this exact format, no other text:
{{
  "correctness": <0-100>,
  "completeness": <0-100>,
  "clarity": <0-100>,
  "depth": <0-100>,
  "reason": "<one sentence explanation>"
}}

Do not include any extra keys. The final LLM judge score weights correctness and
completeness most heavily."""

MAX_RETRIES = 2
TIMEOUT_SECONDS = 15


async def score(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
) -> LLMJudgeResult:
    """
    Use LLM (Groq/Llama) as a judge with rubric subscores.
    Returns normalized overall and rubric subscores on a 0-1 scale.
    """
    settings = get_settings()

    if not settings.GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — skipping LLM scoring")
        return _fallback_result("LLM scoring unavailable (no API key)")

    try:
        prompt = LLM_SCORING_PROMPT.format(
            question=question or "N/A",
            ideal_answer=ideal_answer,
            candidate_answer=candidate_answer,
        )

        def _call_groq():
            client = Groq(
                api_key=settings.GROQ_API_KEY, timeout=TIMEOUT_SECONDS,
            )
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    return client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.1,
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"LLM scorer attempt {attempt + 1}/{MAX_RETRIES}"
                        f" failed: {e}"
                    )
                    if attempt == MAX_RETRIES - 1:
                        raise last_error  # type: ignore[misc]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(_executor, _call_groq)
        response_text = response.choices[0].message.content or ""

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response_text[start:end])
            else:
                raise ValueError("Could not parse LLM score response")

        rubric = _parse_rubric_scores(data)
        raw_score = sum(
            rubric[field] * RUBRIC_WEIGHTS[field] for field in RUBRIC_FIELDS
        )
        reason = data.get("reason", "")

        logger.info(f"LLM judge score: {raw_score}/100 — {reason}")
        return LLMJudgeResult(
            normalized_score=raw_score / 100.0,
            reason=reason,
            correctness=rubric["correctness"] / 100.0,
            completeness=rubric["completeness"] / 100.0,
            clarity=rubric["clarity"] / 100.0,
            depth=rubric["depth"] / 100.0,
            is_fallback=False,
        )

    except Exception as e:
        logger.warning(f"LLM scoring failed: {e}")
        return _fallback_result(f"LLM scoring failed: {str(e)}")


def _fallback_result(reason: str) -> LLMJudgeResult:
    return LLMJudgeResult(
        normalized_score=0.0,
        reason=reason,
        correctness=0.0,
        completeness=0.0,
        clarity=0.0,
        depth=0.0,
        is_fallback=True,
    )


def _normalize_score(value: object) -> float:
    raw = float(value)
    return max(0.0, min(100.0, raw))


def _parse_rubric_scores(data: dict) -> dict[str, float]:
    if all(field in data for field in RUBRIC_FIELDS):
        return {
            field: _normalize_score(data[field])
            for field in RUBRIC_FIELDS
        }

    # Backward-compatible fallback for the older single-score format.
    raw_score = _normalize_score(data.get("score", 50))
    return {field: raw_score for field in RUBRIC_FIELDS}
