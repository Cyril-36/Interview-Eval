"""STAR behavioral answer scorer using LLM-as-judge.

Evaluates answers to behavioral ("tell me about a time…") questions
on the STAR framework: Situation, Task, Action, Result, plus Reflection.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from groq import Groq
from app.config import get_settings
from app.services.scoring.executor import executor as _executor

logger = logging.getLogger(__name__)

STAR_FIELDS = ("situation", "task", "action", "result", "reflection")


@dataclass
class STARResult:
    normalized_score: float  # 0-1 average of all STAR fields
    reason: str
    situation: float   # 0-1
    task: float        # 0-1
    action: float      # 0-1
    result: float      # 0-1
    reflection: float  # 0-1
    is_fallback: bool = False


STAR_SCORING_PROMPT = """You are an expert behavioral interview evaluator. Score the candidate's \
answer using the STAR framework.

Question: {question}

Ideal Answer: {ideal_answer}

Candidate's Answer: {candidate_answer}

Evaluate based on the STAR framework:
1. Situation — Did they clearly describe the context and setting?
2. Task — Did they explain their specific responsibility or challenge?
3. Action — Did they detail the concrete steps they personally took?
4. Result — Did they quantify or clearly state the outcome and impact?
5. Reflection — Did they show learning, growth, or self-awareness?

Respond with ONLY a JSON object in this exact format, no other text:
{{
  "situation": <0-100>,
  "task": <0-100>,
  "action": <0-100>,
  "result": <0-100>,
  "reflection": <0-100>,
  "reason": "<one sentence explanation>"
}}

Do not include any extra keys. The final score is the average of the five dimensions."""

MAX_RETRIES = 2
TIMEOUT_SECONDS = 15


async def score(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
) -> STARResult:
    """Score a behavioral answer using the STAR framework via LLM judge."""
    settings = get_settings()

    if not settings.GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — skipping STAR scoring")
        return _fallback_result("STAR scoring unavailable (no API key)")

    try:
        prompt = STAR_SCORING_PROMPT.format(
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
                        f"STAR scorer attempt {attempt + 1}/{MAX_RETRIES}"
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
                raise ValueError("Could not parse STAR score response")

        star = _parse_star_scores(data)
        raw_score = sum(star.values()) / len(STAR_FIELDS)
        reason = data.get("reason", "")

        logger.info(f"STAR judge score: {raw_score}/100 — {reason}")
        return STARResult(
            normalized_score=raw_score / 100.0,
            reason=reason,
            situation=star["situation"] / 100.0,
            task=star["task"] / 100.0,
            action=star["action"] / 100.0,
            result=star["result"] / 100.0,
            reflection=star["reflection"] / 100.0,
            is_fallback=False,
        )

    except Exception as e:
        logger.warning(f"STAR scoring failed: {e}")
        return _fallback_result(f"STAR scoring failed: {str(e)}")


def _fallback_result(reason: str) -> STARResult:
    return STARResult(
        normalized_score=0.0,
        reason=reason,
        situation=0.0,
        task=0.0,
        action=0.0,
        result=0.0,
        reflection=0.0,
        is_fallback=True,
    )


def _normalize_score(value: object) -> float:
    raw = float(value)
    return max(0.0, min(100.0, raw))


def _parse_star_scores(data: dict) -> dict[str, float]:
    if all(field in data for field in STAR_FIELDS):
        return {
            field: _normalize_score(data[field])
            for field in STAR_FIELDS
        }
    # Fallback: uniform score
    raw_score = _normalize_score(data.get("score", 50))
    return {field: raw_score for field in STAR_FIELDS}
