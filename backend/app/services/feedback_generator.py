import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

FEEDBACK_PROMPT = """You are an expert interview coach. Analyze the candidate's answer \
compared to the ideal answer and provide constructive feedback.

Question: {question}

Ideal Answer: {ideal_answer}

Candidate's Answer: {candidate_answer}

Score Breakdown:
- Semantic Similarity (SBERT): {sbert_score}/100
- Factual Consistency (NLI): {nli_score}/100
- Keyword Coverage: {keyword_score}/100
- Composite Score: {composite_score}/100

Missing Key Concepts: {missing_keywords}

Provide feedback in this exact JSON format, no other text:
{{
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "model_answer": "A concise improved version incorporating missed points"
}}

Be specific and constructive. Reference actual content from the answers."""

MAX_RETRIES = 2
TIMEOUT_SECONDS = 15


async def generate_feedback(
    question: str,
    ideal_answer: str,
    candidate_answer: str,
    sbert_score: float,
    nli_score: float,
    keyword_score: float,
    composite_score: float,
    missing_keywords: list[str],
) -> dict:
    settings = get_settings()

    try:
        return await _llm_feedback(
            question, ideal_answer, candidate_answer,
            sbert_score, nli_score, keyword_score, composite_score,
            missing_keywords, settings,
        )
    except Exception as e:
        logger.warning(f"LLM feedback failed, using fallback: {e}")
        return _fallback_feedback(missing_keywords, composite_score)


async def _llm_feedback(
    question: str,
    ideal_answer: str,
    candidate_answer: str,
    sbert_score: float,
    nli_score: float,
    keyword_score: float,
    composite_score: float,
    missing_keywords: list[str],
    settings,
) -> dict:
    prompt = FEEDBACK_PROMPT.format(
        question=question,
        ideal_answer=ideal_answer,
        candidate_answer=candidate_answer,
        sbert_score=sbert_score,
        nli_score=nli_score,
        keyword_score=keyword_score,
        composite_score=composite_score,
        missing_keywords=", ".join(missing_keywords) if missing_keywords else "None",
    )

    def _call_groq():
        client = Groq(api_key=settings.GROQ_API_KEY, timeout=TIMEOUT_SECONDS)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.3,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Feedback attempt {attempt + 1}/{MAX_RETRIES}"
                    f" failed: {e}"
                )
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Feedback generation failed: {last_error}"
                    )

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
            raise ValueError("Failed to parse feedback response")

    return {
        "strengths": data.get("strengths", []),
        "improvements": data.get("improvements", []),
        "model_answer": data.get("model_answer", ideal_answer),
    }


def _fallback_feedback(
    missing_keywords: list[str], composite_score: float,
) -> dict:
    strengths = []
    improvements = []

    if composite_score >= 60:
        strengths.append(
            "Your answer demonstrates a reasonable understanding of the topic."
        )
    if composite_score >= 80:
        strengths.append("Excellent coverage of the key concepts.")

    if missing_keywords:
        improvements.append(
            f"Consider covering these key concepts:"
            f" {', '.join(missing_keywords[:5])}"
        )
    if composite_score < 60:
        improvements.append(
            "Try to provide more detailed explanations with specific examples."
        )

    if not strengths:
        strengths.append("You attempted to answer the question.")
    if not improvements:
        improvements.append("Keep practicing to reinforce your understanding.")

    return {
        "strengths": strengths,
        "improvements": improvements,
        "model_answer": "Feedback generation unavailable offline."
        " Review the ideal answer for guidance.",
    }
