import asyncio
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from sentence_transformers import util

from app.config import get_settings
from app.models_loader import get_registry

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

GENERATION_PROMPT = """You are an expert technical interviewer. Generate exactly \
{count} interview questions for the following profile:

- Role: {role}
- Experience Level: {level}
- Category: {category}
- Difficulty: {difficulty}

For each question, provide:
1. The question text
2. A comprehensive ideal answer (3-5 sentences)
3. The category label
4. The difficulty label

IMPORTANT: Return ONLY valid JSON in this exact format, no other text:
{{
  "questions": [
    {{
      "question_text": "...",
      "ideal_answer": "...",
      "category": "{category}",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Make questions diverse — cover different sub-topics within {category}. \
Each question should test a distinct concept.

IMPORTANT: Be creative and vary your questions. Use seed {seed} for randomness. \
Do NOT repeat common/obvious questions — dig into lesser-known sub-topics too."""

MAX_RETRIES = 3
TIMEOUT_SECONDS = 30


async def generate_questions(
    role: str,
    level: str,
    category: str,
    difficulty: str,
    num_questions: int,
) -> list[dict]:
    settings = get_settings()

    request_count = min(num_questions * 2, 20)

    prompt = GENERATION_PROMPT.format(
        count=request_count,
        role=role,
        level=level,
        category=category,
        difficulty=difficulty,
        seed=random.randint(1, 100000),
    )

    def _call_groq():
        client = Groq(api_key=settings.GROQ_API_KEY, timeout=TIMEOUT_SECONDS)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.9,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Question generation attempt"
                    f" {attempt + 1}/{MAX_RETRIES} failed: {e}"
                )
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Question generation failed after"
                        f" {MAX_RETRIES} attempts: {last_error}"
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
            raise ValueError("Failed to parse LLM response as JSON")

    raw_questions = data.get("questions", [])
    if not raw_questions:
        raise ValueError("No questions generated")

    # Diversity filter uses SBERT encode (CPU-heavy) — run in executor
    filtered = await loop.run_in_executor(
        _executor, _diversity_filter,
        raw_questions, num_questions, settings.DIVERSITY_THRESHOLD,
    )
    return filtered


def _diversity_filter(
    questions: list[dict],
    target_count: int,
    threshold: float,
) -> list[dict]:
    """Greedy selection: pick questions with max pairwise cosine < threshold."""
    if len(questions) <= 1:
        return questions[:target_count]

    sbert = get_registry().sbert
    texts = [q["question_text"] for q in questions]
    embeddings = sbert.encode(texts, convert_to_tensor=True)

    selected_indices = [0]

    for i in range(1, len(questions)):
        if len(selected_indices) >= target_count:
            break

        is_diverse = True
        for j in selected_indices:
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= threshold:
                is_diverse = False
                break

        if is_diverse:
            selected_indices.append(i)

    return [questions[i] for i in selected_indices]
