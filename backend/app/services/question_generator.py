import json
import logging
from groq import Groq
from sentence_transformers import util

from app.config import get_settings
from app.models_loader import get_registry

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are an expert technical interviewer. Generate exactly {count} interview questions for the following profile:

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

Make questions diverse — cover different sub-topics within {category}. Each question should test a distinct concept."""


async def generate_questions(
    role: str,
    level: str,
    category: str,
    difficulty: str,
    num_questions: int,
) -> list[dict]:
    settings = get_settings()

    # Request 2x questions for diversity filtering
    request_count = min(num_questions * 2, 20)

    prompt = GENERATION_PROMPT.format(
        count=request_count,
        role=role,
        level=level,
        category=category,
        difficulty=difficulty,
    )

    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )

    # Parse response
    response_text = response.choices[0].message.content
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response_text[start:end])
        else:
            raise ValueError("Failed to parse LLM response as JSON")

    raw_questions = data.get("questions", [])
    if not raw_questions:
        raise ValueError("No questions generated")

    # Apply SBERT diversity filter
    filtered = _diversity_filter(raw_questions, num_questions, settings.DIVERSITY_THRESHOLD)
    return filtered


def _diversity_filter(
    questions: list[dict],
    target_count: int,
    threshold: float,
) -> list[dict]:
    """Greedy selection: pick questions with max pairwise cosine similarity < threshold."""
    if len(questions) <= 1:
        return questions[:target_count]

    sbert = get_registry().sbert
    texts = [q["question_text"] for q in questions]
    embeddings = sbert.encode(texts, convert_to_tensor=True)

    selected_indices = [0]

    for i in range(1, len(questions)):
        if len(selected_indices) >= target_count:
            break

        # Check similarity against all already-selected questions
        is_diverse = True
        for j in selected_indices:
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= threshold:
                is_diverse = False
                break

        if is_diverse:
            selected_indices.append(i)

    return [questions[i] for i in selected_indices]
