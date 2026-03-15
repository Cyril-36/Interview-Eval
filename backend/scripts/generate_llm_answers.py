"""
Use Groq LLM to generate realistic candidate answers at 3 quality levels.

This produces answers where:
- GOOD: Semantically correct but uses DIFFERENT vocabulary/structure
- AVERAGE: Partially correct with genuine gaps
- POOR: Vague, off-topic, surface-level

This is the gold standard approach — ensures TF-IDF can't trivially distinguish
while SBERT + NLI can, which is the core thesis of the project.

Usage:
    cd backend && python -m scripts.generate_llm_answers
"""

import json
import time
import logging
from pathlib import Path
from groq import Groq
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "evaluation"


PROMPT_TEMPLATE = """You are simulating interview candidates of varying skill levels.

Given this interview question and ideal answer, generate THREE candidate answers:

**Question:** {question}

**Ideal Answer:** {ideal_answer}

Generate exactly 3 answers:

1. **GOOD answer** (score ~8/10): The candidate understands the concept fully and explains it correctly, but uses their OWN words and structure — NOT copying the ideal answer. They may add a personal example. The answer should be semantically equivalent but lexically different.

2. **AVERAGE answer** (score ~5/10): The candidate knows the general topic but only covers about 40% of the key points. They pad with generic filler statements. Some correct information mixed with vagueness.

3. **POOR answer** (score ~2/10): The candidate barely knows the topic. They give a vague, surface-level response that doesn't demonstrate real understanding. May mention a keyword but can't explain it.

Return ONLY valid JSON in this exact format:
{{
  "good": "the good candidate's answer here",
  "average": "the average candidate's answer here",
  "poor": "the poor candidate's answer here"
}}

IMPORTANT: The GOOD answer must NOT copy phrases from the ideal answer. Use completely different wording to express the same ideas. This is critical."""


def generate_answers_batch(questions, client):
    """Generate candidate answers for a list of questions."""
    results = []

    for i, q in enumerate(questions):
        prompt = PROMPT_TEMPLATE.format(
            question=q["question"],
            ideal_answer=q["ideal_answer"],
        )

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )

            text = response.choices[0].message.content
            # Parse JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                results.append({
                    "question_index": i,
                    "good": data.get("good", ""),
                    "average": data.get("average", ""),
                    "poor": data.get("poor", ""),
                })
                logger.info(f"  [{i+1}/{len(questions)}] Generated answers for: {q['question'][:60]}...")
            else:
                logger.warning(f"  [{i+1}] Failed to parse JSON response")
                results.append(None)

        except Exception as e:
            logger.warning(f"  [{i+1}] Error: {e}")
            results.append(None)

        # Rate limit: Groq free tier = 30 req/min
        if (i + 1) % 25 == 0:
            logger.info("  Pausing 60s for rate limit...")
            time.sleep(60)
        else:
            time.sleep(2)

    return results


def main():
    # Load V2 dataset to get questions
    v2_path = EVAL_DIR / "dataset_v2.json"
    if not v2_path.exists():
        logger.error("Run prepare_evaluation_dataset_v2.py first")
        return

    with open(v2_path) as f:
        v2_data = json.load(f)

    # Get unique questions
    seen = set()
    questions = []
    for entry in v2_data:
        qi = entry["question_index"]
        if qi not in seen:
            seen.add(qi)
            questions.append({
                "question_index": qi,
                "question": entry["question"],
                "ideal_answer": entry["ideal_answer"],
                "role": entry["role"],
                "category": entry["category"],
                "difficulty": entry["difficulty"],
            })

    logger.info(f"Generating LLM answers for {len(questions)} questions...")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')

    if not api_key:
        logger.error("Set GROQ_API_KEY in .env or environment")
        return

    client = Groq(api_key=api_key)
    results = generate_answers_batch(questions, client)

    # Build new dataset
    dataset = []
    for q, r in zip(questions, results):
        if r is None:
            # Fallback to V2 answers
            v2_entries = [e for e in v2_data if e["question_index"] == q["question_index"]]
            for e in v2_entries:
                dataset.append(e)
            continue

        for quality in ["good", "average", "poor"]:
            dataset.append({
                "id": f"q{q['question_index']+1}_{quality}",
                "question_index": q["question_index"],
                "question": q["question"],
                "ideal_answer": q["ideal_answer"],
                "candidate_answer": r[quality],
                "quality_level": quality,
                "role": q["role"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "human_score": None,
                "rater_1": None,
                "rater_2": None,
            })

    output_path = EVAL_DIR / "dataset_llm.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"\nLLM dataset saved: {len(dataset)} instances -> {output_path}")

    # Quick quality check
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    for q in ["good", "average", "poor"]:
        entries = [e for e in dataset if e["quality_level"] == q]
        sims = []
        for e in entries:
            v = TfidfVectorizer()
            tfidf = v.fit_transform([e["ideal_answer"], e["candidate_answer"]])
            sims.append(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        print(f"  {q.upper():>8} avg TF-IDF sim to ideal: {sum(sims)/len(sims):.3f}")


if __name__ == "__main__":
    main()
