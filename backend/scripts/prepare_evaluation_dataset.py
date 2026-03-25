"""
Prepare the human evaluation dataset from downloaded interview QA datasets.

This script:
1. Loads HR interview + ML QA datasets
2. Selects 50 QA pairs across 5 roles (10 per role)
3. Auto-generates 3 quality levels of candidate answers per question:
   - Good: Close paraphrase of ideal answer
   - Average: Partial coverage, some key points missing
   - Poor: Vague, off-topic, or minimal answer
4. Outputs dataset.json for scoring and human rating

Usage:
    cd backend && python -m scripts.prepare_evaluation_dataset
"""

import json
import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = Path(__file__).parent.parent / "evaluation"

TARGET_ROLES = [
    "Software Engineer",
    "Data Scientist",
    "ML Engineer",
    "Product Manager",
    "DevOps Engineer",
]

POOR_TEMPLATES = [
    "I think it has something to do with {keyword}. Not entirely sure about the details though.",
    "This is a common topic in interviews. {keyword} is involved somehow but I'd need to review the specifics.",
    "I would need to look this up to give a complete answer. I know {keyword} is related but can't explain further.",
    "From what I vaguely remember, it's about {keyword}. I'd need to study this more before I could explain it properly.",
    "That's a good question. I believe {keyword} plays a role but I honestly can't explain the mechanism or details.",
    "I'm not very confident on this one. Something about {keyword} maybe? I should have prepared better for this topic.",
]

AVERAGE_PREFIXES = [
    "From my understanding, ",
    "Based on what I've learned, ",
    "I think the key idea is that ",
    "In my experience, ",
    "The way I understand it, ",
]

AVERAGE_SUFFIXES = [
    " However, I'm not fully sure about the remaining aspects and there might be other important points I'm missing.",
    " That said, I think there are additional details I'm not covering here.",
    " I believe there's more to this topic but these are the main points I can recall.",
    " There might be other considerations I'm forgetting right now.",
]

GOOD_REPHRASE_PAIRS = [
    ("is used to", "serves to"),
    ("allows", "enables"),
    ("important", "crucial"),
    ("helps", "assists in"),
    ("provides", "offers"),
    ("commonly", "frequently"),
    ("for example", "for instance"),
    ("however", "that said"),
    ("Additionally", "Moreover"),
    ("In addition", "Furthermore"),
    ("method", "approach"),
    ("process", "procedure"),
    ("system", "framework"),
    ("create", "develop"),
    ("use", "utilize"),
    ("get", "obtain"),
    ("show", "demonstrate"),
    ("make", "construct"),
    ("need", "require"),
    ("give", "provide"),
]

GOOD_INTRO = [
    "In my understanding, ",
    "To answer this, ",
    "Speaking from my knowledge, ",
    "",
    "",
    "",
]


def load_hr_dataset():
    path = DATA_DIR / "hr_interview_sampled.json"
    if not path.exists():
        print(f"HR dataset not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def load_ml_dataset():
    path = DATA_DIR / "ml_interview_questions.csv"
    if not path.exists():
        print(f"ML dataset not found at {path}")
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "question": row["question"],
                "ideal_answer": row["answer"],
                "category": row.get("category", "ML"),
                "difficulty": row.get("difficulty", "medium"),
                "role": "ML Engineer",
                "keywords": row.get("topic_tags", "").split(","),
            })
        return rows


def generate_poor_answer(ideal_answer: str, keywords: list[str]) -> str:
    """Poor: vague, only mentions a keyword, no real substance."""
    keyword = keywords[0] if keywords else "this concept"
    template = random.choice(POOR_TEMPLATES)
    return template.format(keyword=keyword)


AVERAGE_GENERIC_FILLERS = [
    "I think this is an important topic in software development.",
    "This comes up a lot in technical interviews.",
    "There are many perspectives on this question.",
    "I've encountered this in some of my coursework.",
    "I've read about this topic before and found it interesting.",
]


def generate_average_answer(ideal_answer: str) -> str:
    """Average: covers ~40% of the content with generic filler and vague structure."""
    sentences = [s.strip() for s in ideal_answer.split(". ") if s.strip()]
    words = ideal_answer.split()

    if len(words) < 30:
        # Very short ideal answer — extract key phrases and build a vague response
        # Take ~half the words, scramble slightly, add filler
        key_words = [w for w in words if len(w) > 4][:3]
        key_phrase = " and ".join(key_words) if key_words else "this topic"
        filler = random.choice(AVERAGE_GENERIC_FILLERS)
        prefix = random.choice(AVERAGE_PREFIXES)
        return (
            f"{prefix}the concept relates to {key_phrase}. "
            f"{filler} "
            "I know the basics but would need to elaborate more on the specifics and edge cases."
        )

    # Longer answer: take first ~40% of content
    num_to_keep = max(1, len(sentences) * 2 // 5)
    kept = sentences[:num_to_keep]

    result = ". ".join(kept)
    if not result.endswith("."):
        result += "."

    prefix = random.choice(AVERAGE_PREFIXES)
    suffix = random.choice(AVERAGE_SUFFIXES)
    return prefix + result[0].lower() + result[1:] + suffix


GOOD_ELABORATIONS = [
    " This is fundamental because it directly impacts team productivity and project outcomes.",
    " I believe this approach works well because it balances efficiency with thoroughness.",
    " In practice, this means being proactive and constantly looking for ways to improve.",
    " A concrete example would be implementing regular code reviews and knowledge-sharing sessions.",
    " This mindset has helped me consistently deliver quality results in collaborative environments.",
    " I've found that combining technical skills with strong communication makes the biggest difference.",
]


def generate_good_answer(ideal_answer: str) -> str:
    """Good: covers ~85% of content, rephrased with different vocabulary, structure, and added elaboration."""
    sentences = [s.strip() for s in ideal_answer.split(". ") if s.strip()]

    # For short answers, rephrase heavily and add elaboration
    if len(sentences) <= 3:
        # Rephrase the core content
        result = ideal_answer
        swaps = random.sample(GOOD_REPHRASE_PAIRS, min(6, len(GOOD_REPHRASE_PAIRS)))
        for old, new in swaps:
            result = result.replace(old, new, 1)

        # Add intro + elaboration to make it clearly richer than average
        intro = random.choice([
            "Absolutely. ",
            "That's a great question. ",
            "Yes, so ",
            "To elaborate, ",
        ])
        elaboration = random.choice(GOOD_ELABORATIONS)
        result = intro + result[0].lower() + result[1:]
        if not result.endswith("."):
            result += "."
        result += elaboration
        return result

    # Longer answer: drop one sentence, rephrase rest
    if len(sentences) > 4:
        drop_idx = random.randint(1, len(sentences) - 1)
        sentences = [s for i, s in enumerate(sentences) if i != drop_idx]

    result = ". ".join(sentences)
    if not result.endswith("."):
        result += "."

    swaps = random.sample(GOOD_REPHRASE_PAIRS, min(8, len(GOOD_REPHRASE_PAIRS)))
    for old, new in swaps:
        result = result.replace(old, new, 1)

    intro = random.choice(GOOD_INTRO)
    if intro:
        result = intro + result[0].lower() + result[1:]

    return result


def select_questions(hr_data: list, ml_data: list) -> list:
    """Select 10 questions per role, 50 total."""
    selected = []

    role_map = {}
    for item in hr_data:
        role = item.get("role", "")
        # Direct match to target roles
        matched_role = None
        for target in TARGET_ROLES:
            if target.lower() == role.lower():
                matched_role = target
                break
        if not matched_role:
            # Fuzzy mapping
            role_lower = role.lower()
            if "software" in role_lower or "developer" in role_lower:
                matched_role = "Software Engineer"
            elif "data" in role_lower:
                matched_role = "Data Scientist"
            elif "devops" in role_lower:
                matched_role = "DevOps Engineer"
            elif "product" in role_lower:
                matched_role = "Product Manager"
            elif "ml" in role_lower or "machine" in role_lower:
                matched_role = "ML Engineer"

        if matched_role:
            if matched_role not in role_map:
                role_map[matched_role] = []
            role_map[matched_role].append({
                "question": item["question"],
                "ideal_answer": item["ideal_answer"],
                "category": item.get("category", "General"),
                "difficulty": item.get("difficulty", "Medium"),
                "role": matched_role,
                "keywords": item.get("keywords", []),
            })

    # Add ML data
    for item in ml_data[:20]:
        if "ML Engineer" not in role_map:
            role_map["ML Engineer"] = []
        role_map["ML Engineer"].append(item)

    # Select 10 per role
    for role in TARGET_ROLES:
        candidates = role_map.get(role, [])
        # Filter for ones with good ideal answers (> 30 chars)
        candidates = [c for c in candidates if len(c.get("ideal_answer", "")) > 30]
        random.shuffle(candidates)
        selected.extend(candidates[:10])

    return selected


def build_evaluation_dataset():
    random.seed(42)

    hr_data = load_hr_dataset()
    ml_data = load_ml_dataset()

    print(f"Loaded {len(hr_data)} HR questions, {len(ml_data)} ML questions")

    questions = select_questions(hr_data, ml_data)
    print(f"Selected {len(questions)} questions across {len(TARGET_ROLES)} roles")

    dataset = []
    for i, q in enumerate(questions):
        ideal = q["ideal_answer"]
        keywords = q.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]

        # Generate 3 quality levels
        for quality, generator in [
            ("good", lambda: generate_good_answer(ideal)),
            ("average", lambda: generate_average_answer(ideal)),
            ("poor", lambda: generate_poor_answer(ideal, keywords)),
        ]:
            candidate = generator()
            dataset.append({
                "id": f"q{i+1}_{quality}",
                "question_index": i,
                "question": q["question"],
                "ideal_answer": ideal,
                "candidate_answer": candidate,
                "quality_level": quality,
                "role": q["role"],
                "category": q.get("category", "General"),
                "difficulty": q.get("difficulty", "Medium"),
                "human_score": None,  # To be filled by human raters (0-10)
                "rater_1": None,
                "rater_2": None,
            })

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nEvaluation dataset saved: {len(dataset)} answer instances -> {output_path}")
    print(f"  {len(questions)} questions x 3 quality levels = {len(dataset)} instances")
    print("\nRoles covered:")
    role_counts = {}
    for q in questions:
        role_counts[q['role']] = role_counts.get(q['role'], 0) + 1
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count} questions")

    print("\nNext steps:")
    print(f"  1. Have 2-3 raters score each answer (0-10) in {output_path}")
    print("  2. Fill in 'human_score' (average), 'rater_1', 'rater_2' fields")
    print("  3. Run: python -m evaluation.grid_search")


if __name__ == "__main__":
    build_evaluation_dataset()
