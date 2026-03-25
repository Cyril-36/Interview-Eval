"""
V2: Improved evaluation dataset generator with realistic candidate answers.

Key fixes over V1:
- Good answers use DIFFERENT vocabulary and sentence structure (not word swaps)
- Average answers genuinely miss 50% of content, add filler reasoning
- Poor answers are truly off-topic or surface-level only
- This ensures TF-IDF can't trivially match good answers to ideal

Usage:
    cd backend && python -m scripts.prepare_evaluation_dataset_v2
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


def load_hr_dataset():
    path = DATA_DIR / "hr_interview_sampled.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_ml_dataset():
    path = DATA_DIR / "ml_interview_questions.csv"
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return [{
            "question": row["question"],
            "ideal_answer": row["answer"],
            "category": row.get("category", "ML"),
            "difficulty": row.get("difficulty", "medium"),
            "role": "ML Engineer",
            "keywords": row.get("topic_tags", "").split(","),
        } for row in reader]


# ─── POOR ANSWERS: Vague, off-topic, no substance ───

POOR_STRATEGIES = [
    # Strategy 1: Only knows the term
    lambda q, kws: f"I've heard of {kws[0] if kws else 'this'} before but I'm not confident explaining it in detail. It's something we covered briefly in class.",

    # Strategy 2: Completely generic
    lambda q, kws: "That's a common interview question. I would say it depends on the context and requirements of the specific project. Every situation is different so there's no one-size-fits-all answer.",

    # Strategy 3: Deflects
    lambda q, kws: f"I haven't worked with {kws[0] if kws else 'this'} directly, but I'm a fast learner and I'm sure I could pick it up quickly on the job.",

    # Strategy 4: Circular/tautological
    lambda q, kws: f"{kws[0].title() if kws else 'This concept'} is important because it's widely used in the industry. Many companies look for this skill because it's valuable.",

    # Strategy 5: Minimal effort
    lambda q, kws: f"I think {kws[0] if kws else 'it'} is related to how things work in software. I'd need to review my notes to give a better answer.",

    # Strategy 6: Wrong direction
    lambda q, kws: "I believe this is mainly about writing clean code and following best practices. Good documentation and code reviews are always important regardless of the specific topic.",
]

# ─── AVERAGE ANSWERS: Partial coverage, some real content ───

AVERAGE_STRATEGIES = [
    # Strategy 1: Gets the first point, misses the rest
    lambda sentences, kws: _avg_first_point(sentences, kws),

    # Strategy 2: Knows the concept but explains vaguely
    lambda sentences, kws: _avg_vague_explanation(sentences, kws),

    # Strategy 3: Lists keywords without connecting them
    lambda sentences, kws: _avg_keyword_list(sentences, kws),
]


def _avg_first_point(sentences, keywords):
    """Takes first 30-40% of content and adds filler."""
    if not sentences:
        return "I'm not entirely sure about this topic."
    kept = sentences[:max(1, len(sentences) * 2 // 5)]
    text = ". ".join(kept)
    if not text.endswith("."):
        text += "."
    fillers = [
        " I think there are other aspects to this but I can't recall them right now.",
        " There's more to it but those are the main points I remember.",
        " I'm not sure about the remaining details though.",
    ]
    return text + random.choice(fillers)


def _avg_vague_explanation(sentences, keywords):
    """Mentions keywords but doesn't explain relationships."""
    if len(keywords) < 2:
        keywords = ["this concept", "the approach"]
    kw_subset = random.sample(keywords, min(3, len(keywords)))
    return (
        f"This involves {kw_subset[0]} and {kw_subset[1] if len(kw_subset) > 1 else 'related concepts'}. "
        "The basic idea is that you need to understand how these work together in practice. "
        f"{'It also relates to ' + kw_subset[2] + ' which is important for the overall approach. ' if len(kw_subset) > 2 else ''}"
        "I've studied this topic and understand the fundamentals, though I might be missing some of the finer details."
    )


def _avg_keyword_list(sentences, keywords):
    """Lists correct terms but shallow explanation."""
    if len(keywords) < 2:
        return _avg_first_point(sentences, keywords)
    kws = random.sample(keywords, min(4, len(keywords)))
    items = ", ".join(kws[:-1]) + f" and {kws[-1]}"
    return (
        f"The key concepts here include {items}. "
        "Each of these plays a role in how the system works. "
        f"In my understanding, the most important one is {kws[0]} because it forms the foundation. "
        "The others build on top of it in various ways."
    )


# ─── GOOD ANSWERS: Semantically equivalent, different words ───

GOOD_STRATEGIES = [
    # Strategy 1: Restructure — reverse order + rephrase
    lambda sentences, kws: _good_restructured(sentences, kws),

    # Strategy 2: Example-enriched — add a concrete example
    lambda sentences, kws: _good_with_example(sentences, kws),

    # Strategy 3: Explain like teaching — different framing
    lambda sentences, kws: _good_teaching_style(sentences, kws),
]


def _good_restructured(sentences, keywords):
    """Covers same content but in reverse/different order with rephrasing."""
    if len(sentences) <= 1:
        return _good_with_example(sentences, keywords)

    # Reverse order and rephrase slightly
    reordered = list(sentences)
    random.shuffle(reordered)

    # Don't just swap words — genuinely rephrase each sentence opening
    rephrasings = [
        "In other words, ", "Essentially, ", "Put simply, ",
        "What this means is ", "The core idea is that ",
        "To put it another way, ", "Fundamentally, ",
    ]

    result_parts = []
    for i, s in enumerate(reordered):
        if i > 0 and random.random() > 0.5:
            prefix = random.choice(rephrasings)
            # Lowercase first char of sentence if adding prefix
            s = prefix + s[0].lower() + s[1:] if s else s
        result_parts.append(s)

    text = ". ".join(result_parts)
    if not text.endswith("."):
        text += "."
    return text


def _good_with_example(sentences, keywords):
    """Covers the content and adds a practical example."""
    text = ". ".join(sentences)
    if not text.endswith("."):
        text += "."

    kw = keywords[0] if keywords else "this concept"
    examples = [
        f" For instance, when working with {kw}, you'd typically see this in production systems where reliability matters.",
        f" A practical example would be implementing {kw} in a team project — it significantly improves code quality.",
        f" I've applied {kw} in a course project where it helped us structure the solution more effectively.",
    ]
    return text + random.choice(examples)


def _good_teaching_style(sentences, keywords):
    """Same content framed as if teaching someone."""
    if not sentences:
        return "This is an important concept."

    text = ". ".join(sentences)
    if not text.endswith("."):
        text += "."

    intro = random.choice([
        "The way I think about this is: ",
        "To break this down: ",
        "Here's how I'd explain it: ",
        "The fundamental principle is that ",
    ])

    conclusion = random.choice([
        " Understanding this distinction is crucial for building robust systems.",
        " This knowledge directly impacts how you design and implement solutions.",
        " Getting this right makes a significant difference in real-world applications.",
    ])

    return intro + text[0].lower() + text[1:] + conclusion


def extract_keywords_simple(text):
    """Extract key terms from text without NLP models."""
    # Common technical and interview terms to keep
    words = text.lower().split()
    stopwords = set("a an the is are was were be been being have has had do does did will would "
                    "shall should may might must can could need dare that this these those i me my we "
                    "our you your he she it they them their what which who whom how when where why "
                    "to for from in on at by with of and or but not no nor so yet both either neither "
                    "also very more most quite rather really just even still already".split())

    keywords = []
    for w in words:
        cleaned = w.strip(".,;:!?\"'()-")
        if cleaned and len(cleaned) > 3 and cleaned not in stopwords:
            if cleaned not in keywords:
                keywords.append(cleaned)
    return keywords[:10]


def select_questions(hr_data, ml_data):
    selected = []
    role_map = {}

    for item in hr_data:
        role = item.get("role", "")
        matched_role = None
        for target in TARGET_ROLES:
            if target.lower() == role.lower():
                matched_role = target
                break
        if not matched_role:
            rl = role.lower()
            if "software" in rl or "developer" in rl:
                matched_role = "Software Engineer"
            elif "data" in rl:
                matched_role = "Data Scientist"
            elif "devops" in rl:
                matched_role = "DevOps Engineer"
            elif "product" in rl:
                matched_role = "Product Manager"
            elif "ml" in rl or "machine" in rl:
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

    for item in ml_data[:20]:
        if "ML Engineer" not in role_map:
            role_map["ML Engineer"] = []
        role_map["ML Engineer"].append(item)

    for role in TARGET_ROLES:
        candidates = role_map.get(role, [])
        candidates = [c for c in candidates if len(c.get("ideal_answer", "")) > 50]
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
        sentences = [s.strip() for s in ideal.split(". ") if s.strip()]

        keywords = q.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        if not keywords or all(k.strip() == "" for k in keywords):
            keywords = extract_keywords_simple(ideal)

        # Generate good answer — NOT a word-swap copy
        good_fn = random.choice(GOOD_STRATEGIES)
        good_answer = good_fn(sentences, keywords)

        # Generate average answer — partial, with genuine gaps
        avg_fn = random.choice(AVERAGE_STRATEGIES)
        avg_answer = avg_fn(sentences, keywords)

        # Generate poor answer — vague, off-topic
        poor_fn = random.choice(POOR_STRATEGIES)
        poor_answer = poor_fn(q["question"], keywords)

        for quality, candidate in [("good", good_answer), ("average", avg_answer), ("poor", poor_answer)]:
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
                "human_score": None,
                "rater_1": None,
                "rater_2": None,
            })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "dataset_v2.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nV2 Evaluation dataset saved: {len(dataset)} instances -> {output_path}")
    print(f"  {len(questions)} questions x 3 quality levels = {len(dataset)} instances")
    print("\nRoles:")
    role_counts = {}
    for q in questions:
        role_counts[q["role"]] = role_counts.get(q["role"], 0) + 1
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")


if __name__ == "__main__":
    build_evaluation_dataset()
