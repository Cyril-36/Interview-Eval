from dataclasses import dataclass, field


BEHAVIORAL_KEYWORDS = {"behavioral", "behaviour", "behavior", "star", "situational"}


def is_behavioral(category: str) -> bool:
    """Check if a question category should use the behavioral STAR pipeline."""
    return any(kw in category.lower() for kw in BEHAVIORAL_KEYWORDS)


@dataclass
class ScoringResult:
    sbert_raw: float
    nli_raw: float
    keyword_raw: float
    llm_raw: float
    llm_reason: str
    llm_correctness: float
    llm_completeness: float
    llm_clarity: float
    llm_depth: float
    raw_composite: float
    composite: float
    grade: str
    missing_keywords: list[str]
    is_behavioral: bool = False
    star_situation: float = 0.0
    star_task: float = 0.0
    star_action: float = 0.0
    star_result: float = 0.0
    star_reflection: float = 0.0
    claim_matches: list[dict] = field(default_factory=list)


def compute_grade(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Needs Improvement"
    return "Significant Gaps"
