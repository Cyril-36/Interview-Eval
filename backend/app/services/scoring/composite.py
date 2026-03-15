from dataclasses import dataclass
from app.config import get_settings
from app.services.scoring import sbert_scorer, nli_scorer, keyword_scorer


@dataclass
class ScoringResult:
    sbert_raw: float
    nli_raw: float
    keyword_raw: float
    composite: float
    grade: str
    missing_keywords: list[str]


def _compute_grade(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Needs Improvement"
    else:
        return "Significant Gaps"


def evaluate(candidate_answer: str, ideal_answer: str) -> ScoringResult:
    settings = get_settings()

    sbert_raw = sbert_scorer.score(candidate_answer, ideal_answer)
    nli_raw = nli_scorer.score(candidate_answer, ideal_answer)
    keyword_raw, missing_keywords = keyword_scorer.score(candidate_answer, ideal_answer)

    composite_raw = (
        settings.SBERT_WEIGHT * sbert_raw
        + settings.NLI_WEIGHT * nli_raw
        + settings.KEYWORD_WEIGHT * keyword_raw
    )
    composite_100 = round(composite_raw * 100, 1)

    return ScoringResult(
        sbert_raw=round(sbert_raw * 100, 1),
        nli_raw=round(nli_raw * 100, 1),
        keyword_raw=round(keyword_raw * 100, 1),
        composite=composite_100,
        grade=_compute_grade(composite_100),
        missing_keywords=missing_keywords,
    )
