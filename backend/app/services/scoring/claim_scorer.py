from dataclasses import dataclass

from app.config import get_settings
from app.services.scoring.claim_extractor import extract_claims
from app.services.scoring.claim_matcher import ClaimMatch, match_claims


@dataclass
class ClaimScoreResult:
    normalized_score: float
    coverage: float
    hard_coverage: float
    missing_claims: list[str]
    matches: list[ClaimMatch]
    avg_similarity: float
    avg_entailment: float
    avg_contradiction: float


def score(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
) -> ClaimScoreResult:
    settings = get_settings()
    claims = extract_claims(
        ideal_answer,
        question=question,
        max_claims=settings.CLAIM_MAX_CLAIMS,
    )
    if not claims:
        return ClaimScoreResult(
            normalized_score=0.5,
            coverage=0.5,
            hard_coverage=0.5,
            missing_claims=[],
            matches=[],
            avg_similarity=0.0,
            avg_entailment=0.0,
            avg_contradiction=0.0,
        )

    matches = match_claims(
        candidate_answer,
        claims,
        threshold=settings.CLAIM_MATCH_THRESHOLD,
    )

    if not matches:
        return ClaimScoreResult(
            normalized_score=0.0,
            coverage=0.0,
            hard_coverage=0.0,
            missing_claims=claims[: settings.KEYWORD_TOP_N],
            matches=[],
            avg_similarity=0.0,
            avg_entailment=0.0,
            avg_contradiction=0.0,
        )

    normalized = sum(match.combined for match in matches) / len(matches)
    coverage = sum(
        min(
            1.0,
            max(
                0.0,
                (
                    match.combined
                    - (settings.CLAIM_MATCH_THRESHOLD - settings.CLAIM_SOFT_MARGIN)
                ) / settings.CLAIM_SOFT_MARGIN,
            ),
        )
        for match in matches
    ) / len(matches)
    hard_coverage = sum(1 for match in matches if match.covered) / len(matches)
    avg_similarity = sum(match.similarity for match in matches) / len(matches)
    avg_entailment = sum(match.entailment for match in matches) / len(matches)
    avg_contradiction = sum(match.contradiction for match in matches) / len(matches)
    missing = [match.claim for match in matches if not match.covered]

    return ClaimScoreResult(
        normalized_score=max(0.0, min(1.0, normalized)),
        coverage=max(0.0, min(1.0, coverage)),
        hard_coverage=max(0.0, min(1.0, hard_coverage)),
        missing_claims=missing[: settings.KEYWORD_TOP_N],
        matches=matches,
        avg_similarity=max(0.0, min(1.0, avg_similarity)),
        avg_entailment=max(0.0, min(1.0, avg_entailment)),
        avg_contradiction=max(0.0, min(1.0, avg_contradiction)),
    )
