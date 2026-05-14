from dataclasses import dataclass

from app.config import get_settings
from app.services.scoring.claim_extractor import CORE, extract_claim_specs
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


def _soft_coverage(match: ClaimMatch, threshold: float, margin: float) -> float:
    if margin <= 0:
        return 1.0 if match.covered else 0.0
    return min(
        1.0,
        max(
            0.0,
            (match.combined - (threshold - margin)) / margin,
        ),
    )


def score(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
) -> ClaimScoreResult:
    settings = get_settings()
    claim_specs = extract_claim_specs(
        ideal_answer,
        question=question,
        max_claims=settings.CLAIM_MAX_CLAIMS,
    )
    if not claim_specs:
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
        claim_specs,
        threshold=settings.CLAIM_MATCH_THRESHOLD,
    )

    if not matches:
        required_specs = [
            claim for claim in claim_specs if claim.importance == CORE
        ] or claim_specs
        return ClaimScoreResult(
            normalized_score=0.0,
            coverage=0.0,
            hard_coverage=0.0,
            missing_claims=[
                claim.text for claim in required_specs[: settings.KEYWORD_TOP_N]
            ],
            matches=[],
            avg_similarity=0.0,
            avg_entailment=0.0,
            avg_contradiction=0.0,
        )

    required_matches = [
        match for match in matches if match.importance == CORE
    ] or matches
    optional_matches = [
        match for match in matches if match.importance != CORE
    ]

    normalized = (
        sum(match.combined for match in required_matches) / len(required_matches)
    )
    coverage = sum(
        _soft_coverage(
            match,
            settings.CLAIM_MATCH_THRESHOLD,
            settings.CLAIM_SOFT_MARGIN,
        )
        for match in required_matches
    ) / len(required_matches)
    hard_coverage = (
        sum(1 for match in required_matches if match.covered) / len(required_matches)
    )

    if optional_matches:
        optional_coverage = sum(
            _soft_coverage(
                match,
                settings.CLAIM_MATCH_THRESHOLD,
                settings.CLAIM_SOFT_MARGIN,
            )
            for match in optional_matches
        ) / len(optional_matches)
        optional_bonus = getattr(settings, "CLAIM_OPTIONAL_BONUS", 0.05)
        normalized += optional_bonus * optional_coverage
        coverage += optional_bonus * optional_coverage

    avg_similarity = sum(match.similarity for match in matches) / len(matches)
    avg_entailment = sum(match.entailment for match in matches) / len(matches)
    avg_contradiction = sum(match.contradiction for match in matches) / len(matches)
    missing = [match.claim for match in required_matches if not match.covered]

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
