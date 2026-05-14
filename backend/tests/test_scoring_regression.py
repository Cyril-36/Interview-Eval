"""
Regression tests for the technical scoring composite.

Anchored on a real session where weights were unbalanced:
- Q1 (correct, LLM=92, correctness=95): old composite 60  → expected ≥ 65
- Q2 (correct, LLM=86, correctness=90): old composite 64  → expected ≥ 60
- Q3 (wrong, LLM=5,  correctness=0):    old composite 41  → expected ≤ 40
- Q4 (partial, LLM=50, correctness=?):  old composite 58  → 40–65 range

Each test fabricates the per-signal raw scores observed in that session and
runs them through _build_result_v2 directly so the assertions are stable
against unrelated changes (no model loading, no LLM calls).
"""
from app.config import get_settings
from app.services.scoring.claim_matcher import ClaimMatch
from app.services.scoring.claim_scorer import ClaimScoreResult
from app.services.scoring.composite_v2 import (
    _apply_correctness_gate,
    _build_result_v2,
)
from app.services.scoring.llm_scorer import LLMJudgeResult


def _make_llm(
    overall: float,
    correctness: float,
    completeness: float = 80.0,
    clarity: float = 85.0,
    depth: float = 75.0,
    is_fallback: bool = False,
) -> LLMJudgeResult:
    """Fabricate an LLMJudgeResult on the 0-100 input scale."""
    return LLMJudgeResult(
        normalized_score=overall / 100.0,
        reason="test fixture",
        correctness=correctness / 100.0,
        completeness=completeness / 100.0,
        clarity=clarity / 100.0,
        depth=depth / 100.0,
        is_fallback=is_fallback,
    )


def _make_claim_result(coverage_pct: float, matches: int = 5) -> ClaimScoreResult:
    """Fabricate a ClaimScoreResult with a target coverage."""
    coverage = coverage_pct / 100.0
    fake_matches = [
        ClaimMatch(
            claim=f"claim {i}",
            best_sentence=f"sentence {i}",
            similarity=coverage,
            entailment=coverage,
            combined=coverage,
            covered=coverage_pct >= 50,
            contradiction=0.0,
        )
        for i in range(matches)
    ]
    return ClaimScoreResult(
        normalized_score=coverage,
        coverage=coverage,
        hard_coverage=coverage,
        missing_claims=[],
        matches=fake_matches,
        avg_similarity=coverage,
        avg_entailment=coverage,
        avg_contradiction=0.0,
    )


def _score(
    sbert: float,
    nli: float,
    keyword: float,
    coverage: float,
    llm_overall: float,
    llm_correctness: float,
    is_fallback: bool = False,
) -> float:
    """Run the synchronous build path and return the final calibrated composite."""
    settings = get_settings()
    result = _build_result_v2(
        settings,
        sbert_raw=sbert / 100.0,
        nli_raw=nli / 100.0,
        keyword_raw=keyword / 100.0,
        keyword_missing=[],
        claim_result=_make_claim_result(coverage),
        llm_result=_make_llm(
            overall=llm_overall,
            correctness=llm_correctness,
            is_fallback=is_fallback,
        ),
    )
    return result.composite


class TestCorrectAnswersScoreHighEnough:
    """Correct answers should not be punished for phrasing differences."""

    def test_q1_class_imbalance_correct_answer_at_least_65(self):
        # Observed: SBERT=87, NLI=72, KW=88, coverage≈35, LLM=92, correctness=95
        # Old composite: 60 → new should be ≥ 65
        score = _score(
            sbert=87.0, nli=72.3, keyword=87.5,
            coverage=35.0,
            llm_overall=91.7, llm_correctness=95.0,
        )
        assert score >= 65.0, f"Q1 correct answer scored {score}, expected ≥ 65"

    def test_q2_transformer_vs_rnn_correct_answer_at_least_60(self):
        # Observed: SBERT=81, NLI=55, KW=88, coverage≈40, LLM=86, correctness=90
        # Old composite: 64 → new should be ≥ 60 (and ideally higher)
        score = _score(
            sbert=80.8, nli=55.4, keyword=87.5,
            coverage=40.0,
            llm_overall=85.5, llm_correctness=90.0,
        )
        assert score >= 60.0, f"Q2 correct answer scored {score}, expected ≥ 60"


class TestWrongAnswersGetGated:
    """Wrong-but-topical answers must not score in the 'Good' range."""

    def test_q3_factually_wrong_answer_capped_at_or_below_40(self):
        # Observed: SBERT=82, NLI=19, KW=100, coverage≈26, LLM=5, correctness=0
        # Old composite: 41 → with tiered gate at correctness=0, capped at 35
        score = _score(
            sbert=81.7, nli=18.9, keyword=100.0,
            coverage=26.0,
            llm_overall=5.0, llm_correctness=0.0,
        )
        assert score <= 40.0, f"Wrong answer scored {score}, expected ≤ 40"

    def test_gap_between_correct_and_wrong_widens(self):
        """The whole point: correct vs wrong should be clearly distinguishable."""
        correct = _score(
            sbert=87.0, nli=72.3, keyword=87.5, coverage=35.0,
            llm_overall=91.7, llm_correctness=95.0,
        )
        wrong = _score(
            sbert=81.7, nli=18.9, keyword=100.0, coverage=26.0,
            llm_overall=5.0, llm_correctness=0.0,
        )
        # Old gap was 60 - 41 = 19. New gap should be substantially wider.
        assert (correct - wrong) >= 30.0, (
            f"Gap correct({correct}) vs wrong({wrong}) = {correct - wrong},"
            f" expected ≥ 30"
        )


class TestPartialAnswerStaysInMidRange:
    def test_q4_partial_answer_lands_between_40_and_70(self):
        # Observed: SBERT=81, NLI=31, KW=75, coverage≈45, LLM=50
        # Partial answer (some right, some wrong) should sit in the middle.
        # Assume correctness around 50 — a mixed answer.
        score = _score(
            sbert=81.0, nli=30.7, keyword=75.0,
            coverage=45.0,
            llm_overall=50.0, llm_correctness=50.0,
        )
        assert 40.0 <= score <= 70.0, (
            f"Q4 partial answer scored {score}, expected 40–70"
        )


class TestTieredCorrectnessGate:
    """Direct unit tests on the gate function."""

    def test_severe_miss_caps_at_low_cap(self):
        settings = get_settings()
        capped = _apply_correctness_gate(
            settings, calibrated=85.0, llm_result=_make_llm(80, correctness=10),
        )
        assert capped == settings.LLM_CORRECTNESS_GATE_LOW_CAP

    def test_moderate_miss_caps_at_mid_cap(self):
        settings = get_settings()
        capped = _apply_correctness_gate(
            settings, calibrated=85.0, llm_result=_make_llm(80, correctness=30),
        )
        assert capped == settings.LLM_CORRECTNESS_GATE_MID_CAP

    def test_high_correctness_unchanged(self):
        settings = get_settings()
        capped = _apply_correctness_gate(
            settings, calibrated=85.0, llm_result=_make_llm(80, correctness=90),
        )
        assert capped == 85.0

    def test_low_input_below_cap_unchanged(self):
        """If the composite is already below the cap, don't raise it."""
        settings = get_settings()
        capped = _apply_correctness_gate(
            settings, calibrated=20.0, llm_result=_make_llm(80, correctness=10),
        )
        assert capped == 20.0

    def test_gate_skipped_for_fallback_llm(self):
        """When the LLM judge fails, we have no correctness signal; don't gate."""
        settings = get_settings()
        capped = _apply_correctness_gate(
            settings,
            calibrated=70.0,
            llm_result=_make_llm(0, correctness=0, is_fallback=True),
        )
        assert capped == 70.0


class TestClaimMatchesExposePartialAndMatchScore:
    """The UI relies on match_score (combined) and partial flag, not raw similarity."""

    def _build_with_match(self, *, combined: float) -> dict:
        settings = get_settings()
        match = ClaimMatch(
            claim="Should mention caching",
            best_sentence="some sentence",
            similarity=0.71,  # raw SBERT
            entailment=0.10,  # low — drags combined down
            combined=combined,
            covered=combined >= settings.CLAIM_MATCH_THRESHOLD,
            contradiction=0.0,
        )
        claim_result = ClaimScoreResult(
            normalized_score=combined,
            coverage=combined,
            hard_coverage=1.0 if match.covered else 0.0,
            missing_claims=[] if match.covered else [match.claim],
            matches=[match],
            avg_similarity=match.similarity,
            avg_entailment=match.entailment,
            avg_contradiction=0.0,
        )
        result = _build_result_v2(
            settings,
            sbert_raw=0.8,
            nli_raw=0.5,
            keyword_raw=0.7,
            keyword_missing=[],
            claim_result=claim_result,
            llm_result=_make_llm(80, correctness=80),
        )
        return result.claim_matches[0]

    def test_combined_below_threshold_above_soft_floor_is_partial(self):
        """71% similarity but low entailment → combined ~0.55 → partial."""
        cm = self._build_with_match(combined=0.55)
        assert cm["covered"] is False
        assert cm["partial"] is True
        assert cm["match_score"] == 0.55

    def test_combined_below_soft_floor_is_neither_covered_nor_partial(self):
        cm = self._build_with_match(combined=0.30)
        assert cm["covered"] is False
        assert cm["partial"] is False

    def test_combined_above_threshold_is_covered_not_partial(self):
        cm = self._build_with_match(combined=0.80)
        assert cm["covered"] is True
        assert cm["partial"] is False


class TestWeightsSumToOne:
    """Guard against future config edits drifting away from a valid mix."""

    def test_claim_weights_sum_to_one(self):
        s = get_settings()
        total = (
            s.CLAIM_SBERT_WEIGHT
            + s.CLAIM_NLI_WEIGHT
            + s.CLAIM_KEYWORD_WEIGHT
            + s.CLAIM_COVERAGE_WEIGHT
            + s.CLAIM_LLM_WEIGHT
        )
        assert abs(total - 1.0) < 1e-6, f"Claim weights sum to {total}, expected 1.0"

    def test_llm_judge_carries_most_weight_for_correctness(self):
        """Document the design intent: LLM rubric is the dominant signal."""
        s = get_settings()
        assert s.CLAIM_LLM_WEIGHT >= s.CLAIM_COVERAGE_WEIGHT, (
            "LLM judge must outweigh claim coverage to capture correctness"
        )
        assert s.CLAIM_LLM_WEIGHT >= s.CLAIM_SBERT_WEIGHT
