import pytest

from app.services.scoring import claim_extractor
from app.services.scoring.claim_extractor import extract_claims
from app.services.scoring.claim_matcher import ClaimMatch
from app.services.scoring import claim_scorer, pipeline


def test_extract_claims_splits_sentences_and_clauses():
    ideal = (
        "I cleaned the dataset and removed duplicates. "
        "I then trained a baseline model, and I compared it against a tuned model."
    )

    claims = extract_claims(ideal, max_claims=6)

    assert len(claims) >= 3
    assert claims[0] == "I cleaned the dataset and removed duplicates"
    assert any("trained a baseline model" in claim for claim in claims)


def test_extract_claims_uses_llm_when_enabled(monkeypatch):
    monkeypatch.setattr(
        claim_extractor,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "CLAIM_EXTRACTION_MODE": "llm",
                "GROQ_API_KEY": "test-key",
                "CLAIM_EXTRACTION_MODEL": "test-model",
            },
        )(),
    )
    monkeypatch.setattr(
        claim_extractor,
        "_extract_claims_with_llm",
        lambda question, ideal_answer, max_claims, model: (
            "First atomic claim",
            "Second atomic claim",
        ),
    )

    claims = extract_claims(
        "Ideal answer text.",
        question="What is the best answer?",
        max_claims=4,
    )

    assert claims == ["First atomic claim", "Second atomic claim"]


def test_extract_claims_falls_back_to_regex_on_llm_failure(monkeypatch):
    monkeypatch.setattr(
        claim_extractor,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "CLAIM_EXTRACTION_MODE": "llm",
                "GROQ_API_KEY": "test-key",
                "CLAIM_EXTRACTION_MODEL": "test-model",
            },
        )(),
    )

    def boom(question, ideal_answer, max_claims, model):
        raise RuntimeError("api down")

    monkeypatch.setattr(claim_extractor, "_extract_claims_with_llm", boom)

    claims = extract_claims(
        "I cleaned the data. I trained a model and compared results.",
        question="Describe your approach.",
        max_claims=4,
    )

    assert claims
    assert any("cleaned the data" in claim.lower() for claim in claims)


def test_claim_scorer_aggregates_matches(monkeypatch):
    monkeypatch.setattr(
        claim_scorer,
        "extract_claims",
        lambda ideal_answer, question="", max_claims=6: ["claim one", "claim two"],
    )
    monkeypatch.setattr(
        claim_scorer,
        "match_claims",
        lambda candidate_answer, claims, threshold=0.62: [
            ClaimMatch(
                claim="claim one",
                best_sentence="sentence one",
                similarity=0.8,
                entailment=0.7,
                combined=0.77,
                covered=True,
            ),
            ClaimMatch(
                claim="claim two",
                best_sentence="sentence two",
                similarity=0.4,
                entailment=0.5,
                combined=0.43,
                covered=False,
            ),
        ],
    )

    result = claim_scorer.score("candidate", "ideal")

    assert result.coverage == pytest.approx(0.5, rel=1e-6)
    assert result.hard_coverage == pytest.approx(0.5, rel=1e-6)
    assert result.normalized_score == pytest.approx(0.6, rel=1e-6)
    assert result.missing_claims == ["claim two"]


def test_claim_scorer_uses_soft_coverage(monkeypatch):
    monkeypatch.setattr(
        claim_scorer,
        "extract_claims",
        lambda ideal_answer, question="", max_claims=6: ["claim one", "claim two"],
    )
    monkeypatch.setattr(
        claim_scorer,
        "match_claims",
        lambda candidate_answer, claims, threshold=0.62: [
            ClaimMatch(
                claim="claim one",
                best_sentence="sentence one",
                similarity=0.8,
                entailment=0.7,
                combined=0.62,
                covered=True,
                contradiction=0.0,
            ),
            ClaimMatch(
                claim="claim two",
                best_sentence="sentence two",
                similarity=0.5,
                entailment=0.5,
                combined=0.55,
                covered=False,
                contradiction=0.2,
            ),
        ],
    )
    monkeypatch.setattr(
        claim_scorer,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "CLAIM_MAX_CLAIMS": 6,
                "CLAIM_MATCH_THRESHOLD": 0.62,
                "CLAIM_SOFT_MARGIN": 0.15,
                "KEYWORD_TOP_N": 8,
            },
        )(),
    )

    result = claim_scorer.score("candidate", "ideal")

    # One full match plus one partial match inside the soft band.
    assert result.coverage == pytest.approx((1.0 + (0.55 - 0.47) / 0.15) / 2, rel=1e-6)
    assert result.hard_coverage == pytest.approx(0.5, rel=1e-6)
    assert result.avg_contradiction == pytest.approx(0.1, rel=1e-6)


@pytest.mark.asyncio
async def test_pipeline_dispatches_technical_to_claim_v2(monkeypatch):
    async def fake_v2(*args, **kwargs):
        return "claim-v2"

    async def fake_behavioral(*args, **kwargs):
        return "behavioral"

    monkeypatch.setattr(pipeline.composite_v2, "evaluate", fake_v2)
    monkeypatch.setattr(pipeline.behavioral_pipeline, "evaluate", fake_behavioral)

    result = await pipeline.evaluate("candidate", "ideal", category="Databases")

    assert result == "claim-v2"


@pytest.mark.asyncio
async def test_pipeline_dispatches_behavioral_to_behavioral_pipeline(monkeypatch):
    async def fake_v2(*args, **kwargs):
        return "claim-v2"

    async def fake_behavioral(*args, **kwargs):
        return "behavioral"

    monkeypatch.setattr(pipeline.composite_v2, "evaluate", fake_v2)
    monkeypatch.setattr(pipeline.behavioral_pipeline, "evaluate", fake_behavioral)

    result = await pipeline.evaluate("candidate", "ideal", category="Behavioral")

    assert result == "behavioral"
