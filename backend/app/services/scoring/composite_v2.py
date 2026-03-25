import asyncio

from app.config import get_settings
from app.services.scoring import (
    claim_scorer,
    keyword_scorer,
    llm_scorer,
    nli_scorer,
    sbert_scorer,
)
from app.services.scoring.calibration import calibrate
from app.services.scoring.executor import executor as _executor
from app.services.scoring.scoring_types import ScoringResult, compute_grade


def _run_v2_nlp_scorers(candidate_answer: str, ideal_answer: str, question: str):
    sbert_raw = sbert_scorer.score(candidate_answer, ideal_answer)
    nli_raw = nli_scorer.score(candidate_answer, ideal_answer)
    keyword_raw, keyword_missing = keyword_scorer.score(candidate_answer, ideal_answer)
    claim_result = claim_scorer.score(candidate_answer, ideal_answer, question=question)
    return sbert_raw, nli_raw, keyword_raw, keyword_missing, claim_result


def _build_result_v2(
    settings,
    sbert_raw: float,
    nli_raw: float,
    keyword_raw: float,
    keyword_missing: list[str],
    claim_result,
    llm_result,
) -> ScoringResult:
    if llm_result.is_fallback:
        # Exclude LLM weight and renormalize remaining weights
        nlp_total = (
            settings.CLAIM_SBERT_WEIGHT
            + settings.CLAIM_NLI_WEIGHT
            + settings.CLAIM_KEYWORD_WEIGHT
            + settings.CLAIM_COVERAGE_WEIGHT
        )
        composite_raw = (
            (settings.CLAIM_SBERT_WEIGHT / nlp_total) * sbert_raw
            + (settings.CLAIM_NLI_WEIGHT / nlp_total) * nli_raw
            + (settings.CLAIM_KEYWORD_WEIGHT / nlp_total) * keyword_raw
            + (settings.CLAIM_COVERAGE_WEIGHT / nlp_total) * claim_result.coverage
        )
    else:
        composite_raw = (
            settings.CLAIM_SBERT_WEIGHT * sbert_raw
            + settings.CLAIM_NLI_WEIGHT * nli_raw
            + settings.CLAIM_KEYWORD_WEIGHT * keyword_raw
            + settings.CLAIM_COVERAGE_WEIGHT * claim_result.coverage
            + settings.CLAIM_LLM_WEIGHT * llm_result.normalized_score
        )
    composite_100 = round(composite_raw * 100, 1)
    calibrated = calibrate(composite_100, is_behavioral=False)

    missing_concepts = (
        claim_result.missing_claims if claim_result.missing_claims else keyword_missing
    )

    claim_matches_dicts = [
        {
            "claim": m.claim,
            "covered": m.covered,
            "similarity": round(m.similarity, 4),
            "contradiction": round(m.contradiction, 4),
        }
        for m in claim_result.matches
    ]

    return ScoringResult(
        sbert_raw=round(sbert_raw * 100, 1),
        nli_raw=round(nli_raw * 100, 1),
        keyword_raw=round(keyword_raw * 100, 1),
        llm_raw=round(llm_result.normalized_score * 100, 1),
        llm_reason=llm_result.reason,
        llm_correctness=round(llm_result.correctness * 100, 1),
        llm_completeness=round(llm_result.completeness * 100, 1),
        llm_clarity=round(llm_result.clarity * 100, 1),
        llm_depth=round(llm_result.depth * 100, 1),
        raw_composite=composite_100,
        composite=calibrated,
        grade=compute_grade(calibrated),
        missing_keywords=missing_concepts,
        is_behavioral=False,
        claim_matches=claim_matches_dicts,
    )


async def evaluate_stepwise(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
    on_progress=None,
    category: str = "",
) -> ScoringResult:
    del category
    settings = get_settings()
    loop = asyncio.get_event_loop()

    async def emit(step, **data):
        if on_progress:
            await on_progress(step, data)

    await emit("scoring_started", is_behavioral=False, pipeline="claim_v2")

    llm_future = asyncio.ensure_future(
        llm_scorer.score(candidate_answer, ideal_answer, question)
    )

    sbert_raw = await loop.run_in_executor(
        _executor, sbert_scorer.score, candidate_answer, ideal_answer
    )
    await emit("sbert_done", score=round(sbert_raw * 100, 1))

    nli_raw = await loop.run_in_executor(
        _executor, nli_scorer.score, candidate_answer, ideal_answer
    )
    await emit("nli_done", score=round(nli_raw * 100, 1))

    keyword_raw, keyword_missing = await loop.run_in_executor(
        _executor, keyword_scorer.score, candidate_answer, ideal_answer
    )
    await emit(
        "keyword_done",
        score=round(keyword_raw * 100, 1),
        missing=keyword_missing,
    )

    claim_result = await loop.run_in_executor(
        _executor, claim_scorer.score, candidate_answer, ideal_answer, question
    )
    await emit(
        "claim_done",
        score=round(claim_result.coverage * 100, 1),
        missing=claim_result.missing_claims,
    )

    llm_result = await llm_future
    await emit("llm_done", score=round(llm_result.normalized_score * 100, 1))

    result = _build_result_v2(
        settings,
        sbert_raw,
        nli_raw,
        keyword_raw,
        keyword_missing,
        claim_result,
        llm_result,
    )
    await emit("scores_ready")
    return result


async def evaluate(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
    category: str = "",
) -> ScoringResult:
    del category
    settings = get_settings()
    loop = asyncio.get_event_loop()

    nlp_future = loop.run_in_executor(
        _executor, _run_v2_nlp_scorers, candidate_answer, ideal_answer, question
    )
    llm_future = llm_scorer.score(candidate_answer, ideal_answer, question)

    (
        sbert_raw,
        nli_raw,
        keyword_raw,
        keyword_missing,
        claim_result,
    ), llm_result = await asyncio.gather(nlp_future, llm_future)

    return _build_result_v2(
        settings,
        sbert_raw,
        nli_raw,
        keyword_raw,
        keyword_missing,
        claim_result,
        llm_result,
    )
