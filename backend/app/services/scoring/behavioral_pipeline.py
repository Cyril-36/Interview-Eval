import asyncio

from app.config import get_settings
from app.services.scoring import keyword_scorer, nli_scorer, sbert_scorer, star_scorer
from app.services.scoring.calibration import calibrate
from app.services.scoring.executor import executor as _executor
from app.services.scoring.scoring_types import ScoringResult, compute_grade


def _run_nlp_scorers(candidate_answer: str, ideal_answer: str):
    sbert_raw = sbert_scorer.score(candidate_answer, ideal_answer)
    nli_raw = nli_scorer.score(candidate_answer, ideal_answer)
    keyword_raw, missing_keywords = keyword_scorer.score(
        candidate_answer, ideal_answer
    )
    return sbert_raw, nli_raw, keyword_raw, missing_keywords


def _run_sbert(candidate_answer: str, ideal_answer: str):
    return sbert_scorer.score(candidate_answer, ideal_answer)


def _run_nli(candidate_answer: str, ideal_answer: str):
    return nli_scorer.score(candidate_answer, ideal_answer)


def _run_keyword(candidate_answer: str, ideal_answer: str):
    return keyword_scorer.score(candidate_answer, ideal_answer)


def _build_result(
    settings,
    sbert_raw: float,
    nli_raw: float,
    keyword_raw: float,
    missing_keywords: list[str],
    star_result,
) -> ScoringResult:
    if star_result.is_fallback:
        nlp_total = (
            settings.BEHAVIORAL_SBERT_WEIGHT
            + settings.BEHAVIORAL_NLI_WEIGHT
            + settings.BEHAVIORAL_KEYWORD_WEIGHT
        )
        composite_raw = (
            (settings.BEHAVIORAL_SBERT_WEIGHT / nlp_total) * sbert_raw
            + (settings.BEHAVIORAL_NLI_WEIGHT / nlp_total) * nli_raw
            + (settings.BEHAVIORAL_KEYWORD_WEIGHT / nlp_total) * keyword_raw
        )
    else:
        composite_raw = (
            settings.BEHAVIORAL_SBERT_WEIGHT * sbert_raw
            + settings.BEHAVIORAL_NLI_WEIGHT * nli_raw
            + settings.BEHAVIORAL_KEYWORD_WEIGHT * keyword_raw
            + settings.BEHAVIORAL_LLM_WEIGHT * star_result.normalized_score
        )

    composite_100 = round(composite_raw * 100, 1)
    calibrated = calibrate(composite_100, is_behavioral=True)

    return ScoringResult(
        sbert_raw=round(sbert_raw * 100, 1),
        nli_raw=round(nli_raw * 100, 1),
        keyword_raw=round(keyword_raw * 100, 1),
        llm_raw=round(star_result.normalized_score * 100, 1),
        llm_reason=star_result.reason,
        llm_correctness=0.0,
        llm_completeness=0.0,
        llm_clarity=0.0,
        llm_depth=0.0,
        raw_composite=composite_100,
        composite=calibrated,
        grade=compute_grade(calibrated),
        missing_keywords=missing_keywords,
        is_behavioral=True,
        star_situation=round(star_result.situation * 100, 1),
        star_task=round(star_result.task * 100, 1),
        star_action=round(star_result.action * 100, 1),
        star_result=round(star_result.result * 100, 1),
        star_reflection=round(star_result.reflection * 100, 1),
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

    await emit("scoring_started", is_behavioral=True)

    star_future = asyncio.ensure_future(
        star_scorer.score(candidate_answer, ideal_answer, question)
    )

    sbert_raw = await loop.run_in_executor(
        _executor, _run_sbert, candidate_answer, ideal_answer,
    )
    await emit("sbert_done", score=round(sbert_raw * 100, 1))

    nli_raw = await loop.run_in_executor(
        _executor, _run_nli, candidate_answer, ideal_answer,
    )
    await emit("nli_done", score=round(nli_raw * 100, 1))

    keyword_raw, missing_keywords = await loop.run_in_executor(
        _executor, _run_keyword, candidate_answer, ideal_answer,
    )
    await emit(
        "keyword_done",
        score=round(keyword_raw * 100, 1),
        missing=missing_keywords,
    )

    star_result = await star_future
    await emit(
        "llm_done",
        score=round(star_result.normalized_score * 100, 1),
        scorer="star",
    )

    result = _build_result(
        settings,
        sbert_raw,
        nli_raw,
        keyword_raw,
        missing_keywords,
        star_result,
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
        _executor, _run_nlp_scorers, candidate_answer, ideal_answer,
    )
    star_future = star_scorer.score(candidate_answer, ideal_answer, question)

    (
        sbert_raw,
        nli_raw,
        keyword_raw,
        missing_keywords,
    ), star_result = await asyncio.gather(nlp_future, star_future)

    return _build_result(
        settings,
        sbert_raw,
        nli_raw,
        keyword_raw,
        missing_keywords,
        star_result,
    )
