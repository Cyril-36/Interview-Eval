from app.services.scoring import behavioral_pipeline, composite_v2
from app.services.scoring.scoring_types import is_behavioral


async def evaluate(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
    category: str = "",
):
    pipeline = behavioral_pipeline if is_behavioral(category) else composite_v2
    return await pipeline.evaluate(
        candidate_answer,
        ideal_answer,
        question=question,
        category=category,
    )


async def evaluate_stepwise(
    candidate_answer: str,
    ideal_answer: str,
    question: str = "",
    on_progress=None,
    category: str = "",
):
    pipeline = behavioral_pipeline if is_behavioral(category) else composite_v2
    return await pipeline.evaluate_stepwise(
        candidate_answer,
        ideal_answer,
        question=question,
        on_progress=on_progress,
        category=category,
    )
