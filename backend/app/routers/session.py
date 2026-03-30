from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    ClaimFeedback,
    EvaluateAnswerResponse,
    FeedbackOut,
    QuestionResult,
    RubricScores,
    ScoreBreakdown,
    SessionSummaryResponse,
    STARScores,
)
from app.services.session_manager import session_manager

router = APIRouter()


@router.get("/session_status")
async def session_status_endpoint(session_id: str = Query(...)):
    """Lightweight check: does this session exist and how far along is it?"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "state": session.state.value,
        "total_questions": len(session.questions),
        "answers_count": len(session.answers),
    }


@router.get("/answer_result", response_model=EvaluateAnswerResponse)
async def answer_result_endpoint(
    session_id: str = Query(...),
    question_index: int = Query(..., ge=0),
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if question_index >= len(session.questions):
        raise HTTPException(status_code=400, detail="Invalid question index")

    answer = session_manager.get_answer(session_id, question_index)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")

    question = session.questions[question_index]
    questions_remaining = len(session.questions) - len(session.answers)
    star_scores = None
    if answer.is_behavioral:
        star_scores = STARScores(
            situation=answer.star_situation,
            task=answer.star_task,
            action=answer.star_action,
            result=answer.star_result,
            reflection=answer.star_reflection,
        )

    return EvaluateAnswerResponse(
        scores=ScoreBreakdown(
            sbert_score=answer.sbert_score,
            nli_score=answer.nli_score,
            keyword_score=answer.keyword_score,
            llm_score=answer.llm_score,
            llm_reason=answer.llm_reason,
            rubric_scores=RubricScores(
                correctness=answer.llm_correctness,
                completeness=answer.llm_completeness,
                clarity=answer.llm_clarity,
                depth=answer.llm_depth,
            ),
            composite_score=answer.composite_score,
            grade=answer.grade,
            missing_keywords=answer.missing_keywords,
            is_behavioral=answer.is_behavioral,
            star_scores=star_scores,
            claim_matches=[
                ClaimFeedback(**cm) for cm in answer.claim_matches
            ],
        ),
        feedback=FeedbackOut(
            strengths=answer.feedback.get("strengths", []),
            improvements=answer.feedback.get("improvements", []),
            model_answer=answer.feedback.get("model_answer", ""),
        ),
        question_text=question.question_text,
        is_last_question=questions_remaining <= 0,
        questions_remaining=max(0, questions_remaining),
    )


@router.get("/session_summary", response_model=SessionSummaryResponse)
async def session_summary_endpoint(session_id: str = Query(...)):
    summary = session_manager.get_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found or no answers yet")

    avg_star = None
    if summary.get("avg_star"):
        avg_star = STARScores(**summary["avg_star"])

    results = []
    for r in summary["results"]:
        r_copy = dict(r)
        star_data = r_copy.pop("star_scores", None)
        if star_data:
            r_copy["star_scores"] = STARScores(**star_data)
        results.append(QuestionResult(**r_copy))

    return SessionSummaryResponse(
        session_id=summary["session_id"],
        role=summary["role"],
        level=summary["level"],
        category=summary["category"],
        total_questions=summary["total_questions"],
        questions_answered=summary["questions_answered"],
        average_score=summary["average_score"],
        results=results,
        strongest_area=summary["strongest_area"],
        weakest_area=summary["weakest_area"],
        overall_grade=summary["overall_grade"],
        avg_rubric=RubricScores(**summary["avg_rubric"]) if summary.get("avg_rubric") else None,
        avg_star=avg_star,
    )
