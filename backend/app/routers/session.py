from fastapi import APIRouter, HTTPException, Query

from app.schemas import SessionSummaryResponse, QuestionResult
from app.services.session_manager import session_manager

router = APIRouter()


@router.get("/session_summary", response_model=SessionSummaryResponse)
async def session_summary_endpoint(session_id: str = Query(...)):
    summary = session_manager.get_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found or no answers yet")

    return SessionSummaryResponse(
        session_id=summary["session_id"],
        role=summary["role"],
        level=summary["level"],
        category=summary["category"],
        total_questions=summary["total_questions"],
        questions_answered=summary["questions_answered"],
        average_score=summary["average_score"],
        results=[QuestionResult(**r) for r in summary["results"]],
        strongest_area=summary["strongest_area"],
        weakest_area=summary["weakest_area"],
        overall_grade=summary["overall_grade"],
    )
