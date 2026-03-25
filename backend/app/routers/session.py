from fastapi import APIRouter, HTTPException, Query

from app.schemas import SessionSummaryResponse, QuestionResult, RubricScores, STARScores
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
