from fastapi import APIRouter, HTTPException

from app.schemas import (
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    ScoreBreakdown,
    FeedbackOut,
)
from app.services.scoring.composite import evaluate
from app.services.feedback_generator import generate_feedback
from app.services.session_manager import session_manager, AnswerResult

router = APIRouter()


@router.post("/evaluate_answer", response_model=EvaluateAnswerResponse)
async def evaluate_answer_endpoint(request: EvaluateAnswerRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.question_index >= len(session.questions):
        raise HTTPException(status_code=400, detail="Invalid question index")

    question = session.questions[request.question_index]

    # Run scoring pipeline
    scoring_result = evaluate(request.candidate_answer, question.ideal_answer)

    # Generate feedback
    feedback_data = await generate_feedback(
        question=question.question_text,
        ideal_answer=question.ideal_answer,
        candidate_answer=request.candidate_answer,
        sbert_score=scoring_result.sbert_raw,
        nli_score=scoring_result.nli_raw,
        keyword_score=scoring_result.keyword_raw,
        composite_score=scoring_result.composite,
        missing_keywords=scoring_result.missing_keywords,
    )

    # Store answer in session
    answer_result = AnswerResult(
        question_index=request.question_index,
        candidate_answer=request.candidate_answer,
        sbert_score=scoring_result.sbert_raw,
        nli_score=scoring_result.nli_raw,
        keyword_score=scoring_result.keyword_raw,
        composite_score=scoring_result.composite,
        grade=scoring_result.grade,
        missing_keywords=scoring_result.missing_keywords,
        feedback=feedback_data,
    )
    session_manager.add_answer(request.session_id, answer_result)

    questions_remaining = len(session.questions) - len(session.answers)
    is_last = questions_remaining <= 0

    return EvaluateAnswerResponse(
        scores=ScoreBreakdown(
            sbert_score=scoring_result.sbert_raw,
            nli_score=scoring_result.nli_raw,
            keyword_score=scoring_result.keyword_raw,
            composite_score=scoring_result.composite,
            grade=scoring_result.grade,
            missing_keywords=scoring_result.missing_keywords,
        ),
        feedback=FeedbackOut(
            strengths=feedback_data.get("strengths", []),
            improvements=feedback_data.get("improvements", []),
            model_answer=feedback_data.get("model_answer", ""),
        ),
        question_text=question.question_text,
        is_last_question=is_last,
        questions_remaining=max(0, questions_remaining),
    )
