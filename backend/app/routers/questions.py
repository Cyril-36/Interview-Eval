from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionOut,
    ResumeParseResponse,
)
from app.services.question_generator import generate_questions
from app.services.resume_parser import ResumeParseError, parse_pdf
from app.services.session_manager import session_manager, QuestionItem

router = APIRouter()


@router.post("/generate_questions", response_model=GenerateQuestionsResponse)
async def generate_questions_endpoint(request: GenerateQuestionsRequest):
    # Create session
    session = session_manager.create_session(
        role=request.role,
        level=request.level,
        category=request.category,
        difficulty=request.difficulty,
    )

    try:
        # Generate questions via LLM + diversity filter
        raw_questions = await generate_questions(
            role=request.role,
            level=request.level,
            category=request.category,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            resume_context=request.resume_context,
            extracted_skills=request.extracted_skills,
        )
    except Exception as e:
        # Clean up orphan session on failed generation
        session_manager.delete_session(session.session_id)
        raise HTTPException(status_code=500, detail=str(e))

    # Store in session
    question_items = [
        QuestionItem(
            index=i,
            question_text=q["question_text"],
            ideal_answer=q["ideal_answer"],
            category=q.get("category", request.category),
            difficulty=q.get("difficulty", request.difficulty),
        )
        for i, q in enumerate(raw_questions)
    ]
    session_manager.set_questions(session.session_id, question_items)

    # Return questions (without ideal answers)
    questions_out = [
        QuestionOut(
            index=item.index,
            question_text=item.question_text,
            category=item.category,
            difficulty=item.difficulty,
        )
        for item in question_items
    ]

    return GenerateQuestionsResponse(
        session_id=session.session_id,
        session_token=session.access_token or "",
        questions=questions_out,
        total_questions=len(questions_out),
    )


@router.post("/parse_resume", response_model=ResumeParseResponse)
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Parse an uploaded resume PDF into text and extracted skills."""
    settings = get_settings()

    if file.content_type and file.content_type not in {
        "application/pdf", "application/x-pdf",
    }:
        raise HTTPException(
            status_code=415, detail="Only PDF resumes are supported",
        )

    body = await file.read()
    if len(body) > settings.RESUME_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Resume too large: {len(body)} bytes"
                f" (limit {settings.RESUME_MAX_BYTES})"
            ),
        )

    try:
        parsed = parse_pdf(body)
    except ResumeParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ResumeParseResponse(
        extracted_text=parsed.text,
        skills=parsed.skills,
        summary=parsed.summary(),
        page_count=parsed.page_count,
        truncated=parsed.truncated,
    )
