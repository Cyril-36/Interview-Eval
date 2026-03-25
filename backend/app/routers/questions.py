from fastapi import APIRouter, HTTPException

from app.schemas import GenerateQuestionsRequest, GenerateQuestionsResponse, QuestionOut
from app.services.question_generator import generate_questions
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
        questions=questions_out,
        total_questions=len(questions_out),
    )
