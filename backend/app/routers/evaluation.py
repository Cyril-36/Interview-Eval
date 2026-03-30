import asyncio
import json
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import (
    ClaimFeedback,
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    FeedbackOut,
    RubricScores,
    ScoreBreakdown,
    STARScores,
)
from app.services.feedback_generator import generate_feedback
from app.services.scoring.pipeline import evaluate, evaluate_stepwise
from app.services.session_manager import (
    AnswerResult,
    SessionStoreError,
    session_manager,
)

router = APIRouter()


@router.post("/evaluate_answer_sse")
async def evaluate_answer_sse(
    request: EvaluateAnswerRequest,
    http_request: Request,
):
    """SSE endpoint that streams scoring progress and final results."""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.question_index >= len(session.questions):
        raise HTTPException(status_code=400, detail="Invalid question index")

    # Removed AnswerReservationStatus logic

    question = session.questions[request.question_index]

    async def event_stream():
        progress_queue: asyncio.Queue = asyncio.Queue()
        scoring_task = None
        feedback_task = None
        client_disconnected = False

        async def on_progress(step, data):
            await progress_queue.put({"step": step, **data})

        async def cancel_task(task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        try:
            scoring_task = asyncio.create_task(
                evaluate_stepwise(
                    request.candidate_answer,
                    question.ideal_answer,
                    question.question_text,
                    on_progress=on_progress,
                    category=question.category,
                )
            )

            while not scoring_task.done():
                if not client_disconnected and await http_request.is_disconnected():
                    client_disconnected = True
                try:
                    event = await asyncio.wait_for(
                        progress_queue.get(), timeout=0.1
                    )
                    if not client_disconnected:
                        yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    continue

            while not progress_queue.empty():
                event = await progress_queue.get()
                if not client_disconnected:
                    yield f"data: {json.dumps(event)}\n\n"

            scoring_result = scoring_task.result()

            if not client_disconnected:
                yield f"data: {json.dumps({'step': 'feedback_started'})}\n\n"

            feedback_task = asyncio.create_task(
                generate_feedback(
                    question=question.question_text,
                    ideal_answer=question.ideal_answer,
                    candidate_answer=request.candidate_answer,
                    sbert_score=scoring_result.sbert_raw,
                    nli_score=scoring_result.nli_raw,
                    keyword_score=scoring_result.keyword_raw,
                    composite_score=scoring_result.composite,
                    missing_keywords=scoring_result.missing_keywords,
                )
            )

            while not feedback_task.done():
                if not client_disconnected and await http_request.is_disconnected():
                    client_disconnected = True
                await asyncio.sleep(0.1)

            feedback_data = feedback_task.result()

            answer_result = AnswerResult(
                question_index=request.question_index,
                candidate_answer=request.candidate_answer,
                sbert_score=scoring_result.sbert_raw,
                nli_score=scoring_result.nli_raw,
                keyword_score=scoring_result.keyword_raw,
                llm_score=scoring_result.llm_raw,
                llm_reason=scoring_result.llm_reason,
                composite_score=scoring_result.composite,
                grade=scoring_result.grade,
                missing_keywords=scoring_result.missing_keywords,
                feedback=feedback_data,
                claim_matches=scoring_result.claim_matches,
                llm_correctness=scoring_result.llm_correctness,
                llm_completeness=scoring_result.llm_completeness,
                llm_clarity=scoring_result.llm_clarity,
                llm_depth=scoring_result.llm_depth,
                is_behavioral=scoring_result.is_behavioral,
                star_situation=scoring_result.star_situation,
                star_task=scoring_result.star_task,
                star_action=scoring_result.star_action,
                star_result=scoring_result.star_result,
                star_reflection=scoring_result.star_reflection,
            )
            session_manager.add_answer(request.session_id, answer_result)

            updated_session = session_manager.get_session(request.session_id)
            answers_count = len(updated_session.answers) if updated_session else 0
            questions_remaining = len(session.questions) - answers_count
            is_last = questions_remaining <= 0

            star_payload = None
            if scoring_result.is_behavioral:
                star_payload = {
                    "situation": scoring_result.star_situation,
                    "task": scoring_result.star_task,
                    "action": scoring_result.star_action,
                    "result": scoring_result.star_result,
                    "reflection": scoring_result.star_reflection,
                }

            final = {
                "step": "done",
                "scores": {
                    "sbert_score": scoring_result.sbert_raw,
                    "nli_score": scoring_result.nli_raw,
                    "keyword_score": scoring_result.keyword_raw,
                    "llm_score": scoring_result.llm_raw,
                    "llm_reason": scoring_result.llm_reason,
                    "rubric_scores": {
                        "correctness": scoring_result.llm_correctness,
                        "completeness": scoring_result.llm_completeness,
                        "clarity": scoring_result.llm_clarity,
                        "depth": scoring_result.llm_depth,
                    },
                    "composite_score": scoring_result.composite,
                    "grade": scoring_result.grade,
                    "missing_keywords": scoring_result.missing_keywords,
                    "is_behavioral": scoring_result.is_behavioral,
                    "star_scores": star_payload,
                    "claim_matches": scoring_result.claim_matches,
                },
                "feedback": {
                    "strengths": feedback_data.get("strengths", []),
                    "improvements": feedback_data.get("improvements", []),
                    "model_answer": feedback_data.get("model_answer", ""),
                },
                "question_text": question.question_text,
                "is_last_question": is_last,
                "questions_remaining": max(0, questions_remaining),
            }
            if not client_disconnected:
                yield f"data: {json.dumps(final)}\n\n"

        except asyncio.CancelledError:
            return
        except SessionStoreError as exc:
            if not client_disconnected:
                yield f"data: {json.dumps({'step': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            if not client_disconnected and not await http_request.is_disconnected():
                yield f"data: {json.dumps({'step': 'error', 'message': str(exc)})}\n\n"
        finally:
            await cancel_task(scoring_task)
            await cancel_task(feedback_task)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/evaluate_answer", response_model=EvaluateAnswerResponse)
async def evaluate_answer_endpoint(request: EvaluateAnswerRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.question_index >= len(session.questions):
        raise HTTPException(status_code=400, detail="Invalid question index")

    # Removed AnswerReservationStatus logic

    question = session.questions[request.question_index]

    scoring_result = await evaluate(
        request.candidate_answer, question.ideal_answer, question.question_text,
        category=question.category,
    )

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

    answer_result = AnswerResult(
        question_index=request.question_index,
        candidate_answer=request.candidate_answer,
        sbert_score=scoring_result.sbert_raw,
        nli_score=scoring_result.nli_raw,
        keyword_score=scoring_result.keyword_raw,
        llm_score=scoring_result.llm_raw,
        llm_reason=scoring_result.llm_reason,
        composite_score=scoring_result.composite,
        grade=scoring_result.grade,
        missing_keywords=scoring_result.missing_keywords,
        feedback=feedback_data,
        claim_matches=scoring_result.claim_matches,
        llm_correctness=scoring_result.llm_correctness,
        llm_completeness=scoring_result.llm_completeness,
        llm_clarity=scoring_result.llm_clarity,
        llm_depth=scoring_result.llm_depth,
        is_behavioral=scoring_result.is_behavioral,
        star_situation=scoring_result.star_situation,
        star_task=scoring_result.star_task,
        star_action=scoring_result.star_action,
        star_result=scoring_result.star_result,
        star_reflection=scoring_result.star_reflection,
    )
    session_manager.add_answer(request.session_id, answer_result)

    updated_session = session_manager.get_session(request.session_id)
    answers_count = len(updated_session.answers) if updated_session else 0
    questions_remaining = len(session.questions) - answers_count
    is_last = questions_remaining <= 0

    star_scores = None
    if scoring_result.is_behavioral:
        star_scores = STARScores(
            situation=scoring_result.star_situation,
            task=scoring_result.star_task,
            action=scoring_result.star_action,
            result=scoring_result.star_result,
            reflection=scoring_result.star_reflection,
        )

    return EvaluateAnswerResponse(
        scores=ScoreBreakdown(
            sbert_score=scoring_result.sbert_raw,
            nli_score=scoring_result.nli_raw,
            keyword_score=scoring_result.keyword_raw,
            llm_score=scoring_result.llm_raw,
            llm_reason=scoring_result.llm_reason,
            rubric_scores=RubricScores(
                correctness=scoring_result.llm_correctness,
                completeness=scoring_result.llm_completeness,
                clarity=scoring_result.llm_clarity,
                depth=scoring_result.llm_depth,
            ),
            composite_score=scoring_result.composite,
            grade=scoring_result.grade,
            missing_keywords=scoring_result.missing_keywords,
            is_behavioral=scoring_result.is_behavioral,
            star_scores=star_scores,
            claim_matches=[
                ClaimFeedback(**cm) for cm in scoring_result.claim_matches
            ],
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
