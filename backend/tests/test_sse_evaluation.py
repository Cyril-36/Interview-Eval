"""Tests for the SSE evaluation endpoint /api/evaluate_answer_sse."""
import json

import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from app.services.session_manager import (
    SessionManager,
    QuestionItem,
    AnswerResult,
)
from app.services.scoring.scoring_types import ScoringResult


@pytest.fixture()
def client(tmp_path):
    """Create a test client with a temp DB."""
    with patch("app.services.session_manager.DATA_DIR", tmp_path), \
         patch("app.services.session_manager.DB_PATH", tmp_path / "sessions.db"):
        import app.services.session_manager as sm_mod
        original = sm_mod.session_manager
        sm_mod.session_manager = SessionManager()

        with patch("app.routers.evaluation.session_manager", sm_mod.session_manager), \
             patch("app.routers.questions.session_manager", sm_mod.session_manager), \
             patch("app.routers.session.session_manager", sm_mod.session_manager), \
             patch("app.main.ModelRegistry.load_all", return_value=None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as tc:
                yield tc, sm_mod.session_manager

        sm_mod.session_manager = original


def _setup_session_with_questions(manager, n_questions=2):
    """Helper to create a session with questions."""
    session = manager.create_session("SWE", "Junior", "Python", "Medium")
    questions = [
        QuestionItem(
            index=i,
            question_text=f"Question {i}?",
            ideal_answer=f"Answer {i}",
            category="Python",
            difficulty="Medium",
        )
        for i in range(n_questions)
    ]
    manager.set_questions(session.session_id, questions)
    return session


def _make_scoring_result():
    return ScoringResult(
        sbert_raw=75.0, nli_raw=70.0, keyword_raw=80.0,
        llm_raw=65.0, raw_composite=72.5, composite=72.5, grade="Good",
        missing_keywords=[], llm_reason="Good answer.",
        llm_correctness=72.0, llm_completeness=68.0,
        llm_clarity=66.0, llm_depth=54.0,
    )


def _parse_sse_events(response_text):
    """Parse SSE response body into a list of parsed JSON objects."""
    events = []
    for chunk in response_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                data = line[len("data: "):]
                events.append(json.loads(data))
    return events


class TestSSEProgressEventOrdering:
    def test_step_order(self, client):
        """Verify progress events arrive in the expected order."""
        tc, manager = client
        session = _setup_session_with_questions(manager)

        async def mock_stepwise(candidate, ideal, question="", on_progress=None, category=""):
            if on_progress:
                await on_progress("scoring_started", {})
                await on_progress("sbert_done", {"score": 75.0})
                await on_progress("nli_done", {"score": 70.0})
                await on_progress("keyword_done", {"score": 80.0, "missing": []})
                await on_progress("llm_done", {"score": 65.0})
                await on_progress("scores_ready", {})
            return _make_scoring_result()

        mock_feedback = AsyncMock(return_value={
            "strengths": ["Good"], "improvements": ["More detail"],
            "model_answer": "Model answer text",
        })

        with patch("app.routers.evaluation.evaluate_stepwise", side_effect=mock_stepwise), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            resp = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer",
            })

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        steps = [e["step"] for e in events]
        expected = [
            "scoring_started", "sbert_done", "nli_done",
            "keyword_done", "llm_done", "scores_ready",
            "feedback_started", "done",
        ]
        assert steps == expected


class TestSSEErrorEvent:
    def test_server_failure_emits_error(self, client):
        """When evaluate_stepwise raises, stream should emit an error event."""
        tc, manager = client
        session = _setup_session_with_questions(manager)

        async def mock_stepwise_fail(candidate, ideal, question="", on_progress=None, category=""):
            raise RuntimeError("Scoring engine exploded")

        with patch("app.routers.evaluation.evaluate_stepwise", side_effect=mock_stepwise_fail):
            resp = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer",
            })

        assert resp.status_code == 200  # SSE always returns 200; error is in-band
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) == 1
        assert "Scoring engine exploded" in error_events[0]["message"]


class TestSSEDonePayloadStructure:
    def test_done_event_contains_required_keys(self, client):
        """The final 'done' event must contain scores, feedback, and session info."""
        tc, manager = client
        session = _setup_session_with_questions(manager, n_questions=2)

        async def mock_stepwise(candidate, ideal, question="", on_progress=None, category=""):
            if on_progress:
                await on_progress("scoring_started", {})
                await on_progress("sbert_done", {"score": 75.0})
                await on_progress("nli_done", {"score": 70.0})
                await on_progress("keyword_done", {"score": 80.0, "missing": []})
                await on_progress("llm_done", {"score": 65.0})
                await on_progress("scores_ready", {})
            return _make_scoring_result()

        mock_feedback = AsyncMock(return_value={
            "strengths": ["Clear"], "improvements": ["Depth"],
            "model_answer": "Ideal model answer",
        })

        with patch("app.routers.evaluation.evaluate_stepwise", side_effect=mock_stepwise), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            resp = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer",
            })

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e.get("step") == "done"]
        assert len(done_events) == 1
        done = done_events[0]

        # Top-level keys
        assert "scores" in done
        assert "feedback" in done
        assert "question_text" in done
        assert "is_last_question" in done
        assert "questions_remaining" in done

        # Verify values
        assert done["question_text"] == "Question 0?"
        assert done["is_last_question"] is False
        assert done["questions_remaining"] == 1

        # Score structure
        scores = done["scores"]
        assert scores["sbert_score"] == 75.0
        assert scores["nli_score"] == 70.0
        assert scores["keyword_score"] == 80.0
        assert scores["llm_score"] == 65.0
        assert scores["composite_score"] == 72.5
        assert scores["grade"] == "Good"
        assert scores["missing_keywords"] == []
        assert scores["llm_reason"] == "Good answer."
        assert scores["rubric_scores"]["correctness"] == 72.0
        assert scores["rubric_scores"]["completeness"] == 68.0
        assert scores["rubric_scores"]["clarity"] == 66.0
        assert scores["rubric_scores"]["depth"] == 54.0

        # Feedback structure
        feedback = done["feedback"]
        assert feedback["strengths"] == ["Clear"]
        assert feedback["improvements"] == ["Depth"]
        assert feedback["model_answer"] == "Ideal model answer"


class TestSSEDuplicateAnswer:
    def test_duplicate_returns_409(self, client):
        """A second SSE request for the same question_index should get HTTP 409."""
        tc, manager = client
        session = _setup_session_with_questions(manager)

        async def mock_stepwise(candidate, ideal, question="", on_progress=None, category=""):
            if on_progress:
                await on_progress("scoring_started", {})
                await on_progress("scores_ready", {})
            return _make_scoring_result()

        mock_feedback = AsyncMock(return_value={
            "strengths": ["Good"], "improvements": ["More detail"],
            "model_answer": "...",
        })

        with patch("app.routers.evaluation.evaluate_stepwise", side_effect=mock_stepwise), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            # First answer succeeds
            resp1 = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer",
            })
            assert resp1.status_code == 200

            # Duplicate should get 409 (raised before streaming starts)
            resp2 = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer again",
            })
            assert resp2.status_code == 409
            assert "already answered" in resp2.json()["detail"]


class TestSSEBehavioralSTARScores:
    def test_star_scores_are_0_to_100(self, client):
        """STAR scores in the done event must be on a 0-100 scale (not 0-1)."""
        tc, manager = client
        session = manager.create_session("PM", "Senior", "Behavioral", "Medium")
        questions = [
            QuestionItem(
                index=0,
                question_text="Tell me about a time you led a team.",
                ideal_answer="I led a cross-functional team...",
                category="Behavioral",
                difficulty="Medium",
            )
        ]
        manager.set_questions(session.session_id, questions)

        behavioral_result = ScoringResult(
            sbert_raw=60.0, nli_raw=50.0, keyword_raw=40.0,
            llm_raw=78.0, raw_composite=71.0, composite=71.0, grade="Good",
            missing_keywords=[], llm_reason="Solid STAR response.",
            llm_correctness=0.0, llm_completeness=0.0,
            llm_clarity=0.0, llm_depth=0.0,
            is_behavioral=True,
            star_situation=85.0, star_task=70.0, star_action=90.0,
            star_result=75.0, star_reflection=60.0,
        )

        async def mock_stepwise(candidate, ideal, question="", on_progress=None, category=""):
            if on_progress:
                await on_progress("scoring_started", {})
                await on_progress("scores_ready", {})
            return behavioral_result

        mock_feedback = AsyncMock(return_value={
            "strengths": ["Clear narrative"], "improvements": ["Add metrics"],
            "model_answer": "...",
        })

        with patch("app.routers.evaluation.evaluate_stepwise", side_effect=mock_stepwise), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            resp = tc.post("/api/evaluate_answer_sse", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "When I was at Acme Corp...",
            })

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        done = [e for e in events if e.get("step") == "done"][0]

        assert done["scores"]["is_behavioral"] is True
        star = done["scores"]["star_scores"]
        assert star is not None

        # Values must be on the 0-100 scale, not 0-1
        assert star["situation"] == 85.0
        assert star["task"] == 70.0
        assert star["action"] == 90.0
        assert star["result"] == 75.0
        assert star["reflection"] == 60.0

        # Sanity: none should exceed 100
        for v in star.values():
            assert 0 <= v <= 100


class TestSSESessionNotFound:
    def test_session_not_found_returns_404(self, client):
        """SSE request with nonexistent session_id should get HTTP 404."""
        tc, _ = client
        resp = tc.post("/api/evaluate_answer_sse", json={
            "session_id": "nonexistent-uuid",
            "question_index": 0,
            "candidate_answer": "something",
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
