"""Tests for API-level validation: duplicate answers, orphan cleanup, input validation."""
import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from app.services.session_manager import (
    SessionManager,
    QuestionItem,
    AnswerResult,
)


@pytest.fixture()
def client(tmp_path):
    """Create a test client with a temp DB."""
    with patch("app.services.session_manager.DATA_DIR", tmp_path), \
         patch("app.services.session_manager.DB_PATH", tmp_path / "sessions.db"):
        # Re-create the singleton for this test
        import app.services.session_manager as sm_mod
        original = sm_mod.session_manager
        sm_mod.session_manager = SessionManager()

        # Patch the module-level import in routers
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


class TestDuplicateAnswerRejection:
    def test_duplicate_returns_409(self, client):
        tc, manager = client
        session = _setup_session_with_questions(manager)

        # Mock the scoring pipeline to avoid loading models
        mock_result = AsyncMock()
        mock_result.return_value = type("SR", (), {
            "sbert_raw": 75.0, "nli_raw": 70.0, "keyword_raw": 80.0,
            "llm_raw": 65.0, "composite": 72.5, "grade": "Good",
            "missing_keywords": [], "llm_reason": "Balanced answer.",
            "llm_correctness": 72.0, "llm_completeness": 68.0,
            "llm_clarity": 66.0, "llm_depth": 54.0,
            "is_behavioral": False,
            "star_situation": 0.0, "star_task": 0.0, "star_action": 0.0,
            "star_result": 0.0, "star_reflection": 0.0,
            "claim_matches": [],
        })()

        mock_feedback = AsyncMock(return_value={
            "strengths": ["Good"], "improvements": ["More detail"],
            "model_answer": "...",
        })

        with patch("app.routers.evaluation.evaluate", mock_result), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            # First answer should succeed
            resp1 = tc.post("/api/evaluate_answer", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer",
            })
            assert resp1.status_code == 200
            score_payload = resp1.json()["scores"]
            assert score_payload["llm_reason"] == "Balanced answer."
            assert score_payload["rubric_scores"]["correctness"] == 72.0

            # Duplicate should get 409
            resp2 = tc.post("/api/evaluate_answer", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "My answer again",
            })
            assert resp2.status_code == 409
            assert "already answered" in resp2.json()["detail"]


class TestQuestionsRemainingAccuracy:
    def test_remaining_count_on_first_and_last(self, client):
        """questions_remaining and is_last_question must reflect post-insert state."""
        tc, manager = client
        session = _setup_session_with_questions(manager, n_questions=2)

        mock_result = AsyncMock()
        mock_result.return_value = type("SR", (), {
            "sbert_raw": 75.0, "nli_raw": 70.0, "keyword_raw": 80.0,
            "llm_raw": 65.0, "composite": 72.5, "grade": "Good",
            "missing_keywords": [], "llm_reason": "Balanced answer.",
            "llm_correctness": 72.0, "llm_completeness": 68.0,
            "llm_clarity": 66.0, "llm_depth": 54.0,
            "is_behavioral": False,
            "star_situation": 0.0, "star_task": 0.0, "star_action": 0.0,
            "star_result": 0.0, "star_reflection": 0.0,
            "claim_matches": [],
        })()
        mock_feedback = AsyncMock(return_value={
            "strengths": ["Good"], "improvements": ["More detail"],
            "model_answer": "...",
        })

        with patch("app.routers.evaluation.evaluate", mock_result), \
             patch("app.routers.evaluation.generate_feedback", mock_feedback):
            # Answer first question
            resp1 = tc.post("/api/evaluate_answer", json={
                "session_id": session.session_id,
                "question_index": 0,
                "candidate_answer": "answer 0",
            })
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["questions_remaining"] == 1
            assert data1["is_last_question"] is False

            # Answer second (last) question
            resp2 = tc.post("/api/evaluate_answer", json={
                "session_id": session.session_id,
                "question_index": 1,
                "candidate_answer": "answer 1",
            })
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["questions_remaining"] == 0
            assert data2["is_last_question"] is True


class TestOrphanSessionCleanup:
    def test_failed_generation_deletes_session(self, client):
        tc, manager = client

        with patch(
            "app.routers.questions.generate_questions",
            AsyncMock(side_effect=RuntimeError("LLM unavailable")),
        ):
            resp = tc.post("/api/generate_questions", json={
                "role": "SWE",
                "level": "Junior",
                "category": "Python",
                "difficulty": "Medium",
                "num_questions": 3,
            })
            assert resp.status_code == 500

        # The session should have been cleaned up — no orphans in DB
        # (We can't easily get the session_id, but we verify
        #  by checking that no sessions exist at all)
        row = manager._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()
        assert row[0] == 0


class TestInputValidation:
    def test_invalid_level_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/generate_questions", json={
            "role": "SWE",
            "level": "Expert",  # not in Literal enum
            "category": "Python",
            "difficulty": "Medium",
        })
        assert resp.status_code == 422

    def test_invalid_difficulty_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/generate_questions", json={
            "role": "SWE",
            "level": "Junior",
            "category": "Python",
            "difficulty": "Impossible",  # not in Literal enum
        })
        assert resp.status_code == 422

    def test_empty_role_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/generate_questions", json={
            "role": "",
            "level": "Junior",
            "category": "Python",
            "difficulty": "Medium",
        })
        assert resp.status_code == 422

    def test_negative_question_index_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/evaluate_answer", json={
            "session_id": "fake",
            "question_index": -1,
            "candidate_answer": "answer",
        })
        assert resp.status_code == 422

    def test_empty_answer_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/evaluate_answer", json={
            "session_id": "fake",
            "question_index": 0,
            "candidate_answer": "",
        })
        assert resp.status_code == 422

    def test_session_not_found(self, client):
        tc, _ = client
        resp = tc.post("/api/evaluate_answer", json={
            "session_id": "nonexistent-uuid",
            "question_index": 0,
            "candidate_answer": "something",
        })
        assert resp.status_code == 404

    def test_summary_not_found(self, client):
        tc, _ = client
        resp = tc.get("/api/session_summary?session_id=nonexistent")
        assert resp.status_code == 404


class TestSessionRecoveryEndpoints:
    def test_session_status_reports_correctly(self, client):
        tc, manager = client
        session = _setup_session_with_questions(manager, n_questions=2)

        resp = tc.get(f"/api/session_status?session_id={session.session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session.session_id

    def test_answer_result_returns_persisted_scorecard_fields(self, client):
        tc, manager = client
        session = _setup_session_with_questions(manager, n_questions=1)

        manager.add_answer(session.session_id, AnswerResult(
            question_index=0,
            candidate_answer="My answer",
            sbert_score=75.0,
            nli_score=70.0,
            keyword_score=80.0,
            llm_score=65.0,
            llm_reason="Balanced answer.",
            composite_score=72.5,
            grade="Good",
            missing_keywords=["trade-off analysis"],
            feedback={
                "strengths": ["Clear structure"],
                "improvements": ["More depth"],
                "model_answer": "Ideal answer",
            },
            claim_matches=[{
                "claim": "Mentions trade-offs",
                "covered": True,
                "similarity": 0.84,
                "contradiction": 0.0,
            }],
            llm_correctness=72.0,
            llm_completeness=68.0,
            llm_clarity=66.0,
            llm_depth=54.0,
        ))

        resp = tc.get(
            f"/api/answer_result?session_id={session.session_id}&question_index=0"
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["scores"]["llm_reason"] == "Balanced answer."
        assert payload["scores"]["claim_matches"] == [{
            "claim": "Mentions trade-offs",
            "covered": True,
            "similarity": 0.84,
            "contradiction": 0.0,
        }]
        assert payload["feedback"]["model_answer"] == "Ideal answer"
