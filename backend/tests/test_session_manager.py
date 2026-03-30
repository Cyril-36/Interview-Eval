"""Tests for SessionManager (SQLite-backed session store)."""
import os
import tempfile
import threading
from unittest.mock import patch

import pytest

from app.services.session_manager import (
    SessionManager,
    SessionState,
    QuestionItem,
    AnswerResult,
)


@pytest.fixture()
def manager(tmp_path):
    """Create a SessionManager backed by a temp directory."""
    with patch("app.services.session_manager.DATA_DIR", tmp_path), \
         patch("app.services.session_manager.DB_PATH", tmp_path / "sessions.db"):
        yield SessionManager()


def _make_questions(n=3):
    return [
        QuestionItem(
            index=i,
            question_text=f"Q{i}?",
            ideal_answer=f"A{i}",
            category="General",
            difficulty="Medium",
        )
        for i in range(n)
    ]


def _make_answer(question_index=0, score=75.0):
    return AnswerResult(
        question_index=question_index,
        candidate_answer="my answer",
        sbert_score=score,
        nli_score=score,
        keyword_score=score,
        llm_score=score,
        llm_reason="Balanced answer.",
        composite_score=score,
        grade="Good",
        missing_keywords=["kw1"],
        feedback={"strengths": ["ok"]},
        claim_matches=[{
            "claim": "Mentions trade-offs",
            "covered": True,
            "similarity": 0.82,
            "contradiction": 0.0,
        }],
    )


class TestCreateAndRetrieve:
    def test_create_returns_session(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        assert session.session_id
        assert session.role == "SWE"
        assert session.state == SessionState.SETUP

    def test_get_session_returns_created(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        fetched = manager.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id
        assert fetched.role == "SWE"

    def test_get_session_not_found(self, manager):
        assert manager.get_session("nonexistent") is None


class TestSetQuestions:
    def test_set_questions_updates_state(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        questions = _make_questions(3)
        manager.set_questions(session.session_id, questions)

        fetched = manager.get_session(session.session_id)
        assert fetched.state == SessionState.INTERVIEWING
        assert len(fetched.questions) == 3
        assert fetched.questions[0].question_text == "Q0?"


class TestAddAnswer:
    def test_add_answer_succeeds(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(3))

        result = manager.add_answer(session.session_id, _make_answer(0))
        assert result is True

        fetched = manager.get_session(session.session_id)
        assert len(fetched.answers) == 1

    def test_duplicate_answer_rejected(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(3))

        assert manager.add_answer(session.session_id, _make_answer(0)) is True
        assert manager.add_answer(session.session_id, _make_answer(0)) is False

        fetched = manager.get_session(session.session_id)
        assert len(fetched.answers) == 1

    def test_session_completes_when_all_answered(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(2))

        manager.add_answer(session.session_id, _make_answer(0))
        fetched = manager.get_session(session.session_id)
        assert fetched.state == SessionState.INTERVIEWING

        manager.add_answer(session.session_id, _make_answer(1))
        fetched = manager.get_session(session.session_id)
        assert fetched.state == SessionState.COMPLETE

    def test_add_answer_nonexistent_session(self, manager):
        assert manager.add_answer("bad-id", _make_answer(0)) is False

    def test_get_answer_round_trips_reason_and_claim_matches(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(1))

        manager.add_answer(session.session_id, _make_answer(0))

        answer = manager.get_answer(session.session_id, 0)
        assert answer is not None
        assert answer.llm_reason == "Balanced answer."
        assert answer.claim_matches == [{
            "claim": "Mentions trade-offs",
            "covered": True,
            "similarity": 0.82,
            "contradiction": 0.0,
        }]


class TestDeleteSession:
    def test_delete_removes_session(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.delete_session(session.session_id)
        assert manager.get_session(session.session_id) is None

    def test_delete_cascades_answers(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(1))
        manager.add_answer(session.session_id, _make_answer(0))
        manager.delete_session(session.session_id)
        assert manager.get_session(session.session_id) is None

    def test_delete_nonexistent_does_not_error(self, manager):
        manager.delete_session("nonexistent")  # should not raise


class TestGetSummary:
    def test_summary_no_answers(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        assert manager.get_summary(session.session_id) is None

    def test_summary_nonexistent(self, manager):
        assert manager.get_summary("nonexistent") is None

    def test_summary_with_answers(self, manager):
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(2))
        manager.add_answer(session.session_id, _make_answer(0, 80.0))
        manager.add_answer(session.session_id, _make_answer(1, 60.0))

        summary = manager.get_summary(session.session_id)
        assert summary is not None
        assert summary["session_id"] == session.session_id
        assert summary["total_questions"] == 2
        assert summary["questions_answered"] == 2
        assert summary["average_score"] == 70.0
        assert summary["overall_grade"] == "Good"
        assert len(summary["results"]) == 2


class TestConcurrency:
    def test_concurrent_answers_no_duplicates(self, manager):
        """Multiple threads adding the same answer — only one should succeed."""
        session = manager.create_session("SWE", "Junior", "Python", "Easy")
        manager.set_questions(session.session_id, _make_questions(1))

        results = []

        def add():
            ok = manager.add_answer(session.session_id, _make_answer(0))
            results.append(ok)

        threads = [threading.Thread(target=add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 4

        fetched = manager.get_session(session.session_id)
        assert len(fetched.answers) == 1


class TestCrossWorkerAccess:
    def test_second_manager_sees_first_managers_data(self, tmp_path):
        """Two SessionManager instances sharing the same DB see each other's data."""
        with patch("app.services.session_manager.DATA_DIR", tmp_path), \
             patch("app.services.session_manager.DB_PATH", tmp_path / "sessions.db"):
            mgr_a = SessionManager()
            mgr_b = SessionManager()

            session = mgr_a.create_session("SWE", "Junior", "Python", "Easy")
            mgr_a.set_questions(session.session_id, _make_questions(3))
            mgr_a.add_answer(session.session_id, _make_answer(0))

            # Worker B should see the session and answer
            fetched = mgr_b.get_session(session.session_id)
            assert fetched is not None
            assert len(fetched.answers) == 1

            # Worker B adds Q1, Worker A adds Q2
            mgr_b.add_answer(session.session_id, _make_answer(1))
            mgr_a.add_answer(session.session_id, _make_answer(2))

            # Both workers should see all 3 answers
            from_a = mgr_a.get_session(session.session_id)
            from_b = mgr_b.get_session(session.session_id)
            assert len(from_a.answers) == 3
            assert len(from_b.answers) == 3

    def test_summary_across_workers(self, tmp_path):
        """get_summary works from a different worker."""
        with patch("app.services.session_manager.DATA_DIR", tmp_path), \
             patch("app.services.session_manager.DB_PATH", tmp_path / "sessions.db"):
            mgr_a = SessionManager()
            mgr_b = SessionManager()

            session = mgr_a.create_session("SWE", "Junior", "Python", "Easy")
            mgr_a.set_questions(session.session_id, _make_questions(2))
            mgr_a.add_answer(session.session_id, _make_answer(0, 80.0))
            mgr_a.add_answer(session.session_id, _make_answer(1, 60.0))

            summary = mgr_b.get_summary(session.session_id)
            assert summary is not None
            assert summary["questions_answered"] == 2
            assert summary["average_score"] == 70.0
