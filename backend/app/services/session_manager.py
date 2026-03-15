import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SessionState(str, Enum):
    SETUP = "setup"
    INTERVIEWING = "interviewing"
    COMPLETE = "complete"


@dataclass
class QuestionItem:
    index: int
    question_text: str
    ideal_answer: str
    category: str
    difficulty: str


@dataclass
class AnswerResult:
    question_index: int
    candidate_answer: str
    sbert_score: float
    nli_score: float
    keyword_score: float
    composite_score: float
    grade: str
    missing_keywords: list[str]
    feedback: dict


@dataclass
class Session:
    session_id: str
    role: str
    level: str
    category: str
    difficulty: str
    questions: list[QuestionItem] = field(default_factory=list)
    answers: list[AnswerResult] = field(default_factory=list)
    state: SessionState = SessionState.SETUP
    created_at: datetime = field(default_factory=datetime.now)


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(
        self, role: str, level: str, category: str, difficulty: str
    ) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            role=role,
            level=level,
            category=category,
            difficulty=difficulty,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def set_questions(self, session_id: str, questions: list[QuestionItem]):
        session = self._sessions.get(session_id)
        if session:
            session.questions = questions
            session.state = SessionState.INTERVIEWING

    def add_answer(self, session_id: str, result: AnswerResult):
        session = self._sessions.get(session_id)
        if session:
            session.answers.append(result)
            if len(session.answers) >= len(session.questions):
                session.state = SessionState.COMPLETE

    def get_summary(self, session_id: str) -> dict | None:
        session = self._sessions.get(session_id)
        if not session or not session.answers:
            return None

        scores = [a.composite_score for a in session.answers]
        avg_score = round(sum(scores) / len(scores), 1)

        results = []
        for answer in session.answers:
            q = session.questions[answer.question_index]
            results.append({
                "index": answer.question_index,
                "question_text": q.question_text,
                "category": q.category,
                "composite_score": answer.composite_score,
                "sbert_score": answer.sbert_score,
                "nli_score": answer.nli_score,
                "keyword_score": answer.keyword_score,
                "grade": answer.grade,
            })

        best = max(results, key=lambda r: r["composite_score"])
        worst = min(results, key=lambda r: r["composite_score"])

        def _overall_grade(score: float) -> str:
            if score >= 80:
                return "Excellent"
            elif score >= 60:
                return "Good"
            elif score >= 40:
                return "Needs Improvement"
            return "Significant Gaps"

        return {
            "session_id": session.session_id,
            "role": session.role,
            "level": session.level,
            "category": session.category,
            "total_questions": len(session.questions),
            "questions_answered": len(session.answers),
            "average_score": avg_score,
            "results": results,
            "strongest_area": f"Q{best['index'] + 1}: {best['question_text'][:60]}",
            "weakest_area": f"Q{worst['index'] + 1}: {worst['question_text'][:60]}",
            "overall_grade": _overall_grade(avg_score),
        }


# Global singleton
session_manager = SessionManager()
