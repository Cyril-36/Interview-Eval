import uuid
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "sessions.db"


class SessionState(str, Enum):
    SETUP = "setup"
    INTERVIEWING = "interviewing"
    COMPLETE = "complete"


class SessionStoreError(RuntimeError):
    """Raised when SQLite persistence fails for reasons other than duplicates."""


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
    llm_score: float
    composite_score: float
    grade: str
    missing_keywords: list[str]
    feedback: dict
    llm_reason: str = ""
    claim_matches: list[dict] = field(default_factory=list)
    llm_correctness: float = 0.0
    llm_completeness: float = 0.0
    llm_clarity: float = 0.0
    llm_depth: float = 0.0
    is_behavioral: bool = False
    star_situation: float = 0.0
    star_task: float = 0.0
    star_action: float = 0.0
    star_result: float = 0.0
    star_reflection: float = 0.0


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
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SessionManager:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(DB_PATH), check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._migrate()

    def _migrate(self):
        """Add missing answer columns to existing answers tables."""
        cursor = self._conn.execute("PRAGMA table_info(answers)")
        columns = {row[1] for row in cursor.fetchall()}
        migrate_cols = [
            ("llm_reason", "TEXT NOT NULL DEFAULT ''"),
            ("claim_matches", "TEXT NOT NULL DEFAULT '[]'"),
            ("llm_correctness", "REAL NOT NULL DEFAULT 0"),
            ("llm_completeness", "REAL NOT NULL DEFAULT 0"),
            ("llm_clarity", "REAL NOT NULL DEFAULT 0"),
            ("llm_depth", "REAL NOT NULL DEFAULT 0"),
            ("is_behavioral", "INTEGER NOT NULL DEFAULT 0"),
            ("star_situation", "REAL NOT NULL DEFAULT 0"),
            ("star_task", "REAL NOT NULL DEFAULT 0"),
            ("star_action", "REAL NOT NULL DEFAULT 0"),
            ("star_result", "REAL NOT NULL DEFAULT 0"),
            ("star_reflection", "REAL NOT NULL DEFAULT 0"),
        ]
        for col, col_type in migrate_cols:
            if col not in columns:
                self._conn.execute(
                    f"ALTER TABLE answers ADD COLUMN {col} {col_type}"
                )
        self._conn.commit()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'setup',
                created_at TEXT NOT NULL,
                questions TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question_index INTEGER NOT NULL,
                candidate_answer TEXT NOT NULL,
                sbert_score REAL NOT NULL,
                nli_score REAL NOT NULL,
                keyword_score REAL NOT NULL,
                llm_score REAL NOT NULL,
                llm_reason TEXT NOT NULL DEFAULT '',
                composite_score REAL NOT NULL,
                grade TEXT NOT NULL,
                missing_keywords TEXT NOT NULL DEFAULT '[]',
                feedback TEXT NOT NULL DEFAULT '{}',
                claim_matches TEXT NOT NULL DEFAULT '[]',
                llm_correctness REAL NOT NULL DEFAULT 0,
                llm_completeness REAL NOT NULL DEFAULT 0,
                llm_clarity REAL NOT NULL DEFAULT 0,
                llm_depth REAL NOT NULL DEFAULT 0,
                is_behavioral INTEGER NOT NULL DEFAULT 0,
                star_situation REAL NOT NULL DEFAULT 0,
                star_task REAL NOT NULL DEFAULT 0,
                star_action REAL NOT NULL DEFAULT 0,
                star_result REAL NOT NULL DEFAULT 0,
                star_reflection REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                UNIQUE(session_id, question_index)
            );
        """)
        self._conn.commit()

    def _load_session(self, session_id: str) -> Session | None:
        """Load a session and its answers from the database."""
        row = self._conn.execute(
            "SELECT session_id, role, level, category, difficulty, state, "
            "created_at, questions FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None

        questions = [QuestionItem(**q) for q in json.loads(row[7])]

        answer_rows = self._conn.execute(
            "SELECT question_index, candidate_answer, sbert_score, nli_score, "
            "keyword_score, llm_score, llm_reason, composite_score, grade, "
            "missing_keywords, feedback, claim_matches, "
            "llm_correctness, llm_completeness, llm_clarity, llm_depth, "
            "is_behavioral, star_situation, star_task, star_action, "
            "star_result, star_reflection "
            "FROM answers WHERE session_id = ? ORDER BY question_index",
            (session_id,),
        ).fetchall()

        answers = [
            AnswerResult(
                question_index=a[0],
                candidate_answer=a[1],
                sbert_score=a[2],
                nli_score=a[3],
                keyword_score=a[4],
                llm_score=a[5],
                llm_reason=a[6],
                composite_score=a[7],
                grade=a[8],
                missing_keywords=json.loads(a[9]),
                feedback=json.loads(a[10]),
                claim_matches=json.loads(a[11]),
                llm_correctness=a[12],
                llm_completeness=a[13],
                llm_clarity=a[14],
                llm_depth=a[15],
                is_behavioral=bool(a[16]),
                star_situation=a[17],
                star_task=a[18],
                star_action=a[19],
                star_result=a[20],
                star_reflection=a[21],
            )
            for a in answer_rows
        ]

        return Session(
            session_id=row[0],
            role=row[1],
            level=row[2],
            category=row[3],
            difficulty=row[4],
            questions=questions,
            answers=answers,
            state=SessionState(row[5]),
            created_at=row[6],
        )

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
        self._conn.execute(
            "INSERT INTO sessions "
            "(session_id, role, level, category, difficulty, state, created_at, questions) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.session_id,
                session.role,
                session.level,
                session.category,
                session.difficulty,
                session.state.value,
                session.created_at,
                "[]",
            ),
        )
        self._conn.commit()
        return session

    def delete_session(self, session_id: str):
        """Remove a session from the database."""
        self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        return self._load_session(session_id)

    def set_questions(self, session_id: str, questions: list[QuestionItem]):
        questions_json = json.dumps(
            [
                {
                    "index": q.index,
                    "question_text": q.question_text,
                    "ideal_answer": q.ideal_answer,
                    "category": q.category,
                    "difficulty": q.difficulty,
                }
                for q in questions
            ]
        )
        self._conn.execute(
            "UPDATE sessions SET questions = ?, state = ? WHERE session_id = ?",
            (questions_json, SessionState.INTERVIEWING.value, session_id),
        )
        self._conn.commit()



    def add_answer(self, session_id: str, result: AnswerResult) -> bool:
        """Add an answer to a session. Returns False if duplicate."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT questions FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return False

                self._conn.execute(
                    "INSERT INTO answers "
                    "(session_id, question_index, candidate_answer, sbert_score, "
                    "nli_score, keyword_score, llm_score, llm_reason, "
                    "composite_score, grade, missing_keywords, feedback, "
                    "claim_matches, "
                    "llm_correctness, llm_completeness, llm_clarity, llm_depth, "
                    "is_behavioral, star_situation, star_task, star_action, "
                    "star_result, star_reflection) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        result.question_index,
                        result.candidate_answer,
                        result.sbert_score,
                        result.nli_score,
                        result.keyword_score,
                        result.llm_score,
                        result.llm_reason,
                        result.composite_score,
                        result.grade,
                        json.dumps(result.missing_keywords),
                        json.dumps(result.feedback),
                        json.dumps(result.claim_matches),
                        result.llm_correctness,
                        result.llm_completeness,
                        result.llm_clarity,
                        result.llm_depth,
                        int(result.is_behavioral),
                        result.star_situation,
                        result.star_task,
                        result.star_action,
                        result.star_result,
                        result.star_reflection,
                    ),
                )
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return False
            except sqlite3.DatabaseError as exc:
                self._conn.rollback()
                raise SessionStoreError(
                    f"Failed to store answer: {exc}"
                ) from exc

            # Check if all questions are answered; if so, mark complete
            questions = json.loads(row[0])
            answer_count = self._conn.execute(
                "SELECT COUNT(*) FROM answers WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]

            if answer_count >= len(questions):
                self._conn.execute(
                    "UPDATE sessions SET state = ? WHERE session_id = ?",
                    (SessionState.COMPLETE.value, session_id),
                )

            self._conn.commit()
            return True

    def get_answer(self, session_id: str, question_index: int) -> AnswerResult | None:
        session = self._load_session(session_id)
        if session is None:
            return None
        for answer in session.answers:
            if answer.question_index == question_index:
                return answer
        return None

    def get_summary(self, session_id: str) -> dict | None:  # type: ignore[type-arg]
        # Fetch session metadata
        sess_row = self._conn.execute(
            "SELECT session_id, role, level, category, questions "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if sess_row is None:
            return None

        questions = json.loads(sess_row[4])

        # Fetch answers
        answer_rows = self._conn.execute(
            "SELECT question_index, candidate_answer, sbert_score, nli_score, "
            "keyword_score, llm_score, composite_score, grade, "
            "llm_correctness, llm_completeness, llm_clarity, llm_depth, "
            "is_behavioral, star_situation, star_task, star_action, "
            "star_result, star_reflection "
            "FROM answers WHERE session_id = ? ORDER BY question_index",
            (session_id,),
        ).fetchall()

        if not answer_rows:
            return None

        # Build question lookup
        q_by_index = {q["index"]: q for q in questions}

        scores = [a[6] for a in answer_rows]
        avg_score = round(sum(scores) / len(scores), 1)

        results: list[dict[str, object]] = []
        for a in answer_rows:
            q = q_by_index.get(a[0], {})
            entry: dict[str, object] = {
                "index": a[0],
                "question_text": q.get("question_text", ""),
                "category": q.get("category", ""),
                "composite_score": a[6],
                "sbert_score": a[2],
                "nli_score": a[3],
                "keyword_score": a[4],
                "llm_score": a[5],
                "grade": a[7],
                "llm_correctness": a[8],
                "llm_completeness": a[9],
                "llm_clarity": a[10],
                "llm_depth": a[11],
                "is_behavioral": bool(a[12]),
            }
            if bool(a[12]):
                entry["star_scores"] = {
                    "situation": a[13],
                    "task": a[14],
                    "action": a[15],
                    "result": a[16],
                    "reflection": a[17],
                }
            results.append(entry)

        best = max(results, key=lambda r: float(str(r["composite_score"])))
        worst = min(results, key=lambda r: float(str(r["composite_score"])))

        # Compute average rubric scores across technical (non-behavioral) answers
        technical_rows = [a for a in answer_rows if not bool(a[12])]
        if technical_rows:
            tn = len(technical_rows)
            avg_rubric = {
                "correctness": round(sum(a[8] for a in technical_rows) / tn, 1),
                "completeness": round(sum(a[9] for a in technical_rows) / tn, 1),
                "clarity": round(sum(a[10] for a in technical_rows) / tn, 1),
                "depth": round(sum(a[11] for a in technical_rows) / tn, 1),
            }
        else:
            # All-behavioral session: rubric averages are meaningless
            avg_rubric = None

        def _overall_grade(score: float) -> str:
            if score >= 80:
                return "Excellent"
            elif score >= 60:
                return "Good"
            elif score >= 40:
                return "Needs Improvement"
            return "Significant Gaps"

        # Compute average STAR scores for behavioral answers
        behavioral_rows = [a for a in answer_rows if bool(a[12])]
        avg_star = None
        if behavioral_rows:
            bn = len(behavioral_rows)
            avg_star = {
                "situation": round(sum(a[13] for a in behavioral_rows) / bn, 3),
                "task": round(sum(a[14] for a in behavioral_rows) / bn, 3),
                "action": round(sum(a[15] for a in behavioral_rows) / bn, 3),
                "result": round(sum(a[16] for a in behavioral_rows) / bn, 3),
                "reflection": round(sum(a[17] for a in behavioral_rows) / bn, 3),
            }

        return {
            "session_id": sess_row[0],
            "role": sess_row[1],
            "level": sess_row[2],
            "category": sess_row[3],
            "total_questions": len(questions),
            "questions_answered": len(answer_rows),
            "average_score": avg_score,
            "results": results,
            "strongest_area": f"Q{int(str(best['index'])) + 1}: {str(best['question_text'])[:60]}",
            "weakest_area": f"Q{int(str(worst['index'])) + 1}: {str(worst['question_text'])[:60]}",
            "overall_grade": _overall_grade(avg_score),
            "avg_rubric": avg_rubric,
            "avg_star": avg_star,
        }


# Global singleton
session_manager = SessionManager()
