from typing import Literal
from pydantic import BaseModel, Field


class GenerateQuestionsRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=100, examples=["Software Engineer"])
    level: Literal["Intern", "Junior", "Mid-Level", "Senior", "Lead"] = Field(
        ..., examples=["Junior"],
    )
    category: str = Field(..., min_length=1, max_length=100, examples=["Data Structures"])
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., examples=["Medium"],
    )
    num_questions: int = Field(default=5, ge=1, le=10)
    resume_context: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional resume summary to tailor questions.",
    )
    extracted_skills: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Optional list of skill keyphrases from a parsed resume.",
    )


class ResumeParseResponse(BaseModel):
    extracted_text: str
    skills: list[str]
    summary: str
    page_count: int
    truncated: bool


class QuestionOut(BaseModel):
    index: int
    question_text: str
    category: str
    difficulty: str


class GenerateQuestionsResponse(BaseModel):
    session_id: str
    session_token: str
    questions: list[QuestionOut]
    total_questions: int


class EvaluateAnswerRequest(BaseModel):
    session_id: str
    question_index: int = Field(..., ge=0)
    candidate_answer: str = Field(..., min_length=1, max_length=5000)


class RubricScores(BaseModel):
    correctness: float
    completeness: float
    clarity: float
    depth: float


class STARScores(BaseModel):
    situation: float = 0.0
    task: float = 0.0
    action: float = 0.0
    result: float = 0.0
    reflection: float = 0.0


class ClaimFeedback(BaseModel):
    claim: str
    covered: bool
    similarity: float
    contradiction: float
    # New: combined match score (similarity + entailment − contradiction penalty)
    # used by the coverage decision, and a partial flag when the score lands
    # within CLAIM_SOFT_MARGIN of the threshold. Defaulted so older persisted
    # claim_matches (without these keys) still validate.
    match_score: float = 0.0
    partial: bool = False
    importance: Literal["core", "optional"] = "core"


class ScoreBreakdown(BaseModel):
    sbert_score: float
    nli_score: float
    keyword_score: float
    llm_score: float
    llm_reason: str
    rubric_scores: RubricScores
    composite_score: float
    grade: str
    missing_keywords: list[str]
    is_behavioral: bool = False
    star_scores: STARScores | None = None
    claim_matches: list[ClaimFeedback] = Field(default_factory=list)


class FeedbackOut(BaseModel):
    strengths: list[str]
    improvements: list[str]
    model_answer: str


class EvaluateAnswerResponse(BaseModel):
    scores: ScoreBreakdown
    feedback: FeedbackOut
    question_text: str
    is_last_question: bool
    questions_remaining: int


class SessionStatusResponse(BaseModel):
    session_id: str
    state: Literal["setup", "interviewing", "complete"]
    total_questions: int
    answers_count: int
    in_progress_indices: list[int] = Field(default_factory=list)


class QuestionResult(BaseModel):
    index: int
    question_text: str
    category: str
    composite_score: float
    sbert_score: float
    nli_score: float
    keyword_score: float
    llm_score: float
    grade: str
    llm_correctness: float = 0.0
    llm_completeness: float = 0.0
    llm_clarity: float = 0.0
    llm_depth: float = 0.0
    is_behavioral: bool = False
    star_scores: STARScores | None = None


class SessionSummaryResponse(BaseModel):
    session_id: str
    role: str
    level: str
    category: str
    total_questions: int
    questions_answered: int
    average_score: float
    results: list[QuestionResult]
    strongest_area: str
    weakest_area: str
    overall_grade: str
    avg_rubric: RubricScores | None = None
    avg_star: STARScores | None = None
