from pydantic import BaseModel, Field
from enum import Enum


class GenerateQuestionsRequest(BaseModel):
    role: str = Field(..., example="Software Engineer")
    level: str = Field(..., example="Junior")
    category: str = Field(..., example="Data Structures")
    difficulty: str = Field(..., example="Medium")
    num_questions: int = Field(default=5, ge=1, le=20)


class QuestionOut(BaseModel):
    index: int
    question_text: str
    category: str
    difficulty: str


class GenerateQuestionsResponse(BaseModel):
    session_id: str
    questions: list[QuestionOut]
    total_questions: int


class EvaluateAnswerRequest(BaseModel):
    session_id: str
    question_index: int
    candidate_answer: str = Field(..., max_length=5000)


class ScoreBreakdown(BaseModel):
    sbert_score: float
    nli_score: float
    keyword_score: float
    composite_score: float
    grade: str
    missing_keywords: list[str]


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


class QuestionResult(BaseModel):
    index: int
    question_text: str
    category: str
    composite_score: float
    sbert_score: float
    nli_score: float
    keyword_score: float
    grade: str


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
