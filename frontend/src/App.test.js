import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import {
  checkSessionStatus,
  evaluateAnswerSSE,
  generateQuestions,
  getAnswerResult,
  getSessionSummary,
} from "./utils/api";

jest.mock("./utils/api", () => ({
  checkSessionStatus: jest.fn(),
  evaluateAnswer: jest.fn(),
  evaluateAnswerSSE: jest.fn(),
  generateQuestions: jest.fn(),
  getAnswerResult: jest.fn(),
  getSessionSummary: jest.fn(),
}));

const SAVED_SESSION = {
  messages: [
    { role: "ai", type: "text", content: "Saved interview state" },
    {
      role: "ai",
      type: "question",
      content: "Tell me about a time you failed.",
      questionNum: 2,
      totalQuestions: 3,
      category: "Behavioral",
      difficulty: "Medium",
    },
  ],
  phase: "interview",
  sessionId: "sess-123",
  questions: [
    {
      index: 0,
      question_text: "Question 1",
      category: "Behavioral",
      difficulty: "Medium",
    },
    {
      index: 1,
      question_text: "Tell me about a time you failed.",
      category: "Behavioral",
      difficulty: "Medium",
    },
  ],
  currentQuestionIndex: 1,
};

describe("App restore flow", () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    generateQuestions.mockResolvedValue({});
    evaluateAnswerSSE.mockResolvedValue({});
    getAnswerResult.mockResolvedValue({});
  });

  test("resets to setup if restored complete session summary cannot be loaded", async () => {
    localStorage.setItem("interview_session", JSON.stringify(SAVED_SESSION));
    checkSessionStatus.mockResolvedValue({
      kind: "ok",
      data: {
        session_id: "sess-123",
        state: "complete",
        total_questions: 2,
        answers_count: 2,
      },
    });
    getSessionSummary.mockRejectedValue(new Error("summary unavailable"));

    render(<App />);

    expect(
      await screen.findByText(
        "Session was already completed, but the summary could not be loaded. Starting fresh."
      )
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(
        screen.getByText(/Welcome to PrepBuddy!/i)
      ).toBeInTheDocument();
    });

    expect(
      screen.getByPlaceholderText("Type your response...")
    ).toBeEnabled();
  });

  test("keeps restored interview state when session status check is unavailable", async () => {
    localStorage.setItem("interview_session", JSON.stringify(SAVED_SESSION));
    checkSessionStatus.mockResolvedValue({ kind: "unavailable" });

    render(<App />);

    expect(
      await screen.findByText("Saved interview state")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Tell me about a time you failed.")
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("Type your answer... (Shift+Enter for new line)")
    ).toBeEnabled();
  });
});

describe("App setup quick replies", () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    checkSessionStatus.mockResolvedValue({ kind: "unavailable" });
  });

  test("shows role quick reply chips on the first setup step", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: "Software Engineer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Data Scientist" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Other" })).toBeInTheDocument();
  });

  test("clicking Other switches setup input to custom role entry", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Other" }));

    expect(screen.getByPlaceholderText("Type a custom role...")).toBeInTheDocument();
  });

  test("shows level quick reply chips after selecting a role", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Software Engineer" }));

    expect(screen.getByRole("button", { name: "Intern" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Junior" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Lead" })).toBeInTheDocument();
  });
});

describe("App interrupted evaluation recovery", () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    checkSessionStatus.mockResolvedValue({ kind: "unavailable" });
  });

  test("restores a completed evaluation after the SSE request fails", async () => {
    generateQuestions.mockResolvedValue({
      session_id: "sess-123",
      questions: [
        {
          index: 0,
          question_text: "Explain CAP theorem.",
          category: "System Design",
          difficulty: "Medium",
        },
        {
          index: 1,
          question_text: "Explain consistent hashing.",
          category: "System Design",
          difficulty: "Medium",
        },
      ],
      total_questions: 2,
    });

    evaluateAnswerSSE.mockRejectedValue(new Error("network dropped"));
    checkSessionStatus
      .mockResolvedValueOnce({
        kind: "ok",
        data: {
          session_id: "sess-123",
          state: "interviewing",
          total_questions: 2,
          answers_count: 1,
          in_progress_indices: [],
        },
      });
    getAnswerResult.mockResolvedValue({
      scores: {
        sbert_score: 72,
        nli_score: 68,
        keyword_score: 61,
        llm_score: 74,
        llm_reason: "Solid answer, but it stayed a bit surface-level.",
        rubric_scores: {
          correctness: 82,
          completeness: 76,
          clarity: 79,
          depth: 64,
        },
        composite_score: 71,
        grade: "Good",
        missing_keywords: ["trade-off analysis"],
        is_behavioral: false,
        star_scores: null,
        claim_matches: [],
      },
      feedback: {
        strengths: ["Clear explanation"],
        improvements: ["Add more detail"],
        model_answer: "Example answer",
      },
      question_text: "Explain CAP theorem.",
      is_last_question: false,
      questions_remaining: 1,
    });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Software Engineer" }));
    fireEvent.click(screen.getByRole("button", { name: "Junior" }));
    fireEvent.click(screen.getByRole("button", { name: "System Design" }));
    fireEvent.click(screen.getByRole("button", { name: "Medium" }));
    fireEvent.click(screen.getByRole("button", { name: "2" }));

    expect(await screen.findByText("Explain CAP theorem.")).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText("Type your answer... (Shift+Enter for new line)");

    fireEvent.change(textarea, {
      target: { value: "My answer" },
    });
    fireEvent.keyDown(textarea, { key: "Enter", code: "Enter" });

    expect(
      await screen.findByText(
        "Connection interrupted, but your evaluation completed. Restored the result below."
      )
    ).toBeInTheDocument();
    expect(screen.getByText("trade-off analysis")).toBeInTheDocument();
    expect(screen.getByText("Explain consistent hashing.")).toBeInTheDocument();
  });
});
