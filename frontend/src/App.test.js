import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import {
  checkSessionStatus,
  evaluateAnswerSSE,
  generateQuestions,
  getSessionSummary,
} from "./utils/api";

jest.mock("./utils/api", () => ({
  checkSessionStatus: jest.fn(),
  evaluateAnswer: jest.fn(),
  evaluateAnswerSSE: jest.fn(),
  generateQuestions: jest.fn(),
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
