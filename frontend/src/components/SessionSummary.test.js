import React from "react";
import { render, screen } from "@testing-library/react";
import SessionSummary from "./SessionSummary";

const summaryData = {
  session_id: "sess-1",
  role: "Software Engineer",
  level: "Junior",
  category: "Mixed",
  total_questions: 2,
  questions_answered: 2,
  average_score: 74.5,
  strongest_area: "Q2",
  weakest_area: "Q1",
  overall_grade: "Good",
  avg_rubric: {
    correctness: 81,
    completeness: 77,
    clarity: 75,
    depth: 70,
  },
  avg_star: {
    situation: 85,
    task: 70,
    action: 90,
    result: 75,
    reflection: 60,
  },
  results: [
    {
      index: 0,
      question_text: "Explain how indexing works in a database.",
      category: "Databases",
      composite_score: 78,
      sbert_score: 80,
      nli_score: 76,
      keyword_score: 74,
      llm_score: 81,
      grade: "Good",
      is_behavioral: false,
    },
    {
      index: 1,
      question_text: "Tell me about a time you handled a conflict.",
      category: "Behavioral",
      composite_score: 71,
      sbert_score: 60,
      nli_score: 55,
      keyword_score: 48,
      llm_score: 76,
      grade: "Good",
      is_behavioral: true,
      star_scores: {
        situation: 85,
        task: 70,
        action: 90,
        result: 75,
        reflection: 60,
      },
    },
  ],
};

describe("SessionSummary", () => {
  test("renders both technical and behavioral summary sections for mixed sessions", () => {
    render(<SessionSummary data={summaryData} onRestart={jest.fn()} />);

    expect(screen.getByText("Technical Rubric Averages")).toBeInTheDocument();
    expect(screen.getByText("Behavioral STAR Averages")).toBeInTheDocument();
    expect(screen.getByText("Technical")).toBeInTheDocument();
    expect(screen.getByText("Behavioral")).toBeInTheDocument();
    expect(screen.getByText(/S 85 T 70 A 90 R 75 Ref 60/)).toBeInTheDocument();
  });
});
