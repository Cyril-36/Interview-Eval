import React from "react";
import { render, screen } from "@testing-library/react";
import ScoreCard from "./ScoreCard";

const technicalData = {
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
  },
  feedback: {
    strengths: ["Clear explanation"],
    improvements: ["Add more detail"],
    model_answer: "Example answer",
  },
};

describe("ScoreCard", () => {
  test("shows a quick tip when a rubric dimension is weak", () => {
    render(<ScoreCard data={technicalData} />);

    expect(screen.getByText("Quick tip")).toBeInTheDocument();
    expect(
      screen.getByText("Add more reasoning, trade-offs, or detail to show deeper understanding.")
    ).toBeInTheDocument();
  });
});
