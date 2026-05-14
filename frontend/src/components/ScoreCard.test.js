import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
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

  test("renders short missing keywords as inline tags", () => {
    render(<ScoreCard data={technicalData} />);
    // 'trade-off analysis' is short — should not become a list item.
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
    expect(screen.getByText("trade-off analysis")).toBeInTheDocument();
  });

  test("renders long sentence-shaped missing concepts as a bullet list", () => {
    const longConcepts = [
      "The architecture would involve a load balancer and a model serving layer using TensorFlow Serving",
      "To handle high traffic, use containerization with Docker and orchestration with Kubernetes",
    ];
    const data = {
      ...technicalData,
      scores: { ...technicalData.scores, missing_keywords: longConcepts },
    };
    render(<ScoreCard data={data} />);

    const list = screen.getByRole("list");
    expect(list).toBeInTheDocument();
    expect(list.children).toHaveLength(2);
    expect(list).toHaveTextContent(/TensorFlow Serving/);
    expect(list).toHaveTextContent(/Kubernetes/);
  });

  test("shows partial coverage icon when claim is in soft margin", () => {
    const data = {
      ...technicalData,
      scores: {
        ...technicalData.scores,
        claim_matches: [
          {
            claim: "Should mention caching",
            covered: false,
            partial: true,
            match_score: 0.55,
            similarity: 0.71,
            contradiction: 0.0,
          },
          {
            claim: "Should mention auth",
            covered: true,
            partial: false,
            match_score: 0.82,
            similarity: 0.85,
            contradiction: 0.0,
          },
          {
            claim: "Should mention sharding",
            covered: false,
            partial: false,
            match_score: 0.30,
            similarity: 0.32,
            contradiction: 0.0,
          },
        ],
      },
    };
    render(<ScoreCard data={data} />);
    fireEvent.click(screen.getByText(/Show Feedback/));

    // covered uses ✓, partial uses ◑, fully missed uses ✗
    expect(screen.getByText("◑")).toBeInTheDocument();
    expect(screen.getByText("✓")).toBeInTheDocument();
    expect(screen.getByText("✗")).toBeInTheDocument();
    // Displayed % comes from match_score, not similarity.
    expect(screen.getByText("55%")).toBeInTheDocument(); // 0.55
    expect(screen.getByText("82%")).toBeInTheDocument(); // 0.82
    expect(screen.queryByText("71%")).not.toBeInTheDocument(); // similarity
  });

  test("falls back to similarity when match_score is absent (older records)", () => {
    const data = {
      ...technicalData,
      scores: {
        ...technicalData.scores,
        claim_matches: [
          {
            claim: "Older record without match_score",
            covered: false,
            similarity: 0.66,
            contradiction: 0.0,
          },
        ],
      },
    };
    render(<ScoreCard data={data} />);
    fireEvent.click(screen.getByText(/Show Feedback/));
    expect(screen.getByText("66%")).toBeInTheDocument();
  });

  test("renders missed optional claims as optional instead of hard misses", () => {
    const data = {
      ...technicalData,
      scores: {
        ...technicalData.scores,
        claim_matches: [
          {
            claim: "Examples include MongoDB or Cassandra",
            covered: false,
            partial: false,
            match_score: 0.18,
            similarity: 0.22,
            contradiction: 0.0,
            importance: "optional",
          },
        ],
      },
    };
    render(<ScoreCard data={data} />);
    fireEvent.click(screen.getByText(/Show Feedback/));

    expect(screen.getByText("○")).toBeInTheDocument();
    expect(screen.getByText("optional")).toBeInTheDocument();
    expect(screen.queryByText("✗")).not.toBeInTheDocument();
  });
});
