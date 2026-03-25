import React from "react";
import CubeIcon from "./CubeIcon";
import { getColor } from "../utils/colors";

function RubricBar({ label, score }) {
  return (
    <div className="score-bar-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${Math.min(100, score)}%`, background: getColor(score) }}
        />
      </div>
      <span className="score-value">{score.toFixed(0)}</span>
    </div>
  );
}

export default function SessionSummary({ data, onRestart }) {
  const rubric = data.avg_rubric;
  const star = data.avg_star;

  return (
    <div className="message-row ai">
      <div className="ai-avatar"><CubeIcon size={15} /></div>
      <div className="summary-card">
        <div className="summary-header">
          <h3>Session Complete</h3>
          <div className="summary-grade" style={{ color: getColor(data.average_score) }}>
            {data.overall_grade}
          </div>
        </div>

        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Role</span>
            <span className="stat-value">{data.role}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Level</span>
            <span className="stat-value">{data.level}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Category</span>
            <span className="stat-value">{data.category}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Questions</span>
            <span className="stat-value">{data.questions_answered}/{data.total_questions}</span>
          </div>
          <div className="stat-item highlight">
            <span className="stat-label">Average Score</span>
            <span className="stat-value" style={{ color: getColor(data.average_score) }}>
              {data.average_score.toFixed(1)}/100
            </span>
          </div>
        </div>

        {rubric && (
          <div className="summary-rubric">
            <div className="summary-rubric-header">
              <h4>Technical Rubric Averages</h4>
              <span className="summary-rubric-note">Technical questions only</span>
            </div>
            <div className="score-bars">
              <RubricBar label="Correct" score={rubric.correctness} />
              <RubricBar label="Complete" score={rubric.completeness} />
              <RubricBar label="Clarity" score={rubric.clarity} />
              <RubricBar label="Depth" score={rubric.depth} />
            </div>
          </div>
        )}

        {star && (
          <div className="summary-rubric">
            <div className="summary-rubric-header">
              <h4>Behavioral STAR Averages</h4>
              <span className="summary-rubric-note">Behavioral questions only</span>
            </div>
            <div className="score-bars">
              <RubricBar label="Situation" score={star.situation} />
              <RubricBar label="Task" score={star.task} />
              <RubricBar label="Action" score={star.action} />
              <RubricBar label="Result" score={star.result} />
              <RubricBar label="Reflection" score={star.reflection} />
            </div>
          </div>
        )}

        <div className="summary-results">
          <h4>Per-Question Breakdown</h4>
          <div className="results-table">
            <div className="results-header">
              <span>#</span>
              <span>Question</span>
              <span>SBERT</span>
              <span>NLI</span>
              <span>KW</span>
              <span>Judge</span>
              <span>Score</span>
              <span>Grade</span>
            </div>
            {data.results.map((r) => (
              <div key={r.index} className="results-row">
                <span>{r.index + 1}</span>
                <div className="result-question-cell">
                  <span className="result-question" title={r.question_text}>
                    {r.question_text.substring(0, 35)}...
                  </span>
                  <div className="result-meta-row">
                    <span
                      className={`result-type-badge ${
                        r.is_behavioral ? "behavioral" : "technical"
                      }`}
                    >
                      {r.is_behavioral ? "Behavioral" : "Technical"}
                    </span>
                    {r.is_behavioral && r.star_scores && (
                      <span className="result-star-inline">
                        S {r.star_scores.situation.toFixed(0)} T {r.star_scores.task.toFixed(0)} A {r.star_scores.action.toFixed(0)} R {r.star_scores.result.toFixed(0)} Ref {r.star_scores.reflection.toFixed(0)}
                      </span>
                    )}
                  </div>
                </div>
                <span>{r.sbert_score.toFixed(0)}</span>
                <span>{r.nli_score.toFixed(0)}</span>
                <span>{r.keyword_score.toFixed(0)}</span>
                <span>{r.llm_score.toFixed(0)}</span>
                <span style={{ color: getColor(r.composite_score), fontWeight: 600 }}>
                  {r.composite_score.toFixed(0)}
                </span>
                <span style={{ color: getColor(r.composite_score) }}>{r.grade}</span>
              </div>
            ))}
          </div>
        </div>

        {data.results && data.results.length > 1 && (
          <div className="summary-highlights">
            <div className="highlight-item best">
              <span className="highlight-label">Strongest</span>
              <span className="highlight-value">{data.strongest_area}</span>
            </div>
            <div className="highlight-item worst">
              <span className="highlight-label">Needs Work</span>
              <span className="highlight-value">{data.weakest_area}</span>
            </div>
          </div>
        )}

        {onRestart && (
          <button className="restart-btn" onClick={onRestart}>
            Start New Session
          </button>
        )}
      </div>
    </div>
  );
}
