import React from "react";

function getColor(score) {
  if (score >= 80) return "#3fb950";
  if (score >= 60) return "#d29922";
  if (score >= 40) return "#db6d28";
  return "#f85149";
}

export default function SessionSummary({ data }) {
  return (
    <div className="message-row ai">
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

        <div className="summary-results">
          <h4>Per-Question Breakdown</h4>
          <div className="results-table">
            <div className="results-header">
              <span>#</span>
              <span>Question</span>
              <span>SBERT</span>
              <span>NLI</span>
              <span>KW</span>
              <span>Score</span>
              <span>Grade</span>
            </div>
            {data.results.map((r) => (
              <div key={r.index} className="results-row">
                <span>{r.index + 1}</span>
                <span className="result-question" title={r.question_text}>
                  {r.question_text.substring(0, 40)}...
                </span>
                <span>{r.sbert_score.toFixed(0)}</span>
                <span>{r.nli_score.toFixed(0)}</span>
                <span>{r.keyword_score.toFixed(0)}</span>
                <span style={{ color: getColor(r.composite_score), fontWeight: 600 }}>
                  {r.composite_score.toFixed(0)}
                </span>
                <span style={{ color: getColor(r.composite_score) }}>{r.grade}</span>
              </div>
            ))}
          </div>
        </div>

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
      </div>
    </div>
  );
}
