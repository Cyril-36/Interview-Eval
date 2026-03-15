import React, { useState } from "react";

function ScoreBar({ label, score, color }) {
  return (
    <div className="score-bar-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${Math.min(100, score)}%`, background: color }}
        />
      </div>
      <span className="score-value">{score.toFixed(1)}</span>
    </div>
  );
}

function getColor(score) {
  if (score >= 80) return "#3fb950";
  if (score >= 60) return "#d29922";
  if (score >= 40) return "#db6d28";
  return "#f85149";
}

export default function ScoreCard({ data }) {
  const [showDetails, setShowDetails] = useState(false);
  const { scores, feedback } = data;

  return (
    <div className="message-row ai">
      <div className="scorecard">
        <div className="scorecard-header">
          <div className="composite-score" style={{ borderColor: getColor(scores.composite_score) }}>
            <span className="composite-number">{scores.composite_score.toFixed(0)}</span>
            <span className="composite-label">/100</span>
          </div>
          <div className="composite-grade" style={{ color: getColor(scores.composite_score) }}>
            {scores.grade}
          </div>
        </div>

        <div className="score-bars">
          <ScoreBar label="Semantic" score={scores.sbert_score} color={getColor(scores.sbert_score)} />
          <ScoreBar label="NLI" score={scores.nli_score} color={getColor(scores.nli_score)} />
          <ScoreBar label="Keywords" score={scores.keyword_score} color={getColor(scores.keyword_score)} />
        </div>

        {scores.missing_keywords.length > 0 && (
          <div className="missing-keywords">
            <span className="mk-label">Missing concepts: </span>
            {scores.missing_keywords.map((kw, i) => (
              <span key={i} className="keyword-tag">{kw}</span>
            ))}
          </div>
        )}

        <button
          className="details-toggle"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? "Hide Feedback ▲" : "Show Feedback ▼"}
        </button>

        {showDetails && (
          <div className="feedback-section">
            {feedback.strengths.length > 0 && (
              <div className="feedback-group">
                <h4 className="feedback-title strengths-title">Strengths</h4>
                <ul>
                  {feedback.strengths.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {feedback.improvements.length > 0 && (
              <div className="feedback-group">
                <h4 className="feedback-title improvements-title">Areas to Improve</h4>
                <ul>
                  {feedback.improvements.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {feedback.model_answer && (
              <div className="feedback-group">
                <h4 className="feedback-title model-title">Model Answer</h4>
                <p className="model-answer">{feedback.model_answer}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
