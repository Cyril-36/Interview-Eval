import React, { useState } from "react";
import CubeIcon from "./CubeIcon";
import { getColor } from "../utils/colors";

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

function getQuickTip(scores) {
  if (scores.is_behavioral && scores.star_scores) {
    const starTips = {
      situation: "Set the context more clearly before moving into the story.",
      task: "Explain the responsibility or challenge you personally owned.",
      action: "Add more concrete steps you personally took.",
      result: "State the outcome more clearly and include impact if possible.",
      reflection: "Add what you learned or how you grew from the experience.",
    };

    const entries = Object.entries(scores.star_scores);
    const weakest = entries.reduce((lowest, current) => (
      current[1] < lowest[1] ? current : lowest
    ));

    if (weakest[1] < 78) {
      return starTips[weakest[0]];
    }
    return null;
  }

  if (!scores.rubric_scores) {
    return null;
  }

  const rubricTips = {
    correctness: "Check the technical accuracy of the answer before you submit.",
    completeness: "Cover more of the core points the interviewer is looking for.",
    clarity: "Structure the answer more clearly so the explanation is easier to follow.",
    depth: "Add more reasoning, trade-offs, or detail to show deeper understanding.",
  };

  const entries = Object.entries(scores.rubric_scores);
  const weakest = entries.reduce((lowest, current) => (
    current[1] < lowest[1] ? current : lowest
  ));

  if (weakest[1] < 78) {
    return rubricTips[weakest[0]];
  }

  return null;
}

export default function ScoreCard({ data }) {
  const [showDetails, setShowDetails] = useState(false);
  const { scores, feedback } = data;
  const rubricScores = scores.rubric_scores;
  const judgeLabel = scores.is_behavioral ? "STAR Judge" : "LLM Judge";
  const quickTip = getQuickTip(scores);

  return (
    <div className="message-row ai">
      <div className="ai-avatar"><CubeIcon size={15} /></div>
      <div className="scorecard">
        <div className="scorecard-header">
          <div className="composite-score" style={{ borderColor: getColor(scores.composite_score) }}>
            <span className="composite-number">{scores.composite_score.toFixed(0)}</span>
            <span className="composite-label">/100</span>
          </div>
          <div className="composite-grade" style={{ color: getColor(scores.composite_score) }}>
            {scores.grade}
            {scores.is_behavioral && <span className="behavioral-badge">STAR</span>}
          </div>
        </div>

        <div className="score-bars">
          <ScoreBar label="Semantic" score={scores.sbert_score} color={getColor(scores.sbert_score)} />
          <ScoreBar label="NLI" score={scores.nli_score} color={getColor(scores.nli_score)} />
          <ScoreBar label="Keywords" score={scores.keyword_score} color={getColor(scores.keyword_score)} />
          <ScoreBar label={judgeLabel} score={scores.llm_score} color={getColor(scores.llm_score)} />
        </div>

        {quickTip && (
          <div className="quick-tip">
            <span className="quick-tip-label">Quick tip</span>
            <span className="quick-tip-text">{quickTip}</span>
          </div>
        )}

        {scores.missing_keywords.length > 0 && (() => {
          const items = scores.missing_keywords;
          const avgLen =
            items.reduce((sum, item) => sum + item.length, 0) / items.length;
          const looksLikeSentences = avgLen > 30;
          return (
            <div
              className={
                looksLikeSentences
                  ? "missing-keywords missing-keywords-list"
                  : "missing-keywords"
              }
            >
              <span className="mk-label">Missing concepts:</span>
              {looksLikeSentences ? (
                <ul className="missing-list">
                  {items.map((kw, i) => (
                    <li key={i} className="missing-list-item">{kw}</li>
                  ))}
                </ul>
              ) : (
                items.map((kw, i) => (
                  <span key={i} className="keyword-tag">{kw}</span>
                ))
              )}
            </div>
          );
        })()}

        <button
          className="details-toggle"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? "Hide Feedback \u25B2" : "Show Feedback \u25BC"}
        </button>

        {showDetails && (
          <div className="feedback-section">
            {scores.claim_matches && scores.claim_matches.length > 0 && (
              <div className="feedback-group">
                <h4 className="feedback-title">Claim Coverage</h4>
                <ul className="claim-list">
                  {scores.claim_matches.map((cm, i) => {
                    const isContradicted = cm.contradiction > 0.3;
                    const isOptional = cm.importance === "optional";
                    const isPartial = !cm.covered && !isContradicted && cm.partial;
                    let iconClass;
                    let icon;
                    if (isContradicted) {
                      iconClass = "contradicted";
                      icon = "\u26A0";
                    } else if (cm.covered) {
                      iconClass = "covered";
                      icon = "\u2713";
                    } else if (isPartial) {
                      iconClass = "partial";
                      icon = "\u25D1"; // half-filled circle: partial coverage
                    } else if (isOptional) {
                      iconClass = "optional";
                      icon = "\u25CB";
                    } else {
                      iconClass = "missed";
                      icon = "\u2717";
                    }
                    // Display the combined match score the coverage decision
                    // actually uses, falling back to raw similarity for older
                    // persisted records that don't have match_score.
                    const displayScore =
                      typeof cm.match_score === "number" && cm.match_score > 0
                        ? cm.match_score
                        : cm.similarity;
                    return (
                      <li key={i} className="claim-item">
                        <span className={`claim-icon ${iconClass}`}>{icon}</span>
                        <span className="claim-text">
                          {cm.claim}
                          {isOptional && (
                            <span className="claim-importance">optional</span>
                          )}
                        </span>
                        <span className="claim-similarity">{(displayScore * 100).toFixed(0)}%</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
            {scores.is_behavioral && scores.star_scores ? (
              <div className="feedback-group">
                <h4 className="feedback-title">STAR Assessment</h4>
                <div className="score-bars rubric-score-bars">
                  <ScoreBar
                    label="Situation"
                    score={scores.star_scores.situation}
                    color={getColor(scores.star_scores.situation)}
                  />
                  <ScoreBar
                    label="Task"
                    score={scores.star_scores.task}
                    color={getColor(scores.star_scores.task)}
                  />
                  <ScoreBar
                    label="Action"
                    score={scores.star_scores.action}
                    color={getColor(scores.star_scores.action)}
                  />
                  <ScoreBar
                    label="Result"
                    score={scores.star_scores.result}
                    color={getColor(scores.star_scores.result)}
                  />
                  <ScoreBar
                    label="Reflection"
                    score={scores.star_scores.reflection}
                    color={getColor(scores.star_scores.reflection)}
                  />
                </div>
                {scores.llm_reason && (
                  <p className="judge-reason">{scores.llm_reason}</p>
                )}
              </div>
            ) : rubricScores ? (
              <div className="feedback-group">
                <h4 className="feedback-title">LLM Rubric</h4>
                <div className="score-bars rubric-score-bars">
                  <ScoreBar
                    label="Correct"
                    score={rubricScores.correctness}
                    color={getColor(rubricScores.correctness)}
                  />
                  <ScoreBar
                    label="Complete"
                    score={rubricScores.completeness}
                    color={getColor(rubricScores.completeness)}
                  />
                  <ScoreBar
                    label="Clarity"
                    score={rubricScores.clarity}
                    color={getColor(rubricScores.clarity)}
                  />
                  <ScoreBar
                    label="Depth"
                    score={rubricScores.depth}
                    color={getColor(rubricScores.depth)}
                  />
                </div>
                {scores.llm_reason && (
                  <p className="judge-reason">{scores.llm_reason}</p>
                )}
              </div>
            ) : null}
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
