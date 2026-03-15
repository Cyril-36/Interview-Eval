import React from "react";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  if (message.type === "question") {
    return (
      <div className="message-row ai">
        <div className="message-bubble question-bubble">
          <div className="question-header">
            <span className="question-badge">
              Q{message.questionNum}/{message.totalQuestions}
            </span>
            <span className="question-meta">
              {message.category} · {message.difficulty}
            </span>
          </div>
          <div className="question-text">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`message-row ${isUser ? "user" : "ai"}`}>
      <div className={`message-bubble ${isUser ? "user-bubble" : "ai-bubble"}`}>
        {message.content.split("\n").map((line, i) => (
          <React.Fragment key={i}>
            {line}
            {i < message.content.split("\n").length - 1 && <br />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
