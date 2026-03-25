import React from "react";
import CubeIcon from "./CubeIcon";

export default function MessageBubble({
  message,
  showQuickReplies = false,
  quickReplies,
  allowCustom = false,
  onQuickReply,
}) {
  const isUser = message.role === "user";
  const isPlainAiText = !isUser && message.type === "text";

  if (message.type === "question") {
    return (
      <div className="message-row ai">
        <div className="ai-avatar"><CubeIcon size={15} /></div>
        <div className="message-bubble question-bubble">
          <div className="question-header">
            <span className="question-badge">
              Q{message.questionNum}/{message.totalQuestions}
            </span>
            <span className="question-meta">
              {message.category} &middot; {message.difficulty}
            </span>
          </div>
          <div className="question-text">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`message-row ${isUser ? "user" : "ai"} ${isPlainAiText ? "ai-plain-row" : ""}`}>
      {isUser ? (
        <div className="user-avatar" aria-hidden="true">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth="1.5"
            stroke="currentColor"
            width="18"
            height="18"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z"
            />
          </svg>
        </div>
      ) : (
        !isPlainAiText && <div className="ai-avatar"><CubeIcon size={15} /></div>
      )}
      <div className={`message-bubble ${isUser ? "user-bubble" : "ai-bubble"} ${isPlainAiText ? "ai-plain-text" : ""}`}>
        {message.content.split("\n").map((line, i) => (
          <React.Fragment key={i}>
            {line}
            {i < message.content.split("\n").length - 1 && <br />}
          </React.Fragment>
        ))}
        {showQuickReplies && Array.isArray(quickReplies) && quickReplies.length > 0 && (
          <div className="quick-replies">
            {[
              ...quickReplies.map((label) => ({ label, value: label })),
              ...(allowCustom ? [{ label: "Other", value: "__other__" }] : []),
            ].map((option) => (
              <button
                key={option.value}
                type="button"
                className={`quick-reply-chip ${option.value === "__other__" ? "secondary" : ""}`}
                onClick={() => onQuickReply?.(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
