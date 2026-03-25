import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import ScoreCard from "./ScoreCard";
import SessionSummary from "./SessionSummary";
import CubeIcon from "./CubeIcon";

export default function ChatWindow({ messages, isLoading, onRestart, onQuickReply, phase, activeSetupStep }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.map((msg, i) => {
        if (msg.type === "scorecard") {
          return <ScoreCard key={i} data={msg.data} />;
        }
        if (msg.type === "summary") {
          return <SessionSummary key={i} data={msg.data} onRestart={onRestart} />;
        }
        if (msg.type === "progress") {
          return (
            <div key={i} className="message-row ai">
              <div className="ai-avatar"><CubeIcon size={15} /></div>
              <div className="message-bubble ai-bubble progress-bubble">
                <div className="progress-indicator">
                  <span className="progress-spinner" />
                  <span className="progress-text">{msg.content}</span>
                </div>
              </div>
            </div>
          );
        }
        return (
          <MessageBubble
            key={i}
            message={msg}
            showQuickReplies={phase === "setup" && msg.setupKey === activeSetupStep?.key}
            quickReplies={phase === "setup" && msg.setupKey === activeSetupStep?.key ? activeSetupStep?.quickReplies : undefined}
            allowCustom={phase === "setup" && msg.setupKey === activeSetupStep?.key ? activeSetupStep?.allowCustom : false}
            onQuickReply={onQuickReply}
          />
        );
      })}
      {isLoading && !messages.some((m) => m.type === "progress") && (
        <div className="message-row ai">
          <div className="ai-avatar"><CubeIcon size={15} /></div>
          <div className="message-bubble ai-bubble loading-bubble">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
