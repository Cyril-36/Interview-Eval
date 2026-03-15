import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import ScoreCard from "./ScoreCard";
import SessionSummary from "./SessionSummary";

export default function ChatWindow({ messages, isLoading }) {
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
          return <SessionSummary key={i} data={msg.data} />;
        }
        return <MessageBubble key={i} message={msg} />;
      })}
      {isLoading && (
        <div className="message-row ai">
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
