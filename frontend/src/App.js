import React from "react";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import useChat from "./hooks/useChat";

export default function App() {
  const { messages, phase, isLoading, handleSend, isComplete } = useChat();

  const getPlaceholder = () => {
    if (isComplete) return "Session complete. Refresh to start a new interview.";
    if (phase === "setup") return "Type your response...";
    return "Type your answer... (Shift+Enter for new line)";
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="header-title">
            <span className="header-icon">&#9881;</span> AI Interview Coach
          </h1>
          <span className="header-badge">NLP-Powered</span>
        </div>
      </header>
      <ChatWindow messages={messages} isLoading={isLoading} />
      <InputBar
        onSend={handleSend}
        disabled={isLoading || isComplete}
        placeholder={getPlaceholder()}
      />
    </div>
  );
}
