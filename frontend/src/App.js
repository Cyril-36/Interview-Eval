import React from "react";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import CubeIcon from "./components/CubeIcon";
import useChat from "./hooks/useChat";

export default function App() {
  const {
    messages,
    phase,
    isLoading,
    handleSend,
    handleQuickReply,
    handleRestart,
    isComplete,
    setupInputPlaceholder,
    inputFocusSignal,
    activeSetupStep,
  } = useChat();

  const getPlaceholder = () => {
    if (isComplete) return "Session complete. Click 'Start New Session' below.";
    if (phase === "setup") return setupInputPlaceholder || "Type your response...";
    return "Type your answer... (Shift+Enter for new line)";
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="header-title">
            <span className="header-icon"><CubeIcon size={20} /></span> PrepBuddy
          </h1>
          <span className="header-badge">NLP + LLM</span>
        </div>
      </header>
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onRestart={handleRestart}
        onQuickReply={handleQuickReply}
        phase={phase}
        activeSetupStep={activeSetupStep}
      />
      <InputBar
        onSend={handleSend}
        disabled={isLoading || isComplete}
        placeholder={getPlaceholder()}
        focusSignal={inputFocusSignal}
      />
    </div>
  );
}
