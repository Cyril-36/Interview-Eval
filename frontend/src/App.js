import React, { useEffect, useRef, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import CubeIcon from "./components/CubeIcon";
import ResumeUploader from "./components/ResumeUploader";
import useChat from "./hooks/useChat";

export default function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);
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
    resumeData,
    handleResumeUpload,
    handleResumeClear,
    handleResumeRemoveSkill,
  } = useChat();

  const getPlaceholder = () => {
    if (isComplete) return "Session complete. Click 'Start New Session' below.";
    if (phase === "setup") return setupInputPlaceholder || "Type your response...";
    return "Type your answer... (Shift+Enter for new line)";
  };

  useEffect(() => {
    if (!isMenuOpen) return undefined;

    const handlePointerDown = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
      }
    };
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isMenuOpen]);

  const handleNewSession = () => {
    setIsMenuOpen(false);
    handleRestart();
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="header-title">
            <span className="header-icon"><CubeIcon size={20} /></span> PrepBuddy
          </h1>
          <div className="header-actions" ref={menuRef}>
            <span className="header-badge">NLP + LLM</span>
            <button
              type="button"
              className="header-menu-btn"
              aria-label="Open menu"
              aria-expanded={isMenuOpen}
              aria-haspopup="menu"
              onClick={() => setIsMenuOpen((open) => !open)}
            >
              <span />
              <span />
              <span />
            </button>
            {isMenuOpen && (
              <div className="header-menu" role="menu">
                <button
                  type="button"
                  className="header-menu-item"
                  role="menuitem"
                  onClick={handleNewSession}
                  disabled={isLoading}
                >
                  New Session
                </button>
              </div>
            )}
          </div>
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
      {phase === "setup" && (
        <ResumeUploader
          resumeData={resumeData}
          onUpload={handleResumeUpload}
          onClear={handleResumeClear}
          onRemoveSkill={handleResumeRemoveSkill}
          disabled={isLoading}
        />
      )}
      <InputBar
        onSend={handleSend}
        disabled={isLoading || isComplete}
        placeholder={getPlaceholder()}
        focusSignal={inputFocusSignal}
      />
    </div>
  );
}
