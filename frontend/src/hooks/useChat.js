import { useState, useCallback, useRef, useEffect } from "react";
import {
  generateQuestions,
  evaluateAnswerSSE,
  getSessionSummary,
  checkSessionStatus,
  getAnswerResult,
} from "../utils/api";
import { parseSetupValue } from "../utils/setup";

const STORAGE_KEY = "interview_session";
const RECOVERY_POLL_INTERVAL_MS = 1000;
const MAX_RECOVERY_POLLS = 20;

const PHASES = {
  SETUP: "setup",
  INTERVIEW: "interview",
  SUMMARY: "summary",
};

const ROLE_QUICK_REPLIES = [
  "Software Engineer",
  "Data Scientist",
  "ML Engineer",
  "Full Stack Developer",
  "Product Manager",
];

const CATEGORY_QUICK_REPLIES = [
  "Data Structures",
  "Algorithms",
  "System Design",
  "OOP",
  "Databases",
  "Machine Learning",
  "Behavioral",
  "Web Development",
];

const LEVEL_QUICK_REPLIES = [
  "Intern",
  "Junior",
  "Mid-Level",
  "Senior",
  "Lead",
];

const DIFFICULTY_QUICK_REPLIES = [
  "Easy",
  "Medium",
  "Hard",
];

const COUNT_QUICK_REPLIES = [
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
];

const SETUP_STEPS = [
  {
    key: "role",
    prompt: "Welcome to PrepBuddy! Let's set up your practice session.\n\nWhat role are you preparing for?",
    quickReplies: ROLE_QUICK_REPLIES,
    allowCustom: true,
  },
  {
    key: "level",
    prompt: "What experience level?",
    quickReplies: LEVEL_QUICK_REPLIES,
  },
  {
    key: "category",
    prompt: "Which category of questions would you like?",
    quickReplies: CATEGORY_QUICK_REPLIES,
    allowCustom: true,
  },
  {
    key: "difficulty",
    prompt: "What difficulty level?",
    quickReplies: DIFFICULTY_QUICK_REPLIES,
  },
  {
    key: "num_questions",
    prompt: "How many questions would you like? (1-10)",
    quickReplies: COUNT_QUICK_REPLIES,
  },
];

function buildSetupPrompt(step) {
  if (!step) {
    return null;
  }

  return {
    role: "ai",
    type: "text",
    content: step.prompt,
    quickReplies: step.quickReplies,
    allowCustom: step.allowCustom,
    setupKey: step.key,
  };
}

function buildQuestionMessage(question, questionNum, totalQuestions) {
  return {
    role: "ai",
    type: "question",
    content: question.question_text,
    questionNum,
    totalQuestions,
    category: question.category,
    difficulty: question.difficulty,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadSavedSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* corrupt data — ignore */ }
  return null;
}

export default function useChat() {
  const saved = useRef(loadSavedSession());

  const [messages, setMessages] = useState(() => {
    if (saved.current) {
      return saved.current.messages;
    }
    return [buildSetupPrompt(SETUP_STEPS[0])];
  });
  const [phase, setPhase] = useState(() => saved.current?.phase ?? PHASES.SETUP);
  const [setupIndex, setSetupIndex] = useState(0);
  const [setupData, setSetupData] = useState({});
  const [sessionId, setSessionId] = useState(() => saved.current?.sessionId ?? null);
  const [questions, setQuestions] = useState(() => saved.current?.questions ?? []);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(
    () => saved.current?.currentQuestionIndex ?? 0
  );
  const [isLoading, setIsLoading] = useState(false);
  const [setupInputPlaceholder, setSetupInputPlaceholder] = useState("");
  const [inputFocusSignal, setInputFocusSignal] = useState(0);

  const questionsRef = useRef(saved.current?.questions ?? []);
  const abortRef = useRef(null);
  const abortReasonRef = useRef(null);

  const resetToSetup = useCallback((notice) => {
    localStorage.removeItem(STORAGE_KEY);
    const nextMessages = [];
    if (notice) {
      nextMessages.push({ role: "ai", type: "text", content: notice });
    }
    nextMessages.push(buildSetupPrompt(SETUP_STEPS[0]));
    setMessages(nextMessages);
    setPhase(PHASES.SETUP);
    setSetupIndex(0);
    setSetupData({});
    setSessionId(null);
    setQuestions([]);
    setCurrentQuestionIndex(0);
    setIsLoading(false);
    setSetupInputPlaceholder("");
    questionsRef.current = [];
  }, []);

  // Persist interview state to localStorage (strip transient progress messages)
  useEffect(() => {
    if (phase === PHASES.INTERVIEW && sessionId) {
      const persistable = messages.filter((m) => m.type !== "progress");
      const data = { messages: persistable, phase, sessionId, questions, currentQuestionIndex };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }
    if (phase === PHASES.SUMMARY) {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [messages, phase, sessionId, questions, currentQuestionIndex]);

  const addMessage = useCallback((msg) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const appendQuestionMessage = useCallback((questionIndex, allQuestions = questionsRef.current) => {
    if (questionIndex < 0 || questionIndex >= allQuestions.length) {
      return;
    }
    addMessage(
      buildQuestionMessage(
        allQuestions[questionIndex],
        questionIndex + 1,
        allQuestions.length
      )
    );
  }, [addMessage]);

  const applyEvaluationResult = useCallback(async (
    data,
    evaluatedQuestionIndex,
    { recoveryNotice } = {}
  ) => {
    setMessages((prev) => {
      const filtered = prev.filter((m) => m.type !== "progress");
      const next = [...filtered];
      if (recoveryNotice) {
        next.push({ role: "ai", type: "text", content: recoveryNotice });
      }
      next.push({ role: "ai", type: "scorecard", data });
      return next;
    });

    if (data.is_last_question) {
      addMessage({
        role: "ai",
        type: "text",
        content: "That was the last question! Let me compile your session summary...",
      });

      try {
        const summary = await getSessionSummary(sessionId);
        setPhase(PHASES.SUMMARY);
        addMessage({ role: "ai", type: "summary", data: summary });
      } catch {
        addMessage({
          role: "ai",
          type: "text",
          content: "Could not fetch session summary. Your individual scores are shown above.",
        });
        setPhase(PHASES.SUMMARY);
      }
      return;
    }

    const nextIdx = evaluatedQuestionIndex + 1;
    setCurrentQuestionIndex(nextIdx);
    appendQuestionMessage(nextIdx);
  }, [addMessage, appendQuestionMessage, sessionId]);

  const updateProgressMessage = useCallback((text) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.type === "progress") {
        return [...prev.slice(0, -1), { ...last, content: text }];
      }
      return prev;
    });
  }, []);

  const recoverInterruptedEvaluation = useCallback(async (questionIndex) => {
    if (!sessionId) {
      return false;
    }

    updateProgressMessage("Connection interrupted. Checking evaluation status...");

    for (let attempt = 0; attempt < MAX_RECOVERY_POLLS; attempt += 1) {
      const status = await checkSessionStatus(sessionId);

      if (status.kind === "missing") {
        resetToSetup("Your previous session is no longer available. Starting fresh.");
        return true;
      }

      if (status.kind === "unavailable") {
        return false;
      }

      const answered = status.data.answers_count > questionIndex;
      const inProgress = (status.data.in_progress_indices || []).includes(questionIndex);

      if (answered) {
        try {
          const recovered = await getAnswerResult(sessionId, questionIndex);
          await applyEvaluationResult(recovered, questionIndex, {
            recoveryNotice:
              "Connection interrupted, but your evaluation completed. Restored the result below.",
          });
          return true;
        } catch {
          return false;
        }
      }

      if (!inProgress) {
        return false;
      }

      updateProgressMessage("Connection interrupted. Waiting for the evaluation to finish...");
      await sleep(RECOVERY_POLL_INTERVAL_MS);
    }

    return false;
  }, [applyEvaluationResult, resetToSetup, sessionId, updateProgressMessage]);

  // Validate and reconcile restored session against backend on mount
  useEffect(() => {
    const s = saved.current;
    if (!s?.sessionId) return;
    let cancelled = false;

    (async () => {
      try {
        const status = await checkSessionStatus(s.sessionId);
        if (cancelled || status.kind === "unavailable") {
          return;
        }
        if (status.kind === "missing") {
          resetToSetup();
          return;
        }

        const sessionStatus = status.data;

        if (sessionStatus.state === "complete") {
          localStorage.removeItem(STORAGE_KEY);
          setPhase(PHASES.SUMMARY);
          try {
            const summary = await getSessionSummary(s.sessionId);
            if (!cancelled) {
              setMessages((prev) => [...prev, { role: "ai", type: "summary", data: summary }]);
            }
          } catch {
            if (!cancelled) {
              resetToSetup(
                "Session was already completed, but the summary could not be loaded. Starting fresh."
              );
            }
          }
          return;
        }

        const backendIdx = sessionStatus.answers_count;
        const localIdx = s.currentQuestionIndex;
        const allQ = s.questions || [];

        if (backendIdx > localIdx) {
          try {
            const recovered = await getAnswerResult(s.sessionId, localIdx);
            if (!cancelled) {
              await applyEvaluationResult(recovered, localIdx, {
                recoveryNotice:
                  "Recovered a completed evaluation from your previous session.",
              });
            }
          } catch {
            if (!cancelled && backendIdx < allQ.length) {
              setCurrentQuestionIndex(backendIdx);
              setMessages((prev) => {
                const trimmed = prev.filter(
                  (m, i) => !(m.type === "question" && i === prev.length - 1 && m.questionNum !== backendIdx + 1)
                );
                return [...trimmed, buildQuestionMessage(allQ[backendIdx], backendIdx + 1, allQ.length)];
              });
            }
          }
          return;
        }

        if (backendIdx !== localIdx && backendIdx < allQ.length) {
          setCurrentQuestionIndex(backendIdx);
          setMessages((prev) => {
            const trimmed = prev.filter(
              (m, i) => !(m.type === "question" && i === prev.length - 1 && m.questionNum !== backendIdx + 1)
            );
            return [...trimmed, buildQuestionMessage(allQ[backendIdx], backendIdx + 1, allQ.length)];
          });
          return;
        }

        const inProgress = (sessionStatus.in_progress_indices || []).includes(localIdx);
        if (inProgress) {
          setMessages((prev) => {
            if (prev.some((m) => m.type === "progress")) {
              return prev;
            }
            return [
              ...prev,
              {
                role: "ai",
                type: "progress",
                content: "Reconnecting to your in-progress evaluation...",
              },
            ];
          });
          const recovered = await recoverInterruptedEvaluation(localIdx);
          if (!recovered && !cancelled) {
            setMessages((prev) => {
              const filtered = prev.filter((m) => m.type !== "progress");
              return [
                ...filtered,
                {
                  role: "ai",
                  type: "text",
                  content: "The previous evaluation did not complete. You can submit the answer again.",
                },
              ];
            });
          }
        }
      } catch {
        if (!cancelled) {
          resetToSetup();
        }
      }
    })();

    return () => { cancelled = true; };
  }, [applyEvaluationResult, recoverInterruptedEvaluation, resetToSetup]);

  const handleSetupInput = useCallback(
    async (userInput) => {
      const step = SETUP_STEPS[setupIndex];
      const parsed = parseSetupValue(step.key, userInput);
      if (parsed.error) {
        addMessage({ role: "ai", type: "text", content: parsed.error });
        return;
      }
      const trimmed = userInput.trim();
      const normalizedValue = parsed.value;
      const numericValue = parsed.numericValue;

      if (!normalizedValue) {
        addMessage({ role: "ai", type: "text", content: "Please enter a valid value." });
        return;
      }

      const newSetupData = { ...setupData, [step.key]: normalizedValue };
      setSetupData(newSetupData);
      setSetupInputPlaceholder("");

      addMessage({ role: "user", type: "text", content: trimmed });

      if (setupIndex < SETUP_STEPS.length - 1) {
        const nextStep = SETUP_STEPS[setupIndex + 1];
        addMessage(buildSetupPrompt(nextStep));
        setSetupIndex(setupIndex + 1);
      } else {
        // All setup done — generate questions
        setIsLoading(true);
        addMessage({
          role: "ai",
          type: "text",
          content: `Great! Setting up your ${newSetupData.role} interview with ${numericValue} ${newSetupData.difficulty} questions on ${newSetupData.category}...\n\nGenerating questions, please wait...`,
        });

        try {
          const config = {
            role: newSetupData.role,
            level: newSetupData.level,
            category: newSetupData.category,
            difficulty: newSetupData.difficulty,
            num_questions: numericValue,
          };

          const data = await generateQuestions(config);
          setSessionId(data.session_id);
          setQuestions(data.questions);
          questionsRef.current = data.questions;
          setCurrentQuestionIndex(0);
          setPhase(PHASES.INTERVIEW);

          addMessage({
            role: "ai",
            type: "text",
            content: `I've prepared ${data.questions.length} questions for you. Let's begin!\n\nType your answer for each question. Take your time — I'll evaluate your response after you submit.`,
          });
          addMessage({
            role: "ai",
            type: "question",
            content: data.questions[0].question_text,
            questionNum: 1,
            totalQuestions: data.questions.length,
            category: data.questions[0].category,
            difficulty: data.questions[0].difficulty,
          });
        } catch (err) {
          addMessage({
            role: "ai",
            type: "text",
            content: `Error generating questions: ${err.message}. Please check that the backend is running and try again.`,
          });
        } finally {
          setIsLoading(false);
        }
      }
    },
    [setupIndex, setupData, addMessage]
  );

  const handleQuickReply = useCallback(
    (value) => {
      const step = SETUP_STEPS[setupIndex];
      if (!step) {
        return;
      }

      if (value === "__other__") {
        const label = step.key === "role" ? "role" : "category";
        setSetupInputPlaceholder(`Type a custom ${label}...`);
        setInputFocusSignal((count) => count + 1);
        return;
      }

      handleSetupInput(value);
    },
    [handleSetupInput, setupIndex]
  );

  const PROGRESS_LABELS = {
    scoring_started: "Starting evaluation...",
    sbert_done: "Semantic analysis complete",
    nli_done: "Factual consistency checked",
    keyword_done: "Keyword coverage analyzed",
    claim_done: "Concept claims matched",
    llm_done: "LLM judge scored",
    scores_ready: "Computing final score...",
    feedback_started: "Generating feedback...",
  };

  const handleInterviewInput = useCallback(
    async (userInput) => {
      const trimmed = userInput.trim();
      if (!trimmed) return;

      addMessage({ role: "user", type: "text", content: trimmed });
      setIsLoading(true);
      addMessage({ role: "ai", type: "progress", content: "Starting evaluation..." });

      abortReasonRef.current = null;
      abortRef.current = new AbortController();

      try {
        const data = await evaluateAnswerSSE(
          sessionId,
          currentQuestionIndex,
          trimmed,
          (event) => {
            const label = PROGRESS_LABELS[event.step];
            if (label) {
              updateProgressMessage(label);
            }
          },
          { signal: abortRef.current.signal }
        );
        await applyEvaluationResult(data, currentQuestionIndex);
      } catch (err) {
        const manualAbort = err.name === "AbortError" && abortReasonRef.current === "restart";
        if (manualAbort) {
          setMessages((prev) => prev.filter((m) => m.type !== "progress"));
          return;
        }

        const recovered = await recoverInterruptedEvaluation(currentQuestionIndex);
        if (recovered) {
          return;
        }

        setMessages((prev) => {
          const filtered = prev.filter((m) => m.type !== "progress");
          return [
            ...filtered,
            {
              role: "ai",
              type: "text",
              content: `Error evaluating answer: ${err.message}`,
            },
          ];
        });
      } finally {
        abortReasonRef.current = null;
        abortRef.current = null;
        setIsLoading(false);
      }
    },
    [
      sessionId,
      currentQuestionIndex,
      addMessage,
      applyEvaluationResult,
      recoverInterruptedEvaluation,
      updateProgressMessage,
    ]
  );

  const handleSend = useCallback(
    (userInput) => {
      if (isLoading) return;
      if (phase === PHASES.SETUP) {
        handleSetupInput(userInput);
      } else if (phase === PHASES.INTERVIEW) {
        handleInterviewInput(userInput);
      }
      // In SUMMARY phase, no more input accepted
    },
    [phase, isLoading, handleSetupInput, handleInterviewInput]
  );

  const handleRestart = useCallback(() => {
    abortReasonRef.current = "restart";
    abortRef.current?.abort();
    abortRef.current = null;
    resetToSetup();
  }, [resetToSetup]);

  return {
    messages,
    phase,
    isLoading,
    handleSend,
    handleQuickReply,
    handleRestart,
    isComplete: phase === PHASES.SUMMARY,
    setupInputPlaceholder,
    inputFocusSignal,
    activeSetupStep: phase === PHASES.SETUP ? SETUP_STEPS[setupIndex] : null,
  };
}
