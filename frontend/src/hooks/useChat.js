import { useState, useCallback, useRef } from "react";
import { generateQuestions, evaluateAnswer, getSessionSummary } from "../utils/api";

const PHASES = {
  SETUP: "setup",
  INTERVIEW: "interview",
  SUMMARY: "summary",
};

const SETUP_STEPS = [
  {
    key: "role",
    prompt: "Welcome to the AI Interview Coach! Let's set up your practice session.\n\nWhat role are you preparing for?\n\nExamples: Software Engineer, Data Scientist, ML Engineer, Full Stack Developer, Product Manager",
  },
  {
    key: "level",
    prompt: "What experience level?\n\nOptions: Intern, Junior, Mid-Level, Senior, Lead",
  },
  {
    key: "category",
    prompt: "Which category of questions would you like?\n\nExamples: Data Structures, Algorithms, System Design, OOP, Databases, Machine Learning, Behavioral, Web Development",
  },
  {
    key: "difficulty",
    prompt: "What difficulty level?\n\nOptions: Easy, Medium, Hard",
  },
  {
    key: "num_questions",
    prompt: "How many questions would you like? (1-10)",
  },
];

export default function useChat() {
  const [messages, setMessages] = useState([
    { role: "ai", type: "text", content: SETUP_STEPS[0].prompt },
  ]);
  const [phase, setPhase] = useState(PHASES.SETUP);
  const [setupIndex, setSetupIndex] = useState(0);
  const [setupData, setSetupData] = useState({});
  const [sessionId, setSessionId] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const questionsRef = useRef([]);

  const addMessage = useCallback((msg) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const handleSetupInput = useCallback(
    async (userInput) => {
      const step = SETUP_STEPS[setupIndex];
      const trimmed = userInput.trim();

      // Validate num_questions
      if (step.key === "num_questions") {
        const num = parseInt(trimmed, 10);
        if (isNaN(num) || num < 1 || num > 10) {
          addMessage({ role: "ai", type: "text", content: "Please enter a number between 1 and 10." });
          return;
        }
      }

      const newSetupData = { ...setupData, [step.key]: trimmed };
      setSetupData(newSetupData);

      addMessage({ role: "user", type: "text", content: trimmed });

      if (setupIndex < SETUP_STEPS.length - 1) {
        const nextStep = SETUP_STEPS[setupIndex + 1];
        addMessage({ role: "ai", type: "text", content: nextStep.prompt });
        setSetupIndex(setupIndex + 1);
      } else {
        // All setup done — generate questions
        setIsLoading(true);
        addMessage({
          role: "ai",
          type: "text",
          content: `Great! Setting up your ${newSetupData.role} interview with ${trimmed} ${newSetupData.difficulty} questions on ${newSetupData.category}...\n\nGenerating questions, please wait...`,
        });

        try {
          const config = {
            role: newSetupData.role,
            level: newSetupData.level,
            category: newSetupData.category,
            difficulty: newSetupData.difficulty,
            num_questions: parseInt(trimmed, 10),
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

  const handleInterviewInput = useCallback(
    async (userInput) => {
      const trimmed = userInput.trim();
      if (!trimmed) return;

      addMessage({ role: "user", type: "text", content: trimmed });
      setIsLoading(true);
      addMessage({ role: "ai", type: "text", content: "Evaluating your answer..." });

      try {
        const data = await evaluateAnswer(sessionId, currentQuestionIndex, trimmed);

        // Remove "Evaluating..." message and add score card
        setMessages((prev) => {
          const filtered = prev.filter(
            (m, i) => !(i === prev.length - 1 && m.content === "Evaluating your answer...")
          );
          return [
            ...filtered,
            { role: "ai", type: "scorecard", data: data },
          ];
        });

        if (data.is_last_question) {
          // Fetch summary
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
        } else {
          const nextIdx = currentQuestionIndex + 1;
          setCurrentQuestionIndex(nextIdx);
          const allQ = questionsRef.current;
          if (nextIdx < allQ.length) {
            addMessage({
              role: "ai",
              type: "question",
              content: allQ[nextIdx].question_text,
              questionNum: nextIdx + 1,
              totalQuestions: allQ.length,
              category: allQ[nextIdx].category,
              difficulty: allQ[nextIdx].difficulty,
            });
          }
        }
      } catch (err) {
        setMessages((prev) => {
          const filtered = prev.filter(
            (m, i) => !(i === prev.length - 1 && m.content === "Evaluating your answer...")
          );
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
        setIsLoading(false);
      }
    },
    [sessionId, currentQuestionIndex, addMessage]
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

  return {
    messages,
    phase,
    isLoading,
    handleSend,
    isComplete: phase === PHASES.SUMMARY,
  };
}
