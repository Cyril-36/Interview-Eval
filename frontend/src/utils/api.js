const API_BASE = "http://localhost:8000/api";

export async function generateQuestions(config) {
  const res = await fetch(`${API_BASE}/generate_questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Failed to generate questions");
  }
  return res.json();
}

export async function evaluateAnswer(sessionId, questionIndex, candidateAnswer) {
  const res = await fetch(`${API_BASE}/evaluate_answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question_index: questionIndex,
      candidate_answer: candidateAnswer,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Failed to evaluate answer");
  }
  return res.json();
}

export async function getSessionSummary(sessionId) {
  const res = await fetch(`${API_BASE}/session_summary?session_id=${sessionId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Failed to get summary");
  }
  return res.json();
}
