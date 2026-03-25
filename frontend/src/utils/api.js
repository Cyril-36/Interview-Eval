const API_BASE =
  process.env.REACT_APP_API_BASE || "http://localhost:8000/api";

const REQUEST_TIMEOUT_MS = 60000; // 60 seconds

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timeoutId);
  }
}

function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (!item || typeof item !== "object") {
        return String(item);
      }
      const path = Array.isArray(item.loc)
        ? item.loc.filter((part) => part !== "body").join(".")
        : "";
      const message = item.msg || item.detail || JSON.stringify(item);
      return path ? `${path}: ${message}` : message;
    });
    return parts.join("; ");
  }

  if (detail && typeof detail === "object") {
    return detail.detail || JSON.stringify(detail);
  }

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  return null;
}

export async function generateQuestions(config) {
  const res = await fetchWithTimeout(`${API_BASE}/generate_questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(formatErrorDetail(err.detail) || "Failed to generate questions");
  }
  return res.json();
}

export async function evaluateAnswer(sessionId, questionIndex, candidateAnswer) {
  const res = await fetchWithTimeout(`${API_BASE}/evaluate_answer`, {
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
    throw new Error(formatErrorDetail(err.detail) || "Failed to evaluate answer");
  }
  return res.json();
}

const SSE_TIMEOUT_MS = 120000; // 2 minutes for full scoring + feedback pipeline

export async function evaluateAnswerSSE(sessionId, questionIndex, candidateAnswer, onEvent, { signal } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SSE_TIMEOUT_MS);

  // If an external signal is provided, abort the internal controller when it fires
  const onExternalAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", onExternalAbort);
    }
  }

  try {
    const res = await fetch(`${API_BASE}/evaluate_answer_sse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        question_index: questionIndex,
        candidate_answer: candidateAnswer,
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(formatErrorDetail(err.detail) || "Failed to evaluate answer");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const event = JSON.parse(line.slice(6));
            if (event.step === "error") {
              throw new Error(event.message || "Evaluation failed on server");
            }
            if (event.step === "done") {
              finalData = event;
            }
            onEvent(event);
          } catch (e) {
            if (e.message && !e.message.includes("JSON")) throw e;
            // skip malformed JSON
          }
        }
      }
    }

    if (!finalData) {
      throw new Error("Stream ended without final result");
    }
    return finalData;
  } finally {
    clearTimeout(timeoutId);
    if (signal) {
      signal.removeEventListener("abort", onExternalAbort);
    }
  }
}

export async function checkSessionStatus(sessionId) {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/session_status?session_id=${sessionId}`
    );
    if (res.status === 404) {
      return { kind: "missing" };
    }
    if (!res.ok) {
      return { kind: "unavailable" };
    }
    return {
      kind: "ok",
      data: await res.json(),
    };
  } catch {
    return { kind: "unavailable" };
  }
}

export async function getSessionSummary(sessionId) {
  const res = await fetchWithTimeout(
    `${API_BASE}/session_summary?session_id=${sessionId}`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(formatErrorDetail(err.detail) || "Failed to get summary");
  }
  return res.json();
}
