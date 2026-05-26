---
title: PrepBuddy Backend
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hybrid NLP + LLM interview answer evaluator (FastAPI)
---

# PrepBuddy Backend

FastAPI service that evaluates free-form interview answers using a hybrid
multi-signal pipeline: Sentence-BERT semantic similarity, cross-encoder NLI,
KeyBERT keyword coverage, atomic-claim coverage with importance weighting, and
an LLM-as-judge rubric. Includes piecewise-linear calibration and a tiered
correctness gate.

## Endpoints

- `POST /api/generate_questions` — generate role-tailored questions
- `POST /api/parse_resume` — extract skills from a PDF resume
- `POST /api/evaluate_answer` — score a candidate's answer
- `POST /api/evaluate_answer_sse` — same, streamed per-signal
- `GET  /api/session_status` — recovery polling for interrupted evaluations
- `GET  /api/session_summary` — final per-session scorecard
- `GET  /health` — liveness probe

## Required secrets

Set these in **Settings → Variables and secrets** on the HF Space:

| Secret | Required | Purpose |
|--------|:---:|---------|
| `GROQ_API_KEY` | yes | LLM judge + question generation |
| `FRONTEND_URL` | yes | CORS origin (your Vercel URL, e.g. `https://prepbuddy.vercel.app`) |

## Caveats on the free tier

Hugging Face Spaces free tier has ephemeral storage. The SQLite session store
and LLM response cache reset on every container restart. This is fine for a
demo — sessions are single-user and short — but switch to managed Postgres +
Redis if you need real persistence.

## Frontend

The React frontend lives at <https://github.com/your-handle/prepbuddy/tree/main/frontend>
and is deployed to Vercel. Set its `REACT_APP_API_BASE` to this Space's URL
plus `/api`.
