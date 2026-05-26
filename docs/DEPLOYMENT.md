# Deployment Guide

Free-tier deployment of PrepBuddy:
- **Backend** → Hugging Face Spaces (Docker SDK), 16 GB RAM, free
- **Frontend** → Vercel, free
- **LLM** → Groq API (free dev tier)

End state: one public URL for the chat UI on Vercel, calling the FastAPI
service hosted on a Hugging Face Space.

---

## Prerequisites

- A GitHub account with the PrepBuddy repo pushed.
- A free Hugging Face account: <https://huggingface.co/join>
- A free Vercel account: <https://vercel.com/signup>
- A Groq API key: <https://console.groq.com/keys>

---

## Part 1 — Backend on Hugging Face Spaces

### 1. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Name**: `prepbuddy` (or whatever you like).
3. **License**: `mit`.
4. **Space SDK**: select **Docker** → **Blank**.
5. **Hardware**: keep **CPU basic (free)** — that's the 16 GB RAM tier.
6. Set visibility to **Public** so Vercel can reach it.
7. Click **Create Space**.

### 2. Push the `backend/` folder as the Space repo

The Space ships with its own git repo. Push your `backend/` directory into it.

```bash
# Replace <your-username> with your HF username.
HF_SPACE=https://huggingface.co/spaces/<your-username>/prepbuddy

cd /path/to/Prepbuddy/backend

# First time only — create a fresh git repo limited to the backend folder.
git init
git remote add space "$HF_SPACE"

git add Dockerfile README.md .dockerignore app requirements.txt
# (Add other folders you want shipped — e.g. evaluation/ — but skip data/.)

git commit -m "deploy: initial backend Space"
git branch -M main
git push space main
```

Hugging Face will detect the Dockerfile and start building. The first build
takes 8–12 minutes because it downloads PyTorch CPU wheels, spaCy, SBERT,
and the DeBERTa NLI weights. Subsequent pushes that don't change
`requirements.txt` are much faster thanks to Docker layer caching.

### 3. Set the secrets

In the Space UI → **Settings → Variables and secrets**:

| Key | Type | Value |
|-----|------|-------|
| `GROQ_API_KEY` | Secret | Your Groq key from <https://console.groq.com/keys> |
| `FRONTEND_URL` | Variable | Your Vercel URL — set this after Part 2 (e.g. `https://prepbuddy.vercel.app`) |

After saving, restart the Space (top right → **Restart Space**) so the new
env vars take effect.

### 4. Verify the backend

Open `https://<your-username>-prepbuddy.hf.space/health` in a browser. You
should see:

```json
{"status": "ok"}
```

The first request after a restart can take 30–60 s while models load even
with the pre-warm in the Dockerfile — the model cache is read off disk and
mapped into RAM. Subsequent requests are fast.

Also confirm the OpenAPI docs render: `https://<your-username>-prepbuddy.hf.space/docs`.

---

## Part 2 — Frontend on Vercel

### 1. Create the project

1. Push the full Prepbuddy repo to GitHub if you haven't already.
2. Go to <https://vercel.com/new> and import the GitHub repo.
3. **Root directory**: `frontend` — this is critical, otherwise Vercel will
   try to build the whole monorepo.
4. **Framework preset**: Create React App (auto-detected via `vercel.json`).
5. Don't deploy yet — set the env var first.

### 2. Set the API base URL

Project → **Settings → Environment Variables** → Add:

| Key | Environment | Value |
|-----|-------------|-------|
| `REACT_APP_API_BASE` | Production, Preview, Development | `https://<your-username>-prepbuddy.hf.space/api` |

Note the `/api` suffix — the FastAPI routers are mounted there.

### 3. Deploy

Hit **Deploy**. Build takes ~1 minute. Note the URL it gives you
(e.g. `https://prepbuddy.vercel.app`).

### 4. Tell the backend about this URL

Back on the Hugging Face Space → **Settings → Variables and secrets** →
update `FRONTEND_URL` to the Vercel URL (no trailing slash). Restart the
Space. This adds the Vercel origin to the FastAPI CORS allow-list.

---

## Smoke test

1. Open the Vercel URL.
2. Start a new session — pick role, level, category, difficulty,
   `num_questions=2`.
3. Wait ~10–20 s for question generation (Groq call).
4. Type an answer and submit. You should see the streaming evaluation events
   in the browser DevTools network tab against `/api/evaluate_answer_sse`.

If anything 4xx/5xx, check:
- HF Space logs (top right → **Logs**) for the FastAPI traceback.
- Browser console for CORS errors → `FRONTEND_URL` mismatch.
- Vercel build logs if the frontend won't load.

---

## Caveats and known limits

- **Ephemeral storage.** The HF Space free tier wipes the container on
  restart, so `data/sessions.db` and `data/llm_cache.db` reset. Sessions are
  short-lived anyway, but a user who closes their browser and comes back
  after a restart will lose their history.
- **Sleep after inactivity.** HF Space free instances sleep after some
  idle minutes and take ~30 s to wake. The first request after sleep will
  feel slow.
- **Groq rate limits.** The free Groq tier limits requests per minute. If
  you demo to a class, generate the questions ahead of time so the LLM
  judge calls are the only ones competing for tokens.
- **Single worker.** The Dockerfile runs one uvicorn worker on purpose —
  the models sit in process memory and forking would double the RAM
  footprint. Vertical scale up if you ever need more throughput.

---

## What to migrate to when this stops being enough

| Free tier limit | Paid path |
|-----------------|-----------|
| Ephemeral disk | HF Space Pro ($9/mo) for persistent storage, **or** swap SQLite for Neon Postgres + Upstash Redis (both free tiers, persistent) |
| Sleep on idle | HF Space Pro keeps the Space always on |
| 16 GB RAM | HF Space hardware upgrade (paid) or move backend to Fly.io Performance VM ($15/mo) |
| Single CPU | GPU upgrade on HF (~$1.05/hr for an A10G) — only needed if you swap MiniLM for a bigger embedding model |

The architecture doesn't change. Only the host changes.
