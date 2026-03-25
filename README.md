# PrepBuddy — Interview Evaluation System

[![Project Status](https://img.shields.io/badge/status-active-success?style=flat-square)](#)
[![Backend](https://img.shields.io/badge/backend-FastAPI%2012%20%7C%20Python%203.11-00599C?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/frontend-React%2018-61DAFB?style=flat-square&logo=react&logoColor=white)](https://react.dev/)
[![Scoring](https://img.shields.io/badge/scoring-hybrid%20NLP%20LLM-8A2BE2?style=flat-square&logo=huggingface&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-informational?style=flat-square)](LICENSE)

**PrepBuddy** is a full-stack interview preparation platform combining local NLP models with cloud LLMs (Groq Llama 3.3-70B) to generate role-specific questions and evaluate answers using multi-signal hybrid scoring pipelines. Produces rubric-backed feedback, per-question grades, and session summaries.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Role-aware Q&A** | Generate questions by role, experience level, category, difficulty |
| **Real-time scoring** | SSE streaming evaluation with 4-signal hybrid pipeline |
| **Dual pipelines** | Technical (claim-based) + Behavioral (STAR rubric) |
| **Rich feedback** | Per-question scores, missing keywords, claim coverage, LLM feedback |
| **Session analytics** | Overall grade, strongest/weakest areas, rubric averages |

---

## 🏗️ Architecture

```mermaid
graph TD
    A[React Frontend<br/>Chat UI + SSE] -->|HTTP/SSE| B[FastAPI Backend<br/>Python 3.11]
    B --> C[Groq LLM<br/>Llama 3.3-70B]
    B --> D[Local Models<br/>SBERT + DeBERTa + KeyBERT]
    B --> E[Scoring Pipeline<br/>Technical vs Behavioral]
    E --> F[Session Store<br/>In-memory]
```

**Scoring Pipeline Decision Tree:**
```
is_behavioral(category)? 
├── YES → STAR Behavioral Pipeline (60% LLM)
└── NO  → Claim-Based Technical Pipeline (50% Claims)
```

---

## 📁 Project Structure

```
PrepBuddy/
├── backend/                    # FastAPI + NLP scoring
│   ├── app/
│   │   ├── main.py            # FastAPI app entrypoint
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Business logic
│   │   └── scoring/           # Multi-signal pipelines
│   ├── evaluation/            # Research + grid search
│   └── tests/                 # pytest suite
├── frontend/                  # React 18 chat UI
│   ├── src/
│   │   ├── hooks/useChat.js   # Session state machine
│   │   ├── components/        # ChatWindow, ScoreCard, etc.
│   │   └── utils/api.js       # SSE + API wrapper
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | FastAPI, Pydantic v2, PyTorch, Sentence Transformers, spaCy, Groq SDK |
| **Frontend** | React 18, Custom Hooks, SSE (EventSource), Lucide React Icons |
| **Scoring** | SBERT (`all-MiniLM-L6-v2`), DeBERTa NLI, KeyBERT, Llama 3.3-70B/3.1-8B |
| **Infra** | uvicorn ASGI, Thread-safe ModelRegistry, Auto device detection (MPS/CUDA/CPU) |

**Lucide React Icons**: Used in `CubeIcon.js` and planned for ScoreCard status indicators. Install via `npm i lucide-react`.

---

## ⚙️ Quick Setup

1. **Clone & Install Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # Add GROQ_API_KEY
```

2. **Install Frontend**
```bash
cd frontend
npm install
```

3. **Run Both**
```bash
# Backend (with hot reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
npm start                    # http://localhost:3000
```

**API Docs**: http://localhost:8000/docs

---

## 📊 Scoring Pipelines

**Technical Pipeline** (composite_v2) - 0-100 scale
| Signal | Weight | Measures |
|--------|--------|----------|
| SBERT | 25% | Semantic similarity |
| NLI | 10% | Entailment/contradiction |
| Keywords | 5% | KeyBERT term coverage |
| **Claims** | **50%** | Ideal answer claim coverage |
| LLM Judge | 10% | Correctness, completeness, clarity, depth |

**Behavioral Pipeline** (STAR)
| Signal | Weight | Measures |
|--------|--------|----------|
| SBERT | 25% | Semantic alignment |
| NLI | 5% | Logical consistency |
| Keywords | 10% | Domain terms |
| **STAR LLM** | **60%** | Situation/Task/Action/Result/Reflection |

---

## 🔬 Evaluation Results

**Grid Search Optimization** (vs human-labeled dataset):
```
Pearson r = 0.8864 (Excellent correlation)
Optimal weights: SBERT=0.40, NLI=0.10, Keyword=0.30, LLM=0.20
```

**Metrics Used**:
- BERTScore (DeBERTa-XL-MNLI)
- ROUGE-1/L
- Pearson/Spearman correlation

---

## 📖 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate_questions` | POST | Create session with N diverse questions |
| `/api/evaluate_answer_sse` | POST | **Streaming** answer evaluation |
| `/api/evaluate_answer` | POST | Blocking answer evaluation |
| `/api/session/{id}` | GET | Session summary + analytics |

**Example Request**:
```json
{
  "role": "Software Engineer",
  "level": "Junior", 
  "category": "Data Structures",
  "difficulty": "Medium",
  "num_questions": 5
}
```

---

## 🧪 Testing

```bash
cd backend
pytest tests/ -v              # Backend API + scoring
cd ../frontend
npm test                      # Frontend components
```

**Coverage**: Schema validation, claim pipeline, SSE streaming, session management.

---

## ⚖️ Grading Scale

| Score | Grade |
|-------|-------|
| 80-100 | Excellent |
| 60-79 | Good |
| 40-59 | Needs Improvement |
| 0-39 | Significant Gaps |

---

## 🔧 Configuration

**Required**: `GROQ_API_KEY` in `backend/.env`

**Optional Tuning**:
```
CLAIM_EXTRACTION_MODE=regex        # or 'llm'
SBERT_WEIGHT=0.40
CLAIM_MATCH_THRESHOLD=0.62
DEVICE=mps                         # cpu/cuda/mps (auto-detect)
```

---

## 📈 Future Work

- [ ] Persistent session storage (PostgreSQL/Redis)
- [ ] Multi-language support
- [ ] Custom rubrics per company/role
- [ ] Voice input + transcription
- [ ] Leaderboards + sharing

---

## 📄 License

MIT License - see [LICENSE](LICENSE) © 2026 Chaitanya Pudota


