"""
Web-based rating tool for the evaluation dataset.

Opens a simple local web page where you and Surya can rate each answer 0-10.
Saves progress automatically — you can close and resume anytime.

Usage:
    cd backend && python -m scripts.rate_answers

Then open http://localhost:8501 in your browser.

Each rater should use their own URL:
    Rater 1 (Cyril):  http://localhost:8501?rater=1
    Rater 2 (Surya):  http://localhost:8501?rater=2
"""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "evaluation" / "dataset.json"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def save_dataset(data):
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.get("/", response_class=HTMLResponse)
async def rating_ui():
    return HTML_PAGE


@app.get("/api/dataset")
async def get_dataset():
    return load_dataset()


@app.post("/api/rate")
async def rate_answer(entry_id: str = Query(...), score: float = Query(..., ge=0, le=10), rater: int = Query(1)):
    data = load_dataset()
    for entry in data:
        if entry["id"] == entry_id:
            rater_key = f"rater_{rater}"
            entry[rater_key] = score

            # Auto-compute human_score as average of available ratings
            ratings = [entry.get(f"rater_{i}") for i in [1, 2, 3] if entry.get(f"rater_{i}") is not None]
            if ratings:
                entry["human_score"] = round(sum(ratings) / len(ratings), 1)

            save_dataset(data)
            return {"status": "ok", "entry_id": entry_id, "human_score": entry["human_score"]}

    return {"status": "error", "message": "Entry not found"}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer Rating Tool</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
.container { max-width: 900px; margin: 0 auto; padding: 20px; }
h1 { font-size: 24px; margin-bottom: 8px; color: #e6edf3; }
.subtitle { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
.progress-bar { background: #21262d; height: 8px; border-radius: 4px; margin-bottom: 24px; overflow: hidden; }
.progress-fill { height: 100%; background: #58a6ff; border-radius: 4px; transition: width 0.3s; }
.progress-text { font-size: 13px; color: #8b949e; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 6px; font-weight: 600; }
.meta { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.meta-tag { font-size: 12px; padding: 3px 10px; border-radius: 12px; background: #21262d; color: #8b949e; }
.meta-tag.role { background: #1f6feb22; color: #58a6ff; }
.meta-tag.quality-good { background: #23863622; color: #3fb950; }
.meta-tag.quality-average { background: #d2992222; color: #d29922; }
.meta-tag.quality-poor { background: #f8514922; color: #f85149; }
.question-text { font-family: 'JetBrains Mono', monospace; font-size: 15px; line-height: 1.7; color: #e6edf3; margin-bottom: 20px; padding: 16px; background: #0d1117; border-radius: 8px; border-left: 3px solid #58a6ff; }
.answer-section { margin-bottom: 16px; }
.answer-text { font-size: 14px; line-height: 1.6; padding: 12px 16px; background: #0d1117; border-radius: 8px; white-space: pre-wrap; }
.answer-text.ideal { border-left: 3px solid #3fb950; }
.answer-text.candidate { border-left: 3px solid #d29922; }
.rating-section { display: flex; align-items: center; gap: 16px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #21262d; }
.rating-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
.rating-btn { width: 44px; height: 44px; border-radius: 10px; border: 2px solid #30363d; background: #0d1117; color: #c9d1d9; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: 'JetBrains Mono', monospace; }
.rating-btn:hover { border-color: #58a6ff; background: #1f6feb22; }
.rating-btn.selected { border-color: #58a6ff; background: #1f6feb; color: white; }
.rating-btn.rated { border-color: #3fb950; background: #23863633; color: #3fb950; }
.nav-buttons { display: flex; gap: 12px; justify-content: space-between; margin-top: 20px; }
.nav-btn { padding: 10px 24px; border-radius: 8px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; font-size: 14px; cursor: pointer; transition: all 0.2s; }
.nav-btn:hover { background: #30363d; }
.nav-btn.primary { background: #1f6feb; border-color: #1f6feb; color: white; }
.nav-btn.primary:hover { background: #388bfd; }
.nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.filter-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.filter-btn { padding: 6px 14px; border-radius: 16px; border: 1px solid #30363d; background: transparent; color: #8b949e; font-size: 12px; cursor: pointer; transition: all 0.2s; }
.filter-btn:hover, .filter-btn.active { border-color: #58a6ff; color: #58a6ff; background: #1f6feb11; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
.stat-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }
.stat-num { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 11px; color: #8b949e; margin-top: 4px; }
.done-banner { background: #23863622; border: 1px solid #23863644; border-radius: 12px; padding: 24px; text-align: center; margin: 40px 0; }
.done-banner h2 { color: #3fb950; margin-bottom: 8px; }
</style>
</head>
<body>
<div class="container" id="app">
  <h1>Answer Rating Tool</h1>
  <p class="subtitle">Rate each candidate answer on a 0-10 scale compared to the ideal answer</p>
  <div id="content">Loading...</div>
</div>

<script>
const params = new URLSearchParams(window.location.search);
const RATER = parseInt(params.get('rater') || '1');
let dataset = [];
let currentIndex = 0;
let filter = 'unrated'; // 'all', 'unrated', 'rated'

async function loadData() {
  const res = await fetch('/api/dataset');
  dataset = await res.json();
  render();
}

function getFiltered() {
  const raterKey = 'rater_' + RATER;
  if (filter === 'unrated') return dataset.filter(d => d[raterKey] == null);
  if (filter === 'rated') return dataset.filter(d => d[raterKey] != null);
  return dataset;
}

async function rate(entryId, score) {
  await fetch('/api/rate?entry_id=' + entryId + '&score=' + score + '&rater=' + RATER, { method: 'POST' });
  // Update local
  const entry = dataset.find(d => d.id === entryId);
  if (entry) {
    entry['rater_' + RATER] = score;
    const ratings = [1,2,3].map(i => entry['rater_'+i]).filter(r => r != null);
    entry.human_score = ratings.length ? Math.round(ratings.reduce((a,b)=>a+b,0)/ratings.length * 10)/10 : null;
  }
  // Auto-advance to next unrated
  const filtered = getFiltered();
  if (filter === 'unrated' && filtered.length > 0) {
    currentIndex = Math.min(currentIndex, filtered.length - 1);
  } else {
    currentIndex = Math.min(currentIndex + 1, getFiltered().length - 1);
  }
  render();
}

function render() {
  const raterKey = 'rater_' + RATER;
  const rated = dataset.filter(d => d[raterKey] != null).length;
  const total = dataset.length;
  const filtered = getFiltered();

  if (currentIndex >= filtered.length) currentIndex = Math.max(0, filtered.length - 1);

  let html = '';

  // Stats
  html += '<div class="stats">';
  html += '<div class="stat-box"><div class="stat-num">' + rated + '</div><div class="stat-label">Rated</div></div>';
  html += '<div class="stat-box"><div class="stat-num">' + (total - rated) + '</div><div class="stat-label">Remaining</div></div>';
  html += '<div class="stat-box"><div class="stat-num">' + Math.round(rated/total*100) + '%</div><div class="stat-label">Progress</div></div>';
  html += '<div class="stat-box"><div class="stat-num">R' + RATER + '</div><div class="stat-label">Rater</div></div>';
  html += '</div>';

  // Progress bar
  html += '<div class="progress-text">' + rated + ' / ' + total + ' rated</div>';
  html += '<div class="progress-bar"><div class="progress-fill" style="width:' + (rated/total*100) + '%"></div></div>';

  // Filters
  html += '<div class="filter-bar">';
  ['unrated','rated','all'].forEach(f => {
    const count = f === 'all' ? dataset.length : f === 'unrated' ? (total-rated) : rated;
    html += '<button class="filter-btn ' + (filter===f?'active':'') + '" onclick="filter=\\''+f+'\\';currentIndex=0;render()">' + f.charAt(0).toUpperCase()+f.slice(1) + ' (' + count + ')</button>';
  });
  html += '</div>';

  if (filtered.length === 0) {
    if (filter === 'unrated') {
      html += '<div class="done-banner"><h2>All Done!</h2><p>You have rated all ' + total + ' answers. Switch to "All" or "Rated" to review.</p></div>';
    } else {
      html += '<div class="done-banner"><h2>No entries</h2><p>No entries match this filter.</p></div>';
    }
    document.getElementById('content').innerHTML = html;
    return;
  }

  const entry = filtered[currentIndex];
  const currentRating = entry[raterKey];

  html += '<div class="card">';

  // Meta
  html += '<div class="meta">';
  html += '<span class="meta-tag role">' + entry.role + '</span>';
  html += '<span class="meta-tag">' + entry.category + '</span>';
  html += '<span class="meta-tag">' + entry.difficulty + '</span>';
  html += '<span class="meta-tag quality-' + entry.quality_level + '">' + entry.quality_level.toUpperCase() + '</span>';
  html += '<span class="meta-tag">#' + (currentIndex+1) + '/' + filtered.length + '</span>';
  html += '</div>';

  // Question
  html += '<div class="label">Question</div>';
  html += '<div class="question-text">' + escapeHtml(entry.question) + '</div>';

  // Ideal answer
  html += '<div class="answer-section"><div class="label">Ideal Answer (Reference)</div>';
  html += '<div class="answer-text ideal">' + escapeHtml(entry.ideal_answer) + '</div></div>';

  // Candidate answer
  html += '<div class="answer-section"><div class="label">Candidate Answer (Rate This)</div>';
  html += '<div class="answer-text candidate">' + escapeHtml(entry.candidate_answer) + '</div></div>';

  // Rating
  html += '<div class="rating-section">';
  html += '<span class="label" style="margin:0">Score:</span>';
  html += '<div class="rating-buttons">';
  for (let i = 0; i <= 10; i++) {
    const cls = currentRating === i ? 'selected' : (currentRating != null ? 'rated' : '');
    html += '<button class="rating-btn ' + (currentRating===i?'selected':'') + '" onclick="rate(\\''+entry.id+'\\','+i+')">' + i + '</button>';
  }
  html += '</div>';
  if (currentRating != null) {
    html += '<span style="color:#3fb950;font-weight:600;font-size:14px">Rated: ' + currentRating + '/10</span>';
  }
  html += '</div>';

  html += '</div>'; // card

  // Navigation
  html += '<div class="nav-buttons">';
  html += '<button class="nav-btn" onclick="currentIndex=Math.max(0,currentIndex-1);render()" ' + (currentIndex===0?'disabled':'') + '>← Previous</button>';
  html += '<span style="color:#8b949e;align-self:center;font-size:13px">' + (currentIndex+1) + ' of ' + filtered.length + '</span>';
  html += '<button class="nav-btn primary" onclick="currentIndex=Math.min(filtered.length-1,currentIndex+1);render()" ' + (currentIndex>=filtered.length-1?'disabled':'') + '>Next →</button>';
  html += '</div>';

  document.getElementById('content').innerHTML = html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  const filtered = getFiltered();
  if (filtered.length === 0) return;
  const entry = filtered[currentIndex];

  if (e.key >= '0' && e.key <= '9' && !e.ctrlKey && !e.metaKey) {
    rate(entry.id, parseInt(e.key));
  }
  if (e.key === 'ArrowRight' || e.key === 'n') {
    currentIndex = Math.min(filtered.length - 1, currentIndex + 1);
    render();
  }
  if (e.key === 'ArrowLeft' || e.key === 'p') {
    currentIndex = Math.max(0, currentIndex - 1);
    render();
  }
});

loadData();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    logger.info(f"Dataset: {DATASET_PATH}")
    data = load_dataset()
    logger.info(f"Total entries: {len(data)}")
    logger.info("")
    logger.info("Opening rating tool at http://localhost:8501")
    logger.info("")
    logger.info("  Rater 1 (Cyril): http://localhost:8501?rater=1")
    logger.info("  Rater 2 (Surya): http://localhost:8501?rater=2")
    logger.info("")
    logger.info("Keyboard shortcuts:")
    logger.info("  0-9        → Rate current answer")
    logger.info("  → or 'n'   → Next answer")
    logger.info("  ← or 'p'   → Previous answer")
    logger.info("")
    logger.info("Progress auto-saves. Close and reopen anytime.")
    uvicorn.run(app, host="0.0.0.0", port=8501)
