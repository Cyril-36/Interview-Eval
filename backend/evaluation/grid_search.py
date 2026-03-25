"""
Grid search for optimal weight combination (4-signal hybrid).

Sweeps weight quadruplets (w_sbert, w_nli, w_keyword, w_llm) that sum to 1.0,
evaluating each against human-labeled scores to maximize Pearson correlation.

Supports two modes:
  - 3-signal (NLP-only): SBERT + NLI + Keyword
  - 4-signal (hybrid):   SBERT + NLI + Keyword + LLM-as-judge

Usage:
    python -m evaluation.grid_search --dataset evaluation/dataset.json
    python -m evaluation.grid_search --dataset evaluation/dataset.json --signals 4
"""

import json
import asyncio
import argparse
import logging
import time
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_grid_search_3signal(entries, step: float = 0.05):
    """
    Grid search over 3 NLP weights (SBERT, NLI, Keyword).
    Fast — no API calls needed.
    """
    values = [round(v * step, 2) for v in range(1, int(1 / step))]
    best_pearson = -1.0
    best_weights = (0.5, 0.3, 0.2)
    results = []

    for ws, wn in [(ws, wn) for ws in values for wn in values]:
        wk = round(1.0 - ws - wn, 2)
        if wk < 0.01 or wk > 0.99:
            continue

        system_scores = [
            (ws * e["sbert"] + wn * e["nli"] + wk * e["keyword"]) * 100
            for e in entries
        ]
        human_scores = [e["human"] for e in entries]

        try:
            pr, _ = pearsonr(system_scores, human_scores)
            sr, _ = spearmanr(system_scores, human_scores)
        except Exception:
            continue

        results.append({
            "w_sbert": ws, "w_nli": wn, "w_keyword": wk,
            "pearson_r": round(pr, 4), "spearman_r": round(sr, 4),
        })

        if pr > best_pearson:
            best_pearson = pr
            best_weights = (ws, wn, wk)

    results.sort(key=lambda x: x["pearson_r"], reverse=True)
    return best_weights, best_pearson, results


def run_grid_search_4signal(entries, step: float = 0.10):
    """
    Grid search over 4 weights (SBERT, NLI, Keyword, LLM).
    Uses step=0.10 by default (reduces combos from ~10K to ~286).
    """
    values = [round(v * step, 2) for v in range(0, int(1 / step) + 1)]
    best_pearson = -1.0
    best_weights = (0.25, 0.15, 0.20, 0.40)
    results = []

    for ws in values:
        for wn in values:
            for wk in values:
                wl = round(1.0 - ws - wn - wk, 2)
                if wl < 0.0 or wl > 1.0:
                    continue
                # Skip if any weight is exactly 0 (unless
                # it's a valid ablation)
                if (ws + wn + wk + wl) < 0.99:
                    continue

                system_scores = [
                    (ws * e["sbert"] + wn * e["nli"]
                     + wk * e["keyword"] + wl * e["llm"]) * 100
                    for e in entries
                ]
                human_scores = [e["human"] for e in entries]

                try:
                    pr, _ = pearsonr(system_scores, human_scores)
                    sr, _ = spearmanr(system_scores, human_scores)
                except Exception:
                    continue

                results.append({
                    "w_sbert": ws, "w_nli": wn,
                    "w_keyword": wk, "w_llm": wl,
                    "pearson_r": round(pr, 4),
                    "spearman_r": round(sr, 4),
                })

                if pr > best_pearson:
                    best_pearson = pr
                    best_weights = (ws, wn, wk, wl)

    results.sort(key=lambda x: x["pearson_r"], reverse=True)
    return best_weights, best_pearson, results


async def compute_llm_scores(data):
    """Call LLM scorer for each entry in dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.scoring import llm_scorer

    scores = []
    total = len(data)
    for i, item in enumerate(data):
        llm_result = await llm_scorer.score(
            item["candidate_answer"],
            item["ideal_answer"],
            item.get("question", ""),
        )
        scores.append(llm_result.normalized_score)
        if (i + 1) % 10 == 0:
            logger.info(f"  LLM scored {i+1}/{total}")
    return scores


def run_grid_search(
    dataset_path: str,
    step: float = 0.05,
    signals: int = 3,
):
    """
    Full grid search pipeline.

    Args:
        dataset_path: Path to JSON dataset with scored entries
        step: Increment step for weight values
        signals: 3 for NLP-only, 4 for NLP+LLM hybrid
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.scoring import sbert_scorer, nli_scorer, keyword_scorer

    # Initialize models
    from app.config import get_settings
    from app.models_loader import ModelRegistry
    settings = get_settings()
    registry = ModelRegistry(settings)
    registry.load_all()

    with open(dataset_path) as f:
        data = json.load(f)

    # Pre-compute NLP scores
    logger.info(f"Computing NLP scores for {len(data)} samples...")
    start = time.time()
    entries = []
    for i, item in enumerate(data):
        candidate = item["candidate_answer"]
        ideal = item["ideal_answer"]
        human_score = item["human_score"]

        s = sbert_scorer.score(candidate, ideal)
        n = nli_scorer.score(candidate, ideal)
        k, _ = keyword_scorer.score(candidate, ideal)

        entry = {
            "sbert": s,
            "nli": n,
            "keyword": k,
            "human": human_score * 10 if human_score <= 10 else human_score,
        }
        entries.append(entry)

        if (i + 1) % 25 == 0:
            logger.info(f"  NLP scored {i+1}/{len(data)}")

    nlp_time = time.time() - start
    logger.info(f"  NLP scores done in {nlp_time:.1f}s")

    # Compute LLM scores if 4-signal mode
    if signals == 4:
        logger.info("Computing LLM-as-judge scores (Groq API)...")
        llm_start = time.time()
        llm_scores = asyncio.run(compute_llm_scores(data))
        for entry, llm_s in zip(entries, llm_scores):
            entry["llm"] = llm_s
        llm_time = time.time() - llm_start
        logger.info(f"  LLM scores done in {llm_time:.1f}s")

    # Run grid search
    if signals == 4:
        search_step = max(step, 0.10)  # 0.05 would give ~10K combos
        logger.info(
            f"\nRunning 4-signal grid search (step={search_step})..."
        )
        best_weights, best_pearson, results = run_grid_search_4signal(
            entries, search_step,
        )

        logger.info("\n" + "=" * 70)
        logger.info(
            f"4-SIGNAL GRID SEARCH RESULTS"
            f" (top 10 / {len(results)} combinations)"
        )
        logger.info("=" * 70)
        for r in results[:10]:
            logger.info(
                f"  SBERT={r['w_sbert']:.2f}  NLI={r['w_nli']:.2f}"
                f"  KW={r['w_keyword']:.2f}  LLM={r['w_llm']:.2f}"
                f"  | Pearson={r['pearson_r']:.4f}"
                f"  Spearman={r['spearman_r']:.4f}"
            )

        logger.info(
            f"\nBest 4-signal weights:"
            f" SBERT={best_weights[0]}, NLI={best_weights[1]},"
            f" Keyword={best_weights[2]}, LLM={best_weights[3]}"
        )
        logger.info(f"Best Pearson r: {best_pearson:.4f}")

        output_path = Path(dataset_path).parent / "grid_search_4signal.json"
        with open(output_path, "w") as f:
            json.dump({
                "mode": "4-signal (NLP + LLM-as-judge)",
                "best_weights": {
                    "sbert": best_weights[0],
                    "nli": best_weights[1],
                    "keyword": best_weights[2],
                    "llm": best_weights[3],
                },
                "best_pearson": round(best_pearson, 4),
                "all_results": results[:50],
            }, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    else:
        logger.info(f"\nRunning 3-signal grid search (step={step})...")
        best_weights, best_pearson, results = run_grid_search_3signal(
            entries, step,
        )

        logger.info("\n" + "=" * 60)
        logger.info(
            f"3-SIGNAL GRID SEARCH RESULTS"
            f" (top 10 / {len(results)} combinations)"
        )
        logger.info("=" * 60)
        for r in results[:10]:
            logger.info(
                f"  SBERT={r['w_sbert']:.2f}  NLI={r['w_nli']:.2f}"
                f"  KW={r['w_keyword']:.2f}"
                f"  | Pearson={r['pearson_r']:.4f}"
                f"  Spearman={r['spearman_r']:.4f}"
            )

        logger.info(
            f"\nBest 3-signal weights:"
            f" SBERT={best_weights[0]}, NLI={best_weights[1]},"
            f" Keyword={best_weights[2]}"
        )
        logger.info(f"Best Pearson r: {best_pearson:.4f}")

        output_path = Path(dataset_path).parent / "grid_search_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "mode": "3-signal (NLP-only)",
                "best_weights": {
                    "sbert": best_weights[0],
                    "nli": best_weights[1],
                    "keyword": best_weights[2],
                },
                "best_pearson": round(best_pearson, 4),
                "all_results": results[:50],
            }, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return best_weights, best_pearson


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/dataset.json")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--signals", type=int, default=3, choices=[3, 4],
        help="3 = NLP-only (SBERT+NLI+KW), 4 = hybrid (+ LLM judge)",
    )
    args = parser.parse_args()
    run_grid_search(args.dataset, args.step, args.signals)
