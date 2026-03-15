"""
Grid search for optimal weight combination.

Sweeps weight triplets (w_sbert, w_nli, w_keyword) that sum to 1.0,
evaluating each against human-labeled scores to maximize Pearson correlation.

Usage:
    python -m evaluation.grid_search --dataset evaluation/dataset.json
"""

import json
import argparse
import logging
from pathlib import Path
from itertools import product

from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_grid_search(dataset_path: str, step: float = 0.05):
    """
    Perform grid search over weight combinations.

    Args:
        dataset_path: Path to JSON dataset with scored entries
        step: Increment step for weight values (default 0.05)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.scoring import sbert_scorer, nli_scorer, keyword_scorer

    # Need to initialize models first
    from app.config import get_settings
    from app.models_loader import ModelRegistry
    settings = get_settings()
    registry = ModelRegistry(settings)
    registry.load_all()

    with open(dataset_path) as f:
        data = json.load(f)

    # Pre-compute all individual scores
    logger.info(f"Computing individual scores for {len(data)} samples...")
    entries = []
    for item in data:
        candidate = item["candidate_answer"]
        ideal = item["ideal_answer"]
        human_score = item["human_score"]

        s = sbert_scorer.score(candidate, ideal)
        n = nli_scorer.score(candidate, ideal)
        k, _ = keyword_scorer.score(candidate, ideal)

        entries.append({
            "sbert": s,
            "nli": n,
            "keyword": k,
            "human": human_score * 10 if human_score <= 10 else human_score,
        })

    # Generate weight combinations
    values = [round(v * step, 2) for v in range(1, int(1 / step))]
    best_pearson = -1
    best_weights = (0.5, 0.3, 0.2)
    results = []

    for ws, wn in product(values, repeat=2):
        wk = round(1.0 - ws - wn, 2)
        if wk < 0.01 or wk > 0.99:
            continue

        system_scores = [
            (ws * e["sbert"] + wn * e["nli"] + wk * e["keyword"]) * 100
            for e in entries
        ]
        human_scores = [e["human"] for e in entries]

        try:
            pr, pp = pearsonr(system_scores, human_scores)
            sr, sp = spearmanr(system_scores, human_scores)
        except Exception:
            continue

        results.append({
            "w_sbert": ws,
            "w_nli": wn,
            "w_keyword": wk,
            "pearson_r": round(pr, 4),
            "spearman_r": round(sr, 4),
        })

        if pr > best_pearson:
            best_pearson = pr
            best_weights = (ws, wn, wk)

    # Sort by Pearson
    results.sort(key=lambda x: x["pearson_r"], reverse=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"GRID SEARCH RESULTS (top 10 / {len(results)} combinations)")
    logger.info(f"{'='*60}")
    for r in results[:10]:
        logger.info(
            f"  SBERT={r['w_sbert']:.2f}  NLI={r['w_nli']:.2f}  KW={r['w_keyword']:.2f}"
            f"  | Pearson={r['pearson_r']:.4f}  Spearman={r['spearman_r']:.4f}"
        )

    logger.info(f"\nBest weights: SBERT={best_weights[0]}, NLI={best_weights[1]}, Keyword={best_weights[2]}")
    logger.info(f"Best Pearson r: {best_pearson:.4f}")

    # Save results
    output_path = Path(dataset_path).parent / "grid_search_results.json"
    with open(output_path, "w") as f:
        json.dump({
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
    args = parser.parse_args()
    run_grid_search(args.dataset, args.step)
