"""
Full evaluation pipeline — runs all experiments from the PRD.

Computes:
1. Baseline comparisons (TF-IDF only, SBERT only, SBERT+NLI, full hybrid)
2. Pearson & Spearman correlation for each method
3. BERTScore & ROUGE on feedback quality
4. Cohen's Kappa for inter-rater agreement
5. Per-role and per-quality-level breakdown
6. Saves full report to evaluation/report.json

Usage:
    cd backend && python -m scripts.run_full_evaluation
"""

import json
import sys
import logging
import time
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "evaluation" / "dataset.json"
REPORT_PATH = Path(__file__).parent.parent / "evaluation" / "report.json"


def load_data():
    with open(DATASET_PATH) as f:
        return json.load(f)


def init_models():
    from app.config import get_settings
    from app.models_loader import ModelRegistry
    settings = get_settings()
    registry = ModelRegistry(settings)
    registry.load_all()
    return settings, registry


def compute_all_scores(data, settings, registry):
    """Pre-compute individual signal scores for every entry."""
    from app.services.scoring import sbert_scorer, nli_scorer, keyword_scorer

    logger.info(f"Computing scores for {len(data)} entries...")
    start = time.time()

    for i, entry in enumerate(data):
        candidate = entry["candidate_answer"]
        ideal = entry["ideal_answer"]

        s = sbert_scorer.score(candidate, ideal)
        n = nli_scorer.score(candidate, ideal)
        k, missing = keyword_scorer.score(candidate, ideal)

        entry["_sbert"] = s
        entry["_nli"] = n
        entry["_keyword"] = k
        entry["_missing_kw"] = missing

        if (i + 1) % 25 == 0:
            logger.info(f"  Scored {i+1}/{len(data)}")

    elapsed = time.time() - start
    logger.info(f"  Done in {elapsed:.1f}s ({elapsed/len(data):.2f}s per entry)")
    return data


def compute_tfidf_scores(data):
    """Baseline 1: TF-IDF keyword overlap only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    scores = []
    for entry in data:
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([entry["ideal_answer"], entry["candidate_answer"]])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            scores.append(sim * 100)
        except Exception:
            scores.append(0.0)
    return scores


def run_baselines(data):
    """Run all 4 methods and compute correlations."""
    human_scores = [e["human_score"] * 10 for e in data]  # Scale to 0-100

    # Baseline 1: TF-IDF only
    tfidf_scores = compute_tfidf_scores(data)

    # Baseline 2: SBERT only
    sbert_scores = [e["_sbert"] * 100 for e in data]

    # Baseline 3: SBERT + NLI
    sbert_nli_scores = [(0.6 * e["_sbert"] + 0.4 * e["_nli"]) * 100 for e in data]

    # Ours: Full hybrid (optimal weights)
    hybrid_scores = [(0.45 * e["_sbert"] + 0.05 * e["_nli"] + 0.50 * e["_keyword"]) * 100 for e in data]

    # Also compute with default weights for comparison
    default_scores = [(0.5 * e["_sbert"] + 0.3 * e["_nli"] + 0.2 * e["_keyword"]) * 100 for e in data]

    results = {}
    methods = {
        "tfidf_only": tfidf_scores,
        "sbert_only": sbert_scores,
        "sbert_nli": sbert_nli_scores,
        "hybrid_default_weights": default_scores,
        "hybrid_optimal_weights": hybrid_scores,
    }

    for name, scores in methods.items():
        pr, pp = pearsonr(scores, human_scores)
        sr, sp = spearmanr(scores, human_scores)
        results[name] = {
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 6),
            "mean_score": round(np.mean(scores), 1),
            "std_score": round(np.std(scores), 1),
        }

    return results


def run_per_quality_analysis(data):
    """Breakdown scores by quality level."""
    results = {}
    for quality in ["good", "average", "poor"]:
        entries = [e for e in data if e["quality_level"] == quality]
        human = [e["human_score"] for e in entries]
        sbert = [e["_sbert"] * 100 for e in entries]
        nli = [e["_nli"] * 100 for e in entries]
        kw = [e["_keyword"] * 100 for e in entries]
        hybrid = [(0.45 * e["_sbert"] + 0.05 * e["_nli"] + 0.50 * e["_keyword"]) * 100 for e in entries]

        results[quality] = {
            "count": len(entries),
            "avg_human_score": round(np.mean(human), 2),
            "avg_sbert": round(np.mean(sbert), 1),
            "avg_nli": round(np.mean(nli), 1),
            "avg_keyword": round(np.mean(kw), 1),
            "avg_composite": round(np.mean(hybrid), 1),
        }
    return results


def run_per_role_analysis(data):
    """Breakdown scores by role."""
    results = {}
    roles = set(e["role"] for e in data)
    for role in sorted(roles):
        entries = [e for e in data if e["role"] == role]
        human = [e["human_score"] for e in entries]
        hybrid = [(0.45 * e["_sbert"] + 0.05 * e["_nli"] + 0.50 * e["_keyword"]) * 100 for e in entries]

        pr, _ = pearsonr(hybrid, [h * 10 for h in human])
        results[role] = {
            "count": len(entries),
            "avg_human": round(np.mean(human), 2),
            "avg_composite": round(np.mean(hybrid), 1),
            "pearson_r": round(pr, 4),
        }
    return results


def compute_cohens_kappa(data):
    """Compute Cohen's Kappa between rater 1 and rater 2."""
    pairs = [(e["rater_1"], e["rater_2"]) for e in data
             if e.get("rater_1") is not None and e.get("rater_2") is not None]

    if len(pairs) < 10:
        return {"error": "Not enough dual-rated entries", "count": len(pairs)}

    r1 = [int(round(p[0])) for p in pairs]
    r2 = [int(round(p[1])) for p in pairs]

    # Bin into categories for Kappa: 0-3=Poor, 4-6=Average, 7-10=Good
    def binned(score):
        if score <= 3:
            return "poor"
        elif score <= 6:
            return "average"
        return "good"

    r1_binned = [binned(s) for s in r1]
    r2_binned = [binned(s) for s in r2]

    kappa = cohen_kappa_score(r1_binned, r2_binned)

    # Also compute raw correlation between raters
    pr, _ = pearsonr([float(s) for s in r1], [float(s) for s in r2])

    return {
        "cohens_kappa": round(kappa, 4),
        "rater_correlation": round(pr, 4),
        "num_dual_rated": len(pairs),
        "kappa_interpretation": (
            "Almost Perfect" if kappa > 0.8 else
            "Substantial" if kappa > 0.6 else
            "Moderate" if kappa > 0.4 else
            "Fair" if kappa > 0.2 else
            "Slight"
        ),
    }


def compute_feedback_metrics(data):
    """Compute BERTScore and ROUGE on a subset (ideal vs candidate as proxy)."""
    from rouge_score import rouge_scorer

    # Use ideal answers as reference, candidate answers as predictions
    # This measures how well candidates match ideal — a proxy for feedback evaluation
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougel_scores = []

    for entry in data:
        scores = scorer.score(entry["ideal_answer"], entry["candidate_answer"])
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougel_scores.append(scores["rougeL"].fmeasure)

    # Per quality level
    quality_rouge = {}
    for quality in ["good", "average", "poor"]:
        indices = [i for i, e in enumerate(data) if e["quality_level"] == quality]
        quality_rouge[quality] = {
            "rouge1": round(np.mean([rouge1_scores[i] for i in indices]), 4),
            "rougeL": round(np.mean([rougel_scores[i] for i in indices]), 4),
        }

    return {
        "rouge1_mean": round(np.mean(rouge1_scores), 4),
        "rougeL_mean": round(np.mean(rougel_scores), 4),
        "per_quality": quality_rouge,
    }


def compute_bertscore_metrics(data):
    """Compute BERTScore on a sample of 30 entries."""
    try:
        from bert_score import score as bert_score

        # Sample 30 (10 per quality)
        sample = []
        for q in ["good", "average", "poor"]:
            entries = [e for e in data if e["quality_level"] == q]
            sample.extend(entries[:10])

        refs = [e["ideal_answer"] for e in sample]
        cands = [e["candidate_answer"] for e in sample]

        logger.info(f"Computing BERTScore on {len(sample)} samples...")
        P, R, F1 = bert_score(cands, refs, model_type="distilbert-base-uncased", lang="en", verbose=False)

        per_quality = {}
        for qi, q in enumerate(["good", "average", "poor"]):
            start = qi * 10
            end = start + 10
            per_quality[q] = {
                "precision": round(P[start:end].mean().item(), 4),
                "recall": round(R[start:end].mean().item(), 4),
                "f1": round(F1[start:end].mean().item(), 4),
            }

        return {
            "mean_precision": round(P.mean().item(), 4),
            "mean_recall": round(R.mean().item(), 4),
            "mean_f1": round(F1.mean().item(), 4),
            "per_quality": per_quality,
        }
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {"error": str(e)}


def main():
    logger.info("=" * 60)
    logger.info("FULL EVALUATION PIPELINE")
    logger.info("=" * 60)

    data = load_data()
    logger.info(f"Loaded {len(data)} entries")

    # Init models
    settings, registry = init_models()

    # 1. Compute all NLP scores
    logger.info("\n[1/6] Computing NLP scores...")
    data = compute_all_scores(data, settings, registry)

    # 2. Baseline comparisons
    logger.info("\n[2/6] Running baseline comparisons...")
    baselines = run_baselines(data)

    logger.info("\n  BASELINE COMPARISON:")
    logger.info(f"  {'Method':<30} {'Pearson':>8} {'Spearman':>8}")
    logger.info(f"  {'-'*50}")
    for name, result in baselines.items():
        logger.info(f"  {name:<30} {result['pearson_r']:>8.4f} {result['spearman_r']:>8.4f}")

    # 3. Per-quality analysis
    logger.info("\n[3/6] Per-quality breakdown...")
    quality_analysis = run_per_quality_analysis(data)
    for q, stats in quality_analysis.items():
        logger.info(f"  {q.upper():>8}: human={stats['avg_human_score']:.1f} composite={stats['avg_composite']:.1f} "
                     f"sbert={stats['avg_sbert']:.1f} nli={stats['avg_nli']:.1f} kw={stats['avg_keyword']:.1f}")

    # 4. Per-role analysis
    logger.info("\n[4/6] Per-role breakdown...")
    role_analysis = run_per_role_analysis(data)
    for role, stats in role_analysis.items():
        logger.info(f"  {role:<20} Pearson={stats['pearson_r']:.4f} avg_composite={stats['avg_composite']:.1f}")

    # 5. Inter-rater agreement
    logger.info("\n[5/6] Computing inter-rater agreement...")
    kappa = compute_cohens_kappa(data)
    logger.info(f"  Cohen's Kappa: {kappa.get('cohens_kappa', 'N/A')}")
    logger.info(f"  Interpretation: {kappa.get('kappa_interpretation', 'N/A')}")

    # 6. Feedback metrics (ROUGE + BERTScore)
    logger.info("\n[6/6] Computing feedback metrics (ROUGE + BERTScore)...")
    rouge_results = compute_feedback_metrics(data)
    logger.info(f"  ROUGE-1: {rouge_results['rouge1_mean']:.4f}")
    logger.info(f"  ROUGE-L: {rouge_results['rougeL_mean']:.4f}")

    bertscore_results = compute_bertscore_metrics(data)
    if "mean_f1" in bertscore_results:
        logger.info(f"  BERTScore F1: {bertscore_results['mean_f1']:.4f}")

    # Compile full report
    report = {
        "summary": {
            "total_entries": len(data),
            "questions": len(set(e["question_index"] for e in data)),
            "roles": sorted(set(e["role"] for e in data)),
            "quality_levels": ["good", "average", "poor"],
            "optimal_weights": {"sbert": 0.45, "nli": 0.05, "keyword": 0.50},
            "best_pearson_r": baselines["hybrid_optimal_weights"]["pearson_r"],
            "best_spearman_r": baselines["hybrid_optimal_weights"]["spearman_r"],
        },
        "baseline_comparison": baselines,
        "per_quality_analysis": quality_analysis,
        "per_role_analysis": role_analysis,
        "inter_rater_agreement": kappa,
        "rouge_scores": rouge_results,
        "bertscore": bertscore_results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"FULL REPORT SAVED: {REPORT_PATH}")
    logger.info(f"{'='*60}")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n  Method                        Pearson   Spearman")
    logger.info(f"  {'─'*50}")
    for name, result in baselines.items():
        marker = " ◄ OURS" if name == "hybrid_optimal_weights" else ""
        logger.info(f"  {name:<30} {result['pearson_r']:.4f}    {result['spearman_r']:.4f}{marker}")
    logger.info(f"\n  Optimal Weights: SBERT=0.45, NLI=0.05, Keyword=0.50")
    logger.info(f"  Inter-rater Kappa: {kappa.get('cohens_kappa', 'N/A')}")
    logger.info(f"  ROUGE-L: {rouge_results['rougeL_mean']:.4f}")
    if "mean_f1" in bertscore_results:
        logger.info(f"  BERTScore F1: {bertscore_results['mean_f1']:.4f}")


if __name__ == "__main__":
    main()
