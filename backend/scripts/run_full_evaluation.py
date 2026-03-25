"""
Full evaluation pipeline — runs all experiments from the PRD.

Computes:
1. Baseline comparisons (TF-IDF, SBERT, SBERT+NLI, 3-sig hybrid, 4-sig hybrid)
2. Pearson & Spearman correlation for each method
3. BERTScore & ROUGE on feedback quality
4. Cohen's Kappa for inter-rater agreement
5. Per-role, per-quality, per-difficulty, and answer-length breakdowns
6. Saves full report to evaluation/report.json

Usage:
    cd backend && python -m scripts.run_full_evaluation
    cd backend && python -m scripts.run_full_evaluation --with-llm
"""

import json
import sys
import asyncio
import logging
import time
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "evaluation" / "dataset.json"
REPORT_PATH = Path(__file__).parent.parent / "evaluation" / "report.json"
LLM_EVAL_TARGET_RPM = 20
LLM_EVAL_REQUEST_INTERVAL_SECONDS = 60.0 / LLM_EVAL_TARGET_RPM


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
    """Pre-compute individual NLP signal scores for every entry."""
    from app.services.scoring import (
        claim_scorer,
        keyword_scorer,
        nli_scorer,
        sbert_scorer,
    )

    logger.info(f"Computing NLP scores for {len(data)} entries...")
    start = time.time()

    for i, entry in enumerate(data):
        candidate = entry["candidate_answer"]
        ideal = entry["ideal_answer"]

        s = sbert_scorer.score(candidate, ideal)
        n = nli_scorer.score(candidate, ideal)
        k, missing = keyword_scorer.score(candidate, ideal)
        claim = claim_scorer.score(candidate, ideal, question=entry.get("question", ""))

        entry["_sbert"] = s
        entry["_nli"] = n
        entry["_keyword"] = k
        entry["_missing_kw"] = missing
        entry["_claim"] = claim.coverage
        entry["_claim_match_quality"] = claim.normalized_score
        entry["_claim_hard_coverage"] = claim.hard_coverage
        entry["_claim_contradiction"] = claim.avg_contradiction
        entry["_missing_claims"] = claim.missing_claims

        if (i + 1) % 25 == 0:
            logger.info(f"  Scored {i+1}/{len(data)}")

    elapsed = time.time() - start
    logger.info(
        f"  Done in {elapsed:.1f}s"
        f" ({elapsed/len(data):.2f}s/entry)"
    )
    return data


async def compute_llm_scores(data):
    """Compute LLM-as-judge scores for every entry via Groq API."""
    from app.services.scoring import llm_scorer

    logger.info(f"Computing LLM-as-judge scores for {len(data)} entries...")
    logger.info(
        "Applying LLM evaluation throttle at %.1f RPM (%.2fs between requests)",
        LLM_EVAL_TARGET_RPM,
        LLM_EVAL_REQUEST_INTERVAL_SECONDS,
    )
    start = time.time()
    next_request_at = time.monotonic()

    for i, entry in enumerate(data):
        wait_seconds = next_request_at - time.monotonic()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        request_started_at = time.monotonic()
        llm_result = await llm_scorer.score(
            entry["candidate_answer"],
            entry["ideal_answer"],
            entry.get("question", ""),
        )
        entry["_llm"] = llm_result.normalized_score
        entry["_llm_reason"] = llm_result.reason
        entry["_llm_correctness"] = llm_result.correctness
        entry["_llm_completeness"] = llm_result.completeness
        entry["_llm_clarity"] = llm_result.clarity
        entry["_llm_depth"] = llm_result.depth
        next_request_at = request_started_at + LLM_EVAL_REQUEST_INTERVAL_SECONDS

        if (i + 1) % 10 == 0:
            logger.info(f"  LLM scored {i+1}/{len(data)}")

    elapsed = time.time() - start
    logger.info(f"  LLM scoring done in {elapsed:.1f}s")
    return data


def compute_tfidf_scores(data):
    """Baseline: TF-IDF keyword overlap only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    scores = []
    for entry in data:
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(
                [entry["ideal_answer"], entry["candidate_answer"]]
            )
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            scores.append(sim * 100)
        except Exception:
            scores.append(0.0)
    return scores


def _composite_3sig(entry):
    return (
        0.45 * entry["_sbert"]
        + 0.05 * entry["_nli"]
        + 0.50 * entry["_keyword"]
    ) * 100


def _composite_4sig(entry):
    return (
        0.40 * entry["_sbert"]
        + 0.10 * entry["_nli"]
        + 0.30 * entry["_keyword"]
        + 0.20 * entry["_llm"]
    ) * 100


def _composite_claim_nlp(entry):
    return (
        0.20 * entry["_sbert"]
        + 0.10 * entry["_nli"]
        + 0.20 * entry["_keyword"]
        + 0.50 * entry["_claim"]
    ) * 100


def _composite_claim_4sig(entry):
    return (
        0.15 * entry["_sbert"]
        + 0.05 * entry["_nli"]
        + 0.10 * entry["_keyword"]
        + 0.50 * entry["_claim"]
        + 0.20 * entry["_llm"]
    ) * 100


def _weight_grid(signal_count: int, step: float = 0.05):
    units = int(round(1.0 / step))

    def _recurse(remaining: int, dims_left: int):
        if dims_left == 1:
            yield (remaining,)
            return
        for value in range(remaining + 1):
            for tail in _recurse(remaining - value, dims_left - 1):
                yield (value,) + tail

    for weights in _recurse(units, signal_count):
        yield tuple(round(weight * step, 4) for weight in weights)


def _weighted_scores(entries, signal_keys, weights):
    return [
        sum(weight * entry[key] for key, weight in zip(signal_keys, weights)) * 100
        for entry in entries
    ]


def run_weight_search(entries, signal_keys, step: float = 0.05):
    human_scores = [entry["human_score"] * 10 for entry in entries]
    best = {
        "weights": tuple(round(1.0 / len(signal_keys), 4) for _ in signal_keys),
        "pearson_r": -1.0,
        "spearman_r": -1.0,
    }
    top_results = []

    for weights in _weight_grid(len(signal_keys), step):
        scores = _weighted_scores(entries, signal_keys, weights)
        try:
            pr, _ = pearsonr(scores, human_scores)
            sr, _ = spearmanr(scores, human_scores)
        except Exception:
            continue

        result = {
            "weights": {
                key.removeprefix("_"): weight
                for key, weight in zip(signal_keys, weights)
            },
            "pearson_r": round(pr, 4),
            "spearman_r": round(sr, 4),
        }
        top_results.append(result)

        if pr > best["pearson_r"]:
            best = {
                "weights": weights,
                "pearson_r": round(pr, 4),
                "spearman_r": round(sr, 4),
            }

    top_results.sort(key=lambda item: item["pearson_r"], reverse=True)
    return {
        "signal_keys": [key.removeprefix("_") for key in signal_keys],
        "best_weights": {
            key.removeprefix("_"): weight
            for key, weight in zip(signal_keys, best["weights"])
        },
        "best_pearson_r": best["pearson_r"],
        "best_spearman_r": best["spearman_r"],
        "top_results": top_results[:10],
    }


def run_cross_validation(
    entries,
    signal_keys,
    step: float = 0.05,
    n_splits: int = 5,
    seed: int = 42,
):
    if len(entries) < n_splits:
        return {
            "n_splits": n_splits,
            "error": "Not enough entries for cross-validation",
        }

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    entries_array = np.array(entries, dtype=object)
    fold_results = []

    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(entries_array), 1):
        train_entries = entries_array[train_idx].tolist()
        test_entries = entries_array[test_idx].tolist()
        search = run_weight_search(train_entries, signal_keys, step=step)
        weights = tuple(search["best_weights"][key.removeprefix("_")] for key in signal_keys)
        test_scores = _weighted_scores(test_entries, signal_keys, weights)
        human_scores = [entry["human_score"] * 10 for entry in test_entries]
        pr, _ = pearsonr(test_scores, human_scores)
        sr, _ = spearmanr(test_scores, human_scores)
        fold_results.append(
            {
                "fold": fold_index,
                "weights": search["best_weights"],
                "pearson_r": round(pr, 4),
                "spearman_r": round(sr, 4),
            }
        )

    return {
        "n_splits": n_splits,
        "mean_pearson_r": round(float(np.mean([f["pearson_r"] for f in fold_results])), 4),
        "std_pearson_r": round(float(np.std([f["pearson_r"] for f in fold_results])), 4),
        "mean_spearman_r": round(float(np.mean([f["spearman_r"] for f in fold_results])), 4),
        "std_spearman_r": round(float(np.std([f["spearman_r"] for f in fold_results])), 4),
        "folds": fold_results,
    }


def _normalized_difficulty(entry):
    return str(entry.get("difficulty", "Unknown")).strip().title()


def _answer_word_count(entry):
    return len(entry.get("candidate_answer", "").split())


def _safe_pearson(xs, ys):
    if len(xs) < 2:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        pr, _ = pearsonr(xs, ys)
    except Exception:
        return None
    return round(pr, 4)


def run_baselines(
    data,
    include_llm=False,
    claim_nlp_weights=None,
    claim_4sig_weights=None,
):
    """Run all methods and compute correlations."""
    human_scores = [e["human_score"] * 10 for e in data]

    tfidf_scores = compute_tfidf_scores(data)
    sbert_scores = [e["_sbert"] * 100 for e in data]
    sbert_nli_scores = [
        (0.6 * e["_sbert"] + 0.4 * e["_nli"]) * 100 for e in data
    ]
    hybrid_3sig = [_composite_3sig(e) for e in data]
    claim_only = [e["_claim"] * 100 for e in data]
    claim_hybrid_nlp = (
        _weighted_scores(data, ["_sbert", "_nli", "_keyword", "_claim"], claim_nlp_weights)
        if claim_nlp_weights else
        [_composite_claim_nlp(e) for e in data]
    )
    hybrid_default = [
        (0.5 * e["_sbert"] + 0.3 * e["_nli"]
         + 0.2 * e["_keyword"]) * 100
        for e in data
    ]

    methods = {
        "tfidf_only": tfidf_scores,
        "sbert_only": sbert_scores,
        "sbert_nli": sbert_nli_scores,
        "hybrid_3sig_default": hybrid_default,
        "hybrid_3sig_optimal": hybrid_3sig,
        "claim_only": claim_only,
        "claim_hybrid_nlp": claim_hybrid_nlp,
    }

    if include_llm and "_llm" in data[0]:
        llm_only = [e["_llm"] * 100 for e in data]
        methods["llm_judge_only"] = llm_only

        # 4-signal optimal (grid search: Pearson=0.8864)
        hybrid_4sig_opt = [_composite_4sig(e) for e in data]
        methods["hybrid_4sig_optimal"] = hybrid_4sig_opt
        methods["claim_hybrid_4sig"] = (
            _weighted_scores(
                data,
                ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
                claim_4sig_weights,
            )
            if claim_4sig_weights else
            [_composite_claim_4sig(e) for e in data]
        )

        hybrid_4sig_v2 = [
            (0.20 * e["_sbert"] + 0.05 * e["_nli"]
             + 0.25 * e["_keyword"] + 0.50 * e["_llm"]) * 100
            for e in data
        ]
        methods["hybrid_4sig_llm_heavy"] = hybrid_4sig_v2

        hybrid_4sig_v3 = [
            (0.30 * e["_sbert"] + 0.05 * e["_nli"]
             + 0.30 * e["_keyword"] + 0.35 * e["_llm"]) * 100
            for e in data
        ]
        methods["hybrid_4sig_balanced"] = hybrid_4sig_v3

    results = {}
    for name, scores in methods.items():
        pr, pp = pearsonr(scores, human_scores)
        sr, sp = spearmanr(scores, human_scores)
        results[name] = {
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 6),
            "mean_score": round(float(np.mean(scores)), 1),
            "std_score": round(float(np.std(scores)), 1),
        }

    return results


def run_per_quality_analysis(
    data,
    include_llm=False,
    claim_nlp_weights=None,
    claim_4sig_weights=None,
):
    """Breakdown scores by quality level."""
    results = {}
    for quality in ["good", "average", "poor"]:
        entries = [e for e in data if e["quality_level"] == quality]
        human = [e["human_score"] for e in entries]
        sbert = [e["_sbert"] * 100 for e in entries]
        nli = [e["_nli"] * 100 for e in entries]
        kw = [e["_keyword"] * 100 for e in entries]
        claim = [e["_claim"] * 100 for e in entries]
        hybrid = [_composite_3sig(e) for e in entries]
        claim_hybrid = (
            _weighted_scores(entries, ["_sbert", "_nli", "_keyword", "_claim"], claim_nlp_weights)
            if claim_nlp_weights else
            [_composite_claim_nlp(e) for e in entries]
        )

        result = {
            "count": len(entries),
            "avg_human_score": round(float(np.mean(human)), 2),
            "avg_sbert": round(float(np.mean(sbert)), 1),
            "avg_nli": round(float(np.mean(nli)), 1),
            "avg_keyword": round(float(np.mean(kw)), 1),
            "avg_claim": round(float(np.mean(claim)), 1),
            "avg_composite_3sig": round(float(np.mean(hybrid)), 1),
            "avg_composite_claim_nlp": round(float(np.mean(claim_hybrid)), 1),
        }

        if include_llm and "_llm" in entries[0]:
            llm = [e["_llm"] * 100 for e in entries]
            hybrid_4 = [_composite_4sig(e) for e in entries]
            claim_hybrid_4 = (
                _weighted_scores(
                    entries,
                    ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
                    claim_4sig_weights,
                )
                if claim_4sig_weights else
                [_composite_claim_4sig(e) for e in entries]
            )
            result["avg_llm"] = round(float(np.mean(llm)), 1)
            result["avg_composite_4sig"] = round(
                float(np.mean(hybrid_4)), 1,
            )
            result["avg_composite_claim_4sig"] = round(
                float(np.mean(claim_hybrid_4)), 1,
            )
            result["avg_llm_correctness"] = round(
                float(np.mean([e["_llm_correctness"] * 100 for e in entries])),
                1,
            )
            result["avg_llm_completeness"] = round(
                float(np.mean([e["_llm_completeness"] * 100 for e in entries])),
                1,
            )
            result["avg_llm_clarity"] = round(
                float(np.mean([e["_llm_clarity"] * 100 for e in entries])),
                1,
            )
            result["avg_llm_depth"] = round(
                float(np.mean([e["_llm_depth"] * 100 for e in entries])),
                1,
            )

        results[quality] = result
    return results


def run_per_role_analysis(
    data,
    include_llm=False,
    claim_nlp_weights=None,
    claim_4sig_weights=None,
):
    """Breakdown scores by role."""
    results = {}
    roles = set(e["role"] for e in data)
    for role in sorted(roles):
        entries = [e for e in data if e["role"] == role]
        human = [e["human_score"] for e in entries]
        hybrid = [_composite_3sig(e) for e in entries]
        claim_hybrid = (
            _weighted_scores(entries, ["_sbert", "_nli", "_keyword", "_claim"], claim_nlp_weights)
            if claim_nlp_weights else
            [_composite_claim_nlp(e) for e in entries]
        )

        pr, _ = pearsonr(hybrid, [h * 10 for h in human])
        result = {
            "count": len(entries),
            "avg_human": round(float(np.mean(human)), 2),
            "avg_composite_3sig": round(float(np.mean(hybrid)), 1),
            "pearson_r_3sig": round(pr, 4),
            "avg_composite_claim_nlp": round(float(np.mean(claim_hybrid)), 1),
            "pearson_r_claim_nlp": _safe_pearson(claim_hybrid, [h * 10 for h in human]),
        }

        if include_llm and "_llm" in entries[0]:
            hybrid_4 = [_composite_4sig(e) for e in entries]
            claim_hybrid_4 = (
                _weighted_scores(
                    entries,
                    ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
                    claim_4sig_weights,
                )
                if claim_4sig_weights else
                [_composite_claim_4sig(e) for e in entries]
            )
            pr4, _ = pearsonr(hybrid_4, [h * 10 for h in human])
            result["avg_composite_4sig"] = round(
                float(np.mean(hybrid_4)), 1,
            )
            result["pearson_r_4sig"] = round(pr4, 4)
            result["avg_composite_claim_4sig"] = round(
                float(np.mean(claim_hybrid_4)), 1,
            )
            result["pearson_r_claim_4sig"] = _safe_pearson(claim_hybrid_4, [h * 10 for h in human])

        results[role] = result
    return results


def run_per_difficulty_analysis(
    data,
    include_llm=False,
    claim_nlp_weights=None,
    claim_4sig_weights=None,
):
    """Breakdown scores by normalized difficulty."""
    results = {}
    difficulties = sorted({_normalized_difficulty(e) for e in data})
    for difficulty in difficulties:
        entries = [
            e for e in data
            if _normalized_difficulty(e) == difficulty
        ]
        human_scaled = [e["human_score"] * 10 for e in entries]
        hybrid = [_composite_3sig(e) for e in entries]
        claim_hybrid = (
            _weighted_scores(entries, ["_sbert", "_nli", "_keyword", "_claim"], claim_nlp_weights)
            if claim_nlp_weights else
            [_composite_claim_nlp(e) for e in entries]
        )

        result = {
            "count": len(entries),
            "avg_human": round(float(np.mean([e["human_score"] for e in entries])), 2),
            "avg_composite_3sig": round(float(np.mean(hybrid)), 1),
            "pearson_r_3sig": _safe_pearson(hybrid, human_scaled),
            "avg_composite_claim_nlp": round(float(np.mean(claim_hybrid)), 1),
            "pearson_r_claim_nlp": _safe_pearson(claim_hybrid, human_scaled),
        }

        if include_llm and "_llm" in entries[0]:
            hybrid_4 = [_composite_4sig(e) for e in entries]
            claim_hybrid_4 = (
                _weighted_scores(
                    entries,
                    ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
                    claim_4sig_weights,
                )
                if claim_4sig_weights else
                [_composite_claim_4sig(e) for e in entries]
            )
            result["avg_llm"] = round(
                float(np.mean([e["_llm"] * 100 for e in entries])), 1,
            )
            result["avg_composite_4sig"] = round(float(np.mean(hybrid_4)), 1)
            result["pearson_r_4sig"] = _safe_pearson(hybrid_4, human_scaled)
            result["avg_composite_claim_4sig"] = round(float(np.mean(claim_hybrid_4)), 1)
            result["pearson_r_claim_4sig"] = _safe_pearson(claim_hybrid_4, human_scaled)

        results[difficulty] = result
    return results


def run_answer_length_analysis(
    data,
    include_llm=False,
    claim_nlp_weights=None,
    claim_4sig_weights=None,
):
    """Split candidate answers into short vs long by median word count."""
    word_counts = [_answer_word_count(e) for e in data]
    median_words = int(np.median(word_counts))
    groups = {
        "short": [e for e in data if _answer_word_count(e) <= median_words],
        "long": [e for e in data if _answer_word_count(e) > median_words],
    }

    results = {
        "threshold_words": median_words,
        "groups": {},
    }

    for label, entries in groups.items():
        if not entries:
            continue

        human_scaled = [e["human_score"] * 10 for e in entries]
        hybrid = [_composite_3sig(e) for e in entries]
        claim_hybrid = (
            _weighted_scores(entries, ["_sbert", "_nli", "_keyword", "_claim"], claim_nlp_weights)
            if claim_nlp_weights else
            [_composite_claim_nlp(e) for e in entries]
        )
        result = {
            "count": len(entries),
            "avg_words": round(
                float(np.mean([_answer_word_count(e) for e in entries])), 1,
            ),
            "avg_human": round(
                float(np.mean([e["human_score"] for e in entries])), 2,
            ),
            "avg_composite_3sig": round(float(np.mean(hybrid)), 1),
            "pearson_r_3sig": _safe_pearson(hybrid, human_scaled),
            "avg_composite_claim_nlp": round(float(np.mean(claim_hybrid)), 1),
            "pearson_r_claim_nlp": _safe_pearson(claim_hybrid, human_scaled),
        }

        if include_llm and "_llm" in entries[0]:
            hybrid_4 = [_composite_4sig(e) for e in entries]
            claim_hybrid_4 = (
                _weighted_scores(
                    entries,
                    ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
                    claim_4sig_weights,
                )
                if claim_4sig_weights else
                [_composite_claim_4sig(e) for e in entries]
            )
            result["avg_llm"] = round(
                float(np.mean([e["_llm"] * 100 for e in entries])), 1,
            )
            result["avg_composite_4sig"] = round(float(np.mean(hybrid_4)), 1)
            result["pearson_r_4sig"] = _safe_pearson(hybrid_4, human_scaled)
            result["avg_composite_claim_4sig"] = round(float(np.mean(claim_hybrid_4)), 1)
            result["pearson_r_claim_4sig"] = _safe_pearson(claim_hybrid_4, human_scaled)

        results["groups"][label] = result

    return results


def compute_llm_rubric_summary(data):
    """Aggregate rubric subscore averages overall and by quality."""
    if not data or "_llm_correctness" not in data[0]:
        return {}

    def _mean(entries, key):
        return round(float(np.mean([e[key] * 100 for e in entries])), 1)

    summary = {
        "overall": {
            "correctness": _mean(data, "_llm_correctness"),
            "completeness": _mean(data, "_llm_completeness"),
            "clarity": _mean(data, "_llm_clarity"),
            "depth": _mean(data, "_llm_depth"),
        },
        "per_quality": {},
    }

    for quality in ["good", "average", "poor"]:
        entries = [e for e in data if e["quality_level"] == quality]
        summary["per_quality"][quality] = {
            "correctness": _mean(entries, "_llm_correctness"),
            "completeness": _mean(entries, "_llm_completeness"),
            "clarity": _mean(entries, "_llm_clarity"),
            "depth": _mean(entries, "_llm_depth"),
        }

    return summary


def compute_cohens_kappa(data):
    """Compute Cohen's Kappa between rater 1 and rater 2."""
    pairs = [
        (e["rater_1"], e["rater_2"]) for e in data
        if e.get("rater_1") is not None and e.get("rater_2") is not None
    ]

    if len(pairs) < 10:
        return {
            "error": "Not enough dual-rated entries",
            "count": len(pairs),
        }

    r1 = [int(round(p[0])) for p in pairs]
    r2 = [int(round(p[1])) for p in pairs]

    def binned(score):
        if score <= 3:
            return "poor"
        elif score <= 6:
            return "average"
        return "good"

    r1_binned = [binned(s) for s in r1]
    r2_binned = [binned(s) for s in r2]

    kappa = cohen_kappa_score(r1_binned, r2_binned)
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
    """Compute ROUGE on ideal vs candidate answers."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True,
    )

    rouge1_scores = []
    rougel_scores = []

    for entry in data:
        scores = scorer.score(
            entry["ideal_answer"], entry["candidate_answer"],
        )
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougel_scores.append(scores["rougeL"].fmeasure)

    quality_rouge = {}
    for quality in ["good", "average", "poor"]:
        indices = [
            i for i, e in enumerate(data)
            if e["quality_level"] == quality
        ]
        quality_rouge[quality] = {
            "rouge1": round(
                float(np.mean([rouge1_scores[i] for i in indices])), 4,
            ),
            "rougeL": round(
                float(np.mean([rougel_scores[i] for i in indices])), 4,
            ),
        }

    return {
        "rouge1_mean": round(float(np.mean(rouge1_scores)), 4),
        "rougeL_mean": round(float(np.mean(rougel_scores)), 4),
        "per_quality": quality_rouge,
    }


def compute_bertscore_metrics(data):
    """Compute BERTScore on a sample of 30 entries."""
    try:
        from bert_score import score as bert_score

        sample = []
        for q in ["good", "average", "poor"]:
            entries = [e for e in data if e["quality_level"] == q]
            sample.extend(entries[:10])

        refs = [e["ideal_answer"] for e in sample]
        cands = [e["candidate_answer"] for e in sample]

        logger.info(f"Computing BERTScore on {len(sample)} samples...")
        P, R, F1 = bert_score(
            cands, refs,
            model_type="distilbert-base-uncased",
            lang="en", verbose=False,
        )

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


def main(with_llm: bool = False):
    logger.info("=" * 60)
    logger.info("FULL EVALUATION PIPELINE")
    if with_llm:
        logger.info("  Mode: 4-signal (NLP + LLM-as-judge)")
    else:
        logger.info("  Mode: 3-signal (NLP-only)")
    logger.info("=" * 60)

    data = load_data()
    logger.info(f"Loaded {len(data)} entries")

    settings, registry = init_models()

    # 1. Compute NLP scores
    total_steps = 11 if with_llm else 10
    step = 1
    logger.info(f"\n[{step}/{total_steps}] Computing NLP scores...")
    data = compute_all_scores(data, settings, registry)

    # 1b. Compute LLM scores if requested
    if with_llm:
        step += 1
        logger.info(
            f"\n[{step}/{total_steps}]"
            " Computing LLM-as-judge scores..."
        )
        data = asyncio.run(compute_llm_scores(data))

    # 2. Optimize claim-based weights
    step += 1
    logger.info(
        f"\n[{step}/{total_steps}] Optimizing claim-based weights..."
    )
    claim_weight_search = {
        "claim_hybrid_nlp": run_weight_search(
            data,
            ["_sbert", "_nli", "_keyword", "_claim"],
            step=0.05,
        ),
    }
    claim_nlp_weights = tuple(
        claim_weight_search["claim_hybrid_nlp"]["best_weights"][key]
        for key in claim_weight_search["claim_hybrid_nlp"]["signal_keys"]
    )
    logger.info(
        "  Best claim NLP weights: "
        + ", ".join(
            f"{key}={value:.2f}"
            for key, value in claim_weight_search["claim_hybrid_nlp"]["best_weights"].items()
        )
        + f" | Pearson={claim_weight_search['claim_hybrid_nlp']['best_pearson_r']:.4f}"
    )

    claim_4sig_weights = None
    if with_llm:
        claim_weight_search["claim_hybrid_4sig"] = run_weight_search(
            data,
            ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
            step=0.05,
        )
        claim_4sig_weights = tuple(
            claim_weight_search["claim_hybrid_4sig"]["best_weights"][key]
            for key in claim_weight_search["claim_hybrid_4sig"]["signal_keys"]
        )
        logger.info(
            "  Best claim 4-signal weights: "
            + ", ".join(
                f"{key}={value:.2f}"
                for key, value in claim_weight_search["claim_hybrid_4sig"]["best_weights"].items()
            )
            + f" | Pearson={claim_weight_search['claim_hybrid_4sig']['best_pearson_r']:.4f}"
        )

    # 3. Baseline comparisons
    step += 1
    logger.info(
        f"\n[{step}/{total_steps}] Running baseline comparisons..."
    )
    baselines = run_baselines(
        data,
        include_llm=with_llm,
        claim_nlp_weights=claim_nlp_weights,
        claim_4sig_weights=claim_4sig_weights,
    )

    logger.info("\n  BASELINE COMPARISON:")
    logger.info(f"  {'Method':<35} {'Pearson':>8} {'Spearman':>8}")
    logger.info("  " + "-" * 55)
    for name, result in baselines.items():
        logger.info(
            f"  {name:<35} {result['pearson_r']:>8.4f}"
            f" {result['spearman_r']:>8.4f}"
        )

    # 4. Cross-validation
    step += 1
    logger.info(
        f"\n[{step}/{total_steps}] Running 5-fold cross-validation..."
    )
    cross_validation = {
        "hybrid_3sig_optimal": run_cross_validation(
            data,
            ["_sbert", "_nli", "_keyword"],
            step=0.05,
        ),
        "claim_hybrid_nlp": run_cross_validation(
            data,
            ["_sbert", "_nli", "_keyword", "_claim"],
            step=0.05,
        ),
    }
    if with_llm:
        cross_validation["hybrid_4sig_optimal"] = run_cross_validation(
            data,
            ["_sbert", "_nli", "_keyword", "_llm"],
            step=0.10,
        )
        cross_validation["claim_hybrid_4sig"] = run_cross_validation(
            data,
            ["_sbert", "_nli", "_keyword", "_claim", "_llm"],
            step=0.05,
        )

    for name, stats in cross_validation.items():
        if "error" in stats:
            logger.info(f"  {name:<24} error={stats['error']}")
            continue
        logger.info(
            f"  {name:<24}"
            f" mean Pearson={stats['mean_pearson_r']:.4f} ± {stats['std_pearson_r']:.4f}"
            f" | mean Spearman={stats['mean_spearman_r']:.4f} ± {stats['std_spearman_r']:.4f}"
        )

    # 5. Per-quality analysis
    step += 1
    logger.info(f"\n[{step}/{total_steps}] Per-quality breakdown...")
    quality_analysis = run_per_quality_analysis(
        data,
        include_llm=with_llm,
        claim_nlp_weights=claim_nlp_weights,
        claim_4sig_weights=claim_4sig_weights,
    )
    for q, stats in quality_analysis.items():
        msg = (
            f"  {q.upper():>8}:"
            f" human={stats['avg_human_score']:.1f}"
            f" composite={stats['avg_composite_3sig']:.1f}"
            f" claim={stats['avg_claim']:.1f}"
            f" claim_hybrid={stats['avg_composite_claim_nlp']:.1f}"
            f" sbert={stats['avg_sbert']:.1f}"
            f" nli={stats['avg_nli']:.1f}"
            f" kw={stats['avg_keyword']:.1f}"
        )
        if "avg_llm" in stats:
            msg += f" llm={stats['avg_llm']:.1f}"
        logger.info(msg)

    # 6. Per-role analysis
    step += 1
    logger.info(f"\n[{step}/{total_steps}] Per-role breakdown...")
    role_analysis = run_per_role_analysis(
        data,
        include_llm=with_llm,
        claim_nlp_weights=claim_nlp_weights,
        claim_4sig_weights=claim_4sig_weights,
    )
    for role, stats in role_analysis.items():
        msg = (
            f"  {role:<20}"
            f" Pearson={stats['pearson_r_3sig']:.4f}"
            f" avg={stats['avg_composite_3sig']:.1f}"
            f" claim_avg={stats['avg_composite_claim_nlp']:.1f}"
        )
        if "pearson_r_4sig" in stats:
            msg += f" Pearson_4sig={stats['pearson_r_4sig']:.4f}"
        logger.info(msg)

    # 7. Per-difficulty analysis
    step += 1
    logger.info(f"\n[{step}/{total_steps}] Per-difficulty breakdown...")
    difficulty_analysis = run_per_difficulty_analysis(
        data,
        include_llm=with_llm,
        claim_nlp_weights=claim_nlp_weights,
        claim_4sig_weights=claim_4sig_weights,
    )
    for difficulty, stats in difficulty_analysis.items():
        msg = (
            f"  {difficulty:<8}"
            f" avg={stats['avg_composite_3sig']:.1f}"
            f" claim_avg={stats['avg_composite_claim_nlp']:.1f}"
            f" human={stats['avg_human']:.1f}"
        )
        if stats.get("pearson_r_3sig") is not None:
            msg += f" Pearson={stats['pearson_r_3sig']:.4f}"
        logger.info(msg)

    # 8. Short-vs-long answer analysis
    step += 1
    logger.info(f"\n[{step}/{total_steps}] Short-vs-long answer breakdown...")
    answer_length_analysis = run_answer_length_analysis(
        data,
        include_llm=with_llm,
        claim_nlp_weights=claim_nlp_weights,
        claim_4sig_weights=claim_4sig_weights,
    )
    logger.info(
        f"  Median split threshold:"
        f" {answer_length_analysis['threshold_words']} words"
    )
    for label, stats in answer_length_analysis["groups"].items():
        msg = (
            f"  {label.upper():>5}:"
            f" count={stats['count']}"
            f" avg_words={stats['avg_words']:.1f}"
            f" avg={stats['avg_composite_3sig']:.1f}"
            f" claim_avg={stats['avg_composite_claim_nlp']:.1f}"
            f" human={stats['avg_human']:.1f}"
        )
        if stats.get("pearson_r_3sig") is not None:
            msg += f" Pearson={stats['pearson_r_3sig']:.4f}"
        logger.info(msg)

    # 9. Inter-rater agreement
    step += 1
    logger.info(
        f"\n[{step}/{total_steps}]"
        " Computing inter-rater agreement..."
    )
    kappa = compute_cohens_kappa(data)
    logger.info(
        f"  Cohen's Kappa: {kappa.get('cohens_kappa', 'N/A')}"
    )
    logger.info(
        f"  Interpretation: {kappa.get('kappa_interpretation', 'N/A')}"
    )

    # 10. Feedback metrics (ROUGE + BERTScore)
    step += 1
    logger.info(
        f"\n[{step}/{total_steps}]"
        " Computing feedback metrics (ROUGE + BERTScore)..."
    )
    rouge_results = compute_feedback_metrics(data)
    logger.info(f"  ROUGE-1: {rouge_results['rouge1_mean']:.4f}")
    logger.info(f"  ROUGE-L: {rouge_results['rougeL_mean']:.4f}")

    bertscore_results = compute_bertscore_metrics(data)
    if "mean_f1" in bertscore_results:
        logger.info(
            f"  BERTScore F1: {bertscore_results['mean_f1']:.4f}"
        )

    llm_rubric_summary = (
        compute_llm_rubric_summary(data) if with_llm else {}
    )

    # Find the best method
    best_method = max(
        baselines.keys(), key=lambda k: baselines[k]["pearson_r"],
    )

    # Compile full report
    report = {
        "summary": {
            "total_entries": len(data),
            "questions": len(set(e["question_index"] for e in data)),
            "roles": sorted(set(e["role"] for e in data)),
            "quality_levels": ["good", "average", "poor"],
            "mode": "4-signal hybrid" if with_llm else "3-signal NLP",
            "optimal_weights_3sig": {
                "sbert": 0.45, "nli": 0.05, "keyword": 0.50,
            },
            "optimal_weights_claim_nlp": claim_weight_search["claim_hybrid_nlp"]["best_weights"],
            "best_method": best_method,
            "best_pearson_r": baselines[best_method]["pearson_r"],
            "best_spearman_r": baselines[best_method]["spearman_r"],
        },
        "baseline_comparison": baselines,
        "claim_weight_search": claim_weight_search,
        "cross_validation": cross_validation,
        "per_quality_analysis": quality_analysis,
        "per_role_analysis": role_analysis,
        "per_difficulty_analysis": difficulty_analysis,
        "answer_length_analysis": answer_length_analysis,
        "inter_rater_agreement": kappa,
        "rouge_scores": rouge_results,
        "bertscore": bertscore_results,
    }
    if llm_rubric_summary:
        report["llm_rubric_summary"] = llm_rubric_summary
    if with_llm and "claim_hybrid_4sig" in claim_weight_search:
        report["summary"]["optimal_weights_claim_4sig"] = (
            claim_weight_search["claim_hybrid_4sig"]["best_weights"]
        )

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"FULL REPORT SAVED: {REPORT_PATH}")
    logger.info("=" * 60)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "\n  Method                              Pearson   Spearman"
    )
    logger.info("  " + "─" * 55)
    for name, result in baselines.items():
        marker = " ◄ BEST" if name == best_method else ""
        logger.info(
            f"  {name:<35} {result['pearson_r']:.4f}"
            f"    {result['spearman_r']:.4f}{marker}"
        )
    logger.info(f"\n  Best method: {best_method}")
    logger.info(
        "  Best claim NLP weights: "
        + ", ".join(
            f"{key}={value:.2f}"
            for key, value in claim_weight_search["claim_hybrid_nlp"]["best_weights"].items()
        )
    )
    if with_llm and "claim_hybrid_4sig" in claim_weight_search:
        logger.info(
            "  Best claim 4-signal weights: "
            + ", ".join(
                f"{key}={value:.2f}"
                for key, value in claim_weight_search["claim_hybrid_4sig"]["best_weights"].items()
            )
        )
    logger.info(
        f"  Inter-rater Kappa:"
        f" {kappa.get('cohens_kappa', 'N/A')}"
    )
    logger.info(f"  ROUGE-L: {rouge_results['rougeL_mean']:.4f}")
    if "mean_f1" in bertscore_results:
        logger.info(
            f"  BERTScore F1: {bertscore_results['mean_f1']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-llm", action="store_true",
        help="Include LLM-as-judge (Groq API) in evaluation",
    )
    args = parser.parse_args()
    main(with_llm=args.with_llm)
