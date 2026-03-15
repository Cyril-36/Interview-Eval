"""
Evaluation metrics for the scoring pipeline.

Provides:
- BERTScore (F1) for feedback quality evaluation
- ROUGE-1 and ROUGE-L for keyword recall measurement
- Pearson and Spearman correlation for composite score vs human score alignment
"""

import json
import logging
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """
    Compute BERTScore F1 between generated feedback and reference feedback.

    Args:
        predictions: System-generated feedback strings
        references: Human-written reference feedback strings

    Returns:
        Dict with precision, recall, f1 (each a list of per-instance scores)
    """
    from bert_score import score as bert_score

    P, R, F1 = bert_score(
        predictions, references,
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        verbose=False,
    )
    return {
        "precision": [p.item() for p in P],
        "recall": [r.item() for r in R],
        "f1": [f.item() for f in F1],
        "mean_f1": F1.mean().item(),
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L scores.

    Args:
        predictions: System-generated texts
        references: Reference texts

    Returns:
        Dict with rouge1 and rougeL average scores
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougel_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougel_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1_mean": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "rougeL_mean": sum(rougel_scores) / len(rougel_scores) if rougel_scores else 0,
        "rouge1_scores": rouge1_scores,
        "rougeL_scores": rougel_scores,
    }


def compute_correlation(
    system_scores: list[float],
    human_scores: list[float],
) -> dict:
    """
    Compute Pearson and Spearman correlation between system and human scores.

    Args:
        system_scores: Composite scores from the pipeline (0-100)
        human_scores: Average human-labeled scores (0-10, will be scaled to 0-100)

    Returns:
        Dict with pearson_r, pearson_p, spearman_r, spearman_p
    """
    if len(system_scores) < 3:
        return {"error": "Need at least 3 samples for correlation"}

    # Scale human scores to 0-100 if they're on 0-10 scale
    max_human = max(human_scores)
    if max_human <= 10:
        human_scaled = [h * 10 for h in human_scores]
    else:
        human_scaled = human_scores

    pearson_r, pearson_p = pearsonr(system_scores, human_scaled)
    spearman_r, spearman_p = spearmanr(system_scores, human_scaled)

    return {
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
    }


def run_evaluation(dataset_path: str | Path) -> dict:
    """
    Run full evaluation on a human-labeled dataset.

    Dataset format (JSON):
    [
        {
            "question": "...",
            "ideal_answer": "...",
            "candidate_answer": "...",
            "human_score": 7.5,
            "reference_feedback": "..." (optional)
        }
    ]
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.scoring.composite import evaluate

    dataset_path = Path(dataset_path)
    with open(dataset_path) as f:
        data = json.load(f)

    system_scores = []
    human_scores = []
    feedbacks_generated = []
    feedbacks_reference = []

    for entry in data:
        result = evaluate(entry["candidate_answer"], entry["ideal_answer"])
        system_scores.append(result.composite)
        human_scores.append(entry["human_score"])

        if "reference_feedback" in entry:
            feedbacks_reference.append(entry["reference_feedback"])
            # Generate a simple feedback string for comparison
            feedback_str = f"Score: {result.composite}. Missing: {', '.join(result.missing_keywords[:3])}"
            feedbacks_generated.append(feedback_str)

    results = {
        "num_samples": len(data),
        "correlation": compute_correlation(system_scores, human_scores),
        "score_stats": {
            "mean_system": round(sum(system_scores) / len(system_scores), 1),
            "mean_human_scaled": round(sum(human_scores) / len(human_scores) * 10, 1),
        },
    }

    if feedbacks_reference and feedbacks_generated:
        results["bertscore"] = compute_bertscore(feedbacks_generated, feedbacks_reference)
        results["rouge"] = compute_rouge(feedbacks_generated, feedbacks_reference)

    return results
