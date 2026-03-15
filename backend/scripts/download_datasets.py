"""
Download and cache public datasets for calibration and evaluation.

Datasets:
- STS-Benchmark: Validate SBERT similarity and calibrate thresholds
- SNLI: Sanity-check DeBERTa-NLI entailment predictions
- MultiNLI: Additional NLI validation data
- Interview QA (HuggingFace): vinaythan/interview-questions-answers
- HR Interview (Kaggle): aryan208/hr-interview-questions-and-ideal-answers
- ML QA: ML interview questions CSV

Usage:
    cd backend && python -m scripts.download_datasets
"""

import json
import csv
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def download_sts_benchmark():
    """Download STS-Benchmark for SBERT threshold calibration."""
    from datasets import load_dataset

    output_path = DATA_DIR / "sts_benchmark_test.json"
    if output_path.exists():
        logger.info(f"STS-Benchmark already present: {output_path}")
        return

    logger.info("Downloading STS-Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")

    records = []
    for row in dataset:
        records.append({
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "score": row["score"],
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"STS-Benchmark saved: {len(records)} pairs -> {output_path}")


def download_snli():
    """Download SNLI sample for NLI sanity checks."""
    from datasets import load_dataset

    output_path = DATA_DIR / "snli_validation_sample.json"
    if output_path.exists():
        logger.info(f"SNLI already present: {output_path}")
        return

    logger.info("Downloading SNLI (validation split, first 2000)...")
    dataset = load_dataset("stanfordnlp/snli", split="validation")

    records = []
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for row in list(dataset)[:2000]:
        if row["label"] == -1:
            continue
        records.append({
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": label_map.get(row["label"], "unknown"),
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"SNLI saved: {len(records)} pairs -> {output_path}")


def download_multinli():
    """Download MultiNLI sample for additional NLI validation."""
    from datasets import load_dataset

    output_path = DATA_DIR / "multinli_validation_sample.json"
    if output_path.exists():
        logger.info(f"MultiNLI already present: {output_path}")
        return

    logger.info("Downloading MultiNLI (validation_matched, first 2000)...")
    dataset = load_dataset("nyu-mll/multi_nli", split="validation_matched")

    records = []
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for row in list(dataset)[:2000]:
        records.append({
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": label_map.get(row["label"], "unknown"),
            "genre": row.get("genre", ""),
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"MultiNLI saved: {len(records)} pairs -> {output_path}")


def download_interview_qa():
    """Download HuggingFace interview QA dataset (vinaythan)."""
    from datasets import load_dataset

    output_path = DATA_DIR / "interview_qa.json"
    if output_path.exists():
        logger.info(f"Interview QA already present: {output_path}")
        return

    logger.info("Downloading Interview QA dataset...")
    try:
        dataset = load_dataset("vinaythan/interview-questions-answers", split="train")
        records = []
        for row in dataset:
            records.append({
                "question": row.get("Question", row.get("question", "")),
                "answer": row.get("Answer", row.get("answer", "")),
            })

        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Interview QA saved: {len(records)} pairs -> {output_path}")
    except Exception as e:
        logger.warning(f"Could not download interview QA dataset: {e}")


def load_local_hr_interview():
    """
    Process the Kaggle HR Interview dataset (aryan208/hr-interview-questions-and-ideal-answers).

    The full dataset is 2.5M rows. We create a stratified sample of ~1000 rows
    covering all roles and difficulty levels.

    Expected: data/hr_interview_questions_dataset.json (extracted from Kaggle zip)
    Output:   data/hr_interview_sampled.json
    """
    sampled_path = DATA_DIR / "hr_interview_sampled.json"
    if sampled_path.exists():
        with open(sampled_path) as f:
            data = json.load(f)
        logger.info(f"HR Interview dataset already present: {len(data)} sampled pairs")
        return

    full_path = DATA_DIR / "hr_interview_questions_dataset.json"
    if not full_path.exists():
        logger.warning(
            "HR Interview dataset not found.\n"
            "  Download from: https://www.kaggle.com/datasets/aryan208/hr-interview-questions-and-ideal-answers\n"
            "  Extract JSON to: backend/data/hr_interview_questions_dataset.json\n"
            "  Then re-run this script."
        )
        return

    logger.info("Processing full HR interview dataset (stratified sampling)...")
    with open(full_path) as f:
        hr = json.load(f)

    buckets = defaultdict(list)
    for item in hr:
        key = (item.get("role", ""), item.get("difficulty", ""))
        if len(buckets[key]) < 20:
            buckets[key].append(item)

    sampled = []
    for items in buckets.values():
        sampled.extend(items)

    sampled = sampled[:1000]
    with open(sampled_path, "w") as f:
        json.dump(sampled, f, indent=2)

    roles = set(item.get("role", "") for item in sampled)
    logger.info(f"HR Interview sampled: {len(sampled)} pairs across {len(roles)} roles -> {sampled_path}")


def load_local_ml_qa():
    """
    Process the ML QA dataset CSV into JSON.

    Expected: data/ml_interview_questions.csv
    Output:   data/ml_interview_qa.json
    """
    csv_path = DATA_DIR / "ml_interview_questions.csv"
    json_path = DATA_DIR / "ml_interview_qa.json"

    if json_path.exists():
        logger.info(f"ML QA JSON already present: {json_path}")
        return

    if not csv_path.exists():
        logger.warning(
            "ML QA dataset not found.\n"
            "  Download and extract the ML QA dataset zip.\n"
            "  Copy ml_interview_questions.csv to: backend/data/"
        )
        return

    logger.info("Converting ML QA CSV to JSON...")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            records.append({
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": row.get("category", "ML"),
                "difficulty": row.get("difficulty", "medium"),
                "company_tags": row.get("company_tags", ""),
                "topic_tags": row.get("topic_tags", ""),
            })

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"ML QA saved: {len(records)} pairs -> {json_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving datasets to: {DATA_DIR}")

    # Remote datasets (HuggingFace)
    logger.info("\n=== Downloading remote datasets ===")
    download_sts_benchmark()
    download_snli()
    download_multinli()
    download_interview_qa()

    # Local datasets (pre-downloaded from Kaggle)
    logger.info("\n=== Processing local datasets ===")
    load_local_hr_interview()
    load_local_ml_qa()

    logger.info(f"\n{'='*50}")
    logger.info("All datasets processed!")
    logger.info(f"{'='*50}")
    logger.info(f"Contents of {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
