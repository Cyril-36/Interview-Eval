import torch
from app.models_loader import get_registry


def score(candidate_answer: str, ideal_answer: str) -> float:
    """
    Compute NLI entailment probability.
    Premise = ideal_answer, Hypothesis = candidate_answer.
    Label order for nli-deberta-v3-small: [contradiction, neutral, entailment].
    """
    registry = get_registry()
    tokenizer = registry.nli_tokenizer
    model = registry.nli_model
    device = registry.device

    inputs = tokenizer(
        ideal_answer,
        candidate_answer,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)[0]
    # entailment is index 2 for this model
    entailment_prob = probabilities[2].item()
    return max(0.0, min(1.0, entailment_prob))
