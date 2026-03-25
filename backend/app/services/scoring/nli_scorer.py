import torch
from dataclasses import dataclass
from app.models_loader import get_registry


@dataclass
class NLIResult:
    normalized_score: float
    entailment: float
    contradiction: float
    neutral: float


def _label_indices(model):
    label2id = {
        str(label).lower(): idx
        for label, idx in getattr(model.config, "label2id", {}).items()
    }
    contradiction_idx = label2id.get("contradiction", 0)
    entailment_idx = label2id.get("entailment", 1)
    neutral_idx = label2id.get("neutral")
    return contradiction_idx, entailment_idx, neutral_idx


def _pair_distribution(
    tokenizer,
    model,
    device,
    premise: str,
    hypothesis: str,
) -> tuple[float, float, float]:
    inputs = tokenizer(
        premise,
        hypothesis,
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
    contradiction_idx, entailment_idx, neutral_idx = _label_indices(model)
    entailment_prob = probabilities[entailment_idx].item()
    contradiction_prob = probabilities[contradiction_idx].item()
    if neutral_idx is not None:
        neutral_prob = probabilities[neutral_idx].item()
    else:
        neutral_prob = max(0.0, 1.0 - entailment_prob - contradiction_prob)
    return entailment_prob, contradiction_prob, neutral_prob


def score_detailed(candidate_answer: str, ideal_answer: str) -> NLIResult:
    """
    Compute a bidirectional NLI compatibility score and expose the averaged
    entailment / contradiction / neutral probabilities for downstream scorers.
    """
    registry = get_registry()
    tokenizer = registry.nli_tokenizer
    model = registry.nli_model
    device = registry.device

    forward_entail, forward_contra, forward_neutral = _pair_distribution(
        tokenizer,
        model,
        device,
        ideal_answer,
        candidate_answer,
    )
    backward_entail, backward_contra, backward_neutral = _pair_distribution(
        tokenizer,
        model,
        device,
        candidate_answer,
        ideal_answer,
    )

    entailment = (forward_entail + backward_entail) / 2.0
    contradiction = (forward_contra + backward_contra) / 2.0
    neutral = (forward_neutral + backward_neutral) / 2.0

    # Cross-encoder NLI often assigns most mass to neutral for partial answers.
    # Compare entailment directly against contradiction to retain a useful
    # 0..1 discrimination signal instead of treating neutral as near-perfect.
    forward_score = forward_entail / max(forward_entail + forward_contra, 1e-8)
    backward_score = backward_entail / max(backward_entail + backward_contra, 1e-8)
    normalized = (forward_score + backward_score) / 2.0
    return NLIResult(
        normalized_score=max(0.0, min(1.0, normalized)),
        entailment=max(0.0, min(1.0, entailment)),
        contradiction=max(0.0, min(1.0, contradiction)),
        neutral=max(0.0, min(1.0, neutral)),
    )


def score(candidate_answer: str, ideal_answer: str) -> float:
    """
    Compute a bidirectional NLI compatibility score.
    We compare entailment vs contradiction in both directions and average them.
    """
    return score_detailed(candidate_answer, ideal_answer).normalized_score
