import pytest
import torch

from app.services.scoring import nli_scorer


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {}


class _DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyModel:
    def __init__(self, probabilities):
        self._probabilities = probabilities
        self._index = 0
        self.config = type(
            "Config",
            (),
            {
                "label2id": {
                    "entailment": 0,
                    "neutral": 1,
                    "contradiction": 2,
                },
            },
        )()

    def __call__(self, **kwargs):
        probs = torch.tensor([self._probabilities[self._index]], dtype=torch.float32)
        self._index += 1
        return _DummyOutput(torch.log(probs))


class _DummyRegistry:
    def __init__(self, probabilities):
        self.nli_tokenizer = _DummyTokenizer()
        self.nli_model = _DummyModel(probabilities)
        self.device = "cpu"


def test_nli_score_uses_label_mapping_and_bidirectional_ratio(monkeypatch):
    registry = _DummyRegistry([
        [0.4, 0.5, 0.1],  # forward: entailment / (entailment + contradiction) = 0.8
        [0.6, 0.2, 0.2],  # backward: 0.6 / (0.6 + 0.2) = 0.75
    ])
    monkeypatch.setattr(nli_scorer, "get_registry", lambda: registry)

    score = nli_scorer.score("candidate", "ideal")

    assert score == pytest.approx(0.775, rel=1e-6)


def test_nli_score_detailed_exposes_contradiction(monkeypatch):
    registry = _DummyRegistry([
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2],
    ])
    monkeypatch.setattr(nli_scorer, "get_registry", lambda: registry)

    result = nli_scorer.score_detailed("candidate", "ideal")

    assert result.normalized_score == pytest.approx(((0.7 / 0.8) + (0.5 / 0.7)) / 2, rel=1e-6)
    assert result.entailment == pytest.approx(0.6, rel=1e-6)
    assert result.contradiction == pytest.approx(0.15, rel=1e-6)
    assert result.neutral == pytest.approx(0.25, rel=1e-6)
