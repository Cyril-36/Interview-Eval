from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from sentence_transformers import util

from app.config import get_settings
from app.models_loader import get_registry
from app.services.scoring import nli_scorer

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class ClaimInput(Protocol):
    @property
    def text(self) -> str:
        ...

    @property
    def importance(self) -> str:
        ...


ClaimInputs = Sequence[str] | Sequence[ClaimInput]


@dataclass
class ClaimMatch:
    claim: str
    best_sentence: str
    similarity: float
    entailment: float
    combined: float
    covered: bool
    contradiction: float = 0.0
    importance: str = "core"


def _split_candidate_sentences(candidate_answer: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(candidate_answer)
        if len(sentence.strip()) > 5
    ]
    if not sentences:
        return [candidate_answer.strip()]

    # Add 2-sentence sliding windows so claims spanning adjacent
    # sentences can still match well.
    windows = []
    for i in range(len(sentences) - 1):
        windows.append(sentences[i] + " " + sentences[i + 1])

    return sentences + windows


def _normalize_claims(claims: ClaimInputs) -> list[tuple[str, str]]:
    normalized = []
    for claim in claims:
        if isinstance(claim, str):
            normalized.append((claim, "core"))
        else:
            normalized.append((claim.text, claim.importance))
    return normalized


def match_claims(
    candidate_answer: str,
    claims: ClaimInputs,
    threshold: float = 0.62,
) -> list[ClaimMatch]:
    settings = get_settings()
    registry = get_registry()
    sbert = registry.sbert

    sentences = _split_candidate_sentences(candidate_answer)
    normalized_claims = _normalize_claims(claims)
    claim_texts = [claim for claim, _importance in normalized_claims]
    claim_embeddings = sbert.encode(claim_texts, convert_to_tensor=True)
    sentence_embeddings = sbert.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(claim_embeddings, sentence_embeddings)

    matches: list[ClaimMatch] = []
    for claim_index, (claim, importance) in enumerate(normalized_claims):
        similarities = similarity_matrix[claim_index]
        best_index = int(similarities.argmax().item())
        best_sentence = sentences[best_index]
        similarity = float(max(0.0, min(1.0, similarities[best_index].item())))
        nli_result = nli_scorer.score_detailed(best_sentence, claim)
        entailment = nli_result.entailment
        contradiction = nli_result.contradiction
        combined = (
            0.70 * similarity
            + 0.30 * entailment
            - settings.CLAIM_CONTRADICTION_PENALTY * contradiction
        )
        combined = max(0.0, min(1.0, combined))
        matches.append(
            ClaimMatch(
                claim=claim,
                best_sentence=best_sentence,
                similarity=similarity,
                entailment=entailment,
                combined=combined,
                covered=combined >= threshold,
                contradiction=contradiction,
                importance=importance,
            )
        )

    return matches
