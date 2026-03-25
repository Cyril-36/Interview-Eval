from dataclasses import dataclass
import re

from sentence_transformers import util

from app.config import get_settings
from app.models_loader import get_registry
from app.services.scoring import nli_scorer

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class ClaimMatch:
    claim: str
    best_sentence: str
    similarity: float
    entailment: float
    combined: float
    covered: bool
    contradiction: float = 0.0


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


def match_claims(
    candidate_answer: str,
    claims: list[str],
    threshold: float = 0.62,
) -> list[ClaimMatch]:
    settings = get_settings()
    registry = get_registry()
    sbert = registry.sbert

    sentences = _split_candidate_sentences(candidate_answer)
    claim_embeddings = sbert.encode(claims, convert_to_tensor=True)
    sentence_embeddings = sbert.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(claim_embeddings, sentence_embeddings)

    matches: list[ClaimMatch] = []
    for claim_index, claim in enumerate(claims):
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
            )
        )

    return matches
