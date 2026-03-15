from sentence_transformers import util
from app.models_loader import get_registry


def score(candidate_answer: str, ideal_answer: str) -> float:
    sbert = get_registry().sbert
    embeddings = sbert.encode(
        [candidate_answer, ideal_answer], convert_to_tensor=True
    )
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return max(0.0, min(1.0, similarity))
