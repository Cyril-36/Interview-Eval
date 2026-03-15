from sentence_transformers import util
from app.models_loader import get_registry
from app.config import get_settings
from app.services.preprocessor import preprocess_tokens


def score(candidate_answer: str, ideal_answer: str) -> tuple[float, list[str]]:
    """
    Compute keyword coverage ratio and return (score, missing_keywords).
    Uses KeyBERT to extract keywords from ideal answer, then checks
    coverage via string match or SBERT similarity.
    """
    settings = get_settings()
    registry = get_registry()
    keybert_model = registry.keybert
    sbert = registry.sbert

    # Extract top-N keywords from ideal answer
    keywords_with_scores = keybert_model.extract_keywords(
        ideal_answer,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=settings.KEYWORD_TOP_N,
    )

    if not keywords_with_scores:
        return 1.0, []

    keywords = [kw for kw, _ in keywords_with_scores]

    # Preprocess candidate for string matching
    candidate_tokens = preprocess_tokens(candidate_answer)
    candidate_text = " ".join(candidate_tokens)

    # Encode candidate answer for semantic matching
    candidate_embedding = sbert.encode(candidate_answer, convert_to_tensor=True)

    matched = []
    missing = []

    for keyword in keywords:
        # Check direct string match (lemmatized)
        keyword_lower = keyword.lower()
        if keyword_lower in candidate_text or keyword_lower in candidate_answer.lower():
            matched.append(keyword)
            continue

        # Check SBERT semantic similarity
        keyword_embedding = sbert.encode(keyword, convert_to_tensor=True)
        similarity = util.cos_sim(keyword_embedding, candidate_embedding).item()

        if similarity >= settings.KEYWORD_SIMILARITY_THRESHOLD:
            matched.append(keyword)
        else:
            missing.append(keyword)

    coverage = len(matched) / len(keywords) if keywords else 1.0
    return max(0.0, min(1.0, coverage)), missing
