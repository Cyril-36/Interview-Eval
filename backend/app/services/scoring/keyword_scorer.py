from sentence_transformers import util
from app.models_loader import get_registry
from app.config import get_settings
from app.services.preprocessor import preprocess_tokens


def _clean_missing_keywords(keywords: list[str]) -> list[str]:
    cleaned: list[str] = []
    signatures: list[set[str]] = []

    for keyword in keywords:
        tokens = preprocess_tokens(keyword)
        if not tokens:
            continue

        deduped_tokens: list[str] = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped_tokens.append(token)

        if not deduped_tokens:
            continue

        token_set = set(deduped_tokens)

        # Skip exact or near-duplicate shorter variants like
        # "imputation" when "median imputation" is already present.
        duplicate = False
        for existing in signatures:
            if token_set == existing:
                duplicate = True
                break
            if len(token_set) == 1 and token_set.issubset(existing):
                duplicate = True
                break
        if duplicate:
            continue

        # Prefer a more informative phrase over an earlier single-word tag.
        replaced = False
        if len(token_set) > 1:
            for index, existing in enumerate(signatures):
                if len(existing) == 1 and existing.issubset(token_set):
                    signatures[index] = token_set
                    cleaned[index] = " ".join(deduped_tokens[:3])
                    replaced = True
                    break
        if replaced:
            continue

        signatures.append(token_set)
        cleaned.append(" ".join(deduped_tokens[:3]))

    return cleaned


def score(candidate_answer: str, ideal_answer: str) -> tuple[float, list[str]]:
    """
    Compute keyword coverage ratio and return (score, missing_keywords).
    Uses KeyBERT to extract keywords from ideal answer, then checks
    coverage via string match or SBERT semantic similarity against
    individual sentences of the candidate answer.
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

    # Split candidate into sentences for fine-grained semantic matching
    # This prevents comparing a short keyword against a long paragraph
    candidate_sentences = [
        s.strip() for s in candidate_answer.replace(".", ". ").split(". ")
        if len(s.strip()) > 5
    ]
    if not candidate_sentences:
        candidate_sentences = [candidate_answer]

    # Encode all candidate sentences at once for efficiency
    sentence_embeddings = sbert.encode(
        candidate_sentences, convert_to_tensor=True
    )

    matched = []
    missing = []

    for keyword in keywords:
        # Check 1: Direct string match (lemmatized)
        keyword_lower = keyword.lower()
        if keyword_lower in candidate_text:
            matched.append(keyword)
            continue
        if keyword_lower in candidate_answer.lower():
            matched.append(keyword)
            continue

        # Check 2: Individual word match for bigrams
        # If keyword is "learning experience", check if both
        # "learning" and "experience" appear in the candidate
        keyword_parts = keyword_lower.split()
        if len(keyword_parts) > 1:
            parts_found = sum(
                1 for part in keyword_parts
                if part in candidate_text or part in candidate_answer.lower()
            )
            if parts_found == len(keyword_parts):
                matched.append(keyword)
                continue

        # Check 3: SBERT semantic similarity against each sentence
        # Compare keyword to individual sentences, not the whole answer
        keyword_embedding = sbert.encode(keyword, convert_to_tensor=True)
        similarities = util.cos_sim(keyword_embedding, sentence_embeddings)[0]
        max_sim: float = similarities.max().item()

        if max_sim >= settings.KEYWORD_SIMILARITY_THRESHOLD:
            matched.append(keyword)
        else:
            missing.append(keyword)

    coverage = len(matched) / len(keywords) if keywords else 1.0
    return max(0.0, min(1.0, coverage)), _clean_missing_keywords(missing)
