from app.models_loader import get_registry


def preprocess(text: str) -> str:
    nlp = get_registry().spacy_nlp
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


def preprocess_tokens(text: str) -> list[str]:
    nlp = get_registry().spacy_nlp
    doc = nlp(text.lower())
    return [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
