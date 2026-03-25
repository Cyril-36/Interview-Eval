from app.services.scoring.keyword_scorer import _clean_missing_keywords


def test_clean_missing_keywords_dedupes_and_removes_noisy_bigrams(monkeypatch):
    lemma_map = {
        "imputation using": ["imputation"],
        "imputation imputation": ["imputation", "imputation"],
        "median imputation": ["median", "imputation"],
        "feature scaling": ["feature", "scaling"],
    }

    monkeypatch.setattr(
        "app.services.scoring.keyword_scorer.preprocess_tokens",
        lambda text: lemma_map[text],
    )

    cleaned = _clean_missing_keywords([
        "imputation using",
        "imputation imputation",
        "median imputation",
        "feature scaling",
    ])

    assert cleaned == ["median imputation", "feature scaling"]
