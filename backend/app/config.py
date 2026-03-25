import torch
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    CLAIM_EXTRACTION_MODE: str = "regex"
    CLAIM_EXTRACTION_MODEL: str = "llama-3.1-8b-instant"

    # Model names
    SBERT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    BART_MODEL: str = "facebook/bart-large-cnn"
    SPACY_MODEL: str = "en_core_web_sm"

    # Scoring weights (4-signal hybrid: NLP + LLM-as-judge)
    # Optimized via 4-signal grid search (Pearson=0.8864)
    SBERT_WEIGHT: float = 0.40
    NLI_WEIGHT: float = 0.10
    KEYWORD_WEIGHT: float = 0.30
    LLM_WEIGHT: float = 0.20

    # Behavioral question weights (STAR pipeline)
    # Downweights NLI/keyword (poor proxies for story-based answers),
    # upweights LLM judge (STAR rubric) and semantic similarity.
    BEHAVIORAL_SBERT_WEIGHT: float = 0.25
    BEHAVIORAL_NLI_WEIGHT: float = 0.05
    BEHAVIORAL_KEYWORD_WEIGHT: float = 0.10
    BEHAVIORAL_LLM_WEIGHT: float = 0.60

    # Claim-based technical scoring (v2 pipeline)
    CLAIM_SBERT_WEIGHT: float = 0.25
    CLAIM_NLI_WEIGHT: float = 0.10
    CLAIM_KEYWORD_WEIGHT: float = 0.05
    CLAIM_COVERAGE_WEIGHT: float = 0.50
    CLAIM_LLM_WEIGHT: float = 0.10
    CLAIM_MATCH_THRESHOLD: float = 0.62
    CLAIM_SOFT_MARGIN: float = 0.15
    CLAIM_CONTRADICTION_PENALTY: float = 0.20
    CLAIM_MAX_CLAIMS: int = 6

    # Thresholds
    DIVERSITY_THRESHOLD: float = 0.6
    KEYWORD_SIMILARITY_THRESHOLD: float = 0.55
    KEYWORD_TOP_N: int = 8

    # Device
    DEVICE: str = ""

    # CORS
    FRONTEND_URL: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "extra": "ignore"}

    def get_device(self) -> str:
        if self.DEVICE:
            return self.DEVICE
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
