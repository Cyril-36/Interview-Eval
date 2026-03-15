import torch
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""

    # Model names
    SBERT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    BART_MODEL: str = "facebook/bart-large-cnn"
    SPACY_MODEL: str = "en_core_web_sm"

    # Scoring weights (optimized via grid search: Pearson r=0.8862)
    SBERT_WEIGHT: float = 0.45
    NLI_WEIGHT: float = 0.05
    KEYWORD_WEIGHT: float = 0.50

    # Thresholds
    DIVERSITY_THRESHOLD: float = 0.6
    KEYWORD_SIMILARITY_THRESHOLD: float = 0.75
    KEYWORD_TOP_N: int = 10

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
