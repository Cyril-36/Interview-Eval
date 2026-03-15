import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
import spacy
import torch

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Settings | None = None):
        if self._initialized:
            return
        self._settings = settings
        self._sbert = None
        self._nli_tokenizer = None
        self._nli_model = None
        self._keybert = None
        self._spacy_nlp = None
        self._device = None
        self._initialized = True

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._settings.get_device()
        return self._device

    @property
    def sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self._sbert

    @property
    def nli_tokenizer(self):
        if self._nli_tokenizer is None:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self._nli_tokenizer

    @property
    def nli_model(self):
        if self._nli_model is None:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self._nli_model

    @property
    def keybert(self) -> KeyBERT:
        if self._keybert is None:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self._keybert

    @property
    def spacy_nlp(self):
        if self._spacy_nlp is None:
            raise RuntimeError("Models not loaded. Call load_all() first.")
        return self._spacy_nlp

    def load_all(self):
        logger.info(f"Loading models on device: {self.device}")

        logger.info(f"Loading SBERT: {self._settings.SBERT_MODEL}")
        self._sbert = SentenceTransformer(
            self._settings.SBERT_MODEL, device=self.device
        )

        logger.info(f"Loading NLI: {self._settings.NLI_MODEL}")
        self._nli_tokenizer = AutoTokenizer.from_pretrained(self._settings.NLI_MODEL)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self._settings.NLI_MODEL
        )
        if self.device != "cpu":
            self._nli_model = self._nli_model.to(self.device)
        self._nli_model.eval()

        logger.info("Loading KeyBERT (reusing SBERT)")
        self._keybert = KeyBERT(model=self._sbert)

        logger.info(f"Loading spaCy: {self._settings.SPACY_MODEL}")
        self._spacy_nlp = spacy.load(self._settings.SPACY_MODEL)

        logger.info("All models loaded successfully.")

    @classmethod
    def reset(cls):
        cls._instance = None


def get_registry() -> ModelRegistry:
    return ModelRegistry()
