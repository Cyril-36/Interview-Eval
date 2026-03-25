import logging
import threading
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
import spacy

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # type: ignore[attr-defined]
        return cls._instance

    def __init__(self, settings: Settings | None = None):
        if self._initialized:  # type: ignore[has-type]
            return
        self._settings = settings
        self._sbert: SentenceTransformer | None = None
        self._nli_tokenizer: object | None = None
        self._nli_model: object | None = None
        self._keybert: KeyBERT | None = None
        self._spacy_nlp: object | None = None
        self._device: str | None = None
        self._load_lock = threading.RLock()
        self._initialized = True

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._settings.get_device()  # type: ignore[union-attr]
        return self._device

    @property
    def sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            with self._load_lock:
                if self._sbert is None:
                    logger.info(f"Loading SBERT: {self._settings.SBERT_MODEL}")
                    self._sbert = SentenceTransformer(
                        self._settings.SBERT_MODEL, device=self.device
                    )
        return self._sbert

    @property
    def nli_tokenizer(self):
        if self._nli_tokenizer is None:
            with self._load_lock:
                if self._nli_tokenizer is None:
                    logger.info(f"Loading NLI tokenizer: {self._settings.NLI_MODEL}")
                    self._nli_tokenizer = AutoTokenizer.from_pretrained(
                        self._settings.NLI_MODEL
                    )
        return self._nli_tokenizer

    @property
    def nli_model(self):
        if self._nli_model is None:
            with self._load_lock:
                if self._nli_model is None:
                    logger.info(f"Loading NLI model: {self._settings.NLI_MODEL}")
                    self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                        self._settings.NLI_MODEL
                    )
                    if self.device != "cpu":
                        self._nli_model = self._nli_model.to(self.device)
                    self._nli_model.eval()
        return self._nli_model

    @property
    def keybert(self) -> KeyBERT:
        if self._keybert is None:
            with self._load_lock:
                if self._keybert is None:
                    logger.info("Loading KeyBERT (reusing SBERT)")
                    self._keybert = KeyBERT(model=self.sbert)
        return self._keybert

    @property
    def spacy_nlp(self):
        if self._spacy_nlp is None:
            with self._load_lock:
                if self._spacy_nlp is None:
                    logger.info(f"Loading spaCy: {self._settings.SPACY_MODEL}")
                    self._spacy_nlp = spacy.load(self._settings.SPACY_MODEL)
        return self._spacy_nlp

    def load_all(self):
        """Pre-warm all models. Safe to skip; models load lazily on first access."""
        logger.info(f"Pre-warming models on device: {self.device}")
        _ = self.sbert
        _ = self.nli_tokenizer
        _ = self.nli_model
        _ = self.keybert
        _ = self.spacy_nlp
        logger.info("All models pre-warmed successfully.")

    @classmethod
    def reset(cls):
        cls._instance = None


def get_registry() -> ModelRegistry:
    return ModelRegistry()
