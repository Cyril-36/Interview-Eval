import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models_loader import ModelRegistry
from app.routers import questions, evaluation, session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting model loading...")
    registry = ModelRegistry(settings)
    registry.load_all()
    logger.info("All models ready. Server is accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Interview Evaluation System",
    description="Hybrid NLP-based interview question generation and answer evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(questions.router, prefix="/api", tags=["Questions"])
app.include_router(evaluation.router, prefix="/api", tags=["Evaluation"])
app.include_router(session.router, prefix="/api", tags=["Session"])


@app.get("/health")
async def health():
    return {"status": "ok"}
