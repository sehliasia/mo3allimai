import os
from urllib.parse import urlsplit

# Docling's OCR stack does not need TorchInductor compilation on Windows.
# This must precede every FastAPI/application import, which may load Docling lazily.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import admin, assistant, auth, health, teacher
from app.core.config import get_settings
from app.core.logging import (
    configure_arabic_review_console_logging,
    configure_embedding_provider_console_logging,
    configure_retrieval_pipeline_console_logging,
    configure_teacher_library_ingestion_console_logging,
)
from app.services.retrieval_pipeline import effective_retrieval_pipeline, validate_retrieval_pipeline
import logging

app = FastAPI(title="Mo3allimAI API", version="1.0.0")
settings = get_settings()
validate_retrieval_pipeline(settings)
configure_arabic_review_console_logging()
configure_retrieval_pipeline_console_logging()
configure_teacher_library_ingestion_console_logging()
configure_embedding_provider_console_logging()
logging.getLogger(__name__).info("retrieval_pipeline_config=%s", effective_retrieval_pipeline(settings))


def cors_origins(frontend_url: str) -> list[str]:
    """Keep local Vite's localhost and loopback origins equivalent.

    Credentials require explicit origins; this intentionally never uses "*".
    """
    origin = frontend_url.rstrip("/")
    parsed = urlsplit(origin)
    origins = [origin]
    if parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1"}:
        alternate_host = "127.0.0.1" if parsed.hostname == "localhost" else "localhost"
        port = f":{parsed.port}" if parsed.port else ""
        origins.append(f"{parsed.scheme}://{alternate_host}{port}")
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(settings.frontend_url),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")
app.include_router(teacher.router, prefix="/api")
