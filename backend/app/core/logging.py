"""Small, application-owned logging configuration."""

from __future__ import annotations

import logging


ARABIC_REVIEW_LOGGER_NAME = "app.services.arabic_linguistic_review_service"
_ARABIC_REVIEW_HANDLER_MARKER = "_mo3allimai_arabic_review_console_handler"
RETRIEVAL_PIPELINE_LOGGER_NAME = "app.services.pedagogical_knowledge_service"
_RETRIEVAL_PIPELINE_HANDLER_MARKER = "_mo3allimai_retrieval_pipeline_console_handler"
TEACHER_LIBRARY_INGESTION_LOGGER_NAME = "app.services.teacher_library_ingestion_service"
_TEACHER_LIBRARY_INGESTION_HANDLER_MARKER = "_mo3allimai_teacher_library_ingestion_console_handler"
EMBEDDING_PROVIDER_LOGGER_NAME = "app.services.embedding_providers.sentence_transformer_provider"
_EMBEDDING_PROVIDER_HANDLER_MARKER = "_mo3allimai_embedding_provider_console_handler"


def configure_arabic_review_console_logging() -> logging.Logger:
    """Show Arabic review diagnostics once in Uvicorn's normal console.

    Uvicorn intentionally leaves application logger levels alone.  This
    dedicated non-propagating handler avoids enabling verbose third-party
    logging and avoids duplicate records through Uvicorn's root handlers.
    """
    review_logger = logging.getLogger(ARABIC_REVIEW_LOGGER_NAME)
    review_logger.setLevel(logging.INFO)
    review_logger.propagate = False

    if not any(getattr(handler, _ARABIC_REVIEW_HANDLER_MARKER, False) for handler in review_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _ARABIC_REVIEW_HANDLER_MARKER, True)
        review_logger.addHandler(handler)

    return review_logger


def configure_retrieval_pipeline_console_logging() -> logging.Logger:
    """Emit only the compact H7 pipeline trace once per request."""
    pipeline_logger = logging.getLogger(RETRIEVAL_PIPELINE_LOGGER_NAME)
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.propagate = False
    if not any(getattr(handler, _RETRIEVAL_PIPELINE_HANDLER_MARKER, False) for handler in pipeline_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _RETRIEVAL_PIPELINE_HANDLER_MARKER, True)
        pipeline_logger.addHandler(handler)
    return pipeline_logger


def configure_teacher_library_ingestion_console_logging() -> logging.Logger:
    """Emit one compact, stage-by-stage teacher ingestion trace to Uvicorn."""
    ingestion_logger = logging.getLogger(TEACHER_LIBRARY_INGESTION_LOGGER_NAME)
    ingestion_logger.setLevel(logging.INFO)
    ingestion_logger.propagate = False
    if not any(getattr(handler, _TEACHER_LIBRARY_INGESTION_HANDLER_MARKER, False) for handler in ingestion_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _TEACHER_LIBRARY_INGESTION_HANDLER_MARKER, True)
        ingestion_logger.addHandler(handler)
    return ingestion_logger


def configure_embedding_provider_console_logging() -> logging.Logger:
    """Expose bounded model-load and batch timings without third-party noise."""
    embedding_logger = logging.getLogger(EMBEDDING_PROVIDER_LOGGER_NAME)
    embedding_logger.setLevel(logging.INFO)
    embedding_logger.propagate = False
    if not any(getattr(handler, _EMBEDDING_PROVIDER_HANDLER_MARKER, False) for handler in embedding_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _EMBEDDING_PROVIDER_HANDLER_MARKER, True)
        embedding_logger.addHandler(handler)
    return embedding_logger
