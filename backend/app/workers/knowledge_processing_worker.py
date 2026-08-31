"""Run the persistent Knowledge Base worker with one process/concurrency slot."""

from __future__ import annotations

import logging
import time

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.services.knowledge_processing_queue import KnowledgeProcessingQueue

logging.basicConfig(level=logging.INFO)


def main() -> None:
    queue = KnowledgeProcessingQueue()
    settings = get_settings()
    with SessionLocal() as db:
        recovered = queue.recover_stale_jobs(db)
        logging.info("knowledge_worker_started concurrency=1 stale_jobs_recovered=%s", recovered)
    while True:
        with SessionLocal() as db:
            job = queue.process_next(db)
        if job is None:
            time.sleep(settings.knowledge_worker_poll_seconds)


if __name__ == "__main__":
    main()
