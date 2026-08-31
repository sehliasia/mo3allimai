import logging

from app.core.logging import (
    ARABIC_REVIEW_LOGGER_NAME,
    _ARABIC_REVIEW_HANDLER_MARKER,
    configure_arabic_review_console_logging,
)


def test_arabic_review_logger_accepts_info_and_installs_one_dedicated_console_handler():
    review_logger = logging.getLogger(ARABIC_REVIEW_LOGGER_NAME)
    previous_handlers = list(review_logger.handlers)
    previous_level = review_logger.level
    previous_propagate = review_logger.propagate
    review_logger.handlers = [
        handler for handler in review_logger.handlers
        if not getattr(handler, _ARABIC_REVIEW_HANDLER_MARKER, False)
    ]

    try:
        configure_arabic_review_console_logging()
        configure_arabic_review_console_logging()

        dedicated_handlers = [
            handler for handler in review_logger.handlers
            if getattr(handler, _ARABIC_REVIEW_HANDLER_MARKER, False)
        ]
        assert review_logger.getEffectiveLevel() == logging.INFO
        assert review_logger.propagate is False
        assert len(dedicated_handlers) == 1
        assert isinstance(dedicated_handlers[0], logging.StreamHandler)
        assert not isinstance(dedicated_handlers[0], logging.FileHandler)
    finally:
        review_logger.handlers = previous_handlers
        review_logger.setLevel(previous_level)
        review_logger.propagate = previous_propagate
