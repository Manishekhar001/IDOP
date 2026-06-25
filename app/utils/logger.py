import datetime
import json
import logging
import sys
from functools import lru_cache


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO") -> None:
    try:
        from app.config import get_settings

        use_json = get_settings().log_format == "json"
    except Exception:
        use_json = False

    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    for noisy_lib in (
        "httpx",
        "httpcore",
        "openai",
        "qdrant_client",
        "urllib3",
        "groq",
        "langgraph",
    ):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


@lru_cache
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
