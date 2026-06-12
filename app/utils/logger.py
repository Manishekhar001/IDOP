import logging
import sys
from functools import lru_cache


def setup_logging(log_level: str = "INFO") -> None:
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
