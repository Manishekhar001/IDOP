import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure application-wide logging with console and rotating file handlers.
    """
    # Create main application logger
    logger = logging.getLogger("idop_app")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Rotating File Handler (app.log)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # Error Rotating File Handler (error.log)
    error_handler = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)

    # Formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(simple_formatter)
    file_handler.setFormatter(detailed_formatter)
    error_handler.setFormatter(detailed_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    # Optional CloudWatch Logs Integration via Watchtower
    import os

    cloudwatch_group = os.getenv("CLOUDWATCH_LOG_GROUP")
    if cloudwatch_group:
        try:
            import watchtower

            cw_handler = watchtower.CloudWatchLogHandler(
                log_group_name=cloudwatch_group,
                log_stream_name=os.getenv("CLOUDWATCH_LOG_STREAM", "idop-api-stream"),
                send_interval=10,
                create_log_group=True,
            )
            cw_handler.setLevel(logging.INFO)
            cw_handler.setFormatter(detailed_formatter)
            logger.addHandler(cw_handler)
            logger.info(
                f"AWS CloudWatch logs handler added successfully to group: '{cloudwatch_group}'"
            )
        except Exception as cw_err:
            logger.warning(
                f"Could not initialize AWS CloudWatch logging stream: {cw_err}"
            )

    # Suppress verbose loggers from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    return logger


def get_logger(name: str = "idop_app") -> logging.Logger:
    """
    Get a logger instance for a specific module.
    """
    return logging.getLogger(name)
