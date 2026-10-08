"""
src/utils/logger.py
-------------------
Centralized logging setup.
Call get_logger(__name__) at the top of every module.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name:     Logger name — use __name__ in every module.
        log_file: Optional path to write logs to a file as well.
        level:    Logging level. Default: INFO.

    Returns:
        Configured logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting preprocessing...")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 to avoid cp1252 errors on Windows
    console_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
