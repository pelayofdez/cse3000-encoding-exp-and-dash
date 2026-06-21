"""Lightweight logging setup shared by all entry points."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "encoding_experiments", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
