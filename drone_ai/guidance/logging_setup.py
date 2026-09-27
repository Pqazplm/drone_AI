"""
Logging configuration for the guidance module.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

GUIDANCE_LOG = logging.getLogger("marker_guidance")


def configure_logging(path: str = "logs/marker_guidance.log"):
    """Configure the guidance logger without replacing application handlers."""
    if GUIDANCE_LOG.handlers:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(message)s"
    )
    console = logging.StreamHandler()
    disk = RotatingFileHandler(
        path, maxBytes=5_000_000, backupCount=4, encoding="utf-8"
    )
    for handler in (console, disk):
        handler.setFormatter(formatter)
        GUIDANCE_LOG.addHandler(handler)
    GUIDANCE_LOG.setLevel(logging.INFO)
    GUIDANCE_LOG.propagate = False
