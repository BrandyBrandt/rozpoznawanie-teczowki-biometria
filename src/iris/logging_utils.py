"""Logging helpers for consistent console output."""

from __future__ import annotations

import logging


def setup_logging(level: str) -> None:
    """Configure root logger for the CLI application."""
    normalized_level = level.upper()
    if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        normalized_level = "INFO"

    logging.basicConfig(
        level=getattr(logging, normalized_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
