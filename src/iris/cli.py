"""CLI parsing for project commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="iris-pipeline",
        description="Iris recognition pipeline for biomedical engineering project.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.json"),
        help="Path to JSON configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check-env",
        help="Validate configuration and required input directory.",
    )
    subparsers.add_parser(
        "run",
        help="Run the full pipeline (stages 2-8).",
    )
    return parser
