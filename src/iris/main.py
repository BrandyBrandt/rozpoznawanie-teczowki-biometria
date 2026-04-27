"""Entry point for the iris recognition CLI."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Support running this file directly from IDE ("Run file")
# when package root ("src") is not on PYTHONPATH.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from iris.cli import build_parser
from iris.config import load_config
from iris.errors import ConfigError, InputDataError, IrisPipelineError
from iris.logging_utils import setup_logging
from iris.pipeline import (
    run_stage2_preprocessing,
    run_stage3_pupil_segmentation,
    run_stage4_iris_segmentation,
    run_stage5_normalization,
    run_stage6_encoding,
    run_stage7_matching,
    run_stage8_evaluation,
)

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Parse arguments and execute a selected command."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        app_config = load_config(args.config)
        setup_logging(app_config.log_level)

        LOGGER.info("Loaded config: %s", app_config.config_path)
        LOGGER.info("Input directory: %s", app_config.input_dir)
        LOGGER.info("Output directory: %s", app_config.output_dir)

        if args.command == "check-env":
            LOGGER.info("Environment validation finished successfully.")
            return 0

        if args.command == "run":
            run_stage2_preprocessing(app_config)
            run_stage3_pupil_segmentation(app_config)
            run_stage4_iris_segmentation(app_config)
            run_stage5_normalization(app_config)
            run_stage6_encoding(app_config)
            run_stage7_matching(app_config)
            run_stage8_evaluation(app_config)
            return 0

        raise IrisPipelineError(f"Unsupported command: {args.command}")
    except (ConfigError, InputDataError, IrisPipelineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Defensive catch for unexpected failures.
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
