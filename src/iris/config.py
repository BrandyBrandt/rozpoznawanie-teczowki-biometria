"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris.errors import ConfigError, InputDataError


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the pipeline."""

    raw: dict[str, Any]
    config_path: Path
    input_dir: Path
    output_dir: Path
    log_level: str


def load_config(config_path: Path) -> AppConfig:
    """Load JSON config and validate essential fields."""
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ConfigError("Only JSON config files are supported.")

    try:
        config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file: {exc}") from exc

    paths = config_raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("Missing or invalid 'paths' section in config.")

    input_dir_value = paths.get("input_dir")
    output_dir_value = paths.get("output_dir")
    if not isinstance(input_dir_value, str) or not input_dir_value.strip():
        raise ConfigError("Config key 'paths.input_dir' must be a non-empty string.")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise ConfigError("Config key 'paths.output_dir' must be a non-empty string.")

    runtime = config_raw.get("runtime", {})
    log_level_value = runtime.get("log_level", "INFO")
    if not isinstance(log_level_value, str):
        raise ConfigError("Config key 'runtime.log_level' must be a string.")

    base_dir = config_path.parent.parent
    input_dir = (base_dir / input_dir_value).resolve()
    output_dir = (base_dir / output_dir_value).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise InputDataError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise InputDataError(f"Input path is not a directory: {input_dir}")

    return AppConfig(
        raw=config_raw,
        config_path=config_path.resolve(),
        input_dir=input_dir,
        output_dir=output_dir,
        log_level=log_level_value,
    )
