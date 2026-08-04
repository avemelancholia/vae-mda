"""YAML configuration loading shared by the public entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_REFERENCES = {"data_config", "model_config"}


class ConfigError(ValueError):
    """Raised when a public configuration cannot be resolved safely."""


def _resolve_reference(value: str, config_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = (PROJECT_ROOT / path, config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((PROJECT_ROOT / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, expand, and validate one YAML mapping."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration must contain a YAML mapping: {config_path}")

    config = payload
    for key in _CONFIG_REFERENCES:
        if key in config:
            config[key] = _resolve_reference(str(config[key]), config_path)
    config["_config_path"] = str(config_path)
    return config


def add_common_arguments(parser: argparse.ArgumentParser, *, logging: bool = True) -> None:
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--device", help="Override the configured torch device")
    if logging:
        parser.add_argument(
            "--logger",
            choices=("auto", "local", "clearml"),
            help="Experiment logging backend (auto uses ClearML when .env credentials exist)",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration dependencies without loading research data",
    )


def load_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    from vae_mda.env import load_dotenv

    load_dotenv()
    config = load_config(args.config)
    if getattr(args, "device", None):
        config["device"] = args.device
    if getattr(args, "logger", None):
        config["logger"] = args.logger
    config.setdefault("logger", "auto")
    return config


def validate_references(config: dict[str, Any]) -> list[Path]:
    """Load nested configs and return their resolved paths."""

    paths: list[Path] = []
    for key in sorted(_CONFIG_REFERENCES):
        if key not in config:
            continue
        path = Path(config[key])
        load_config(path)
        paths.append(path)
    return paths
