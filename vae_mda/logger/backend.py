"""Local-first experiment logging with an optional ClearML adapter."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class LocalLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figure_dir = output_dir / "figures"
        self.metrics_path = output_dir / "metrics.csv"
        self.summary_path = output_dir / "summary.json"
        self._summary: dict[str, Any] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

    def report_scalar(self, *, title, series, value, iteration, **_):
        new_file = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("iteration", "title", "series", "value")
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "iteration": int(iteration),
                    "title": str(title),
                    "series": str(series),
                    "value": _json_value(value),
                }
            )

    def report_single_value(self, *, name, value, **_):
        self._summary[str(name)] = _json_value(value)
        self.summary_path.write_text(
            json.dumps(self._summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    def report_matplotlib_figure(self, *, title, series, figure, iteration, **_):
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{int(iteration):04d}_{_safe_name(str(title))}_{_safe_name(str(series))}.png"
        )
        figure.savefig(self.figure_dir / filename, dpi=120, bbox_inches="tight")


class LocalTask:
    def __init__(self, *, project_name: str, task_name: str, output_root: Path):
        self.output_dir = output_root / _safe_name(project_name) / _safe_name(task_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = LocalLogger(self.output_dir)
        self._config_index = 0

    def connect(self, config):
        label = ("run", "model", "data")[min(self._config_index, 2)]
        self._config_index += 1
        (self.output_dir / f"{label}_config.json").write_text(
            json.dumps(_json_value(config), indent=2, sort_keys=True), encoding="utf-8"
        )
        return config

    def get_logger(self):
        return self._logger

    def close(self):
        return None


def init_task(config: dict[str, Any], *, project_name: str, task_name: str):
    from vae_mda.env import clearml_is_configured, load_dotenv

    load_dotenv()

    def _local():
        output_root = Path(config.get("output_root", "outputs"))
        return LocalTask(
            project_name=project_name,
            task_name=task_name,
            output_root=output_root,
        )

    backend = str(config.get("logger", "auto")).lower()
    if backend in ("", "auto"):
        backend = "clearml" if clearml_is_configured() else "local"
    if backend == "local":
        return _local()
    if backend == "clearml":
        if not clearml_is_configured():
            print(
                "ClearML logging requested but no credentials were found; set "
                "the CLEARML_* variables in .env (or run clearml-init). Logging "
                "locally to outputs/ instead."
            )
            return _local()
        try:
            from clearml import Task
        except ImportError:
            print(
                "ClearML credentials were found but the optional clearml package "
                "is not installed; run 'python -m pip install clearml' to enable "
                "online logging. Logging locally to outputs/ instead."
            )
            return _local()
        return Task.init(project_name=project_name, task_name=task_name)
    raise ValueError(f"Unknown logger backend: {backend!r}")
