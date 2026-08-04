"""Summaries for local assimilation metrics emitted by the public logger."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def moving_block_skill(
    model: np.ndarray,
    analysis: np.ndarray,
    *,
    block_size: int = 14,
    samples: int = 5000,
    seed: int = 20260518,
) -> tuple[float, float, float, float]:
    finite = np.isfinite(model) & np.isfinite(analysis)
    model = np.asarray(model[finite], dtype=float)
    analysis = np.asarray(analysis[finite], dtype=float)
    if not len(model) or float(model.mean()) == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    point = 1.0 - float(analysis.mean()) / float(model.mean())
    if len(model) < block_size or samples < 2:
        return point, float("nan"), float("nan"), float("nan")

    generator = np.random.default_rng(seed)
    starts = np.arange(len(model) - block_size + 1)
    blocks = math.ceil(len(model) / block_size)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = generator.choice(starts, size=blocks, replace=True)
        sample = np.concatenate(
            [np.arange(start, start + block_size) for start in chosen]
        )[: len(model)]
        denominator = float(model[sample].mean())
        values[index] = (
            float("nan")
            if denominator == 0.0
            else 1.0 - float(analysis[sample].mean()) / denominator
        )
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return point, float("nan"), float("nan"), float("nan")
    low, high = np.percentile(values, (2.5, 97.5))
    return point, float(values.std()), float(low), float(high)


def load_local_metrics(path: Path):
    grouped = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            series = row["series"]
            if series not in {"model", "assim"}:
                continue
            title = row["title"]
            if not title.endswith(("_mae", "_rmse")):
                continue
            grouped[title][series][int(row["iteration"])] = float(row["value"])
    return grouped


def summarize_files(
    specifications: Iterable[tuple[str, Path]],
    *,
    block_size: int = 14,
    samples: int = 5000,
    seed: int = 20260518,
):
    rows = []
    for run_name, path in specifications:
        for metric, series in sorted(load_local_metrics(path).items()):
            common = sorted(set(series["model"]) & set(series["assim"]))
            model = np.asarray([series["model"][key] for key in common], dtype=float)
            analysis = np.asarray([series["assim"][key] for key in common], dtype=float)
            point, std, low, high = moving_block_skill(
                model,
                analysis,
                block_size=block_size,
                samples=samples,
                seed=seed,
            )
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "n_cases": len(common),
                    "model_mean": float(np.mean(model)) if len(model) else float("nan"),
                    "analysis_mean": float(np.mean(analysis)) if len(analysis) else float("nan"),
                    "skill": point,
                    "bootstrap_std": std,
                    "ci_2_5": low,
                    "ci_97_5": high,
                }
            )
    return rows


def write_summary(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "run",
        "metric",
        "n_cases",
        "model_mean",
        "analysis_mean",
        "skill",
        "bootstrap_std",
        "ci_2_5",
        "ci_97_5",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
