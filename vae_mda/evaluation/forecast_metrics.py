#!/usr/bin/env python3
"""Recalculate forecast metrics with the actual sparse SRAL observation mask.

The original forecast notebook transformed the sparse SRAL/AMSR2 target, filled
its off-track NaNs with zero, and then evaluated over every ocean pixel.  This
script preserves the existing dense-AMSR2 convention, maps non-finite ocean
forecast fill values to zero ice concentration, and evaluates the sparse target
only where the transformed SRAL-conditioned target is finite:

    ocean & finite(sral_disc[11](target))

It uses the exact forecast files and time indices stored in the existing wide
metric manifests.  No assimilation, VAE, or NEMO experiment is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vae_mda.data.transform.sral_transform import sral_disc
from .forecast_metric_case_selection import select_observation_cases
from .forecast_metric_overlap_selection import select_forecast_overlap_cases


FIELD = "siconc"
TARGET_DENSE = "amsr2_dense"
TARGET_SPARSE = "sral_corrected_sparse"
TARGETS = (TARGET_DENSE, TARGET_SPARSE)
METRICS = ("mae", "rmse", "mse", "iiee")
CASE_COLUMNS = ("target", "lead_hours", "init_date", "valid_date")
KEY_COLUMNS = ("lead_hours", "init_date", "valid_date")

DENSE_AMSR2_CHANNEL = 7
SRAL_CLASS_CHANNEL = 1
SRAL_TRANSFORM_INDEX = 11
SRAL_TRANSFORM = sral_disc[SRAL_TRANSFORM_INDEX]["transform"]
SRAL_TRANSFORM_NAME = sral_disc[SRAL_TRANSFORM_INDEX]["name"]

DEFAULT_LAND_MASK = Path(os.environ.get("VAE_MDA_LAND_MASK", "data/land_mask.npy"))
DEFAULT_MAX_CASES_PER_LEAD = 200
DEFAULT_LEADS = (0, 24, 48, 72)
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 14
DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 20260518
DEFAULT_CLEARML_PROJECT = "paper_vae_borey/forecast_validation"
DEFAULT_CLEARML_TASK = "corrected_sparse_sral_forecast_metrics_v2"
DEFAULT_ARTIFACT_NAME = "forecast_sral_metrics_corrected_v3"
DEFAULT_TAGS = ("paper", "forecast", "sral", "mask-fix", "metrics-only")

RUN_OUTPUT_DIRS = {
    "ice_only": "forecast_metrics_ice_only",
    "phys_corr": "ice_ocean_phys_corr",
}
RUN_LABELS = {
    "ice_only": "Ice-only",
    "phys_corr": "Ice + physical ocean correction",
}


class ValidationError(RuntimeError):
    """Raised when an input or scientific invariant is not satisfied."""


@dataclass(frozen=True)
class PathMap:
    old: str
    new: str


@dataclass
class PreflightResult:
    records_by_run: dict[str, dict[tuple[int, str, str], dict[str, Any]]]
    candidate_keys: list[tuple[int, str, str]]
    selected_keys: list[tuple[int, str, str]]
    available_by_lead: dict[int, int]
    observed_by_lead: dict[int, int]
    total_by_lead: dict[int, int]
    path_report: pd.DataFrame
    forecast_overlap_report: pd.DataFrame
    missing_report: pd.DataFrame
    target_report: pd.DataFrame
    land_mask_path: Path


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def default_metrics_root() -> Path:
    output_root = Path(os.environ.get("VAE_MDA_OUTPUT_ROOT", project_dir().parents[1] / "outputs"))
    candidates = (output_root / "metrics",)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def default_manifest_specs() -> list[str]:
    root = default_metrics_root()
    return [
        f"ice_only={root / 'forecast_metrics_ice_only' / 'forecast_metrics_common_daily_wide.csv'}",
        f"phys_corr={root / 'ice_ocean_phys_corr' / 'forecast_metrics_common_daily_wide.csv'}",
    ]


def parse_assignment(raw: str, label: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"{label} must have the form NAME=VALUE: {raw!r}")
    name, value = raw.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{label} must have a non-empty name and value: {raw!r}")
    return name, value


def parse_manifest_specs(values: Sequence[str] | None) -> dict[str, Path]:
    specs = values if values else default_manifest_specs()
    parsed: dict[str, Path] = {}
    for raw in specs:
        name, value = parse_assignment(raw, "--manifest")
        if name in parsed:
            raise ValidationError(f"duplicate manifest name: {name}")
        parsed[name] = Path(value).expanduser().resolve()
    if len(parsed) < 2:
        raise ValidationError("at least two comparison manifests are required")
    return parsed


def parse_path_maps(values: Sequence[str] | None) -> list[PathMap]:
    maps: list[PathMap] = []
    for raw in values or ():
        old, new = parse_assignment(raw, "--path-map")
        old = old.rstrip("/") or "/"
        new = new.rstrip("/") or "/"
        maps.append(PathMap(old=old, new=new))
    return sorted(maps, key=lambda item: len(item.old), reverse=True)


def remap_path(raw: str | Path, path_maps: Sequence[PathMap]) -> Path:
    value = str(raw)
    for mapping in path_maps:
        if value == mapping.old:
            return Path(mapping.new)
        prefix = mapping.old.rstrip("/") + "/"
        if value.startswith(prefix):
            suffix = value[len(mapping.old.rstrip("/")) :]
            return Path(mapping.new.rstrip("/") + suffix)
    return Path(value)


def _equal_values(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left) == str(right)


def _normalise_date(value: Any, column: str) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception as exc:
        raise ValidationError(f"invalid {column} value {value!r}") from exc


def load_manifest(
    run_key: str,
    path: Path,
    leads: Sequence[int],
    path_maps: Sequence[PathMap],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    frame = pd.read_csv(path)
    required = {
        *CASE_COLUMNS,
        "forecast_file_assimilated",
        "forecast_file_baseline",
        "target_file_assimilated",
        "target_file_baseline",
        "time_index_assimilated",
        "time_index_baseline",
        *[f"{metric}_assimilated" for metric in METRICS],
        *[f"{metric}_baseline" for metric in METRICS],
        "n_pixels_assimilated",
        "n_pixels_baseline",
        *[
            f"{name}_{side}"
            for name in ("amsr2_channel", "sral_class_channel", "sral_transform_index", "sral_transform_name")
            for side in ("assimilated", "baseline")
        ],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValidationError(f"{path} is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["lead_hours"] = pd.to_numeric(frame["lead_hours"], errors="raise").astype(int)
    frame = frame[frame["lead_hours"].isin([int(lead) for lead in leads])].copy()
    frame["init_date"] = frame["init_date"].map(lambda value: _normalise_date(value, "init_date"))
    frame["valid_date"] = frame["valid_date"].map(lambda value: _normalise_date(value, "valid_date"))

    expected_manifest_values = {
        "amsr2_channel": DENSE_AMSR2_CHANNEL,
        "sral_class_channel": SRAL_CLASS_CHANNEL,
        "sral_transform_index": SRAL_TRANSFORM_INDEX,
        "sral_transform_name": SRAL_TRANSFORM_NAME,
    }
    for base_name, expected in expected_manifest_values.items():
        for side in ("assimilated", "baseline"):
            column = f"{base_name}_{side}"
            actual = frame[column]
            if isinstance(expected, int):
                matches = pd.to_numeric(actual, errors="coerce") == expected
            else:
                matches = actual.astype(str) == expected
            if not matches.all():
                unexpected = sorted(actual.loc[~matches].astype(str).unique())
                raise ValidationError(f"{path}: {column} must be {expected!r}, found {unexpected}")

    unexpected_targets = sorted(set(frame["target"].dropna()) - set(TARGETS))
    if unexpected_targets:
        raise ValidationError(f"{path} contains unexpected targets: {unexpected_targets}")

    duplicate_mask = frame.duplicated([*KEY_COLUMNS, "target"], keep=False)
    if duplicate_mask.any():
        sample = frame.loc[duplicate_mask, [*KEY_COLUMNS, "target"]].head().to_dict("records")
        raise ValidationError(f"{path} has duplicate case/target rows, for example: {sample}")

    dense = frame[frame["target"] == TARGET_DENSE].copy()
    sparse = frame[frame["target"] == TARGET_SPARSE].copy()
    dense_keys = set(map(tuple, dense[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    sparse_keys = set(map(tuple, sparse[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    if dense_keys != sparse_keys:
        raise ValidationError(
            f"{path} dense/sparse case mismatch: "
            f"dense_only={len(dense_keys - sparse_keys)}, sparse_only={len(sparse_keys - dense_keys)}"
        )

    sparse_lookup = sparse.set_index(list(KEY_COLUMNS))
    if not sparse_lookup.index.is_unique:
        raise ValidationError(f"{path} has duplicate sparse case keys")
    metadata_columns = [
        "forecast_file_assimilated",
        "forecast_file_baseline",
        "target_file_assimilated",
        "target_file_baseline",
        "time_index_assimilated",
        "time_index_baseline",
    ]
    metadata_columns.extend(
        f"{name}_{side}"
        for name in (
            "amsr2_channel",
            "sral_class_channel",
            "sral_transform_index",
            "sral_transform_name",
        )
        for side in ("assimilated", "baseline")
    )
    optional_metadata = [
        "time_value_assimilated",
        "time_value_baseline",
        "selection_method_assimilated",
        "selection_method_baseline",
    ]
    metadata_columns.extend(column for column in optional_metadata if column in frame.columns)
    for _, row in dense.iterrows():
        key = tuple(row[column] for column in KEY_COLUMNS)
        counterpart = sparse_lookup.loc[key]
        for column in metadata_columns:
            if not _equal_values(row[column], counterpart[column]):
                raise ValidationError(f"{path}: dense/sparse metadata differs for {key}, column {column}")

    for _, row in dense.iterrows():
        if not _equal_values(row["target_file_baseline"], row["target_file_assimilated"]):
            key = tuple(row[column] for column in KEY_COLUMNS)
            raise ValidationError(f"{path}: baseline/assimilated targets differ for {key}")

    dense["run_key"] = run_key
    dense["manifest_path"] = str(path)
    dense["forecast_file_baseline_source"] = dense["forecast_file_baseline"].astype(str)
    dense["forecast_file_assimilated_source"] = dense["forecast_file_assimilated"].astype(str)
    dense["target_file_source"] = dense["target_file_baseline"].astype(str)
    dense["forecast_file_baseline"] = dense["forecast_file_baseline_source"].map(
        lambda value: str(remap_path(value, path_maps))
    )
    dense["forecast_file_assimilated"] = dense["forecast_file_assimilated_source"].map(
        lambda value: str(remap_path(value, path_maps))
    )
    dense["target_file"] = dense["target_file_source"].map(lambda value: str(remap_path(value, path_maps)))
    dense["time_index_baseline"] = pd.to_numeric(dense["time_index_baseline"], errors="raise").astype(int)
    dense["time_index_assimilated"] = pd.to_numeric(
        dense["time_index_assimilated"], errors="raise"
    ).astype(int)
    return dense.sort_values(["lead_hours", "init_date", "valid_date"]).reset_index(drop=True)


def case_key(record: Mapping[str, Any]) -> tuple[int, str, str]:
    return int(record["lead_hours"]), str(record["init_date"]), str(record["valid_date"])


def _file_status(path: Path) -> tuple[bool, bool, int | None, int | None]:
    exists = path.is_file()
    readable = exists and os.access(path, os.R_OK)
    if not exists:
        return False, False, None, None
    stat = path.stat()
    return True, readable, int(stat.st_size), int(stat.st_mtime)


def build_preflight(
    manifests: Mapping[str, Path],
    leads: Sequence[int],
    max_cases_per_lead: int,
    path_maps: Sequence[PathMap],
    land_mask_path: Path,
) -> PreflightResult:
    frames = {
        run_key: load_manifest(run_key, path, leads, path_maps)
        for run_key, path in manifests.items()
    }
    records_by_run: dict[str, dict[tuple[int, str, str], dict[str, Any]]] = {}
    for run_key, frame in frames.items():
        records_by_run[run_key] = {
            case_key(record): record
            for record in frame.to_dict("records")
        }

    key_sets = [set(records) for records in records_by_run.values()]
    global_keys = set.intersection(*key_sets)
    if not global_keys:
        raise ValidationError("the manifests have no globally common forecast cases")

    first_run = next(iter(records_by_run))
    shared_columns = (
        "forecast_file_baseline",
        "target_file",
        "time_index_baseline",
    )
    for key in sorted(global_keys):
        reference = records_by_run[first_run][key]
        for run_key, records in records_by_run.items():
            candidate = records[key]
            for column in shared_columns:
                if not _equal_values(reference[column], candidate[column]):
                    raise ValidationError(
                        f"manifests disagree on shared {column} for case {key}: "
                        f"{first_run}={reference[column]!r}, {run_key}={candidate[column]!r}"
                    )

    remapped_land_mask = remap_path(land_mask_path, path_maps)
    path_refs: dict[tuple[str, str, str], dict[str, Any]] = {}

    def register_path(kind: str, run_key: str, source: str, resolved: str, key: Any) -> None:
        identity = (kind, run_key, resolved)
        row = path_refs.setdefault(
            identity,
            {
                "kind": kind,
                "run_key": run_key,
                "source_path": source,
                "resolved_path": resolved,
                "case_keys": set(),
            },
        )
        row["case_keys"].add(str(key))

    register_path("land_mask", "shared", str(land_mask_path), str(remapped_land_mask), "all")
    for key in sorted(global_keys):
        reference = records_by_run[first_run][key]
        register_path(
            "baseline_forecast",
            "shared",
            str(reference["forecast_file_baseline_source"]),
            str(reference["forecast_file_baseline"]),
            key,
        )
        register_path(
            "target",
            "shared",
            str(reference["target_file_source"]),
            str(reference["target_file"]),
            key,
        )
        for run_key, records in records_by_run.items():
            record = records[key]
            register_path(
                "assimilated_forecast",
                run_key,
                str(record["forecast_file_assimilated_source"]),
                str(record["forecast_file_assimilated"]),
                key,
            )

    path_rows: list[dict[str, Any]] = []
    status_by_path: dict[str, bool] = {}
    for row in path_refs.values():
        resolved = Path(row["resolved_path"])
        exists, readable, size, mtime = _file_status(resolved)
        status_by_path[str(resolved)] = bool(exists and readable)
        path_rows.append(
            {
                "kind": row["kind"],
                "run_key": row["run_key"],
                "source_path": row["source_path"],
                "resolved_path": row["resolved_path"],
                "exists": exists,
                "readable": readable,
                "size_bytes": size,
                "mtime_unix": mtime,
                "n_case_references": len(row["case_keys"]),
            }
        )
    path_report = pd.DataFrame(path_rows).sort_values(["kind", "run_key", "resolved_path"])

    land_ok = status_by_path.get(str(remapped_land_mask), False)
    if not land_ok:
        raise ValidationError(f"land mask is missing or unreadable: {remapped_land_mask}")

    total_by_lead: dict[int, int] = {}
    available_by_lead: dict[int, int] = {}
    candidate_keys: list[tuple[int, str, str]] = []
    selected_keys: list[tuple[int, str, str]] = []
    missing_rows: list[dict[str, Any]] = []
    for lead in [int(value) for value in leads]:
        lead_keys = sorted(
            (key for key in global_keys if key[0] == lead),
            key=lambda key: (key[1], key[2]),
        )
        total_by_lead[lead] = len(lead_keys)
        available: list[tuple[int, str, str]] = []
        for key in lead_keys:
            reference = records_by_run[first_run][key]
            requirements = [
                ("baseline_forecast", "shared", str(reference["forecast_file_baseline"])),
                ("target", "shared", str(reference["target_file"])),
            ]
            requirements.extend(
                (
                    "assimilated_forecast",
                    run_key,
                    str(records[key]["forecast_file_assimilated"]),
                )
                for run_key, records in records_by_run.items()
            )
            failures = [item for item in requirements if not status_by_path.get(item[2], False)]
            if failures:
                for kind, run_key, missing_path in failures:
                    missing_rows.append(
                        {
                            "lead_hours": key[0],
                            "init_date": key[1],
                            "valid_date": key[2],
                            "kind": kind,
                            "run_key": run_key,
                            "resolved_path": missing_path,
                            "error": "missing_or_unreadable",
                        }
                    )
            else:
                available.append(key)
        available_by_lead[lead] = len(available)
        candidate_keys.extend(available)
        if not available:
            raise ValidationError(
                f"lead {lead} h has no globally complete cases; metrics cannot be "
                "calculated for this lead"
            )
        selected_keys.extend(available[:max_cases_per_lead])

    missing_columns = [
        "lead_hours",
        "init_date",
        "valid_date",
        "kind",
        "run_key",
        "resolved_path",
        "error",
    ]
    missing_report = pd.DataFrame(missing_rows, columns=missing_columns)
    return PreflightResult(
        records_by_run=records_by_run,
        candidate_keys=sorted(candidate_keys, key=lambda key: (key[1], key[0], key[2])),
        selected_keys=sorted(selected_keys, key=lambda key: (key[1], key[0], key[2])),
        available_by_lead=available_by_lead,
        observed_by_lead={},
        total_by_lead=total_by_lead,
        path_report=path_report,
        missing_report=missing_report,
        target_report=pd.DataFrame(),
        forecast_overlap_report=pd.DataFrame(),
        land_mask_path=remapped_land_mask,
    )


def ensure_grid_shape(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == shape:
        return array
    if array.ndim == 2 and array.T.shape == shape:
        return array.T
    raise ValidationError(f"{name} has shape {array.shape}, expected {shape} (or its transpose)")


def field_for_dense_eval(array: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Reproduce the original dense-target full-ocean convention."""
    result = np.asarray(array, dtype=np.float32).copy()
    ocean = ~land_mask
    result[ocean & ~np.isfinite(result)] = 0.0
    result[land_mask] = np.nan
    return result


def target_views(target: np.ndarray, land_mask: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 2:
        target = target[None, ...]
    if target.ndim != 3:
        raise ValidationError(f"target must have shape (channel, y, x), got {target.shape}")
    if target.shape[0] <= max(DENSE_AMSR2_CHANNEL, SRAL_CLASS_CHANNEL):
        raise ValidationError(
            f"target has {target.shape[0]} channels, but channel {DENSE_AMSR2_CHANNEL} is required"
        )
    aligned = np.asarray(
        [
            ensure_grid_shape(channel, land_mask.shape, f"target channel {index}")
            for index, channel in enumerate(target)
        ],
        dtype=np.float32,
    )
    corrected = np.asarray(SRAL_TRANSFORM(aligned.copy()), dtype=np.float32)
    corrected = ensure_grid_shape(corrected, land_mask.shape, "transformed SRAL target")
    return {
        TARGET_DENSE: aligned[DENSE_AMSR2_CHANNEL],
        TARGET_SPARSE: corrected,
    }


def metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    ice_threshold: float,
) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    n_pixels = int(valid.sum())
    if n_pixels == 0:
        return {"n_pixels": 0, "mae": np.nan, "rmse": np.nan, "mse": np.nan, "iiee": np.nan}
    difference = np.asarray(prediction[valid] - target[valid], dtype=np.float64)
    mse = float(np.mean(difference**2))
    return {
        "n_pixels": n_pixels,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "iiee": float(
            np.mean((prediction[valid] > ice_threshold) != (target[valid] > ice_threshold))
        ),
    }


def metrics_for_prediction(
    prediction: np.ndarray,
    views: Mapping[str, np.ndarray],
    land_mask: np.ndarray,
    ice_threshold: float,
) -> dict[str, dict[str, float | int]]:
    ocean = ~land_mask
    dense_prediction = field_for_dense_eval(prediction, land_mask)
    dense_target = field_for_dense_eval(views[TARGET_DENSE], land_mask)
    dense = metric_values(dense_prediction, dense_target, ocean, ice_threshold)

    # Ocean forecast fill values encode zero ice and are normalized above. Keep
    # off-track target NaNs untouched so only actual SRAL-track pixels are scored.
    sparse_mask = ocean & np.isfinite(views[TARGET_SPARSE])
    sparse = metric_values(
        dense_prediction, views[TARGET_SPARSE], sparse_mask, ice_threshold
    )
    return {TARGET_DENSE: dense, TARGET_SPARSE: sparse}


def _time_dimension(data_array: Any) -> str:
    for dimension in data_array.dims:
        if "time" in dimension.lower():
            return str(dimension)
    return str(data_array.dims[0])


def load_forecast_slices(
    path: Path,
    indices: Iterable[int],
    field: str,
    grid_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise RuntimeError("xarray is required to read prognosis NetCDF files") from exc

    unique_indices = sorted({int(index) for index in indices})
    slices: dict[int, np.ndarray] = {}
    with xr.open_dataset(path) as dataset:
        if field not in dataset:
            raise ValidationError(f"variable {field!r} not found in {path}")
        data = dataset[field]
        time_dimension = _time_dimension(data)
        n_times = int(data.sizes[time_dimension])
        for index in unique_indices:
            if index < 0 or index >= n_times:
                raise ValidationError(
                    f"time index {index} is outside 0..{n_times - 1} for {path}"
                )
            selected = data.isel({time_dimension: index})
            while selected.ndim > 2:
                selected = selected.isel({selected.dims[0]: 0})
            array = np.asarray(selected.values, dtype=np.float32)
            array = ensure_grid_shape(array, grid_shape, f"forecast {path} at index {index}")
            slices[index] = np.clip(array, 0.0, 1.0)
    return slices


def load_fields_for_records(
    records: Mapping[tuple[int, str, str], Mapping[str, Any]],
    keys: Sequence[tuple[int, str, str]],
    file_column: str,
    index_column: str,
    field: str,
    grid_shape: tuple[int, int],
) -> dict[tuple[int, str, str], np.ndarray]:
    grouped: dict[str, list[tuple[tuple[int, str, str], int]]] = defaultdict(list)
    for key in keys:
        record = records[key]
        grouped[str(record[file_column])].append((key, int(record[index_column])))

    result: dict[tuple[int, str, str], np.ndarray] = {}
    for raw_path, members in grouped.items():
        slices = load_forecast_slices(
            Path(raw_path),
            [index for _, index in members],
            field,
            grid_shape,
        )
        for key, index in members:
            result[key] = slices[index]
    return result


class TargetCache:
    def __init__(self, land_mask: np.ndarray, max_items: int = 32):
        self.land_mask = land_mask
        self.max_items = max_items
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str | Path) -> dict[str, np.ndarray]:
        key = str(path)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        array = np.asarray(np.load(key), dtype=np.float32)
        views = target_views(array, self.land_mask)
        self.cache[key] = views
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return views


def _metadata_value(record: Mapping[str, Any], base: str, side: str) -> Any:
    value = record.get(f"{base}_{side}", None)
    if pd.isna(value):
        return None
    return value


def make_daily_row(
    run_side: str,
    record: Mapping[str, Any],
    target_name: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run": run_side,
        "init_date": str(record["init_date"]),
        "valid_date": str(record["valid_date"]),
        "lead_hours": int(record["lead_hours"]),
        "target": target_name,
        "forecast_file": str(record[f"forecast_file_{run_side}"]),
        "time_index": int(record[f"time_index_{run_side}"]),
        "time_value": _metadata_value(record, "time_value", run_side),
        "selection_method": _metadata_value(record, "selection_method", run_side),
        "target_file": str(record["target_file"]),
        "amsr2_channel": DENSE_AMSR2_CHANNEL,
        "sral_class_channel": SRAL_CLASS_CHANNEL,
        "sral_transform_index": SRAL_TRANSFORM_INDEX,
        "sral_transform_name": SRAL_TRANSFORM_NAME,
        **values,
    }


def assert_dense_regression(
    calculated: Mapping[str, Any],
    record: Mapping[str, Any],
    run_side: str,
    key: tuple[int, str, str],
    run_key: str,
    rtol: float,
    atol: float,
) -> None:
    old_n = int(record[f"n_pixels_{run_side}"])
    if int(calculated["n_pixels"]) != old_n:
        raise ValidationError(
            f"dense regression failed for {run_key}/{run_side}/{key}: "
            f"n_pixels={calculated['n_pixels']} (old {old_n})"
        )
    for metric in METRICS:
        old = float(record[f"{metric}_{run_side}"])
        new = float(calculated[metric])
        if not np.isclose(new, old, rtol=rtol, atol=atol, equal_nan=True):
            raise ValidationError(
                f"dense regression failed for {run_key}/{run_side}/{key}/{metric}: "
                f"new={new:.12g}, old={old:.12g}, rtol={rtol}, atol={atol}"
            )


def calculate_daily_metrics(
    preflight: PreflightResult,
    field: str,
    ice_threshold: float,
    dense_regression_rtol: float,
    max_cases_per_lead: int,
    dense_regression_atol: float,
) -> dict[str, pd.DataFrame]:
    land_mask = np.asarray(np.load(preflight.land_mask_path), dtype=bool)
    if land_mask.ndim != 2:
        raise ValidationError(f"land mask must be 2D, got {land_mask.shape}")
    ocean_pixels = int((~land_mask).sum())
    if ocean_pixels == 0:
        raise ValidationError("land mask contains no ocean pixels")

    first_run = next(iter(preflight.records_by_run))
    baseline_records = preflight.records_by_run[first_run]
    keys_by_init: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for key in preflight.selected_keys:
        keys_by_init[key[1]].append(key)

    rows_by_run: dict[str, list[dict[str, Any]]] = {
        run_key: [] for run_key in preflight.records_by_run
    }
    target_cache = TargetCache(land_mask)
    completed = 0
    total = len(preflight.selected_keys)

    for init_date in sorted(keys_by_init):
        keys = sorted(keys_by_init[init_date], key=lambda key: key[0])
        baseline_fields = load_fields_for_records(
            baseline_records,
            keys,
            "forecast_file_baseline",
            "time_index_baseline",
            field,
            land_mask.shape,
        )
        target_views_by_key = {
            key: target_cache.get(baseline_records[key]["target_file"])
            for key in keys
        }
        baseline_metrics = {
            key: metrics_for_prediction(
                baseline_fields[key], target_views_by_key[key], land_mask, ice_threshold
            )
            for key in keys
        }

        for run_key, records in preflight.records_by_run.items():
            assimilated_fields = load_fields_for_records(
                records,
                keys,
                "forecast_file_assimilated",
                "time_index_assimilated",
                field,
                land_mask.shape,
            )
            for key in keys:
                record = records[key]
                assimilated_metrics = metrics_for_prediction(
                    assimilated_fields[key], target_views_by_key[key], land_mask, ice_threshold
                )
                assert_dense_regression(
                    baseline_metrics[key][TARGET_DENSE],
                    record,
                    "baseline",
                    key,
                    run_key,
                    dense_regression_rtol,
                    dense_regression_atol,
                )
                assert_dense_regression(
                    assimilated_metrics[TARGET_DENSE],
                    record,
                    "assimilated",
                    key,
                    run_key,
                    dense_regression_rtol,
                    dense_regression_atol,
                )
                for target_name in TARGETS:
                    rows_by_run[run_key].append(
                        make_daily_row(
                            "baseline", record, target_name, baseline_metrics[key][target_name]
                        )
                    )
                    rows_by_run[run_key].append(
                        make_daily_row(
                            "assimilated", record, target_name, assimilated_metrics[target_name]
                        )
                    )
        completed += len(keys)
        if completed == total or completed % 40 == 0:
            print(f"calculated {completed}/{total} globally paired forecast cases", flush=True)

    raw_frames = {run_key: pd.DataFrame(rows) for run_key, rows in rows_by_run.items()}
    selected_frames, selected_keys, overlap_report, overlap_by_lead = (
        select_forecast_overlap_cases(
            frames_by_run=raw_frames,
            leads=sorted(preflight.total_by_lead),
            max_cases_per_lead=max_cases_per_lead,
            ocean_pixels=ocean_pixels,
            sparse_target_name=TARGET_SPARSE,
        )
    )
    preflight.selected_keys = selected_keys
    preflight.forecast_overlap_report = overlap_report
    selected_set = set(selected_keys)
    preflight.target_report["selected"] = [
        (int(row.lead_hours), str(row.init_date), str(row.valid_date)) in selected_set
        for row in preflight.target_report.itertuples(index=False)
    ]
    print(f"positive forecast/track overlap cases by lead: {overlap_by_lead}", flush=True)
    result: dict[str, pd.DataFrame] = {}
    expected_by_lead = {
        lead: sum(key[0] == lead for key in preflight.selected_keys)
        for lead in sorted(preflight.total_by_lead)
    }
    expected_counts = {
        (target_name, lead, run_side): n_cases
        for target_name in TARGETS
        for lead, n_cases in expected_by_lead.items()
        for run_side in ("baseline", "assimilated")
    }
    for run_key, frame in selected_frames.items():
        frame = frame.sort_values(["target", "lead_hours", "init_date", "valid_date", "run"]).reset_index(drop=True)
        counts = {
            (str(target_name), int(lead), str(run_side)): int(n_cases)
            for (target_name, lead, run_side), n_cases in frame.groupby(
                ["target", "lead_hours", "run"]
            ).size().items()
        }
        sparse_counts = frame.loc[
            frame["target"] == TARGET_SPARSE,
            [*KEY_COLUMNS, "run", "n_pixels"],
        ].pivot(index=list(KEY_COLUMNS), columns="run", values="n_pixels")
        if not np.array_equal(
            sparse_counts["baseline"].to_numpy(),
            sparse_counts["assimilated"].to_numpy(),
        ):
            raise ValidationError(
                f"sparse pixel counts depend on forecast values for {run_key}; "
                "ocean no-ice fill values must be evaluated as zero concentration"
            )

        if counts != expected_counts:
            raise ValidationError(
                f"unexpected case counts for {run_key}: actual={counts}, "
                f"expected={expected_counts}"
            )
        dense_pixels = frame.loc[frame["target"] == TARGET_DENSE, "n_pixels"]
        if not (dense_pixels == ocean_pixels).all():
            raise ValidationError(f"dense target no longer evaluates over all {ocean_pixels} ocean pixels")
        sparse_pixels = frame.loc[frame["target"] == TARGET_SPARSE, "n_pixels"]
        if sparse_pixels.empty or (sparse_pixels <= 0).any() or (sparse_pixels >= ocean_pixels).any():
            raise ValidationError(
                f"invalid corrected sparse pixel counts for {run_key}: "
                f"min={sparse_pixels.min()}, max={sparse_pixels.max()}, ocean={ocean_pixels}"
            )
        result[run_key] = frame

    assert_baseline_identity(result)
    return result


def assert_baseline_identity(frames: Mapping[str, pd.DataFrame]) -> None:
    reference_key = next(iter(frames))
    keys = [*CASE_COLUMNS]
    value_columns = ["n_pixels", *METRICS]
    reference = (
        frames[reference_key]
        .loc[lambda frame: frame["run"] == "baseline", [*keys, *value_columns]]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    for run_key, frame in frames.items():
        candidate = (
            frame.loc[frame["run"] == "baseline", [*keys, *value_columns]]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        if not reference[keys].equals(candidate[keys]):
            raise ValidationError(f"baseline case keys differ between {reference_key} and {run_key}")
        if not np.array_equal(reference["n_pixels"].to_numpy(), candidate["n_pixels"].to_numpy()):
            raise ValidationError(f"baseline pixel counts differ between {reference_key} and {run_key}")
        for metric in METRICS:
            if not np.allclose(
                reference[metric].to_numpy(dtype=float),
                candidate[metric].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValidationError(f"baseline {metric} differs between {reference_key} and {run_key}")


def daily_to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "forecast_file",
        "time_index",
        "time_value",
        "selection_method",
        "target_file",
        "amsr2_channel",
        "sral_class_channel",
        "sral_transform_index",
        "sral_transform_name",
        "n_pixels",
        *METRICS,
    ]
    sides: list[pd.DataFrame] = []
    for run_side in ("baseline", "assimilated"):
        selected = frame.loc[frame["run"] == run_side, [*CASE_COLUMNS, *value_columns]].copy()
        selected = selected.rename(columns={column: f"{column}_{run_side}" for column in value_columns})
        sides.append(selected)
    wide = sides[0].merge(sides[1], on=list(CASE_COLUMNS), validate="one_to_one")
    for metric in METRICS:
        denominator = wide[f"{metric}_baseline"].to_numpy(dtype=float)
        numerator = wide[f"{metric}_assimilated"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            wide[f"{metric}_skill"] = 1.0 - numerator / denominator
    return wide.sort_values(["target", "lead_hours", "init_date", "valid_date"]).reset_index(drop=True)


def summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_long = (
        frame.groupby(["target", "lead_hours", "run"], as_index=False)
        .agg(
            n_cases=("init_date", "count"),
            n_pixels_mean=("n_pixels", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mse=("mse", "mean"),
            iiee=("iiee", "mean"),
        )
        .sort_values(["target", "lead_hours", "run"])
    )
    summary_wide = summary_long.pivot(
        index=["target", "lead_hours"],
        columns="run",
        values=["n_cases", "n_pixels_mean", *METRICS],
    )
    summary_wide.columns = [f"{metric}_{run}" for metric, run in summary_wide.columns]
    summary_wide = summary_wide.reset_index()
    summary_wide["n_common_cases"] = summary_wide[
        ["n_cases_baseline", "n_cases_assimilated"]
    ].min(axis=1).astype(int)
    for metric in METRICS:
        summary_wide[f"{metric}_skill"] = (
            1.0 - summary_wide[f"{metric}_assimilated"] / summary_wide[f"{metric}_baseline"]
        )
    return summary_long, summary_wide


def block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n < 1:
        raise ValidationError("cannot bootstrap an empty sample")
    if block_size <= 1 or n < block_size:
        return rng.integers(0, n, size=n)
    starts = np.arange(0, n - block_size + 1)
    sampled_starts = rng.choice(starts, size=math.ceil(n / block_size), replace=True)
    return np.concatenate(
        [np.arange(start, start + block_size) for start in sampled_starts]
    )[:n]


def ratio_skill(assimilated: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.mean(baseline))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.mean(assimilated)) / denominator


def stable_seed(base_seed: int, *parts: Any) -> int:
    token = ":".join(str(part) for part in parts).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
    return (int(base_seed) + offset) % (2**32)


def bootstrap_summary(
    wide_by_run: Mapping[str, pd.DataFrame],
    block_size: int,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_key, frame in wide_by_run.items():
        for (target_name, lead_hours), group in frame.groupby(["target", "lead_hours"], sort=True):
            group = group.sort_values(["init_date", "valid_date"]).reset_index(drop=True)
            arrays = {
                f"{metric}_{side}": group[f"{metric}_{side}"].to_numpy(dtype=float)
                for metric in METRICS
                for side in ("baseline", "assimilated")
            }
            row: dict[str, Any] = {
                "target": target_name,
                "lead_hours": int(lead_hours),
                "run_key": run_key,
                "run_label": RUN_LABELS.get(run_key, run_key),
                "n_common_cases": len(group),
            }
            for metric in METRICS:
                row[f"{metric}_baseline"] = float(np.mean(arrays[f"{metric}_baseline"]))
                row[f"{metric}_assimilated"] = float(np.mean(arrays[f"{metric}_assimilated"]))
                row[f"{metric}_skill"] = ratio_skill(
                    arrays[f"{metric}_assimilated"], arrays[f"{metric}_baseline"]
                )
            row["mean_skill"] = 0.5 * (row["mae_skill"] + row["rmse_skill"])

            rng = np.random.default_rng(stable_seed(seed, run_key, target_name, lead_hours))
            boot_names = [
                *[
                    f"{metric}_{quantity}"
                    for metric in METRICS
                    for quantity in ("baseline", "assimilated", "skill")
                ],
                "mean_skill",
            ]
            boot = {name: np.empty(samples, dtype=float) for name in boot_names}
            for index in range(samples):
                sampled = block_indices(len(group), block_size, rng)
                for metric in METRICS:
                    baseline = arrays[f"{metric}_baseline"][sampled]
                    assimilated = arrays[f"{metric}_assimilated"][sampled]
                    boot[f"{metric}_baseline"][index] = float(np.mean(baseline))
                    boot[f"{metric}_assimilated"][index] = float(np.mean(assimilated))
                    boot[f"{metric}_skill"][index] = ratio_skill(assimilated, baseline)
                boot["mean_skill"][index] = 0.5 * (
                    boot["mae_skill"][index] + boot["rmse_skill"][index]
                )
            for name, values in boot.items():
                finite = values[np.isfinite(values)]
                if finite.size:
                    low, high = np.percentile(finite, [2.5, 97.5])
                    row[f"{name}_ci_lo"] = float(low)
                    row[f"{name}_ci_hi"] = float(high)
                    row[f"{name}_bootstrap_std"] = float(np.std(finite, ddof=1))
                else:
                    row[f"{name}_ci_lo"] = np.nan
                    row[f"{name}_ci_hi"] = np.nan
                    row[f"{name}_bootstrap_std"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "lead_hours", "run_key"]).reset_index(drop=True)


def safe_output_directory(path: Path, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = {Path("/"), Path.home().resolve(), project_dir().resolve(), project_dir().parent.resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValidationError(f"refusing unsafe output directory: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValidationError(f"output path exists and is not a directory: {resolved}")
    if resolved.is_dir() and any(resolved.iterdir()):
        if not overwrite:
            raise ValidationError(
                f"output directory is not empty: {resolved}; pass --overwrite to replace it"
            )
        for child in resolved.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def output_dir_for_run(run_key: str) -> str:
    if run_key in RUN_OUTPUT_DIRS:
        return RUN_OUTPUT_DIRS[run_key]
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_key)
    return f"forecast_metrics_{safe}"


def write_metric_outputs(
    output_dir: Path,
    daily_by_run: Mapping[str, pd.DataFrame],
    preflight: PreflightResult,
    bootstrap_block_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    wide_by_run: dict[str, pd.DataFrame] = {}
    pixel_rows: list[dict[str, Any]] = []
    for run_key, daily in daily_by_run.items():
        run_dir = output_dir / output_dir_for_run(run_key)
        wide = daily_to_wide(daily)
        summary_long, summary_wide = summary_tables(daily)
        wide_by_run[run_key] = wide
        write_csv(daily, run_dir / "forecast_metrics_daily.csv")
        write_csv(daily, run_dir / "forecast_metrics_common_daily.csv")
        write_csv(wide, run_dir / "forecast_metrics_common_daily_wide.csv")
        write_csv(summary_long, run_dir / "forecast_metrics_summary_long.csv")
        write_csv(summary_wide, run_dir / "forecast_metrics_summary_wide.csv")
        write_csv(preflight.missing_report, run_dir / "forecast_metrics_missing.csv")
        for (target_name, lead_hours, run_side), group in daily.groupby(
            ["target", "lead_hours", "run"], sort=True
        ):
            pixel_rows.append(
                {
                    "run_key": run_key,
                    "target": target_name,
                    "lead_hours": int(lead_hours),
                    "forecast": run_side,
                    "n_cases": len(group),
                    "n_pixels_min": int(group["n_pixels"].min()),
                    "n_pixels_mean": float(group["n_pixels"].mean()),
                    "n_pixels_max": int(group["n_pixels"].max()),
                }
            )

    combined = bootstrap_summary(
        wide_by_run,
        block_size=bootstrap_block_size,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    write_csv(combined, output_dir / "combined_summary.csv")
    write_csv(combined, output_dir / "forecast_restart_comparison_summary.csv")
    write_csv(pd.DataFrame(pixel_rows), output_dir / "pixel_count_summary.csv")
    write_csv(preflight.target_report, output_dir / "sparse_target_case_selection.csv")
    write_csv(preflight.forecast_overlap_report, output_dir / "forecast_overlap_case_selection.csv")
    write_csv(preflight.path_report, output_dir / "preflight_report.csv")
    write_csv(preflight.missing_report, output_dir / "missing_cases.csv")
    return wide_by_run, combined


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_provenance(
    args: argparse.Namespace,
    manifests: Mapping[str, Path],
    path_maps: Sequence[PathMap],
    preflight: PreflightResult,
    combined: pd.DataFrame,
) -> dict[str, Any]:
    selected_by_lead = {
        str(lead): sum(key[0] == lead for key in preflight.selected_keys)
        for lead in sorted(preflight.total_by_lead)
    }
    return {
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "correct forecast verification over the finite sparse SRAL target mask",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(("numpy", "pandas", "xarray", "netCDF4", "clearml")),
        "inputs": {
            "manifests": {
                run_key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for run_key, path in manifests.items()
            },
            "land_mask": str(preflight.land_mask_path),
            "path_maps": [{"old": item.old, "new": item.new} for item in path_maps],
        },
        "selection": {
            "leads_hours": [int(value) for value in args.leads],
            "max_cases_per_lead": int(args.max_cases_per_lead),
            "global_manifest_cases_by_lead": preflight.total_by_lead,
            "globally_complete_cases_by_lead": preflight.available_by_lead,
            "cases_with_sparse_observations_by_lead": preflight.observed_by_lead,
            "selected_cases_by_lead": selected_by_lead,
            "ordering": "first globally complete cases with finite sparse SRAL target pixels, sorted by init_date then valid_date independently per lead",
        },
        "metric_definition": {
            "field": args.field,
            "forecast_clip": [0.0, 1.0],
            "ice_threshold": float(args.ice_threshold),
            "dense_target": {
                "name": TARGET_DENSE,
                "channel": DENSE_AMSR2_CHANNEL,
                "mask": "all ocean pixels; non-finite ocean forecast/target values filled with zero (legacy regression convention)",
            },
            "sparse_target": {
                "name": TARGET_SPARSE,
                "class_channel": SRAL_CLASS_CHANNEL,
                "transform_index": SRAL_TRANSFORM_INDEX,
                "transform_name": SRAL_TRANSFORM_NAME,
                "transform_callable": SRAL_TRANSFORM.__name__,
                "mask": "(~land_mask) & finite(transformed_target)",
                "nonfinite_ocean_forecast_fill": 0.0,
                "off_track_imputation": False,
            },
            "metrics": list(METRICS),
            "iiee_definition": "mean binary ice-edge disagreement at the configured threshold",
        },
        "validation": {
            "dense_regression_rtol": float(args.dense_regression_rtol),
            "dense_regression_atol": float(args.dense_regression_atol),
            "dense_regression_passed": True,
            "baseline_identity_across_comparisons": True,
            "sparse_pixel_count_strictly_between_zero_and_ocean_count": True,
        },
        "bootstrap": {
            "method": "paired moving-block bootstrap within each run/target/lead",
            "block_size": int(args.bootstrap_block_size),
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
            "confidence_interval_percent": 95,
            "skill": "1 - mean(assimilated_metric) / mean(baseline_metric)",
        },
        "outputs": {
            "directory": str(Path(args.output_dir).expanduser().resolve()),
            "combined_summary_rows": len(combined),
        },
    }


def load_env_file(path: Path | None) -> None:
    if path is None or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def upload_to_clearml(
    output_dir: Path,
    combined: pd.DataFrame,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
) -> tuple[str, str | None]:
    load_env_file(args.env_file)
    try:
        from clearml import Task
    except ModuleNotFoundError as exc:
        raise RuntimeError("clearml is required unless --no-clearml is supplied") from exc

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task,
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
    )
    try:
        task.add_tags(list(args.tags))
        task.connect(dict(configuration), name="metric_recalculation")
        logger = task.get_logger()
        for row in combined.to_dict("records"):
            lead = int(row["lead_hours"])
            title = f"{row['run_key']}/{row['target']}"
            for metric in METRICS:
                logger.report_scalar(
                    title=title,
                    series=f"{metric}_skill",
                    value=float(row[f"{metric}_skill"]),
                    iteration=lead,
                )
            logger.report_scalar(
                title=title,
                series="sparse_n_common_cases" if row["target"] == TARGET_SPARSE else "dense_n_common_cases",
                value=float(row["n_common_cases"]),
                iteration=lead,
            )

        task_info = {
            "task_id": task.id,
            "project": args.clearml_project,
            "task_name": args.clearml_task,
            "artifact_name": args.artifact_name,
            "task_url": task.get_output_log_web_page(),
        }
        (output_dir / "clearml_task.json").write_text(
            json.dumps(task_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        uploaded = task.upload_artifact(
            name=args.artifact_name,
            artifact_object=str(output_dir),
            wait_on_upload=True,
        )
        if not uploaded:
            raise RuntimeError("ClearML rejected the artifact upload request")
        task.flush(wait_for_uploads=True)
        return str(task.id), task_info["task_url"]
    finally:
        task.close()


def preflight_console_summary(preflight: PreflightResult, manifests: Mapping[str, Path]) -> dict[str, Any]:
    missing_paths = int((~preflight.path_report["readable"].astype(bool)).sum())
    return {
        "manifests": {key: str(value) for key, value in manifests.items()},
        "land_mask": str(preflight.land_mask_path),
        "cases_with_sparse_observations_by_lead": preflight.observed_by_lead,
        "manifest_common_cases_by_lead": preflight.total_by_lead,
        "globally_complete_cases_by_lead": preflight.available_by_lead,
        "selected_cases": len(preflight.selected_keys),
        "missing_or_unreadable_unique_paths": missing_paths,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate dense and corrected sparse SRAL forecast metrics from saved prognosis "
            "NetCDF files, then optionally upload the result folder as one ClearML artifact."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        metavar="RUN=CSV",
        help="wide metric manifest for a comparison run; repeat for each run",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        metavar="OLD=NEW",
        help="remap a recorded absolute path prefix; repeat as needed",
    )
    parser.add_argument("--land-mask", type=Path, default=DEFAULT_LAND_MASK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_metrics_root() / DEFAULT_ARTIFACT_NAME,
    )
    parser.add_argument("--field", default=FIELD)
    parser.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS))
    parser.add_argument(
        "--max-cases-per-lead",
        type=int,
        default=DEFAULT_MAX_CASES_PER_LEAD,
        help=(
            "maximum number of valid globally paired cases per lead; if fewer are "
            "available, use all of them (default: %(default)s)"
        ),
    )
    parser.add_argument("--ice-threshold", type=float, default=0.15)
    parser.add_argument("--dense-regression-rtol", type=float, default=1e-6)
    parser.add_argument("--dense-regression-atol", type=float, default=1e-7)
    parser.add_argument(
        "--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clearml", action="store_true", help="Upload outputs to ClearML")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument("--clearml-task", default=DEFAULT_CLEARML_TASK)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="optional KEY=VALUE file for ClearML credentials; values are never logged",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_cases_per_lead < 1:
        raise ValidationError("--max-cases-per-lead must be positive")
    if not args.leads or len(set(args.leads)) != len(args.leads):
        raise ValidationError("--leads must be non-empty and unique")
    if args.bootstrap_block_size < 1:
        raise ValidationError("--bootstrap-block-size must be positive")
    if args.bootstrap_samples < 2:
        raise ValidationError("--bootstrap-samples must be at least 2")
    if not 0.0 <= args.ice_threshold <= 1.0:
        raise ValidationError("--ice-threshold must be between zero and one")
    if args.dense_regression_rtol < 0 or args.dense_regression_atol < 0:
        raise ValidationError("dense regression tolerances must be non-negative")


def run(args: argparse.Namespace) -> int:
    validate_arguments(args)
    manifests = parse_manifest_specs(args.manifest)
    path_maps = parse_path_maps(args.path_map)
    print("preflighting manifests and saved forecast/target files", flush=True)
    preflight = build_preflight(
        manifests=manifests,
        leads=args.leads,
        max_cases_per_lead=args.max_cases_per_lead,
        path_maps=path_maps,
        land_mask_path=args.land_mask,
    )
    selection_land_mask = np.asarray(np.load(preflight.land_mask_path), dtype=bool)
    selection_target_cache = TargetCache(selection_land_mask, max_items=256)
    first_run = next(iter(preflight.records_by_run))
    selected_keys, observed_by_lead, target_report = select_observation_cases(
        candidate_keys=preflight.candidate_keys,
        records=preflight.records_by_run[first_run],
        land_mask=selection_land_mask,
        load_sparse_target=lambda path: selection_target_cache.get(path)[TARGET_SPARSE],
        leads=args.leads,
        max_cases_per_lead=args.max_cases_per_lead,
    )
    preflight.selected_keys = selected_keys
    preflight.observed_by_lead = observed_by_lead
    preflight.target_report = target_report
    print(json.dumps(preflight_console_summary(preflight, manifests), indent=2, sort_keys=True))
    if args.preflight_only:
        print("preflight complete; no metrics were calculated and no ClearML task was created")
        return 0
    observation_rows = preflight.target_report.loc[
        preflight.target_report["has_sparse_observations"].astype(bool)
    ]
    preflight.selected_keys = [
        (int(row.lead_hours), str(row.init_date), str(row.valid_date))
        for row in observation_rows.itertuples(index=False)
    ]

    output_dir = safe_output_directory(args.output_dir, args.overwrite)
    daily_by_run = calculate_daily_metrics(
        preflight=preflight,
        field=args.field,
        ice_threshold=args.ice_threshold,
        dense_regression_rtol=args.dense_regression_rtol,
        dense_regression_atol=args.dense_regression_atol,
        max_cases_per_lead=args.max_cases_per_lead,
    )
    _, combined = write_metric_outputs(
        output_dir=output_dir,
        daily_by_run=daily_by_run,
        preflight=preflight,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    provenance = build_provenance(args, manifests, path_maps, preflight, combined)
    (output_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated corrected metrics written to {output_dir}", flush=True)

    if not args.clearml:
        print("ClearML upload disabled by --no-clearml")
        return 0

    clearml_configuration = {
        "manifests": {key: str(value) for key, value in manifests.items()},
        "land_mask": str(preflight.land_mask_path),
        "output_dir": str(output_dir),
        "leads": [int(value) for value in args.leads],
        "max_cases_per_lead": int(args.max_cases_per_lead),
        "field": args.field,
        "ice_threshold": float(args.ice_threshold),
        "sral_transform_index": SRAL_TRANSFORM_INDEX,
        "sral_transform_name": SRAL_TRANSFORM_NAME,
        "sparse_mask": "ocean & finite(transformed_target); nonfinite ocean forecast fill -> 0",
        "bootstrap_block_size": int(args.bootstrap_block_size),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "path_maps": [{"old": item.old, "new": item.new} for item in path_maps],
    }
    task_id, task_url = upload_to_clearml(output_dir, combined, args, clearml_configuration)
    print(f"ClearML artifact {args.artifact_name!r} uploaded on task {task_id}")
    if task_url:
        print(task_url)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValidationError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
