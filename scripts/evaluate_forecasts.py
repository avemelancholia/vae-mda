#!/usr/bin/env python3
"""Calculate and upload final NEMO forecast-verification metrics.

This entrypoint reads the baseline, ice-only, and ice+ocean NEMO forecast
directories directly.  It deliberately does not reuse historical metric
manifests: those manifests contain dense scores from older forecast restarts
and cannot validate the selected reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vae_mda.evaluation.forecast_metric_case_selection import select_observation_cases
from vae_mda.evaluation.forecast_metrics import (
    DENSE_AMSR2_CHANNEL,
    METRICS,
    RUN_LABELS,
    SRAL_CLASS_CHANNEL,
    SRAL_TRANSFORM,
    SRAL_TRANSFORM_INDEX,
    SRAL_TRANSFORM_NAME,
    TARGET_DENSE,
    TARGET_SPARSE,
    TARGETS,
    TargetCache,
    ValidationError,
    assert_baseline_identity,
    bootstrap_summary,
    daily_to_wide,
    load_env_file,
    load_fields_for_records,
    make_daily_row,
    metrics_for_prediction,
    output_dir_for_run,
    safe_output_directory,
    summary_tables,
    write_csv,
)


CaseKey = tuple[int, str, str]

DEFAULT_BASELINE_ROOT = None
DEFAULT_ICE_ONLY_ROOT = None
DEFAULT_PHYS_CORR_ROOT = None
DEFAULT_TARGET_DIR = None
DEFAULT_LAND_MASK = None
DEFAULT_DATE_START = "2023-01-01"
DEFAULT_DATE_END = "2023-08-09"
DEFAULT_LEADS = (0, 24, 48, 72)
DEFAULT_CASES_PER_LEAD = 200
DEFAULT_FIELD = "siconc"
DEFAULT_FILE_SUFFIX = "_forecast.1h_icemod.nc"
DEFAULT_TIME_TOLERANCE_HOURS = 0.51
DEFAULT_ICE_THRESHOLD = 0.15
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 14
DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 20260518
DEFAULT_CLEARML_PROJECT = "paper_vae_borey/forecast_validation"
DEFAULT_CLEARML_TASK = "forecast_metrics"
DEFAULT_ARTIFACT_NAME = "forecast_sral_metrics"
DEFAULT_TAGS = ("paper", "forecast", "sral", "metrics")

RUN_KEYS = ("ice_only", "phys_corr")
RUN_COLORS = {
    "baseline": "0.30",
    "ice_only": "tab:orange",
    "phys_corr": "tab:green",
}
TARGET_LABELS = {
    TARGET_DENSE: "Dense AMSR2",
    TARGET_SPARSE: "SRAL-conditioned AMSR2",
}


@dataclass(frozen=True)
class ForecastInspection:
    path: Path | None
    selections: dict[int, dict[str, Any]]
    errors: dict[int, str]
    report: dict[str, Any]


@dataclass
class ScanResult:
    records_by_run: dict[str, dict[CaseKey, dict[str, Any]]]
    candidate_keys: list[CaseKey]
    path_report: pd.DataFrame
    missing_report: pd.DataFrame


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    output_root = Path(
        os.environ.get("VAE_MDA_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
    )
    return output_root / DEFAULT_ARTIFACT_NAME


def default_env_file() -> Path | None:
    return None


def parse_iso_date(raw: str, option: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError(f"{option} must be an ISO date (YYYY-MM-DD): {raw!r}") from exc


def inclusive_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValidationError("--date-end must not precede --date-start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def run_directory(root: Path, init_date: date) -> Path:
    return root / f"run_{init_date.isoformat()}T00:00:00+00:00"


def _path_status(path: Path) -> tuple[bool, bool, int | None, int | None]:
    exists = path.exists()
    readable = exists and os.access(path, os.R_OK)
    if not exists:
        return False, False, None, None
    stat = path.stat()
    return True, bool(readable), int(stat.st_size), int(stat.st_mtime)


def _time_dimension(data_array: Any) -> str:
    for dimension in data_array.dims:
        if "time" in str(dimension).lower():
            return str(dimension)
    return str(data_array.dims[0])


def _fallback_index(lead_hours: int) -> int:
    if lead_hours < 0:
        raise ValidationError(f"lead hours must be non-negative: {lead_hours}")
    return 0 if lead_hours == 0 else lead_hours - 1


def inspect_forecast(
    root: Path,
    run_key: str,
    init_date: date,
    leads: Sequence[int],
    field: str,
    suffix: str,
    tolerance_hours: float,
) -> ForecastInspection:
    folder = run_directory(root, init_date)
    base_report: dict[str, Any] = {
        "kind": "forecast",
        "run_key": run_key,
        "init_date": init_date.isoformat(),
        "source_root": str(root),
    }
    if not folder.is_dir():
        reason = f"run directory not found: {folder}"
        return ForecastInspection(
            path=None,
            selections={},
            errors={int(lead): reason for lead in leads},
            report={
                **base_report,
                "resolved_path": str(folder),
                "exists": False,
                "readable": False,
                "size_bytes": None,
                "mtime_unix": None,
                "error": reason,
            },
        )

    matches = sorted(folder.glob(f"*{suffix}"))
    if len(matches) != 1:
        reason = (
            f"expected exactly one *{suffix} file in {folder}, found {len(matches)}"
        )
        return ForecastInspection(
            path=None,
            selections={},
            errors={int(lead): reason for lead in leads},
            report={
                **base_report,
                "resolved_path": str(folder),
                "exists": True,
                "readable": os.access(folder, os.R_OK),
                "size_bytes": None,
                "mtime_unix": int(folder.stat().st_mtime),
                "error": reason,
            },
        )

    path = matches[0]
    exists, readable, size, mtime = _path_status(path)
    report = {
        **base_report,
        "resolved_path": str(path),
        "exists": exists,
        "readable": readable,
        "size_bytes": size,
        "mtime_unix": mtime,
        "error": None,
    }
    if not readable:
        reason = f"forecast file is not readable: {path}"
        report["error"] = reason
        return ForecastInspection(
            path=path,
            selections={},
            errors={int(lead): reason for lead in leads},
            report=report,
        )

    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise RuntimeError("xarray is required to read NEMO forecast files") from exc

    selections: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    try:
        with xr.open_dataset(path) as dataset:
            if field not in dataset:
                raise ValidationError(f"variable {field!r} not found in {path}")
            data_array = dataset[field]
            time_dimension = _time_dimension(data_array)
            n_times = int(data_array.sizes[time_dimension])
            coordinate_times: Any = None
            coordinate_error: str | None = None
            if time_dimension in dataset.coords:
                try:
                    coordinate_times = pd.to_datetime(
                        np.asarray(dataset[time_dimension].values).ravel()
                    )
                    if getattr(coordinate_times, "tz", None) is not None:
                        coordinate_times = coordinate_times.tz_localize(None)
                except Exception as exc:
                    coordinate_error = str(exc)

            for raw_lead in leads:
                lead = int(raw_lead)
                target_time = pd.Timestamp(
                    datetime.combine(init_date, time.min) + timedelta(hours=lead)
                )
                selected: dict[str, Any] | None = None
                if coordinate_times is not None and len(coordinate_times):
                    try:
                        deltas = np.abs(
                            (coordinate_times - target_time)
                            .to_numpy(dtype="timedelta64[s]")
                            .astype(np.int64)
                        )
                        index = int(np.argmin(deltas))
                        delta_hours = float(deltas[index]) / 3600.0
                        if delta_hours <= tolerance_hours:
                            selected = {
                                "time_index": index,
                                "time_value": str(coordinate_times[index]),
                                "selection_method": "time_coordinate",
                                "time_delta_hours": delta_hours,
                                "n_times": n_times,
                            }
                    except Exception as exc:
                        coordinate_error = str(exc)

                if selected is None:
                    index = _fallback_index(lead)
                    if 0 <= index < n_times:
                        selected = {
                            "time_index": index,
                            "time_value": None,
                            "selection_method": "fallback_index",
                            "time_delta_hours": None,
                            "n_times": n_times,
                        }
                    else:
                        detail = (
                            f"; time-coordinate error: {coordinate_error}"
                            if coordinate_error
                            else ""
                        )
                        errors[lead] = (
                            f"no time within {tolerance_hours} h of {target_time} and "
                            f"fallback index {index} is outside 0..{n_times - 1} in {path}"
                            f"{detail}"
                        )
                        continue
                selections[lead] = selected
    except Exception as exc:
        reason = f"could not inspect {path}: {type(exc).__name__}: {exc}"
        report["error"] = reason
        return ForecastInspection(
            path=path,
            selections={},
            errors={int(lead): reason for lead in leads},
            report=report,
        )

    if errors:
        report["error"] = "; ".join(sorted(set(errors.values())))
    return ForecastInspection(
        path=path,
        selections=selections,
        errors=errors,
        report=report,
    )


def scan_inputs(
    baseline_root: Path,
    run_roots: Mapping[str, Path],
    target_dir: Path,
    land_mask_path: Path,
    dates: Sequence[date],
    leads: Sequence[int],
    field: str,
    suffix: str,
    tolerance_hours: float,
) -> ScanResult:
    path_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    records_by_run: dict[str, dict[CaseKey, dict[str, Any]]] = {
        run_key: {} for run_key in run_roots
    }
    candidate_keys: list[CaseKey] = []

    exists, readable, size, mtime = _path_status(land_mask_path)
    path_rows.append(
        {
            "kind": "land_mask",
            "run_key": "shared",
            "init_date": None,
            "source_root": str(land_mask_path.parent),
            "resolved_path": str(land_mask_path),
            "exists": exists,
            "readable": readable,
            "size_bytes": size,
            "mtime_unix": mtime,
            "error": None if readable else "land mask is missing or unreadable",
        }
    )
    if not readable:
        raise ValidationError(f"land mask is missing or unreadable: {land_mask_path}")

    forecast_roots = {"baseline": baseline_root, **run_roots}
    for position, init_date in enumerate(dates, start=1):
        inspections = {
            run_key: inspect_forecast(
                root=root,
                run_key=run_key,
                init_date=init_date,
                leads=leads,
                field=field,
                suffix=suffix,
                tolerance_hours=tolerance_hours,
            )
            for run_key, root in forecast_roots.items()
        }
        path_rows.extend(inspection.report for inspection in inspections.values())

        for raw_lead in leads:
            lead = int(raw_lead)
            valid_date = init_date + timedelta(hours=lead)
            target_path = target_dir / f"{valid_date.isoformat()}.npy"
            target_exists, target_readable, target_size, target_mtime = _path_status(
                target_path
            )
            path_rows.append(
                {
                    "kind": "target",
                    "run_key": "shared",
                    "init_date": init_date.isoformat(),
                    "source_root": str(target_dir),
                    "resolved_path": str(target_path),
                    "exists": target_exists,
                    "readable": target_readable,
                    "size_bytes": target_size,
                    "mtime_unix": target_mtime,
                    "error": None if target_readable else "target is missing or unreadable",
                }
            )

            key: CaseKey = (lead, init_date.isoformat(), valid_date.isoformat())
            case_errors: list[tuple[str, str, str]] = []
            if not target_readable:
                case_errors.append(
                    ("target", "shared", f"target is missing or unreadable: {target_path}")
                )
            for run_key, inspection in inspections.items():
                if lead in inspection.errors:
                    case_errors.append(("forecast", run_key, inspection.errors[lead]))
                elif lead not in inspection.selections or inspection.path is None:
                    case_errors.append(
                        ("forecast", run_key, f"forecast selection is unavailable for {key}")
                    )

            if case_errors:
                for kind, run_key, reason in case_errors:
                    missing_rows.append(
                        {
                            "lead_hours": lead,
                            "init_date": init_date.isoformat(),
                            "valid_date": valid_date.isoformat(),
                            "kind": kind,
                            "run_key": run_key,
                            "reason": reason,
                        }
                    )
                continue

            baseline_inspection = inspections["baseline"]
            baseline_selection = baseline_inspection.selections[lead]
            for run_key in run_roots:
                run_inspection = inspections[run_key]
                run_selection = run_inspection.selections[lead]
                records_by_run[run_key][key] = {
                    "lead_hours": lead,
                    "init_date": init_date.isoformat(),
                    "valid_date": valid_date.isoformat(),
                    "forecast_file_baseline": str(baseline_inspection.path),
                    "forecast_file_assimilated": str(run_inspection.path),
                    "time_index_baseline": int(baseline_selection["time_index"]),
                    "time_index_assimilated": int(run_selection["time_index"]),
                    "time_value_baseline": baseline_selection["time_value"],
                    "time_value_assimilated": run_selection["time_value"],
                    "selection_method_baseline": baseline_selection[
                        "selection_method"
                    ],
                    "selection_method_assimilated": run_selection[
                        "selection_method"
                    ],
                    "target_file": str(target_path),
                }
            candidate_keys.append(key)

        if position == len(dates) or position % 40 == 0:
            print(f"scanned {position}/{len(dates)} initialization dates", flush=True)

    path_columns = [
        "kind",
        "run_key",
        "init_date",
        "source_root",
        "resolved_path",
        "exists",
        "readable",
        "size_bytes",
        "mtime_unix",
        "error",
    ]
    path_report = (
        pd.DataFrame(path_rows, columns=path_columns)
        .drop_duplicates(["kind", "run_key", "resolved_path"])
        .sort_values(["kind", "run_key", "resolved_path"])
        .reset_index(drop=True)
    )
    missing_columns = [
        "lead_hours",
        "init_date",
        "valid_date",
        "kind",
        "run_key",
        "reason",
    ]
    missing_report = pd.DataFrame(missing_rows, columns=missing_columns)
    if not missing_report.empty:
        missing_report = missing_report.sort_values(
            ["lead_hours", "init_date", "run_key", "kind"]
        ).reset_index(drop=True)
    candidate_keys = sorted(set(candidate_keys), key=lambda key: (key[1], key[0], key[2]))
    return ScanResult(
        records_by_run=records_by_run,
        candidate_keys=candidate_keys,
        path_report=path_report,
        missing_report=missing_report,
    )


def select_cases(
    scan: ScanResult,
    land_mask: np.ndarray,
    leads: Sequence[int],
    cases_per_lead: int,
) -> tuple[list[CaseKey], dict[int, int], pd.DataFrame]:
    first_run = next(iter(scan.records_by_run))
    cache = TargetCache(land_mask, max_items=256)
    selected, observed_by_lead, report = select_observation_cases(
        candidate_keys=scan.candidate_keys,
        records=scan.records_by_run[first_run],
        land_mask=land_mask,
        load_sparse_target=lambda path: cache.get(path)[TARGET_SPARSE],
        leads=leads,
        max_cases_per_lead=cases_per_lead,
    )
    insufficient = {
        int(lead): int(observed_by_lead.get(int(lead), 0))
        for lead in leads
        if observed_by_lead.get(int(lead), 0) < cases_per_lead
    }
    if insufficient:
        raise ValidationError(
            f"fewer than {cases_per_lead} globally complete observation-bearing "
            f"cases are available for these leads: {insufficient}"
        )
    selected_by_lead = {
        int(lead): sum(key[0] == int(lead) for key in selected) for lead in leads
    }
    if any(count != cases_per_lead for count in selected_by_lead.values()):
        raise ValidationError(
            f"selected case count mismatch: expected {cases_per_lead}, got {selected_by_lead}"
        )
    return selected, observed_by_lead, report


def calculate_daily_metrics(
    records_by_run: Mapping[str, Mapping[CaseKey, Mapping[str, Any]]],
    selected_keys: Sequence[CaseKey],
    land_mask_path: Path,
    field: str,
    ice_threshold: float,
    leads: Sequence[int],
    cases_per_lead: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    land_mask = np.asarray(np.load(land_mask_path), dtype=bool)
    if land_mask.ndim != 2:
        raise ValidationError(f"land mask must be 2D, got {land_mask.shape}")
    ocean_pixels = int((~land_mask).sum())
    if ocean_pixels == 0:
        raise ValidationError("land mask contains no ocean pixels")

    first_run = next(iter(records_by_run))
    baseline_records = records_by_run[first_run]
    keys_by_init: dict[str, list[CaseKey]] = defaultdict(list)
    for key in selected_keys:
        keys_by_init[key[1]].append(key)

    rows_by_run: dict[str, list[dict[str, Any]]] = {
        run_key: [] for run_key in records_by_run
    }
    target_cache = TargetCache(land_mask)
    completed = 0
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
        views_by_key = {
            key: target_cache.get(baseline_records[key]["target_file"]) for key in keys
        }
        baseline_metrics = {
            key: metrics_for_prediction(
                baseline_fields[key], views_by_key[key], land_mask, ice_threshold
            )
            for key in keys
        }

        for run_key, records in records_by_run.items():
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
                    assimilated_fields[key], views_by_key[key], land_mask, ice_threshold
                )
                for target_name in TARGETS:
                    rows_by_run[run_key].append(
                        make_daily_row(
                            "baseline",
                            record,
                            target_name,
                            baseline_metrics[key][target_name],
                        )
                    )
                    rows_by_run[run_key].append(
                        make_daily_row(
                            "assimilated",
                            record,
                            target_name,
                            assimilated_metrics[target_name],
                        )
                    )
        completed += len(keys)
        if completed == len(selected_keys) or completed % 40 == 0:
            print(
                f"calculated {completed}/{len(selected_keys)} paired forecast cases",
                flush=True,
            )

    expected_counts = {
        (target_name, int(lead), run_side): cases_per_lead
        for target_name in TARGETS
        for lead in leads
        for run_side in ("baseline", "assimilated")
    }
    daily_by_run: dict[str, pd.DataFrame] = {}
    overlap_rows: list[dict[str, Any]] = []
    for run_key, rows in rows_by_run.items():
        frame = pd.DataFrame(rows).sort_values(
            ["target", "lead_hours", "init_date", "valid_date", "run"]
        ).reset_index(drop=True)
        actual_counts = {
            (str(target), int(lead), str(run_side)): int(count)
            for (target, lead, run_side), count in frame.groupby(
                ["target", "lead_hours", "run"]
            ).size().items()
        }
        if actual_counts != expected_counts:
            raise ValidationError(
                f"unexpected case counts for {run_key}: actual={actual_counts}, "
                f"expected={expected_counts}"
            )
        dense_pixels = frame.loc[frame["target"] == TARGET_DENSE, "n_pixels"]
        if not (dense_pixels == ocean_pixels).all():
            raise ValidationError(
                f"dense target does not evaluate over all {ocean_pixels} ocean pixels"
            )

        sparse = frame.loc[
            frame["target"] == TARGET_SPARSE,
            ["lead_hours", "init_date", "valid_date", "run", "n_pixels"],
        ]
        sparse_wide = sparse.pivot(
            index=["lead_hours", "init_date", "valid_date"],
            columns="run",
            values="n_pixels",
        ).reset_index()
        if not np.array_equal(
            sparse_wide["baseline"].to_numpy(),
            sparse_wide["assimilated"].to_numpy(),
        ):
            raise ValidationError(
                f"sparse pixel counts depend on forecast values for {run_key}"
            )
        for row in sparse_wide.itertuples(index=False):
            baseline_count = int(row.baseline)
            assimilated_count = int(row.assimilated)
            eligible = (
                0 < baseline_count < ocean_pixels
                and 0 < assimilated_count < ocean_pixels
            )
            if not eligible:
                raise ValidationError(
                    f"invalid sparse pixel count for {run_key}/"
                    f"{row.init_date}/lead-{row.lead_hours}: "
                    f"baseline={baseline_count}, assimilated={assimilated_count}"
                )
            overlap_rows.append(
                {
                    "run_key": run_key,
                    "lead_hours": int(row.lead_hours),
                    "init_date": str(row.init_date),
                    "valid_date": str(row.valid_date),
                    "n_pixels_baseline": baseline_count,
                    "n_pixels_assimilated": assimilated_count,
                    "eligible_for_run": True,
                    "globally_eligible": True,
                    "selected": True,
                }
            )
        daily_by_run[run_key] = frame

    assert_baseline_identity(daily_by_run)
    overlap_report = pd.DataFrame(overlap_rows).sort_values(
        ["lead_hours", "init_date", "valid_date", "run_key"]
    ).reset_index(drop=True)
    return daily_by_run, overlap_report


def write_metric_outputs(
    output_dir: Path,
    daily_by_run: Mapping[str, pd.DataFrame],
    path_report: pd.DataFrame,
    missing_report: pd.DataFrame,
    target_report: pd.DataFrame,
    overlap_report: pd.DataFrame,
    bootstrap_block_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
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
        write_csv(missing_report, run_dir / "forecast_metrics_missing.csv")
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
    write_csv(target_report, output_dir / "sparse_target_case_selection.csv")
    write_csv(overlap_report, output_dir / "forecast_overlap_case_selection.csv")
    write_csv(path_report, output_dir / "preflight_report.csv")
    write_csv(missing_report, output_dir / "missing_cases.csv")
    return combined


def plot_summary(combined: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to draw the forecast figure") from exc

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 6.8), sharex="col", constrained_layout=True
    )
    for column, target_name in enumerate(TARGETS):
        target_rows = combined.loc[combined["target"] == target_name]
        by_run = {
            run_key: target_rows.loc[target_rows["run_key"] == run_key]
            .sort_values("lead_hours")
            .reset_index(drop=True)
            for run_key in RUN_KEYS
        }
        if any(frame.empty for frame in by_run.values()):
            raise ValidationError(f"missing summary rows for target {target_name}")
        reference = by_run[RUN_KEYS[0]]
        leads = reference["lead_hours"].to_numpy(dtype=float)
        baseline = reference["mae_baseline"].to_numpy(dtype=float)

        top = axes[0, column]
        top.plot(
            leads,
            baseline,
            color=RUN_COLORS["baseline"],
            marker="o",
            linewidth=1.8,
            label="Baseline",
        )
        for run_key, frame in by_run.items():
            run_leads = frame["lead_hours"].to_numpy(dtype=float)
            run_baseline = frame["mae_baseline"].to_numpy(dtype=float)
            if not np.array_equal(leads, run_leads) or not np.allclose(
                baseline, run_baseline, rtol=0.0, atol=0.0
            ):
                raise ValidationError(f"summary baseline mismatch for {target_name}")
            values = frame["mae_assimilated"].to_numpy(dtype=float)
            spread = frame["mae_assimilated_bootstrap_std"].to_numpy(dtype=float)
            top.plot(
                leads,
                values,
                color=RUN_COLORS[run_key],
                marker="o",
                linewidth=1.8,
                label=RUN_LABELS[run_key],
            )
            top.fill_between(
                leads,
                values - spread,
                values + spread,
                color=RUN_COLORS[run_key],
                alpha=0.15,
                linewidth=0,
            )
        top.set_title(f"{TARGET_LABELS[target_name]}: MAE")
        top.set_ylabel("MAE")
        top.grid(True, color="0.86", linewidth=0.8)
        if column == 1:
            top.legend(loc="best")

        bottom = axes[1, column]
        bottom.axhline(0.0, color="black", linewidth=1.0)
        for run_key, frame in by_run.items():
            values = frame["mean_skill"].to_numpy(dtype=float)
            spread = frame["mean_skill_bootstrap_std"].to_numpy(dtype=float)
            bottom.plot(
                leads,
                values,
                color=RUN_COLORS[run_key],
                marker="o",
                linewidth=1.8,
                label=RUN_LABELS[run_key],
            )
            bottom.fill_between(
                leads,
                values - spread,
                values + spread,
                color=RUN_COLORS[run_key],
                alpha=0.15,
                linewidth=0,
            )
        bottom.set_title(f"{TARGET_LABELS[target_name]}: skill")
        bottom.set_xlabel("Lead time, hours")
        bottom.set_ylabel("Mean MAE/RMSE skill")
        bottom.set_xticks(leads)
        bottom.grid(True, color="0.86", linewidth=0.8)
        if column == 1:
            bottom.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    run_roots: Mapping[str, Path],
    scan: ScanResult,
    selected_keys: Sequence[CaseKey],
    observed_by_lead: Mapping[int, int],
    combined: pd.DataFrame,
) -> dict[str, Any]:
    candidate_by_lead = {
        str(lead): sum(key[0] == int(lead) for key in scan.candidate_keys)
        for lead in args.leads
    }
    selected_by_lead = {
        str(lead): sum(key[0] == int(lead) for key in selected_keys)
        for lead in args.leads
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "final forecast verification from raw NEMO outputs",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ("numpy", "pandas", "xarray", "netCDF4", "matplotlib", "clearml")
        ),
        "inputs": {
            "baseline_root": str(args.baseline_root),
            "run_roots": {key: str(path) for key, path in run_roots.items()},
            "target_dir": str(args.target_dir),
            "land_mask": str(args.land_mask),
            "date_start": args.date_start,
            "date_end": args.date_end,
            "forecast_file_suffix": args.file_suffix,
            "path_report": "preflight_report.csv",
            "missing_report": "missing_cases.csv",
        },
        "selection": {
            "leads_hours": [int(value) for value in args.leads],
            "required_cases_per_lead": int(args.cases_per_lead),
            "globally_file_complete_cases_by_lead": candidate_by_lead,
            "observation_bearing_cases_by_lead": {
                str(key): int(value) for key, value in observed_by_lead.items()
            },
            "selected_cases_by_lead": selected_by_lead,
            "ordering": (
                "first globally complete cases with finite sparse SRAL target "
                "pixels, sorted chronologically independently per lead"
            ),
        },
        "metric_definition": {
            "field": args.field,
            "forecast_clip": [0.0, 1.0],
            "ice_threshold": float(args.ice_threshold),
            "dense_target": {
                "name": TARGET_DENSE,
                "channel": DENSE_AMSR2_CHANNEL,
                "mask": "all ocean pixels; non-finite ocean values filled with zero",
            },
            "sparse_target": {
                "name": TARGET_SPARSE,
                "class_channel": SRAL_CLASS_CHANNEL,
                "transform_index": SRAL_TRANSFORM_INDEX,
                "transform_name": SRAL_TRANSFORM_NAME,
                "transform_callable": SRAL_TRANSFORM.__name__,
                "mask": "(~land_mask) & finite(selected_sral_transform(target))",
                "nonfinite_ocean_forecast_fill": 0.0,
                "off_track_imputation": False,
            },
            "metrics": list(METRICS),
            "iiee_definition": "mean binary ice-edge disagreement at the configured threshold",
        },
        "validation": {
            "sral_transform_callable_verified": SRAL_TRANSFORM.__name__ == "transform_selected",
            "exact_case_count_verified": True,
            "baseline_identity_across_comparisons": True,
            "sparse_pixel_counts_independent_of_forecast_fill": True,
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
            "figure": "forecast_restart_comparison.png",
        },
    }


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
        task.connect(dict(configuration), name="forecast_metrics")
        logger = task.get_logger()
        for row in combined.to_dict("records"):
            lead = int(row["lead_hours"])
            title = f"{row['run_key']}/{row['target']}"
            for metric in METRICS:
                for quantity in ("baseline", "assimilated", "skill"):
                    name = f"{metric}_{quantity}"
                    value = float(row[name])
                    if np.isfinite(value):
                        logger.report_scalar(
                            title=title,
                            series=name,
                            value=value,
                            iteration=lead,
                        )
            mean_skill = float(row["mean_skill"])
            if np.isfinite(mean_skill):
                logger.report_scalar(
                    title=title,
                    series="mean_skill",
                    value=mean_skill,
                    iteration=lead,
                )
            logger.report_scalar(
                title=title,
                series="n_common_cases",
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
            json.dumps(task_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        uploaded = task.upload_artifact(
            name=args.artifact_name,
            artifact_object=str(output_dir),
            wait_on_upload=True,
        )
        if not uploaded:
            raise RuntimeError("ClearML rejected the forecast metric artifact upload")
        task.flush(wait_for_uploads=True)
        return str(task.id), task_info["task_url"]
    finally:
        task.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ice-only-root", type=Path, required=True)
    parser.add_argument("--phys-corr-root", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--land-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--date-start", default=DEFAULT_DATE_START)
    parser.add_argument("--date-end", default=DEFAULT_DATE_END)
    parser.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS))
    parser.add_argument("--cases-per-lead", type=int, default=DEFAULT_CASES_PER_LEAD)
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--file-suffix", default=DEFAULT_FILE_SUFFIX)
    parser.add_argument(
        "--time-tolerance-hours",
        type=float,
        default=DEFAULT_TIME_TOLERANCE_HOURS,
    )
    parser.add_argument("--ice-threshold", type=float, default=DEFAULT_ICE_THRESHOLD)
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=DEFAULT_BOOTSTRAP_BLOCK_SIZE,
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clearml", action="store_true", help="Upload outputs to ClearML")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument("--clearml-task", default=DEFAULT_CLEARML_TASK)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument("--env-file", type=Path, default=default_env_file())
    return parser


def validate_arguments(args: argparse.Namespace) -> tuple[date, date]:
    if SRAL_TRANSFORM.__name__ != "transform_selected":
        raise ValidationError(
            f"Selected SRAL preprocessing is invalid: found {SRAL_TRANSFORM.__name__}"
        )
    if not args.leads or len(set(args.leads)) != len(args.leads):
        raise ValidationError("--leads must be non-empty and unique")
    if any(int(lead) < 0 for lead in args.leads):
        raise ValidationError("--leads must be non-negative")
    if args.cases_per_lead < 1:
        raise ValidationError("--cases-per-lead must be positive")
    if args.time_tolerance_hours < 0:
        raise ValidationError("--time-tolerance-hours must be non-negative")
    if not 0.0 <= args.ice_threshold <= 1.0:
        raise ValidationError("--ice-threshold must be between zero and one")
    if args.bootstrap_block_size < 1:
        raise ValidationError("--bootstrap-block-size must be positive")
    if args.bootstrap_samples < 2:
        raise ValidationError("--bootstrap-samples must be at least two")
    start = parse_iso_date(args.date_start, "--date-start")
    end = parse_iso_date(args.date_end, "--date-end")
    if end < start:
        raise ValidationError("--date-end must not precede --date-start")
    roots = [args.baseline_root, args.ice_only_root, args.phys_corr_root]
    if len({str(path.expanduser().resolve()) for path in roots}) != len(roots):
        raise ValidationError("baseline and assimilated run roots must be distinct")
    return start, end


def run(args: argparse.Namespace) -> int:
    start, end = validate_arguments(args)
    args.baseline_root = args.baseline_root.expanduser().resolve()
    args.ice_only_root = args.ice_only_root.expanduser().resolve()
    args.phys_corr_root = args.phys_corr_root.expanduser().resolve()
    args.target_dir = args.target_dir.expanduser().resolve()
    args.land_mask = args.land_mask.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    run_roots = {
        "ice_only": args.ice_only_root,
        "phys_corr": args.phys_corr_root,
    }
    dates = inclusive_dates(start, end)

    print(
        f"Selected SRAL preprocessing: {SRAL_TRANSFORM_NAME}",
        flush=True,
    )
    print("scanning raw NEMO forecasts and validation targets", flush=True)
    scan = scan_inputs(
        baseline_root=args.baseline_root,
        run_roots=run_roots,
        target_dir=args.target_dir,
        land_mask_path=args.land_mask,
        dates=dates,
        leads=args.leads,
        field=args.field,
        suffix=args.file_suffix,
        tolerance_hours=args.time_tolerance_hours,
    )
    land_mask = np.asarray(np.load(args.land_mask), dtype=bool)
    selected_keys, observed_by_lead, target_report = select_cases(
        scan=scan,
        land_mask=land_mask,
        leads=args.leads,
        cases_per_lead=args.cases_per_lead,
    )
    print(
        json.dumps(
            {
                "globally_complete_cases": len(scan.candidate_keys),
                "observation_bearing_cases_by_lead": observed_by_lead,
                "selected_cases": len(selected_keys),
                "missing_case_requirements": len(scan.missing_report),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    daily_by_run, overlap_report = calculate_daily_metrics(
        records_by_run=scan.records_by_run,
        selected_keys=selected_keys,
        land_mask_path=args.land_mask,
        field=args.field,
        ice_threshold=args.ice_threshold,
        leads=args.leads,
        cases_per_lead=args.cases_per_lead,
    )
    output_dir = safe_output_directory(args.output_dir, args.overwrite)
    combined = write_metric_outputs(
        output_dir=output_dir,
        daily_by_run=daily_by_run,
        path_report=scan.path_report,
        missing_report=scan.missing_report,
        target_report=target_report,
        overlap_report=overlap_report,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    figure_path = output_dir / "forecast_restart_comparison.png"
    plot_summary(combined, figure_path)
    provenance = build_provenance(
        args=args,
        run_roots=run_roots,
        scan=scan,
        selected_keys=selected_keys,
        observed_by_lead=observed_by_lead,
        combined=combined,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"validated forecast metrics written to {output_dir}", flush=True)

    if not args.clearml:
        print("ClearML upload disabled")
        return 0

    clearml_configuration = {
        "baseline_root": str(args.baseline_root),
        "run_roots": {key: str(value) for key, value in run_roots.items()},
        "target_dir": str(args.target_dir),
        "land_mask": str(args.land_mask),
        "date_start": args.date_start,
        "date_end": args.date_end,
        "leads": [int(value) for value in args.leads],
        "cases_per_lead": int(args.cases_per_lead),
        "field": args.field,
        "ice_threshold": float(args.ice_threshold),
        "sral_transform_index": SRAL_TRANSFORM_INDEX,
        "sral_transform_name": SRAL_TRANSFORM_NAME,
        "sral_transform_callable": SRAL_TRANSFORM.__name__,
        "bootstrap_block_size": int(args.bootstrap_block_size),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "output_dir": str(output_dir),
    }
    task_id, task_url = upload_to_clearml(
        output_dir=output_dir,
        combined=combined,
        args=args,
        configuration=clearml_configuration,
    )
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
