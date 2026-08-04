"""Select a globally paired sample with positive forecast/track overlap."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


CaseKey = tuple[int, str, str]
CASE_KEY_COLUMNS = ["lead_hours", "init_date", "valid_date"]


def _row_key(row: Any) -> CaseKey:
    return int(row.lead_hours), str(row.init_date), str(row.valid_date)


def select_forecast_overlap_cases(
    frames_by_run: Mapping[str, pd.DataFrame],
    leads: Sequence[int],
    max_cases_per_lead: int,
    ocean_pixels: int,
    sparse_target_name: str,
) -> tuple[dict[str, pd.DataFrame], list[CaseKey], pd.DataFrame, dict[int, int]]:
    """Intersect positive sparse overlap across every forecast comparison."""
    eligible_sets: list[set[CaseKey]] = []
    report_rows: list[dict[str, Any]] = []

    for run_key, frame in frames_by_run.items():
        sparse = frame.loc[frame["target"] == sparse_target_name].copy()
        wide = sparse.pivot(
            index=CASE_KEY_COLUMNS,
            columns="run",
            values="n_pixels",
        ).reset_index()
        required = {"baseline", "assimilated"}
        if not required.issubset(wide.columns):
            raise ValueError(f"missing baseline/assimilated sparse rows for {run_key}")

        eligible_for_run: set[CaseKey] = set()
        for row in wide.itertuples(index=False):
            key = _row_key(row)
            baseline_pixels = int(row.baseline)
            assimilated_pixels = int(row.assimilated)
            eligible = (
                0 < baseline_pixels < ocean_pixels
                and 0 < assimilated_pixels < ocean_pixels
            )
            if eligible:
                eligible_for_run.add(key)
            report_rows.append(
                {
                    "run_key": run_key,
                    "lead_hours": key[0],
                    "init_date": key[1],
                    "valid_date": key[2],
                    "n_pixels_baseline": baseline_pixels,
                    "n_pixels_assimilated": assimilated_pixels,
                    "eligible_for_run": eligible,
                }
            )
        eligible_sets.append(eligible_for_run)

    globally_eligible = set.intersection(*eligible_sets) if eligible_sets else set()
    selected: list[CaseKey] = []
    overlap_by_lead: dict[int, int] = {}
    for lead in [int(value) for value in leads]:
        lead_keys = sorted(
            (key for key in globally_eligible if key[0] == lead),
            key=lambda key: (key[1], key[2]),
        )
        overlap_by_lead[lead] = len(lead_keys)
        if not lead_keys:
            raise ValueError(
                f"lead {lead} h has no cases with positive sparse overlap in every "
                "forecast; a corrected sparse metric cannot be calculated for this lead"
            )
        # The requested count is a cap: use all valid cases when fewer exist.
        selected.extend(lead_keys[:max_cases_per_lead])

    selected_set = set(selected)
    for row in report_rows:
        key = (int(row["lead_hours"]), str(row["init_date"]), str(row["valid_date"]))
        row["globally_eligible"] = key in globally_eligible
        row["selected"] = key in selected_set

    filtered: dict[str, pd.DataFrame] = {}
    for run_key, frame in frames_by_run.items():
        keys = [
            (int(lead), str(init_date), str(valid_date))
            for lead, init_date, valid_date in frame[CASE_KEY_COLUMNS].itertuples(
                index=False, name=None
            )
        ]
        filtered[run_key] = frame.loc[
            [key in selected_set for key in keys]
        ].reset_index(drop=True)

    report = pd.DataFrame(report_rows).sort_values(
        ["lead_hours", "init_date", "valid_date", "run_key"]
    ).reset_index(drop=True)
    selected = sorted(selected, key=lambda key: (key[1], key[0], key[2]))
    return filtered, selected, report, overlap_by_lead
