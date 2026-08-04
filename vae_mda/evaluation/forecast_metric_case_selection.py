"""Observation-aware case selection for sparse forecast verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


CaseKey = tuple[int, str, str]


def select_observation_cases(
    candidate_keys: Sequence[CaseKey],
    records: Mapping[CaseKey, Mapping[str, Any]],
    land_mask: np.ndarray,
    load_sparse_target: Callable[[str], np.ndarray],
    leads: Sequence[int],
    max_cases_per_lead: int,
) -> tuple[list[CaseKey], dict[int, int], pd.DataFrame]:
    """Choose the first file-complete cases that contain sparse observations.

    Track coverage is target-dependent but run-independent. Screening it before
    loading forecasts prevents no-observation dates from entering paired metric
    means or bootstrap samples.
    """
    land_mask = np.asarray(land_mask, dtype=bool)
    if land_mask.ndim != 2:
        raise ValueError(f"land mask must be 2D, got {land_mask.shape}")
    ocean = ~land_mask

    selected: list[CaseKey] = []
    observed_by_lead: dict[int, int] = {}
    report_rows: list[dict[str, Any]] = []

    for lead in [int(value) for value in leads]:
        lead_candidates = sorted(
            (key for key in candidate_keys if key[0] == lead),
            key=lambda key: (key[1], key[2]),
        )
        observed: list[CaseKey] = []
        for key in lead_candidates:
            target_path = str(records[key]["target_file"])
            sparse_target = np.asarray(load_sparse_target(target_path))
            if sparse_target.shape != land_mask.shape:
                raise ValueError(
                    f"sparse target {target_path} has shape {sparse_target.shape}, "
                    f"expected {land_mask.shape}"
                )
            n_pixels = int((ocean & np.isfinite(sparse_target)).sum())
            if n_pixels > 0:
                observed.append(key)
            report_rows.append(
                {
                    "lead_hours": key[0],
                    "init_date": key[1],
                    "valid_date": key[2],
                    "target_file": target_path,
                    "n_sparse_target_pixels": n_pixels,
                    "has_sparse_observations": n_pixels > 0,
                }
            )

        observed_by_lead[lead] = len(observed)
        if not observed:
            raise ValueError(
                f"lead {lead} h has no file-complete cases with sparse SRAL "
                "observations; a corrected sparse metric cannot be calculated for this lead"
            )
        # The requested count is a cap: use all observed cases when fewer exist.
        selected.extend(observed[:max_cases_per_lead])

    selected_set = set(selected)
    for row in report_rows:
        key = (int(row["lead_hours"]), str(row["init_date"]), str(row["valid_date"]))
        row["selected"] = key in selected_set

    report = pd.DataFrame(report_rows).sort_values(
        ["lead_hours", "init_date", "valid_date"]
    ).reset_index(drop=True)
    selected = sorted(selected, key=lambda key: (key[1], key[0], key[2]))
    return selected, observed_by_lead, report
