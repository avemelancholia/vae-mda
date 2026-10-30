"""Score forecast branches against AMSR2 and compare them with paired moving-block bootstrap statistics.

Used for the selected-configuration forecasts (Table 3, Fig. 4b, Supplementary Tables S11 and S12).

Scoring. A branch directory holds one folder per cycle, ``run_<D>T00:00:00+00:00``, with exactly one
``*_forecast.1h_icemod.nc`` and a ``forecast-done`` marker; it is the 72-hour forecast initialized from the
analysis of date D - 1. The hourly-mean SIC records 23, 47 and 71 (lead 24, 48 and 72 h) are verified against
the observation file of the valid date: dense AMSR2 (channel 7, missing values as zero) on all ocean cells, and
sparse SRAL-conditioned AMSR2 (rule 11) on its finite track cells. Nonfinite forecast values count as zero SIC.

Comparison. Only dates scored for every branch, lead and target are used, and the verification support of each
case must be the same for every branch. Skill is 1 - mean error / mean baseline error; for several leads, errors
are averaged over the leads first. The mean skill averages MAE and RMSE skill, the combined skill averages the
mean skills against the two targets, and the IIEE skill uses the area-integrated IIEE averaged over targets.
Uncertainty: paired moving-block bootstrap (consecutive dates, 14-day blocks, 5000 replicates), with the same
resampled dates for every branch, so differences between branches are paired.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

LEADS = {24: 23, 48: 47, 72: 71}            # lead hours -> hourly record ending at that lead
TARGETS = ('amsr2_dense', 'sral_rule11')
DENSE_CHANNEL = 7
ERRORS = ('mae', 'rmse', 'iiee_km2')
LEAD_SETS = {'24': (24,), '48': (48,), '72': (72,), '24-72': (24, 48, 72)}


# ----------------------------------------------------------------------------------------------- scoring
def cell_area(geometry, ocean):
    """Cell areas (m2) from a NEMO domain_cfg file (e1t * e2t), checked against the land mask."""
    import netCDF4 as nc
    with nc.Dataset(geometry) as ds:
        area = np.asarray(ds['e1t'][:]).squeeze() * np.asarray(ds['e2t'][:]).squeeze()
        if area.shape != ocean.shape:
            raise ValueError(f'Geometry shape differs from the land mask: {geometry}')
        if 'bottom_level' in ds.variables and not np.array_equal(np.asarray(ds['bottom_level'][:]).squeeze() > 0, ocean):
            raise ValueError(f'Geometry wet cells differ from the land mask: {geometry}')
    if not np.isfinite(area[ocean]).all() or np.any(area[ocean] <= 0):
        raise ValueError('Cell areas must be finite and positive on ocean cells')
    return area


def forecasts(branch_dir):
    """{analysis date: icemod path} for the completed forecasts of one branch."""
    found = {}
    for folder in sorted(Path(branch_dir).glob('run_*T00:00:00+00:00')):
        paths = list(folder.glob('*_forecast.1h_icemod.nc'))
        if len(paths) != 1 or not (folder / 'forecast-done').is_file():
            continue
        start = date.fromisoformat(folder.name[4:14])
        found[str(start - timedelta(days=1))] = paths[0]
    return found


def read_forecast(path, folder_day):
    """SIC at the three leads; checks that record r ends at the expected valid date."""
    import netCDF4 as nc
    fields = {}
    with nc.Dataset(path) as ds:
        ds.set_auto_mask(False)
        sic, clock = ds['siconc'], ds['time_counter']
        for lead, index in LEADS.items():
            stamp = nc.num2date(clock[index], clock.units, calendar=getattr(clock, 'calendar', 'standard'))
            centre = datetime(stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute) + timedelta(hours=12)
            if centre.date() != folder_day + timedelta(days=lead // 24) or stamp.hour != 23:
                raise ValueError(f'Unexpected time of record {index} in {path}: {stamp}')
            raw = np.asarray(sic[index], np.float64)
            missing = ~np.isfinite(raw)
            for key in ('_FillValue', 'missing_value'):
                for value in np.asarray(getattr(sic, key, [])).reshape(-1):
                    missing |= raw == value
            fields[lead] = np.clip(np.where(missing, 0., raw), 0., 1.)
    return fields


def read_targets(path, ocean):
    from ..data.transform.sral_transform import sral_disc
    obs = np.load(path, allow_pickle=False)
    if obs.ndim != 3 or obs.shape[0] <= DENSE_CHANNEL or obs.shape[1:] != ocean.shape:
        raise ValueError(f'Observation channels or grid differ: {path}: {obs.shape}')
    dense = np.asarray(obs[DENSE_CHANNEL], np.float64)
    return {'amsr2_dense': np.where(np.isfinite(dense), dense, 0.),
            'sral_rule11': np.asarray(sral_disc[11]['transform'](obs.copy()), np.float64)}


def case_metrics(prediction, truth, mask, area):
    p, t = prediction[mask], truth[mask]
    if not len(p):
        raise ValueError('Empty verification support')
    mismatch = (p > .15) != (t > .15)
    return dict(n_pixels=int(mask.sum()), support_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                mae=float(np.mean(np.abs(p - t))), rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                iiee_km2=float(area[mask][mismatch].sum() / 1e6))


def score_branches(branches, observations, land_mask, geometry, dates=None):
    """Per-case rows for {name: directory}. ``dates``: optional analysis dates to score."""
    ocean = ~np.load(land_mask, allow_pickle=False).astype(bool)
    area = cell_area(geometry, ocean)
    found = {name: forecasts(path) for name, path in branches.items()}
    days = sorted(set().union(*found.values()) if dates is None else dates)
    rows = []
    for day in days:
        folder_day = date.fromisoformat(day) + timedelta(days=1)
        valid = {lead: folder_day + timedelta(days=lead // 24) for lead in LEADS}
        obs_files = {lead: Path(observations) / f'{d}.npy' for lead, d in valid.items()}
        if not all(p.is_file() for p in obs_files.values()):
            continue
        targets = {lead: read_targets(p, ocean) for lead, p in obs_files.items()}
        for name, paths in found.items():
            if day not in paths:
                continue
            fields = read_forecast(paths[day], folder_day)
            for lead in LEADS:
                for target in TARGETS:
                    truth = targets[lead][target]
                    rows.append(dict(branch=name, analysis_date=day, lead_hours=lead, target=target,
                                     valid_date=str(valid[lead]), forecast_file=str(paths[day]),
                                     **case_metrics(fields[lead], truth, ocean & np.isfinite(truth), area)))
    return rows


# -------------------------------------------------------------------------------------------- statistics
def moving_block_indices(n, block_days, replicates, seed):
    """Indices of ``replicates`` moving-block resamples of n consecutive days (the paper's sampler)."""
    block = min(block_days, n)
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(replicates, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(replicates, -1)[:, :n]


def block_indices(dates, block_days, replicates, seed):
    """Moving-block bootstrap indices over consecutive dates (as in the paper's evaluation)."""
    days = [date.fromisoformat(d) for d in dates]
    if not days or any((b - a).days != 1 for a, b in zip(days, days[1:])):
        raise ValueError('The moving-block bootstrap requires consecutive calendar dates')
    return moving_block_indices(len(days), block_days, replicates, seed)


def interval(point, draws):
    low, high = np.percentile(draws, [2.5, 97.5])
    return dict(estimate=float(point), std=float(np.std(draws)), low=float(low), high=float(high))


def _errors(table, branch, days):
    return {(target, lead): np.array([[table[branch, day, lead, target][m] for m in ERRORS] for day in days])
            for target in TARGETS for lead in LEADS}


def compare(rows, baseline, reference=None, block_days=14, replicates=5000, seed=20261022):
    """Skill of every branch relative to ``baseline`` and paired differences from ``reference``."""
    table = {(r['branch'], r['analysis_date'], int(r['lead_hours']), r['target']): r for r in rows}
    branches = sorted({r['branch'] for r in rows})
    if baseline not in branches or (reference is not None and reference not in branches):
        raise ValueError('Baseline or reference branch has no scores')
    days = sorted(d for d in {r['analysis_date'] for r in rows}
                  if all((b, d, lead, t) in table for b in branches for lead in LEADS for t in TARGETS))
    if not days:
        raise ValueError('No date is scored for every branch, lead and target')
    for day in days:
        for lead in LEADS:
            for target in TARGETS:
                supports = {table[b, day, lead, target]['support_sha256'] for b in branches}
                if len(supports) != 1:
                    raise ValueError(f'Verification support differs between branches: {day} {lead} h {target}')
    weights = np.stack([np.bincount(i, minlength=len(days)) for i in block_indices(days, block_days, replicates, seed)])
    weights = weights / len(days)
    control = _errors(table, baseline, days)
    draws, skill = {}, {}
    for branch in branches:
        errors = _errors(table, branch, days)
        for name, leads in LEAD_SETS.items():
            per_target, per_target_draws, iiee, iiee_draws = {}, {}, [], []
            for target in TARGETS:
                a = np.mean([errors[target, lead] for lead in leads], axis=0)
                c = np.mean([control[target, lead] for lead in leads], axis=0)
                point = 1 - a.mean(axis=0) / c.mean(axis=0)
                boot = 1 - (weights @ a) / (weights @ c)
                per_target[target], per_target_draws[target] = point[:2].mean(), boot[:, :2].mean(axis=1)
                iiee.append(point[2]); iiee_draws.append(boot[:, 2])
            combined = np.mean(list(per_target.values())), np.mean(list(per_target_draws.values()), axis=0)
            draws[branch, name] = combined[1]
            skill[branch, name] = dict(
                branch=branch, leads=name, n_dates=len(days),
                **{f'{t}_mean_skill': float(per_target[t]) for t in TARGETS},
                **{f'combined_{k}': v for k, v in interval(*combined).items()},
                **{f'iiee_{k}': v for k, v in interval(np.mean(iiee), np.mean(iiee_draws, axis=0)).items()})
    differences = []
    if reference is not None:
        for branch in branches:
            if branch in (reference, baseline):
                continue
            for name in LEAD_SETS:
                point = skill[branch, name]['combined_estimate'] - skill[reference, name]['combined_estimate']
                differences.append(dict(branch=branch, reference=reference, leads=name,
                                        **interval(point, draws[branch, name] - draws[reference, name])))
    return days, [skill[k] for k in sorted(skill) if k[0] != baseline], differences
