#!/usr/bin/env python3
"""Paired comparison of assimilation methods with simultaneous 95% intervals (Supplementary Table S7).

Input: the local metrics.csv of each run (title, series model|assim, iteration = date index, value), as written
by the assimilation logger and read by evaluate_assimilation.py. All runs must cover the same consecutive dates
and share the background series of each target (same dates and verification cells).

For each target (``--target``, e.g. model_siconc in M2M; amsr2_siconc and sral_siconc in S2M) the mean skill is
the average of MAE and RMSE skill, 1 - mean analysis error / mean background error. Uncertainty comes from a
moving-block bootstrap (14-day blocks, 5000 replicates) with the same resampled dates for every method, so the
pairwise differences are paired. Pointwise intervals are bootstrap percentiles. Simultaneous intervals use one
multiplier per target: the 95th percentile, over replicates, of the largest absolute standardized deviation
across all pairs; interval = difference +- multiplier * bootstrap standard deviation.

Example (M2M, the eight VAE models and three classical baselines of Table 1):
  python scripts/compare_models.py --target model_siconc --out outputs/m2m_pairs \
      --run vae_1f_d512=outputs/m2m/vae_1f_d512/metrics.csv --run BLUE=outputs/m2m/blue/metrics.csv ...
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae_mda.evaluation.forecast_branches import moving_block_indices  # noqa: E402


def load(path, target):
    """{(metric, series): {iteration: value}} for one target of one run."""
    values = {}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            for metric in ('mae', 'rmse'):
                if row['title'] == f'{target}_{metric}' and row['series'] in ('model', 'assim'):
                    values.setdefault((metric, row['series']), {})[int(row['iteration'])] = float(row['value'])
    if len(values) != 4:
        raise ValueError(f'{path}: missing {target}_mae/_rmse model and assim series')
    return values


def mean_skill(values, days, indices):
    """Point estimate and bootstrap draws of the mean of MAE and RMSE skill."""
    points, draws = [], []
    for metric in ('mae', 'rmse'):
        a = np.array([values[metric, 'assim'][d] for d in days]); b = np.array([values[metric, 'model'][d] for d in days])
        points.append(1 - a.mean() / b.mean()); draws.append(1 - a[indices].mean(axis=1) / b[indices].mean(axis=1))
    return float(np.mean(points)), np.mean(draws, axis=0)


def compare(runs, target, block_days=14, replicates=5000, seed=20261022):
    loaded = {name: load(path, target) for name, path in runs.items()}
    days = sorted(set.intersection(*(set(v[m, s]) for v in loaded.values() for m in ('mae', 'rmse') for s in ('model', 'assim'))))
    if not days or days != list(range(days[0], days[0] + len(days))):
        raise ValueError(f'{target}: runs must share consecutive date indices')
    first = next(iter(loaded.values()))
    for name, values in loaded.items():
        for metric in ('mae', 'rmse'):
            reference = np.array([first[metric, 'model'][d] for d in days])
            if not np.allclose([values[metric, 'model'][d] for d in days], reference, rtol=0, atol=2e-7):
                raise ValueError(f'{target}: background {metric} of {name} differs; runs are not on the same cases')
    indices = moving_block_indices(len(days), block_days, replicates, seed)
    skill = {name: mean_skill(values, days, indices) for name, values in loaded.items()}
    pairs = []
    for a, b in combinations(runs, 2):
        estimate, draws = skill[a][0] - skill[b][0], skill[a][1] - skill[b][1]
        pairs.append((a, b, estimate, draws, float(np.std(draws))))
    critical = float(np.percentile(np.max(np.array([np.abs(d - e) / s for _, _, e, d, s in pairs]), axis=0), 95))
    rows = []
    for a, b, estimate, draws, std in pairs:
        low, high = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(target=target, a=a, b=b, estimate=estimate, bootstrap_std=std, ci95_low=float(low),
                         ci95_high=float(high), sim_low=estimate - critical * std, sim_high=estimate + critical * std,
                         multiplier=critical, n_dates=len(days)))
    return {name: dict(target=target, run=name, mean_skill=s[0], bootstrap_std=float(np.std(s[1])), n_dates=len(days))
            for name, s in skill.items()}, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, metavar='NAME=metrics.csv')
    parser.add_argument('--target', action='append', required=True, help='metric prefix, e.g. model_siconc')
    parser.add_argument('--block-days', type=int, default=14)
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    runs = dict(item.split('=', 1) for item in args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    skills, pairs = [], []
    for target in args.target:
        s, p = compare(runs, target, args.block_days, args.replicates, args.seed)
        skills += list(s.values()); pairs += p
        print(f'{target}: {len(runs)} runs, {p[0]["n_dates"]} dates, {len(p)} pairs, simultaneous multiplier {p[0]["multiplier"]:.3f}')
        for row in p:
            flag = '' if row['sim_low'] <= 0 <= row['sim_high'] else '  *'
            print(f'  {row["a"]} - {row["b"]}: {row["estimate"]:+.4f}  simultaneous [{row["sim_low"]:+.4f}, {row["sim_high"]:+.4f}]{flag}')
    for name, rows in (('skill.csv', skills), ('pairs.csv', pairs)):
        with (args.out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
