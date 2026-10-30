#!/usr/bin/env python3
"""Score forecast branches against AMSR2 and compare them (Table 3, Fig. 4b, Supplementary Tables S11-S12).

  score    per-case MAE, RMSE and IIEE of every branch, including the baseline forecast -> case_scores.csv
  compare  skill relative to the baseline forecast and paired differences from a reference branch

Example (the 2023 selected-configuration forecasts):
  python scripts/compare_forecast_branches.py score --out outputs/forecast_2023 \
      --branch control=/path/to/vae-mda-outputs/forecast_2023/control \
      --branch s2m_vae_1f_d512=/path/to/vae-mda-outputs/forecast_2023/s2m_vae_1f_d512 ... \
      --observations /path/to/vae-mda-data/sral \
      --land-mask /path/to/vae-mda-data/static/land_mask.npy --geometry /path/to/vae-mda-data/static/BARKA12_domain_cfg_v1.3.nc
  python scripts/compare_forecast_branches.py compare --scores outputs/forecast_2023/case_scores.csv \
      --baseline control --reference s2m_vae_1f_d512 --out outputs/forecast_2023
See vae_mda/evaluation/forecast_branches.py for the folder layout and the statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_scores(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline='') as stream:
            for row in csv.DictReader(stream):
                row['lead_hours'] = int(row['lead_hours'])
                for key in ('mae', 'rmse', 'iiee_km2'):
                    row[key] = float(row[key])
                rows.append(row)
    return rows


def score(args):
    from vae_mda.evaluation.forecast_branches import score_branches
    branches = dict(item.split('=', 1) for item in args.branch)
    rows = score_branches(branches, args.observations, args.land_mask, args.geometry)
    if not rows:
        raise SystemExit('No forecast was scored; check the branch folders and observation files')
    write_csv(args.out / 'case_scores.csv', rows)
    print(f'{len(rows)} case rows for {len(branches)} branches -> {args.out / "case_scores.csv"}')


def compare(args):
    from vae_mda.evaluation.forecast_branches import compare as paired
    days, skill, differences = paired(read_scores(args.scores), args.baseline, args.reference,
                                      args.block_days, args.replicates, args.seed)
    write_csv(args.out / 'skill.csv', skill)
    if differences:
        write_csv(args.out / 'differences.csv', differences)
    (args.out / 'summary.json').write_text(json.dumps(dict(
        baseline=args.baseline, reference=args.reference, n_dates=len(days), first_date=days[0], last_date=days[-1],
        bootstrap=dict(block_days=args.block_days, replicates=args.replicates, seed=args.seed)), indent=1) + '\n')
    print(f'{len(days)} common dates ({days[0]} to {days[-1]}); combined skill relative to {args.baseline}:')
    ranked = sorted({r['branch'] for r in skill}, key=lambda b: -next(r['combined_estimate'] for r in skill
                                                                    if r['branch'] == b and r['leads'] == '24-72'))
    print(f'  {"branch":42s}' + ''.join(f'{lead:>16s}' for lead in ('24', '48', '72', '24-72')))
    for branch in ranked:
        cells = {r['leads']: r for r in skill if r['branch'] == branch}
        print(f'  {branch:42s}' + ''.join(f'{cells[k]["combined_estimate"]:9.3f} ±{cells[k]["combined_std"]:.3f}'
                                          for k in ('24', '48', '72', '24-72')))
    for row in differences:
        if row['leads'] == '72':
            print(f'  72 h: {row["branch"]} - {row["reference"]} = {row["estimate"]:+.3f} [{row["low"]:+.3f}, {row["high"]:+.3f}]')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score', help='score forecast branches')
    s.add_argument('--branch', action='append', required=True, metavar='NAME=DIR')
    s.add_argument('--observations', type=Path, required=True, help='daily <date>.npy observation files')
    s.add_argument('--land-mask', type=Path, required=True)
    s.add_argument('--geometry', type=Path, required=True, help='NEMO domain_cfg with e1t/e2t')
    s.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare', help='skill and paired differences from case scores')
    c.add_argument('--scores', type=Path, nargs='+', required=True)
    c.add_argument('--baseline', required=True, help='branch without assimilation')
    c.add_argument('--reference', help='branch for paired differences (e.g. s2m_vae_1f_d512)')
    c.add_argument('--block-days', type=int, default=14)
    c.add_argument('--replicates', type=int, default=5000)
    c.add_argument('--seed', type=int, default=20261022)
    c.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    {'score': score, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
