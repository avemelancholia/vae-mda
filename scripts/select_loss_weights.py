#!/usr/bin/env python3
"""Select the background and latent loss weights of each VAE model (Supplementary Note on loss-weight selection).

For one experiment (--setting M2M or S2M), every (w_b, w_z) pair of config/experiments/loss_weights.yaml is fitted
on the 60 development dates with the model's base config; the observation weight w_y keeps its base value (1 in
M2M, 6 in S2M). Each pair runs scripts/run_assimilation.py on a derived config (sampled dates, the pair's weights,
local logging) and is scored from its metrics.csv:
  M2M  mean over the fields the model decodes of the mean skill against the following-year NEMO field;
  S2M  combined skill: mean of the SIC mean skill against dense AMSR2 and sparse SRAL-conditioned AMSR2.
Mean skill = average of MAE and RMSE skill; skill = 1 - mean analysis error / mean background error over the dates.
The highest score is selected. Completed runs are reused, so an interrupted search resumes.

  python scripts/select_loss_weights.py --setting S2M                 # all models
  python scripts/select_loss_weights.py --setting M2M --models vae_4f_um --dry-run
  python scripts/select_loss_weights.py --setting S2M --score-only    # rank existing runs only
"""
from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / 'config/experiments/loss_weights.yaml'
TARGETS = {'M2M': 'model', 'S2M': ('amsr2', 'sral')}


def pairs(plan, setting, model):
    grid = plan[setting]
    points = [(b, z) for b in grid['background_weights'] for z in grid['latent_weights']]
    points += [tuple(p) for p in grid.get('extra_pairs', {}).get(model, []) if tuple(p) not in points]
    return points


def run_name(model, wb, wz):
    return f'{model}_wb{wb:g}_wz{wz:g}'


def derived_config(plan, setting, model, wb, wz, out):
    """Write the data and assimilation configs of one grid point and return the assimilation config path."""
    base = yaml.safe_load((PROJECT_ROOT / plan['models'][model][setting]).read_text())
    data = yaml.safe_load((PROJECT_ROOT / base['data_config']).read_text())
    data.update(dataset_name='Fields3DVARSampledDS', sample_dates=list(plan['dates']))
    name = run_name(model, wb, wz)
    folder = out / 'configs' / setting
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / f'{name}_data.yaml'
    data_path.write_text(yaml.safe_dump(data, sort_keys=False))
    config = copy.deepcopy(base)
    config.update(data_config=str(data_path), mod_weight=wb, lat_weight=wz, logger='local', output_root=str(out),
                  project_name=f'loss_weights_{setting.lower()}', task_name=name)
    path = folder / f'{name}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path, config, data


def metrics_path(out, config):
    return out / config['project_name'] / f"{config['model_name']}_{config['task_name']}" / 'metrics.csv'


def load_metrics(path):
    """{title: {series: {iteration: value}}} of the model/assim series."""
    values = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['series'] in ('model', 'assim'):
                values.setdefault(row['title'], {}).setdefault(row['series'], {})[int(row['iteration'])] = float(row['value'])
    return values


def mean_skill(values, prefix, n_dates):
    skills = []
    for metric in ('mae', 'rmse'):
        series = values.get(f'{prefix}_{metric}', {})
        keys = sorted(set(series.get('model', {})) & set(series.get('assim', {})))
        if len(keys) != n_dates:
            return None
        model = np.array([series['model'][k] for k in keys]); assim = np.array([series['assim'][k] for k in keys])
        skills.append(1 - assim.mean() / model.mean())
    return float(np.mean(skills))


def score(setting, values, fields, n_dates):
    if setting == 'M2M':
        parts = {field: mean_skill(values, f'model_{field}', n_dates) for field in fields}
    else:
        parts = {target: mean_skill(values, f'{target}_siconc', n_dates) for target in TARGETS['S2M']}
    if any(v is None for v in parts.values()):
        return None, parts
    return float(np.mean(list(parts.values()))), parts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--setting', choices=('M2M', 'S2M'), required=True)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--models', nargs='+', help='default: all models of the plan')
    parser.add_argument('--out', type=Path, help='default: output_root of the plan')
    parser.add_argument('--dry-run', action='store_true', help='write the derived configs and list the runs')
    parser.add_argument('--score-only', action='store_true', help='rank existing runs; do not fit')
    args = parser.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text())
    out = (args.out or Path(plan['output_root'])).resolve()
    models = args.models or list(plan['models'])
    n_dates = len(plan['dates'])
    rows = []
    for model in models:
        for wb, wz in pairs(plan, args.setting, model):
            path, config, data = derived_config(plan, args.setting, model, wb, wz, out)
            metrics = metrics_path(out, config)
            if args.dry_run:
                print(f'{args.setting} {model} w_b={wb:g} w_z={wz:g}: {path}')
                continue
            if not metrics.is_file() and not args.score_only:
                print(f'Fitting {args.setting} {model} w_b={wb:g} w_z={wz:g}', flush=True)
                subprocess.run([sys.executable, str(PROJECT_ROOT / 'scripts/run_assimilation.py'), '--config', str(path)],
                               cwd=PROJECT_ROOT, check=True)
            if not metrics.is_file():
                continue
            fields = [f for f in data['back_field'] if f in data['val_field'][0]]
            objective, parts = score(args.setting, load_metrics(metrics), fields, n_dates)
            if objective is None:
                print(f'Incomplete run skipped: {metrics}')
                continue
            rows.append(dict(setting=args.setting, model=model, w_b=wb, w_z=wz, objective=objective,
                             **{f'mean_skill_{k}': v for k, v in parts.items()}))
    if args.dry_run or not rows:
        print('No completed runs to rank.' if not args.dry_run else f'{len(models)} models listed.')
        return 0
    fieldnames = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('setting', 'model', 'w_b', 'w_z', 'objective'), k))
    with (out / f'{args.setting.lower()}_grid.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    selected = []
    print(f'{args.setting} selection ({n_dates} development dates):')
    for model in models:
        ranked = sorted((r for r in rows if r['model'] == model), key=lambda r: -r['objective'])
        if not ranked:
            continue
        best = ranked[0]; selected.append(best)
        near = sum(r['objective'] >= best['objective'] - 0.01 for r in ranked)
        reference = plan.get('selected', {}).get(args.setting, {}).get(model)
        note = '' if reference is None else ('  (paper: same)' if [best['w_b'], best['w_z']] == reference
                                             else f'  (paper: w_b={reference[0]:g}, w_z={reference[1]:g})')
        print(f'  {model:36s} w_b={best["w_b"]:g} w_z={best["w_z"]:g} score {best["objective"]:.4f}; '
              f'{near}/{len(ranked)} pairs within 0.01{note}')
    with (out / f'{args.setting.lower()}_selected.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(selected)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
