#!/usr/bin/env python3
"""Apply VAE-MDA analyses to a bounded range of NEMO/SI3 restart dates."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae_mda.config import add_common_arguments, load_cli_config, load_config, validate_references


def resolve_device(config):
    import torch

    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        config["device"] = "cpu"
    return config


def wait_for(path: Path, *, watch: bool, poll_seconds: float):
    if not watch:
        if not path.exists():
            raise FileNotFoundError(
                f"Required forecast marker is missing: {path}; use --watch to poll"
            )
        return
    while not path.exists():
        print(f"Waiting for {path}", flush=True)
        time.sleep(poll_seconds)


def run_date(config):
    from torch.utils.data import DataLoader

    from vae_mda.assimilator import InferenceHandler
    from vae_mda.data.inference_data import InferenceDS
    from vae_mda.model.vae_model import VAEModel3Dvar, VAEModel3DvarEXP
    from vae_mda.model.var_model import Model3Dvar, Model3DvarEXP

    models = {
        cls.name: cls
        for cls in (Model3Dvar, Model3DvarEXP, VAEModel3Dvar, VAEModel3DvarEXP)
    }
    data_config = load_config(config["data_config"])
    model_config = load_config(config["model_config"])
    dataset = InferenceDS(config, data_config)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), shuffle=False)
    try:
        model_class = models[config["model_name"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported public forecast model: {exc.args[0]}") from exc
    InferenceHandler(
        config,
        dataloader=loader,
        model_class=model_class,
        model_config=model_config,
        data_config=data_config,
    ).run_assimilation()


def inclusive_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--start-date", help="First ISO date; defaults to config date")
    parser.add_argument("--end-date", help="Last ISO date; defaults to start date")
    parser.add_argument("--watch", action="store_true", help="Poll for forecast-done markers")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    base_config = load_cli_config(args)
    references = validate_references(base_config)
    start = date.fromisoformat(args.start_date or base_config["date"])
    end = date.fromisoformat(args.end_date or start.isoformat())
    if end < start:
        raise ValueError("--end-date must not precede --start-date")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": base_config,
                    "references": [str(p) for p in references],
                    "dates": [start.isoformat(), end.isoformat()],
                },
                indent=2,
            )
        )
        return 0

    from vae_mda.logger.backend import init_task

    project_name = base_config.get("project_name", "paper_forecast_assimilation")
    task_name = base_config.get("task_name", Path(args.config).stem)
    task = init_task(
        base_config,
        project_name=project_name,
        task_name=f"{base_config['model_name']}_{task_name}",
    )
    base_config = task.connect(base_config)
    task.connect(load_config(base_config["model_config"]))
    task.connect(load_config(base_config["data_config"]))
    logger = task.get_logger()
    try:
        for iteration, day in enumerate(inclusive_dates(start, end)):
            config = resolve_device(copy.deepcopy(base_config))
            config["date"] = day.isoformat()
            run_dir = Path(config["restart_dir"]) / f"run_{day.isoformat()}T00:00:00+00:00"
            wait_for(run_dir / "forecast-done", watch=args.watch, poll_seconds=args.poll_seconds)
            print(f"Applying restart analysis for {day.isoformat()}", flush=True)
            run_date(config)
            (run_dir / "assim-done").touch()
            logger.report_scalar(
                title="forecast_cycle",
                series="completed",
                value=1,
                iteration=iteration,
            )
    finally:
        task.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
