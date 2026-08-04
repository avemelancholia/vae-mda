#!/usr/bin/env python3
"""Run one paper VAE-MDA or classical-baseline assimilation configuration."""

from __future__ import annotations

import argparse
import json
import sys
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


def run(config):
    from torch.utils.data import DataLoader, Subset

    from vae_mda.assimilator import AssimilationHandler
    from vae_mda.data.assimilation_data import (
        Fields3DVARDS,
        Fields3DVAREXPDS,
        Fields3DVARSampledDS,
    )
    from vae_mda.model.blue_model import ModelBLUE
    from vae_mda.model.enkf_model import ModelEnKF
    from vae_mda.model.vae_model import VAEModel3Dvar, VAEModel3DvarEXP
    from vae_mda.model.var_model import Model3Dvar, Model3DvarEXP

    models = {
        cls.name: cls
        for cls in (
            Model3Dvar,
            Model3DvarEXP,
            ModelBLUE,
            ModelEnKF,
            VAEModel3Dvar,
            VAEModel3DvarEXP,
        )
    }
    datasets = {
        cls.name: cls
        for cls in (Fields3DVARDS, Fields3DVAREXPDS, Fields3DVARSampledDS)
    }

    data_config = load_config(config["data_config"])
    if "depth" in config:
        data_config["depth"] = config["depth"]
    if "diagnostic_dir" in config:
        data_config["diagnostic_dir"] = config["diagnostic_dir"]
    try:
        dataset_class = datasets[data_config["dataset_name"]]
        model_class = models[config["model_name"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported public dataset/model name: {exc.args[0]}") from exc

    dataset = dataset_class(data_config)
    if "dataset_limit" in config:
        dataset = Subset(dataset, range(min(int(config["dataset_limit"]), len(dataset))))
    print(f"Dataset length: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
    )
    model_config = load_config(config["model_config"])
    handler = AssimilationHandler(
        config,
        dataloader=loader,
        model_class=model_class,
        model_config=model_config,
        data_config=data_config,
    )
    handler.run_assimilation()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--dataset-limit", type=int, help="Limit cases for smoke runs")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    config = load_cli_config(args)
    if args.dataset_limit is not None:
        if args.dataset_limit < 1:
            raise ValueError("--dataset-limit must be positive")
        config["dataset_limit"] = args.dataset_limit
    references = validate_references(config)
    if args.dry_run:
        print(json.dumps({"config": config, "references": [str(p) for p in references]}, indent=2))
        return 0
    run(resolve_device(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
