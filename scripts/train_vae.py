#!/usr/bin/env python3
"""Train a VAE variant reported in the paper."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae_mda.config import add_common_arguments, load_cli_config, load_config, validate_references


def seed_everything(seed=42):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config):
    from torch.utils.data import Subset

    from vae_mda.data.vae_data import VAETrainDS
    from vae_mda.model.vae.model import create_model
    from vae_mda.trainer import Trainer

    if config.get("model_name") != "diffusers":
        raise ValueError("The public artifact supports the reported diffusers VAE family only")

    seed_everything(int(config.get("seed", 42)))
    data_config = load_config(config["data_config"])
    model_config = load_config(config["model_config"])
    train_dataset = VAETrainDS(data_config, tag="train")
    validation_dataset = VAETrainDS(data_config, tag="val")
    if "dataset_limit" in config:
        limit = int(config["dataset_limit"])
        train_dataset = Subset(train_dataset, range(min(limit, len(train_dataset))))
        validation_dataset = Subset(
            validation_dataset, range(min(limit, len(validation_dataset)))
        )
    print(
        f"Training samples: {len(train_dataset)}; "
        f"validation samples: {len(validation_dataset)}"
    )
    model = create_model(model_config)
    checkpoint = config.get("init_checkpoint_path")
    if checkpoint:
        model.from_pretrained(Path(checkpoint))
    Trainer(
        config,
        model,
        train_dataset,
        validation_dataset,
        data_config,
        model_config,
    ).train()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--dataset-limit", type=int, help="Limit train and validation samples"
    )
    parser.add_argument(
        "--epoch-limit", type=int, help="Cap epochs for a bounded smoke run"
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    config = load_cli_config(args)
    if args.dataset_limit is not None:
        if args.dataset_limit < 1:
            raise ValueError("--dataset-limit must be positive")
        config["dataset_limit"] = args.dataset_limit
    if args.epoch_limit is not None:
        if args.epoch_limit < 1:
            raise ValueError("--epoch-limit must be positive")
        config["epoch"] = min(int(config["epoch"]), args.epoch_limit)
    references = validate_references(config)
    if args.dry_run:
        print(json.dumps({"config": config, "references": [str(p) for p in references]}, indent=2))
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
