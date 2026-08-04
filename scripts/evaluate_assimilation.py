#!/usr/bin/env python3
"""Summarize local assimilation logs with the paper's block-bootstrap skill."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae_mda.evaluation.assimilation_metrics import summarize_files, write_summary


def parse_run(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have the form NAME=metrics.csv")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("--run must have a non-empty name and path")
    return name, Path(raw_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=14)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args(argv)
    if args.block_size < 1 or args.samples < 2:
        raise ValueError("--block-size must be positive and --samples at least two")
    missing = [str(path) for _, path in args.run if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Metrics file(s) not found: {', '.join(missing)}")
    rows = summarize_files(
        args.run,
        block_size=args.block_size,
        samples=args.samples,
        seed=args.seed,
    )
    write_summary(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
