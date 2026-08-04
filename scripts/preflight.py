#!/usr/bin/env python3
"""Validate real data and weights before starting an experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae_mda.preflight import PreflightError, run_preflight


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--date", help="Override restart configuration date")
    parser.add_argument(
        "--load-sample",
        action="store_true",
        help="Open and transform the first real sample after structural checks",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        messages = run_preflight(
            args.config, load_sample=args.load_sample, run_date=args.date
        )
    except (PreflightError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2
    for message in messages:
        print(f"[ok] {message}")
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
