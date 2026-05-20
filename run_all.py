"""Run the Geordie Miner pipeline on every ./data/* directory, then write a comparison report.

Cross-platform replacement for the old run.bat. Usage:

    python run_all.py                                # auto-discovers ./data/*
    python run_all.py data/a data/b data/c           # explicit list
    python run_all.py --config config/config.txt --no-compare data/*
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import compare
import geordie_miner


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-run Geordie Miner over multiple corpora and compare results.")
    p.add_argument("data_dirs", nargs="*", help="Data directories. If empty, auto-discovers ./data/*.")
    p.add_argument("--config", default=geordie_miner.DEFAULT_CONFIG, help=f"Config file (default: {geordie_miner.DEFAULT_CONFIG}).")
    p.add_argument("--out", default=geordie_miner.DEFAULT_OUTPUT_BASE, help=f"Output base directory (default: {geordie_miner.DEFAULT_OUTPUT_BASE}).")
    p.add_argument("--stages", default=",".join(geordie_miner.ALL_STAGES), help="Stages to run (comma-separated).")
    p.add_argument("--no-compare", action="store_true", help="Skip the cross-run comparison report.")
    p.add_argument("--top", type=int, default=50, help="Top-N terms used in the comparison (default: 50).")
    return p.parse_args(argv)


def discover_data_dirs() -> List[str]:
    return sorted(d for d in glob.glob(os.path.join("data", "*")) if os.path.isdir(d))


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dirs = args.data_dirs or discover_data_dirs()
    if not dirs:
        print("No data directories found. Pass one or more, or place them under ./data/*.", file=sys.stderr)
        return 1

    print(f"Running pipeline on {len(dirs)} corpora: {', '.join(dirs)}")
    rc = geordie_miner.main([*dirs, "--config", args.config, "--out", args.out, "--stages", args.stages])
    if rc != 0:
        return rc

    if not args.no_compare:
        output_dirs = [os.path.join(args.out, os.path.basename(os.path.normpath(d))) for d in dirs]
        output_dirs = [d for d in output_dirs if os.path.isdir(d)]
        if output_dirs:
            compare.main([*output_dirs, "--top", str(args.top)])

    return 0


if __name__ == "__main__":
    sys.exit(main())
