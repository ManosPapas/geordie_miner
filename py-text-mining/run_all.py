"""Run the Geordie Miner pipeline on every ./data_* directory, then write a comparison report.

Cross-platform replacement for the old run.bat. Usage:

    python run_all.py                          # auto-discovers ./data_*
    python run_all.py data_a data_b data_c     # explicit list
    python run_all.py --config config.txt --no-compare data_*
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
    p.add_argument("data_dirs", nargs="*", help="Data directories. If empty, auto-discovers ./data_*.")
    p.add_argument("--config", default="config.txt", help="Config file (default: config.txt).")
    p.add_argument("--stages", default=",".join(geordie_miner.ALL_STAGES), help="Stages to run (comma-separated).")
    p.add_argument("--no-compare", action="store_true", help="Skip the cross-run comparison report.")
    p.add_argument("--top", type=int, default=50, help="Top-N terms used in the comparison (default: 50).")
    return p.parse_args(argv)


def discover_data_dirs() -> List[str]:
    return sorted(d for d in glob.glob("data_*") if os.path.isdir(d))


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dirs = args.data_dirs or discover_data_dirs()
    if not dirs:
        print("No data directories found. Pass one or more, or place them under ./data_*.", file=sys.stderr)
        return 1

    print(f"Running pipeline on {len(dirs)} corpora: {', '.join(dirs)}")
    rc = geordie_miner.main([args.config, *dirs, "--stages", args.stages])
    if rc != 0:
        return rc

    if not args.no_compare:
        analysis_dirs = [f"analysis_{os.path.basename(os.path.normpath(d))}" for d in dirs]
        analysis_dirs = [d for d in analysis_dirs if os.path.isdir(d)]
        if len(analysis_dirs) >= 1:
            compare.main([*analysis_dirs, "--top", str(args.top)])

    return 0


if __name__ == "__main__":
    sys.exit(main())
