"""Geordie Miner — text mining pipeline for academic corpora.

Usage:
    python geordie_miner.py DATA_DIR [DATA_DIR ...] [options]

Examples:
    python geordie_miner.py data/fulltext
    python geordie_miner.py data/fulltext data/no_refs data/no_method
    python geordie_miner.py data/fulltext --stages topics
    python geordie_miner.py data/fulltext --config config/config.txt

Stages (in order):
    ingest     PDF -> text + numbered prefix
    preprocess stopwords, substitutions, low-frequency filter
    terms      term frequencies, TF-IDF, word clouds
    phrases    n-grams, co-occurrence, network, dendrogram
    topics     KMeans, LDA, NMF, HDP

Run `python geordie_miner.py --help` for full argument reference.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List

# Make src/ importable when invoked from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import Config, init_directories, load_config, write_config_log  # noqa: E402
from ingest import convert_pdfs_to_text, ensure_nltk_resources  # noqa: E402
from logger import make_logger  # noqa: E402
from phrases import run_hierarchical_clustering, run_phrase_analysis, load_processed_corpus  # noqa: E402
from preprocess import descriptive_stats, load_stopwords, load_substitutions, preprocess_corpus  # noqa: E402
from terms import run_term_analysis  # noqa: E402
from topics import run_topic_models  # noqa: E402


ALL_STAGES = ["ingest", "preprocess", "terms", "phrases", "topics"]
DEFAULT_CONFIG = os.path.join("config", "config.txt")
DEFAULT_OUTPUT_BASE = "output"


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="geordie_miner",
        description="Text mining pipeline: PDF -> tokens -> frequencies / topics / networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See README.md for full documentation.",
    )
    p.add_argument("data_dirs", nargs="+", help="One or more directories of PDFs/.txt files to process.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Path to .ini config file (default: {DEFAULT_CONFIG}).")
    p.add_argument("--out", default=DEFAULT_OUTPUT_BASE, help=f"Base directory for analysis output (default: {DEFAULT_OUTPUT_BASE}).")
    p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages to run. Available: {','.join(ALL_STAGES)}. Default: all.",
    )
    return p.parse_args(argv)


def run_one(cfg: Config, stages: List[str]) -> None:
    """Run the pipeline once for a single data directory."""
    init_directories(cfg, stages)
    log = make_logger(os.path.join(cfg.directory_analysis, "_log_output.log"))
    write_config_log(cfg)

    log("")
    log("=========================================================")
    log("Geordie Miner")
    log("=========================================================")
    log(f"Runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config:  {cfg.config_path}")
    log(f"Data:    {cfg.directory_data}")
    log(f"Output:  {cfg.directory_analysis}")
    log(f"Stages:  {', '.join(stages)}")
    log("")

    ensure_nltk_resources(log)

    if "ingest" in stages:
        log("--- Stage: ingest ---")
        convert_pdfs_to_text(cfg.directory_data, cfg.directory_text, log)
        descriptive_stats(cfg, cfg.directory_text, log)

    if "preprocess" in stages:
        log("--- Stage: preprocess ---")
        stopwords = load_stopwords(cfg, log)
        substitutions = load_substitutions(cfg, log)
        preprocess_corpus(cfg, stopwords, substitutions, log)
        descriptive_stats(cfg, cfg.directory_processed, log)

    if "terms" in stages:
        log("--- Stage: terms ---")
        corpus = _read_processed_as_token_lists(cfg)
        run_term_analysis(cfg, corpus, log)

    if "phrases" in stages:
        log("--- Stage: phrases ---")
        tokens, _ = load_processed_corpus(cfg)
        cooccurrence = run_phrase_analysis(cfg, tokens, log)
        run_hierarchical_clustering(cfg, cooccurrence, log)

    if "topics" in stages:
        log("--- Stage: topics ---")
        run_topic_models(cfg, log)

    log("")
    log(f"Done. Outputs in: {cfg.directory_analysis}")
    log("")


def _read_processed_as_token_lists(cfg: Config) -> List[List[str]]:
    """Read processed/*.txt and return one whitespace-tokenised list per document."""
    corpus: List[List[str]] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            corpus.append(f.read().split())
    return corpus


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        print(f"Error: unknown stage(s): {unknown}. Available: {ALL_STAGES}", file=sys.stderr)
        return 2

    for data_dir in args.data_dirs:
        try:
            cfg = load_config(args.config, data_dir, output_base=args.out)
            run_one(cfg, stages)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error processing '{data_dir}': {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
