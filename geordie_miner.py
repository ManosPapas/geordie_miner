"""Geordie Miner — text mining pipeline for academic corpora.

One CLI with three subcommands:

    python geordie_miner.py run     DATA_DIR [DATA_DIR ...]   # one or more corpora
    python geordie_miner.py batch                              # all ./data/* + comparison
    python geordie_miner.py compare [DIR ...]                  # standalone comparison

Examples:
    python geordie_miner.py run data/fulltext
    python geordie_miner.py run data/fulltext data/no_refs --stages topics
    python geordie_miner.py batch --top 100
    python geordie_miner.py compare output/fulltext output/no_refs

Run `python geordie_miner.py <subcommand> --help` for per-subcommand options.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from typing import List

# Make src/ importable when invoked from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from coherence import compute_and_save as compute_coherence  # noqa: E402
import compare as cmp  # noqa: E402
from config import Config, init_directories, load_config, write_config_log  # noqa: E402
from ingest import convert_pdfs_to_text, ensure_nltk_resources  # noqa: E402
from logger import make_logger  # noqa: E402
from phrases import run_hierarchical_clustering, run_phrase_analysis  # noqa: E402
from preprocess import descriptive_stats, load_stopwords, load_substitutions, preprocess_corpus  # noqa: E402
from summary import write_summary  # noqa: E402
from terms import run_term_analysis  # noqa: E402
from topics import run_topic_models  # noqa: E402


ALL_STAGES = ["ingest", "preprocess", "terms", "phrases", "topics"]
DEFAULT_CONFIG = os.path.join("config", "config.ini")
DEFAULT_OUTPUT_BASE = "output"
DEFAULT_DATA_BASE = "data"
DEFAULT_COMPARE_TOP_N = 50


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geordie_miner",
        description="Text mining pipeline: PDF -> tokens -> frequencies / topics / networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See README.md for full documentation.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{run,batch,compare}")

    run_p = sub.add_parser("run", help="Run the pipeline on one or more data directories.")
    run_p.add_argument("data_dirs", nargs="+", help="One or more directories of PDFs/.txt files.")
    run_p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Path to config file (default: {DEFAULT_CONFIG}).")
    run_p.add_argument("--out", default=DEFAULT_OUTPUT_BASE, help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).")
    run_p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages: {','.join(ALL_STAGES)}. Default: all.",
    )

    batch_p = sub.add_parser(
        "batch",
        help="Run pipeline on every ./data/* directory and write a comparison report.",
    )
    batch_p.add_argument("data_dirs", nargs="*", help=f"Data directories. Empty = auto-discover {DEFAULT_DATA_BASE}/*.")
    batch_p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config file (default: {DEFAULT_CONFIG}).")
    batch_p.add_argument("--out", default=DEFAULT_OUTPUT_BASE, help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).")
    batch_p.add_argument("--stages", default=",".join(ALL_STAGES), help="Stages (comma-separated).")
    batch_p.add_argument("--no-compare", action="store_true", help="Skip the cross-run comparison report.")
    batch_p.add_argument("--top", type=int, default=DEFAULT_COMPARE_TOP_N, help=f"Top-N terms for comparison (default: {DEFAULT_COMPARE_TOP_N}).")

    cmp_p = sub.add_parser("compare", help="Compare existing output directories without re-running the pipeline.")
    cmp_p.add_argument("dirs", nargs="*", help=f"Output directories. Empty = auto-discover {DEFAULT_OUTPUT_BASE}/*.")
    cmp_p.add_argument("--report", default="comparison_report.md", help="Markdown output path (default: comparison_report.md).")
    cmp_p.add_argument("--top", type=int, default=DEFAULT_COMPARE_TOP_N, help=f"Top-N terms (default: {DEFAULT_COMPARE_TOP_N}).")

    return p


def _parse_stages(value: str) -> List[str]:
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        sys.exit(f"Error: unknown stage(s) {unknown}. Available: {ALL_STAGES}")
    return stages


def run_pipeline(cfg: Config, stages: List[str]) -> None:
    """Execute the configured stages on a single corpus."""
    init_directories(cfg, stages)
    log = make_logger(cfg.log_path("run.log"))
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

    topics_artefacts: dict = {}

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
        tokens = _flatten_processed_tokens(cfg)
        cooccurrence = run_phrase_analysis(cfg, tokens, log)
        run_hierarchical_clustering(cfg, cooccurrence, log)

    if "topics" in stages:
        log("--- Stage: topics ---")
        topics_artefacts = run_topic_models(cfg, log)
        if topics_artefacts:
            log("--- Stage: coherence ---")
            compute_coherence(
                cfg,
                topics_artefacts["tokenised_docs"],
                topics_artefacts["top_words"],
                log,
            )

    log("--- Writing summary ---")
    write_summary(cfg, log)

    log("")
    log(f"Done. Outputs in: {cfg.directory_analysis}")
    log("")


def _read_processed_as_token_lists(cfg: Config) -> List[List[str]]:
    corpus: List[List[str]] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            corpus.append(f.read().split())
    return corpus


def _flatten_processed_tokens(cfg: Config) -> List[str]:
    tokens: List[str] = []
    from preprocess import lemmatise_text  # local import to avoid loading at module import time

    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            tokens.extend(lemmatise_text(f.read()))
    return tokens


def cmd_run(args: argparse.Namespace) -> int:
    stages = _parse_stages(args.stages)
    for data_dir in args.data_dirs:
        try:
            cfg = load_config(args.config, data_dir, output_base=args.out)
            run_pipeline(cfg, stages)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error processing '{data_dir}': {e}", file=sys.stderr)
            return 1
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    dirs = args.data_dirs or sorted(
        d for d in glob.glob(os.path.join(DEFAULT_DATA_BASE, "*")) if os.path.isdir(d)
    )
    if not dirs:
        print(
            f"No data directories found. Pass one or more, or place them under ./{DEFAULT_DATA_BASE}/*.",
            file=sys.stderr,
        )
        return 1

    print(f"Running pipeline on {len(dirs)} corpora: {', '.join(dirs)}")
    run_args = argparse.Namespace(
        data_dirs=dirs, config=args.config, out=args.out, stages=args.stages
    )
    rc = cmd_run(run_args)
    if rc != 0:
        return rc

    if not args.no_compare:
        output_dirs = [os.path.join(args.out, os.path.basename(os.path.normpath(d))) for d in dirs]
        output_dirs = [d for d in output_dirs if os.path.isdir(d)]
        if output_dirs:
            cmp.write_report("comparison_report.md", output_dirs, args.top)
            print(f"Comparison report written: comparison_report.md ({len(output_dirs)} run(s))")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    dirs = args.dirs or cmp.discover_dirs(DEFAULT_OUTPUT_BASE)
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print(
            f"No output directories provided or discovered under ./{DEFAULT_OUTPUT_BASE}/*.",
            file=sys.stderr,
        )
        return 1
    cmp.write_report(args.report, dirs, args.top)
    print(f"Comparison report written: {args.report} ({len(dirs)} run(s))")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "compare":
        return cmd_compare(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
