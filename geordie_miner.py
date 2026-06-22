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

# Cap loky/joblib workers BEFORE any sklearn import (the imports below pull in
# sklearn). Also silences the "Could not find the number of physical cores"
# loky warning on some machines. Re-applied per-run from config.max_cpu_count.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# Make src/ importable when invoked from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from coherence import compute_and_save as compute_coherence  # noqa: E402
import compare as cmp  # noqa: E402
from config import Config, init_directories, load_config, write_config_log  # noqa: E402
from ingest import ensure_nltk_resources, ingest_corpus  # noqa: E402
from logger import make_logger, log_stage  # noqa: E402
from phrases import run_hierarchical_clustering, run_phrase_analysis  # noqa: E402
from preprocess import descriptive_stats, load_stopwords, load_substitutions, preprocess_corpus  # noqa: E402
from summary import write_summary  # noqa: E402
from terms import run_term_analysis  # noqa: E402
from topics import run_topic_models  # noqa: E402


ALL_STAGES = [
    "ingest",
    "metadata",        # extract + enrich bibliometric fields, provenance, entity tables
    "references",      # citation network + impact / co-citation / coupling
    "collaboration",   # optional co-authorship networks (gated by collaboration_enable)
    "bibliometrics",   # descriptive bibliometrics + charts + Excel
    "preprocess",
    "annotate",        # optional linguistic annotation (gated by annotation_enable)
    "lexical",         # optional dictionary concept analysis (gated by lexical_enable)
    "terms",
    "phrases",
    "topics",          # KMeans, LDA, NMF, HDP and (when installed) BERTopic
    "stability",       # multi-seed stability check
    "topic_evolution", # optional topic prevalence/splits/merges over time (gated)
    "map",             # 2D document map (needs BERTopic embeddings)
    "science_map",     # optional journal map + thematic evolution (gated)
    "longitudinal",    # optional per-period analysis (gated by longitudinal_enable)
]
DEFAULT_CONFIG = os.path.join("config", "config.txt")
DEFAULT_OUTPUT_BASE = "output"
DEFAULT_DATA_BASE = "data"
DEFAULT_COMPARE_TOP_N = 50
DEFAULT_COMPARE_REPORT = os.path.join("output", "comparison_report.md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geordie_miner",
        description="Text mining pipeline: PDF -> tokens -> frequencies / topics / networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See README.md for full documentation.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{run,batch,compare,fetch}")

    run_p = sub.add_parser("run", help="Run the pipeline on one or more data directories.")
    run_p.add_argument("data_dirs", nargs="+", help="One or more directories of PDFs/.txt/.bib/.ris/.csv files.")
    run_p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Path to config file (default: {DEFAULT_CONFIG}).")
    run_p.add_argument(
        "--profile",
        default=None,
        choices=["bibliometrics", "full_text", "balanced"],
        help="Use a bundled config profile from config/profiles/ (ignored when --config is given explicitly).",
    )
    run_p.add_argument("--out", default=DEFAULT_OUTPUT_BASE, help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).")
    run_p.add_argument(
        "--name",
        default=None,
        help=(
            "Override the output sub-folder name. "
            "Default is the data folder's basename. "
            "Use this to keep multiple analyses of the same data (different configs / K) "
            "side-by-side under output/<name>/. Only valid with a single DATA_DIR."
        ),
    )
    run_p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages: {','.join(ALL_STAGES)}. Default: all.",
    )
    run_p.add_argument(
        "--no-coherence",
        action="store_true",
        help=(
            "Skip the coherence stage (c_v + u_mass scoring). "
            "Saves ~5–18 minutes on a 100-paper corpus. "
            "Use when iterating; re-enable when you want to pick the best K objectively."
        ),
    )

    batch_p = sub.add_parser(
        "batch",
        help="Run pipeline on every ./data/* directory and write a comparison report.",
    )
    batch_p.add_argument("data_dirs", nargs="*", help=f"Data directories. Empty = auto-discover {DEFAULT_DATA_BASE}/*.")
    batch_p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config file (default: {DEFAULT_CONFIG}).")
    batch_p.add_argument("--profile", default=None, choices=["bibliometrics", "full_text", "balanced"], help="Bundled config profile (ignored when --config is given).")
    batch_p.add_argument("--out", default=DEFAULT_OUTPUT_BASE, help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).")
    batch_p.add_argument("--stages", default=",".join(ALL_STAGES), help="Stages (comma-separated).")
    batch_p.add_argument("--no-compare", action="store_true", help="Skip the cross-run comparison report.")
    batch_p.add_argument("--no-coherence", action="store_true", help="Skip the coherence stage across all runs.")
    batch_p.add_argument("--top", type=int, default=DEFAULT_COMPARE_TOP_N, help=f"Top-N terms for comparison (default: {DEFAULT_COMPARE_TOP_N}).")

    cmp_p = sub.add_parser("compare", help="Compare existing output directories without re-running the pipeline.")
    cmp_p.add_argument("dirs", nargs="*", help=f"Output directories. Empty = auto-discover {DEFAULT_OUTPUT_BASE}/*.")
    cmp_p.add_argument("--report", default=DEFAULT_COMPARE_REPORT, help=f"Markdown output path (default: {DEFAULT_COMPARE_REPORT}).")
    cmp_p.add_argument("--top", type=int, default=DEFAULT_COMPARE_TOP_N, help=f"Top-N terms (default: {DEFAULT_COMPARE_TOP_N}).")

    fetch_p = sub.add_parser("fetch", help="Build a corpus by keyword search from an external provider (OpenAlex/Crossref/Scopus).")
    fetch_p.add_argument("query", help="Keyword query, e.g. \"virtual reality retail\".")
    fetch_p.add_argument("--provider", default="openalex", choices=["openalex", "crossref", "scopus"], help="Source (default: openalex; scopus needs SCOPUS_API_KEY).")
    fetch_p.add_argument("--limit", type=int, default=200, help="Max records to fetch (default: 200).")
    fetch_p.add_argument("--out", required=True, help="Data directory to create, e.g. data/myquery.")
    fetch_p.add_argument("--mailto", default="", help="Contact email for the OpenAlex/Crossref polite pool.")

    return p


def _parse_stages(value: str) -> List[str]:
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        sys.exit(f"Error: unknown stage(s) {unknown}. Available: {ALL_STAGES}")
    return stages


def _resolve_config(args: argparse.Namespace):
    """Resolve config to a single path, or [base, profile] overlay when --profile is used
    (and --config wasn't given explicitly). Later files override earlier ones."""
    profile = getattr(args, "profile", None)
    if profile and getattr(args, "config", DEFAULT_CONFIG) == DEFAULT_CONFIG:
        path = os.path.join("config", "profiles", f"{profile}.txt")
        if not os.path.exists(path):
            sys.exit(f"Error: profile '{profile}' not found at {path}.")
        return [DEFAULT_CONFIG, path]
    return args.config


def _run_stage(stages, name, log, fn, *, enabled=True, fatal=False):
    """Run one pipeline stage when it's selected and enabled.

    Optional stages are non-fatal (errors are logged and skipped so the rest of
    the run continues); core stages (`fatal=True`) let exceptions propagate to
    the caller. Centralising this is why the body below has no per-stage
    try/except boilerplate.
    """
    if name not in stages or not enabled:
        return
    log_stage(log, name)
    if fatal:
        fn()
    else:
        try:
            fn()
        except Exception as e:
            log(f"  {name} stage failed (non-fatal): {e}")


def run_pipeline(cfg: Config, stages: List[str], skip_coherence: bool = False) -> None:
    """Execute the configured stages on a single corpus.

    `skip_coherence=True` bypasses the (slow) coherence scoring even when
    `topics` is in the stages — useful when iterating on other settings.
    """
    init_directories(cfg, stages)
    log = make_logger(cfg.log_path("run.log"))
    write_config_log(cfg)

    # Honour the configured worker cap for this run (loky reads it at executor
    # creation time, so setting it here still takes effect for coherence/sklearn).
    os.environ["LOKY_MAX_CPU_COUNT"] = str(getattr(cfg, "max_cpu_count", 4))

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

    # Pick lemmatisation backend (NLTK or spaCy) once per run.
    from preprocess import set_lemmatisation_backend
    set_lemmatisation_backend("spacy" if getattr(cfg, "use_spacy", False) else "nltk", log)

    topics_artefacts: dict = {}

    # Each stage is a small closure; _run_stage handles selection, gating, the
    # banner, and (for optional stages) non-fatal error handling — so there is no
    # repetitive per-stage try/except here.

    def _ingest():
        ingest_corpus(cfg, log)
        descriptive_stats(cfg, cfg.directory_text, log)

    def _metadata():
        from entities import build_entity_tables
        from metadata import enrich_metadata, extract_metadata
        extract_metadata(cfg, log)
        if cfg.enrich_enable:
            enrich_metadata(cfg, log)
        build_entity_tables(cfg, log)

    def _references():
        from citation_analysis import run_citation_analysis
        from references import extract_references
        extract_references(cfg, log)
        run_citation_analysis(cfg, log)

    def _collaboration():
        from collaboration import run_collaboration
        run_collaboration(cfg, log)

    def _bibliometrics():
        from bibliometrics import build_bibliometrics
        build_bibliometrics(cfg, log)

    def _preprocess():
        stopwords = load_stopwords(cfg, log)
        substitutions = load_substitutions(cfg, log)
        preprocess_corpus(cfg, stopwords, substitutions, log)
        descriptive_stats(cfg, cfg.directory_processed, log)

    def _annotate():
        from annotate import run_annotation
        run_annotation(cfg, log)

    def _lexical():
        from lexical import run_lexical
        run_lexical(cfg, log)

    def _terms():
        run_term_analysis(cfg, _read_processed_as_token_lists(cfg), log)

    def _phrases():
        tokens = _flatten_processed_tokens(cfg)
        sentences = _read_processed_sentences(cfg) if cfg.preserve_sentences else None
        cooccurrence = run_phrase_analysis(cfg, tokens, log, sentences=sentences)
        run_hierarchical_clustering(cfg, cooccurrence, log)

    def _topics():
        nonlocal topics_artefacts
        topics_artefacts = run_topic_models(cfg, log)
        if topics_artefacts and not skip_coherence:
            log_stage(log, "coherence")
            compute_coherence(cfg, topics_artefacts["tokenised_docs"], topics_artefacts["top_words"], log)
        elif skip_coherence:
            log("  coherence skipped (--no-coherence)")

    def _stability():
        from stability import run_stability
        if topics_artefacts and topics_artefacts.get("tokenised_docs"):
            docs = [" ".join(d) for d in topics_artefacts["tokenised_docs"]]
        else:
            from topics import load_documents_for_topics
            docs, _ = load_documents_for_topics(cfg)
        run_stability(cfg, docs, log)

    def _topic_evolution():
        from topic_evolution import run_topic_evolution
        run_topic_evolution(cfg, log)

    def _map():
        from document_map import write_document_map
        embeddings = topics_artefacts.get("embeddings") if topics_artefacts else None
        if embeddings is None:
            log("  document map: needs BERTopic embeddings — make sure 'topics' ran and BERTopic is installed.")
            return
        bertopic_label = next((k for k in topics_artefacts["assignments"] if k.startswith("BERTopic")), None)
        topic_labels = topics_artefacts["assignments"][bertopic_label] if bertopic_label else None
        if topic_labels is None:
            import numpy as np
            topic_labels = np.zeros(len(embeddings), dtype=int)
        write_document_map(cfg, embeddings, topics_artefacts["doc_ids"], topic_labels, log)

    def _science_map():
        from science_map import run_science_map
        run_science_map(cfg, log)

    def _longitudinal():
        from longitudinal import run_longitudinal
        run_longitudinal(cfg, log, skip_coherence=skip_coherence)

    _run_stage(stages, "ingest", log, _ingest, fatal=True)
    _run_stage(stages, "metadata", log, _metadata)
    _run_stage(stages, "references", log, _references)
    _run_stage(stages, "collaboration", log, _collaboration, enabled=cfg.collaboration_enable)
    _run_stage(stages, "bibliometrics", log, _bibliometrics)
    _run_stage(stages, "preprocess", log, _preprocess, fatal=True)
    _run_stage(stages, "annotate", log, _annotate, enabled=cfg.annotation_enable)
    _run_stage(stages, "lexical", log, _lexical, enabled=cfg.lexical_enable)
    _run_stage(stages, "terms", log, _terms, fatal=True)
    _run_stage(stages, "phrases", log, _phrases, fatal=True)
    _run_stage(stages, "topics", log, _topics, fatal=True)
    _run_stage(stages, "stability", log, _stability)
    _run_stage(stages, "topic_evolution", log, _topic_evolution, enabled=cfg.topic_evolution_enable)
    _run_stage(stages, "map", log, _map)
    _run_stage(stages, "science_map", log, _science_map, enabled=cfg.science_map_enable)
    _run_stage(stages, "longitudinal", log, _longitudinal, enabled=cfg.longitudinal_enable)

    # Reproducibility export (non-fatal) then the always-on summary.
    try:
        from runconfig import write_run_config
        write_run_config(cfg, stages, log)
    except Exception as e:
        log(f"  run-config export failed (non-fatal): {e}")

    log_stage(log, "summary")
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


def _read_processed_sentences(cfg: Config) -> List[List[str]]:
    """Read processed files as a list of per-line (sentence) lemmatised token lists.

    Used for sentence-aware n-grams / co-occurrence when `preserve_sentences` is on
    (each processed file then holds one sentence per line).
    """
    from preprocess import lemmatise_text

    sentences: List[List[str]] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            for line in f:
                toks = lemmatise_text(line)
                if toks:
                    sentences.append(toks)
    return sentences


def cmd_run(args: argparse.Namespace) -> int:
    stages = _parse_stages(args.stages)
    name = getattr(args, "name", None)
    skip_coherence = getattr(args, "no_coherence", False)
    if name and len(args.data_dirs) > 1:
        sys.exit("Error: --name only makes sense with a single data directory; remove it or pass one DATA_DIR.")
    config_path = _resolve_config(args)
    for data_dir in args.data_dirs:
        try:
            cfg = load_config(config_path, data_dir, output_base=args.out, run_name=name)
            run_pipeline(cfg, stages, skip_coherence=skip_coherence)
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
        data_dirs=dirs,
        config=args.config,
        profile=getattr(args, "profile", None),
        out=args.out,
        stages=args.stages,
        no_coherence=getattr(args, "no_coherence", False),
        name=None,
    )
    rc = cmd_run(run_args)
    if rc != 0:
        return rc

    if not args.no_compare:
        output_dirs = [os.path.join(args.out, os.path.basename(os.path.normpath(d))) for d in dirs]
        output_dirs = [d for d in output_dirs if os.path.isdir(d)]
        if output_dirs:
            report_path = os.path.join(args.out, "comparison_report.md")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            cmp.write_report(report_path, output_dirs, args.top)
            print(f"Comparison report written: {report_path} ({len(output_dirs)} run(s))")

    return 0


def _looks_like_output_dir(d: str) -> bool:
    """Heuristic: does `d` contain analysis artefacts (i.e. is it an OUTPUT dir)?"""
    return os.path.exists(os.path.join(d, "terms_lemmatised.csv")) or bool(
        glob.glob(os.path.join(d, "topics_*.txt"))
    )


def cmd_compare(args: argparse.Namespace) -> int:
    dirs = args.dirs or cmp.discover_dirs(DEFAULT_OUTPUT_BASE)
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print(
            f"No output directories provided or discovered under ./{DEFAULT_OUTPUT_BASE}/*.",
            file=sys.stderr,
        )
        return 1
    # `compare` reads finished runs — point users at `run`/`batch` if they pass data dirs.
    suspicious = [d for d in dirs if not _looks_like_output_dir(d)]
    if suspicious:
        print(
            "Note: `compare` expects OUTPUT directories (e.g. output/<name>), not data folders.\n"
            f"      No analysis artefacts found in: {', '.join(suspicious)}\n"
            "      Process raw data with `run` or `batch` first; `compare` then diffs the output/ dirs.",
            file=sys.stderr,
        )
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    cmp.write_report(args.report, dirs, args.top)
    print(f"Comparison report written: {args.report} ({len(dirs)} run(s))")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Build a corpus CSV by keyword search from an external provider."""
    import csv as _csv
    from types import SimpleNamespace

    from bibimport import STANDARD_FIELDS
    from providers import make_provider

    ns = SimpleNamespace(provider_mailto=args.mailto, provider_fallback="openalex")
    provider = make_provider(args.provider, ns, lambda m: print(m))
    if provider is None:
        print("Error: no usable provider (Scopus needs SCOPUS_API_KEY).", file=sys.stderr)
        return 1

    print(f"Searching {provider.name} for {args.query!r} (limit {args.limit})...")
    records = provider.search_keywords(args.query, args.limit)
    if not records:
        print("No records returned.", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"fetched_{provider.name}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=STANDARD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in STANDARD_FIELDS})

    print(f"Fetched {len(records)} record(s) -> {out_csv}")
    print(f"Next:  python geordie_miner.py run {args.out}")
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
    if args.command == "fetch":
        return cmd_fetch(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
