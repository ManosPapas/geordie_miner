"""Longitudinal analysis: split the corpus into publication-year brackets and
run the term/phrase/topic analysis on each subset, so clusters can be compared
period-to-period.

Reads years from `metadata.csv`, partitions the *already preprocessed* documents
into 5- or 10-year brackets, and for each bracket reuses the standard stage
functions (`run_term_analysis`, `run_phrase_analysis`, `run_topic_models`) writing
into `output/<name>/longitudinal-<start>-<end>/`. Finishes with a cross-period
comparison report (reusing `compare.write_report`).
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from dataclasses import replace
from typing import Callable, Dict, List

from compare import write_report
from config import Config
from corpus_io import bracket_size, doc_years, processed_files_by_doc
from phrases import load_processed_corpus, run_hierarchical_clustering, run_phrase_analysis
from terms import run_term_analysis
from topics import run_topic_models


def _read_token_lists(processed_dir: str) -> List[List[str]]:
    out: List[List[str]] = []
    for fn in sorted(os.listdir(processed_dir)):
        if fn.endswith(".txt"):
            with open(os.path.join(processed_dir, fn), "r", encoding="utf-8") as f:
                out.append(f.read().split())
    return out


def run_longitudinal(cfg: Config, log: Callable[[str], None], skip_coherence: bool = False) -> None:
    years = doc_years(cfg, as_int=True)
    files = processed_files_by_doc(cfg)
    paired = {d: years[d] for d in years if d in files}
    if len(paired) < cfg.longitudinal_min_docs:
        log(
            f"  longitudinal: only {len(paired)} doc(s) have both a year and processed text "
            f"(need >= {cfg.longitudinal_min_docs}) — skipping. Did the metadata stage run?"
        )
        return

    yr_values = sorted(set(paired.values()))
    span = yr_values[-1] - yr_values[0]
    size = bracket_size(cfg, span)
    log(f"  longitudinal: {len(paired)} dated docs spanning {yr_values[0]}-{yr_values[-1]} -> {size}-year brackets.")

    # Partition doc_ids into brackets keyed by bracket start year.
    brackets: Dict[int, List[str]] = defaultdict(list)
    for doc_id, yr in paired.items():
        start = (yr // size) * size
        brackets[start].append(doc_id)

    produced_dirs: List[str] = []
    for start in sorted(brackets):
        doc_ids = brackets[start]
        end = start + size - 1
        if len(doc_ids) < cfg.longitudinal_min_docs:
            log(f"  longitudinal: bracket {start}-{end} has {len(doc_ids)} doc(s) (< min) — skipped.")
            continue

        sub_dir = cfg.output_path(f"longitudinal-{start}-{end}")
        sub_processed = os.path.join(sub_dir, "text_processed")
        os.makedirs(sub_processed, exist_ok=True)
        for doc_id in doc_ids:
            src = os.path.join(cfg.directory_processed, files[doc_id])
            shutil.copy(src, os.path.join(sub_processed, files[doc_id]))

        n = len(doc_ids)
        # Keep K within the subset size and run a single K per model (fast, robust
        # on small brackets where K > n_docs would otherwise crash KMeans/NMF).
        safe_k = max(2, min(cfg.kmeans_topics, n - 1))
        sub_cfg = replace(
            cfg,
            directory_analysis=sub_dir,
            directory_processed=sub_processed,
            kmeans_topics=safe_k, lda_topics=safe_k, nmf_topics=safe_k,
            topic_modelling_multi1=0, topic_modelling_multi2=0, topic_modelling_multi3=0,
            enable_bertopic=False,
            longitudinal_enable=False, annotation_enable=False,
        )

        log(f"  longitudinal: bracket {start}-{end} — {n} docs (K={safe_k}).")
        try:
            corpus = _read_token_lists(sub_processed)
            run_term_analysis(sub_cfg, corpus, log)
            tokens, _ = load_processed_corpus(sub_cfg)
            cooc = run_phrase_analysis(sub_cfg, tokens, log)
            run_hierarchical_clustering(sub_cfg, cooc, log)
            run_topic_models(sub_cfg, log)
            produced_dirs.append(sub_dir)
        except Exception as e:
            log(f"  longitudinal: bracket {start}-{end} failed (non-fatal): {e}")

    if len(produced_dirs) >= 2:
        report = cfg.output_path("longitudinal_comparison.md")
        write_report(report, produced_dirs, 50)
        log(f"  longitudinal: cross-period comparison written -> {os.path.basename(report)} (+ .html).")
    elif produced_dirs:
        log("  longitudinal: only one bracket produced — no cross-period comparison.")
