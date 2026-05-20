"""Topic-stability analysis: run LDA at the base K with multiple random seeds
and report how stable each topic is across runs.

Stable topics (high pairwise Jaccard of their top words) are likely real
signal. Unstable topics (low Jaccard) are likely artefacts of random
initialisation.

Output: `topic_stability.csv` with one row per (seed-1, seed-2, topic-A, topic-B)
matched pair, plus an aggregated `topic_stability_summary.csv` with each
"canonical" topic's mean Jaccard across seeds.
"""

from __future__ import annotations

import csv
from typing import Callable, List, Tuple

import numpy as np
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

from config import Config


def _topics_from_lda(lda: LdaModel, k: int, top_n: int) -> List[List[str]]:
    """Return a list of `k` topics, each a list of `top_n` words."""
    return [[w for w, _ in lda.show_topic(i, topn=top_n)] for i in range(k)]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_match(topic: List[str], candidates: List[List[str]]) -> Tuple[int, float]:
    """Return (best_idx, best_jaccard) for the candidate topic most similar to `topic`."""
    best_idx, best_score = -1, -1.0
    for j, cand in enumerate(candidates):
        s = _jaccard(topic, cand)
        if s > best_score:
            best_idx, best_score = j, s
    return best_idx, best_score


def run_stability(
    cfg: Config,
    docs: List[str],
    log: Callable[[str], None],
    n_seeds: int = 5,
    top_n_words: int = 10,
) -> None:
    """Run LDA at `cfg.lda_topics` with N different seeds, compute stability.

    Stability per topic = mean Jaccard between this topic's top words and the
    best-matching topic from each other seed's run.
    """
    if not docs:
        log("  stability: no documents; skipping.")
        return

    k = cfg.lda_topics
    if k < 2:
        log("  stability: lda_topics < 2; skipping.")
        return

    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = list(matutils.Sparse2Corpus(dtm, documents_columns=False))

    log(f"  stability: running LDA at K={k} with {n_seeds} different seeds...")
    runs: List[List[List[str]]] = []
    for seed in range(n_seeds):
        lda = LdaModel(
            corpus=bow,
            num_topics=k,
            id2word=id2word,
            passes=cfg.lda_passes,
            iterations=1000,
            alpha="auto",
            eta="auto",
            random_state=42 + seed,
            per_word_topics=False,
        )
        runs.append(_topics_from_lda(lda, k, top_n_words))

    # For each topic in run 0, find its best match in every other run.
    rows: List[Tuple[int, float, str]] = []
    for i, topic in enumerate(runs[0]):
        match_scores: List[float] = []
        for run in runs[1:]:
            _, score = _best_match(topic, run)
            match_scores.append(score)
        mean = float(np.mean(match_scores)) if match_scores else 0.0
        rows.append((i + 1, mean, ", ".join(topic[:5])))

    rows.sort(key=lambda r: -r[1])
    with open(cfg.output_path("topic_stability.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "mean_jaccard", "top_5_words"])
        for topic_id, mean, top_words in rows:
            writer.writerow([topic_id, f"{mean:.3f}", top_words])

    high = sum(1 for _, m, _ in rows if m >= 0.7)
    low = sum(1 for _, m, _ in rows if m < 0.3)
    log(f"  stability: {high}/{len(rows)} topics highly stable (≥0.7), {low} unstable (<0.3).")
    log(f"  Stability scores written: topic_stability.csv")
