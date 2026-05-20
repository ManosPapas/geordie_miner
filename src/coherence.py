"""Topic coherence scoring (c_v and u_mass) for LDA + NMF runs.

Higher c_v → topics are semantically more coherent (typical range ~0.3 to 0.8).
u_mass is log-based and negative; closer to zero is better.

Used after topics.py to give the researcher an objective signal for "which K is
best?" rather than eyeballing the topic words.
"""

from __future__ import annotations

import csv
from typing import Callable, Dict, List, Tuple

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from config import Config


def compute_and_save(
    cfg: Config,
    tokenised_docs: List[List[str]],
    top_words: Dict[str, List[List[Tuple[str, float]]]],
    log: Callable[[str], None],
) -> None:
    """Compute c_v + u_mass coherence for each LDA/NMF/HDP topic model and write coherence_scores.csv.

    `top_words` maps a model label (e.g. "LDA_5", "NMF_10") to a list of topics,
    each topic being a list of (word, score) tuples.
    """
    if not tokenised_docs:
        return

    dictionary = Dictionary(tokenised_docs)
    rows: List[Tuple[str, int, float, float]] = []

    for label, topics in top_words.items():
        if label.startswith("KMeans"):
            continue  # coherence on KMeans centroids isn't meaningful in the same way
        try:
            model_name, k_str = label.rsplit("_", 1)
            k = int(k_str)
        except ValueError:
            model_name, k = label, 0

        topic_word_lists = [[w for w, _ in topic] for topic in topics if topic]
        if not topic_word_lists:
            continue

        try:
            cv = CoherenceModel(
                topics=topic_word_lists,
                texts=tokenised_docs,
                dictionary=dictionary,
                coherence="c_v",
            ).get_coherence()
            um = CoherenceModel(
                topics=topic_word_lists,
                texts=tokenised_docs,
                dictionary=dictionary,
                coherence="u_mass",
            ).get_coherence()
        except Exception as e:
            log(f"  coherence failed for {label}: {e}")
            continue

        rows.append((model_name, k, float(cv), float(um)))
        log(f"  coherence {label}: c_v = {cv:.3f}, u_mass = {um:.3f}")

    rows.sort(key=lambda r: (r[0], r[1]))
    with open(cfg.output_path("coherence_scores.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "K", "coherence_c_v", "coherence_u_mass"])
        writer.writerows(rows)
    log(f"Coherence scores written: {len(rows)} model(s).")
