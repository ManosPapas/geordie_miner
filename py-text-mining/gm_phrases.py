"""Phrase analysis: bigrams, trigrams, co-occurrence, network export, hierarchical clustering."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe (no display server required)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from gm_config import Config
from gm_preprocess import lemmatise_text


def load_processed_corpus(cfg: Config) -> Tuple[List[str], List[str]]:
    """Read all processed .txt files, return (concatenated tokens, file_names)."""
    tokens: List[str] = []
    names: List[str] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            tokens.extend(lemmatise_text(f.read()))
        names.append(filename)
    return tokens, names


def _export_ngrams(corpus: List[str], n: int, threshold: int, top_k: int, out_path: str) -> int:
    freq = Counter(ngrams(corpus, n))
    rows = sorted(
        ((" ".join(gram), count) for gram, count in freq.items() if count >= threshold),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ngram", "frequency"])
        writer.writerows(rows)
    return len(rows)


def run_phrase_analysis(
    cfg: Config,
    corpus: List[str],
    log: Callable[[str], None],
) -> Dict[str, Counter]:
    """Export bigrams, trigrams, co-occurrence CSV + GEXF network. Return cooccurrence_counts."""
    n2 = _export_ngrams(
        corpus, 2, cfg.bigram_threshold, cfg.bigram_export_count,
        os.path.join(cfg.directory_analysis, "analysis_terms_ngram2.csv"),
    )
    log(f"Bigrams exported: {n2} (threshold >= {cfg.bigram_threshold}).")

    n3 = _export_ngrams(
        corpus, 3, cfg.trigram_threshold, cfg.trigram_export_count,
        os.path.join(cfg.directory_analysis, "analysis_terms_ngram3.csv"),
    )
    log(f"Trigrams exported: {n3} (threshold >= {cfg.trigram_threshold}).")

    cooccurrence: Dict[str, Counter] = defaultdict(Counter)
    for sentence in sent_tokenize(" ".join(corpus)):
        toks = lemmatise_text(sentence)
        for i, t in enumerate(toks):
            for j in range(i + 1, min(i + 1 + cfg.window_size, len(toks))):
                cooccurrence[t][toks[j]] += 1
                cooccurrence[toks[j]][t] += 1

    cooc_path = os.path.join(cfg.directory_analysis, "analysis_cooccurrence.csv")
    with open(cooc_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term1", "term2", "cooccurrence_count"])
        for t1, neighbours in cooccurrence.items():
            for t2, count in neighbours.items():
                writer.writerow([t1, t2, count])
    log("Co-occurrence matrix exported.")

    graph = nx.Graph()
    for t1, neighbours in cooccurrence.items():
        for t2, count in neighbours.items():
            if count >= cfg.cooccurrence_threshold:
                graph.add_edge(t1, t2, weight=count)
    nx.write_gexf(graph, os.path.join(cfg.directory_analysis, "analysis_cooccurrence_network.gexf"))
    log(f"Co-occurrence network saved ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).")

    return cooccurrence


def run_hierarchical_clustering(
    cfg: Config,
    cooccurrence: Dict[str, Counter],
    log: Callable[[str], None],
) -> None:
    """Agglomerative hierarchical clustering on a binary co-occurrence matrix → dendrogram."""
    terms = list(cooccurrence.keys())
    n = len(terms)
    if n < 2:
        log("Skipping hierarchical clustering: not enough terms.")
        return

    matrix = np.zeros((n, n))
    for i, t1 in enumerate(terms):
        for j, t2 in enumerate(terms):
            if cooccurrence[t1][t2] > 0:
                matrix[i, j] = 1

    distances = pdist(matrix, metric=cfg.clustering_metric)
    linkage_matrix = linkage(distances, method=cfg.linkage_method)

    plt.figure(figsize=cfg.dendrogram_figsize)
    plt.title(f"Agglomerative Hierarchical Clustering ({cfg.clustering_metric})")
    dendrogram(linkage_matrix, labels=terms, orientation="right")
    plt.tight_layout()
    out_path = os.path.join(cfg.directory_analysis, "analysis_ahc_dendrogram_jaccard.png")
    plt.savefig(out_path)
    plt.close()
    log(f"Hierarchical clustering dendrogram saved ({cfg.clustering_metric} / {cfg.linkage_method}).")
