"""Phrase analysis: bigrams, trigrams, co-occurrence, network export, hierarchical clustering."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from config import Config
from plotting import plt
from preprocess import collapse_consecutive_list, lemmatise_text
from vosviewer import export_vosviewer


def load_processed_corpus(cfg: Config) -> Tuple[List[str], List[str]]:
    """Read all processed .txt files, return (deduped concatenated tokens, file_names).

    Lemmatisation can re-introduce consecutive duplicates (e.g. `products` and
    `product` both become `product`), so we collapse again per-file before
    concatenating.
    """
    tokens: List[str] = []
    names: List[str] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            file_tokens = lemmatise_text(f.read())
        tokens.extend(collapse_consecutive_list(file_tokens))
        names.append(filename)
    return tokens, names


def _export_ngrams(units: List[List[str]], n: int, threshold: int, top_k: int, out_path: str) -> int:
    """Export top n-grams to CSV. Skip n-grams with repeated tokens (e.g. A-B-A) — they
    add no information beyond their constituent (n-1)-grams and clutter the top list.

    `units` is a list of token sequences; n-grams are counted within each unit and
    never span across units. The default caller passes a single unit (the whole
    flattened corpus); with `preserve_sentences` each sentence is its own unit so
    n-grams don't cross sentence boundaries.
    """
    freq: Counter = Counter()
    for unit in units:
        freq.update(ngrams(unit, n))
    rows = sorted(
        (
            (" ".join(gram), count)
            for gram, count in freq.items()
            if count >= threshold and len(set(gram)) == n
        ),
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
    sentences: List[List[str]] | None = None,
) -> Dict[str, Counter]:
    """Export bigrams, trigrams, co-occurrence CSV + GEXF/VOSviewer networks.

    `corpus` is the flattened token list (back-compatible). When `sentences` is
    provided (preserve_sentences mode) it is a list of per-sentence token lists;
    n-grams and co-occurrence windows then respect sentence boundaries.
    Returns the co-occurrence counts.
    """
    ngram_units = sentences if sentences is not None else [corpus]

    n2 = _export_ngrams(
        ngram_units, 2, cfg.bigram_threshold, cfg.bigram_export_count,
        cfg.output_path("bigrams.csv"),
    )
    log(f"Bigrams exported: {n2} (threshold >= {cfg.bigram_threshold}).")

    n3 = _export_ngrams(
        ngram_units, 3, cfg.trigram_threshold, cfg.trigram_export_count,
        cfg.output_path("trigrams.csv"),
    )
    log(f"Trigrams exported: {n3} (threshold >= {cfg.trigram_threshold}).")

    # Co-occurrence units: pre-segmented sentences when available, else re-segment
    # the joined corpus the original way (preserves default behaviour exactly).
    if sentences is not None:
        cooc_units = sentences
    else:
        cooc_units = [lemmatise_text(s) for s in sent_tokenize(" ".join(corpus))]

    cooccurrence: Dict[str, Counter] = defaultdict(Counter)
    for toks in cooc_units:
        for i, t in enumerate(toks):
            for j in range(i + 1, min(i + 1 + cfg.window_size, len(toks))):
                cooccurrence[t][toks[j]] += 1
                cooccurrence[toks[j]][t] += 1

    cooc_path = cfg.output_path("cooccurrence.csv")
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
    nx.write_gexf(graph, cfg.output_path("network.gexf"))
    log(f"Co-occurrence network saved ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).")

    # Also export the same network in VOSviewer's native map + network format.
    # Guarded so this value-add export can't abort the (un-wrapped) phrases stage.
    try:
        export_vosviewer(
            graph,
            cfg.output_path("network_vosviewer_map.txt"),
            cfg.output_path("network_vosviewer_network.txt"),
            log,
        )
    except Exception as e:
        log(f"  VOSviewer export failed (non-fatal): {e}")

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

    width, height, font_size = _dendrogram_dimensions(terms, cfg.dendrogram_figsize)

    plt.figure(figsize=(width, height))
    plt.title(f"Agglomerative Hierarchical Clustering ({cfg.clustering_metric})")
    dendrogram(linkage_matrix, labels=terms, orientation="right", leaf_font_size=font_size)
    plt.xlabel("distance")
    # bbox_inches="tight" expands the saved canvas to fit the (possibly long) leaf
    # labels, so they're never clipped regardless of label length.
    plt.savefig(cfg.output_path("dendrogram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log(
        f"Hierarchical clustering dendrogram saved ({cfg.clustering_metric} / {cfg.linkage_method}) — "
        f"{n} leaves, {width:.0f}x{height:.0f}in @ {font_size}pt (adaptive)."
    )


def _dendrogram_dimensions(
    labels: List[str],
    floor_figsize: Tuple[int, int],
) -> Tuple[float, float, int]:
    """Pick (width_in, height_in, leaf_font_pt) so leaf labels stay legible.

    Complexity is estimated from the number of leaves, the mean label length and
    the maximum label length. The configured `dendrogram_figsize` acts as a floor
    (minimum size), never a cap, so small dendrograms look unchanged.
    """
    n = len(labels)
    lengths = [len(s) for s in labels] or [1]
    mean_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    floor_w, floor_h = float(floor_figsize[0]), float(floor_figsize[1])

    # Right-oriented: one leaf per row. Give each leaf ~0.18in; grow height with
    # leaf count, capped so the PNG stays within matplotlib's pixel limits.
    per_leaf_in = 0.18
    height = min(120.0, max(floor_h, n * per_leaf_in + 2.0))

    # Font follows the spacing actually achieved (shrinks only if we hit the cap).
    spacing_in = (height - 2.0) / max(n, 1)
    font_size = int(max(5, min(11, round(spacing_in * 72 * 0.8))))

    # Width grows a little with label length so the tree isn't cramped beside
    # long labels (bbox_inches="tight" then handles the exact label extent).
    width = min(40.0, max(floor_w, 9.0 + (mean_len + max_len) * 0.04))

    return width, height, font_size
