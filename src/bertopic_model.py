"""BERTopic — embedding-based topic modelling.

Uses sentence-transformers (default `all-MiniLM-L6-v2`) to embed each document,
then BERTopic to cluster and label topics. Runs **offline after the first
download** — the embedding model lands in `~/.cache/huggingface/`.

Trade-offs vs LDA/NMF:
- Quality: usually noticeably more coherent topics on small/medium corpora.
- Speed: similar (embedding is fast; clustering is fast).
- Installation footprint: heavy (PyTorch + transformers ≈ 1.5-2.5 GB on disk).

The module is fully optional: it lazy-imports bertopic so the rest of the
pipeline keeps running if the dep isn't installed. When missing, returns None
and the caller skips this model.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np


# Default model — small, fast, good general-purpose English embedder.
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def _try_import_bertopic(log: Callable[[str], None]):
    try:
        from bertopic import BERTopic  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception as e:
        log(
            f"  BERTopic: not available ({e.__class__.__name__}). "
            f"Install with: pip install bertopic sentence-transformers"
        )
        return False


def run_bertopic(
    cfg,
    docs: List[str],
    doc_ids: List[str],
    log: Callable[[str], None],
    embedder_name: str = DEFAULT_EMBEDDER,
    min_topic_size: int = 5,
) -> Optional[dict]:
    """Run BERTopic on `docs`. Return a dict with labels/top_words/counts/top_docs/embeddings.

    Returns None if bertopic isn't installed (with an informative log message).
    """
    if not _try_import_bertopic(log):
        return None

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # 1) Embed documents
    log(f"  BERTopic: embedding {len(docs)} documents with {embedder_name}...")
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(docs, show_progress_bar=False, convert_to_numpy=True)

    # 2) Fit BERTopic
    # Adjust min_topic_size for small corpora — BERTopic's default (10) is too
    # aggressive on <100 doc corpora and ends up putting almost everything in -1.
    effective_min = max(2, min(min_topic_size, max(2, len(docs) // 10)))
    log(f"  BERTopic: clustering (min_topic_size={effective_min})...")
    topic_model = BERTopic(
        embedding_model=embedder,
        min_topic_size=effective_min,
        verbose=False,
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    topics = np.array(topics)

    # BERTopic uses -1 for "outlier" (unassigned). Re-map to 0-based topic ids,
    # collapsing -1 into its own topic id at the end so downstream code is happy.
    unique_topics = sorted(set(int(t) for t in topics))
    if -1 in unique_topics:
        unique_topics.remove(-1)
    remap = {old: new for new, old in enumerate(unique_topics)}
    outlier_id = len(unique_topics)  # last index reserved for outliers
    labels = np.array([remap[int(t)] if int(t) != -1 else outlier_id for t in topics])
    n_topics = len(unique_topics) + (1 if -1 in set(int(t) for t in topics) else 0)

    # 3) Top words per topic
    top_words: List[List[Tuple[str, float]]] = []
    for old in unique_topics:
        words = topic_model.get_topic(old) or []
        top_words.append([(w, float(s)) for w, s in words[:10]])
    if -1 in set(int(t) for t in topics):
        words = topic_model.get_topic(-1) or []
        top_words.append([(w, float(s)) for w, s in words[:10]])

    counts = [int((labels == i).sum()) for i in range(n_topics)]

    # 4) Top docs per topic — use distance to topic centroid (cosine similarity)
    centroids: List[np.ndarray] = []
    for i in range(n_topics):
        mask = labels == i
        if mask.any():
            centroids.append(embeddings[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    centroids_arr = np.array(centroids)
    # Cosine similarity = normalised dot product
    embeddings_n = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    centroids_n = centroids_arr / np.linalg.norm(centroids_arr, axis=1, keepdims=True).clip(min=1e-9)
    scores = embeddings_n @ centroids_n.T  # (n_docs, n_topics)

    # 5) Build top_docs rows (same shape as topics.py expects)
    from topics import _top_docs_for_topic, TOP_DOCS_PER_TOPIC  # reuse helper

    top_docs = _top_docs_for_topic("BERTopic", n_topics, scores, doc_ids, labels, n_topics)

    log(f"  BERTopic completed: {n_topics} topics (incl. outliers if any).")

    # Save the per-topic word file (same convention as the other models).
    with open(cfg.output_path(f"topics_bertopic.txt"), "w", encoding="utf-8") as f:
        for i, words in enumerate(top_words):
            label = f"Topic {i + 1}" if i < len(unique_topics) else "Topic (outliers)"
            f.write(f"{label} ({counts[i] if i < len(counts) else 0} docs): ")
            f.write(", ".join(f"{w} ({s:.4f})" for w, s in words) + "\n")

    # And the [doc_id, topic_number] membership file (matches the old code's convention).
    with open(cfg.output_path("topics_bertopic_doc2topic.txt"), "w", encoding="utf-8") as f:
        for idx, lbl in enumerate(labels):
            f.write(f"{doc_ids[idx]}, {int(lbl) + 1}\n")

    return {
        "label": f"BERTopic_{n_topics}",
        "labels": labels,
        "top_words": top_words,
        "counts": counts,
        "top_docs": top_docs,
        "embeddings": embeddings,
        "n_topics": n_topics,
    }
