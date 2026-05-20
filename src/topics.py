"""Topic modelling: KMeans, LDA, NMF, HDP.

Each model runs at K, K*m1, K*m2, K*m3 topics (multipliers from config; 0 = skip).
Outputs:
    topics_<model>_<K>.txt              top words per topic + doc counts
    topic_assignments.csv               doc_id + one column per model/K
    topic_top_docs.csv                  top-N most representative docs per topic
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim import matutils
from gensim.models.hdpmodel import HdpModel
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import Config
from preprocess import lemmatise_text


TOP_DOCS_PER_TOPIC = 5  # how many representative documents to list per topic


def load_documents_for_topics(cfg: Config) -> Tuple[List[str], List[str]]:
    """Load each processed file as a single lemmatised string. Return (docs, file_names)."""
    docs: List[str] = []
    names: List[str] = []
    for filename in sorted(os.listdir(cfg.directory_processed)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(cfg.directory_processed, filename), "r", encoding="utf-8") as f:
            docs.append(" ".join(lemmatise_text(f.read())))
        names.append(filename)
    return docs, names


def _file_id(name: str) -> str:
    return name.split("__", 1)[0]


def _topic_ks(base: int, multipliers: List[int]) -> List[int]:
    """Return unique, positive K values: [base, base*m1, base*m2, ...]."""
    ks = [base] + [base * m for m in multipliers if m > 0]
    out: List[int] = []
    for k in ks:
        if k > 0 and k not in out:
            out.append(k)
    return out


def run_topic_models(cfg: Config, log: Callable[[str], None]) -> Dict:
    """Run all four topic models. Write per-model topic-word files plus unified outputs.

    Returns a dict of artefacts useful to the summary/coherence stages:
        {
            "names": [...],                    # file names in doc order
            "doc_ids": [...],                   # short ids
            "tokenised_docs": [[...]],          # for coherence
            "assignments": {label: array},
            "top_words": {label: [[w,...],...]},
            "doc_counts": {label: [n, ...]},
        }
    """
    docs, names = load_documents_for_topics(cfg)
    if not docs:
        log("No documents to topic-model; skipping.")
        return {}
    doc_ids = [_file_id(n) for n in names]
    tokenised_docs = [d.split() for d in docs]

    multipliers = [
        cfg.topic_modelling_multi1,
        cfg.topic_modelling_multi2,
        cfg.topic_modelling_multi3,
    ]

    assignments: Dict[str, np.ndarray] = {}
    top_words: Dict[str, List[List[str]]] = {}
    doc_counts: Dict[str, List[int]] = {}
    top_docs_rows: List[Tuple] = []

    for k in _topic_ks(cfg.kmeans_topics, multipliers):
        labels, words, counts, top = _kmeans(cfg, docs, doc_ids, k, log)
        assignments[f"KMeans_{k}"] = labels
        top_words[f"KMeans_{k}"] = words
        doc_counts[f"KMeans_{k}"] = counts
        top_docs_rows.extend(top)

    for k in _topic_ks(cfg.lda_topics, multipliers):
        labels, words, counts, top = _lda(cfg, docs, doc_ids, k, log)
        assignments[f"LDA_{k}"] = labels
        top_words[f"LDA_{k}"] = words
        doc_counts[f"LDA_{k}"] = counts
        top_docs_rows.extend(top)

    for k in _topic_ks(cfg.nmf_topics, multipliers):
        labels, words, counts, top = _nmf(cfg, docs, doc_ids, k, log)
        assignments[f"NMF_{k}"] = labels
        top_words[f"NMF_{k}"] = words
        doc_counts[f"NMF_{k}"] = counts
        top_docs_rows.extend(top)

    labels, words, counts, top, n_hdp = _hdp(cfg, docs, doc_ids, log)
    assignments[f"HDP_{n_hdp}"] = labels
    top_words[f"HDP_{n_hdp}"] = words
    doc_counts[f"HDP_{n_hdp}"] = counts
    top_docs_rows.extend(top)

    _write_assignments_csv(cfg, doc_ids, assignments)
    _write_top_docs_csv(cfg, top_docs_rows)
    log(f"Unified topic assignments + top-{TOP_DOCS_PER_TOPIC} docs per topic exported.")

    return {
        "names": names,
        "doc_ids": doc_ids,
        "tokenised_docs": tokenised_docs,
        "assignments": assignments,
        "top_words": top_words,
        "doc_counts": doc_counts,
    }


def _write_topic_words_file(
    cfg: Config,
    filename: str,
    top_words: List[List[Tuple[str, float]]],
    doc_counts: List[int],
) -> None:
    """Write a `topics_<model>_<K>.txt` file with weighted terms + per-topic doc counts."""
    with open(cfg.output_path(filename), "w", encoding="utf-8") as f:
        for i, words in enumerate(top_words):
            n_docs = doc_counts[i] if i < len(doc_counts) else 0
            f.write(f"Topic {i + 1} ({n_docs} docs): ")
            f.write(", ".join(f"{w} ({s:.4f})" for w, s in words) + "\n")


def _top_docs_for_topic(
    model_label: str,
    k: int,
    doc_topic_scores: np.ndarray,
    doc_ids: List[str],
    assigned_topics: np.ndarray,
    n_topics: int,
) -> List[Tuple]:
    """For each topic, return rows of (model, K, topic, rank, doc_id, score) for the top-N docs.

    `doc_topic_scores` is shape (n_docs, n_topics) — higher = more representative.
    `assigned_topics` is the argmax per doc.
    Only docs assigned to topic t are eligible for topic t's top list.
    """
    rows: List[Tuple] = []
    for t in range(n_topics):
        eligible = np.where(assigned_topics == t)[0]
        if eligible.size == 0:
            continue
        scores = doc_topic_scores[eligible, t]
        order = np.argsort(-scores)[:TOP_DOCS_PER_TOPIC]
        for rank, idx in enumerate(order, start=1):
            d_idx = eligible[idx]
            rows.append((model_label, k, t + 1, rank, doc_ids[d_idx], float(scores[idx])))
    return rows


def _kmeans(cfg: Config, docs: List[str], doc_ids: List[str], k: int, log: Callable[[str], None]):
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(dtm)
    feature_names = vec.get_feature_names_out()

    labels = model.labels_
    top_words: List[List[Tuple[str, float]]] = []
    for centroid in model.cluster_centers_:
        idx = centroid.argsort()[-10:][::-1]
        top_words.append([(feature_names[j], float(centroid[j])) for j in idx])

    counts = [int((labels == i).sum()) for i in range(k)]
    _write_topic_words_file(cfg, f"topics_kmeans_{k}.txt", top_words, counts)

    # For top docs: use negative distance to centroid as "score" (closer = higher).
    distances = model.transform(dtm)  # shape (n_docs, k), smaller = better
    scores = -distances
    top_docs = _top_docs_for_topic(f"KMeans", k, scores, doc_ids, labels, k)

    log(f"KMeans completed: {k} topics.")
    return labels, top_words, counts, top_docs


def _lda(cfg: Config, docs: List[str], doc_ids: List[str], k: int, log: Callable[[str], None]):
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = list(matutils.Sparse2Corpus(dtm, documents_columns=False))

    lda = LdaModel(
        corpus=bow,
        num_topics=k,
        id2word=id2word,
        passes=cfg.lda_passes,
        iterations=1000,
        alpha="auto",
        eta="auto",
        random_state=42,
        per_word_topics=bool(cfg.lda_per_word_topics),
    )

    scores = np.zeros((len(bow), k))
    for i, doc in enumerate(bow):
        for t, p in lda.get_document_topics(doc, minimum_probability=0):
            scores[i, t] = p

    labels = scores.argmax(axis=1)
    counts = [int((labels == i).sum()) for i in range(k)]

    top_words = [
        [(w, float(p)) for w, p in lda.show_topic(i, topn=cfg.lda_terms_per_topic)]
        for i in range(k)
    ]
    _write_topic_words_file(cfg, f"topics_lda_{k}.txt", top_words, counts)

    top_docs = _top_docs_for_topic("LDA", k, scores, doc_ids, labels, k)
    log(f"LDA completed: {k} topics.")
    return labels, top_words, counts, top_docs


def _nmf(cfg: Config, docs: List[str], doc_ids: List[str], k: int, log: Callable[[str], None]):
    vec = TfidfVectorizer(lowercase=True)
    dtm = vec.fit_transform(docs)
    model = NMF(n_components=k, random_state=42, max_iter=cfg.nmf_max_iter)
    model.fit(dtm)
    terms = vec.get_feature_names_out()

    scores = model.transform(dtm)  # shape (n_docs, k)
    labels = scores.argmax(axis=1)
    counts = [int((labels == i).sum()) for i in range(k)]

    top_words = []
    for topic in model.components_:
        idx = topic.argsort()[: -cfg.nmf_terms_per_topic - 1 : -1]
        top_words.append([(terms[j], float(topic[j])) for j in idx])
    _write_topic_words_file(cfg, f"topics_nmf_{k}.txt", top_words, counts)

    top_docs = _top_docs_for_topic("NMF", k, scores, doc_ids, labels, k)
    log(f"NMF completed: {k} topics.")
    return labels, top_words, counts, top_docs


def _hdp(cfg: Config, docs: List[str], doc_ids: List[str], log: Callable[[str], None]):
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = list(matutils.Sparse2Corpus(dtm, documents_columns=False))

    hdp = HdpModel(corpus=bow, id2word=id2word)
    results = hdp.show_topics(num_words=cfg.terms_per_topic_hdp, formatted=False)
    n_topics = len(results)

    scores = np.zeros((len(bow), n_topics))
    for i, doc in enumerate(bow):
        for t, p in hdp[doc]:
            if t < n_topics:
                scores[i, t] = p

    labels = scores.argmax(axis=1) if scores.size else np.array([], dtype=int)
    counts = [int((labels == i).sum()) for i in range(n_topics)]

    top_words = [[(w, float(p)) for w, p in tw] for (_, tw) in results]
    _write_topic_words_file(cfg, "topics_hdp.txt", top_words, counts)

    top_docs = _top_docs_for_topic("HDP", n_topics, scores, doc_ids, labels, n_topics)
    log(f"HDP completed: {n_topics} topics.")
    return labels, top_words, counts, top_docs, n_topics


def _write_assignments_csv(cfg: Config, doc_ids: List[str], assignments: Dict[str, np.ndarray]) -> None:
    """Write topic_assignments.csv with one column per model/K. Topic ids are 1-indexed."""
    df = pd.DataFrame({"doc_id": doc_ids})
    for label, arr in assignments.items():
        df[label] = [int(x) + 1 for x in arr]
    df.to_csv(cfg.output_path("topic_assignments.csv"), index=False)


def _write_top_docs_csv(cfg: Config, rows: List[Tuple]) -> None:
    """Write topic_top_docs.csv: (model, K, topic, rank, doc_id, score)."""
    with open(cfg.output_path("topic_top_docs.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "K", "topic", "rank", "doc_id", "score"])
        writer.writerows(rows)
