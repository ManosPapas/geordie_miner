"""Topic modelling: KMeans, LDA, NMF, HDP.

Each model is run at K, K*m1, K*m2, K*m3 topics (multipliers from config; 0 = skip).
"""

from __future__ import annotations

import os
from typing import Callable, List

from gensim import matutils
from gensim.models.hdpmodel import HdpModel
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gm_config import Config
from gm_preprocess import lemmatise_text


def load_documents_for_topics(cfg: Config) -> tuple[list[str], list[str]]:
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


def _topic_ks(base: int, multipliers: list[int]) -> list[int]:
    """Return list of unique, positive K values: base + base*m for non-zero multipliers."""
    ks = [base] + [base * m for m in multipliers if m > 0]
    seen = []
    for k in ks:
        if k > 0 and k not in seen:
            seen.append(k)
    return seen


def run_topic_models(cfg: Config, log: Callable[[str], None]) -> None:
    docs, names = load_documents_for_topics(cfg)
    if not docs:
        log("No documents to topic-model; skipping.")
        return

    multipliers = [cfg.topic_modelling_multi1, cfg.topic_modelling_multi2, cfg.topic_modelling_multi3]

    for k in _topic_ks(cfg.kmeans_topics, multipliers):
        _kmeans(cfg, docs, names, k, log)
    for k in _topic_ks(cfg.lda_topics, multipliers):
        _lda(cfg, docs, names, k, log)
    for k in _topic_ks(cfg.nmf_topics, multipliers):
        _nmf(cfg, docs, names, k, log)
    _hdp(cfg, docs, names, log)


def _file_id(name: str) -> str:
    """Extract numeric prefix '001' from '001__some_paper.txt'."""
    return name.split("__", 1)[0]


def _kmeans(cfg: Config, docs: list[str], names: list[str], k: int, log: Callable[[str], None]) -> None:
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(dtm)

    feature_names = vec.get_feature_names_out()
    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_KMEANS_cluster_centroids_{k}.txt"), "w", encoding="utf-8") as f:
        for i, centroid in enumerate(model.cluster_centers_):
            top_idx = centroid.argsort()[-10:][::-1]
            f.write(f"Cluster {i + 1}: {', '.join(feature_names[j] for j in top_idx)}\n")

    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_KMeans_{k}_doc2topic_assignments.txt"), "w", encoding="utf-8") as f:
        for idx, cluster in enumerate(model.labels_):
            f.write(f"{_file_id(names[idx])}, {cluster + 1}\n")

    log(f"KMeans completed: {k} topics.")


def _lda(cfg: Config, docs: list[str], names: list[str], k: int, log: Callable[[str], None]) -> None:
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = matutils.Sparse2Corpus(dtm, documents_columns=False)

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

    doc_topics: list[int] = []
    counts = {f"Topic {i + 1}": 0 for i in range(k)}
    for doc in bow:
        probs = lda.get_document_topics(doc)
        assigned = max(probs, key=lambda x: x[1])[0] if probs else 0
        doc_topics.append(assigned)
        counts[f"Topic {assigned + 1}"] += 1

    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_LDA_{k}.txt"), "w", encoding="utf-8") as f:
        for i in range(k):
            label = f"Topic {i + 1}"
            terms = lda.show_topic(i, topn=cfg.lda_terms_per_topic)
            f.write(f"{label} ({counts[label]} docs): ")
            f.write(", ".join(f"{w} ({p:.4f})" for w, p in terms) + "\n")

    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_LDA_{k}_doc2topic_assignments.txt"), "w", encoding="utf-8") as f:
        for idx, topic in enumerate(doc_topics):
            if idx < len(names):
                f.write(f"{_file_id(names[idx])}, {topic + 1}\n")

    log(f"LDA completed: {k} topics.")


def _nmf(cfg: Config, docs: list[str], names: list[str], k: int, log: Callable[[str], None]) -> None:
    vec = TfidfVectorizer(lowercase=True)
    dtm = vec.fit_transform(docs)
    model = NMF(n_components=k, random_state=42, max_iter=cfg.nmf_max_iter)
    model.fit(dtm)

    terms = vec.get_feature_names_out()
    doc_topics = model.transform(dtm).argmax(axis=1)
    counts = {f"Topic {i + 1}": int((doc_topics == i).sum()) for i in range(k)}

    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_NMF_{k}.txt"), "w", encoding="utf-8") as f:
        for i, topic in enumerate(model.components_):
            label = f"Topic {i + 1}"
            top_idx = topic.argsort()[: -cfg.nmf_terms_per_topic - 1 : -1]
            f.write(f"{label} ({counts[label]} docs): ")
            f.write(", ".join(f"{terms[j]} ({topic[j]:.4f})" for j in top_idx) + "\n")

    with open(os.path.join(cfg.directory_analysis, f"analysis_topicmodel_nmf_{k}_doc2topic_assignments.txt"), "w", encoding="utf-8") as f:
        for idx, topic in enumerate(doc_topics):
            f.write(f"{_file_id(names[idx])}, {int(topic) + 1}\n")

    log(f"NMF completed: {k} topics.")


def _hdp(cfg: Config, docs: list[str], names: list[str], log: Callable[[str], None]) -> None:
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = matutils.Sparse2Corpus(dtm, documents_columns=False)

    hdp = HdpModel(corpus=bow, id2word=id2word)
    results = hdp.show_topics(num_words=cfg.terms_per_topic_hdp, formatted=False)
    n_topics = len(results)

    doc_topics: list[int] = []
    counts: dict[str, int] = {}
    for doc in bow:
        probs = hdp[doc]
        assigned = max(probs, key=lambda x: x[1])[0] if probs else -1
        doc_topics.append(assigned)
        label = f"Topic {assigned + 1}" if assigned != -1 else "No Topic"
        counts[label] = counts.get(label, 0) + 1

    with open(os.path.join(cfg.directory_analysis, "analysis_topicmodel_HDP.txt"), "w", encoding="utf-8") as f:
        for idx, (_, topic_words) in enumerate(results):
            label = f"Topic {idx + 1}"
            f.write(f"{label} ({counts.get(label, 0)} docs): ")
            f.write(", ".join(f"{w} ({p:.4f})" for w, p in topic_words) + "\n")

    with open(os.path.join(cfg.directory_analysis, "analysis_topicmodel_HDP_doc2topic_assignments.txt"), "w", encoding="utf-8") as f:
        for idx, topic in enumerate(doc_topics):
            label = topic + 1 if topic != -1 else "No Topic"
            f.write(f"{_file_id(names[idx])}, {label}\n")

    log(f"HDP completed: {n_topics} topics.")
