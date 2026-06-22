"""Topic evolution over time brackets.

Fits a quick LDA within each publication-year bracket, tracks how topic
prevalence shifts across brackets, and detects splits (one topic -> several) and
merges (several -> one) by matching topics in adjacent brackets on top-term
Jaccard overlap.

Outputs:
- `topic_evolution.csv`   — bracket, topic_id, n_docs, share, top_terms
- `topic_transitions.csv` — from_bracket, from_topic, to_bracket, to_topic, jaccard, type
- `topic_evolution.html`  — Sankey of flows between bracket topics (if plotly available)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

from config import Config
from corpus_io import bracket_size, doc_years, processed_files_by_doc

_MATCH_THRESHOLD = 0.2
_TOP_TERMS = 10


def _fit_bracket(docs: List[str], k: int, passes: int) -> Tuple[List[Set[str]], List[int]]:
    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = list(matutils.Sparse2Corpus(dtm, documents_columns=False))
    lda = LdaModel(corpus=bow, num_topics=k, id2word=id2word, passes=passes,
                   alpha="auto", eta="auto", random_state=42, per_word_topics=False)
    topics = [set(w for w, _ in lda.show_topic(i, topn=_TOP_TERMS)) for i in range(k)]
    sizes = [0] * k
    for doc in bow:
        probs = lda.get_document_topics(doc, minimum_probability=0)
        if probs:
            sizes[max(probs, key=lambda x: x[1])[0]] += 1
    return topics, sizes


def _jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def run_topic_evolution(cfg: Config, log: Callable[[str], None]) -> None:
    years = doc_years(cfg, as_int=True)
    files = processed_files_by_doc(cfg)
    paired = {d: years[d] for d in years if d in files}
    if len(paired) < max(4, cfg.longitudinal_min_docs * 2):
        log(f"  topic_evolution: too few dated docs ({len(paired)}); skipping.")
        return

    yr_values = sorted(set(paired.values()))
    size = bracket_size(cfg, yr_values[-1] - yr_values[0])
    brackets: Dict[int, List[str]] = defaultdict(list)
    for doc_id, yr in paired.items():
        brackets[(yr // size) * size].append(doc_id)

    log(f"  topic_evolution: {len(paired)} dated docs, {size}-year brackets.")

    bracket_topics: Dict[str, List[Tuple[int, Set[str], int]]] = {}
    evo_rows: List[dict] = []
    for start in sorted(brackets):
        ids = brackets[start]
        if len(ids) < cfg.longitudinal_min_docs:
            continue
        label = f"{start}-{start + size - 1}"
        docs = []
        for d in ids:
            with open(os.path.join(cfg.directory_processed, files[d]), "r", encoding="utf-8") as f:
                docs.append(f.read())
        k = max(2, min(cfg.lda_topics, len(docs) - 1))
        try:
            topics, sizes = _fit_bracket(docs, k, cfg.lda_passes)
        except Exception as e:
            log(f"  topic_evolution: bracket {label} failed ({e}); skipped.")
            continue
        bracket_topics[label] = [(i, topics[i], sizes[i]) for i in range(k)]
        total = sum(sizes) or 1
        for i in range(k):
            evo_rows.append({"bracket": label, "topic_id": i + 1, "n_docs": sizes[i],
                             "share": round(sizes[i] / total, 3),
                             # persist all top terms so the science-map heatmap's Jaccard
                             # matches the transition Jaccard (both use these term sets).
                             "top_terms": ", ".join(sorted(topics[i]))})

    with open(cfg.output_path("topic_evolution.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bracket", "topic_id", "n_docs", "share", "top_terms"])
        w.writeheader(); w.writerows(evo_rows)

    produced = [lbl for lbl in (f"{s}-{s + size - 1}" for s in sorted(brackets)) if lbl in bracket_topics]
    trans_rows: List[dict] = []
    for a, b in zip(produced, produced[1:]):
        TA, TB = bracket_topics[a], bracket_topics[b]
        out_matches: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        in_matches: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for ti, terms_i, _ in TA:
            for tj, terms_j, _ in TB:
                jac = _jaccard(terms_i, terms_j)
                if jac >= _MATCH_THRESHOLD:
                    out_matches[ti].append((tj, jac))
                    in_matches[tj].append((ti, jac))
        for ti, _, _ in TA:
            outs = out_matches.get(ti, [])
            if not outs:
                trans_rows.append({"from_bracket": a, "from_topic": ti + 1, "to_bracket": b,
                                   "to_topic": "", "jaccard": 0.0, "type": "dissolves"})
            for tj, jac in outs:
                typ = "split" if len(outs) >= 2 else ("merge" if len(in_matches.get(tj, [])) >= 2 else "continuation")
                trans_rows.append({"from_bracket": a, "from_topic": ti + 1, "to_bracket": b,
                                   "to_topic": tj + 1, "jaccard": round(jac, 3), "type": typ})
        for tj, _, _ in TB:
            if not in_matches.get(tj):
                trans_rows.append({"from_bracket": a, "from_topic": "", "to_bracket": b,
                                   "to_topic": tj + 1, "jaccard": 0.0, "type": "emerges"})

    with open(cfg.output_path("topic_transitions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from_bracket", "from_topic", "to_bracket", "to_topic", "jaccard", "type"])
        w.writeheader(); w.writerows(trans_rows)

    _sankey(bracket_topics, trans_rows, cfg, log)
    n_split = sum(1 for r in trans_rows if r["type"] == "split")
    n_merge = sum(1 for r in trans_rows if r["type"] == "merge")
    log(f"  topic_evolution: {len(produced)} bracket(s); {n_split} split(s), {n_merge} merge(s) "
        f"-> topic_evolution.csv, topic_transitions.csv.")


def _sankey(bracket_topics, trans_rows, cfg: Config, log: Callable[[str], None]) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    node_labels: List[str] = []
    node_idx: Dict[Tuple[str, int], int] = {}
    for label, topics in bracket_topics.items():
        for ti, _, _ in topics:
            node_idx[(label, ti + 1)] = len(node_labels)
            node_labels.append(f"{label} T{ti + 1}")
    src, tgt, val = [], [], []
    for r in trans_rows:
        if r["to_topic"] == "" or r["from_topic"] == "":
            continue
        s = node_idx.get((r["from_bracket"], r["from_topic"]))
        t = node_idx.get((r["to_bracket"], r["to_topic"]))
        if s is not None and t is not None:
            src.append(s); tgt.append(t); val.append(max(0.05, r["jaccard"]))
    if not src:
        return
    fig = go.Figure(go.Sankey(node=dict(label=node_labels, pad=15, thickness=15),
                              link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(title_text="Thematic evolution (topic flows across time brackets)", font_size=10)
    fig.write_html(cfg.output_path("topic_evolution.html"), include_plotlyjs="inline")
    log("  topic_evolution: Sankey -> topic_evolution.html")
