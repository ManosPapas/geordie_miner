"""Citation-based impact indicators + co-citation / bibliographic-coupling networks.

Builds on the within-corpus citation graph produced by `references.py`
(`references.csv` column `cited_doc_in_corpus`). Within-corpus citation links are
usually sparse (a paper rarely cites another paper *in the same corpus*), so this
module degrades gracefully and says so when the graph is thin.

Outputs (when there are edges):
- `citation_impact.csv`     — per document: in/out degree + PageRank/betweenness/eigenvector
- `journal_impact.csv`      — per venue: n_docs, local citations received, external cited_by
- `cocitation_documents.*`  — documents frequently cited together (GEXF + VOSviewer)
- `coupling_documents.*`    — documents sharing references (GEXF + VOSviewer)
- `cocitation_journals.gexf`, `coupling_journals.gexf`, `cocitation_authors.gexf`,
  `coupling_authors.gexf` — the same, aggregated to venues / authors
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

from config import Config
from corpus_io import metadata_by_doc, split_multi
from vosviewer import export_vosviewer


def _citation_edges(cfg: Config) -> List[Tuple[str, str]]:
    """(citer_doc_id, cited_doc_id) pairs within the corpus, from references.csv."""
    path = cfg.output_path("references.csv")
    edges: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        return edges
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            citer, cited = (r.get("doc_id") or "").strip(), (r.get("cited_doc_in_corpus") or "").strip()
            if citer and cited and citer != cited:
                edges.append((citer, cited))
    return edges


def _centrality(fn, nodes) -> Dict:
    """Run a centrality function, returning zeros for every node if it can't converge."""
    try:
        return fn()
    except Exception:
        return {n: 0.0 for n in nodes}


def _export_matrix_network(labels, matrix, threshold, gexf_path, vos_prefix, log, kind) -> None:
    g = nx.Graph()
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i][j]
            if w >= threshold:  # callers pass threshold >= 1
                g.add_edge(labels[i], labels[j], weight=int(w))
    if g.number_of_edges() == 0:
        log(f"  citation: {kind} network empty (no pairs >= {threshold}); skipped.")
        return
    nx.write_gexf(g, gexf_path)
    if vos_prefix:
        export_vosviewer(g, vos_prefix + "_map.txt", vos_prefix + "_network.txt", log)
    log(f"  citation: {kind} network — {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {os.path.basename(gexf_path)}")


def _aggregate(matrix, doc_entities: List[List[str]]):
    """Project a doc x doc matrix up to entity x entity (e.g. journals, authors)."""
    entities = sorted({e for es in doc_entities for e in es})
    idx = {e: k for k, e in enumerate(entities)}
    m = np.zeros((len(entities), len(entities)))
    n = len(doc_entities)
    for i in range(n):
        for j in range(n):
            if i == j or matrix[i][j] <= 0:
                continue
            for ei in doc_entities[i]:
                for ej in doc_entities[j]:
                    if ei != ej:
                        m[idx[ei]][idx[ej]] += matrix[i][j]
    return entities, m


def run_citation_analysis(cfg: Config, log: Callable[[str], None]) -> None:
    meta = metadata_by_doc(cfg)
    edges = _citation_edges(cfg)
    doc_ids = list(meta.keys()) or sorted({d for e in edges for d in e})
    if not doc_ids:
        log("  citation: no documents; skipping.")
        return
    if not edges:
        log("  citation: within-corpus citation graph is empty (0 internal links) — "
            "impact/co-citation/coupling skipped. This is normal when papers don't cite "
            "others in the same corpus.")
        return

    idx = {d: i for i, d in enumerate(doc_ids)}
    n = len(doc_ids)
    M = np.zeros((n, n))  # M[i][j] = 1 if doc i cites doc j
    for citer, cited in edges:
        if citer in idx and cited in idx:
            M[idx[citer]][idx[cited]] = 1

    # ---- Impact ----
    if cfg.compute_impact:
        dg = nx.DiGraph()
        dg.add_nodes_from(doc_ids)
        for citer, cited in edges:
            if citer in idx and cited in idx:
                dg.add_edge(citer, cited)
        pr = _centrality(lambda: nx.pagerank(dg), doc_ids)
        bt = _centrality(lambda: nx.betweenness_centrality(dg), doc_ids)
        ev = _centrality(lambda: nx.eigenvector_centrality(dg, max_iter=500), doc_ids)

        with open(cfg.output_path("citation_impact.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "title", "journal", "local_in_degree", "local_out_degree",
                        "pagerank", "betweenness", "eigenvector", "external_cited_by"])
            for d in sorted(doc_ids, key=lambda x: -int(M[:, idx[x]].sum())):
                r = meta.get(d, {})
                w.writerow([d, (r.get("title") or "")[:120], r.get("journal", ""),
                            int(M[:, idx[d]].sum()), int(M[idx[d], :].sum()),
                            f"{pr.get(d, 0):.5f}", f"{bt.get(d, 0):.5f}", f"{ev.get(d, 0):.5f}",
                            r.get("cited_by", "")])

        # Venue impact
        venue_docs: Dict[str, List[str]] = defaultdict(list)
        for d in doc_ids:
            j = (meta.get(d, {}).get("journal") or "").strip()
            if j:
                venue_docs[j].append(d)
        with open(cfg.output_path("journal_impact.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["journal", "n_docs", "local_citations_in", "external_cited_by"])
            for j, ds in sorted(venue_docs.items(), key=lambda x: -len(x[1])):
                local_in = int(sum(M[:, idx[d]].sum() for d in ds))
                ext = sum(int(meta[d]["cited_by"]) for d in ds if str(meta[d].get("cited_by", "")).isdigit())
                w.writerow([j, len(ds), local_in, ext])
        log(f"  citation: impact computed for {n} docs / {len(venue_docs)} venues -> citation_impact.csv, journal_impact.csv")

    # doc -> entities maps for aggregation
    doc_journals = [[(meta.get(d, {}).get("journal") or "").strip()] if (meta.get(d, {}).get("journal") or "").strip() else [] for d in doc_ids]
    doc_authors = [split_multi(meta.get(d, {}).get("authors", "")) for d in doc_ids]

    # ---- Co-citation: C = M^T M (docs cited together) ----
    if cfg.compute_cocitation:
        C = M.T @ M
        np.fill_diagonal(C, 0)
        _export_matrix_network(doc_ids, C, 2, cfg.output_path("cocitation_documents.gexf"),
                               cfg.output_path("cocitation_documents_vosviewer"), log, "co-citation (documents)")
        je, jm = _aggregate(C, doc_journals)
        _export_matrix_network(je, jm, 1, cfg.output_path("cocitation_journals.gexf"), "", log, "co-citation (journals)")
        ae, am = _aggregate(C, doc_authors)
        _export_matrix_network(ae, am, 1, cfg.output_path("cocitation_authors.gexf"), "", log, "co-citation (authors)")

    # ---- Bibliographic coupling: B = M M^T (docs sharing references) ----
    if cfg.compute_coupling:
        B = M @ M.T
        np.fill_diagonal(B, 0)
        _export_matrix_network(doc_ids, B, 2, cfg.output_path("coupling_documents.gexf"),
                               cfg.output_path("coupling_documents_vosviewer"), log, "bibliographic coupling (documents)")
        je, jm = _aggregate(B, doc_journals)
        _export_matrix_network(je, jm, 1, cfg.output_path("coupling_journals.gexf"), "", log, "coupling (journals)")
        ae, am = _aggregate(B, doc_authors)
        _export_matrix_network(ae, am, 1, cfg.output_path("coupling_authors.gexf"), "", log, "coupling (authors)")
