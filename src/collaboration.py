"""Collaboration / co-authorship structures.

Builds co-authorship networks at author, institution and country level from
metadata.csv, computes basic network measures (degree, component membership,
greedy-modularity community), summarises the structure in text, and exports each
network as GEXF + VOSviewer + CSV edge/node lists.

Outputs per level <lvl> in {authors, institutions, countries}:
  collab_<lvl>.gexf, collab_<lvl>_edges.csv, collab_<lvl>_nodes.csv,
  collab_<lvl>_vosviewer_map.txt/_network.txt
plus `collaboration_summary.txt`.
"""

from __future__ import annotations

import csv
from typing import Callable, List

import networkx as nx

from config import Config
from corpus_io import communities, read_metadata, split_multi
from vosviewer import export_vosviewer


def _build_network(doc_entities: List[List[str]]) -> nx.Graph:
    g = nx.Graph()
    for ents in doc_entities:
        uniq = sorted(set(ents))
        g.add_nodes_from(uniq)
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                u, v = uniq[a], uniq[b]
                if g.has_edge(u, v):
                    g[u][v]["weight"] += 1
                else:
                    g.add_edge(u, v, weight=1)
    return g


def _analyse_level(level: str, doc_entities: List[List[str]], cfg: Config, log: Callable[[str], None]) -> List[str]:
    g = _build_network(doc_entities)
    if g.number_of_nodes() == 0:
        log(f"  collaboration: no {level} data; skipped.")
        return [f"### {level.title()}", "", "_(no data)_", ""]

    community_of = communities(g)
    components = list(nx.connected_components(g))
    comp_of = {node: i for i, comp in enumerate(components) for node in comp}
    degree = dict(g.degree())
    wdegree = dict(g.degree(weight="weight"))

    # Node + edge exports
    with open(cfg.output_path(f"collab_{level}_nodes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "degree", "weighted_degree", "component", "community"])
        for node in sorted(g.nodes(), key=lambda x: -degree[x]):
            w.writerow([node, degree[node], wdegree[node], comp_of[node], community_of.get(node, 0)])
    with open(cfg.output_path(f"collab_{level}_edges.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight"])
        for u, v, d in g.edges(data=True):
            w.writerow([u, v, d.get("weight", 1)])

    for node, cid in community_of.items():
        g.nodes[node]["community"] = cid
    nx.write_gexf(g, cfg.output_path(f"collab_{level}.gexf"))
    if g.number_of_edges() > 0:
        export_vosviewer(g, cfg.output_path(f"collab_{level}_vosviewer_map.txt"),
                         cfg.output_path(f"collab_{level}_vosviewer_network.txt"), log)

    isolated = [n for n in g.nodes() if degree[n] == 0]
    top = sorted(g.nodes(), key=lambda x: -degree[x])[:10]
    n_comm = len(set(community_of.values()))

    log(f"  collaboration: {level} — {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
        f"{len(components)} component(s), {n_comm} cluster(s), {len(isolated)} isolated.")

    lines = [f"### {level.title()}", ""]
    lines.append(f"- Nodes: **{g.number_of_nodes()}**, edges: **{g.number_of_edges()}**, "
                 f"components: **{len(components)}**, clusters: **{n_comm}**, isolated: **{len(isolated)}**.")
    collaborators = ", ".join(f"{n} ({degree[n]})" for n in top if degree[n] > 0)
    if collaborators:
        lines.append(f"- Top collaborators (by degree): {collaborators}")
    if components:
        biggest = max(components, key=len)
        lines.append(f"- Largest connected component: **{len(biggest)}** members.")
    lines.append("")
    return lines


def run_collaboration(cfg: Config, log: Callable[[str], None]) -> None:
    rows = read_metadata(cfg)
    if not rows:
        log("  collaboration: no metadata.csv; skipping.")
        return

    authors = [split_multi(r.get("authors", "")) for r in rows]
    institutions = [split_multi(r.get("affiliations", "")) for r in rows]
    countries = [split_multi(r.get("country", "")) for r in rows]

    summary: List[str] = ["# Collaboration structure", ""]
    summary += _analyse_level("authors", authors, cfg, log)
    summary += _analyse_level("institutions", institutions, cfg, log)
    summary += _analyse_level("countries", countries, cfg, log)

    with open(cfg.output_path("collaboration_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
