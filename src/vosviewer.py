"""Export a networkx graph in VOSviewer's native map + network file format.

VOSviewer (https://www.vosviewer.com/) reads a *pair* of tab-separated files:

- **map file**: one row per node — `id`, `label`, `cluster`,
  `weight<Links>`, `weight<Total link strength>`. (VOSviewer computes the layout
  itself from the network when x/y are absent.)
- **network file**: one row per edge — `id1`, `id2`, `weight` — referencing the
  integer ids from the map file. No header (classic VOSviewer network format).

This complements the existing Gephi-oriented GEXF export so the same
co-occurrence / citation graphs can be opened directly in VOSviewer.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import networkx as nx

from corpus_io import communities


def export_vosviewer(
    graph: nx.Graph,
    map_path: str,
    network_path: str,
    log: Callable[[str], None],
) -> None:
    """Write `map_path` and `network_path` for `graph` in VOSviewer format."""
    nodes: List = list(graph.nodes())
    if not nodes:
        log("  VOSviewer: graph is empty — nothing to export.")
        return

    node_id = {node: i + 1 for i, node in enumerate(nodes)}
    clusters = communities(graph)

    # Per-node metrics: Links = degree, Total link strength = summed edge weight.
    links: Dict = {n: 0 for n in nodes}
    strength: Dict = {n: 0.0 for n in nodes}
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1))
        links[u] += 1
        links[v] += 1
        strength[u] += w
        strength[v] += w

    def _num(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:.4f}"

    with open(map_path, "w", encoding="utf-8", newline="") as f:
        f.write("id\tlabel\tcluster\tweight<Links>\tweight<Total link strength>\n")
        for node in nodes:
            label = str(graph.nodes[node].get("label", node))
            f.write(
                f"{node_id[node]}\t{label}\t{clusters.get(node, 1)}\t"
                f"{links[node]}\t{_num(strength[node])}\n"
            )

    with open(network_path, "w", encoding="utf-8", newline="") as f:
        for u, v, data in graph.edges(data=True):
            f.write(f"{node_id[u]}\t{node_id[v]}\t{_num(float(data.get('weight', 1)))}\n")

    log(
        f"  VOSviewer export: {len(nodes)} nodes, {graph.number_of_edges()} edges -> "
        f"map + network files (import the pair via VOSviewer 'Map: VOSviewer files')."
    )
