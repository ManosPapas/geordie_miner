"""Shared corpus / CSV readers and small helpers.

Centralises the things that were being re-implemented in many modules: reading
`metadata.csv`, splitting multi-valued fields, mapping documents to years /
processed files, year-bracket sizing, country naming, and community detection.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from config import Config

_MULTI_RE = re.compile(r";|\|")


def read_dicts(path: str) -> List[Dict[str, str]]:
    """Read a CSV into a list of row dicts (empty list if the file is absent)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def read_metadata(cfg: Config) -> List[Dict[str, str]]:
    """All rows of `metadata.csv` (empty list if it doesn't exist yet)."""
    return read_dicts(cfg.output_path("metadata.csv"))


def metadata_by_doc(cfg: Config) -> Dict[str, Dict[str, str]]:
    """`metadata.csv` keyed by doc_id."""
    return {r["doc_id"]: r for r in read_metadata(cfg) if r.get("doc_id")}


def split_multi(value: str) -> List[str]:
    """Split a multi-valued field on ';' / '|' into trimmed, non-empty parts."""
    return [v.strip() for v in _MULTI_RE.split(str(value or "")) if v.strip()]


def doc_years(cfg: Config, as_int: bool = False) -> Dict:
    """doc_id -> publication year. With as_int=True, only rows with a numeric year
    are included (as ints); otherwise the raw string year is returned for every doc."""
    out: Dict = {}
    for r in read_metadata(cfg):
        year = (r.get("year") or "").strip()
        if as_int:
            if year.isdigit():
                out[r["doc_id"]] = int(year)
        else:
            out[r["doc_id"]] = year
    return out


def processed_files_by_doc(cfg: Config) -> Dict[str, str]:
    """doc_id -> processed filename in cfg.directory_processed."""
    out: Dict[str, str] = {}
    if os.path.isdir(cfg.directory_processed):
        for fn in sorted(os.listdir(cfg.directory_processed)):
            if fn.endswith(".txt"):
                out[fn.split("__", 1)[0]] = fn
    return out


def bracket_size(cfg: Config, span: int) -> int:
    """Year-bracket width: config 5/10, else auto (5 if span <= 25 years, else 10)."""
    raw = str(getattr(cfg, "longitudinal_bracket_years", "auto")).lower()
    if raw in ("5", "10"):
        return int(raw)
    return 5 if span <= 25 else 10


def country_name(code: str) -> str:
    """ISO alpha-2 -> country name (best-effort; returns the code if unresolved)."""
    code = (code or "").strip()
    if len(code) != 2:
        return code
    try:
        import pycountry
        rec = pycountry.countries.get(alpha_2=code.upper())
        return rec.name if rec else code
    except Exception:
        return code


def top_terms(directory: str, top: int) -> List[str]:
    """Top-`top` lemmatised terms (lowercased) from a run dir's terms_lemmatised.csv.

    Shared by the markdown and HTML comparison reports so they can't drift.
    """
    import pandas as pd
    path = os.path.join(directory, "terms_lemmatised.csv")
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).head(top)["Term"].astype(str).str.lower().tolist()


def topic_files(directory: str) -> Dict[str, str]:
    """Map topic-model label -> file contents for `topics_*.txt` in a run dir."""
    out: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if fn.startswith("topics_") and fn.endswith(".txt"):
            label = fn[len("topics_"):-len(".txt")]
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
                out[label] = f.read()
    return out


def communities(graph: nx.Graph) -> Dict:
    """Map each node to a 1-based greedy-modularity community id.

    Falls back to a single community when the algorithm can't run.
    """
    undirected = graph.to_undirected() if graph.is_directed() else graph
    try:
        comms = greedy_modularity_communities(undirected, weight="weight")
    except Exception:
        return {node: 1 for node in graph.nodes()}
    mapping: Dict = {}
    for cid, members in enumerate(comms, start=1):
        for node in members:
            mapping[node] = cid
    return mapping
