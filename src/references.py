"""Reference / bibliography extraction + cross-corpus citation network.

Pure regex, no external dependencies. Works on the References / Bibliography
section detected by `sections.py`. For each reference, attempts to extract
authors, year, title, journal. Then matches references against other papers in
the same corpus (by publication year + title-keyword overlap) to build a
citation graph.

Output:
- `references.csv`     — one row per reference: doc_id, ref_num, authors, year, title, journal, raw
- `citation_network.gexf` — directed graph: edges from citer to cited (only for
  references that match another paper in the same corpus)
"""

from __future__ import annotations

import csv
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from config import Config
from sections import detect_sections
from vosviewer import export_vosviewer


# Detect a year in parens or standalone.
_YEAR_RE = re.compile(r"\((19[89]\d|20\d{2})\)|(?<!\d)((?:19[89]\d|20\d{2}))(?!\d)")

# Split a references section into individual entries. Real-world bibliographies
# use many formats; we use multiple heuristics and pick the best.
_NUMBERED_RE = re.compile(r"\n\s*\[?\s*\d+\s*[\]\.\)]\s+")
_AUTHOR_INIT_RE = re.compile(r"\n(?=[A-Z][\w\-]+,\s*[A-Z]\.)")  # Smith, J.


def _split_references(text: str) -> List[str]:
    """Split a references-section blob into individual reference entries."""
    text = text.strip()
    if not text:
        return []

    # Try numbered ([1], 1., etc.)
    parts = _NUMBERED_RE.split(text)
    if len(parts) > 5:
        return [p.strip() for p in parts if p.strip()]

    # Try "Surname, Initial."
    parts = _AUTHOR_INIT_RE.split(text)
    if len(parts) > 5:
        return [p.strip() for p in parts if p.strip()]

    # Fallback: blank-line-separated
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _parse_reference(raw: str) -> Dict[str, str]:
    """Best-effort parse of one reference string into (authors, year, title, journal)."""
    raw = re.sub(r"\s+", " ", raw).strip()

    year_m = _YEAR_RE.search(raw)
    year = ""
    if year_m:
        year = year_m.group(1) or year_m.group(2) or ""

    # Authors = text before the year (truncated)
    authors = ""
    if year and year_m:
        authors_chunk = raw[: year_m.start()].rstrip(" .,;:(")
        # Clip to first 4 author groups for readability
        authors = authors_chunk[:200]

    # Title = first sentence-like chunk after the year
    title = ""
    rest = raw[year_m.end():].strip(" .,;:)") if year_m else raw
    title_m = re.match(r"\s*([^.]+\.)", rest)
    if title_m:
        title = title_m.group(1).rstrip(".").strip()

    # Journal = next chunk after title
    journal = ""
    if title_m:
        after = rest[title_m.end():].strip(" .,;:")
        journal_m = re.match(r"([^.,0-9]+)", after)
        if journal_m:
            journal = journal_m.group(1).strip()

    return {"authors": authors, "year": year, "title": title, "journal": journal, "raw": raw}


def _title_keywords(title: str) -> set:
    """Return a set of lowercase content words from a title (≥4 chars)."""
    words = re.findall(r"[A-Za-z]{4,}", title.lower())
    stop = {"with", "from", "this", "that", "what", "when", "where", "study", "research", "paper", "analysis"}
    return {w for w in words if w not in stop}


def _build_corpus_index(text_dir: str) -> Dict[str, Dict]:
    """Index every doc in `text_dir` by year + title keywords for matching.

    Returns dict: doc_id → {file, title, year, title_keywords}.
    The "title" is heuristically taken as the first long line of the doc.
    """
    from metadata import _detect_title, _detect_year  # reuse helpers

    index: Dict[str, Dict] = {}
    for filename in sorted(os.listdir(text_dir)):
        if not filename.endswith(".txt"):
            continue
        doc_id = filename.split("__", 1)[0]
        with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
            head = f.read(2500)
        title = _detect_title(head) or ""
        year = _detect_year(head)
        index[doc_id] = {
            "file": filename,
            "title": title,
            "year": str(year) if year else "",
            "title_keywords": _title_keywords(title),
        }
    return index


def _match_reference_to_doc(ref: Dict[str, str], index: Dict[str, Dict]) -> Optional[str]:
    """Return the doc_id (in `index`) that this reference best matches, or None."""
    ref_year = ref.get("year") or ""
    ref_kw = _title_keywords(ref.get("title", ""))
    if not ref_kw:
        return None

    best: Tuple[int, Optional[str]] = (0, None)
    for doc_id, info in index.items():
        if ref_year and info["year"] and ref_year != info["year"]:
            continue
        overlap = len(ref_kw & info["title_keywords"])
        # Require at least 2 keyword overlap to count as a match
        if overlap >= 2 and overlap > best[0]:
            best = (overlap, doc_id)
    return best[1]


def extract_references(cfg: Config, log: Callable[[str], None]) -> None:
    """Extract bibliographies from every .txt in `cfg.directory_text` and build a citation network."""
    text_dir = cfg.directory_text
    files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))
    if not files:
        log("  references: no input files; skipping.")
        return

    log(f"  references: building corpus index for cross-citation matching...")
    index = _build_corpus_index(text_dir)

    rows: List[Dict[str, str]] = []
    edges: List[Tuple[str, str]] = []
    n_with_refs = 0

    for filename in files:
        doc_id = filename.split("__", 1)[0]
        with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()

        # Find the references section using sections.py
        refs_text = ""
        for name, start, end in detect_sections(text):
            if name == "references":
                refs_text = text[start:end]
                break
        if not refs_text:
            continue
        n_with_refs += 1

        entries = _split_references(refs_text)
        for ref_num, raw in enumerate(entries[:200], start=1):  # cap to 200 refs per paper
            parsed = _parse_reference(raw)
            cited_doc = _match_reference_to_doc(parsed, index)
            rows.append({
                "doc_id": doc_id,
                "ref_num": str(ref_num),
                "authors": parsed["authors"][:150],
                "year": parsed["year"],
                "title": parsed["title"][:200],
                "journal": parsed["journal"][:100],
                "cited_doc_in_corpus": cited_doc or "",
                "raw": parsed["raw"][:300],
            })
            if cited_doc and cited_doc != doc_id:
                edges.append((doc_id, cited_doc))

    # Write references.csv
    out_path = cfg.output_path("references.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "doc_id", "ref_num", "authors", "year", "title", "journal", "cited_doc_in_corpus", "raw"
        ])
        writer.writeheader()
        writer.writerows(rows)
    log(
        f"  references: extracted {len(rows)} references from {n_with_refs}/{len(files)} files. "
        f"{len(edges)} matched another paper in this corpus."
    )

    # Build + write the citation network (directed: citer → cited)
    g = nx.DiGraph()
    for doc_id in index:
        g.add_node(doc_id, label=index[doc_id]["title"][:80] if index[doc_id]["title"] else doc_id)
    for citer, cited in edges:
        if g.has_edge(citer, cited):
            g[citer][cited]["weight"] += 1
        else:
            g.add_edge(citer, cited, weight=1)
    nx.write_gexf(g, cfg.output_path("citation_network.gexf"))
    log(
        f"  citation network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges. "
        f"Open citation_network.gexf in Gephi to explore."
    )

    # Also export the citation graph in VOSviewer's native format. (The whole
    # references stage is already wrapped non-fatally by the caller.)
    if g.number_of_edges() > 0:
        export_vosviewer(
            g,
            cfg.output_path("citation_network_vosviewer_map.txt"),
            cfg.output_path("citation_network_vosviewer_network.txt"),
            log,
        )
