"""Shared helpers for reading topic-model output and labelling topics.

Kept dependency-free and separate from summary.py / html_report.py so both can
import it without a circular dependency.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

# A topics_<model>_<K>.txt line looks like:
#   Topic 1 (12 docs): term (0.1234), other (0.0987), ...
_HEADER_RE = re.compile(r"^(Topic[^:(]*?)\((\d+)\s*docs?\):\s*(.*)$", re.IGNORECASE)


def _parse_term_weight(item: str) -> Optional[Tuple[str, float]]:
    item = item.strip()
    if not item:
        return None
    # term may itself contain spaces, so split on the LAST " (".
    if " (" in item and item.endswith(")"):
        term, rest = item.rsplit(" (", 1)
        try:
            return term.strip(), float(rest.rstrip(")"))
        except ValueError:
            return term.strip(), 0.0
    return item, 0.0


def parse_topics_file(path: str) -> List[Tuple[str, int, List[Tuple[str, float]]]]:
    """Parse a topics_*.txt file into [(topic_header, n_docs, [(term, weight), ...]), ...]."""
    out: List[Tuple[str, int, List[Tuple[str, float]]]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _HEADER_RE.match(line)
            if not m:
                continue
            header = m.group(1).strip()
            n_docs = int(m.group(2))
            terms = [tw for tw in (_parse_term_weight(it) for it in m.group(3).split(", ")) if tw]
            out.append((header, n_docs, terms))
    return out


def topic_label(terms: List[Tuple[str, float]], n: int = 3) -> str:
    """Build a short, human-readable suggested label from the top-weighted terms.

    A descriptive hint (e.g. "Virtual, Reality, Store") — explicitly a suggestion,
    not a definitive classification.
    """
    picked: List[str] = []
    for term, _ in terms:
        t = term.strip()
        if t and t.lower() not in {p.lower() for p in picked}:
            picked.append(t)
        if len(picked) >= n:
            break
    return ", ".join(p.title() for p in picked) if picked else "(no terms)"


def model_label_from_filename(filename: str) -> str:
    """`topics_lda_5.txt` -> `LDA K=5`; `topics_bertopic.txt` -> `BERTOPIC`."""
    stem = os.path.basename(filename)
    stem = stem[len("topics_"):] if stem.startswith("topics_") else stem
    stem = stem[:-4] if stem.endswith(".txt") else stem
    return stem.upper().replace("_", " K=")
