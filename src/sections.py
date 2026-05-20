"""Section detection for academic papers.

Heuristic, regex-based, pure Python. No new dependencies. Detects common
section headers (Abstract / Introduction / Methods / Results / Discussion /
References / Acknowledgements) and splits a paper into per-section text.

When enabled, writes `<doc>__section_<name>.txt` files alongside the standard
`<doc>.txt` in `text_sections/`, and lets the preprocess stage skip sections
listed in `exclude_sections` in config.txt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import Config


# Canonical section names → list of regex patterns that match real-world variants.
# Patterns are case-insensitive and anchored to start of line.
SECTION_PATTERNS: Dict[str, List[str]] = {
    "abstract": [
        r"^\s*abstract\s*$",
        r"^\s*abstract[\.:\-—]\s*",
        r"^\s*summary\s*$",
    ],
    "introduction": [
        r"^\s*\d*[\.\)]?\s*introduction\s*$",
        r"^\s*1\.?\s+introduction",
        r"^\s*background\s*$",
    ],
    "literature_review": [
        r"^\s*\d*[\.\)]?\s*(?:literature\s+review|related\s+work|theoretical\s+(?:framework|background))\s*$",
    ],
    "methodology": [
        r"^\s*\d*[\.\)]?\s*(?:methodology|methods?|materials?\s+and\s+methods?|research\s+(?:method|design|methodology)|experimental\s+(?:design|setup|methods?))\s*$",
    ],
    "results": [
        r"^\s*\d*[\.\)]?\s*(?:results?|findings?|analyses?|data\s+analysis|empirical\s+results?)\s*$",
    ],
    "discussion": [
        r"^\s*\d*[\.\)]?\s*(?:discussion|implications?|interpretation)\s*$",
    ],
    "conclusion": [
        r"^\s*\d*[\.\)]?\s*(?:conclusions?|concluding\s+remarks?|final\s+remarks?|summary\s+and\s+conclusions?)\s*$",
    ],
    "references": [
        r"^\s*references?\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*works?\s+cited\s*$",
    ],
    "acknowledgements": [
        r"^\s*acknowledge?ments?\s*$",
    ],
    "appendix": [
        r"^\s*appendix(?:\s+[a-z\d]+)?\s*$",
        r"^\s*supplementary\s+(?:material|information)\s*$",
    ],
}

# Compile once.
_COMPILED: Dict[str, List[re.Pattern]] = {
    name: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    for name, patterns in SECTION_PATTERNS.items()
}

CANONICAL_ORDER = list(SECTION_PATTERNS.keys())


def detect_sections(text: str) -> List[Tuple[str, int, int]]:
    """Find section boundaries in `text`.

    Returns a list of (section_name, start_offset, end_offset). Ranges are
    contiguous and cover the whole text. Sections appear in the order they
    occur. Text before the first detected header is labelled `preamble`.
    """
    # Collect (offset, section_name) for every header hit in the text.
    hits: List[Tuple[int, str]] = []
    for name, patterns in _COMPILED.items():
        for pat in patterns:
            for m in pat.finditer(text):
                hits.append((m.start(), name))

    if not hits:
        return [("body", 0, len(text))]

    # Sort by position, then deduplicate adjacent hits with same name.
    hits.sort()
    deduped: List[Tuple[int, str]] = []
    last_name = None
    for offset, name in hits:
        if name == last_name:
            continue
        deduped.append((offset, name))
        last_name = name

    # Build contiguous ranges.
    spans: List[Tuple[str, int, int]] = []
    if deduped[0][0] > 0:
        spans.append(("preamble", 0, deduped[0][0]))
    for i, (offset, name) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        spans.append((name, offset, end))
    return spans


def split_file_into_sections(text: str) -> Dict[str, str]:
    """Return a dict: section_name → concatenated text for that section.

    If the same section name appears twice (rare), the chunks are concatenated.
    """
    out: Dict[str, str] = {}
    for name, start, end in detect_sections(text):
        out.setdefault(name, "")
        out[name] += text[start:end]
    return out


def write_sections_for_corpus(
    cfg: Config,
    text_dir: str,
    sections_dir: str,
    log: Callable[[str], None],
) -> Dict[str, int]:
    """For each .txt in `text_dir`, write per-section files into `sections_dir/`.

    Returns counts of how many docs contained each section (for the log).
    """
    os.makedirs(sections_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))
    counts: Dict[str, int] = {}

    for filename in files:
        with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        sections = split_file_into_sections(text)
        stem = filename[:-4]  # strip .txt
        for section_name, body in sections.items():
            if not body.strip():
                continue
            counts[section_name] = counts.get(section_name, 0) + 1
            out_path = os.path.join(sections_dir, f"{stem}__section_{section_name}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(body)

    if counts:
        log(f"  sections: {len(files)} files split. Section coverage:")
        for name in CANONICAL_ORDER + ["preamble", "body"]:
            if name in counts:
                pct = counts[name] / len(files) * 100
                log(f"    {name:<20s} {counts[name]:>4d} files ({pct:.0f}%)")
    return counts


def filter_text_excluding_sections(text: str, exclude: List[str]) -> str:
    """Return `text` with the named sections removed.

    `exclude` is a list of canonical section names (e.g. ["references",
    "acknowledgements"]).
    """
    if not exclude:
        return text
    exclude_set = {e.strip().lower() for e in exclude}
    spans = detect_sections(text)
    kept = [text[start:end] for name, start, end in spans if name not in exclude_set]
    return "\n".join(kept)
