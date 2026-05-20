"""Extract per-paper metadata (year, title, DOI) from PDFs and text files.

Best-effort, fully offline. Combines:
- Embedded PDF metadata (when the file is a PDF — pypdf reads the title/author fields)
- Regex/heuristics over the first ~2000 characters (for `.txt` files or PDFs
  with missing metadata)

Output: `metadata.csv` with columns: doc_id, file, title, year, doi.

Not perfect — paper metadata varies wildly. Treat outputs as a starting point,
not gospel.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import Config


# Regex patterns
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19[89]\d|20\d{2})\b")  # 1980–2099
_ACCEPTED_RE = re.compile(r"(?:received|accepted|published|copyright|©)\s*[^\n]*?\b(19[89]\d|20\d{2})\b", re.IGNORECASE)


def _doc_id(filename: str) -> str:
    return filename.split("__", 1)[0]


def _read_head(path: Path, max_chars: int = 2500) -> str:
    """Read the first chunk of a file (handles encoding issues gracefully)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _detect_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    if not m:
        return None
    # Strip trailing punctuation that's often part of the surrounding text
    doi = m.group(0).rstrip(".,;:)]}'\"")
    return doi


def _detect_year(text: str) -> Optional[int]:
    """Find a publication-y year. Prefer one near 'copyright/accepted/published';
    otherwise the earliest 4-digit year in the first ~2500 chars."""
    m = _ACCEPTED_RE.search(text)
    if m:
        return int(m.group(1))
    candidates = [int(y) for y in _YEAR_RE.findall(text[:2500])]
    if not candidates:
        return None
    # Heuristic: prefer years that look like recent publications (1990–current+1)
    plausible = [y for y in candidates if 1990 <= y <= 2030]
    return min(plausible) if plausible else min(candidates)


_BANNER_PATTERNS = re.compile(
    r"(?i)(contents lists available|sciencedirect|sciverse|elsevier|"
    r"taylor\s*&\s*francis|wiley online library|springer link|"
    r"journal homepage|cite this article|published:|received:|accepted:|"
    r"©\s*\d{4}|copyright|all rights reserved|"
    r"page \d+ of \d+|vol\.?\s*\d+|issue\s*\d+|"
    r"^[A-Z][A-Z\s,&]+$)"
)


def _detect_title(text: str) -> Optional[str]:
    """First non-trivial, non-banner line is usually the title."""
    for line in text.splitlines():
        cleaned = line.strip()
        if len(cleaned) < 20 or len(cleaned) > 300:
            continue
        # Skip URL / DOI / page-number / digits-only lines
        if cleaned.lower().startswith(("http", "doi", "www")):
            continue
        if cleaned.replace(" ", "").isdigit():
            continue
        # Skip common journal banner / metadata text
        if _BANNER_PATTERNS.search(cleaned):
            continue
        # Skip all-caps lines (running headers)
        if cleaned.isupper() and len(cleaned) > 30:
            continue
        # Plausible title
        return cleaned
    return None


def _pdf_metadata(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (title, doi) from embedded PDF metadata if available."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        info = reader.metadata or {}
        title = info.get("/Title") or info.get("title") or None
        # DOI is rarely in PDF metadata directly; check Subject as a long-shot
        subject = info.get("/Subject") or ""
        doi = _detect_doi(subject) if subject else None
        return (title.strip() if title else None, doi)
    except Exception:
        return (None, None)


def extract_metadata(cfg: Config, log: Callable[[str], None]) -> None:
    """Scan files in cfg.directory_data (the original input) for metadata.

    Falls back to scanning cfg.directory_text for .txt-only corpora.
    """
    source = cfg.directory_data if os.path.isdir(cfg.directory_data) else cfg.directory_text
    files = sorted(os.listdir(source))
    if not files:
        log("  metadata: no files found.")
        return

    rows: List[Dict[str, str]] = []
    for idx, name in enumerate(files):
        path = Path(source) / name
        text = ""
        title = None
        doi = None

        if name.lower().endswith(".pdf"):
            title, doi = _pdf_metadata(path)
            # Also grab some text for year/title fallback
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                pages_to_scan = min(2, len(reader.pages))
                text = "\n".join(reader.pages[i].extract_text() or "" for i in range(pages_to_scan))
            except Exception:
                text = ""
        else:
            text = _read_head(path)

        if not title:
            title = _detect_title(text)
        if not doi:
            doi = _detect_doi(text)
        year = _detect_year(text)

        rows.append({
            "doc_id": f"{idx + 1:03d}",
            "file": name,
            "title": title or "",
            "year": str(year) if year else "",
            "doi": doi or "",
        })

    out_path = cfg.output_path("metadata.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "file", "title", "year", "doi"])
        writer.writeheader()
        writer.writerows(rows)

    n_with_year = sum(1 for r in rows if r["year"])
    n_with_title = sum(1 for r in rows if r["title"])
    n_with_doi = sum(1 for r in rows if r["doi"])
    log(
        f"  metadata: extracted for {len(rows)} files — "
        f"{n_with_title} with title, {n_with_year} with year, {n_with_doi} with DOI."
    )
