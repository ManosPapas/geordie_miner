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
from typing import Callable, Dict, List, Optional

from config import Config
from corpus_io import read_dicts
from sections import detect_sections


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


def _pdf_info(path: Path) -> Dict[str, str]:
    """Return embedded PDF metadata fields (title, author, doi, producer)."""
    out = {"title": "", "author": "", "doi": "", "producer": ""}
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        info = reader.metadata or {}
        out["title"] = (info.get("/Title") or "").strip()
        out["author"] = (info.get("/Author") or "").strip()
        out["producer"] = ((info.get("/Producer") or "") + " " + (info.get("/Creator") or "")).strip()
        subject = info.get("/Subject") or ""
        out["doi"] = _detect_doi(subject) or ""
    except Exception:
        pass
    return out


# ISSN: four digits, hyphen, three digits, then a check digit (digit or X).
_ISSN_RE = re.compile(r"\b\d{4}-\d{3}[\dxX]\b")

# Publisher / source detection: (canonical name, keyword regex).
_PUBLISHER_KEYWORDS = [
    ("Elsevier", r"elsevier|sciencedirect|sciverse"),
    ("Springer", r"springer"),
    ("Wiley", r"wiley"),
    ("Taylor & Francis", r"taylor\s*&\s*francis|tandfonline|routledge"),
    ("IEEE", r"\bieee\b"),
    ("SAGE", r"\bsage\s+publications?\b|\bsagepub\b"),
    ("MDPI", r"\bmdpi\b"),
    ("Emerald", r"emerald"),
    ("ACM", r"\bacm\b|association for computing machinery"),
    ("Frontiers", r"\bfrontiers\b"),
    ("Springer Nature / Nature", r"nature publishing|springer nature"),
    ("Oxford University Press", r"oxford university press|\boup\b"),
    ("Cambridge University Press", r"cambridge university press"),
]

# Journal-ish line: contains a strong journal keyword and is a plausible length.
_JOURNAL_LINE_RE = re.compile(
    r"(journal|proceedings|transactions|review|letters|bulletin|annals|conference|quarterly)",
    re.IGNORECASE,
)


def _detect_issn(text: str) -> str:
    m = _ISSN_RE.search(text)
    return m.group(0) if m else ""


def _detect_publisher(text: str, producer: str = "") -> str:
    hay = f"{text[:3000]} {producer}".lower()
    for name, pattern in _PUBLISHER_KEYWORDS:
        if re.search(pattern, hay, re.IGNORECASE):
            return name
    return ""


def _detect_journal(text: str) -> str:
    """Best-effort source/journal title from a journal-ish line in the front matter."""
    for line in text[:2500].splitlines():
        cleaned = line.strip()
        if not (12 <= len(cleaned) <= 120):
            continue
        if cleaned.lower().startswith(("http", "doi", "www", "©")):
            continue
        if _JOURNAL_LINE_RE.search(cleaned) and not cleaned.replace(" ", "").isdigit():
            return cleaned
    return ""


def _detect_authors(text: str, pdf_author: str = "") -> str:
    """Prefer embedded PDF author; else a heuristic author line after the title."""
    if pdf_author and not pdf_author.lower().startswith(("microsoft", "adobe", "pdf")):
        return pdf_author[:200]

    title = _detect_title(text) or ""
    lines = [ln.strip() for ln in text.splitlines()]
    try:
        start = lines.index(title.strip()) + 1 if title else 0
    except ValueError:
        start = 0
    for line in lines[start:start + 6]:
        if not (5 <= len(line) <= 200):
            continue
        if any(ch.isdigit() for ch in line):
            continue
        if _BANNER_PATTERNS.search(line):
            continue
        # Author-ish: commas or "and", and a couple of capitalised tokens.
        caps = sum(1 for w in line.split() if w[:1].isupper())
        if ("," in line or " and " in line.lower()) and caps >= 2:
            return line[:200]
    return ""


def _detect_methodology(full_text: str) -> str:
    """Return a short snippet of the methodology section if one is detected."""
    for name, start, end in detect_sections(full_text):
        if name == "methodology":
            snippet = re.sub(r"\s+", " ", full_text[start:end]).strip()
            return snippet[:160]
    return ""


FIELDNAMES = [
    "doc_id", "file", "title", "authors", "affiliations", "year", "journal",
    "volume", "issue", "pages", "issn", "publisher", "doi", "abstract",
    "keywords", "country", "cited_by", "methodology", "notes",
]

# Per-field provenance precedence (low -> high). A higher source overlays a lower
# one; a user override is never overwritten by enrichment.
_PROVENANCE_RANK = {"pdf": 0, "import": 1, "openalex": 2, "crossref": 2, "scopus": 2, "override": 3}


def _original_pdf(data_dir: str, text_filename: str) -> Optional[Path]:
    """Map a converted `NNN__stem.txt` back to its source PDF in `data_dir`, if any."""
    stem = text_filename.split("__", 1)[-1]
    if stem.lower().endswith(".txt"):
        stem = stem[:-4]
    cand = Path(data_dir) / (stem + ".pdf")
    return cand if cand.exists() else None


def _load_keyed(path: str) -> Dict[str, Dict[str, str]]:
    """Load a CSV with a doc_id column into {doc_id: {field: value}}."""
    out: Dict[str, Dict[str, str]] = {}
    for row in read_dicts(path):
        did = (row.get("doc_id") or "").strip()
        if did:
            out[did] = row
    return out


def _load_overrides(cfg: Config) -> Dict[str, Dict[str, str]]:
    """User overrides from `metadata_overrides.csv` (doc_id, field, value); data dir wins over config dir."""
    candidates = [
        os.path.join(cfg.directory_data, "metadata_overrides.csv"),
        os.path.join(os.path.dirname(cfg.config_path), "metadata_overrides.csv"),
    ]
    out: Dict[str, Dict[str, str]] = {}
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                did, field, value = (row.get("doc_id") or "").strip(), (row.get("field") or "").strip(), (row.get("value") or "").strip()
                if did and field in FIELDNAMES:
                    out.setdefault(did, {})[field] = value
        break
    return out


def _overlay(record: Dict[str, str], prov: Dict[str, str], values: Dict[str, str], source: str) -> None:
    """Apply non-empty `values` from `source` if they outrank the current provenance."""
    rank = _PROVENANCE_RANK.get(source, 0)
    for field, value in values.items():
        if field not in FIELDNAMES or field in ("doc_id", "file", "notes"):
            continue
        value = (value or "").strip()
        if not value:
            continue
        if rank >= _PROVENANCE_RANK.get(prov.get(field, ""), -1):
            record[field] = value
            prov[field] = source


def _write_metadata(rows: List[Dict[str, str]], prov_rows: List[Dict[str, str]], cfg: Config) -> None:
    with open(cfg.output_path("metadata.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(cfg.output_path("metadata_provenance.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "field", "source"])
        writer.writerows([(r["doc_id"], r["field"], r["source"]) for r in prov_rows])


_KEY_FIELDS = ("title", "authors", "year", "journal", "doi")


def extract_metadata(cfg: Config, log: Callable[[str], None]) -> None:
    """Extract per-document bibliometric fields, then merge external import + user
    overrides on top, recording per-field provenance.

    Precedence (low -> high): PDF heuristics -> imported bibliographic file ->
    user overrides. Provider enrichment (highest but below overrides) is applied
    separately by `enrich_metadata`. Fields that can't be filled are left blank
    and noted in `notes` — never inferred.
    """
    text_dir = cfg.directory_text
    files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))
    if not files:
        log("  metadata: no converted text files found.")
        return

    imported = _load_keyed(cfg.output_path("imported_metadata.csv"))
    overrides = _load_overrides(cfg)

    rows: List[Dict[str, str]] = []
    prov_rows: List[Dict[str, str]] = []
    for name in files:
        doc_id = _doc_id(name)
        record = {f: "" for f in FIELDNAMES}
        record["doc_id"], record["file"] = doc_id, name
        prov: Dict[str, str] = {}

        full_text = _read_head(Path(text_dir) / name, max_chars=200000)
        head = full_text[:2500]
        pdf = _original_pdf(cfg.directory_data, name)
        pdf_info = _pdf_info(pdf) if pdf else {"title": "", "author": "", "doi": "", "producer": ""}

        heuristic = {
            "title": pdf_info["title"] or _detect_title(head) or "",
            "authors": _detect_authors(head, pdf_info["author"]),
            "year": str(_detect_year(head) or ""),
            "journal": _detect_journal(head),
            "issn": _detect_issn(full_text),
            "publisher": _detect_publisher(full_text, pdf_info["producer"]),
            "doi": pdf_info["doi"] or _detect_doi(full_text) or "",
            "methodology": _detect_methodology(full_text),
        }
        _overlay(record, prov, heuristic, "pdf")
        if doc_id in imported:
            _overlay(record, prov, imported[doc_id], "import")
        if doc_id in overrides:
            _overlay(record, prov, overrides[doc_id], "override")

        missing = [f for f in _KEY_FIELDS if not record[f]]
        record["notes"] = ("not extracted: " + ", ".join(missing)) if missing else ""
        rows.append(record)
        prov_rows.extend({"doc_id": doc_id, "field": f, "source": s} for f, s in prov.items())

    _write_metadata(rows, prov_rows, cfg)

    def _n(field: str) -> int:
        return sum(1 for r in rows if r[field])

    extra = f" ({len(imported)} from import)" if imported else ""
    log(
        f"  metadata: {len(rows)} docs{extra} — {_n('title')} title, {_n('authors')} authors, "
        f"{_n('year')} year, {_n('journal')} journal, {_n('doi')} DOI, {_n('abstract')} abstract, "
        f"{_n('country')} country. Provenance -> metadata_provenance.csv."
    )


def enrich_metadata(cfg: Config, log: Callable[[str], None]) -> None:
    """Override/enrich metadata fields from an online provider (by DOI then title)."""
    rows = read_dicts(cfg.output_path("metadata.csv"))
    if not rows:
        log("  enrich: no metadata.csv rows; skipping.")
        return

    from providers import make_provider
    provider = make_provider(cfg.provider, cfg, log)
    if provider is None:
        log("  enrich: no usable provider; skipping.")
        return

    prov_lookup = {(r["doc_id"], r["field"]): r["source"]
                   for r in read_dicts(cfg.output_path("metadata_provenance.csv"))}
    cap = cfg.provider_max or len(rows)
    log(f"  enrich: querying {provider.name} for up to {cap} doc(s)...")

    n_enriched = 0
    for row in rows[:cap]:
        doi, title = (row.get("doi") or "").strip(), (row.get("title") or "").strip()
        rec = provider.fetch_by_doi(doi) if doi else (provider.fetch_by_title(title) if title else None)
        if not rec:
            continue
        changed = False
        for field, value in rec.items():
            if field not in FIELDNAMES or field in ("doc_id", "file", "notes"):
                continue
            value = str(value or "").strip()
            if value and prov_lookup.get((row["doc_id"], field)) != "override":
                row[field] = value
                prov_lookup[(row["doc_id"], field)] = provider.name
                changed = True
        n_enriched += int(changed)

    prov_rows = [{"doc_id": d, "field": f, "source": s} for (d, f), s in prov_lookup.items()]
    _write_metadata(rows, prov_rows, cfg)
    log(f"  enrich: updated {n_enriched}/{min(cap, len(rows))} doc(s) from {provider.name}.")
