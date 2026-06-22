"""Bibliographic file import: BibTeX / RIS / CSV -> a common record schema.

Lets users feed a reference-manager export (or a Scopus/WoS CSV) as the corpus.
Each record becomes a document whose text body is the title + abstract, and a
sidecar `imported_metadata.csv` is written for the metadata stage to merge in.

The common record schema (all values are strings; multi-valued fields are
"; "-joined) is:
    title, abstract, keywords, authors, affiliations, year, journal,
    volume, issue, pages, doi, country, cited_by
"""

from __future__ import annotations

import csv
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from config import Config
from textutil import sanitize_text

STANDARD_FIELDS = [
    "title", "abstract", "keywords", "authors", "affiliations", "year",
    "journal", "volume", "issue", "pages", "doi", "country", "cited_by",
]

# Case-insensitive default column names for CSV import (covers Scopus/WoS exports).
_CSV_DEFAULTS: Dict[str, List[str]] = {
    "title": ["title", "article title", "document title"],
    "abstract": ["abstract"],
    "keywords": ["keywords", "author keywords", "index keywords"],
    "authors": ["authors", "author", "author full names", "author(s)"],
    "affiliations": ["affiliations", "author affiliations", "affiliation"],
    "year": ["year", "publication year"],
    "journal": ["journal", "source title", "source", "publication", "journaltitle"],
    "volume": ["volume", "vol"],
    "issue": ["issue", "number"],
    "pages": ["pages", "page start"],
    "doi": ["doi"],
    "country": ["country", "countries"],
    "cited_by": ["cited by", "cited by count", "times cited", "citation count"],
}


def _empty_record() -> Dict[str, str]:
    return {f: "" for f in STANDARD_FIELDS}


def _join(values) -> str:
    if isinstance(values, (list, tuple)):
        return "; ".join(str(v).strip() for v in values if str(v).strip())
    return str(values or "").strip()


def _year(value: str) -> str:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return m.group(0) if m else ""


def parse_bibtex(path: str) -> List[Dict[str, str]]:
    import bibtexparser
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        db = bibtexparser.load(f)
    out: List[Dict[str, str]] = []
    for e in db.entries:
        rec = _empty_record()
        rec["title"] = _join(e.get("title", "").replace("{", "").replace("}", ""))
        rec["abstract"] = _join(e.get("abstract", ""))
        rec["keywords"] = _join([k.strip() for k in re.split(r"[;,]", e.get("keywords", "")) if k.strip()])
        rec["authors"] = _join([a.strip() for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()])
        rec["year"] = _year(e.get("year", "") or e.get("date", ""))
        rec["journal"] = _join(e.get("journal", "") or e.get("journaltitle", "") or e.get("booktitle", ""))
        rec["volume"] = _join(e.get("volume", ""))
        rec["issue"] = _join(e.get("number", ""))
        rec["pages"] = _join(e.get("pages", ""))
        rec["doi"] = _join(e.get("doi", "")).lower()
        out.append(rec)
    return out


def parse_ris(path: str) -> List[Dict[str, str]]:
    import rispy
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        entries = rispy.load(f)
    out: List[Dict[str, str]] = []
    for e in entries:
        rec = _empty_record()
        rec["title"] = _join(e.get("primary_title") or e.get("title") or e.get("translated_title", ""))
        rec["abstract"] = _join(e.get("abstract", ""))
        rec["keywords"] = _join(e.get("keywords", []))
        rec["authors"] = _join(e.get("authors") or e.get("first_authors", []))
        rec["year"] = _year(e.get("publication_year") or e.get("year", ""))
        rec["journal"] = _join(e.get("journal_name") or e.get("secondary_title") or e.get("alternate_title1", ""))
        rec["volume"] = _join(e.get("volume", ""))
        rec["issue"] = _join(e.get("number", ""))
        sp, ep = e.get("start_page", ""), e.get("end_page", "")
        rec["pages"] = f"{sp}-{ep}".strip("-") if (sp or ep) else ""
        rec["doi"] = _join(e.get("doi", "")).lower()
        out.append(rec)
    return out


def _parse_mapping(csv_mapping: str) -> Dict[str, str]:
    """`title:Col, year:Yr` -> {'title': 'col', 'year': 'yr'} (lowercased columns)."""
    out: Dict[str, str] = {}
    for pair in csv_mapping.split(","):
        if ":" in pair:
            field, col = pair.split(":", 1)
            field, col = field.strip().lower(), col.strip().lower()
            if field in STANDARD_FIELDS and col:
                out[field] = col
    return out


def parse_csv(path: str, csv_mapping: str = "") -> List[Dict[str, str]]:
    import pandas as pd
    df = pd.read_csv(path, dtype=str).fillna("")
    lower_cols = {c.lower().strip(): c for c in df.columns}
    overrides = _parse_mapping(csv_mapping)

    def column_for(field: str) -> Optional[str]:
        if field in overrides and overrides[field] in lower_cols:
            return lower_cols[overrides[field]]
        for cand in _CSV_DEFAULTS.get(field, []):
            if cand in lower_cols:
                return lower_cols[cand]
        return None

    field_cols = {f: column_for(f) for f in STANDARD_FIELDS}
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        rec = _empty_record()
        for f, col in field_cols.items():
            if col is not None:
                rec[f] = str(row[col]).strip()
        rec["year"] = _year(rec["year"])
        rec["doi"] = rec["doi"].lower()
        out.append(rec)
    return out


def find_bib_file(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (path, kind) of the first .bib/.ris/.csv in data_dir, else (None, None)."""
    for name in sorted(os.listdir(data_dir)):
        ext = name.lower().rsplit(".", 1)[-1]
        if ext in ("bib", "bibtex"):
            return os.path.join(data_dir, name), "bibtex"
        if ext == "ris":
            return os.path.join(data_dir, name), "ris"
        if ext == "csv":
            return os.path.join(data_dir, name), "csv"
    return None, None


def detect_and_parse(path: str, kind: str, csv_mapping: str = "") -> List[Dict[str, str]]:
    if kind == "bibtex":
        return parse_bibtex(path)
    if kind == "ris":
        return parse_ris(path)
    return parse_csv(path, csv_mapping)


def run_bibliographic_import(cfg: Config, src_path: str, kind: str, log: Callable[[str], None]) -> int:
    """Parse `src_path`, write one text file per record + `imported_metadata.csv`."""
    log(f"Bibliographic import mode — parsing {kind} file '{os.path.basename(src_path)}'.")
    try:
        records = detect_and_parse(src_path, kind, cfg.csv_mapping)
    except Exception as e:
        log(f"  ERROR: could not parse {kind} file: {e}. Check the format / csv_mapping.")
        return 0

    if not records:
        log("  bibliographic import: no records found.")
        return 0

    width = max(3, len(str(len(records))))
    rows: List[Dict[str, str]] = []
    written = n_empty = 0
    for rec in records:
        body = sanitize_text((rec.get("title", "") + ". " + rec.get("abstract", "")).strip(" ."))
        if not body.strip():
            n_empty += 1
            continue
        doc_id = f"{written + 1:0{width}d}"
        with open(os.path.join(cfg.directory_text, f"{doc_id}__bib.txt"), "w", encoding="utf-8") as f:
            f.write(body)
        row = {"doc_id": doc_id, "file": f"{doc_id}__bib.txt"}
        row.update({f: sanitize_text(rec.get(f, "")) for f in STANDARD_FIELDS})
        rows.append(row)
        written += 1

    out_csv = cfg.output_path("imported_metadata.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "file"] + STANDARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_with_abstract = sum(1 for r in rows if r["abstract"])
    log(
        f"  bibliographic import: {written} record(s) -> documents "
        f"({n_with_abstract} with abstracts, {n_empty} empty skipped); "
        f"sidecar imported_metadata.csv written for enrichment/merge."
    )
    return written
