"""Normalised author / institution / country tables built from metadata.csv.

These underpin the collaboration analyses and the descriptive bibliometric
rankings. Author disambiguation is heuristic (case/whitespace fold); it is far
more reliable when provider enrichment (OpenAlex) has supplied clean names.

Outputs: `authors.csv`, `institutions.csv`, `countries.csv`.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from typing import Callable, Dict

from config import Config
from corpus_io import country_name, read_metadata, split_multi


def _norm_author(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def build_entity_tables(cfg: Config, log: Callable[[str], None]) -> None:
    rows = read_metadata(cfg)
    if not rows:
        log("  entities: no metadata.csv; skipping entity tables.")
        return

    authors: Dict[str, dict] = defaultdict(lambda: {"display": "", "docs": set(), "years": set(), "cited": 0})
    institutions: Dict[str, set] = defaultdict(set)
    countries: Dict[str, set] = defaultdict(set)

    for r in rows:
        doc_id = r.get("doc_id", "")
        year = (r.get("year") or "").strip()
        cited = int(r["cited_by"]) if str(r.get("cited_by", "")).strip().isdigit() else 0
        for name in split_multi(r.get("authors", "")):
            key = _norm_author(name)
            a = authors[key]
            a["display"] = a["display"] or name
            a["docs"].add(doc_id)
            if year:
                a["years"].add(year)
            a["cited"] += cited
        for inst in split_multi(r.get("affiliations", "")):
            institutions[inst].add(doc_id)
        for c in split_multi(r.get("country", "")):
            countries[c].add(doc_id)

    # authors.csv
    with open(cfg.output_path("authors.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["author", "n_papers", "first_year", "last_year", "total_cited_by", "doc_ids"])
        for a in sorted(authors.values(), key=lambda x: (-len(x["docs"]), x["display"].lower())):
            yrs = sorted(a["years"])
            w.writerow([a["display"], len(a["docs"]), yrs[0] if yrs else "", yrs[-1] if yrs else "",
                        a["cited"], ";".join(sorted(a["docs"]))])

    # institutions.csv
    with open(cfg.output_path("institutions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["institution", "n_papers", "doc_ids"])
        for inst, docs in sorted(institutions.items(), key=lambda x: (-len(x[1]), x[0].lower())):
            w.writerow([inst, len(docs), ";".join(sorted(docs))])

    # countries.csv
    with open(cfg.output_path("countries.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_code", "country", "n_papers", "doc_ids"])
        for code, docs in sorted(countries.items(), key=lambda x: (-len(x[1]), x[0])):
            w.writerow([code, country_name(code), len(docs), ";".join(sorted(docs))])

    log(
        f"  entities: {len(authors)} author(s), {len(institutions)} institution(s), "
        f"{len(countries)} country/countries -> authors.csv, institutions.csv, countries.csv."
    )
