"""Descriptive bibliometrics: publication trends, ranked entities with impact
proxies, charts, and a multi-sheet Excel export.

Reads `metadata.csv` plus (when present) the normalised entity tables
(`authors.csv`, `institutions.csv`, `countries.csv`) and citation impact
(`journal_impact.csv`). Writes:
- `bibliometric_*.csv` (one ranked table each, documented schemas)
- `bibliometrics.xlsx` (the same tables as sheets + a `_schema` sheet)
- `bibliometrics_summary.csv` (long-format counts, kept for the summary page)
- charts: bib_publication_trends.png, bib_journals.png, bib_authors.png,
  bib_top_institutions.png, bib_top_countries.png (+ optional choropleth)
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from typing import Callable, List, Optional

import pandas as pd

from config import Config
from corpus_io import read_dicts
from plotting import plt

_AUTHOR_SPLIT_RE = re.compile(r"\s*(?:;| and |&)\s*", re.IGNORECASE)


def _bar(path: str, labels: List[str], values: List[float], title: str, xlabel: str) -> None:
    if not labels:
        return
    plt.figure(figsize=(9, max(3.5, 0.4 * len(labels) + 1)))
    y = range(len(labels))
    plt.barh(list(y), values, color="#2563eb")
    plt.yticks(list(y), labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _rolling(values: List[int], window: int = 3) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(round(sum(values[i - window + 1:i + 1]) / window, 2))
    return out


def build_bibliometrics(cfg: Config, log: Callable[[str], None], top_n: int = 20) -> None:
    meta_path = cfg.output_path("metadata.csv")
    if not os.path.exists(meta_path):
        log("  bibliometrics: no metadata.csv — skipping aggregates.")
        return
    df = pd.read_csv(meta_path, dtype=str).fillna("")
    if df.empty:
        log("  bibliometrics: metadata.csv is empty — skipping.")
        return

    out_csv = lambda n: cfg.output_path(n)
    journal_impact = {r["journal"]: r for r in read_dicts(out_csv("journal_impact.csv"))}
    authors_tbl = read_dicts(out_csv("authors.csv"))
    institutions_tbl = read_dicts(out_csv("institutions.csv"))
    countries_tbl = read_dicts(out_csv("countries.csv"))

    # ---- Publication trends ----
    years = Counter(y for y in df.get("year", pd.Series(dtype=str)).tolist() if str(y).strip().isdigit())
    trend_rows = []
    if years:
        yr_keys = list(range(min(int(y) for y in years), max(int(y) for y in years) + 1))
        counts = [years.get(str(y), 0) for y in yr_keys]
        roll = _rolling(counts, 3)
        for i, y in enumerate(yr_keys):
            prev = counts[i - 1] if i > 0 else 0
            growth = round((counts[i] - prev) / prev * 100, 1) if prev else ""
            trend_rows.append({"year": y, "n_publications": counts[i],
                               "rolling_avg_3y": roll[i] if roll[i] is not None else "",
                               "growth_pct": growth})
    trends_df = pd.DataFrame(trend_rows, columns=["year", "n_publications", "rolling_avg_3y", "growth_pct"])

    # ---- Ranked journals (+ impact proxies) ----
    journals = Counter(j.strip() for j in df.get("journal", pd.Series(dtype=str)).tolist() if str(j).strip())
    jrows = []
    for j, c in journals.most_common():
        imp = journal_impact.get(j, {})
        jrows.append({"journal": j, "n_documents": c,
                      "local_citations_in": imp.get("local_citations_in", ""),
                      "external_cited_by": imp.get("external_cited_by", "")})
    journals_df = pd.DataFrame(jrows, columns=["journal", "n_documents", "local_citations_in", "external_cited_by"])

    # ---- Ranked authors / institutions / countries (prefer entity tables) ----
    if authors_tbl:
        authors_df = pd.DataFrame([
            {"author": r["author"], "n_papers": int(r["n_papers"]),
             "total_cited_by": r.get("total_cited_by", ""),
             "first_year": r.get("first_year", ""), "last_year": r.get("last_year", "")}
            for r in authors_tbl
        ])
    else:
        ac: Counter = Counter()
        for field in df.get("authors", pd.Series(dtype=str)).tolist():
            ac.update(a for a in _AUTHOR_SPLIT_RE.split(field or "") if len(a.strip()) >= 3)
        authors_df = pd.DataFrame([{"author": a, "n_papers": c, "total_cited_by": "",
                                    "first_year": "", "last_year": ""} for a, c in ac.most_common()])

    institutions_df = (pd.DataFrame([{"institution": r["institution"], "n_papers": int(r["n_papers"])} for r in institutions_tbl])
                       if institutions_tbl else pd.DataFrame(columns=["institution", "n_papers"]))
    countries_df = (pd.DataFrame([{"country_code": r["country_code"], "country": r["country"], "n_papers": int(r["n_papers"])} for r in countries_tbl])
                    if countries_tbl else pd.DataFrame(columns=["country_code", "country", "n_papers"]))

    # ---- Write per-table CSVs ----
    trends_df.to_csv(out_csv("bibliometric_publication_trends.csv"), index=False)
    journals_df.to_csv(out_csv("bibliometric_top_journals.csv"), index=False)
    authors_df.to_csv(out_csv("bibliometric_top_authors.csv"), index=False)
    institutions_df.to_csv(out_csv("bibliometric_top_institutions.csv"), index=False)
    countries_df.to_csv(out_csv("bibliometric_top_countries.csv"), index=False)

    # ---- Long-format summary (kept for the summary page) ----
    with open(out_csv("bibliometrics_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "key", "count"])
        for _, r in trends_df.iterrows():
            w.writerow(["year", r["year"], r["n_publications"]])
        for _, r in journals_df.iterrows():
            w.writerow(["journal", r["journal"], r["n_documents"]])
        for _, r in authors_df.iterrows():
            w.writerow(["author", r["author"], r["n_papers"]])

    # ---- Charts ----
    if not trends_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(trends_df["year"], trends_df["n_publications"], marker="o", label="publications")
        roll = pd.to_numeric(trends_df["rolling_avg_3y"], errors="coerce")
        if roll.notna().any():
            plt.plot(trends_df["year"], roll, linestyle="--", color="#d97706", label="3-yr rolling avg")
        plt.xlabel("year"); plt.ylabel("publications"); plt.title("Publications per year")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_csv("bib_publication_trends.png"), dpi=130); plt.close()

    _bar(out_csv("bib_journals.png"), journals_df["journal"].head(top_n).tolist(),
         journals_df["n_documents"].head(top_n).tolist(), f"Top {min(top_n, len(journals_df))} sources / journals", "documents")
    _bar(out_csv("bib_authors.png"), authors_df["author"].head(top_n).tolist(),
         authors_df["n_papers"].head(top_n).tolist(), f"Top {min(top_n, len(authors_df))} authors", "papers")
    if not institutions_df.empty:
        _bar(out_csv("bib_top_institutions.png"), institutions_df["institution"].head(top_n).tolist(),
             institutions_df["n_papers"].head(top_n).tolist(), f"Top {min(top_n, len(institutions_df))} institutions", "papers")
    if not countries_df.empty:
        _bar(out_csv("bib_top_countries.png"), countries_df["country"].head(top_n).tolist(),
             countries_df["n_papers"].head(top_n).tolist(), f"Top {min(top_n, len(countries_df))} countries", "papers")
        _choropleth(cfg, countries_df, log)

    # ---- Multi-sheet Excel ----
    _write_excel(cfg, trends_df, journals_df, authors_df, institutions_df, countries_df, log)

    log(
        f"  bibliometrics: {len(trends_df)} year(s), {len(journals_df)} journal(s), "
        f"{len(authors_df)} author(s), {len(institutions_df)} institution(s), {len(countries_df)} country/countries "
        f"-> bibliometric_*.csv, bibliometrics.xlsx, charts."
    )


def _choropleth(cfg: Config, countries_df: pd.DataFrame, log: Callable[[str], None]) -> None:
    """Optional world choropleth of document counts (needs plotly; ISO-3 via pycountry)."""
    try:
        import plotly.express as px
        import pycountry
    except Exception:
        return
    iso3, counts = [], []
    for _, r in countries_df.iterrows():
        code = str(r["country_code"]).strip().upper()
        rec = pycountry.countries.get(alpha_2=code) if len(code) == 2 else None
        if rec:
            iso3.append(rec.alpha_3)
            counts.append(r["n_papers"])
    if not iso3:
        return
    fig = px.choropleth(locations=iso3, color=counts, color_continuous_scale=cfg.visual_colour_scheme,
                        labels={"color": "documents"}, title="Documents by country")
    fig.write_html(cfg.output_path("bib_country_map.html"), include_plotlyjs="inline")
    log("  bibliometrics: country choropleth -> bib_country_map.html")


_SCHEMA = [
    ("Publication trends", "year / n_publications / rolling_avg_3y / growth_pct", "Articles per year with a 3-year rolling average and year-on-year % growth."),
    ("Top journals", "journal / n_documents / local_citations_in / external_cited_by", "Source ranking; local citations are within-corpus, external is provider cited-by."),
    ("Top authors", "author / n_papers / total_cited_by / first_year / last_year", "Author ranking with activity span and summed external citations."),
    ("Top institutions", "institution / n_papers", "Institution ranking (populated when affiliations are available)."),
    ("Top countries", "country_code / country / n_papers", "Country ranking (populated when country data is available)."),
]


def _write_excel(cfg, trends_df, journals_df, authors_df, institutions_df, countries_df, log) -> None:
    schema_df = pd.DataFrame(_SCHEMA, columns=["sheet", "columns", "description"])
    try:
        with pd.ExcelWriter(cfg.output_path("bibliometrics.xlsx"), engine="openpyxl") as xl:
            trends_df.to_excel(xl, sheet_name="Publication trends", index=False)
            journals_df.to_excel(xl, sheet_name="Top journals", index=False)
            authors_df.to_excel(xl, sheet_name="Top authors", index=False)
            institutions_df.to_excel(xl, sheet_name="Top institutions", index=False)
            countries_df.to_excel(xl, sheet_name="Top countries", index=False)
            schema_df.to_excel(xl, sheet_name="_schema", index=False)
    except Exception as e:
        log(f"  bibliometrics: Excel export failed (non-fatal): {e}")
