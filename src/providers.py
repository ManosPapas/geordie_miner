"""Pluggable external metadata providers.

A `Provider` fetches bibliographic records (in the common schema from
`bibimport.STANDARD_FIELDS`, plus `openalex_id` / `ext_references` when known)
from an online source, by DOI, by title, or by keyword search.

Backends:
- OpenAlexProvider  — free, no key, ToS-friendly (default). Rich: authors,
  institutions, countries, citations, references, concepts.
- CrossrefProvider  — free, no key. Good DOI/title metadata.
- ScopusProvider    — requires an Elsevier API key in $SCOPUS_API_KEY (untested
  here). Falls back to the configured `provider_fallback` when the key is absent.

Network calls are defensive: timeouts + try/except returning None, never crashing
the pipeline.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional

import requests

from bibimport import STANDARD_FIELDS

_TIMEOUT = 20


def _empty() -> Dict[str, str]:
    rec = {f: "" for f in STANDARD_FIELDS}
    rec["openalex_id"] = ""
    rec["ext_references"] = []
    return rec


def _get_json(url: str, params: dict, log: Callable[[str], None]) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT,
                            headers={"User-Agent": "geordie_miner/0.2 (research tool)"})
        if resp.status_code != 200:
            log(f"  providers: {url} returned HTTP {resp.status_code}.")
            return None
        return resp.json()
    except Exception as e:
        log(f"  providers: request failed ({e.__class__.__name__}: {e}).")
        return None


# ---------------- OpenAlex ----------------

def _abstract_from_inverted(inv: Optional[dict]) -> str:
    if not inv:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlexProvider:
    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def __init__(self, mailto: str, log: Callable[[str], None]):
        self.mailto = mailto
        self.log = log

    def _params(self, extra: dict) -> dict:
        p = dict(extra)
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    def _to_record(self, w: dict) -> Dict[str, str]:
        rec = _empty()
        rec["title"] = w.get("display_name") or w.get("title") or ""
        rec["abstract"] = _abstract_from_inverted(w.get("abstract_inverted_index"))
        authors, affils, countries = [], [], []
        for a in w.get("authorships", []):
            nm = (a.get("author") or {}).get("display_name")
            if nm:
                authors.append(nm)
            for inst in a.get("institutions", []):
                if inst.get("display_name"):
                    affils.append(inst["display_name"])
                if inst.get("country_code"):
                    countries.append(inst["country_code"])
        rec["authors"] = "; ".join(authors)
        rec["affiliations"] = "; ".join(dict.fromkeys(affils))
        rec["country"] = "; ".join(dict.fromkeys(countries))
        rec["year"] = str(w.get("publication_year") or "")
        src = (w.get("primary_location") or {}).get("source") or {}
        rec["journal"] = src.get("display_name") or ""
        biblio = w.get("biblio") or {}
        rec["volume"] = biblio.get("volume") or ""
        rec["issue"] = biblio.get("issue") or ""
        rec["pages"] = "-".join(p for p in (biblio.get("first_page"), biblio.get("last_page")) if p)
        rec["doi"] = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        kws = [k.get("display_name") for k in (w.get("keywords") or []) if k.get("display_name")]
        if not kws:
            kws = [c.get("display_name") for c in (w.get("concepts") or [])[:6] if c.get("display_name")]
        rec["keywords"] = "; ".join(kws)
        rec["cited_by"] = str(w.get("cited_by_count") or "")
        rec["openalex_id"] = w.get("id") or ""
        rec["ext_references"] = w.get("referenced_works") or []
        return rec

    def fetch_by_doi(self, doi: str) -> Optional[Dict[str, str]]:
        data = _get_json(f"{self.BASE}/doi:{doi}", self._params({}), self.log)
        return self._to_record(data) if data else None

    def fetch_by_title(self, title: str) -> Optional[Dict[str, str]]:
        data = _get_json(self.BASE, self._params({"filter": f"title.search:{title}", "per-page": 1}), self.log)
        results = (data or {}).get("results") or []
        return self._to_record(results[0]) if results else None

    def search_keywords(self, query: str, limit: int) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        per_page = min(200, limit)
        page = 1
        while len(out) < limit:
            data = _get_json(self.BASE, self._params({"search": query, "per-page": per_page, "page": page}), self.log)
            results = (data or {}).get("results") or []
            if not results:
                break
            out.extend(self._to_record(w) for w in results)
            if len(results) < per_page:
                break
            page += 1
        return out[:limit]


# ---------------- Crossref ----------------

class CrossrefProvider:
    name = "crossref"
    BASE = "https://api.crossref.org/works"

    def __init__(self, mailto: str, log: Callable[[str], None]):
        self.mailto = mailto
        self.log = log

    def _params(self, extra: dict) -> dict:
        p = dict(extra)
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    def _to_record(self, m: dict) -> Dict[str, str]:
        rec = _empty()
        rec["title"] = (m.get("title") or [""])[0]
        abstract = m.get("abstract") or ""
        rec["abstract"] = re.sub(r"<[^>]+>", " ", abstract).strip()
        authors, affils = [], []
        for a in m.get("author", []) or []:
            nm = " ".join(p for p in (a.get("given"), a.get("family")) if p)
            if nm:
                authors.append(nm)
            for aff in a.get("affiliation", []) or []:
                if aff.get("name"):
                    affils.append(aff["name"])
        rec["authors"] = "; ".join(authors)
        rec["affiliations"] = "; ".join(dict.fromkeys(affils))
        issued = (m.get("issued") or {}).get("date-parts") or [[""]]
        rec["year"] = str(issued[0][0] or "")
        rec["journal"] = (m.get("container-title") or [""])[0]
        rec["volume"] = m.get("volume") or ""
        rec["issue"] = m.get("issue") or ""
        rec["pages"] = m.get("page") or ""
        rec["doi"] = (m.get("DOI") or "").lower()
        rec["keywords"] = "; ".join(m.get("subject") or [])
        rec["cited_by"] = str(m.get("is-referenced-by-count") or "")
        return rec

    def fetch_by_doi(self, doi: str) -> Optional[Dict[str, str]]:
        data = _get_json(f"{self.BASE}/{doi}", self._params({}), self.log)
        msg = (data or {}).get("message")
        return self._to_record(msg) if msg else None

    def fetch_by_title(self, title: str) -> Optional[Dict[str, str]]:
        data = _get_json(self.BASE, self._params({"query.bibliographic": title, "rows": 1}), self.log)
        items = ((data or {}).get("message") or {}).get("items") or []
        return self._to_record(items[0]) if items else None

    def search_keywords(self, query: str, limit: int) -> List[Dict[str, str]]:
        data = _get_json(self.BASE, self._params({"query": query, "rows": min(limit, 1000)}), self.log)
        items = ((data or {}).get("message") or {}).get("items") or []
        return [self._to_record(m) for m in items[:limit]]


# ---------------- Scopus (key-gated; untested without a key) ----------------

class ScopusProvider:
    name = "scopus"
    SEARCH = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: str, log: Callable[[str], None]):
        self.api_key = api_key
        self.log = log

    def _headers(self) -> dict:
        return {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}

    def _search(self, query: str, count: int) -> List[dict]:
        try:
            resp = requests.get(self.SEARCH, params={"query": query, "count": min(count, 25)},
                                headers=self._headers(), timeout=_TIMEOUT)
            if resp.status_code != 200:
                self.log(f"  providers(scopus): HTTP {resp.status_code}.")
                return []
            return (resp.json().get("search-results") or {}).get("entry") or []
        except Exception as e:
            self.log(f"  providers(scopus): request failed ({e}).")
            return []

    def _to_record(self, e: dict) -> Dict[str, str]:
        rec = _empty()
        rec["title"] = e.get("dc:title") or ""
        rec["abstract"] = e.get("dc:description") or ""
        rec["authors"] = e.get("dc:creator") or ""
        rec["year"] = (e.get("prism:coverDate") or "")[:4]
        rec["journal"] = e.get("prism:publicationName") or ""
        rec["volume"] = e.get("prism:volume") or ""
        rec["issue"] = e.get("prism:issueIdentifier") or ""
        rec["pages"] = e.get("prism:pageRange") or ""
        rec["doi"] = (e.get("prism:doi") or "").lower()
        rec["cited_by"] = str(e.get("citedby-count") or "")
        affils = [a.get("affilname") for a in (e.get("affiliation") or []) if a.get("affilname")]
        countries = [a.get("affiliation-country") for a in (e.get("affiliation") or []) if a.get("affiliation-country")]
        rec["affiliations"] = "; ".join(dict.fromkeys(affils))
        rec["country"] = "; ".join(dict.fromkeys(countries))
        return rec

    def fetch_by_doi(self, doi: str) -> Optional[Dict[str, str]]:
        entries = self._search(f"DOI({doi})", 1)
        return self._to_record(entries[0]) if entries else None

    def fetch_by_title(self, title: str) -> Optional[Dict[str, str]]:
        entries = self._search(f"TITLE({title})", 1)
        return self._to_record(entries[0]) if entries else None

    def search_keywords(self, query: str, limit: int) -> List[Dict[str, str]]:
        return [self._to_record(e) for e in self._search(query, limit)]


def make_provider(name: str, cfg, log: Callable[[str], None]):
    """Construct a provider by name, honouring the Scopus key gate + fallback."""
    name = (name or "openalex").lower()
    mailto = getattr(cfg, "provider_mailto", "")
    if name == "crossref":
        return CrossrefProvider(mailto, log)
    if name == "scopus":
        key = os.environ.get("SCOPUS_API_KEY")
        if not key:
            fb = getattr(cfg, "provider_fallback", "openalex")
            log(f"  providers: Scopus selected but SCOPUS_API_KEY is not set — falling back to '{fb}'.")
            return make_provider(fb, cfg, log) if fb and fb != "scopus" else None
        return ScopusProvider(key, log)
    return OpenAlexProvider(mailto, log)
