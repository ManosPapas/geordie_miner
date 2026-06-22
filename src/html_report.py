"""Self-contained HTML report generator.

Produces `output/<name>/summary.html` (single run) and
`output/comparison_report.html` (cross-run). Both files are self-contained:
images are base64-embedded, JS is inline, no external CSS/JS dependencies.

Features:
- Sortable tables (click any column header)
- Collapsible sections via native <details>/<summary>
- Best-K row highlighted in the coherence table
- Same data as the .md versions; different rendering
"""

from __future__ import annotations

import base64
import html as _html
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import networkx as nx
import pandas as pd

from config import Config
from corpus_io import top_terms, topic_files
from topicutil import model_label_from_filename, parse_topics_file, topic_label


# ---------- visual style ----------

CSS = """
:root {
  --fg: #222;
  --muted: #666;
  --accent: #2563eb;
  --bg-alt: #f5f5f7;
  --border: #e3e3e3;
  --highlight: #fff7c2;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1200px;
  margin: 2rem auto;
  padding: 0 1rem 4rem;
  color: var(--fg);
  line-height: 1.55;
}
h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
h2 { margin-top: 2.5rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }
h3 { margin-top: 1.5rem; color: var(--muted); }
h4 { margin-top: 1rem; color: var(--muted); font-weight: 600; }
p.meta { color: var(--muted); font-size: 0.9rem; margin: 0.2rem 0; }
table { border-collapse: collapse; margin: 1rem 0; width: auto; max-width: 100%; }
th, td { padding: 0.4rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--bg-alt); font-weight: 600; cursor: pointer; user-select: none; white-space: nowrap; }
th.sortable:hover { background: #ececef; }
th.asc::after { content: " ↑"; color: var(--accent); }
th.desc::after { content: " ↓"; color: var(--accent); }
tr.highlight { background: var(--highlight); }
tr.highlight td:first-child::before { content: "★ "; color: #b58900; }
details { margin: 0.5rem 0; }
summary { cursor: pointer; padding: 0.4rem 0.7rem; background: var(--bg-alt); border-radius: 4px; font-weight: 500; }
summary:hover { background: #ececef; }
img { max-width: 100%; height: auto; margin: 1rem 0; border: 1px solid var(--border); border-radius: 4px; }
pre { background: var(--bg-alt); padding: 0.8rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }
code { background: var(--bg-alt); padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.9em; }
.callout { background: var(--bg-alt); border-left: 3px solid var(--accent); padding: 0.6rem 1rem; margin: 1rem 0; border-radius: 0 4px 4px 0; }
.muted { color: var(--muted); font-size: 0.9rem; }
"""

JS_SORTABLE = """
document.querySelectorAll('table.sortable th').forEach(th => {
  th.classList.add('sortable');
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const index = Array.from(th.parentNode.children).indexOf(th);
    const wasAsc = th.classList.contains('asc');
    rows.sort((a, b) => {
      const aText = (a.children[index]?.textContent || '').trim();
      const bText = (b.children[index]?.textContent || '').trim();
      const aNum = parseFloat(aText.replace(/[^0-9.\\-]/g, ''));
      const bNum = parseFloat(bText.replace(/[^0-9.\\-]/g, ''));
      const bothNum = !isNaN(aNum) && !isNaN(bNum) && aText !== '' && bText !== '';
      if (bothNum) return wasAsc ? aNum - bNum : bNum - aNum;
      return wasAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
    });
    rows.forEach(row => tbody.appendChild(row));
    table.querySelectorAll('th').forEach(h => h.classList.remove('asc', 'desc'));
    th.classList.toggle('desc', !wasAsc);
    th.classList.toggle('asc', wasAsc);
  });
});
"""


# ---------- helpers ----------

def _esc(value) -> str:
    """HTML-escape any value."""
    return _html.escape(str(value), quote=True)


def _embed_image(path: Path) -> str:
    """Return a base64 data URI for the given image path, or empty string."""
    if not path.exists():
        return ""
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _table(headers: List[str], rows: List[List], *, sortable: bool = True, highlight_idx: int | None = None) -> str:
    """Build an HTML table. `highlight_idx` (0-indexed body row) gets the highlight class."""
    cls = "sortable" if sortable else ""
    out = [f'<table class="{cls}">']
    out.append("<thead><tr>")
    for h in headers:
        out.append(f"<th>{_esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        row_cls = ' class="highlight"' if i == highlight_idx else ""
        out.append(f"<tr{row_cls}>")
        for cell in row:
            out.append(f"<td>{_esc(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _doc_template(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
{body}
<script>{JS_SORTABLE}</script>
</body>
</html>
"""


# ---------- single-run sections ----------

def _section_header(cfg: Config) -> str:
    name = os.path.basename(os.path.normpath(cfg.directory_data))
    return (
        f"<h1>Geordie Miner — {_esc(name)}</h1>\n"
        f"<p class='meta'><strong>Data:</strong> <code>{_esc(cfg.directory_data)}</code></p>\n"
        f"<p class='meta'><strong>Config:</strong> <code>{_esc(cfg.config_path)}</code></p>\n"
        f"<p class='meta'><strong>Output:</strong> <code>{_esc(cfg.directory_analysis)}</code></p>"
    )


def _section_corpus_stats(cfg: Config) -> str:
    path = Path(cfg.output_path("corpus_stats.txt"))
    if not path.exists():
        return ""
    return f"<h2>Corpus statistics</h2>\n<pre>{_esc(path.read_text(encoding='utf-8').strip())}</pre>"


def _section_top_terms(cfg: Config, top: int = 50) -> str:
    path = Path(cfg.output_path("terms_lemmatised.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path).head(top)
    rows = [
        [i + 1, row["Term"], int(row["Count"]), f"{row['% of Docs']:.1f}", f"{row['TF-IDF']:.2f}"]
        for i, row in df.iterrows()
    ]
    return (
        f"<h2>Top {top} lemmatised terms</h2>\n"
        f"<p class='muted'>Click any column header to sort.</p>\n"
        + _table(["Rank", "Term", "Count", "% of Docs", "TF-IDF"], rows)
    )


def _section_top_phrases(cfg: Config, top: int = 50) -> str:
    parts = ["<h2>Top phrases</h2>"]
    for label, fname in (("Bigrams", "bigrams.csv"), ("Trigrams", "trigrams.csv")):
        path = Path(cfg.output_path(fname))
        if not path.exists():
            continue
        df = pd.read_csv(path).head(top)
        if df.empty:
            continue
        rows = [[i + 1, row["ngram"], int(row["frequency"])] for i, row in df.iterrows()]
        parts.append(
            f"<details open><summary>{label} (top {len(df)})</summary>\n"
            + _table(["Rank", "n-gram", "Frequency"], rows)
            + "\n</details>"
        )
    return "\n".join(parts) if len(parts) > 1 else ""


def _section_word_clouds(cfg: Config) -> str:
    images = [
        ("Lemmatised", "wordcloud_lemmatised.jpg"),
        ("Raw", "wordcloud_raw.jpg"),
    ]
    available = [(label, Path(cfg.output_path(fname))) for label, fname in images]
    available = [(l, p) for l, p in available if p.exists()]
    if not available:
        return ""
    parts = ["<h2>Word clouds</h2>"]
    for label, path in available:
        data_uri = _embed_image(path)
        parts.append(f"<h4>{_esc(label)}</h4>\n<img src='{data_uri}' alt='{_esc(label)} word cloud'>")
    return "\n".join(parts)


def _section_topic_models(cfg: Config) -> str:
    topic_files = sorted(p for p in Path(cfg.directory_analysis).glob("topics_*.txt"))
    if not topic_files:
        return ""
    parts = [
        "<h2>Topics &amp; clusters</h2>",
        "<p class='muted'>Every method that produces topics/clusters (LDA, KMeans, NMF, HDP, "
        "BERTopic) is listed below. Each topic shows a <strong>suggested label</strong> derived "
        "from its highest-weighted terms — indicative, not a definitive classification — followed "
        "by its terms with weights/scores. Click a model to expand.</p>",
    ]
    for path in topic_files:
        topics = parse_topics_file(str(path))
        if not topics:
            continue
        inner: List[str] = []
        for header, n_docs, terms in topics:
            suggested = topic_label(terms)
            rows = [[term, f"{weight:.4f}"] for term, weight in terms]
            inner.append(
                f"<h4>{_esc(header)} <span class='muted'>({n_docs} docs)</span> — "
                f"<em>{_esc(suggested)}</em></h4>"
            )
            inner.append(_table(["Term", "Weight / score"], rows, sortable=False))
        parts.append(
            f"<details><summary>{_esc(model_label_from_filename(path.name))}</summary>\n"
            + "\n".join(inner)
            + "\n</details>"
        )
    return "\n".join(parts)


def _section_coherence(cfg: Config) -> str:
    path = Path(cfg.output_path("coherence_scores.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    df_sorted = df.sort_values("coherence_c_v", ascending=False).reset_index(drop=True)
    rows = [
        [row["model"], int(row["K"]), f"{row['coherence_c_v']:.3f}", f"{row['coherence_u_mass']:.3f}"]
        for _, row in df_sorted.iterrows()
    ]
    best_label = f"{df_sorted.iloc[0]['model']} at K={int(df_sorted.iloc[0]['K'])}"
    return (
        "<h2>Topic coherence</h2>\n"
        "<div class='callout'>Higher <code>c_v</code> = topics are more semantically coherent (typical range 0.3–0.8). "
        "<code>u_mass</code> closer to zero is better. <strong>Best model on this corpus:</strong> "
        f"{_esc(best_label)} (c_v = {df_sorted.iloc[0]['coherence_c_v']:.3f}).</div>\n"
        "<p class='muted'><strong>How to read:</strong> compare c_v <em>within</em> a model to choose K — "
        "pick the K at (or just before) a plateau rather than the single highest value, ignore small gaps, "
        "and confirm the winner holds up in the stability section below.</p>\n"
        + _table(["Model", "K", "c_v", "u_mass"], rows, highlight_idx=0)
    )


def _links_list(cfg: Config, items: List[Tuple[str, str]]) -> str:
    """Build a <ul> of relative links to artefacts that exist, else ''."""
    present = [(label, fn) for label, fn in items if Path(cfg.output_path(fn)).exists()]
    if not present:
        return ""
    lis = "\n".join(f"<li><a href='{_esc(fn)}'>{_esc(label)}</a> &nbsp;<code>{_esc(fn)}</code></li>" for label, fn in present)
    return f"<ul>\n{lis}\n</ul>"


def _section_bibliometrics(cfg: Config) -> str:
    path = Path(cfg.output_path("bibliometrics_summary.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    parts = [
        "<h2>Bibliometrics &amp; science mapping</h2>",
        "<p class='muted'>Best-effort from PDF heuristics / imported files / external providers. Unreliable "
        "fields are blank in <code>metadata.csv</code> (see <code>notes</code> and <code>metadata_provenance.csv</code>), "
        "never inferred.</p>",
    ]
    for cat, header in (("year", "Publication years"), ("journal", "Top sources / journals"), ("author", "Top authors")):
        sub = df[df["category"] == cat].head(15)
        if sub.empty:
            continue
        rows = [[r["key"], int(r["count"])] for _, r in sub.iterrows()]
        parts.append(f"<details open><summary>{_esc(header)}</summary>\n" + _table(["Value", "Count"], rows) + "\n</details>")
    for label, fname in (
        ("Publications per year", "bib_publication_trends.png"),
        ("Top sources / journals", "bib_journals.png"),
        ("Top authors", "bib_authors.png"),
        ("Top institutions", "bib_top_institutions.png"),
        ("Top countries", "bib_top_countries.png"),
        ("Journal map (term similarity)", "journal_map.png"),
        ("Thematic correspondence (start vs end)", "thematic_evolution_matrix.png"),
    ):
        uri = _embed_image(Path(cfg.output_path(fname)))
        if uri:
            parts.append(f"<h4>{_esc(label)}</h4>\n<img src='{uri}' alt='{_esc(label)}'>")
    links = _links_list(cfg, [
        ("All bibliometric tables (Excel, multi-sheet)", "bibliometrics.xlsx"),
        ("Publication trends (CSV)", "bibliometric_publication_trends.csv"),
        ("Top journals (CSV)", "bibliometric_top_journals.csv"),
        ("Top authors (CSV)", "bibliometric_top_authors.csv"),
        ("Top institutions (CSV)", "bibliometric_top_institutions.csv"),
        ("Top countries (CSV)", "bibliometric_top_countries.csv"),
        ("Normalised author table", "authors.csv"),
        ("Citation impact per document", "citation_impact.csv"),
        ("Venue impact per journal", "journal_impact.csv"),
        ("Metadata provenance", "metadata_provenance.csv"),
        ("Collaboration summary", "collaboration_summary.txt"),
        ("Co-authorship network (authors, GEXF)", "collab_authors.gexf"),
        ("Co-citation network (documents, GEXF)", "cocitation_documents.gexf"),
        ("Bibliographic coupling (documents, GEXF)", "coupling_documents.gexf"),
        ("Interactive journal map", "journal_map.html"),
        ("Country choropleth", "bib_country_map.html"),
    ])
    if links:
        parts.append("<h3>Detailed tables, networks &amp; maps</h3>")
        parts.append("<p class='muted'>CSVs open in any spreadsheet; GEXF in Gephi/VOSviewer; HTML maps in a browser.</p>")
        parts.append(links)
    return "\n".join(parts)


def _section_lexical(cfg: Config) -> str:
    path = Path(cfg.output_path("concept_counts.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    parts = [
        "<h2>Full-text &amp; lexical analysis</h2>",
        "<p class='muted'>Dictionary-based concept counts from your lexicons (<code>config/lexicons/</code>).</p>",
        _table(["Concept", "Docs", "Occurrences"], [[r["concept"], int(r["n_docs"]), int(r["total_occurrences"])] for _, r in df.iterrows()]),
    ]
    for label, fname in (("Concept frequencies", "concept_frequencies.png"), ("Concept prevalence over time", "concept_trends.png")):
        uri = _embed_image(Path(cfg.output_path(fname)))
        if uri:
            parts.append(f"<h4>{_esc(label)}</h4>\n<img src='{uri}' alt='{_esc(label)}'>")
    links = _links_list(cfg, [
        ("Concept co-occurrence (CSV)", "concept_cooccurrence.csv"),
        ("Concept trends (CSV)", "concept_trends.csv"),
        ("Context samples (CSV)", "concept_contexts.csv"),
    ])
    if links:
        parts.append(links)
    return "\n".join(parts)


def _section_topic_evolution(cfg: Config) -> str:
    path = Path(cfg.output_path("topic_evolution.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    rows = [[r["bracket"], int(r["topic_id"]), int(r["n_docs"]), f"{float(r['share']):.2f}", r["top_terms"]] for _, r in df.iterrows()]
    parts = [
        "<h2>Topics &amp; evolution</h2>",
        "<p class='muted'>Topics modelled within each time bracket; transitions flag splits (one&rarr;many) "
        "and merges (many&rarr;one) between adjacent periods.</p>",
        "<details open><summary>Topic prevalence per bracket</summary>\n" + _table(["Bracket", "Topic", "Docs", "Share", "Top terms"], rows) + "\n</details>",
    ]
    tpath = Path(cfg.output_path("topic_transitions.csv"))
    if tpath.exists():
        t = pd.read_csv(tpath)
        parts.append(f"<div class='callout'>{int((t['type'] == 'split').sum())} split(s) and "
                     f"{int((t['type'] == 'merge').sum())} merge(s) detected across brackets.</div>")
    if Path(cfg.output_path("topic_evolution.html")).exists():
        parts.append("<p><strong>Interactive flow:</strong> <a href='topic_evolution.html'>topic_evolution.html</a> (Sankey of topic flows over time).</p>")
    parts.append(_links_list(cfg, [("Topic transitions (CSV)", "topic_transitions.csv")]))
    return "\n".join(parts)


def _section_stability(cfg: Config) -> str:
    path = Path(cfg.output_path("stability_report.json"))
    if not path.exists():
        return ""
    rep = json.loads(path.read_text(encoding="utf-8"))
    per_seed = rep.get("per_seed_coherence_c_v", {}) or {}
    rows = [[seed, "n/a" if value is None else value] for seed, value in per_seed.items()]
    judged = str(rep.get("judgement", "")).upper()
    return (
        "<h2>Topic-model stability</h2>\n"
        "<div class='callout'>Most-coherent model "
        f"<strong>{_esc(rep.get('model'))}</strong> at K={_esc(rep.get('K'))}, re-run across a fixed "
        f"seed schedule. Mean cross-seed topic Jaccard <strong>{_esc(rep.get('mean_cross_seed_jaccard'))}</strong>; "
        f"coherence c_v mean {_esc(rep.get('coherence_mean'))} (variance {_esc(rep.get('coherence_variance'))}). "
        f"Judgement: <strong>{_esc(judged)}</strong>.</div>\n"
        "<p class='muted'><strong>How to read:</strong> a <em>stable</em> solution (high cross-seed Jaccard, "
        "low coherence variance) means the topics are robust to random initialisation and safe to interpret; "
        "<em>moderately sensitive</em> warrants caution on individual topics; <em>unstable</em> suggests the "
        "themes are largely artefacts — reduce K or revisit preprocessing.</p>\n"
        f"<p class='muted'>Seeds: {_esc(rep.get('seeds'))}. Per-topic detail in <code>topic_stability.csv</code>.</p>\n"
        + _table(["Seed", "Coherence c_v"], rows, sortable=False)
    )


def _section_reproducibility(cfg: Config) -> str:
    if not Path(cfg.output_path("run_config.json")).exists():
        return ""
    has_yaml = Path(cfg.output_path("run_config.yaml")).exists()
    files = "run_config.json" + (" / run_config.yaml" if has_yaml else "")
    return (
        "<h2>Reproducibility</h2>\n"
        f"<p>This run's full configuration is exported to <code>{files}</code> — input sources, "
        "preprocessing options, model parameters, selected algorithms, random seeds, library "
        "versions, the application version, data sources and the execution environment. "
        "A readable summary is in <a href='reproducibility.md'><code>reproducibility.md</code></a>.</p>\n"
        "<p class='muted'>Documents the environment rather than guaranteeing cross-machine identity "
        "of results; exact reproducibility can vary with hardware and library versions.</p>"
    )


def _section_figures(cfg: Config) -> str:
    figures = [
        ("Hierarchical clustering dendrogram", "dendrogram.png"),
    ]
    available = [(label, Path(cfg.output_path(fname))) for label, fname in figures]
    available = [(l, p) for l, p in available if p.exists()]
    if not available:
        return ""
    parts = ["<h2>Figures</h2>"]
    for label, path in available:
        data_uri = _embed_image(path)
        parts.append(f"<h4>{_esc(label)}</h4>\n<img src='{data_uri}' alt='{_esc(label)}'>")
    return "\n".join(parts)


def _section_network(cfg: Config) -> str:
    path = Path(cfg.output_path("network.gexf"))
    if not path.exists():
        return ""
    g = nx.read_gexf(str(path))
    viewer = ""
    if Path(cfg.output_path("network.html")).exists():
        viewer = (
            "<p><strong>Interactive viewer:</strong> <a href='network.html'>network.html</a> "
            "— renders the network in your browser, and falls back to the GEXF download if the "
            "viewer libraries can't load.</p>"
        )
    return (
        "<h2>Co-occurrence network</h2>\n"
        f"<p>Nodes: <strong>{g.number_of_nodes()}</strong>, edges: <strong>{g.number_of_edges()}</strong>.</p>\n"
        f"{viewer}"
        "<p class='muted'>Open <code>network.gexf</code> in <a href='https://gephi.org/'>Gephi</a>, or import "
        "<code>network_vosviewer_map.txt</code> + <code>network_vosviewer_network.txt</code> into "
        "<a href='https://www.vosviewer.com/'>VOSviewer</a>.</p>"
    )


# ---------- public entry points ----------

def _safe_section(fn, cfg: Config, log: Callable[[str], None]) -> str:
    """Build one report section; return '' (and log) on error so a single bad
    section can't blank the whole report. One place for all section error handling."""
    try:
        return fn(cfg)
    except Exception as e:
        log(f"  summary: section '{fn.__name__}' skipped ({e}).")
        return ""


def write_summary_html(cfg: Config, log: Callable[[str], None]) -> None:
    """Write the interactive single-run report to output/<name>/summary.html."""
    section_fns = [
        _section_header, _section_corpus_stats, _section_bibliometrics, _section_top_terms,
        _section_top_phrases, _section_word_clouds, _section_lexical, _section_topic_models,
        _section_coherence, _section_stability, _section_topic_evolution, _section_figures,
        _section_network, _section_reproducibility,
    ]
    body = "\n\n".join(s for s in (_safe_section(fn, cfg, log) for fn in section_fns) if s)
    name = os.path.basename(os.path.normpath(cfg.directory_data))
    doc = _doc_template(f"Geordie Miner — {name}", body)
    out_path = cfg.output_path("summary.html")
    Path(out_path).write_text(doc, encoding="utf-8")
    log(f"HTML summary written: {out_path}")


# ---------- cross-run comparison ----------

def write_comparison_html(out_path: str, dirs: List[str], top: int) -> None:
    """Write output/comparison_report.html with side-by-side runs."""
    dirs_p = [Path(d) for d in dirs]
    short = {d: d.name for d in dirs_p}
    term_lists = {short[d]: top_terms(str(d), top) for d in dirs_p}

    parts: List[str] = []
    parts.append(f"<h1>Geordie Miner — comparison report</h1>")
    parts.append(f"<p class='meta'>Comparing <strong>{len(dirs_p)} run(s)</strong> at top <strong>{top}</strong> terms.</p>")

    # Runs list
    parts.append("<h2>Runs</h2>\n<ul>")
    for d in dirs_p:
        parts.append(f"<li><code>{_esc(short[d])}</code> → <code>{_esc(str(d))}</code></li>")
    parts.append("</ul>")

    # Side-by-side terms table
    parts.append(f"<h2>Top {top} lemmatised terms — side by side</h2>")
    max_len = max((len(v) for v in term_lists.values()), default=0)
    headers = ["Rank"] + [short[d] for d in dirs_p]
    rows: List[List] = []
    for i in range(max_len):
        row = [i + 1]
        for d in dirs_p:
            terms = term_lists[short[d]]
            row.append(terms[i] if i < len(terms) else "")
        rows.append(row)
    parts.append(_table(headers, rows))

    # Pairwise overlap
    if len(dirs_p) >= 2:
        parts.append("<h2>Pairwise Jaccard overlap</h2>")
        parts.append("<p class='muted'>Higher Jaccard = the two runs agree more on which top terms appear. Rule of thumb: ≥ 0.7 means themes are robust.</p>")
        overlap_rows: List[List] = []
        names = list(term_lists)
        for i, a in enumerate(names):
            sa = set(term_lists[a])
            for b in names[i + 1:]:
                sb = set(term_lists[b])
                common = sa & sb
                union = sa | sb
                jaccard = len(common) / len(union) if union else 0.0
                overlap_rows.append([a, b, f"{len(common)} / {top}", f"{jaccard:.3f}"])
        parts.append(_table(["Run A", "Run B", "Common terms", "Jaccard"], overlap_rows))

        # Terms unique to each run
        parts.append("<h2>Terms unique to each run</h2>")
        all_sets = {name: set(terms) for name, terms in term_lists.items()}
        for name, terms in all_sets.items():
            others = set().union(*(s for n, s in all_sets.items() if n != name))
            unique = sorted(terms - others)
            inner = ", ".join(_esc(t) for t in unique) if unique else "<em>(none)</em>"
            parts.append(
                f"<details><summary>{_esc(name)} — {len(unique)} unique term(s)</summary>\n"
                f"<p>{inner}</p>\n</details>"
            )

    # Topic models per run
    parts.append("<h2>Topic-model outputs per run</h2>")
    for d in dirs_p:
        topics = topic_files(str(d))
        parts.append(f"<details><summary><code>{_esc(short[d])}</code></summary>")
        if not topics:
            parts.append("<p class='muted'><em>(no topic-model output found)</em></p>")
        else:
            for label, body in topics.items():
                parts.append(
                    f"<details><summary>{_esc(label)}</summary>\n<pre>{_esc(body.strip())}</pre>\n</details>"
                )
        parts.append("</details>")

    body = "\n".join(parts)
    doc = _doc_template("Geordie Miner — comparison report", body)
    Path(out_path).write_text(doc, encoding="utf-8")
