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
import csv as _csv
import html as _html
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from config import Config


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


def _section_top_terms(cfg: Config, top: int = 30) -> str:
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


def _section_top_phrases(cfg: Config, top: int = 25) -> str:
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
    parts = ["<h2>Topic models</h2>",
             "<p class='muted'>Each model is collapsed by default — click to expand.</p>"]
    for path in topic_files:
        label = path.stem.replace("topics_", "").upper().replace("_", " K=")
        body = path.read_text(encoding="utf-8").strip()
        parts.append(
            f"<details><summary>{_esc(label)}</summary>\n<pre>{_esc(body)}</pre>\n</details>"
        )
    return "\n".join(parts)


def _section_coherence(cfg: Config) -> str:
    path = Path(cfg.output_path("coherence_scores.csv"))
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
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
        + _table(["Model", "K", "c_v", "u_mass"], rows, highlight_idx=0)
    )


def _section_top_docs(cfg: Config) -> str:
    path = Path(cfg.output_path("topic_top_docs.csv"))
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    parts = [
        "<h2>Top documents per topic</h2>",
        "<p class='muted'>The most representative papers for each topic in each model. Use these to read the actual papers behind a theme.</p>",
    ]
    for (model, k), group in df.groupby(["model", "K"], sort=False):
        rows = [
            [int(r["topic"]), int(r["rank"]), r["doc_id"], f"{r['score']:.4f}"]
            for _, r in group.iterrows()
        ]
        parts.append(
            f"<details><summary>{_esc(model)} (K={int(k)})</summary>\n"
            + _table(["Topic", "Rank", "Doc ID", "Score"], rows)
            + "\n</details>"
        )
    return "\n".join(parts)


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
    try:
        import networkx as nx
        g = nx.read_gexf(str(path))
        return (
            "<h2>Co-occurrence network</h2>\n"
            f"<p>Nodes: <strong>{g.number_of_nodes()}</strong>, edges: <strong>{g.number_of_edges()}</strong>.</p>\n"
            f"<p class='muted'>Open <code>network.gexf</code> in <a href='https://gephi.org/'>Gephi</a> for interactive visual exploration.</p>"
        )
    except Exception:
        return ""


# ---------- public entry points ----------

def write_summary_html(cfg: Config, log: Callable[[str], None]) -> None:
    """Write the interactive single-run report to output/<name>/summary.html."""
    sections = [
        _section_header(cfg),
        _section_corpus_stats(cfg),
        _section_top_terms(cfg),
        _section_top_phrases(cfg),
        _section_word_clouds(cfg),
        _section_topic_models(cfg),
        _section_coherence(cfg),
        _section_top_docs(cfg),
        _section_figures(cfg),
        _section_network(cfg),
    ]
    body = "\n\n".join(s for s in sections if s)
    name = os.path.basename(os.path.normpath(cfg.directory_data))
    doc = _doc_template(f"Geordie Miner — {name}", body)
    out_path = cfg.output_path("summary.html")
    Path(out_path).write_text(doc, encoding="utf-8")
    log(f"HTML summary written: {out_path}")


# ---------- cross-run comparison ----------

def _load_top_terms(directory: Path, top: int) -> List[str]:
    csv_path = directory / "terms_lemmatised.csv"
    if not csv_path.exists():
        return []
    return pd.read_csv(csv_path).head(top)["Term"].astype(str).str.lower().tolist()


def _load_topic_files(directory: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("topics_*.txt")):
        label = path.stem.replace("topics_", "")
        out[label] = path.read_text(encoding="utf-8")
    return out


def write_comparison_html(out_path: str, dirs: List[str], top: int) -> None:
    """Write output/comparison_report.html with side-by-side runs."""
    dirs_p = [Path(d) for d in dirs]
    short = {d: d.name for d in dirs_p}
    term_lists = {short[d]: _load_top_terms(d, top) for d in dirs_p}

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
        topics = _load_topic_files(d)
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
