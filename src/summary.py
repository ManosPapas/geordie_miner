"""Per-run summary report (summary.md).

Reads everything in `output/<name>/` and produces a single readable markdown
file. Saves the researcher from clicking through 30 individual outputs.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List

import networkx as nx
import pandas as pd

from config import Config
from html_report import write_summary_html
from network_viz import write_network_html
from topicutil import model_label_from_filename, parse_topics_file, topic_label


def write_summary(cfg: Config, log: Callable[[str], None]) -> None:
    """Build summary.md + summary.html from artefacts in the output directory."""
    name = os.path.basename(os.path.normpath(cfg.directory_data))
    lines: List[str] = []
    lines.append(f"# Geordie Miner — `{name}` summary")
    lines.append("")
    lines.append(f"_Data:_ `{cfg.directory_data}`  ")
    lines.append(f"_Config:_ `{cfg.config_path}`  ")
    lines.append(f"_Output:_ `{cfg.directory_analysis}`  ")
    lines.append("")

    # Build the standalone interactive network page (linked from both reports).
    # Guarded: this runs at the top of write_summary, which is itself un-wrapped,
    # so a viewer hiccup must not stop summary.md being written.
    gexf = cfg.output_path("network.gexf")
    if os.path.exists(gexf):
        try:
            write_network_html(gexf, cfg.output_path("network.html"), log)
        except Exception as e:
            log(f"Network viewer page failed (non-fatal): {e}")

    # One guard for all sections — a single bad section can't blank the report.
    for add in (_add_corpus_stats, _add_bibliometrics, _add_top_terms, _add_top_phrases,
                _add_lexical, _add_network_summary, _add_topic_models, _add_coherence,
                _add_stability, _add_topic_evolution, _add_reproducibility, _add_figures):
        try:
            add(cfg, lines)
        except Exception as e:
            log(f"  summary.md: section '{add.__name__}' skipped ({e}).")

    out_path = cfg.output_path("summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Summary written: {out_path}")

    # Also produce the interactive HTML version.
    try:
        write_summary_html(cfg, log)
    except Exception as e:
        log(f"HTML summary failed (non-fatal): {e}")


def _add_corpus_stats(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("corpus_stats.txt")
    if not os.path.exists(path):
        return
    lines.append("## Corpus statistics")
    lines.append("")
    lines.append("```")
    with open(path, "r", encoding="utf-8") as f:
        lines.append(f.read().strip())
    lines.append("```")
    lines.append("")


def _add_top_terms(cfg: Config, lines: List[str], top_n: int = 50) -> None:
    csv_path = cfg.output_path("terms_lemmatised.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path).head(top_n)
    lines.append(f"## Top {top_n} lemmatised terms")
    lines.append("")
    lines.append("| Rank | Term | Count | % of Docs | TF-IDF |")
    lines.append("|------|------|------:|----------:|-------:|")
    for i, row in df.iterrows():
        lines.append(
            f"| {i + 1} | {row['Term']} | {int(row['Count'])} | "
            f"{row['% of Docs']:.1f} | {row['TF-IDF']:.2f} |"
        )
    lines.append("")


def _add_top_phrases(cfg: Config, lines: List[str], top_n: int = 50) -> None:
    for label, fname in (("bigrams", "bigrams.csv"), ("trigrams", "trigrams.csv")):
        path = cfg.output_path(fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).head(top_n)
        if df.empty:
            continue
        lines.append(f"## Top {len(df)} {label}")
        lines.append("")
        lines.append("| Rank | n-gram | Frequency |")
        lines.append("|------|--------|----------:|")
        for i, row in df.iterrows():
            lines.append(f"| {i + 1} | {row['ngram']} | {int(row['frequency'])} |")
        lines.append("")


def _add_network_summary(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("network.gexf")
    if not os.path.exists(path):
        return
    g = nx.read_gexf(path)
    lines.append("## Co-occurrence network")
    lines.append("")
    lines.append(f"- Nodes: **{g.number_of_nodes()}**")
    lines.append(f"- Edges: **{g.number_of_edges()}**")
    lines.append("- File: `network.gexf` — open in [Gephi](https://gephi.org/) or [VOSviewer](https://www.vosviewer.com/).")
    lines.append("- VOSviewer: `network_vosviewer_map.txt` + `network_vosviewer_network.txt`.")
    if os.path.exists(cfg.output_path("network.html")):
        lines.append("- Interactive in-browser viewer: [`network.html`](network.html) (falls back to the GEXF download if the viewer can't load).")
    lines.append("")


def _add_topic_models(cfg: Config, lines: List[str]) -> None:
    topic_files = sorted(
        f for f in os.listdir(cfg.directory_analysis)
        if f.startswith("topics_") and f.endswith(".txt")
    )
    if not topic_files:
        return
    lines.append("## Topic models")
    lines.append("")
    lines.append(
        "_Each topic shows a **suggested label** derived from its highest-weighted "
        "terms (indicative, not a definitive classification) followed by the terms "
        "with their weights/scores._"
    )
    lines.append("")
    for fname in topic_files:
        topics = parse_topics_file(cfg.output_path(fname))
        if not topics:
            continue
        lines.append(f"### {model_label_from_filename(fname)}")
        lines.append("")
        for header, n_docs, terms in topics:
            label = topic_label(terms)
            term_str = ", ".join(f"{t} ({w:.4f})" for t, w in terms)
            lines.append(f"- **{header} ({n_docs} docs) — _{label}_**: {term_str}")
        lines.append("")


def _add_coherence(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("coherence_scores.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    lines.append("## Topic coherence")
    lines.append("")
    lines.append("Higher `c_v` is better (typical range 0.3–0.8). `u_mass` closer to zero is better.")
    lines.append("")
    lines.append("| Model | K | c_v | u_mass |")
    lines.append("|-------|--:|----:|-------:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['K'])} | "
            f"{row['coherence_c_v']:.3f} | {row['coherence_u_mass']:.3f} |"
        )
    lines.append("")


def _add_bibliometrics(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("bibliometrics_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    lines.append("## Bibliometrics & science mapping")
    lines.append("")
    lines.append(
        "_Best-effort from PDF heuristics / imported files / external providers. Unreliable "
        "fields are blank in `metadata.csv` (see `notes` + `metadata_provenance.csv`), not inferred._"
    )
    lines.append("")
    for cat, header in (("year", "Publication years"), ("journal", "Top sources / journals"), ("author", "Top authors")):
        sub = df[df["category"] == cat].head(15)
        if sub.empty:
            continue
        lines.append(f"### {header}")
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|-------|------:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['key']} | {int(r['count'])} |")
        lines.append("")
    for label, fname in (
        ("Publications per year", "bib_publication_trends.png"),
        ("Top sources / journals", "bib_journals.png"),
        ("Top authors", "bib_authors.png"),
        ("Top institutions", "bib_top_institutions.png"),
        ("Top countries", "bib_top_countries.png"),
        ("Journal map", "journal_map.png"),
        ("Thematic correspondence", "thematic_evolution_matrix.png"),
    ):
        if os.path.exists(cfg.output_path(fname)):
            lines.append(f"![{label}]({fname})")
            lines.append("")
    detail = [
        ("All tables (Excel)", "bibliometrics.xlsx"),
        ("Citation impact", "citation_impact.csv"),
        ("Collaboration summary", "collaboration_summary.txt"),
        ("Author table", "authors.csv"),
        ("Interactive journal map", "journal_map.html"),
    ]
    present = [(lbl, fn) for lbl, fn in detail if os.path.exists(cfg.output_path(fn))]
    if present:
        lines.append("**Detailed files:** " + " · ".join(f"[{lbl}]({fn})" for lbl, fn in present))
        lines.append("")


def _add_lexical(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("concept_counts.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    lines.append("## Full-text & lexical analysis")
    lines.append("")
    lines.append("_Dictionary concept counts from your lexicons (`config/lexicons/`)._")
    lines.append("")
    lines.append("| Concept | Docs | Occurrences |")
    lines.append("|---------|-----:|------------:|")
    for _, r in df.iterrows():
        lines.append(f"| {r['concept']} | {int(r['n_docs'])} | {int(r['total_occurrences'])} |")
    lines.append("")
    for label, fname in (("Concept frequencies", "concept_frequencies.png"), ("Concept prevalence over time", "concept_trends.png")):
        if os.path.exists(cfg.output_path(fname)):
            lines.append(f"![{label}]({fname})")
            lines.append("")
    if os.path.exists(cfg.output_path("concept_contexts.csv")):
        lines.append("_Context samples: [`concept_contexts.csv`](concept_contexts.csv)._")
        lines.append("")


def _add_topic_evolution(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("topic_evolution.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    lines.append("## Topics & evolution")
    lines.append("")
    tpath = cfg.output_path("topic_transitions.csv")
    if os.path.exists(tpath):
        t = pd.read_csv(tpath)
        lines.append(f"_{int((t['type'] == 'split').sum())} split(s), {int((t['type'] == 'merge').sum())} "
                     "merge(s) detected across time brackets._")
        lines.append("")
    lines.append("| Bracket | Topic | Docs | Share | Top terms |")
    lines.append("|---------|------:|-----:|------:|-----------|")
    for _, r in df.iterrows():
        lines.append(f"| {r['bracket']} | {int(r['topic_id'])} | {int(r['n_docs'])} | {float(r['share']):.2f} | {r['top_terms']} |")
    lines.append("")
    if os.path.exists(cfg.output_path("topic_evolution.html")):
        lines.append("_Interactive Sankey: [`topic_evolution.html`](topic_evolution.html)._")
        lines.append("")
    if os.path.exists(cfg.output_path("thematic_evolution_matrix.png")):
        lines.append("![Thematic correspondence](thematic_evolution_matrix.png)")
        lines.append("")


def _add_stability(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("stability_report.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    lines.append("## Topic-model stability")
    lines.append("")
    lines.append(f"- Primary model (most coherent): **{rep.get('model')}** at K={rep.get('K')}")
    lines.append(f"- Fixed seed schedule: `{rep.get('seeds')}`")
    lines.append(f"- Coherence `c_v` across reruns: mean **{rep.get('coherence_mean')}**, variance {rep.get('coherence_variance')} (std {rep.get('coherence_std')})")
    lines.append(f"- Mean cross-seed topic Jaccard: **{rep.get('mean_cross_seed_jaccard')}**")
    lines.append(f"- Judgement: **{str(rep.get('judgement', '')).upper()}**")
    lines.append("")
    lines.append("_Full detail in `stability_report.json` / `stability_report.txt`; per-topic stability in `topic_stability.csv`._")
    lines.append("")


def _add_reproducibility(cfg: Config, lines: List[str]) -> None:
    if not os.path.exists(cfg.output_path("run_config.json")):
        return
    has_yaml = os.path.exists(cfg.output_path("run_config.yaml"))
    lines.append("## Reproducibility")
    lines.append("")
    files = "`run_config.json`" + (" / `run_config.yaml`" if has_yaml else "")
    lines.append(
        f"This run's full configuration — input sources, preprocessing options, model "
        f"parameters, algorithms, random seeds, library versions, app version and the "
        f"execution environment — is exported to {files} so collaborators can rerun it."
    )
    lines.append("")
    lines.append(
        "_Documents the environment rather than guaranteeing cross-machine identity; "
        "exact results can vary with hardware and library versions._"
    )
    lines.append("")


def _add_figures(cfg: Config, lines: List[str]) -> None:
    figures = [
        ("Lemmatised word cloud", "wordcloud_lemmatised.jpg"),
        ("Raw word cloud", "wordcloud_raw.jpg"),
        ("Hierarchical clustering dendrogram", "dendrogram.png"),
    ]
    available = [(label, fname) for label, fname in figures if os.path.exists(cfg.output_path(fname))]
    if not available:
        return
    lines.append("## Figures")
    lines.append("")
    for label, fname in available:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"![{label}]({fname})")
        lines.append("")
