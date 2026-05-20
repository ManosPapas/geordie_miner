"""Per-run summary report (summary.md).

Reads everything in `output/<name>/` and produces a single readable markdown
file. Saves the researcher from clicking through 30 individual outputs.
"""

from __future__ import annotations

import csv
import os
from typing import Callable, List

import pandas as pd

from config import Config


def write_summary(cfg: Config, log: Callable[[str], None]) -> None:
    """Build summary.md from whatever artefacts exist in the output directory."""
    name = os.path.basename(os.path.normpath(cfg.directory_data))
    lines: List[str] = []
    lines.append(f"# Geordie Miner — `{name}` summary")
    lines.append("")
    lines.append(f"_Data:_ `{cfg.directory_data}`  ")
    lines.append(f"_Config:_ `{cfg.config_path}`  ")
    lines.append(f"_Output:_ `{cfg.directory_analysis}`  ")
    lines.append("")

    _add_corpus_stats(cfg, lines)
    _add_top_terms(cfg, lines)
    _add_top_phrases(cfg, lines)
    _add_network_summary(cfg, lines)
    _add_topic_models(cfg, lines)
    _add_coherence(cfg, lines)
    _add_figures(cfg, lines)

    out_path = cfg.output_path("summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Summary written: {out_path}")


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


def _add_top_terms(cfg: Config, lines: List[str], top_n: int = 25) -> None:
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


def _add_top_phrases(cfg: Config, lines: List[str], top_n: int = 15) -> None:
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
    try:
        import networkx as nx
        g = nx.read_gexf(path)
        lines.append("## Co-occurrence network")
        lines.append("")
        lines.append(f"- Nodes: **{g.number_of_nodes()}**")
        lines.append(f"- Edges: **{g.number_of_edges()}**")
        lines.append(f"- File: `network.gexf` — open in [Gephi](https://gephi.org/) to colour by modularity.")
        lines.append("")
    except Exception:
        return


def _add_topic_models(cfg: Config, lines: List[str]) -> None:
    topic_files = sorted(
        f for f in os.listdir(cfg.directory_analysis)
        if f.startswith("topics_") and f.endswith(".txt")
    )
    if not topic_files:
        return
    lines.append("## Topic models")
    lines.append("")
    for fname in topic_files:
        with open(cfg.output_path(fname), "r", encoding="utf-8") as f:
            body = f.read().strip()
        if not body:
            continue
        label = fname.removeprefix("topics_").removesuffix(".txt").upper().replace("_", " K=")
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```")
        lines.append(body)
        lines.append("```")
        lines.append("")


def _add_coherence(cfg: Config, lines: List[str]) -> None:
    path = cfg.output_path("coherence_scores.csv")
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
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
