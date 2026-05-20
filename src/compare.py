"""Cross-run comparison report.

Reads `terms_lemmatised.csv` and the `topics_*.txt` files from each output
directory, then writes a single markdown report that highlights overlap and
divergence in top terms / topic words across runs.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import pandas as pd


def discover_dirs(base: str = "output") -> List[str]:
    return sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))


def short_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or path


def load_top_terms(directory: str, top: int) -> List[str]:
    csv_path = os.path.join(directory, "terms_lemmatised.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path).head(top)
    return df["Term"].astype(str).str.lower().tolist()


def load_topic_files(directory: str) -> Dict[str, str]:
    """Map topic-model label -> file contents for the per-topic word files."""
    out: Dict[str, str] = {}
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("topics_") and filename.endswith(".txt"):
            label = filename.removeprefix("topics_").removesuffix(".txt")
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                out[label] = f.read()
    return out


def compute_overlap(term_lists: Dict[str, List[str]]) -> List[Tuple[str, str, int, float]]:
    """Return pairwise overlap stats: (run_a, run_b, common_count, jaccard)."""
    names = list(term_lists)
    rows = []
    for i, a in enumerate(names):
        sa = set(term_lists[a])
        for b in names[i + 1:]:
            sb = set(term_lists[b])
            common = sa & sb
            union = sa | sb
            jaccard = len(common) / len(union) if union else 0.0
            rows.append((a, b, len(common), jaccard))
    return rows


def write_report(out_path: str, dirs: List[str], top: int) -> None:
    short = {d: short_name(d) for d in dirs}
    term_lists = {short[d]: load_top_terms(d, top) for d in dirs}
    overlap = compute_overlap(term_lists)

    lines: List[str] = []
    lines.append("# Geordie Miner — comparison report")
    lines.append("")
    lines.append(f"Comparing **{len(dirs)} run(s)** at top **{top}** terms.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for d in dirs:
        lines.append(f"- `{short[d]}` → `{d}`")
    lines.append("")

    lines.append(f"## Top {top} lemmatised terms")
    lines.append("")
    max_len = max((len(v) for v in term_lists.values()), default=0)
    header = "| rank | " + " | ".join(short[d] for d in dirs) + " |"
    sep = "|------|" + "|".join(["------"] * len(dirs)) + "|"
    lines.append(header)
    lines.append(sep)
    for i in range(max_len):
        row = [f"| {i + 1}"]
        for d in dirs:
            terms = term_lists[short[d]]
            row.append(terms[i] if i < len(terms) else "")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    if len(dirs) >= 2:
        lines.append("## Pairwise overlap of top terms")
        lines.append("")
        lines.append("| run A | run B | common terms | Jaccard |")
        lines.append("|-------|-------|--------------|---------|")
        for a, b, count, jacc in overlap:
            lines.append(f"| {a} | {b} | {count} / {top} | {jacc:.3f} |")
        lines.append("")

        lines.append("## Terms unique to each run")
        lines.append("")
        all_sets = {name: set(terms) for name, terms in term_lists.items()}
        for name, terms in all_sets.items():
            others = set().union(*(s for n, s in all_sets.items() if n != name))
            unique = sorted(terms - others)
            lines.append(f"### `{name}` ({len(unique)} unique)")
            lines.append("")
            lines.append(", ".join(unique) if unique else "_(none)_")
            lines.append("")

    lines.append("## Topic-model outputs per run")
    lines.append("")
    for d in dirs:
        lines.append(f"### `{short[d]}`")
        lines.append("")
        topics = load_topic_files(d)
        if not topics:
            lines.append("_(no topic-model output found)_")
            lines.append("")
            continue
        for label, body in topics.items():
            lines.append(f"#### {label}")
            lines.append("")
            lines.append("```")
            lines.append(body.strip())
            lines.append("```")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also write the interactive HTML version alongside the markdown.
    try:
        from html_report import write_comparison_html
        html_path = out_path[:-3] + ".html" if out_path.endswith(".md") else out_path + ".html"
        write_comparison_html(html_path, dirs, top)
    except Exception:
        pass
