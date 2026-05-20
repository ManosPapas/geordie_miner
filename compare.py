"""Compare two or more Geordie Miner runs.

Reads `analysis_terms_single_lemmatised.csv` and the topic-model output files from each
output directory, then writes a single markdown report that highlights overlap and
divergence in top terms / topic words across runs.

Usage:
    python compare.py                                # auto-discovers ./output/*
    python compare.py output/a output/b output/c     # explicit list
    python compare.py --out my_report.md output/*
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare top terms + topic models across analysis runs.")
    p.add_argument("dirs", nargs="*", help="Analysis directories. If empty, auto-discovers ./output/*.")
    p.add_argument("--out", default="comparison_report.md", help="Markdown output path (default: comparison_report.md).")
    p.add_argument("--top", type=int, default=50, help="How many top terms to compare per run (default: 50).")
    return p.parse_args(argv)


def discover_dirs() -> List[str]:
    return sorted(d for d in glob.glob(os.path.join("output", "*")) if os.path.isdir(d))


def short_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or path


def load_top_terms(directory: str, top: int) -> List[str]:
    csv_path = os.path.join(directory, "analysis_terms_single_lemmatised.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path).head(top)
    return df["Term"].astype(str).str.lower().tolist()


def load_topic_files(directory: str) -> Dict[str, str]:
    """Map topic-model label -> file contents for the per-topic word files."""
    out: Dict[str, str] = {}
    for filename in sorted(os.listdir(directory)):
        # Match files like analysis_topicmodel_LDA_5.txt, analysis_topicmodel_NMF_10.txt, analysis_topicmodel_HDP.txt
        if filename.startswith("analysis_topicmodel_") and filename.endswith(".txt") and "doc2topic" not in filename and "cluster_centroids" not in filename:
            label = filename.replace("analysis_topicmodel_", "").removesuffix(".txt")
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


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dirs = args.dirs or discover_dirs()
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("No analysis directories provided or discovered.", file=sys.stderr)
        return 1
    write_report(args.out, dirs, args.top)
    print(f"Report written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
