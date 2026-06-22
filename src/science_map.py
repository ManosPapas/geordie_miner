"""Science mapping: a journal map + a thematic-correspondence matrix.

- Journal map: each journal is positioned by the similarity of its term
  distribution (TF-IDF over its concatenated documents, projected to 2D with
  TruncatedSVD), sized by document count. Interactive (plotly) + static (png).
- Thematic correspondence: a Jaccard heatmap between the first and last time
  bracket's topics (read from `topic_evolution.csv`), showing which early themes
  persist into recent ones.

Visual styling honours `[visuals]` config (density caps points, label_filtering
thins labels, colour_scheme sets the palette).
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Callable, Dict, List

from config import Config
from corpus_io import metadata_by_doc
from plotting import plt

_DENSITY_CAP = {"low": 15, "medium": 30, "high": 60}


def run_science_map(cfg: Config, log: Callable[[str], None]) -> None:
    _journal_map(cfg, log)
    _thematic_matrix(cfg, log)


def _doc_journals(cfg: Config) -> Dict[str, str]:
    return {d: (r.get("journal") or "").strip() for d, r in metadata_by_doc(cfg).items()}


def _journal_map(cfg: Config, log: Callable[[str], None]) -> None:
    journals = _doc_journals(cfg)
    if not any(journals.values()):
        log("  science_map: no journal metadata — journal map skipped.")
        return

    # Concatenate each journal's processed documents.
    texts: Dict[str, List[str]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    if not os.path.isdir(cfg.directory_processed):
        log("  science_map: no processed text — journal map skipped.")
        return
    for fn in sorted(os.listdir(cfg.directory_processed)):
        if not fn.endswith(".txt"):
            continue
        doc_id = fn.split("__", 1)[0]
        j = journals.get(doc_id, "")
        if not j:
            continue
        with open(os.path.join(cfg.directory_processed, fn), "r", encoding="utf-8") as f:
            texts[j].append(f.read())
        counts[j] += 1

    cap = _DENSITY_CAP.get(cfg.visual_density, 30)
    top = sorted(counts, key=lambda j: -counts[j])[:cap]
    if len(top) < 3:
        log(f"  science_map: only {len(top)} journal(s) with text — need >= 3 for a map; skipped.")
        return

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [" ".join(texts[j]) for j in top]
    tfidf = TfidfVectorizer(max_features=2000).fit_transform(docs)
    n_comp = min(2, tfidf.shape[1] - 1, len(top) - 1)
    if n_comp < 2:
        log("  science_map: not enough term variation for a 2D journal map; skipped.")
        return
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(tfidf)

    sizes = [counts[j] for j in top]
    # Static PNG
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], s=[20 + 12 * c for c in sizes],
                     c=sizes, cmap=cfg.visual_colour_scheme, alpha=0.8, edgecolors="grey")
    label_every = 1 if (not cfg.visual_label_filtering or len(top) <= 20) else 2
    for i, j in enumerate(top):
        if i % label_every == 0:
            plt.annotate(j[:30], (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)
    plt.colorbar(sc, label="documents")
    plt.title("Journal map (term-distribution similarity)")
    plt.xlabel("SVD-1"); plt.ylabel("SVD-2"); plt.tight_layout()
    plt.savefig(cfg.output_path("journal_map.png"), dpi=130); plt.close()

    # Interactive
    try:
        import plotly.express as px
        fig = px.scatter(x=coords[:, 0], y=coords[:, 1], size=sizes, color=sizes,
                         hover_name=top, color_continuous_scale=cfg.visual_colour_scheme,
                         labels={"x": "SVD-1", "y": "SVD-2", "color": "documents"},
                         title="Journal map (term-distribution similarity)")
        fig.write_html(cfg.output_path("journal_map.html"), include_plotlyjs="inline")
    except Exception as e:
        log(f"  science_map: interactive journal map skipped ({e}).")

    log(f"  science_map: journal map of {len(top)} journal(s) -> journal_map.png/.html")


def _thematic_matrix(cfg: Config, log: Callable[[str], None]) -> None:
    path = cfg.output_path("topic_evolution.csv")
    if not os.path.exists(path):
        log("  science_map: no topic_evolution.csv — thematic matrix skipped (enable topic_evolution).")
        return
    by_bracket: Dict[str, List[tuple]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            terms = set(t.strip() for t in (r.get("top_terms") or "").split(",") if t.strip())
            by_bracket[r["bracket"]].append((r["topic_id"], terms))
    brackets = list(by_bracket.keys())
    if len(brackets) < 2:
        log("  science_map: need >= 2 brackets for a thematic matrix; skipped.")
        return

    a, b = brackets[0], brackets[-1]
    TA, TB = by_bracket[a], by_bracket[b]
    mat = [[(len(ti & tj) / len(ti | tj) if (ti or tj) else 0.0) for _, tj in TB] for _, ti in TA]

    plt.figure(figsize=(max(5, len(TB)), max(4, len(TA))))
    plt.imshow(mat, cmap=cfg.visual_colour_scheme, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="top-term Jaccard")
    plt.xticks(range(len(TB)), [f"T{t}" for t, _ in TB])
    plt.yticks(range(len(TA)), [f"T{t}" for t, _ in TA])
    plt.xlabel(f"{b} topics"); plt.ylabel(f"{a} topics")
    plt.title(f"Thematic correspondence: {a} vs {b}")
    plt.tight_layout()
    plt.savefig(cfg.output_path("thematic_evolution_matrix.png"), dpi=130); plt.close()
    log(f"  science_map: thematic correspondence matrix ({a} vs {b}) -> thematic_evolution_matrix.png")
