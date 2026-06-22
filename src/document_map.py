"""Interactive 2D map of every document in the corpus.

Uses the document embeddings produced by BERTopic and projects them to 2D
with UMAP. Each paper becomes a dot, similar papers cluster together visually,
hover to see the title.

Output:
- `document_map.html` (interactive, self-contained — open in any browser)
- `document_map.png`  (static, for inclusion in papers / slides)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from config import Config


def _try_imports(log: Callable[[str], None]) -> bool:
    try:
        import umap  # noqa: F401
        import plotly  # noqa: F401
        return True
    except Exception as e:
        log(
            f"  document map: missing deps ({e.__class__.__name__}). "
            f"Install with: pip install umap-learn plotly"
        )
        return False


def _hover_labels(cfg: Config, doc_ids: List[str]) -> List[str]:
    """Build a hover label per doc — uses title from metadata.csv if available, else doc_id."""
    metadata_path = Path(cfg.output_path("metadata.csv"))
    titles_by_id: Dict[str, str] = {}
    if metadata_path.exists():
        try:
            df = pd.read_csv(metadata_path)
            for _, row in df.iterrows():
                if row.get("title"):
                    titles_by_id[str(row["doc_id"])] = str(row["title"])[:100]
        except Exception:
            pass
    return [titles_by_id.get(did, did) for did in doc_ids]


def _journal_labels(cfg: Config, doc_ids: List[str]) -> List[str]:
    """Journal per doc from metadata.csv (empty list if unavailable)."""
    path = Path(cfg.output_path("metadata.csv"))
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return []
    by_id = {str(r["doc_id"]): (str(r.get("journal", "")) or "(unknown)") for _, r in df.iterrows()}
    return [by_id.get(did, "(unknown)") for did in doc_ids]


def write_document_map(
    cfg: Config,
    embeddings: np.ndarray,
    doc_ids: List[str],
    topic_labels: np.ndarray,
    log: Callable[[str], None],
) -> None:
    """Project embeddings to 2D with UMAP, render as plotly interactive + matplotlib static."""
    if not _try_imports(log):
        return
    if embeddings is None or len(embeddings) == 0:
        log("  document map: no embeddings to project.")
        return

    import umap
    import plotly.express as px

    n_neighbors = min(15, max(2, len(embeddings) - 1))
    log(f"  document map: projecting {len(embeddings)}-D embeddings to 2D (UMAP, n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    hover_text = _hover_labels(cfg, doc_ids)
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "doc_id": doc_ids,
        "topic": [f"Topic {int(t) + 1}" for t in topic_labels],
        "hover": hover_text,
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="topic",
        hover_data={"x": False, "y": False, "doc_id": True, "hover": True, "topic": True},
        title=f"Document map — {os.path.basename(os.path.normpath(cfg.directory_data))}",
        labels={"x": "UMAP-1", "y": "UMAP-2"},
        height=700,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(legend_title_text="BERTopic cluster", template="plotly_white")
    html_path = cfg.output_path("document_map.html")
    fig.write_html(html_path, include_plotlyjs="inline")
    log(f"  document map: interactive HTML written to {html_path}")

    # Optional journal-level view: same coordinates, coloured by source/journal.
    journals = _journal_labels(cfg, doc_ids)
    if journals and len(set(journals)) > 1:
        df["journal"] = journals
        jfig = px.scatter(
            df, x="x", y="y", color="journal",
            hover_data={"x": False, "y": False, "doc_id": True, "hover": True, "journal": True},
            title=f"Document map by journal — {os.path.basename(os.path.normpath(cfg.directory_data))}",
            labels={"x": "UMAP-1", "y": "UMAP-2"}, height=700,
        )
        jfig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
        jfig.update_layout(legend_title_text="journal", template="plotly_white")
        jfig.write_html(cfg.output_path("document_map_by_journal.html"), include_plotlyjs="inline")
        log("  document map: journal-coloured view -> document_map_by_journal.html")

    # Also a static PNG (matplotlib — already a dep).
    try:
        from plotting import plt

        plt.figure(figsize=(10, 7))
        unique = sorted(set(topic_labels))
        for t in unique:
            mask = topic_labels == t
            plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Topic {int(t) + 1}", s=20, alpha=0.7)
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"Document map — {os.path.basename(os.path.normpath(cfg.directory_data))}")
        plt.tight_layout()
        png_path = cfg.output_path("document_map.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        log(f"  document map: static PNG written to {png_path}")
    except Exception as e:
        log(f"  document map: static PNG skipped ({e})")
