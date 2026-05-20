"""Term analysis: raw + lemmatised frequencies, TF-IDF, word clouds."""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

from gm_config import Config
from gm_preprocess import lemmatise_text


def _term_table(corpus: List[List[str]], top_n: int) -> pd.DataFrame:
    """Build a Term/Count/% of Docs/TF-IDF table for the top-N terms."""
    counter = Counter(t for doc in corpus for t in doc)
    doc_freq = {t: sum(1 for doc in corpus if t in doc) for t in counter}
    n_docs = len(corpus)

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([" ".join(doc) for doc in corpus])
    vocab = vec.vocabulary_

    rows = []
    for term, count in counter.most_common(top_n):
        pct = (doc_freq[term] / n_docs * 100) if n_docs else 0.0
        if term in vocab:
            tfidf_value = float(tfidf[:, vocab[term]].sum())
        else:
            tfidf_value = 0.0
        rows.append((term, count, pct, tfidf_value))
    return pd.DataFrame(rows, columns=["Term", "Count", "% of Docs", "TF-IDF"])


def run_term_analysis(
    cfg: Config,
    corpus: List[List[str]],
    log: Callable[[str], None],
) -> None:
    """Run term frequency, TF-IDF and word-cloud generation for raw + lemmatised corpora."""
    lemmatised = [lemmatise_text(" ".join(doc)) for doc in corpus]

    raw_df = _term_table(corpus, cfg.top_n_terms)
    lem_df = _term_table(lemmatised, cfg.top_n_terms)

    raw_df.to_excel(os.path.join(cfg.directory_analysis, "analysis_terms_single_raw.xlsx"), index=False)
    lem_df.to_excel(os.path.join(cfg.directory_analysis, "analysis_terms_single_lemmatised.xlsx"), index=False)
    raw_df.to_csv(os.path.join(cfg.directory_analysis, "analysis_terms_single_raw.csv"), index=False)
    lem_df.to_csv(os.path.join(cfg.directory_analysis, "analysis_terms_single_lemmatised.csv"), index=False)
    log(f"Term analysis: top {cfg.top_n_terms} raw + lemmatised terms exported.")

    if cfg.output_wordcloud:
        raw_counts = Counter(t for doc in corpus for t in doc)
        lem_counts = Counter(t for doc in lemmatised for t in doc)
        _save_wordcloud(cfg, raw_counts, "analysis_descriptive_raw_wordcloud.jpg")
        _save_wordcloud(cfg, lem_counts, "analysis_descriptive_lemmatised_wordcloud.jpg")
        log("Word clouds exported (raw + lemmatised).")


def _save_wordcloud(cfg: Config, counts: Counter, filename: str) -> None:
    if not counts:
        return
    wc = WordCloud(
        width=cfg.wordcloud_width,
        height=cfg.wordcloud_height,
        background_color=cfg.wordcloud_background_color,
    ).generate_from_frequencies(counts)
    wc.to_file(os.path.join(cfg.directory_analysis, filename))
