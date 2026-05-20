"""Configuration loader for Geordie Miner.

Replaces the original `_functions.load_config()` with a typed dataclass.
No global state — pass the returned `Config` object around explicitly.
"""

from __future__ import annotations

import configparser
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple


@dataclass
class Config:
    # Paths
    config_path: str
    directory_data: str
    directory_analysis: str
    directory_text: str
    directory_processed: str

    # Language
    language: str

    # Preprocessing
    stopwords_file: str
    substitutions_file: str
    min_frequency: int

    # Term analysis
    top_n_terms: int
    output_wordcloud: bool
    wordcloud_width: int
    wordcloud_height: int
    wordcloud_background_color: str

    # Phrase analysis
    bigram_threshold: int
    trigram_threshold: int
    bigram_export_count: int
    trigram_export_count: int
    window_size: int
    cooccurrence_threshold: int
    clustering_metric: str
    linkage_method: str
    dendrogram_figsize: Tuple[int, int]

    # Topic-modelling multipliers (each model is re-run at K, K*m1, K*m2, K*m3)
    topic_modelling_multi1: int
    topic_modelling_multi2: int
    topic_modelling_multi3: int

    # Topic models
    kmeans_topics: int
    lda_topics: int
    lda_terms_per_topic: int
    lda_passes: int
    lda_per_word_topics: int
    nmf_topics: int
    nmf_terms_per_topic: int
    nmf_max_iter: int
    terms_per_topic_hdp: int


def load_config(config_path: str, data_dir: str) -> Config:
    """Parse the .ini-style config file and return a typed Config.

    Validates that the config file and data directory exist before returning.
    Output directory names are derived from `data_dir`.
    """
    if not os.path.exists(config_path):
        sys.exit(f"Error: configuration file '{config_path}' not found.")
    if not os.path.exists(data_dir):
        sys.exit(f"Error: data directory '{data_dir}' not found.")

    cp = configparser.ConfigParser()
    cp.read(config_path)
    if not cp.sections():
        sys.exit(f"Error: no sections found in config '{config_path}'.")

    directory_analysis = f"analysis_{os.path.basename(os.path.normpath(data_dir))}"
    directory_text = os.path.join(directory_analysis, "text")
    directory_processed = os.path.join(directory_analysis, "text_processed")

    figsize_raw = cp.get("phrase_analysis", "dendrogram_figsize", fallback="10,7")
    dendrogram_figsize = tuple(int(x.strip()) for x in figsize_raw.split(","))  # type: ignore[assignment]

    return Config(
        config_path=os.path.abspath(config_path),
        directory_data=data_dir,
        directory_analysis=directory_analysis,
        directory_text=directory_text,
        directory_processed=directory_processed,

        language=cp.get("default", "language", fallback="english"),

        stopwords_file=cp.get("preprocessing", "stopwords_file", fallback="stopwords.txt"),
        substitutions_file=cp.get("preprocessing", "substitutions_file", fallback="substitutions.txt"),
        min_frequency=cp.getint("preprocessing", "min_frequency", fallback=5),

        top_n_terms=cp.getint("term_analysis", "top_n_terms", fallback=200),
        output_wordcloud=cp.getboolean("term_analysis", "output_wordcloud", fallback=True),
        wordcloud_width=cp.getint("term_analysis", "wordcloud_width", fallback=800),
        wordcloud_height=cp.getint("term_analysis", "wordcloud_height", fallback=400),
        wordcloud_background_color=cp.get("term_analysis", "wordcloud_background_color", fallback="white"),

        bigram_threshold=cp.getint("phrase_analysis", "bigram_threshold", fallback=10),
        trigram_threshold=cp.getint("phrase_analysis", "trigram_threshold", fallback=10),
        bigram_export_count=cp.getint("phrase_analysis", "bigram_export_count", fallback=100),
        trigram_export_count=cp.getint("phrase_analysis", "trigram_export_count", fallback=100),
        window_size=cp.getint("phrase_analysis", "window_size", fallback=5),
        cooccurrence_threshold=cp.getint("phrase_analysis", "cooccurrence_threshold", fallback=3),
        clustering_metric=cp.get("phrase_analysis", "clustering_metric", fallback="jaccard"),
        linkage_method=cp.get("phrase_analysis", "linkage_method", fallback="ward"),
        dendrogram_figsize=dendrogram_figsize,

        topic_modelling_multi1=cp.getint("topic_modelling", "topic_modelling_multi1", fallback=0),
        topic_modelling_multi2=cp.getint("topic_modelling", "topic_modelling_multi2", fallback=0),
        topic_modelling_multi3=cp.getint("topic_modelling", "topic_modelling_multi3", fallback=0),

        kmeans_topics=cp.getint("topic_kmeans", "kmeans_topics", fallback=10),

        lda_topics=cp.getint("topic_lda", "lda_topics", fallback=10),
        lda_terms_per_topic=cp.getint("topic_lda", "lda_terms_per_topic", fallback=10),
        lda_passes=cp.getint("topic_lda", "lda_passes", fallback=25),
        lda_per_word_topics=cp.getint("topic_lda", "lda_per_word_topics", fallback=0),

        nmf_topics=cp.getint("topic_nmf", "nmf_topics", fallback=10),
        nmf_terms_per_topic=cp.getint("topic_nmf", "nmf_terms_per_topic", fallback=10),
        nmf_max_iter=cp.getint("topic_nmf", "nmf_max_iter", fallback=500),

        terms_per_topic_hdp=cp.getint("topic_hdp", "terms_per_topic_hdp", fallback=10),
    )


def init_directories(cfg: Config, stages: list[str] | None = None) -> None:
    """Create output directories. Only wipe the ones that the current stages will rebuild.

    Without a `stages` argument, wipes everything (full reset).
    With `stages`, preserves directories that earlier (skipped) stages produced.
    """
    os.makedirs(cfg.directory_analysis, exist_ok=True)

    if stages is None or "ingest" in stages:
        for d in (cfg.directory_text, cfg.directory_processed):
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs(cfg.directory_text)
        os.makedirs(cfg.directory_processed)
        # corpus stats file is append-only — clear it on full ingest
        stats_path = os.path.join(cfg.directory_analysis, "analysis_corpus.txt")
        if os.path.exists(stats_path):
            os.remove(stats_path)
        return

    if "preprocess" in stages:
        if os.path.exists(cfg.directory_processed):
            shutil.rmtree(cfg.directory_processed)
        os.makedirs(cfg.directory_processed)


def write_config_log(cfg: Config) -> None:
    """Write a snapshot of the resolved configuration to the analysis directory."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(cfg.directory_analysis, "_log_config.log")
    lines = [f"Log created on: {stamp}", "", "Resolved configuration:", ""]
    for key, value in asdict(cfg).items():
        lines.append(f"{key} = {value}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
