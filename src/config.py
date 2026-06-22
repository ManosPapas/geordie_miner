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

    @property
    def directory_logs(self) -> str:
        return os.path.join(self.directory_analysis, "logs")

    def output_path(self, *parts: str) -> str:
        return os.path.join(self.directory_analysis, *parts)

    def log_path(self, name: str) -> str:
        return os.path.join(self.directory_logs, name)

    # Language
    language: str

    # Preprocessing
    stopwords_file: str
    substitutions_file: str
    min_frequency: int
    use_spacy: bool
    exclude_sections: list

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

    # Runtime / parallelism
    max_cpu_count: int

    # Ingest
    bulk_text_mode: str          # auto | on | off
    bulk_record_split: str       # auto | line | blank

    # Preprocessing (extended)
    remove_references: bool
    strip_citation_markers: bool
    preserve_sentences: bool

    # Topic-model enable switches
    enable_kmeans: bool
    enable_lda: bool
    enable_nmf: bool
    enable_hdp: bool
    enable_bertopic: bool

    # Linguistic annotation
    annotation_enable: bool
    annotation_engine: str       # spacy | stanza
    annotation_model: str
    annotation_tasks: list       # subset of: sentence, pos, ner

    # Longitudinal analysis
    longitudinal_enable: bool
    longitudinal_bracket_years: str   # auto | 5 | 10
    longitudinal_min_docs: int

    # Stability
    stability_seeds: list        # explicit, fixed seed schedule

    # --- Wave 2 ---

    # Bibliographic import (BibTeX / RIS / CSV)
    csv_mapping: str             # "field:Column,field:Column" overrides for CSV import

    # External metadata providers / enrichment
    enrich_enable: bool
    provider: str                # openalex | crossref | scopus
    provider_fallback: str       # provider to fall back to (e.g. openalex)
    provider_max: int            # cap records to enrich (0 = no cap)
    provider_mailto: str         # contact email for OpenAlex/Crossref "polite pool"

    # References / citation analysis
    compute_impact: bool
    compute_cocitation: bool
    compute_coupling: bool

    # Collaboration / co-authorship
    collaboration_enable: bool

    # Full-text lexical analysis
    lexical_enable: bool
    lexicons_dir: str
    lexical_context_samples: int

    # Topic evolution
    topic_evolution_enable: bool

    # Science mapping
    science_map_enable: bool

    # Visuals
    visual_density: str          # low | medium | high
    visual_label_filtering: bool
    visual_colour_scheme: str


def load_config(
    config_path: str,
    data_dir: str,
    output_base: str = "output",
    run_name: str | None = None,
) -> Config:
    """Parse the .ini-style config file and return a typed Config.

    Validates that the config file and data directory exist before returning.
    Output directory is `<output_base>/<run_name or basename(data_dir)>`.
    Stopwords / substitutions paths in the config are resolved relative to the
    config file's location (so `config/config.ini` referencing `stopwords.txt`
    points at `config/stopwords.txt`).
    """
    # `config_path` may be a single path or a list of paths (base + profile
    # overlay); later files override earlier ones.
    paths = [config_path] if isinstance(config_path, str) else list(config_path)
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"Error: configuration file '{p}' not found.")
    if not os.path.exists(data_dir):
        sys.exit(f"Error: data directory '{data_dir}' not found.")

    # inline_comment_prefixes lets users put `# comment` after a value on the same
    # line (as the README examples show), not just on their own line.
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.read(paths)
    if not cp.sections():
        sys.exit(f"Error: no sections found in config '{paths}'.")

    # Relative paths (stopwords, lexicons, ...) resolve against the base config dir.
    config_dir = os.path.dirname(os.path.abspath(paths[0]))

    def resolve_relative_to_config(value: str) -> str:
        return value if os.path.isabs(value) else os.path.join(config_dir, value)

    name = run_name if run_name else os.path.basename(os.path.normpath(data_dir))
    directory_analysis = os.path.join(output_base, name)
    directory_text = os.path.join(directory_analysis, "text")
    directory_processed = os.path.join(directory_analysis, "text_processed")

    figsize_raw = cp.get("phrase_analysis", "dendrogram_figsize", fallback="10,7")
    dendrogram_figsize = tuple(int(x.strip()) for x in figsize_raw.split(","))  # type: ignore[assignment]

    def _int_list(value: str) -> list:
        return [int(s.strip()) for s in value.split(",") if s.strip()]

    def _str_list(value: str) -> list:
        return [s.strip().lower() for s in value.split(",") if s.strip()]

    cfg = Config(
        config_path=os.path.abspath(paths[-1]),
        directory_data=data_dir,
        directory_analysis=directory_analysis,
        directory_text=directory_text,
        directory_processed=directory_processed,

        language=cp.get("default", "language", fallback="english"),

        stopwords_file=resolve_relative_to_config(cp.get("preprocessing", "stopwords_file", fallback="stopwords.txt")),
        substitutions_file=resolve_relative_to_config(cp.get("preprocessing", "substitutions_file", fallback="substitutions.txt")),
        min_frequency=cp.getint("preprocessing", "min_frequency", fallback=5),
        use_spacy=cp.getboolean("preprocessing", "use_spacy", fallback=False),
        exclude_sections=[s.strip() for s in cp.get("preprocessing", "exclude_sections", fallback="").split(",") if s.strip()],

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

        max_cpu_count=cp.getint("default", "max_cpu_count", fallback=4),

        bulk_text_mode=cp.get("ingest", "bulk_text_mode", fallback="auto").strip().lower(),
        bulk_record_split=cp.get("ingest", "bulk_record_split", fallback="auto").strip().lower(),

        remove_references=cp.getboolean("preprocessing", "remove_references", fallback=False),
        strip_citation_markers=cp.getboolean("preprocessing", "strip_citation_markers", fallback=False),
        preserve_sentences=cp.getboolean("preprocessing", "preserve_sentences", fallback=False),

        enable_kmeans=cp.getboolean("topic_modelling", "enable_kmeans", fallback=True),
        enable_lda=cp.getboolean("topic_modelling", "enable_lda", fallback=True),
        enable_nmf=cp.getboolean("topic_modelling", "enable_nmf", fallback=True),
        enable_hdp=cp.getboolean("topic_modelling", "enable_hdp", fallback=True),
        enable_bertopic=cp.getboolean("topic_modelling", "enable_bertopic", fallback=True),

        annotation_enable=cp.getboolean("annotation", "enable", fallback=False),
        annotation_engine=cp.get("annotation", "annotation_engine", fallback="spacy").strip().lower(),
        annotation_model=cp.get("annotation", "annotation_model", fallback="en_core_web_sm").strip(),
        annotation_tasks=_str_list(cp.get("annotation", "annotation_tasks", fallback="sentence,pos,ner")),

        longitudinal_enable=cp.getboolean("longitudinal", "enable", fallback=False),
        longitudinal_bracket_years=cp.get("longitudinal", "longitudinal_bracket_years", fallback="auto").strip().lower(),
        longitudinal_min_docs=cp.getint("longitudinal", "longitudinal_min_docs", fallback=3),

        stability_seeds=_int_list(cp.get("stability", "stability_seeds", fallback="42,123,2024,7,99")),

        csv_mapping=cp.get("import", "csv_mapping", fallback="").strip(),

        enrich_enable=cp.getboolean("providers", "enrich_enable", fallback=False),
        provider=cp.get("providers", "provider", fallback="openalex").strip().lower(),
        provider_fallback=cp.get("providers", "provider_fallback", fallback="openalex").strip().lower(),
        provider_max=cp.getint("providers", "provider_max", fallback=0),
        provider_mailto=cp.get("providers", "provider_mailto", fallback="").strip(),

        compute_impact=cp.getboolean("references", "compute_impact", fallback=True),
        compute_cocitation=cp.getboolean("references", "compute_cocitation", fallback=True),
        compute_coupling=cp.getboolean("references", "compute_coupling", fallback=True),

        collaboration_enable=cp.getboolean("collaboration", "enable", fallback=False),

        lexical_enable=cp.getboolean("lexical", "enable", fallback=False),
        lexicons_dir=resolve_relative_to_config(cp.get("lexical", "lexicons_dir", fallback="lexicons")),
        lexical_context_samples=cp.getint("lexical", "context_samples", fallback=5),

        topic_evolution_enable=cp.getboolean("topic_evolution", "enable", fallback=False),

        science_map_enable=cp.getboolean("science_map", "enable", fallback=False),

        visual_density=cp.get("visuals", "density", fallback="medium").strip().lower(),
        visual_label_filtering=cp.getboolean("visuals", "label_filtering", fallback=True),
        visual_colour_scheme=cp.get("visuals", "colour_scheme", fallback="viridis").strip(),
    )

    # `remove_references` is a friendly shortcut for excluding the references
    # section during preprocessing — fold it into the existing mechanism.
    if cfg.remove_references and "references" not in cfg.exclude_sections:
        cfg.exclude_sections.append("references")

    return cfg


def init_directories(cfg: Config, stages: list[str] | None = None) -> None:
    """Prepare output directories for a run.

    Full runs (any run that includes the `ingest` stage, or `stages=None`) wipe
    `directory_analysis` entirely so stale files from previous runs — old K
    values, leftover topic files, accumulated logs — can't bleed through.

    Partial-stage runs (e.g. `--stages topics`) only wipe what the running
    stages own; everything else is preserved.
    """
    if stages is None or "ingest" in stages:
        # Full rebuild — clean slate.
        if os.path.exists(cfg.directory_analysis):
            try:
                shutil.rmtree(cfg.directory_analysis)
            except OSError as e:
                sys.exit(
                    f"Error: could not wipe '{cfg.directory_analysis}' — "
                    f"close any files open from previous runs (e.g. Gephi on "
                    f"network.gexf, image viewers on the word clouds) and try again.\n"
                    f"Underlying error: {e}"
                )
        os.makedirs(cfg.directory_analysis)
        os.makedirs(cfg.directory_logs)
        os.makedirs(cfg.directory_text)
        os.makedirs(cfg.directory_processed)
        return

    # Partial run: create the base dirs if missing, otherwise leave them alone.
    os.makedirs(cfg.directory_analysis, exist_ok=True)
    os.makedirs(cfg.directory_logs, exist_ok=True)

    if "preprocess" in stages:
        if os.path.exists(cfg.directory_processed):
            shutil.rmtree(cfg.directory_processed)
        os.makedirs(cfg.directory_processed)


def write_config_log(cfg: Config) -> None:
    """Write a snapshot of the resolved configuration to the analysis directory."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Log created on: {stamp}", "", "Resolved configuration:", ""]
    for key, value in asdict(cfg).items():
        lines.append(f"{key} = {value}")
    with open(cfg.log_path("config_used.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
