"""Reproducibility: export a machine-readable run configuration.

Writes `run_config.json` (always; stdlib) and `run_config.yaml` (when PyYAML is
available) capturing everything needed to rerun the analysis as closely as
possible: input sources, all resolved settings, model parameters, selected
algorithms, random seeds, library versions, the application version, and the
execution environment.

The export documents the environment rather than implying cross-machine
identity of results — exact reproducibility can vary with hardware and library
versions.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Callable, Dict, List, Optional

from config import Config
from version import __version__

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# Libraries whose versions materially affect results — recorded for the run log.
KEY_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "nltk", "gensim", "spacy", "stanza",
    "bertopic", "sentence-transformers", "umap-learn", "hdbscan", "networkx",
    "matplotlib", "wordcloud", "plotly", "pypdf", "PyYAML", "tqdm", "openpyxl",
]


def _library_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in KEY_PACKAGES:
        try:
            out[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            out[pkg] = None  # not installed
    return out


def write_run_config(cfg: Config, stages: List[str], log: Callable[[str], None]) -> None:
    data_dir = cfg.directory_data
    files = sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []

    # Round-trip settings through JSON so tuples become lists and everything is
    # YAML/JSON-safe (asdict keeps dendrogram_figsize as a tuple otherwise).
    settings = json.loads(json.dumps(asdict(cfg), default=str))

    payload = {
        "geordie_miner_version": __version__,
        "input": {
            "data_dir": data_dir,
            "n_input_files": len(files),
            "input_files": files[:500],
            "config_file": cfg.config_path,
        },
        "stages_requested": list(stages),
        "settings": settings,
        "random_seeds": {
            "stability_seeds": list(cfg.stability_seeds),
            "model_random_state": 42,  # KMeans / LDA / NMF / UMAP all use a fixed 42
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "data_sources": {
            "bibliographic_import": os.path.exists(cfg.output_path("imported_metadata.csv")),
            "enrichment_enabled": bool(getattr(cfg, "enrich_enable", False)),
            "provider": getattr(cfg, "provider", ""),
            "input_file_types": sorted({f.rsplit(".", 1)[-1].lower() for f in files if "." in f}),
        },
        "library_versions": _library_versions(),
        "note": (
            "Documents the execution environment for reproducibility. Exact results "
            "can vary across machines and library versions; this records the run "
            "rather than guaranteeing cross-machine identity of results."
        ),
    }

    with open(cfg.output_path("run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    written = "run_config.json"
    if yaml is not None:
        # payload was round-tripped through JSON above, so it is YAML-safe.
        with open(cfg.output_path("run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        written += " + run_config.yaml"

    _write_reproducibility_md(cfg, payload)
    n_libs = sum(1 for v in payload["library_versions"].values() if v)
    log(f"  reproducibility: {written} + reproducibility.md written ({n_libs} library versions + environment).")


def _write_reproducibility_md(cfg: Config, payload: dict) -> None:
    """Human-readable companion to run_config.json."""
    s = payload["settings"]
    env = payload["environment"]
    ds = payload["data_sources"]
    key_libs = ["numpy", "pandas", "scikit-learn", "gensim", "nltk", "spacy", "stanza", "bertopic"]
    lines = [
        f"# Reproducibility report — {os.path.basename(os.path.normpath(cfg.directory_analysis))}",
        "",
        f"- **Geordie Miner version:** {payload['geordie_miner_version']}",
        f"- **Python:** {env['python_version']} on {env['platform']}",
        f"- **Input:** {payload['input']['n_input_files']} file(s) ({', '.join(ds['input_file_types']) or 'n/a'}) "
        f"from `{payload['input']['data_dir']}`",
        f"- **Data sources:** bibliographic import = {ds['bibliographic_import']}; "
        f"enrichment = {ds['enrichment_enabled']} ({ds['provider']})",
        "",
        "## Key configuration",
        "",
        f"- Stages: {', '.join(payload['stages_requested'])}",
        f"- Topic K (kmeans/lda/nmf): {s.get('kmeans_topics')}/{s.get('lda_topics')}/{s.get('nmf_topics')}; "
        f"multipliers {s.get('topic_modelling_multi1')},{s.get('topic_modelling_multi2')},{s.get('topic_modelling_multi3')}",
        f"- min_frequency: {s.get('min_frequency')}; preserve_sentences: {s.get('preserve_sentences')}; "
        f"remove_references: {s.get('remove_references')}",
        f"- Random seeds (stability): {payload['random_seeds']['stability_seeds']}; "
        f"model random_state: {payload['random_seeds']['model_random_state']}",
        "",
        "## Key library versions",
        "",
    ]
    lines += [f"- {p}: {payload['library_versions'].get(p) or 'not installed'}" for p in key_libs]
    lines += ["", f"_{payload['note']}_", ""]
    with open(cfg.output_path("reproducibility.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
