"""Topic-stability analysis.

After coherence scoring has identified the best-performing model, this stage
re-runs that model across a FIXED, explicit seed schedule (declared in config,
not chosen opportunistically) and reports how consistent the solution is:

- per-seed coherence (c_v), plus its mean and variance across reruns
- mean cross-seed topic-term agreement (Jaccard, best-matched topics)
- a plain judgement: stable / moderately sensitive / unstable

Outputs:
- `topic_stability.csv`      — per-topic mean Jaccard vs the other seeds
- `stability_report.json`    — machine-readable summary
- `stability_report.txt`     — human-readable summary

Stable topics (high Jaccard, low coherence variance) are likely real signal;
unstable ones are likely artefacts of random initialisation.
"""

from __future__ import annotations

import csv
import json
import os
from statistics import mean, pvariance
from typing import Callable, List, Tuple

import pandas as pd
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import Config


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_match(topic: List[str], candidates: List[List[str]]) -> Tuple[int, float]:
    best_idx, best_score = -1, -1.0
    for j, cand in enumerate(candidates):
        s = _jaccard(topic, cand)
        if s > best_score:
            best_idx, best_score = j, s
    return best_idx, best_score


def _select_primary_model(cfg: Config, log: Callable[[str], None]) -> Tuple[str, int]:
    """Pick the most coherent model (by c_v) as the primary candidate to validate.

    Only LDA/NMF are seed-refittable here; if the best model is something else
    (HDP/KMeans/BERTopic) we validate LDA at the same K and note it.
    """
    path = cfg.output_path("coherence_scores.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            best = df.sort_values("coherence_c_v", ascending=False).iloc[0]
            model, k = str(best["model"]).upper(), int(best["K"])
            if model not in ("LDA", "NMF"):
                log(f"  stability: most-coherent model is {model} (K={k}); validating LDA at K={k} (seed-refittable).")
                model = "LDA"
            else:
                log(f"  stability: most-coherent model is {model} at K={k} — selected as primary candidate.")
            return model, (k if k >= 2 else cfg.lda_topics)
    log(f"  stability: no coherence scores available; defaulting to LDA at K={cfg.lda_topics}.")
    return "LDA", cfg.lda_topics


def _fit_lda_topics(bow, id2word, k: int, passes: int, seed: int, top_n: int) -> List[List[str]]:
    lda = LdaModel(
        corpus=bow, num_topics=k, id2word=id2word, passes=passes, iterations=1000,
        alpha="auto", eta="auto", random_state=seed, per_word_topics=False,
    )
    return [[w for w, _ in lda.show_topic(i, topn=top_n)] for i in range(k)]


def _fit_nmf_topics(docs: List[str], k: int, max_iter: int, seed: int, top_n: int) -> List[List[str]]:
    vec = TfidfVectorizer()
    dtm = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    model = NMF(n_components=k, random_state=seed, max_iter=max_iter)
    model.fit(dtm)
    out: List[List[str]] = []
    for topic in model.components_:
        idx = topic.argsort()[: -top_n - 1 : -1]
        out.append([terms[j] for j in idx])
    return out


def _coherence_cv(topic_word_lists: List[List[str]], tokenised_docs: List[List[str]], dictionary: Dictionary) -> float:
    lists = [w for w in topic_word_lists if w]
    if not lists:
        return float("nan")
    try:
        return CoherenceModel(
            topics=lists, texts=tokenised_docs, dictionary=dictionary,
            coherence="c_v", processes=1,
        ).get_coherence()
    except Exception:
        return float("nan")


def _judge(mean_jaccard: float) -> str:
    if mean_jaccard >= 0.7:
        return "stable"
    if mean_jaccard >= 0.4:
        return "moderately sensitive"
    return "unstable"


def run_stability(cfg: Config, docs: List[str], log: Callable[[str], None], top_n_words: int = 10) -> None:
    if not docs:
        log("  stability: no documents; skipping.")
        return

    seeds = list(cfg.stability_seeds) or [42, 123, 2024, 7, 99]
    model_name, k = _select_primary_model(cfg, log)
    if k < 2:
        log("  stability: K < 2; skipping.")
        return
    if len(docs) <= k:
        log(f"  stability: only {len(docs)} docs for K={k}; skipping.")
        return

    tokenised = [d.split() for d in docs]
    dictionary = Dictionary(tokenised)

    vec = CountVectorizer()
    dtm = vec.fit_transform(docs)
    id2word = dict(enumerate(vec.get_feature_names_out()))
    bow = list(matutils.Sparse2Corpus(dtm, documents_columns=False))

    log(f"  stability: re-running {model_name} (K={k}) across {len(seeds)} fixed seed(s): {seeds}")

    runs: List[List[List[str]]] = []
    per_seed_coherence: List[Tuple[int, float]] = []
    for seed in seeds:
        if model_name == "NMF":
            topics = _fit_nmf_topics(docs, k, cfg.nmf_max_iter, seed, top_n_words)
        else:
            topics = _fit_lda_topics(bow, id2word, k, cfg.lda_passes, seed, top_n_words)
        runs.append(topics)
        cv = _coherence_cv(topics, tokenised, dictionary)
        per_seed_coherence.append((seed, cv))
        log(f"    seed {seed}: c_v = {cv:.3f}" if cv == cv else f"    seed {seed}: c_v = n/a")

    # Per-topic stability (relative to the first seed's run).
    rows: List[Tuple[int, float, str]] = []
    for i, topic in enumerate(runs[0]):
        scores = [_best_match(topic, runs[j])[1] for j in range(len(runs)) if j != 0]
        rows.append((i + 1, float(mean(scores)) if scores else 0.0, ", ".join(topic[:5])))
    rows.sort(key=lambda r: -r[1])
    with open(cfg.output_path("topic_stability.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "mean_jaccard", "top_5_words"])
        for topic_id, mj, words in rows:
            writer.writerow([topic_id, f"{mj:.3f}", words])

    # Cross-seed agreement: average best-match Jaccard over all seed pairs.
    pair_means: List[float] = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            pair_means.append(float(mean(_best_match(t, runs[b])[1] for t in runs[a])))
    mean_jaccard = float(mean(pair_means)) if pair_means else 0.0

    valid_cv = [c for _, c in per_seed_coherence if c == c]
    coh_mean = float(mean(valid_cv)) if valid_cv else float("nan")
    coh_var = float(pvariance(valid_cv)) if len(valid_cv) > 1 else 0.0
    judgement = _judge(mean_jaccard)

    report = {
        "model": model_name,
        "K": k,
        "seeds": seeds,
        "per_seed_coherence_c_v": {str(s): (None if c != c else round(c, 4)) for s, c in per_seed_coherence},
        "coherence_mean": None if coh_mean != coh_mean else round(coh_mean, 4),
        "coherence_variance": round(coh_var, 6),
        "coherence_std": round(coh_var ** 0.5, 4),
        "mean_cross_seed_jaccard": round(mean_jaccard, 4),
        "judgement": judgement,
        "notes": (
            "Seeds are a fixed, pre-declared schedule (config.stability_seeds). "
            "Judgement is based on mean cross-seed topic-term Jaccard; coherence "
            "variance indicates how much the fit quality moves with initialisation."
        ),
    }
    with open(cfg.output_path("stability_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(cfg.output_path("stability_report.txt"), "w", encoding="utf-8") as f:
        f.write("Topic-model stability report\n")
        f.write("============================\n\n")
        f.write(f"Primary model      : {model_name} (K={k})\n")
        f.write(f"Seed schedule      : {', '.join(map(str, seeds))}\n")
        f.write(
            f"Coherence (c_v)    : mean {report['coherence_mean']} "
            f"(variance {report['coherence_variance']}, std {report['coherence_std']})\n"
        )
        f.write("Per-seed c_v       : " + ", ".join(
            f"{s}={'n/a' if c != c else round(c, 3)}" for s, c in per_seed_coherence
        ) + "\n")
        f.write(f"Mean cross-seed Jaccard : {mean_jaccard:.3f}\n")
        f.write(f"Judgement          : {judgement.upper()}\n")

    log(
        f"  stability: judgement = {judgement.upper()} "
        f"(mean cross-seed Jaccard {mean_jaccard:.3f}; c_v mean "
        f"{'n/a' if coh_mean != coh_mean else round(coh_mean, 3)}, var {coh_var:.4f}). "
        f"-> stability_report.json/.txt"
    )
