"""Dictionary-based concept analysis (full-text / lexical layer).

Users drop small thematic lexicons in `config/lexicons/<concept>.txt` (one
term/phrase per line). This stage counts concept occurrences, concept
co-occurrences, their longitudinal shifts, and exports context samples (example
sentences) for qualitative reading.

Outputs:
- `concept_counts.csv`        — concept, n_docs, total_occurrences
- `concept_cooccurrence.csv`  — concept_a, concept_b, n_docs_together
- `concept_trends.csv`        — year, concept, n_docs
- `concept_contexts.csv`      — concept, term, doc_id, year, topic, sentence
- `concept_frequencies.png`, `concept_trends.png`
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

from nltk.tokenize import sent_tokenize

from config import Config
from corpus_io import doc_years
from plotting import plt


def _load_lexicons(lexicons_dir: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(lexicons_dir):
        return out
    for fname in sorted(os.listdir(lexicons_dir)):
        if not fname.endswith(".txt"):
            continue
        concept = fname[:-4]
        terms = []
        with open(os.path.join(lexicons_dir, fname), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line.lower())
        if terms:
            out[concept] = terms
    return out


def _doc_topics(cfg: Config) -> Dict[str, str]:
    path = cfg.output_path("topic_assignments.csv")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c != "doc_id"]
        if not cols:
            return {}
        model = cols[0]
        return {r["doc_id"]: f"{model}={r[model]}" for r in reader}


def run_lexical(cfg: Config, log: Callable[[str], None]) -> None:
    lexicons = _load_lexicons(cfg.lexicons_dir)
    if not lexicons:
        log(f"  lexical: no lexicons found in '{cfg.lexicons_dir}' (one <concept>.txt per concept); skipping.")
        return

    patterns = {c: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
                for c, terms in lexicons.items()}
    years = doc_years(cfg)
    topics = _doc_topics(cfg)
    samples_target = max(1, cfg.lexical_context_samples)

    text_dir = cfg.directory_text
    files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))

    total_occ: Counter = Counter()
    doc_freq: Counter = Counter()
    cooc: Counter = Counter()                      # (concept_a, concept_b) -> n docs together
    trends: Dict[Tuple[str, str], set] = defaultdict(set)  # (year, concept) -> doc_ids
    contexts: Dict[str, List[tuple]] = defaultdict(list)

    for name in files:
        doc_id = name.split("__", 1)[0]
        with open(os.path.join(text_dir, name), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        low = text.lower()
        present = []
        for concept, pat in patterns.items():
            hits = pat.findall(low)
            if hits:
                total_occ[concept] += len(hits)
                doc_freq[concept] += 1
                present.append(concept)
                yr = years.get(doc_id, "")
                if yr:
                    trends[(yr, concept)].add(doc_id)
        # co-occurrence (concepts co-present in a document)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                cooc[tuple(sorted((present[i], present[j])))] += 1
        # context sampling (only sent-tokenise if a present concept still needs samples)
        if present and any(len(contexts[c]) < samples_target for c in present):
            for sent in sent_tokenize(text):
                sl = sent.lower()
                for concept in present:
                    if len(contexts[concept]) >= samples_target:
                        continue
                    m = patterns[concept].search(sl)
                    if m:
                        contexts[concept].append((m.group(0), doc_id, years.get(doc_id, ""),
                                                  topics.get(doc_id, ""), re.sub(r"\s+", " ", sent).strip()[:300]))

    # ---- write CSVs ----
    with open(cfg.output_path("concept_counts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["concept", "n_docs", "total_occurrences"])
        for c in sorted(lexicons, key=lambda x: -total_occ[x]):
            w.writerow([c, doc_freq[c], total_occ[c]])

    with open(cfg.output_path("concept_cooccurrence.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_a", "concept_b", "n_docs_together"])
        for (a, b), n in cooc.most_common():
            w.writerow([a, b, n])

    with open(cfg.output_path("concept_trends.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["year", "concept", "n_docs"])
        for (yr, c) in sorted(trends):
            w.writerow([yr, c, len(trends[(yr, c)])])

    with open(cfg.output_path("concept_contexts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["concept", "term", "doc_id", "year", "topic", "sentence"])
        for concept, rows in contexts.items():
            for term, doc_id, yr, topic, sent in rows:
                w.writerow([concept, term, doc_id, yr, topic, sent])

    # ---- charts ----
    concepts_sorted = sorted(lexicons, key=lambda x: -total_occ[x])
    if any(total_occ.values()):
        plt.figure(figsize=(9, max(3.5, 0.4 * len(concepts_sorted) + 1)))
        plt.barh(range(len(concepts_sorted)), [total_occ[c] for c in concepts_sorted], color="#7c3aed")
        plt.yticks(range(len(concepts_sorted)), concepts_sorted, fontsize=9)
        plt.gca().invert_yaxis(); plt.xlabel("occurrences"); plt.title("Concept frequencies")
        plt.tight_layout(); plt.savefig(cfg.output_path("concept_frequencies.png"), dpi=130); plt.close()

    yrs = sorted({yr for (yr, _) in trends})
    if len(yrs) >= 2:
        plt.figure(figsize=(10, 5))
        for c in concepts_sorted:
            series = [len(trends.get((yr, c), set())) for yr in yrs]
            if any(series):
                plt.plot(yrs, series, marker="o", label=c)
        plt.xlabel("year"); plt.ylabel("documents"); plt.title("Concept prevalence over time")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(cfg.output_path("concept_trends.png"), dpi=130); plt.close()

    log(
        f"  lexical: {len(lexicons)} concept(s) — "
        f"{sum(1 for c in lexicons if doc_freq[c])} present; "
        f"{sum(len(v) for v in contexts.values())} context sample(s) "
        f"-> concept_counts.csv, concept_cooccurrence.csv, concept_trends.csv, concept_contexts.csv."
    )
