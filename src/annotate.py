"""Optional linguistic annotation stage with a pluggable spaCy / Stanza backend.

Runs sentence segmentation, POS tagging and named-entity recognition over the
converted full text (`text/`, NOT the stopword-stripped processed text, so casing
and structure are preserved for the taggers), and exports structured annotations
that later analysis can reuse without rerunning the model:

- `annotations/<doc>.jsonl`  — one JSON object per sentence:
    {"text": ..., "tokens": [{"text","lemma","pos"}], "entities": [{"text","label"}]}
- `annotations_manifest.json` — per-doc source-text hash + counts; unchanged docs
  are skipped on re-run (`--stages annotate`).
- `entity_counts.csv`         — corpus-wide NER aggregate (entity, label, count).

The stage degrades gracefully: if the selected engine/model can't be loaded it
logs install instructions and no-ops rather than crashing the pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from typing import Callable, List, Optional

from tqdm import tqdm

from config import Config


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class Annotator:
    """Backend interface: annotate(text) -> list of sentence dicts."""

    name = "base"

    def annotate(self, text: str) -> List[dict]:  # pragma: no cover - interface
        raise NotImplementedError


class SpacyAnnotator(Annotator):
    name = "spacy"

    def __init__(self, model: str, tasks: List[str]):
        import spacy
        self.nlp = spacy.load(model)
        self.nlp.max_length = 3_000_000
        self.tasks = tasks

    def annotate(self, text: str) -> List[dict]:
        want_pos = "pos" in self.tasks
        want_ner = "ner" in self.tasks
        doc = self.nlp(text)
        sentences: List[dict] = []
        for sent in doc.sents:
            tokens = (
                [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_} for t in sent if not t.is_space]
                if want_pos else []
            )
            entities = [{"text": e.text, "label": e.label_} for e in sent.ents] if want_ner else []
            sentences.append({"text": sent.text.strip(), "tokens": tokens, "entities": entities})
        return sentences


class StanzaAnnotator(Annotator):
    name = "stanza"

    def __init__(self, model: str, tasks: List[str]):
        import stanza
        lang = model or "en"
        processors = "tokenize"
        if "pos" in tasks:
            processors += ",pos,lemma"
        if "ner" in tasks:
            processors += ",ner"
        try:
            self.nlp = stanza.Pipeline(lang=lang, processors=processors, verbose=False)
        except Exception:
            # First use: download the model bundle once, then retry.
            stanza.download(lang, verbose=False)
            self.nlp = stanza.Pipeline(lang=lang, processors=processors, verbose=False)
        self.tasks = tasks

    def annotate(self, text: str) -> List[dict]:
        want_pos = "pos" in self.tasks
        want_ner = "ner" in self.tasks
        doc = self.nlp(text)
        sentences: List[dict] = []
        for sent in doc.sentences:
            tokens = (
                [{"text": w.text, "lemma": (w.lemma or ""), "pos": (w.upos or "")} for w in sent.words]
                if want_pos else []
            )
            entities = [{"text": e.text, "label": e.type} for e in sent.ents] if want_ner else []
            sentences.append({"text": sent.text.strip(), "tokens": tokens, "entities": entities})
        return sentences


def make_annotator(engine: str, model: str, tasks: List[str], log: Callable[[str], None]) -> Optional[Annotator]:
    try:
        if engine == "stanza":
            return StanzaAnnotator(model, tasks)
        return SpacyAnnotator(model, tasks)
    except Exception as e:
        log(f"  annotation: could not initialise '{engine}' ({e.__class__.__name__}: {e}).")
        log("  install: spaCy -> `python -m spacy download en_core_web_sm`; Stanza -> `pip install stanza`.")
        return None


def run_annotation(cfg: Config, log: Callable[[str], None]) -> None:
    src_dir = cfg.directory_text
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
    if not files:
        log("  annotation: no text files to annotate.")
        return

    tasks = list(cfg.annotation_tasks) or ["sentence", "pos", "ner"]
    annotator = make_annotator(cfg.annotation_engine, cfg.annotation_model, tasks, log)
    if annotator is None:
        return

    ann_dir = cfg.output_path("annotations")
    os.makedirs(ann_dir, exist_ok=True)
    manifest_path = cfg.output_path("annotations_manifest.json")

    old_manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
        except Exception:
            old_manifest = {}

    manifest: dict = {}
    entity_counter: Counter = Counter()
    n_new = n_reused = 0

    log(f"  annotation: engine={annotator.name}, model={cfg.annotation_model}, tasks={', '.join(tasks)}")

    for name in tqdm(files, desc="Annotating"):
        with open(os.path.join(src_dir, name), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        h = _hash(text)
        out_jsonl = os.path.join(ann_dir, name[:-4] + ".jsonl")
        prev = old_manifest.get(name)

        if prev and prev.get("hash") == h and os.path.exists(out_jsonl):
            with open(out_jsonl, "r", encoding="utf-8") as f:
                sentences = [json.loads(line) for line in f if line.strip()]
            n_reused += 1
        else:
            sentences = annotator.annotate(text)
            with open(out_jsonl, "w", encoding="utf-8") as f:
                for s in sentences:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_new += 1

        n_ent = 0
        for s in sentences:
            for e in s.get("entities", []):
                key = (str(e.get("text", "")).strip(), str(e.get("label", "")))
                if key[0]:
                    entity_counter[key] += 1
                    n_ent += 1
        manifest[name] = {"hash": h, "n_sentences": len(sentences), "n_entities": n_ent}

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(cfg.output_path("entity_counts.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "label", "count"])
        for (entity, label), count in entity_counter.most_common():
            writer.writerow([entity, label, count])

    log(
        f"  annotation: {n_new} annotated, {n_reused} reused from cache; "
        f"{len(entity_counter)} distinct entities -> annotations/*.jsonl, entity_counts.csv"
    )
