# Geordie Miner

A text-mining pipeline for academic corpora. Drop a folder of PDFs (or `.txt`
files) into `data/`, tweak the config, and get back: term frequencies, TF-IDF
tables, word clouds, n-grams, co-occurrence networks (Gephi- **and VOSviewer**-ready
with an in-browser viewer), hierarchical clusters, and four flavours of topic
models (KMeans, LDA, NMF, HDP) — plus an automatically-generated `summary.md` so
you don't have to click through 30 files.

It also does **bibliometric descriptive statistics** with charts (publication
years / journals / authors), an optional **linguistic annotation** stage
(spaCy or Stanza: sentence/POS/NER), **longitudinal** analysis by publication-year
brackets, multi-seed **stability** checks, a **bulk text import** mode for large
collections of short documents, and a machine-readable **run-config** export for
reproducibility.

On top of that there's a full **bibliometric / science-mapping layer**: import
reference-manager files (**BibTeX / RIS / CSV**) or build a corpus by keyword from
**OpenAlex / Crossref / Scopus** (`fetch`); enrich metadata with provenance flags
and normalised author / institution / country tables; **citation networks**
(impact, co-citation, bibliographic coupling), **co-authorship networks**,
**dictionary concept** analysis, **topic evolution** (splits/merges over time), and
a **journal map** — all surfaced in the report and exportable for Gephi/VOSviewer +
multi-sheet Excel. Ready-made **`--profile`** presets keep runs focused.

Designed for researchers who want one command to go from a folder of papers to
a readable report of what's in them.

---

## Table of contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [Folder layout](#folder-layout)
4. [CLI reference](#cli-reference)
5. [What lands in `output/<name>/`](#what-lands-in-outputname)
6. [Configuration (`config/config.txt`)](#configuration-configconfigtxt)
7. [Comparing multiple runs](#comparing-multiple-runs)
8. [Notebook](#notebook)
9. [Troubleshooting](#troubleshooting)

---

## Install

You need **Python 3.10 or newer** and **git**. Check with:

```bash
python --version
git --version
```

If `python --version` shows `3.9` or older (or "command not found"), install
Python from https://www.python.org/downloads/ — tick *Add Python to PATH* on
Windows, then **open a new terminal** so the new `python` is on your path.
Missing git? https://git-scm.com/downloads.

### Step-by-step (Windows / PowerShell)

> 💡 **One-time setup if PowerShell blocks venv activation.** Run this once
> in PowerShell (no admin needed): `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.
> Without it, step 2 below errors with "running scripts is disabled". This is
> the single most common Windows install blocker.

```powershell
# 1. Clone
git clone https://github.com/<you>/geordie_miner.git
cd geordie_miner

# 2. Create + activate a virtual environment (keeps deps isolated from your system Python)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
python geordie_miner.py --help
```

> ⚠️ **Heads up on install size.** `requirements.txt` includes `bertopic` +
> `sentence-transformers` for the embedding-based topic model and 2D map.
> Those pull in PyTorch (~1.5–2.5 GB on disk). First install can take 5–10
> minutes on a slow connection. If you don't need BERTopic or the visual map,
> you can comment those lines out of `requirements.txt` and the rest still
> works (LDA/NMF/HDP cover the topic-modelling base case).
>
> `requirements.txt` also installs **Stanza** and the **spaCy `en_core_web_sm`**
> model (used by the optional annotation stage and the spaCy lemmatiser). Stanza
> shares PyTorch with BERTopic. If the pinned spaCy-model wheel URL is awkward in
> your environment, drop that line and run `python -m spacy download en_core_web_sm`
> instead; Stanza downloads its language model on first use.
>
> The bibliometric layer adds three small pure-Python deps (`bibtexparser`,
> `rispy`, `pycountry`) plus `requests` for the OpenAlex/Crossref/Scopus
> providers — all lightweight.

### Step-by-step (macOS / Linux)

```bash
# 1. Clone
git clone https://github.com/<you>/geordie_miner.git
cd geordie_miner

# 2. Create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
python geordie_miner.py --help
```

### Every new terminal session: re-activate the venv

Closing your terminal deactivates the virtual environment. **Each time you
open a new one**, re-activate before running anything:

```powershell
# Windows
.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
source .venv/bin/activate
```

You'll know it's active when your prompt has `(.venv)` in front. If you run
`python geordie_miner.py ...` without activating and get `ModuleNotFoundError`,
that's the cause — activate and try again.

---

## Quickstart

```bash
# 1. Create a folder for your corpus and drop PDFs / .txt files into it
mkdir data/myproject
# (copy your files into data/myproject/)

# 2. Run
python geordie_miner.py run data/myproject

# 3. Read the report
# Open output/myproject/summary.md in any markdown viewer.
```

That's it. The first run downloads NLTK resources (~50 MB) into `.cache/nltk/`;
subsequent runs reuse them. Make sure your virtual environment is active
(see [the activation reminder above](#every-new-terminal-session-re-activate-the-venv)).

### Bulk text import

For **large collections of short documents** (social-media posts, abstracts, …),
put a **single** plain-text file in the data folder. When the folder contains
exactly one `.txt` file and no PDFs, the pipeline switches to **bulk text import**:
it treats that file as a one-column document table — **one document per record,
order preserved** — instead of one-document-per-file.

```bash
mkdir data/posts            # then drop ONE file, e.g. posts.txt, inside
python geordie_miner.py run data/posts
```

Record splitting auto-detects: blank-line-separated blocks become records,
otherwise each **line** is a document (the typical posts case). Override with
`[ingest] bulk_record_split = line|blank` and force/disable the mode with
`bulk_text_mode = on|off|auto`. Empty/whitespace-only records are skipped and
counted, malformed encodings fall back through `utf-8 → utf-8-sig → cp1252` with
an explicit log line.

### Bibliographic input & external data

**Import a reference-manager file.** Drop a single `.bib`, `.ris`, or `.csv`
(e.g. a Scopus/WoS export) into the data folder — each record becomes a document
(title + abstract as the text), and its fields populate `metadata.csv`. CSV
columns are auto-mapped (common names); override with `[import] csv_mapping =
title:Article Title, year:Year, ...`.

**Build a corpus by keyword search.** No file? Pull one from an external
provider:

```bash
python geordie_miner.py fetch "virtual reality retail" --out data/vr --provider openalex --mailto you@example.com
python geordie_miner.py run data/vr --profile bibliometrics
```

- **OpenAlex** (default) and **Crossref** are free, need no key, and are used both
  by `fetch` and by metadata **enrichment** (`[providers] enrich_enable = true`),
  which fills/overrides fields by DOI or title and records the source in
  `metadata_provenance.csv`.
- **Scopus** works too if you have an Elsevier key — set `SCOPUS_API_KEY` in your
  environment and `provider = scopus`. Without a key it logs a notice and falls
  back to OpenAlex. (Google Scholar is intentionally unsupported — no official API.)
- User corrections: a `metadata_overrides.csv` (`doc_id,field,value`) in the data
  or config folder always wins, with provenance `override`.

---

## Folder layout

```
geordie_miner/
├── README.md
├── requirements.txt
├── geordie_miner.py            ← THE CLI — one file, three subcommands
│
├── src/                        ← internal modules (you usually won't touch these)
│   ├── config.py
│   ├── logger.py
│   ├── ingest.py
│   ├── preprocess.py
│   ├── terms.py
│   ├── phrases.py
│   ├── topics.py
│   ├── coherence.py
│   ├── summary.py
│   └── compare.py
│
├── config/                     ← what you'll actually edit
│   ├── config.txt              ← all tunable parameters
│   ├── stopwords.txt           ← words to drop (one per line)
│   └── substitutions.txt       ← `original,replacement` pairs (one per line)
│
├── notebooks/
│   └── explore.ipynb           ← interactive view of a finished run
│
├── data/                       ← create subfolders here for each corpus
│   ├── myproject/              ← your PDFs / .txt files
│   └── ...
│
├── output/                     ← created by the pipeline
│   ├── myproject/              ← one folder per corpus (analysis artefacts)
│   │   └── ...
│   ├── comparison_report.md    ← written by `batch` / `compare` subcommands
│   └── comparison_report.html  ← interactive version (open in browser)
│
└── .cache/                     ← auto-downloaded NLTK resources
    └── nltk/
```

`data/`, `output/`, and `.cache/` are gitignored. Your corpus and your large
analysis files don't leak into commits.

---

## CLI reference

One entry point, three subcommands:

```text
python geordie_miner.py run     DATA_DIR [DATA_DIR ...]   [--config ...] [--profile NAME] [--out ...] [--stages ...]
python geordie_miner.py batch   [DATA_DIR ...]            [--config ...] [--profile NAME] [--out ...] [--stages ...] [--no-compare] [--top N]
python geordie_miner.py compare [OUTPUT_DIR ...]          [--report PATH] [--top N]
python geordie_miner.py fetch   "KEYWORDS" --out DATA_DIR [--provider openalex|crossref|scopus] [--limit N] [--mailto EMAIL]
```

`--profile {bibliometrics,full_text,balanced}` overlays a bundled preset from
`config/profiles/` on top of `config.txt` (ignored if you pass `--config`
explicitly). `fetch` builds a corpus by keyword search from an external provider
— see [Bibliographic input & external data](#bibliographic-input--external-data).

### `run` — process one or more specific corpora

```bash
python geordie_miner.py run data/myproject
python geordie_miner.py run data/full data/no_refs data/no_method
python geordie_miner.py run data/myproject --stages topics       # rerun just the slow part
python geordie_miner.py run data/myproject --config my_config.txt
```

#### Multiple analyses on the same data: `--name`

If you run twice on the same data folder, the second run overwrites the first.
To keep both — e.g. comparing a baseline against an aggressive-K config — pass
`--name` to label each run:

```bash
python geordie_miner.py run data/myproject --name baseline
# → output/baseline/

python geordie_miner.py run data/myproject --name k20 --config config/k20.ini
# → output/k20/

python geordie_miner.py compare output/baseline output/k20
# → output/comparison_report.md
```

Without `--name`, output still lands at `output/<basename(data_dir)>/` (so
existing scripts keep working). `--name` is only valid when running a single
data directory.

### `batch` — process every `data/*` folder + write a comparison report

```bash
python geordie_miner.py batch                  # auto-discovers all data/*
python geordie_miner.py batch --top 100        # comparison at top-100 terms
python geordie_miner.py batch --no-compare     # skip the cross-run report
```

### `compare` — generate a comparison report from existing output dirs

```bash
python geordie_miner.py compare                                # all output/*
python geordie_miner.py compare output/a output/b              # explicit pair
python geordie_miner.py compare --report diff.md --top 100 output/*
```

### `--stages` for fast iteration

The pipeline runs these stages, in order:

| Stage          | What it does |
|----------------|--------------|
| `ingest`       | Convert PDFs to text; copy `.txt` files. Numbers each doc `001__…`. Surrogate-laden text is sanitised, empty/scanned PDFs are skipped. Treats a **single** `.txt` file as a **bulk document table**, and a **`.bib`/`.ris`/`.csv`** file as a [bibliographic import](#bibliographic-input--external-data). |
| `metadata`     | Bibliometric extraction per doc → `metadata.csv` (title, authors, affiliations, year, journal, volume, issue, pages, ISSN, publisher, DOI, abstract, keywords, country; blanks noted, never inferred). Merges imported files + optional **provider enrichment** + user overrides with per-field **provenance** (`metadata_provenance.csv`), and builds normalised `authors.csv` / `institutions.csv` / `countries.csv`. |
| `references`   | Parse bibliographies → citation network (`references.csv`, `citation_network.gexf` + VOSviewer). Then **impact** (in/out degree, PageRank/betweenness/eigenvector → `citation_impact.csv`, `journal_impact.csv`) and **co-citation** + **bibliographic coupling** networks for documents/journals/authors. |
| `collaboration`| *(optional)* Co-authorship networks at author / institution / country level + degree/component/community measures + a text summary → `collab_*.gexf` / `_edges.csv` / `_nodes.csv` + VOSviewer. Enable via `[collaboration] enable = true`. |
| `bibliometrics`| Publication trends (annual counts, 3-yr rolling avg, growth), ranked journals/authors/institutions/countries with impact proxies, charts, a country choropleth, and a multi-sheet `bibliometrics.xlsx` (+ `bibliometric_*.csv`). |
| `preprocess`   | Lowercase, strip URLs / parens / non-alphabetic, stopwords + substitutions, drop low-frequency tokens, collapse duplicates. Optional `remove_references`, `strip_citation_markers`, `preserve_sentences`. |
| `annotate`     | *(optional; off by default)* Linguistic annotation with a pluggable **spaCy or Stanza** back end — sentence/POS/NER → reusable `annotations/*.jsonl` + `entity_counts.csv`, hash-cached. Enable via `[annotation] enable = true`. |
| `lexical`      | *(optional)* Dictionary **concept** counting from `config/lexicons/<concept>.txt` → frequencies, co-occurrence, longitudinal shifts, and context samples (`concept_*.csv`). Enable via `[lexical] enable = true`. |
| `terms`        | Term frequencies, TF-IDF, word clouds (raw + lemmatised). |
| `phrases`      | Bigrams, trigrams, co-occurrence GEXF + **VOSviewer** network, and an **adaptive** clustering dendrogram. |
| `topics`       | KMeans + LDA + NMF + HDP at K, K·m1, K·m2, K·m3. Plus **BERTopic**. Each method toggleable (`enable_kmeans/lda/nmf/hdp/bertopic`). Plus coherence (`c_v` + `u_mass`). |
| `stability`    | Pick the most-coherent model, re-run across a **fixed seed schedule**, report coherence mean/variance + cross-seed Jaccard + a **stable/moderate/unstable** judgement. |
| `topic_evolution`| *(optional)* Per-bracket topics + prevalence + **split/merge** detection across periods → `topic_evolution.csv`, `topic_transitions.csv`, Sankey `topic_evolution.html`. Enable via `[topic_evolution] enable = true`. |
| `map`          | 2D document map (BERTopic embeddings + UMAP) → `document_map.html` (+ a journal-coloured view) + `.png`. |
| `science_map`  | *(optional)* **Journal map** (term-similarity) + thematic correspondence matrix → `journal_map.html/.png`, `thematic_evolution_matrix.png`. Enable via `[science_map] enable = true`. |
| `longitudinal` | *(optional; off by default)* Split the corpus into 5/10-year brackets and re-run terms/phrases/topics per subset → `longitudinal-<start>-<end>/` + `longitudinal_comparison.md`. Enable via `[longitudinal] enable = true`. |

Two stages — `annotate` and `longitudinal` — are in the default stage list but are
**gated by their own `enable` flag**, so a normal run is unaffected until you turn
them on in `config.txt`. Every run also writes `run_config.json` / `run_config.yaml`
(all settings, seeds, library versions, environment) for reproducibility.

When you change config and don't want to redo the whole pipeline, skip stages:

```bash
# Tweak topic count in config.txt, then re-run only topics (keeps text_processed/):
python geordie_miner.py run data/myproject --stages topics

# Just regenerate word clouds:
python geordie_miner.py run data/myproject --stages terms
```

### What gets wiped on re-run

- **Full pipeline run** (any run that includes `ingest`, i.e. the default
  command) — `output/<name>/` is wiped wholesale and rebuilt from scratch.
  No stale files from previous K values or earlier configs survive.
- **Partial-stage run** (`--stages topics`, etc.) — only the directories the
  running stages own are wiped. Skipped stages' outputs are preserved so you
  can iterate fast on one stage.
- **Different `--name`** — each name has its own output folder; runs with
  different names never touch each other. This is how you compare configs:

  ```bash
  python geordie_miner.py run data/myproject --name baseline
  python geordie_miner.py run data/myproject --name k20 --config config/k20.ini
  python geordie_miner.py compare output/baseline output/k20
  ```

If the wipe fails with a "file in use" error on Windows, close anything that
has the output open (Gephi on `network.gexf`, image viewer on a word cloud)
and rerun.

---

## What lands in `output/<name>/`

```
output/myproject/
├── summary.md                  ← read this first (plain markdown)
├── summary.html                ← interactive version (open in browser)
├── corpus_stats.txt
│
├── metadata.csv                ← title / year / DOI per paper (best-effort)
├── references.csv              ← parsed bibliographies + cross-corpus matches
├── citation_network.gexf       ← citation graph (open in Gephi)
│
├── terms_raw.csv               ← top-N terms (raw)
├── terms_raw.xlsx
├── terms_lemmatised.csv        ← top-N terms (lemmatised) — the canonical one
├── terms_lemmatised.xlsx
│
├── wordcloud_raw.jpg
├── wordcloud_lemmatised.jpg
│
├── bigrams.csv
├── trigrams.csv
│
├── cooccurrence.csv            ← every pair with count
├── network.gexf                ← co-occurrence graph (open in Gephi)
├── dendrogram.png
│
├── topics_kmeans_5.txt              ← top terms per topic, per model/K
├── topics_kmeans_5_doc2topic.txt    ← `doc_id, topic_number` per doc for this model/K
├── topics_kmeans_10.txt
├── topics_kmeans_10_doc2topic.txt
├── ...
├── topics_lda_5.txt
├── topics_lda_5_doc2topic.txt
├── ...
├── topics_nmf_5.txt
├── topics_nmf_5_doc2topic.txt
├── ...
├── topics_hdp.txt
├── topics_hdp_doc2topic.txt
├── topics_bertopic.txt              ← embedding-based topic model (if BERTopic installed)
├── topics_bertopic_doc2topic.txt
│
├── topic_assignments.csv       ← doc_id × every model/K — one big wide table
├── topic_top_docs.csv          ← top 5 most-representative docs per topic per model
├── coherence_scores.csv        ← c_v + u_mass for LDA / NMF / HDP — pick K objectively
├── topic_stability.csv         ← multi-seed LDA stability (high Jaccard = real signal)
│
├── document_map.html           ← interactive 2D scatter of every paper (UMAP)
├── document_map.png            ← static version of the same map
│
├── text/                       ← per-doc raw text
├── text_processed/             ← per-doc preprocessed text
│
└── logs/
    ├── run.log
    ├── config_used.txt         ← resolved config at runtime
    ├── stopwords_used.txt
    └── substitutions_used.txt
```

**Also written by the newer stages:**

```
├── bibliometrics_summary.csv   ← year / journal / author counts
├── bib_years.png, bib_journals.png, bib_authors.png
├── network_vosviewer_map.txt   ← VOSviewer map + network pair (co-occurrence)
├── network_vosviewer_network.txt
├── citation_network_vosviewer_map.txt / _network.txt
├── network.html                ← in-browser GEXF viewer (graceful fallback to download)
├── annotations/                ← per-doc sentence/POS/NER JSONL (if annotate ran)
├── annotations_manifest.json   ← source-text hashes for incremental re-annotation
├── entity_counts.csv           ← corpus-wide NER aggregate
├── stability_report.json/.txt  ← multi-seed stability + judgement
├── run_config.json / .yaml     ← full run configuration (reproducibility)
├── reproducibility.md          ← readable versions / config / data sources
└── longitudinal-<start>-<end>/ ← per-period analysis (+ longitudinal_comparison.md)
```

**Bibliometric / science-mapping layer** (when those stages run):

```
├── metadata_provenance.csv     ← per-field source (pdf / import / provider / override)
├── authors.csv, institutions.csv, countries.csv   ← normalised entity tables
├── bibliometrics.xlsx          ← multi-sheet; + bibliometric_*.csv per table
├── citation_impact.csv, journal_impact.csv        ← citation-based impact
├── cocitation_*.gexf, coupling_*.gexf             ← co-citation / bibliographic coupling
├── collab_authors|institutions|countries.gexf     ← co-authorship (+ _nodes/_edges.csv)
├── collaboration_summary.txt
├── concept_counts.csv, concept_cooccurrence.csv, concept_trends.csv, concept_contexts.csv
├── topic_evolution.csv, topic_transitions.csv, topic_evolution.html  ← topic splits/merges
├── journal_map.html/.png, thematic_evolution_matrix.png             ← science maps
└── imported_metadata.csv       ← parsed bibliographic input (BibTeX/RIS/CSV)
```

**Start with `summary.md`** — it stitches the most important pieces together.
Drill into the individual files when you want raw numbers.

---

## Configuration (`config/config.txt`)

```ini
[default]
language      = english                  # NLTK stopword language
max_cpu_count = 4                         # cap loky/joblib workers (silences a core-count warning)

[ingest]
bulk_text_mode    = auto                  # auto|on|off — single .txt → bulk document table
bulk_record_split = auto                  # auto|line|blank — how to split bulk records

[preprocessing]
stopwords_file    = stopwords.txt        # paths are relative to config.txt
substitutions_file = substitutions.txt
min_frequency     = 25                   # drop tokens with < N occurrences
use_spacy         = false                # true = use spaCy instead of NLTK (better, heavier)
exclude_sections  =                      # e.g. references,acknowledgements,appendix (uses section detector)
remove_references      = false           # shortcut for excluding the references section
strip_citation_markers = false           # drop [12] / (Smith, 2020) inline markers
preserve_sentences     = false           # one cleaned sentence per line in text_processed/

[term_analysis]
top_n_terms                = 200
output_wordcloud           = true        # real boolean
wordcloud_width            = 800
wordcloud_height           = 400
wordcloud_background_color = white

[phrase_analysis]
bigram_threshold       = 10
trigram_threshold      = 10
bigram_export_count    = 100
trigram_export_count   = 100
clustering_metric      = jaccard
linkage_method         = ward
dendrogram_figsize     = 10, 7
cooccurrence_threshold = 15
window_size            = 5

[topic_modelling]
# Each topic model is run at K, K·m1, K·m2, K·m3. Set a multiplier to 0 to skip.
topic_modelling_multi1 = 2
topic_modelling_multi2 = 3
topic_modelling_multi3 = 4
enable_kmeans   = true                   # turn individual methods on/off
enable_lda      = true
enable_nmf      = true
enable_hdp      = true
enable_bertopic = true

[topic_kmeans]
kmeans_topics = 5

[topic_lda]
lda_topics            = 5
lda_terms_per_topic   = 10
lda_passes            = 50
lda_per_word_topics   = 0

[topic_nmf]
nmf_topics            = 5
nmf_terms_per_topic   = 10
nmf_max_iter          = 1000

[topic_hdp]
terms_per_topic_hdp = 10

[annotation]                              # optional annotation stage (off by default)
enable            = false
annotation_engine = spacy                 # spacy | stanza
annotation_model  = en_core_web_sm        # spaCy model name, or a Stanza language code (e.g. en)
annotation_tasks  = sentence, pos, ner

[longitudinal]                            # optional per-period analysis (off by default)
enable                     = false
longitudinal_bracket_years = auto         # auto | 5 | 10
longitudinal_min_docs      = 3

[stability]
stability_seeds = 42, 123, 2024, 7, 99    # fixed, explicit seed schedule
```

The bibliometric layer adds further sections — `[import]`, `[providers]`,
`[references]`, `[collaboration]`, `[lexical]`, `[topic_evolution]`,
`[science_map]`, `[visuals]` — each documented inline in `config/config.txt`, or
just use a `--profile`. Every optional stage has an `enable` flag (off by default)
so minimal runs stay fast.

### Tuning checklist

- **Too many junk tokens** in word clouds → add to `config/stopwords.txt`.
- **Same concept appearing as different words** (e.g. *VR* vs *virtual reality*)
  → add `vr,virtualreality` and `virtual reality,virtualreality` to
  `config/substitutions.txt`.
- **Topics blur into one big bucket** → raise `kmeans_topics` / `lda_topics`,
  or check `coherence_scores.csv` to pick K objectively.
- **Topics fragment into noise** → lower K, or check `min_frequency` isn't too
  high.

---

## Comparing multiple runs

> **Which directories do I pass?** `run` and `batch` take **data** directories
> (folders of PDFs/`.txt` under `data/`). `compare` takes **output** directories
> — the finished `output/<name>/` folders, *not* the raw data:
> `python geordie_miner.py compare output/baseline output/k20`. With no arguments
> it auto-discovers every `output/*`. (`batch` runs the pipeline on `data/*` and
> then compares the resulting outputs for you.) If you hand `compare` a folder
> with no analysis artefacts, it prints a hint pointing you at `run`/`batch`.

Useful when you want to know whether removing certain sections (references,
methodology, appendix) changes your themes. Drop each variant into its own
`data/<name>/` folder, then:

```bash
python geordie_miner.py batch
```

This processes each corpus and writes both `output/comparison_report.md` (plain
text, paste-into-paper friendly) and `output/comparison_report.html`
(interactive, open in your browser — sortable tables, collapsible sections),
showing:

- Top-N terms side-by-side per run
- Pairwise Jaccard overlap between term lists (higher = more agreement)
- Terms unique to each run
- Full topic-model output dumped per run

> **Rule of thumb:** if Jaccard ≥ 0.7 across all pairs, your themes are robust
> to the variation.

You can also run `compare` standalone if you already have output dirs from a
previous batch:

```bash
python geordie_miner.py compare --top 100
```

---

## Notebooks

Two Jupyter notebooks ship with the project. They do different things — pick
the one that matches your goal:

| Notebook | Purpose |
|----------|---------|
| `notebooks/demo.ipynb` | **Run the full pipeline from inside a notebook**, with every config value inline at the top so you can tweak and re-run without touching `config/config.txt`. Each run writes to its own `output/<RUN_NAME>/`. |
| `notebooks/explore.ipynb` | **Inspect a finished run.** Doesn't execute the pipeline — auto-loads the most recently modified `output/<name>/` and renders the artefacts inline. |

### `notebooks/demo.ipynb` — runnable demo

Open it (see [How to open](#how-to-open-either-notebook) below), edit the first
code cell to set `DATA_DIR` and `RUN_NAME`, then **Run All Cells**. The notebook:

1. Writes the inline config values to a temporary config file.
2. Calls the same `run_pipeline` function the CLI uses.
3. Renders results inline as the pipeline progresses (term tables, word clouds,
   topic models, coherence scores, etc.).
4. Prints the best model/K from the coherence table at the end.

To compare two configurations, change a value (e.g. `LDA_TOPICS = 10`) and
re-run with a different `RUN_NAME`. Both runs are preserved side-by-side under
`output/`.

### `notebooks/explore.ipynb` — inspect an existing run

An interactive Jupyter view of a finished run. It auto-loads the most recently
modified `output/<name>/` and renders every artefact inline — term tables,
top phrases, word clouds, topic models, coherence scores, the dendrogram, and
network stats — so you don't have to open 30 separate files.

### When to use it instead of `summary.md`

| Use `summary.md` when… | Use `explore.ipynb` when… |
|------------------------|---------------------------|
| You want a quick read or something to paste into a paper. | You want to slice and explore the data interactively. |
| You're sharing a plain-text artefact. | You want sortable / scrollable tables. |
| You don't have Jupyter installed. | You want to add your own cells (custom queries, plots, comparisons). |

### What's in it

| Cell | Output |
|------|--------|
| 1 — Setup | Picks the most recent `output/<name>/` automatically. |
| 2 — Corpus statistics | Doc count, total words, unique terms, etc. |
| 3 — Top terms | Sortable DataFrame from `terms_lemmatised.csv`. |
| 4 — Top phrases | Bigram + trigram tables. |
| 5 — Word clouds | Both images displayed inline. |
| 6 — Topic models | Every `topics_*.txt` rendered. |
| 7 — Topic assignments | First 20 rows of `topic_assignments.csv`. |
| 8 — Top documents per topic | Example slice of `topic_top_docs.csv`. |
| 9 — Coherence scores | Full `coherence_scores.csv`. |
| 10 — Dendrogram + network | Dendrogram image + node/edge counts for `network.gexf`. |

### Inspecting a specific run

By default `explore.ipynb` picks whichever `output/<name>/` was modified most
recently. To pin a specific one, edit the first cell:

```python
RUN = "fulltext_noreference_nomethodology"   # the folder name under output/
```

Then Run All again.

### How to open either notebook

**Option A — Jupyter (browser):**

```bash
pip install jupyter
jupyter lab notebooks/                # opens both notebooks in the file browser
```

Then in the menu: **Kernel → Restart Kernel and Run All Cells**.

**Option B — VS Code:**

Just open `notebooks/demo.ipynb` or `notebooks/explore.ipynb` — VS Code's
built-in notebook editor works out of the box. Click **Run All** at the top.
First time, you may be prompted to pick a Python kernel — choose the one in
`.venv`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named '<anything>'` when running.**
Your virtual environment isn't active. Run `.venv\Scripts\Activate.ps1` on
Windows or `source .venv/bin/activate` on macOS/Linux, then try again. Your
prompt should show `(.venv)` when it's active.

**`pip install` fails to compile a package.**
Try `python -m pip install --upgrade pip` then re-run. Modern wheels work on
every common platform; if you're on something exotic (older Linux, unsupported
Python version) you may need conda-forge as a last resort:
`conda install -c conda-forge gensim scipy`.

**`Resource punkt not found` / `Resource wordnet not found`.**
The first run downloads NLTK resources to `.cache/nltk/`. If you're offline,
run once while online:
```bash
python -c "import nltk; [nltk.download(p) for p in ['stopwords','punkt','punkt_tab','wordnet']]"
```

**PDFs come out garbled or empty.**
Some PDFs are scanned images; `pypdf` can't OCR them. Pre-convert with
[`ocrmypdf`](https://github.com/ocrmypdf/OCRmyPDF) or
[`pdfplumber`](https://github.com/jsvine/pdfplumber) before dropping into
`data/<name>/`.

**The pipeline is slow — most of the time is "Stage: coherence".**
Coherence scoring (c_v + u_mass for every LDA/NMF/HDP model) is the slow part
— typically 60–80% of total runtime. It runs in parallel across CPU cores by
default, but on small machines or for quick iteration you can skip it:

```bash
python geordie_miner.py run data/myproject --no-coherence
```

You lose `coherence_scores.csv` (the objective signal for picking K) but get
the result back in a quarter of the time. Re-enable it once you've settled
on stopwords / config / preprocessing.

**LDA at high K is also slow.**
LDA scales linearly with passes and roughly with K. If `topic_modelling_multi3 = 4`
and `lda_topics = 5`, you're running LDA at K=5, 10, 15, 20. Drop one or two
multipliers to 0 in `config.txt` if you don't need that range.

**The dendrogram is an unreadable forest of labels.**
The dendrogram now **sizes itself adaptively** — figure height, font size and
margins scale with the number of leaves and the label lengths, and it's saved
with `bbox_inches="tight"` so labels aren't clipped. `dendrogram_figsize` in
`config.txt` is now a *minimum* (floor), not a cap. If it's still dense, raise
`cooccurrence_threshold` (default 15) so fewer terms enter the matrix.

**Stopwords: how do I drop a phrase like "et al"?**
Just put the phrase on its own line in `config/stopwords.txt` (e.g. `et al`).
Entries are matched longest-first, so a multi-word phrase is removed as a unit
before its constituent words.

**Why aren't "would" / "could" removed?**
NLTK's English stopword list ships `should` and the negative-contraction stems
(`couldn`, `wouldn`, …) but **omits the bare modals** `would`, `could`, `may`,
`might`, `must`, `shall`. Those are now included in the default
`config/stopwords.txt` (with a comment) — delete any you want to keep.

**Should `text_processed/` keep sentence structure?**
By default it's flattened. Set `preserve_sentences = true` to write one cleaned
sentence per line — this stops n-gram and co-occurrence windows from crossing
sentence boundaries and gives the annotation stage clean sentences. Useful when
sentence-level patterns matter; leave it off for plain bag-of-words analysis.

**A PDF crashes ingestion with a "surrogates not allowed" UTF-8 error.**
Fixed — extracted text is sanitised (lone surrogate code points like `\ud835`
from maths symbols are stripped) before writing. PDFs that yield no extractable
text (scanned images) are reported and skipped rather than written empty.

**N-gram tables still look noisy.**
Consecutive duplicate tokens are collapsed automatically (so
`metaverse metaverse metaverse` → `metaverse`). If you still see noise, check
that your `substitutions.txt` isn't introducing weird boundaries.

**`topics_bertopic.txt` and `document_map.html` are missing.**
BERTopic + sentence-transformers aren't installed. Run
`pip install -r requirements.txt` (or `pip install bertopic sentence-transformers umap-learn plotly`).
The first BERTopic run downloads the embedding model (~80 MB) into your
HuggingFace cache.

**Skip individual heavy stages.**
The new stages are independent. To skip them, use `--stages` with only what
you want, e.g.:

```bash
# Run everything except BERTopic + map + stability
python geordie_miner.py run data/myproject --stages ingest,metadata,references,preprocess,terms,phrases,topics

# Just metadata + references
python geordie_miner.py run data/myproject --stages ingest,metadata,references
```

**`metadata.csv` titles are wrong / generic.**
Title detection is a heuristic. It scans the first chunk of each `.txt`
looking for the first non-banner line. Works for ~80% of typical academic
papers; fails when the PDF→text extraction produced unusual whitespace or
when the publisher's banner uses non-standard text. Edit `metadata.csv`
manually for the bad rows.

**`references.csv` is empty or sparse.**
The references extractor needs to find a "References" / "Bibliography"
section. If your corpus has had references stripped (e.g. the
`fulltext_noreference_*` variant), there's nothing to extract — that's
correct behaviour. For other corpora, check `logs/run.log` for the
"references: extracted N references from M/N files" line — if M is low,
the section detector isn't finding bibliography headers in your papers.

**spaCy says "model not found".**
After `pip install spacy`, you also need to download the English model
once: `python -m spacy download en_core_web_sm`. Then set `use_spacy = true`
in `config/config.txt`.
