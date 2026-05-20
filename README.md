# Geordie Miner

A text-mining pipeline for academic corpora. Drop a folder of PDFs (or `.txt`
files) into `data/`, tweak the config, and get back: term frequencies, TF-IDF
tables, word clouds, n-grams, co-occurrence networks (Gephi-ready),
hierarchical clusters, and four flavours of topic models (KMeans, LDA, NMF,
HDP) — plus an automatically-generated `summary.md` so you don't have to click
through 30 files.

Designed for researchers who want one command to go from a folder of papers to
a readable report of what's in them.

---

## Table of contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [Folder layout](#folder-layout)
4. [CLI reference](#cli-reference)
5. [What lands in `output/<name>/`](#what-lands-in-outputname)
6. [Configuration (`config/config.ini`)](#configuration-configconfigini)
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
│   ├── config.ini              ← all tunable parameters
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
│   └── comparison_report.md    ← written by `batch` / `compare` subcommands
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
python geordie_miner.py run     DATA_DIR [DATA_DIR ...]   [--config ...] [--out ...] [--stages ...]
python geordie_miner.py batch   [DATA_DIR ...]            [--config ...] [--out ...] [--stages ...] [--no-compare] [--top N]
python geordie_miner.py compare [OUTPUT_DIR ...]          [--report PATH] [--top N]
```

### `run` — process one or more specific corpora

```bash
python geordie_miner.py run data/myproject
python geordie_miner.py run data/full data/no_refs data/no_method
python geordie_miner.py run data/myproject --stages topics       # rerun just the slow part
python geordie_miner.py run data/myproject --config my_config.ini
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

The pipeline has five stages, run in order:

| Stage        | What it does |
|--------------|--------------|
| `ingest`     | Convert PDFs to text; copy `.txt` files. Numbers each doc `001__…`. |
| `preprocess` | Lowercase, strip URLs/parens/non-alphabetic, apply stopwords + substitutions, drop low-frequency tokens, collapse consecutive duplicates. |
| `terms`      | Term frequencies, TF-IDF, word clouds (raw + lemmatised). |
| `phrases`    | Bigrams, trigrams, co-occurrence matrix, GEXF network, hierarchical clustering dendrogram. |
| `topics`     | KMeans + LDA + NMF + HDP at K, K·m1, K·m2, K·m3. Plus coherence scores. |

When you change config and don't want to redo the whole pipeline, skip stages:

```bash
# Tweak topic count in config.ini, then re-run only topics (keeps text_processed/):
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
├── summary.md                  ← read this first
├── corpus_stats.txt
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
├── network.gexf                ← open in Gephi
├── dendrogram.png
│
├── topics_kmeans_5.txt         ← top terms per topic, per model/K
├── topics_kmeans_10.txt
├── topics_kmeans_15.txt
├── topics_kmeans_20.txt
├── topics_lda_5.txt
├── topics_lda_10.txt
├── ...
├── topics_nmf_5.txt
├── ...
├── topics_hdp.txt
│
├── topic_assignments.csv       ← doc_id × every model/K — one big table
├── topic_top_docs.csv          ← top 5 docs per topic per model
├── coherence_scores.csv        ← c_v + u_mass for LDA / NMF / HDP — pick K objectively
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

**Start with `summary.md`** — it stitches the most important pieces together.
Drill into the individual files when you want raw numbers.

---

## Configuration (`config/config.ini`)

```ini
[default]
language = english                       # NLTK stopword language

[preprocessing]
stopwords_file    = stopwords.txt        # paths are relative to config.ini
substitutions_file = substitutions.txt
min_frequency     = 25                   # drop tokens with < N occurrences

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
```

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

Useful when you want to know whether removing certain sections (references,
methodology, appendix) changes your themes. Drop each variant into its own
`data/<name>/` folder, then:

```bash
python geordie_miner.py batch
```

This processes each corpus and writes `output/comparison_report.md`, showing:

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

## Notebook (`notebooks/explore.ipynb`)

### What it is

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

### How to open it

**Option A — Jupyter (browser):**

```bash
pip install jupyter
jupyter lab notebooks/explore.ipynb
```

Then in the menu: **Kernel → Restart Kernel and Run All Cells**.

**Option B — VS Code:**

Just open `notebooks/explore.ipynb` — VS Code's built-in notebook editor
works out of the box. Click **Run All** at the top.

### Inspecting a specific run

By default the notebook picks whichever `output/<name>/` was modified most
recently. To pin a specific one, edit the first cell:

```python
RUN = "fulltext_noreference_nomethodology"   # the folder name under output/
```

Then Run All again.

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

**LDA at high K is very slow.**
LDA scales linearly with the number of passes and quadratically-ish with K.
If `topic_modelling_multi3 = 4` and `lda_topics = 5`, you're running LDA at
K=5, 10, 15, 20 which can take 10–20 minutes on a few-hundred-paper corpus.
Drop one or two multipliers to 0 in `config.ini`, or use `--stages topics`
to iterate.

**The dendrogram is an unreadable forest of labels.**
Raise `cooccurrence_threshold` in `config.ini` (default 15) so fewer terms
make it into the matrix. Or raise `dendrogram_figsize` (default `10, 7`).

**N-gram tables still look noisy.**
Consecutive duplicate tokens are collapsed automatically (so
`metaverse metaverse metaverse` → `metaverse`). If you still see noise, check
that your `substitutions.txt` isn't introducing weird boundaries.
