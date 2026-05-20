# Geordie Miner

A text-mining pipeline for academic corpora. Drop a folder of PDFs (or `.txt`
files) into `data/`, tweak the config, and get back: term frequencies, TF-IDF
tables, word clouds, n-grams, co-occurrence networks (Gephi-ready),
hierarchical clusters, and four flavours of topic models (KMeans, LDA, NMF,
HDP).

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/<you>/geordie_miner.git
cd geordie_miner

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put your PDFs / .txt files in a folder, e.g. data/fulltext/

# 5. Run
python geordie_miner.py data/fulltext
```

Outputs land in `output/fulltext/`. Open the `.xlsx` / `.csv` / `.png`
files directly, and the `.gexf` network in [Gephi](https://gephi.org/).

---

## Folder layout

```
geordie_miner/
├── README.md
├── requirements.txt
├── run.bat                 ← Windows wrapper, calls run_all.py
├── geordie_miner.py        ← main CLI entry point
├── compare.py              ← cross-run comparison report
├── run_all.py              ← batch runner for ./data/*
│
├── src/                    ← internal modules (don't edit these casually)
│   ├── config.py
│   ├── logger.py
│   ├── ingest.py
│   ├── preprocess.py
│   ├── terms.py
│   ├── phrases.py
│   └── topics.py
│
├── config/                 ← all the knobs you'll actually want to turn
│   ├── config.txt
│   ├── stopwords.txt
│   └── substitutions.txt
│
├── data/                   ← your input corpora (gitignored)
│   ├── fulltext/
│   ├── fulltext_no_refs/
│   └── fulltext_no_method/
│
└── output/                 ← analysis results (gitignored)
    ├── fulltext/
    ├── fulltext_no_refs/
    └── fulltext_no_method/
```

`data/` and `output/` are gitignored so your corpus (likely copyrighted) and
your large analysis files don't end up in the repo. Each `data/<name>/` you
add automatically gets a matching `output/<name>/`.

---

## Running multiple corpora

Pass several `data/...` paths in one go — each gets its own `output/<name>/`:

```bash
python geordie_miner.py data/fulltext data/fulltext_no_refs data/fulltext_no_method
```

Or auto-discover every subfolder of `data/` and write a cross-run **comparison
report**:

```bash
python run_all.py
```

That writes `comparison_report.md` at the repo root. Use it to sanity-check
that removing references / methodology doesn't shift your thematic findings.

---

## CLI reference

```text
python geordie_miner.py DATA_DIR [DATA_DIR ...]
                        [--config config/config.txt]
                        [--out output]
                        [--stages STAGES]
```

| Argument     | Default              | Meaning |
|--------------|----------------------|---------|
| `data_dir`   | (required, ≥1)       | One or more folders of PDFs / `.txt` files. |
| `--config`   | `config/config.txt`  | Path to `.ini`-style config. Stopwords / substitutions paths inside it are resolved relative to *this* file. |
| `--out`      | `output`             | Base directory for analysis output. Each run produces `<out>/<basename(data_dir)>/`. |
| `--stages`   | all                  | Comma-separated subset of: `ingest, preprocess, terms, phrases, topics`. |

`python geordie_miner.py --help` shows the same info inline.

### Skipping stages (iteration)

When tweaking config, skip the slow parts:

```bash
# Re-run only the topic models — keeps preprocessed text on disk
python geordie_miner.py data/fulltext --stages topics
```

`init_directories` is stage-aware: it only wipes folders the running stages
will rebuild, so `--stages topics` won't delete your `text_processed/`.

---

## Stages

1. **`ingest`** — copy each `.txt` file and convert each `.pdf` to text via
   [`pypdf`](https://pypi.org/project/pypdf/). Files are numbered (`001__`,
   `002__`, …) to give every document a stable id.
2. **`preprocess`** — lowercase, join hyphen-broken words, strip URLs and
   parenthesised text, drop non-alphabetic characters and single-character
   tokens, apply stopwords + substitutions (twice — catches plurals), drop
   low-frequency terms.
3. **`terms`** — top-N term frequencies, document frequency, summed TF-IDF,
   raw and lemmatised. Word clouds.
4. **`phrases`** — bigrams + trigrams, co-occurrence matrix in a configurable
   sentence window, a GEXF network for Gephi, and an agglomerative
   hierarchical clustering dendrogram.
5. **`topics`** — four topic models: KMeans, LDA, NMF, HDP. Each runs at the
   configured base `K` plus optional multiples (`K*m1`, `K*m2`, `K*m3`).

---

## Configuration (`config/config.txt`)

```ini
[default]
language = english

[preprocessing]
stopwords_file    = stopwords.txt        # path relative to config.txt
substitutions_file = substitutions.txt
min_frequency     = 25                   # drop tokens with < N occurrences

[term_analysis]
top_n_terms                = 200
output_wordcloud           = true        # real boolean (true / false)
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
# Each topic model is run at K, K*m1, K*m2, K*m3 — set a multiplier to 0 to skip.
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

---

## What lands in `output/<name>/`

| File | Content |
|------|---------|
| `_log_output.log`            | Full timestamped run log. |
| `_log_config.log`            | Resolved configuration snapshot. |
| `_log_stopwords.txt`         | Effective stopword list (custom + NLTK). |
| `_log_substitutions.txt`     | Applied substitution pairs. |
| `text/`                      | Per-document text after PDF extraction. |
| `text_processed/`            | Per-document text after preprocessing. |
| `analysis_corpus.txt`        | Corpus-level descriptive stats. |
| `analysis_terms_single_raw.{csv,xlsx}`        | Top-N raw term table. |
| `analysis_terms_single_lemmatised.{csv,xlsx}` | Top-N lemmatised term table. |
| `analysis_descriptive_*_wordcloud.jpg`        | Word cloud images. |
| `analysis_terms_ngram2.csv`  | Top bigrams. |
| `analysis_terms_ngram3.csv`  | Top trigrams. |
| `analysis_cooccurrence.csv`  | Pairwise co-occurrence counts. |
| `analysis_cooccurrence_network.gexf` | Network for Gephi. |
| `analysis_ahc_dendrogram_jaccard.png` | Hierarchical-clustering dendrogram. |
| `analysis_topicmodel_KMEANS_cluster_centroids_<K>.txt` | KMeans centroids. |
| `analysis_topicmodel_KMeans_<K>_doc2topic_assignments.txt` | KMeans doc→cluster. |
| `analysis_topicmodel_LDA_<K>.txt`             | LDA topic words. |
| `analysis_topicmodel_LDA_<K>_doc2topic_assignments.txt`   | LDA doc→topic. |
| `analysis_topicmodel_NMF_<K>.txt`             | NMF topic words. |
| `analysis_topicmodel_nmf_<K>_doc2topic_assignments.txt`   | NMF doc→topic. |
| `analysis_topicmodel_HDP.txt`                 | HDP topic words. |
| `analysis_topicmodel_HDP_doc2topic_assignments.txt`       | HDP doc→topic. |

---

## Comparison report

```bash
python compare.py                              # auto-discovers ./output/*
python compare.py output/a output/b output/c   # explicit list
python compare.py --out diff.md --top 100 output/*
```

Writes a markdown file with:

- Top-N terms in each run, side-by-side
- Pairwise Jaccard overlap between term lists
- Terms unique to each run
- Topic-model output dumped for each run

This is the research artefact when comparing variants (e.g. "do references
distort my themes?").

---

## Troubleshooting

- **`gensim` / `scipy` fail to install on Windows.** Use conda for those:
  `conda install -c conda-forge gensim scipy`, then `pip install -r requirements.txt`
  for the rest.
- **`Resource punkt not found`.** The first run downloads NLTK resources into
  `./_nltk_data/`. If you're offline, run once while online:
  `python -c "import nltk; [nltk.download(p) for p in ['stopwords','punkt','punkt_tab','wordnet']]"`
- **PDFs come out garbled.** Some PDFs are scanned images; `pypdf` can't OCR
  them. Pre-convert with [`ocrmypdf`](https://github.com/ocrmypdf/OCRmyPDF)
  or [`pdfplumber`](https://github.com/jsvine/pdfplumber) before dropping into
  `data/`.
- **N-gram output is full of `metaverse metaverse`.** When substitutions
  collapse `virtual reality` → `virtualreality` and stopwords between repeated
  mentions get stripped, you get duplicate-token n-grams. A future improvement
  is to collapse consecutive duplicate tokens during preprocessing.

---

## Notes for users of the older version

- The wrapper `py-text-mining/` folder is gone; the repo root *is* the project.
- Output dirs moved from `analyis_<dataset>/` (typo) → `output/<dataset>/`.
- Config moved from a flat file to `config/config.txt`.
- CLI signature changed: data dirs are positional, config is `--config`:
  - **Old:** `python geordie_miner.py config.txt data_fulltext`
  - **New:** `python geordie_miner.py data/fulltext`
- The lemmatised word cloud is now built from lemmatised counts (was a bug).
- `output_wordcloud` is a real boolean — `false` actually skips it.
- The unused `enable_lemmatization` key has been removed.
