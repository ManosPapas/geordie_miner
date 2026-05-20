# Geordie Miner

A text-mining pipeline for academic corpora. Drop in a folder of PDFs (or `.txt`
files), tweak the config, and get back: term frequencies, TF-IDF tables, word
clouds, n-grams, co-occurrence networks (Gephi-ready), hierarchical clusters,
and four flavours of topic models (KMeans, LDA, NMF, HDP).

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/<you>/geordie_miner.git
cd geordie_miner/py-text-mining

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put your PDFs / .txt files in a folder, e.g. ./data_fulltext

# 5. Run
python geordie_miner.py config.txt data_fulltext
```

Outputs land in `analysis_data_fulltext/`. Open the `.xlsx` / `.csv` / `.png`
files directly, and the `.gexf` network in [Gephi](https://gephi.org/).

---

## Running multiple corpora

Pass several data directories on one line — each gets its own
`analysis_<name>/` folder:

```bash
python geordie_miner.py config.txt data_fulltext data_no_refs data_no_method
```

Or auto-discover every `data_*` folder and produce a cross-run **comparison
report**:

```bash
python run_all.py
```

That writes `comparison_report.md` next to your analysis folders. Use it to
sanity-check that removing references / methodology sections doesn't shift the
thematic findings.

---

## Project layout

| File | What it does |
|------|--------------|
| `geordie_miner.py` | CLI entry point. Orchestrates the stages below. |
| `gm_config.py`     | Parses `config.txt` into a typed `Config`. |
| `gm_logging.py`    | Timestamped logger writing to console + file. |
| `gm_ingest.py`     | PDF→text conversion, NLTK resource download. |
| `gm_preprocess.py` | Stopwords, substitutions, low-frequency filter. |
| `gm_terms.py`      | Frequency / TF-IDF tables + word clouds. |
| `gm_phrases.py`    | Bigrams, trigrams, co-occurrence, GEXF, dendrogram. |
| `gm_topics.py`     | KMeans + LDA + NMF + HDP topic models. |
| `compare.py`       | Cross-run comparison report (markdown). |
| `run_all.py`       | Batch runner across all `data_*` directories. |
| `config.txt`       | All tunable parameters. |
| `stopwords.txt`    | Custom stopwords (one per line). |
| `substitutions.txt`| `original,replacement` pairs (one per line). |

Beginners can read each `gm_*.py` top-to-bottom — they're short and
single-purpose. Production users can `import` any of them.

---

## CLI reference

```text
python geordie_miner.py [config] [data_dir ...] [--stages STAGES]
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `config`     | `config.txt` | Path to `.ini`-style config file. |
| `data_dir`   | `data`       | One or more folders of PDFs / `.txt` files. |
| `--stages`   | all          | Comma-separated subset of: `ingest, preprocess, terms, phrases, topics`. |

`python geordie_miner.py --help` shows the same info inline.

### Skipping stages

Useful when iterating on config:

```bash
# Re-run only the topic-modelling step (preprocessed files are kept)
python geordie_miner.py config.txt data_fulltext --stages topics
```

---

## Stages

1. **`ingest`** — copy each `.txt` file and convert each `.pdf` to text via
   [`pypdf`](https://pypi.org/project/pypdf/). Files are numbered (`001__`,
   `002__`, …) to give every document a stable id.
2. **`preprocess`** — lowercase, join hyphen-broken words, strip URLs and
   parenthesised text, drop non-alphabetic characters and single-character
   tokens, apply stopwords + substitutions (twice, so plurals are caught), drop
   low-frequency terms.
3. **`terms`** — top-N term frequencies, document frequency, summed TF-IDF,
   raw and lemmatised. Word clouds.
4. **`phrases`** — bigrams + trigrams, co-occurrence matrix in a configurable
   sentence window, a GEXF network for Gephi, and an agglomerative
   hierarchical clustering dendrogram.
5. **`topics`** — four topic models: KMeans, LDA, NMF, HDP. Each runs at the
   configured base `K`, plus optional multiples (`K*m1`, `K*m2`, `K*m3`).

---

## Configuration reference (`config.txt`)

```ini
[default]
language = english                    # NLTK stopword language

[preprocessing]
stopwords_file    = stopwords.txt
substitutions_file = substitutions.txt
min_frequency     = 25                # drop tokens appearing < N times

[term_analysis]
top_n_terms                = 200
output_wordcloud           = true     # generate raw + lemmatised word clouds
wordcloud_width            = 800
wordcloud_height           = 400
wordcloud_background_color = white

[phrase_analysis]
bigram_threshold       = 10           # min frequency to be exported
trigram_threshold      = 10
bigram_export_count    = 100
trigram_export_count   = 100
clustering_metric      = jaccard      # any scipy pdist metric
linkage_method         = ward
dendrogram_figsize     = 10, 7
cooccurrence_threshold = 15           # min edge weight in the network
window_size            = 5            # sentence-local co-occurrence window

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

## Output files

Every run creates a `analysis_<dataset>/` directory containing:

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
| `analysis_topicmodel_LDA_<K>.txt`         | LDA topic words. |
| `analysis_topicmodel_LDA_<K>_doc2topic_assignments.txt` | LDA doc→topic. |
| `analysis_topicmodel_NMF_<K>.txt`         | NMF topic words. |
| `analysis_topicmodel_nmf_<K>_doc2topic_assignments.txt` | NMF doc→topic. |
| `analysis_topicmodel_HDP.txt`             | HDP topic words. |
| `analysis_topicmodel_HDP_doc2topic_assignments.txt`     | HDP doc→topic. |

---

## Comparison report

```bash
python compare.py                                   # auto-discovers ./analysis_*
python compare.py analysis_a analysis_b analysis_c  # explicit list
python compare.py --out diff.md --top 100 analysis_*
```

Writes a markdown file with:

- Top-N terms in each run, side-by-side
- Pairwise Jaccard overlap between term lists
- Terms unique to each run
- Topic-model output dumped for each run

This is the actual research artefact when you want to check whether removing
references / methodology / etc. changes your thematic findings.

---

## Troubleshooting

- **`gensim` / `scipy` fail to install on Windows.** Use conda for those:
  `conda install -c conda-forge gensim scipy`, then `pip install -r requirements.txt`
  for the rest.
- **`Resource punkt not found`.** The first run downloads NLTK resources into
  `./_nltk_data/`. If you're offline, run
  `python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"`
  while online once.
- **PDFs come out garbled.** Some PDFs are scanned images; `pypdf` can't OCR
  them. Pre-convert to text with [`pdfplumber`](https://github.com/jsvine/pdfplumber)
  or an OCR tool (e.g. `ocrmypdf`) before dropping into `data_*`.

---

## Notes for users coming from the original script

- Output directory is now spelled `analysis_*` (was `analyis_*`).
- The lemmatised word cloud is now generated from lemmatised counts (was
  identical to the raw one due to a bug).
- `output_wordcloud` in `config.txt` is now parsed as a real boolean — set
  `false` to actually skip word clouds.
- The unused `enable_lemmatization` key has been removed; lemmatisation is
  always applied in the term / phrase / topic stages.
- `run.bat` (Windows-only) is kept as a one-line wrapper, but `python run_all.py`
  is the cross-platform path going forward.
