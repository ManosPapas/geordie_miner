"""Text preprocessing: stopwords, substitutions, low-frequency filter, descriptive stats.

Also exposes the single `lemmatise_text` helper used across modules.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Callable, Dict, List, Tuple

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from gm_config import Config


_lemmatiser = WordNetLemmatizer()


def lemmatise_text(text: str) -> List[str]:
    """Tokenise, lowercase, drop non-alphabetic tokens, lemmatise."""
    tokens = word_tokenize(text)
    return [_lemmatiser.lemmatize(t.lower()) for t in tokens if t.isalpha()]


def load_stopwords(cfg: Config, log: Callable[[str], None]) -> List[str]:
    """Load custom stopwords from file + NLTK stopwords for the configured language."""
    custom: List[str] = []
    if os.path.exists(cfg.stopwords_file):
        with open(cfg.stopwords_file, "r", encoding="utf-8") as f:
            custom = [line.strip().lower() for line in f if line.strip()]

    nltk_list = nltk_stopwords.words(cfg.language)
    combined = custom + nltk_list

    log_path = os.path.join(cfg.directory_analysis, "_log_stopwords.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    log(f"Stopwords loaded: {len(custom)} custom + {len(nltk_list)} NLTK ({cfg.language}).")

    return combined


def load_substitutions(cfg: Config, log: Callable[[str], None]) -> Dict[str, str]:
    """Load comma-separated substitution pairs from file."""
    subs: Dict[str, str] = {}
    if not os.path.exists(cfg.substitutions_file):
        log(f"No substitutions file at '{cfg.substitutions_file}'; skipping.")
        return subs

    with open(cfg.substitutions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            original, replacement = line.split(",", 1)
            subs[original.strip()] = replacement.strip()

    log_path = os.path.join(cfg.directory_analysis, "_log_substitutions.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for k, v in subs.items():
            f.write(f"{k} -> {v}\n")
    log(f"Substitutions loaded: {len(subs)} pairs.")

    return subs


def _remove_stopwords(text: str, words: List[str]) -> str:
    if not words:
        return text
    pattern = r"\b(" + "|".join(map(re.escape, words)) + r")\b"
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _apply_substitutions(text: str, subs: Dict[str, str]) -> str:
    for original, replacement in subs.items():
        text = re.sub(rf"\b{re.escape(original)}\b[\s.,]*", replacement + "  ", text, flags=re.IGNORECASE)
    return text


def _remove_low_frequency_terms(text: str, min_frequency: int) -> str:
    if min_frequency <= 1:
        return text
    counts = Counter(text.split())
    out_lines = []
    for line in text.splitlines():
        out_lines.append(" ".join(w for w in line.split() if counts[w] >= min_frequency))
    return "\n".join(out_lines)


def preprocess_corpus(
    cfg: Config,
    stopwords: List[str],
    substitutions: Dict[str, str],
    log: Callable[[str], None],
) -> List[List[str]]:
    """Clean each .txt file in `cfg.directory_text` and write to `cfg.directory_processed`.

    Returns the corpus as a list of token lists (one per document).
    """
    corpus: List[List[str]] = []
    files = sorted(f for f in os.listdir(cfg.directory_text) if f.endswith(".txt"))

    for filename in tqdm(files, desc="Preprocessing"):
        with open(os.path.join(cfg.directory_text, filename), "r", encoding="utf-8") as f:
            text = f.read().lower()

        text = re.sub(r"-\s*\n\s*", "", text)                  # join hyphen-broken words
        text = _apply_substitutions(text, substitutions)
        text = _remove_stopwords(text, stopwords)
        text = re.sub(r"http\S+|www\.\S+", " ", text)          # links
        text = re.sub(r"\(.*?\)", " ", text)                   # parens
        text = re.sub(r"[^a-zA-Z\s]", " ", text)               # non-alphabetic
        text = re.sub(r"\b\w{1}\b", " ", text)                 # single chars
        text = _apply_substitutions(text, substitutions)       # again — picks up new boundaries
        text = _remove_stopwords(text, stopwords)              # again
        text = _remove_low_frequency_terms(text, cfg.min_frequency)

        with open(os.path.join(cfg.directory_processed, filename), "w", encoding="utf-8") as f:
            f.write(text)
        corpus.append(text.split())

    log("Preprocessing complete.")
    return corpus


def descriptive_stats(cfg: Config, directory: str, log: Callable[[str], None]) -> None:
    """Append corpus-level descriptive statistics for `directory` to analysis_corpus.txt."""
    docs: List[List[str]] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            docs.append(lemmatise_text(f.read()))

    total_docs = len(docs)
    total_words = sum(len(d) for d in docs)
    unique = len({t for d in docs for t in d})
    lengths = [len(d) for d in docs]
    mean_len = round(total_words / total_docs) if total_docs else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    out_path = os.path.join(cfg.directory_analysis, "analysis_corpus.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(
            f"\nDirectory: {directory}\n"
            f"Number of documents: {total_docs}\n"
            f"Total words in corpus: {total_words}\n"
            f"Total unique terms: {unique}\n"
            f"Mean terms per document: {mean_len}\n"
            f"Minimum terms in a document: {min_len}\n"
            f"Maximum terms in a document: {max_len}\n"
        )
    log(f"Descriptive stats appended for '{directory}'.")
