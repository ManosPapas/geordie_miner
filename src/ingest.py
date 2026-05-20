"""Data ingestion: NLTK resource setup + PDF/text → numbered .txt files."""

from __future__ import annotations

import os
import shutil
from typing import Callable

import nltk
from pypdf import PdfReader
from tqdm import tqdm


def ensure_nltk_resources(log: Callable[[str], None]) -> None:
    """Download required NLTK corpora into ./_nltk_data if not already present."""
    nltk_data_dir = os.path.join(os.getcwd(), "_nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    log("NLTK setup:")
    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "wordnet": "corpora/wordnet",
    }
    for name, locator in resources.items():
        try:
            nltk.data.find(locator)
            log(f"  exists: {name}")
        except LookupError:
            log(f"  downloading: {name}")
            nltk.download(name, download_dir=nltk_data_dir, quiet=True)
    log("NLTK setup complete.")


def convert_pdfs_to_text(data_dir: str, text_dir: str, log: Callable[[str], None]) -> int:
    """Copy .txt files and convert .pdf files in `data_dir` into numbered .txt files in `text_dir`.

    Returns the number of files successfully written.
    """
    files = sorted(os.listdir(data_dir))
    n_txt = n_pdf = n_skip = 0

    for index, file_name in enumerate(tqdm(files, desc="Ingesting files")):
        src = os.path.join(data_dir, file_name)
        prefix = f"{index + 1:03d}__"
        ext = file_name.lower().rsplit(".", 1)[-1]

        if ext == "txt":
            dst = os.path.join(text_dir, prefix + file_name)
            try:
                shutil.copy(src, dst)
                n_txt += 1
            except Exception as e:
                log(f"  failed to copy '{file_name}': {e}")
                n_skip += 1

        elif ext == "pdf":
            dst = os.path.join(text_dir, prefix + file_name[:-4] + ".txt")
            try:
                reader = PdfReader(src)
                pages = [page.extract_text() or "" for page in reader.pages]
                with open(dst, "w", encoding="utf-8") as f:
                    f.write("\n".join(pages))
                n_pdf += 1
            except Exception as e:
                log(f"  failed to convert '{file_name}': {e}")
                n_skip += 1
        else:
            n_skip += 1

    log(f"Ingestion complete: {n_txt} .txt copied, {n_pdf} .pdf converted, {n_skip} skipped.")
    return n_txt + n_pdf
