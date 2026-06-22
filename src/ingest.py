"""Data ingestion: NLTK resource setup + PDF/text → numbered .txt files.

Two modes:
- **Standard** (default): one input file → one document. PDFs are converted to
  text with pypdf; .txt files are copied.
- **Bulk text import**: a SINGLE plain-text file is treated as a document table
  (one short document per record). Intended for large collections of short
  documents such as social-media posts or abstracts — thousands of them.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Callable, List, Tuple

import nltk
from pypdf import PdfReader
from tqdm import tqdm

from config import Config
from textutil import sanitize_text


NLTK_CACHE_DIR = os.path.join(".cache", "nltk")

# Encodings tried, in order, when decoding a bulk text file.
_DECODE_CHAIN = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def ensure_nltk_resources(log: Callable[[str], None]) -> None:
    """Download required NLTK corpora into ./.cache/nltk if not already present."""
    nltk_data_dir = os.path.join(os.getcwd(), NLTK_CACHE_DIR)
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


def _list_inputs(data_dir: str) -> Tuple[List[str], List[str]]:
    """Return (txt_files, pdf_files) sorted, for `data_dir`."""
    files = sorted(os.listdir(data_dir))
    txts = [f for f in files if f.lower().endswith(".txt")]
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    return txts, pdfs


def _should_bulk_import(cfg: Config, log: Callable[[str], None]) -> str | None:
    """Decide whether to use bulk text import. Returns the single .txt file name
    to bulk-import, or None for standard ingestion."""
    txts, pdfs = _list_inputs(cfg.directory_data)
    mode = (cfg.bulk_text_mode or "auto").lower()

    if mode == "off":
        return None
    if len(txts) != 1:
        if mode == "on":
            log(
                f"  bulk_text_mode=on but the input has {len(txts)} .txt file(s) "
                f"(need exactly 1) — falling back to standard ingestion."
            )
        return None
    if mode == "auto" and pdfs:
        return None  # mixed corpus — treat as standard
    if mode == "on" and pdfs:
        log(f"  bulk_text_mode=on — ignoring {len(pdfs)} PDF(s); importing the single .txt as a document table.")
    return txts[0]


def ingest_corpus(cfg: Config, log: Callable[[str], None]) -> int:
    """Top-level ingestion dispatcher. Returns number of documents written.

    Priority: bibliographic file (.bib/.ris, or a sole .csv) -> bulk single .txt ->
    standard per-file PDF/txt conversion.
    """
    from bibimport import find_bib_file, run_bibliographic_import

    txts, pdfs = _list_inputs(cfg.directory_data)
    bib_path, kind = find_bib_file(cfg.directory_data)
    # .bib/.ris are unambiguously bibliographic; a .csv only when it's the sole
    # corpus file (so a stray metadata CSV next to PDFs doesn't hijack ingestion).
    if bib_path is not None and not pdfs and (kind in ("bibtex", "ris") or len(txts) == 0):
        return run_bibliographic_import(cfg, bib_path, kind, log)

    bulk_file = _should_bulk_import(cfg, log)
    if bulk_file is not None:
        return bulk_text_import(
            os.path.join(cfg.directory_data, bulk_file),
            cfg.directory_text,
            cfg.bulk_record_split,
            log,
        )
    return convert_pdfs_to_text(cfg.directory_data, cfg.directory_text, log)


def convert_pdfs_to_text(data_dir: str, text_dir: str, log: Callable[[str], None]) -> int:
    """Copy .txt files and convert .pdf files in `data_dir` into numbered .txt files in `text_dir`.

    Returns the number of files successfully written. Surrogate-laden PDF text is
    sanitised before writing; PDFs that yield no extractable text (e.g. scanned
    images) are reported and skipped rather than written as empty documents.
    """
    files = sorted(os.listdir(data_dir))
    n_txt = n_pdf = n_skip = n_empty = 0

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
                text = sanitize_text("\n".join(pages))
                if not text.strip():
                    log(f"  '{file_name}': no extractable text (likely a scanned/image PDF) — skipped.")
                    n_empty += 1
                    continue
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(text)
                n_pdf += 1
            except Exception as e:
                log(f"  failed to convert '{file_name}': {e}")
                n_skip += 1
        else:
            n_skip += 1

    log(
        f"Ingestion complete: {n_txt} .txt copied, {n_pdf} .pdf converted, "
        f"{n_empty} empty/unreadable skipped, {n_skip} other skipped."
    )
    return n_txt + n_pdf


def _decode_bulk_file(path: str, log: Callable[[str], None]) -> str:
    """Read a bulk text file, trying a chain of encodings. Logs the one used."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in _DECODE_CHAIN:
        try:
            text = raw.decode(enc)
            if enc != "utf-8":
                log(f"  bulk import: decoded with '{enc}' (not valid UTF-8) — check for mojibake.")
            return text
        except UnicodeDecodeError:
            continue
    # Last resort: never crash on a bad byte.
    log("  bulk import: file is not decodable in any tried encoding — decoding UTF-8 with replacement.")
    return raw.decode("utf-8", "replace")


def _split_records(text: str, split: str, log: Callable[[str], None]) -> List[str]:
    """Split bulk text into raw records per the chosen strategy."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    split = (split or "auto").lower()

    if split == "line":
        return text.split("\n")
    if split == "blank":
        return re.split(r"\n\s*\n", text)

    # auto: prefer blank-line-separated blocks when records clearly span multiple
    # lines; otherwise treat each line as a document (the social-media case).
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    multiline = sum(1 for b in blocks if b.strip().count("\n") >= 1)
    if len(blocks) >= 2 and multiline >= max(1, len(blocks) // 5):
        log("  bulk import: auto-detected blank-line-separated records.")
        return blocks
    log("  bulk import: auto-detected one document per line.")
    return text.split("\n")


def bulk_text_import(
    src_path: str,
    text_dir: str,
    split: str,
    log: Callable[[str], None],
) -> int:
    """Treat a single text file as a document table: one document per record.

    Preserves record order, validates empty/malformed rows and encoding issues,
    and writes numbered `NNNNN__bulk.txt` files into `text_dir`. Returns the
    number of documents written. Suitable for thousands of short documents
    (social-media posts, abstracts, ...).
    """
    log(f"Bulk text import mode — treating '{os.path.basename(src_path)}' as a document table")
    log("  (one short document per record — intended for large collections of short texts).")

    text = sanitize_text(_decode_bulk_file(src_path, log))
    records = _split_records(text, split, log)

    width = max(3, len(str(len(records))))
    written = 0
    n_empty = 0
    n_short = 0  # 1-character rows etc. — kept, but counted as a heads-up

    for record in tqdm(records, desc="Importing records"):
        body = record.strip()
        if not body:
            n_empty += 1
            continue
        if len(body) < 2:
            n_short += 1
        dst = os.path.join(text_dir, f"{written + 1:0{width}d}__bulk.txt")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(body)
        written += 1

    if written == 0:
        log(
            f"  ERROR: bulk import produced 0 documents from {len(records)} record(s) "
            f"({n_empty} empty). Check the file and `bulk_record_split`."
        )
    else:
        log(
            f"  bulk import complete: {written} document(s) written, "
            f"{n_empty} empty record(s) skipped"
            + (f", {n_short} very short (<2 char) record(s) flagged" if n_short else "")
            + "."
        )
    return written
