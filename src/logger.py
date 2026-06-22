"""Logger factory for Geordie Miner.

Returns a small logger function that prints to stdout and appends to a log file.
Each pipeline run gets its own logger pointed at its own output directory.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable


def make_logger(log_path: str) -> Callable[[str], None]:
    """Return a logger function that writes timestamped messages to console + file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{stamp} - {message}"
        try:
            print(entry)
        except UnicodeEncodeError:
            # A non-ASCII char (em-dash, ≥, …) in the message must never crash the
            # run on a strict cp1252 Windows console — degrade to a safe rendering.
            enc = (sys.stdout.encoding or "utf-8")
            print(entry.encode(enc, "replace").decode(enc, "replace"))
        # The log FILE is always clean UTF-8.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    return log


# ASCII only — a Unicode rule (─) would crash print() on a cp1252 Windows console.
_STAGE_RULE = "-" * 60


def log_stage(log: Callable[[str], None], name: str) -> None:
    """Emit a blank line + a horizontal rule + the stage banner.

    Keeps consecutive stages visually separated in both the console and run.log.
    """
    log("")
    log(_STAGE_RULE)
    log(f"--- Stage: {name} ---")
