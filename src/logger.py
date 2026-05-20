"""Logger factory for Geordie Miner.

Returns a small logger function that prints to stdout and appends to a log file.
Each pipeline run gets its own logger pointed at its own output directory.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable


def make_logger(log_path: str) -> Callable[[str], None]:
    """Return a logger function that writes timestamped messages to console + file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{stamp} - {message}"
        print(entry)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    return log
