"""Small text utilities shared across stages (intentionally no heavy imports)."""

from __future__ import annotations

import re

# Lone UTF-16 surrogate code points. pypdf can emit these when a PDF uses
# mathematical alphanumeric symbols (e.g. U+1D400 decomposed into a surrogate
# half like '\ud835'); Python then refuses to UTF-8-encode the string with
# "surrogates not allowed", crashing the file write.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def sanitize_text(text: str) -> str:
    """Make `text` safe to write to a UTF-8 file.

    Drops lone surrogate code points, then round-trips through UTF-8 with
    replacement so any other un-encodable code point becomes U+FFFD rather than
    raising. Returns the input unchanged when it is empty/None-ish.
    """
    if not text:
        return text
    text = _SURROGATE_RE.sub("", text)
    return text.encode("utf-8", "replace").decode("utf-8")
