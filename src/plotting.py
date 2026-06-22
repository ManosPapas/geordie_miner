"""Single Matplotlib setup point.

Importing this module configures the headless ``Agg`` backend once and exposes
``plt``, so chart-producing modules just do ``from plotting import plt`` instead
of repeating the backend dance in every file.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after use() is required)

__all__ = ["plt"]
