"""Single source of truth for the application version.

Bumped when the analysis pipeline changes in a way that affects outputs, so the
version recorded in `run_config.json` is meaningful for reproducibility.
"""

__version__ = "0.2.0"
