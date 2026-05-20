# `output/`

Analysis results land here — one subfolder per run, named after the input
data folder (or `--name`). Each subfolder contains `summary.md`, term tables,
topic models, word clouds, networks, logs, etc.

Cross-run reports (from `compare` / `batch`) land at `output/comparison_report.md`.
Contents are gitignored and rebuilt fresh on every full pipeline run.
