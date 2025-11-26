# pyspectral

Machine learning for processing spectral imaging.

## Getting started
- Python 3.13 with `uv` is expected. Create the env with `make create_environment`,
  activate it, then install dependencies via `make requirements`.
- Run checks with `make format`, `make lint`, `make type`, and `make test`
  (use `pytest -m "not slow"` to skip slow tests).
- Convert raw HSI `.txt` files to structured `.npz`/`.json` assets in `data/ready`
  with the interactive CLI: `uv run pyspec`. The command looks for a metadata
  CSV in `data/` (columns like `raw_path` and `presence`) and writes outputs next
  to the raw files.

## Project layout
```
Makefile           # Helper commands for env setup, linting, typing, testing
pyproject.toml     # Project metadata and tool configuration (ruff, mypy, pytest)
data/              # Project data (raw/interim/processed/external/ready, keep large files out of git)
docs/              # MkDocs site (docs/mkdocs.yml + docs/docs/*)
models/            # Trained weights and exported artifacts
notebooks/         # Experiment notebooks (n.n-initials-title.ipynb)
reports/           # Generated reports and figures
references/        # Background material and papers
stubs/             # Type stub overrides
tests/             # Pytest suite
pyspectral/        # Library code
  config.py        # Paths, logging, RNG seed, and shared constants
  core.py          # Spectral cube/flat helpers, z-scoring, and typed containers
  cli.py           # Interactive entrypoint (`uv run pyspec`) for raw-to-ready conversion
  types.py         # Numpy typing aliases
  data/            # Data ingestion, preprocessing, simulation, and I/O helpers
  modeling/        # Models, training loops, and out-of-fold utilities
  result/          # Inference, comparison, and plotting helpers
uv.lock            # Locked dependencies synced via `make requirements`
```
