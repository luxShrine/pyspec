# pyspectral

Machine learning for processing spectral imaging. The library targets pixel-wise
(and secondarily scene-level) classification of hyperspectral Raman maps.
Specifically, this was designed to detect the presence of alginate in a scene,
and was built to compare a neural-network classifier against a PCA + SVM baseline under the same
preprocessing and cross-validation protocol.

## Background

Given a set of Raman maps, the question is how best to detect whether alginate
is present in each pixel of each map. The well-understood baseline is
dimensionality reduction via PCA followed by an SVM classifier. PCA optimizes
for variance in `X`, not for structure in `(X, y)`, so it can discard
low-variance information that is physically relevant for the label. The
hypothesis behind the ML side of the library is that a small neural network,
trained directly on physically motivated spectral features, can recover that
information and supplement, or potentially outperform, the PCA+SVM pipeline.

`pyspectral` implements both pipelines end-to-end so they can be evaluated on
the same scenes, same folds, and same held-out test split.

## Getting started

Python 3.13 and [`uv`](https://docs.astral.sh/uv/) are expected. The repo uses
[`just`](https://github.com/casey/just) as a task runner.

```bash
# Create the environment and install dependencies
uv venv --python 3.13
uv sync

# Format, lint, type-check, test
just format        # ruff check --fix && ruff format
just lint          # ruff format --check && ruff check
just type          # uv run mypy pyspectral
just test          # uv run pytest tests

# Subsets of the test suite
uv run pytest tests -m "not slow"
uv run pytest tests -m unit
uv run pytest tests/test_features.py

# Interactive marimo notebooks
just marimo
```

### Converting raw HSI files

`uv run pyspec` launches an interactive CLI that converts raw HSI `.txt` files
into the structured `.npz` + `.json` pairs the rest of the library consumes.
It looks for a metadata CSV in `data/` (columns like `raw_path` and
`presence`), prompts for the file to use, and writes converted assets next to
the raw files. The same CSV is later used by `build_classification()` to
assemble a `ClassPair` for training.

## Pipeline overview

The high-level data flow is:

```
raw HSI .txt  ──►  preprocessing 
                      │
                      ▼
                      features  ──►  folds  ──►  train
                                                  │
                                                  ▼
                                                  predictions + metrics
```

- **Ingestion (`pyspectral/data/io.py`).** `HSIMap` holds wavelengths,
  coordinates, spectra, the reshaped cube, and presence labels for a single
  scene. `DataArtifacts` bundles per-scene metadata. `build_classification()`
  assembles a `ClassPair` (flattened spectra + artifacts) from a CSV manifest,
  and `save_outer` / `restore_arrays` handle JSON + NPZ round-tripping.
- **Preprocessing (`pyspectral/data/preprocessing.py`).** `PreConfig` composes
  the per-spectrum pipeline: spike removal -> polynomial or ALS baseline
  subtraction -> Savitzky–Golay smoothing -> peak normalization.
  `preprocess_cube()` runs the whole sequence and emits per-step stats;
  `SameGridCubes` resamples a collection of scenes to a shared wavelength axis.
- **Features (`pyspectral/data/features.py`).** Three spectral windows
  (low / mid / high) defined by `RegionSet`. `PresenceWindows` reduces each
  window to a 3-element band ratio; `FullPresenceWindows` returns the full
  per-window intensity spectra. 
  `create_specband_feats()` extracts peak heights, FWHM, and log-ratio
  features used by both the ML and the PCA+SVM pipelines.
- **Datasets and folds (`pyspectral/data/{dataset,shared}.py`).**
  `PixelSpectraDataset` and `SceneSpectralDataset` are PyTorch datasets.
  `KFolds` performs **scene-aware** K-fold splitting - every pixel of a scene
  is either fully in train or fully in validation, so no spatial leakage can
  occur. `apply_scaler()` fits a `StandardScaler` on the training pixels of
  each fold and applies it to the matching validation / test splits.
- **Models (`pyspectral/modeling/models.py`).** Three architectures, selected
  by the `ClassModelType` enum:
  - `CONV` - small 2D CNN over spectral tiles.
  - `MIL` - multiple-instance-learning head with mean pooling over a scene.
  - `MILMULTI` - MIL variant with a multi-class head.
  `get_model()` instantiates the chosen architecture.
- **Training (`pyspectral/modeling/train.py`).** `train_pixel()` and
  `train_scene()` are the two training loops. `pick_device()` auto-selects
  CUDA / MPS / CPU. Ragged scene batches are handled by custom collates.
- **Out-of-fold tracking (`pyspectral/modeling/oof.py`).** `FoldLoss`,
  `EpochLosses`, `PxlStats`, and `SceneStats` aggregate per-fold pixel-level
  and scene-level statistics across the CV run.
- **Baseline + evaluation (`pyspectral/result/`).**
  - `class_ml.py` - `create_svm_pipeline()` builds a
    `StandardScaler -> PCA -> SVC` baseline pipeline.
  - `class_trad.py` - `svm_class_predict_pixel()` evaluates a fitted SVM on a
    new scene and returns a `PredCompare`. Assumes the SVC was trained with
    binary `{0, 1}` labels.
  - `predict.py` - `MaskedValues` separates positive / negative / maybe
    predictions; `PredCompare` pairs ground truth with predictions and computes
    IoU.
  - `compare.py` - `_check_prediction_polarity()` catches the common failure
    mode where the model has learned an inverted target.
  - `plots.py` - boundary generation (Otsu, hysteresis), confusion matrices,
    and loss curves.

## Evaluation protocol

The repository implements the same protocol for both the ML and PCA+SVM paths.

**Scene-level cross-validation.** Scenes are stratified by approximate positive
rate (`mean(presence) > 0.5`). Folds are constructed so that every pixel of a
scene lands entirely in train or entirely in validation. This keeps pixels
identifiable by scene for debugging and prevents spatial leakage. The scikit
`Pipeline` does the same separation on the PCA+SVM side.

**Held-out test scene.** One full scene is held back from all folds and is
only touched after CV is complete. Roughly 98% of scenes go to CV
(train + eval); about 2% (a single scene at current dataset size) is reserved
as the unseen test set.

**Label encoding.** Presence is encoded ternary: `{0.0, 0.5, 1.0}` for
negative / uncertain / positive. The legacy integer encoding `{0, 1, 2}` is
still handled in a few code paths. For IoU and confusion-matrix scoring, the
`0.5` (maybe) pixels are dropped.

**Metrics.**

- **IoU** between a binary truth mask `(label == 1.0)` and a binary prediction
  mask `(P(pos) >= threshold)`.
- **Confusion matrix** restricted to the binary `{neg, pos}` labels.
- **Probability histograms** of `P(pos)` split by true class.
- **ROC-style FPR–TPR curve** swept over threshold.

The two pipelines produce qualitatively different probability distributions:
PCA+SVM yields sharply bimodal probabilities (so IoU and the confusion matrix
look near-perfect, but the ROC is degenerate), while the ML model produces
more continuous probabilities that spread across the `[0, 1]` interval.

## Headline results

Numbers from the current CV + held-out test setup:

| Model    | Split | IoU   | RMSE  | FPR     | FNR     |
|----------|-------|-------|-------|---------|---------|
| ML       | CV    | 0.766 | 0.594 | 0.0879% | 0.0806% |
| PCA+SVM  | CV    | 0.978 | 0.555 | 0.818%  | 1.0199% |
| ML       | Test  | -     | 0.125 | 0%      | 1.56%   |
| PCA+SVM  | Test  | -     | 0.000 | 0%      | 0%      |

PCA+SVM is near-perfect under current labels and preprocessing.
The ML model trails on IoU during CV but is competitive on the held-out scene
and has a near-perfect true-positive rate, which motivates further work
training a network directly on PCA features rather than band features alone.

The full discussion, including loss curves, ROC, probability histograms, PCA geometry
plots, and the analysis of "maybe" label behaviour, is described at: <https://holograph.luxrin.com/projects/spectral/project-pyspec-classification/>.

## Project layout

```
justfile           # Common dev commands (format, lint, type, test, marimo)
pyproject.toml     # Project metadata; ruff, mypy, pytest, basedpyright config
data/              # raw / interim / processed / external / ready - kept out of git
models/            # Trained weights and exported artifacts
notebooks/         # Marimo experiment notebooks
reports/           # Generated reports and figures
references/        # Background material and papers
stubs/             # Type stub overrides
tests/             # Pytest suite (markers: unit / integration / slow)
pyspectral/        # Library code
  config.py        # Paths, logging, RNG seed, domain constants
  core.py          # Cube / FlatMap / TruePredPair containers
  cli.py           # `pyspec` interactive entrypoint
  types.py         # jaxtyping + beartype array aliases
  data/            # Ingestion, preprocessing, features, datasets, simulations
  modeling/        # Models, training loops, OOF tracking
  result/          # Inference, comparison, plotting, SVM baseline
uv.lock            # Locked dependencies synced via `uv sync`
```

## Conventions

- **Ruff:** 88-char lines, double quotes, isort with `pyspectral` as
  first-party. `F722` is ignored so jaxtyping shape strings
  (e.g. `Float[Array, "H W M"]`) lint cleanly.
- **Typing:** `disallow_untyped_defs = true`, `no_implicit_optional = true`.
  Annotate array shapes with jaxtyping; runtime shape checks come from beartype
  via the aliases in `pyspectral/types.py`.
- **Test markers:** tag new tests `unit`, `integration`, or `slow`. Some tests
  in `test_data.py` require real assets in `data/raw`, `data/processed`, and
  `data/*.csv`.
- **New configuration** lives in `pyspectral/config.py` as typed dataclasses -
  avoid ad-hoc module-level globals.
- **Data files:** large files stay out of git. Raw data in `data/raw/`,
  intermediates in `data/processed/`, ready assets in `data/ready/`.

## Limitations

- **Dataset size.** Few real scenes; most additional scenes are simulated from
  the original maps, so they are correlated with the seed scene.
- **Label quality.** Presence masks are derived from boundary thresholding,
  which embeds the bias of that thresholding step. The boundary regions are
  genuinely un-classifiable in places and end up in the "maybe" bucket.
- **Scope.** Current models are binary alginate-presence detectors. Detecting
  other chemistries, or going multi-class, would require new training data and
  matching changes to the metrics layer.

## Future work

- **Label-quality analysis.** Spatial entropy maps and boundary smoothness
  metrics to surface likely mislabels - Raman chemical distributions ought to
  be spatially smooth, so isolated pixels or jagged interfaces are suspicious.
- **Model inputs.** Train the network on full spectra or on PCA / SVD features
  directly, not just on the band-ratio features.
- **Dataset extension.** Add new chemistries and new acquisition settings,
  then test whether the PCA+SVM baseline still holds or whether domain shift
  is what finally separates it from the ML pipeline.
