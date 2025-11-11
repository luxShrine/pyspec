from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pyspectral.config import ArrayF, ArrayF32, ModelType
from pyspectral.dataset import (
    IndexArray,
    KFolds,
    PixelSpectraDataset,
    SpectralData,
    SpectraPair,
)
from pyspectral.features import DataArtifacts, FoldStat
from pyspectral.modeling.models import (
    BandPenalty,
    Penalty,
    SpectralMapper,
    get_loss_managers,
    get_model,
)


@dataclass(frozen=True, slots=True)
class TestResult:
    # BUG: not always using rmse, correct downstream use of this RMSE value
    test_loss: float
    fraction_improved: float
    predictions: list[ArrayF32]


@dataclass(frozen=True, slots=True)
class EpochLoss:
    train: ArrayF  # shape: (Epoch count)
    test: ArrayF


@dataclass(frozen=True, slots=True)
class FoldLoss:
    train: ArrayF  # shape: (folds, Epoch count)
    test: ArrayF
    frac_improved: float
    best_test: float

    def repr(self) -> str:
        out = (
            f"train loss={self.train:.3g} | "
            + f"test loss={self.test:.3g} | "
            + f"best test loss={self.best_test:.3g} | "
            + f"frac improved vs identity={self.frac_improved:.3f}"
        )
        return out


@dataclass
class EpochLosses:
    # Mean epoch loss across folds
    epoch_loss_store: EpochLoss = field(init=False)
    # Loss, epoch data for each fold, ignore from repr this is not meant to be used
    _fold_loss_store: list[FoldLoss] = field(repr=False)

    def __post_init__(self) -> None:
        """Average across each fold at same epoch at the end for per epoch metric"""
        # collapse from fold: (epoch_idx: Loss) -> epoch_idx: avg_fold(Loss)
        tr_all_folds = []
        te_all_folds = []
        folds = len(self._fold_loss_store)
        for fold in self._fold_loss_store:
            tr_all_folds.append(fold.train / folds)
            te_all_folds.append(fold.test / folds)

        tr_all_folds_ep = np.einsum("ij->j", np.vstack(tr_all_folds))
        te_all_folds_ep = np.einsum("ij->j", np.vstack(te_all_folds))
        self.epoch_loss_store = EpochLoss(train=tr_all_folds_ep, test=te_all_folds_ep)


class OOFStats:
    """Store per fold stats."""

    def __init__(self, prc_spectra: ArrayF32, artifacts: DataArtifacts):
        # all shapes are the same: (N,C)
        self.pred_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.true_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.diag_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.diag_orig: ArrayF32 = np.zeros_like(prc_spectra)

        # to compute RMSE in original units we’ll store per-sample inverse later
        self.pred_orig: ArrayF32 = np.zeros_like(prc_spectra)
        self._prc: ArrayF32 = prc_spectra

        # training & test loss stats
        self.epoch_loss: list[FoldLoss] = []

        self.artifacts: DataArtifacts = artifacts

    def store(
        self,
        fold_stat: FoldStat,
        best_preds_fold: list[ArrayF32] | None,
        yhat_te_std: ArrayF,
        yhat_te_orig: ArrayF,
        test_idx: IndexArray,
        fold_epoch_loss: FoldLoss,
    ) -> None:
        """Update records of out of fold stats each fold."""
        if best_preds_fold is None:
            raise RuntimeError(
                "best_preds_fold is None; did training produce any predictions?"
            )
        preds_fold = np.concatenate(best_preds_fold, axis=0)
        self.pred_std[test_idx] = preds_fold
        self.true_std[test_idx] = fold_stat.test_prc_z
        self.diag_std[test_idx] = yhat_te_std
        # inverse transform to original Y units: y = y_std*sd + mu
        y_pred_orig = preds_fold * fold_stat.y_std + fold_stat.y_mean
        self.pred_orig[test_idx] = y_pred_orig
        self.diag_orig[test_idx] = yhat_te_orig

        self.epoch_loss.append(fold_epoch_loss)

    def get_loss(self) -> EpochLosses:
        return EpochLosses(_fold_loss_store=self.epoch_loss)


def pick_device() -> torch.device:
    """Select the best available PyTorch device, preferring GPU backends when present.

    Returns:
        Torch device pointing at CUDA, Metal (MPS), or CPU in that order of priority.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def rmse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the root-mean-squared error between predictions and targets.

    Args:
        y_true: Reference tensor containing ground truth spectra.
        y_pred: Tensor with model predictions in the same shape as ``y_true``.

    Returns:
        Scalar tensor containing the RMSE value.
    """
    return torch.sqrt(torch.mean((y_true - y_pred).pow(2)))


@torch.no_grad()
def cross_entropy(y: Tensor, p: Tensor) -> Tensor:
    ce = torch.nn.functional.cross_entropy(p, y)
    return ce


def create_dataloader(
    training_data: Dataset[SpectralData],
    test_data: Dataset[SpectralData],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader[SpectralData], DataLoader[SpectralData]]:
    """Build train and test dataloaders with convenience logging of sample shapes.

    Args:
        training_data: Dataset yielding raw and processed spectra for fitting.
        test_data: Held-out dataset used for validation loss computation.
        device: Torch device used to decide whether to pin loader memory.
        batch_size: Number of samples per stochastic batch.
        num_workers: Number of parallel workers

    Returns:
        Tuple with the training and test dataloaders.
    """
    # Create data loaders.
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=num_workers
    )

    for x, y in test_dataloader:
        logger.debug(f"Shape of spectra [N, C, H, W]: {x.shape}")
        logger.debug(f"Shape of processed spectra: {y.shape} {y.dtype}")
        break

    return (train_dataloader, test_dataloader)


def train_epoch(
    dataloader: DataLoader[SpectralData],
    model: SpectralMapper,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    penalty: Penalty,
) -> float:
    """Run a single training epoch and return the average batch loss.

    Args:
        dataloader: Iterator over mini-batches of raw and processed spectra.
        model: Spectral mapper being optimized.
        loss_fn: Criterion measuring reconstruction error in processed space.
        optimizer: Optimizer compatible with the mapper AdamW.
        device: Device onto which inputs are streamed during the epoch.
        penalty: Optional regularizer applied to the mapper parameters per batch.

    Returns:
        Mean loss across all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for spectra, target in dataloader:
        spectra, target = (
            spectra.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        optimizer.zero_grad(set_to_none=True)
        pred = model(spectra)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            target = target.long()

        batch_loss = loss_fn(pred, target)
        batch_loss = penalty.apply_penalty(model, batch_loss)

        # backpropagation
        batch_loss.backward()
        optimizer.step()

        total_loss += float(batch_loss.detach())
    return total_loss / len(dataloader)


@torch.no_grad()
def test_epoch(
    loader: DataLoader[SpectralData],
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> TestResult:
    """Evaluate the model for one epoch and capture summary statistics.

    Args:
        loader: Validation dataloader yielding raw and processed spectra pairs.
        model: Spectral mapper evaluated in inference mode.
        loss_fn: Loss function matching the training objective.
        device: Device hosting the model and receiving inputs.

    Returns:
        TestResult instance with MSE, RMSE, improvement rate, and predictions.
    """
    model.eval()
    vl_loss = total_improv = 0.0
    n_total = 0
    preds_fold = []
    for spectra, target in loader:
        spectra, target = spectra.to(device), target.to(device)
        pred = model(spectra)
        preds_fold.append(pred.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32))

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            target = target.long()
        loss = loss_fn(pred, target)
        # weight loss by batch size
        batches = spectra.shape[0]
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            true_diff = cross_entropy(target, pred)
            pred_loss = loss
        else:
            true_diff = rmse(target, pred)
            pred_loss = torch.sqrt(loss)

        vl_loss += float(loss) * batches
        total_improv += float(pred_loss < true_diff)
        n_total += batches
    frac_improved = total_improv / max(1, n_total)
    vl_loss /= max(1, n_total)
    return TestResult(vl_loss, frac_improved, preds_fold)


# -- main training function


def run_epochs(
    epochs: int,
    train_dl: DataLoader[SpectralData],
    test_dl: DataLoader[SpectralData],
    model: SpectralMapper,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    penalty: Penalty,
) -> tuple[FoldLoss, list[np.ndarray] | None]:
    # metrics
    best_vl_loss, best_preds, fi, tr_loss, vl_loss = float("inf"), None, 0.0, [], []
    for _ep in range(1, epochs + 1):
        tr_loss.append(
            train_epoch(train_dl, model, loss_fn, optimizer, device, penalty)
        )
        test_result = test_epoch(test_dl, model, loss_fn, device)
        fi = test_result.fraction_improved

        vl_loss.append(test_result.test_loss)
        if test_result.test_loss < best_vl_loss:
            best_vl_loss = test_result.test_loss
            best_preds = test_result.predictions

    fold_loss = FoldLoss(np.asarray(tr_loss), np.asarray(vl_loss), fi, best_vl_loss)
    return fold_loss, best_preds


def cv_train_model(
    spectral_pairs: SpectraPair,
    arts: DataArtifacts,
    n_splits: int = 4,
    batch_size: int = 2,
    epochs: int = 20,
    lr: float = 2e-4,
    wd: float = 1e-4,
    model_type: str | ModelType = ModelType.LRSM,
    rank: int = 8,
    band_penalty: BandPenalty | None = None,
    groups: bool = True,
    verbose: bool = False,
) -> OOFStats:
    """Train the mapper with cross-validation and accumulate out-of-fold diagnostics.

    Args:
        n_splits: Number of cross-validation folds to construct.
        groups: Optional grouping labels controlling the fold assignments.
        batch_size: Mini-batch size fed to the training loop.
        epochs: Number of training epochs per fold.
        lr: Learning rate supplied to the optimizer factory.
        wd: Weight decay applied when using AdamW.
        model_type: Name of the spectral mapper variant to train.
        rank: Rank parameter for low-rank spectral mappers.
        verbose: Whether to stream fold-level diagnostics to stdout.

    Returns:
        Aggregated out-of-fold statistics capturing losses and predictions.
    """
    band_penalty = BandPenalty() if band_penalty is None else band_penalty
    device = pick_device()
    model_type = (
        model_type if isinstance(model_type, ModelType) else ModelType(model_type)
    )
    raw = spectral_pairs.X_raw.astype(np.float32)  # (N,C)
    prc = spectral_pairs.Y_proc.astype(np.float32)  # (N,C)

    # TODO: calculate this instead of magic
    h = w = 8

    kfolds = KFolds(n_splits, raw)
    cv, split_iter = kfolds.get_splits((h, w))

    # out of fold metrics container
    oof_stats = OOFStats(prc_spectra=prc, artifacts=arts)

    total_folds = cv.get_n_splits()
    for fold, (tr_idx, te_idx) in tqdm(enumerate(split_iter), total=total_folds):
        # take out the sections of the arrays by index & normalize around average
        fold_stat = FoldStat.from_subset(
            raw[tr_idx], prc[tr_idx], raw[te_idx], prc[te_idx]
        )

        # create baseline to compare to.
        yhat_te_std, yhat_te_orig = fold_stat.get_baseline()

        # must recreate model, datasets & loaders each fold
        train_ds = PixelSpectraDataset(fold_stat.train_raw_z, fold_stat.train_prc_z)
        test_ds = PixelSpectraDataset(fold_stat.test_raw_z, fold_stat.test_prc_z)
        train_dl, test_dl = create_dataloader(
            train_ds, test_ds, device, batch_size=batch_size
        )
        c_in = train_ds[0][0].shape[0]
        c_out = train_ds[0][1].shape[0]

        # setup model
        model, penalty = get_model(
            model_type, fold_stat, band_penalty, c_in, rank, device
        )

        loss_fn, optimizer = get_loss_managers(model, lr=lr, wd=wd)
        if fold == 0:
            details = f"model type: {model} | channels in -> channels out: {c_in} -> {c_out}\n"
            details += f"loss function: {loss_fn} | optimizer: {optimizer}\n"
            details += f"Fold option: {type(cv)}"
            logger.debug(details)

        fold_loss, best_preds = run_epochs(
            epochs, train_dl, test_dl, model, loss_fn, optimizer, device, penalty
        )

        if verbose:
            out = f"Fold {fold + 1}/{total_folds} |" + fold_loss.repr()
            tqdm.write(out)

        # store metrics
        oof_stats.store(
            fold_stat,
            best_preds,
            yhat_te_std,
            yhat_te_orig,
            te_idx,
            fold_loss,
        )

    return oof_stats


# TODO:
# Molecule Scoring


def get_concentration(absorption: np.ndarray, k: np.ndarray):
    # Regression, we can use least squares to measure concentration:
    # minimizing function: S = \sum_i=1^n (y_i - f_i)^2
    # => c = (K^T A) / (K^T K)
    # with K being the absorbance coefficient & A the light absorption
    # with variance of:
    # σ^2 ≈ S/(N-1)
    numerator = k.T @ absorption
    denom = np.linalg.inv(k.T @ k)
    return denom @ numerator


@dataclass
class RegionSet:
    """Characteristic spectral features in 800–1000 cm⁻¹, 1300–1500 cm⁻¹, and 1500–1800 cm⁻¹"""

    low: float = 900.0
    mid: float = 1400.0
    high: float = 1650.0
    lo_window: tuple[float, float] = field(init=False)
    mid_window: tuple[float, float] = field(init=False)
    hi_window: tuple[float, float] = field(init=False)
    window_range: float | int = 100  # percent: float or absolute measure: int

    def __post_init__(self) -> None:
        if isinstance(self.window_range, float):
            r1 = self.window_range * self.low
            r2 = self.window_range * self.mid
            r3 = self.window_range * self.high
        else:
            r1 = self.window_range
            r2 = self.window_range
            r3 = self.window_range
        # prevent negative region, could also add high boundaries
        self.lo_window = (min(self.low - r1, 0), self.low + r1)
        self.mid_window = (min(self.mid - r2, 0), self.mid + r2)
        self.hi_window = (min(self.high - r3, 0), self.high + r3)

    def __iter__(self):
        yield from [self.lo_window, self.mid_window, self.hi_window]

    def get_low(self) -> tuple[float, float]:
        return self.lo_window

    def get_mid(self) -> tuple[float, float]:
        return self.mid_window

    def get_high(self) -> tuple[float, float]:
        return self.hi_window

    def match(self, sample: float) -> Literal["low", "middle", "high"] | None:
        """Find if sample is found within expected features, if so, return which feature."""
        if self.lo_window[0] <= sample and sample <= self.lo_window[1]:
            return "low"
        elif self.mid_window[0] <= sample and sample <= self.mid_window[1]:
            return "middle"
        elif self.hi_window[0] <= sample and sample <= self.hi_window[1]:
            return "high"
        return None


def _get_region_area(
    sample: np.ndarray,
    reference: np.ndarray,
    region: tuple[float, float],
    aggregation: Literal["mean", "median"] = "mean",
) -> tuple[np.ndarray | np.floating, np.ndarray | np.floating]:
    r1 = region[0]
    r2 = region[1]
    distance = abs(r1 - r2)

    sample_dx = distance / len(sample)
    ref_dx = distance / len(reference)

    sample_area = sample * sample_dx
    ref_area = reference * ref_dx

    # take average or median of these values?
    if aggregation.lower() == "mean":
        sample_area = np.mean(sample_area)
        ref_area = np.mean(ref_area)
    elif aggregation.lower() == "median":
        sample_area = np.median(sample_area)
        ref_area = np.median(ref_area)
    else:
        NotImplementedError(f"{aggregation=}")

    return sample_area, ref_area


def _filter_array(
    array: np.ndarray, upper_bound: float, lower_bound: float
) -> np.ndarray:
    mask = (array >= lower_bound) & (array <= upper_bound)
    return array[mask]


@dataclass
class PresenceMap:
    lo: np.ndarray
    mid: np.ndarray
    hi: np.ndarray


# integrate area or height in small windows around molecule's bands
# get ratio of the sample vs reference
# Use some threshold, if the ratio surpasses the threshold it is present
# This ought be binary at first, but can be probabilistic with more data
# Regardless, for each pixel, create a map of 1/0 presence of desired molecule
def create_binary_spectra_map(
    sample: np.ndarray,
    reference: np.ndarray,
    regions: RegionSet,
    threshold: float = 0.5,
    aggregation: str = "mean",
):
    # TODO: get values of array within each window, ought handle case
    # where window finds nothing, in such a case, should expand the
    # window a reasonable amount before raising an exception
    bool_maps = []
    for region in regions:
        # find the area under this curve
        reference_filtered = _filter_array(reference, *region)
        sample_filtered = _filter_array(sample, *region)
        sample_area, ref_area = _get_region_area(
            sample_filtered, reference_filtered, region
        )

        ratio = np.asarray(ref_area / sample_area)
        bool_map = ratio > threshold
        bool_maps.append(bool_map)
    return PresenceMap(*bool_maps)
