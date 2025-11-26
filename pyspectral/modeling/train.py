from __future__ import annotations

from collections.abc import Callable

from loguru import logger
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pyspectral.core import TestResult
from pyspectral.data.dataset import (
    build_pix_scene_datasets,
    build_scene_datasets,
    build_spec_datasets,
    pixel_collate,
    scene_collate,
)
from pyspectral.data.features import RegionSet
from pyspectral.data.io import ClassPair, DataArtifacts, SpectraPair
import pyspectral.modeling.models as pm
import pyspectral.modeling.oof as oof
from pyspectral.types import (
    ArrayF32,
    ClassModelType,
    SpecModelType,
)

# -- helpers -------------------------------------------------------


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


def compute_iou_from_masks(
    base_mask: np.ndarray,
    cmp_mask: np.ndarray,
) -> float:
    """Compute IoU between two boolean / {0,1} masks."""
    if base_mask.shape != cmp_mask.shape:
        raise ValueError("Masks must share shape")
    base_mask = np.asarray(base_mask, dtype=bool)
    cmp_mask = np.asarray(cmp_mask, dtype=bool)

    intersection = np.logical_and(base_mask, cmp_mask).sum()
    union = np.logical_or(base_mask, cmp_mask).sum()
    return float(intersection / union) if union else 0.0


def create_dataloader[T](
    training_data: Dataset[T],
    test_data: Dataset[T],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    collate_fn: Callable | None = None,
) -> tuple[DataLoader[T], DataLoader[T]]:
    """Build train and test dataloaders with convenience logging of sample shapes.

    Args:
        training_data: Dataset yielding raw and processed spectra for fitting.
        test_data: Held-out dataset used for validation loss computation.
        device: Torch device used to decide whether to pin loader memory.
        batch_size: Number of samples per stochastic batch.
        num_workers: Number of parallel workers

    Returns:
        Tuple with the training and test dataloaders. When ``collate_fn`` is supplied
        it is forwarded to both loaders (used for variable-length batches).
    """
    # Create data loaders.
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    for batch in test_dataloader:
        logger.debug(f"Shape of raw data: {batch[0].shape}")
        logger.debug(f"Shape of processed data: {batch[1].shape} {batch[1].dtype}")
        break

    return (train_dataloader, test_dataloader)


def _pred_to_numpy(pred: Tensor) -> ArrayF32:
    arr: ArrayF32 = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    while arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def train_epoch(
    dataloader: DataLoader,
    model: pm.Model,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    penalty: pm.Penalty,
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

    for batch in dataloader:
        inputs, target, _ = model.prepare(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model.pred(inputs)
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
    loader: DataLoader,
    model: pm.Model,
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
    preds_fold: list[ArrayF32] = []
    for batch in loader:
        inputs, target, _ = model.prepare(batch, device)
        pred = model.pred(inputs)
        preds_fold.append(_pred_to_numpy(pred))

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            target = target.long()
        loss = loss_fn(pred, target)
        # weight loss by batch size
        batches = int(target.shape[0]) if target.ndim >= 1 else 1
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
    train_dl: DataLoader,
    test_dl: DataLoader,
    train_setup: pm.TrainSetup,
    device: torch.device,
) -> tuple[oof.FoldLoss, list[ArrayF32] | None]:
    # metrics
    best_vl_loss, best_preds, fi, tr_loss, vl_loss = float("inf"), None, 0.0, [], []
    model = train_setup.model
    loss_fn = train_setup.loss_fn
    optimizer = train_setup.optimizer
    penalty = train_setup.penalty

    for _ep in tqdm(range(1, epochs + 1), desc="Running Epochs..."):
        tr_loss.append(
            train_epoch(train_dl, model, loss_fn, optimizer, device, penalty)
        )
        test_result = test_epoch(test_dl, model, loss_fn, device)
        fi = test_result.fraction_improved

        vl_loss.append(test_result.test_loss)
        if test_result.test_loss < best_vl_loss:
            best_vl_loss = test_result.test_loss
            best_preds = test_result.predictions

    fold_loss = oof.FoldLoss(np.asarray(tr_loss), np.asarray(vl_loss), fi, best_vl_loss)
    return fold_loss, best_preds


def cv_train_model(
    spectral_pairs: SpectraPair,
    arts: DataArtifacts,
    n_splits: int = 4,
    batch_size: int = 2,
    epochs: int = 20,
    lr: float = 2e-4,
    wd: float = 1e-4,
    model_type: str | SpecModelType = SpecModelType.LRSM,
    rank: int = 8,
    band_penalty: pm.BandPenalty | None = None,
    verbose: bool = False,
) -> oof.Stats:
    """Train the mapper with cross-validation and accumulate out-of-fold diagnostics.

    Args:
        n_splits: Number of cross-validation folds to construct.
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
    band_penalty = pm.BandPenalty() if band_penalty is None else band_penalty
    device = pick_device()
    model_type = (
        model_type
        if isinstance(model_type, SpecModelType)
        else SpecModelType(model_type)
    )

    # out of fold metrics container
    oof_stats = oof.Stats(prc_spectra=spectral_pairs.Y_proc, artifacts=arts)

    for fold, ((train_ds, _), (test_ds, te_idx), fold_stat) in enumerate(
        build_spec_datasets(n_splits, spectral_pairs)
    ):
        train_dl, test_dl = create_dataloader(
            train_ds, test_ds, device, batch_size=batch_size
        )
        c_in = train_ds[0][0].shape[0]
        _c_out = train_ds[0][1].shape[0]

        # setup model
        cfg = pm.SpectraCfg(fold_stat, model_type, band_penalty=band_penalty, rank=rank)
        train_setup = pm.TrainSetup.get_train_setup(
            c_in, device, lr=lr, wd=wd, goal=cfg
        )

        train_setup.log_init(fold)

        fold_loss, best_preds = run_epochs(
            epochs,
            train_dl,
            test_dl,
            train_setup,
            device,
        )

        if verbose:
            out = f"Fold {fold + 1}/{n_splits} |" + fold_loss.repr()
            tqdm.write(out)

        # store metrics
        oof_stats.store(
            fold_stat,
            best_preds,
            te_idx,
            fold_loss,
        )

    return oof_stats


def train_class(
    class_pair: ClassPair,
    n_splits: int = 4,
    batch_size: int = 2,
    epochs: int = 20,
    lr: float = 2e-4,
    wd: float = 1e-4,
    verbose: bool = False,
) -> oof.ClassStats:
    """
    Args:
        n_splits: Number of cross-validation folds to construct.
        batch_size: Mini-batch size fed to the training loop.
        epochs: Number of training epochs per fold.
        lr: Learning rate supplied to the optimizer factory.
        wd: Weight decay applied when using AdamW.
        verbose: Whether to stream fold-level diagnostics to stdout.

    Returns:
        Aggregated out-of-fold statistics capturing losses and predictions.
    """
    device = pick_device()

    # TODO: calculate this instead of magic
    h = w = 8
    _pixels_per_tile = int(h / 2)

    region_set = RegionSet()
    oof_stats = oof.ClassStats(class_pair.arts)

    for fold, ((tr_ds, _), (te_ds, te_i), _) in enumerate(
        build_scene_datasets(n_splits, class_pair, region_set)
    ):
        # recreate model, datasets & loaders each fold
        train_dl, test_dl = create_dataloader(
            tr_ds, te_ds, device, batch_size, collate_fn=scene_collate
        )
        c_in = tr_ds[0][0].shape[0]
        d_in = tr_ds[0][0].shape[1]
        # c_out = tr_ds[0][1].shape[0]
        class_cfg = pm.ClassCfg(h, w, d_in=d_in, model_type=ClassModelType.MIL)

        # setup model
        train_setup = pm.TrainSetup.get_train_setup(
            c_in=c_in, device=device, lr=lr, wd=wd, goal=class_cfg
        )

        train_setup.log_init(fold)

        fold_loss, best_preds = run_epochs(
            epochs,
            train_dl,
            test_dl,
            train_setup,
            device,
        )

        if verbose:
            out = f"Fold {fold + 1} |" + fold_loss.repr()
            tqdm.write(out)

        # store metrics for held-out scenes
        oof_stats.store(te_i, best_preds, fold_loss)

    return oof_stats


def train_pixel(
    class_pair: ClassPair,
    n_splits: int = 4,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 2e-4,
    wd: float = 1e-4,
    cpu_override: bool = False,
    verbose: bool = False,
    random_state: int = 47,
) -> oof.PxlStats:
    """Train a pixel-level classifier using PixelSpectraDataset cross-validation.

    Returns
        Per-pixel out-of-fold diagnostics.
    """
    device = pick_device() if not cpu_override else torch.device("cpu")
    # TODO: calculate this for pixels (always 1x1)
    h = w = 1

    region_set = RegionSet()
    oof_stats = oof.PxlStats(class_pair.arts)

    for fold, ((tr_ds, _, _), (te_ds, te_pix_idx, _), _) in enumerate(
        build_pix_scene_datasets(
            n_splits=n_splits,
            data=class_pair,
            region_set=region_set,
            random_state=random_state,
        )
    ):
        # recreate model, datasets & loaders each fold
        train_dl, test_dl = create_dataloader(
            tr_ds, te_ds, device, batch_size, collate_fn=pixel_collate
        )
        d_in = tr_ds[0][0].shape[0]
        class_cfg = pm.ClassCfg(h, w, d_in=d_in, model_type=ClassModelType.MIL)

        # setup model
        train_setup = pm.TrainSetup.get_train_setup(
            c_in=d_in, device=device, lr=lr, wd=wd, goal=class_cfg
        )
        train_setup.log_init(fold)

        fold_loss, best_preds = run_epochs(
            epochs,
            train_dl,
            test_dl,
            train_setup,
            device,
        )

        if verbose:
            out = f"Fold {fold + 1} |" + fold_loss.repr()
            tqdm.write(out)

        # store metrics for held-out pixels
        oof_stats.store(te_pix_idx, best_preds, fold_loss)

    return oof_stats
