from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pyspectral.config import (
    ArrayF,
    ArrayF32,
    ArrayI,
    ClassModelType,
    IndexArray,
    SpecModelType,
)
from pyspectral.data.dataset import (
    KFolds,
    PixelSpectraDataset,
    build_scene_datasets,
    build_spec_datasets,
    scene_collate,
)
from pyspectral.data.features import FoldStat, RegionSet
from pyspectral.data.io import ClassPair, DataArtifacts, SpectraPair
from pyspectral.modeling.models import (
    BandPenalty,
    ClassCfg,
    MILMeanHead,
    Model,
    Penalty,
    SpectraCfg,
    get_train_setup,
)


@dataclass(frozen=True, slots=True)
class TestResult:
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
            f"train loss={self.train.mean():.3g} | "
            + f"test loss={self.test.mean():.3g} | "
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

    def __init__(self, prc_spectra: ArrayF32 | ArrayF, artifacts: DataArtifacts):
        # all shapes are the same: (N,C)
        prc_spectra = prc_spectra.astype(np.float32, copy=False)
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
        # inverse transform to original Y units: y = y_std*sd + mu
        y_pred_orig = preds_fold * fold_stat.y_std + fold_stat.y_mean
        self.pred_orig[test_idx] = y_pred_orig

        self.epoch_loss.append(fold_epoch_loss)

    def get_loss(self) -> EpochLosses:
        return EpochLosses(_fold_loss_store=self.epoch_loss)


class ClassOOFStats:
    """Per-scene diagnostics for classification cross-validation."""

    def __init__(self, artifacts: DataArtifacts):
        self.artifacts: DataArtifacts = artifacts
        self.scene_ids: ArrayI = np.asarray(artifacts.scene_ids, dtype=np.int64)
        # binary presence labels stored for convenience
        self.true: ArrayF32 = np.asarray(artifacts.presences, dtype=np.float32).reshape(
            -1, 1
        )
        self.pred: ArrayF32 | None = None
        self.epoch_loss: list[FoldLoss] = []

    def _ensure_storage(self, pred_dim: int) -> None:
        if self.pred is None:
            n_scenes = self.scene_ids.size
            self.pred = np.zeros((n_scenes, pred_dim), dtype=np.float32)

    def store(
        self,
        scene_indices: IndexArray,
        best_preds_fold: list[ArrayF32] | None,
        fold_loss: FoldLoss,
    ) -> None:
        if best_preds_fold is None:
            raise RuntimeError(
                "best_preds_fold is None; did the classification fold run evaluation?"
            )
        preds = np.concatenate(best_preds_fold, axis=0).astype(np.float32, copy=False)
        if preds.ndim == 1:
            preds = preds[:, None]
        elif preds.ndim > 2:
            preds = preds.reshape(preds.shape[0], -1)

        scene_indices = np.asarray(scene_indices, dtype=np.int64)
        if preds.shape[0] != scene_indices.size:
            raise ValueError(
                "Prediction count mismatch "
                + f"(preds={preds.shape[0]} vs scenes={scene_indices.size})."
            )

        self._ensure_storage(preds.shape[1] if preds.ndim > 1 else 1)
        assert self.pred is not None  # for type checker
        self.pred[scene_indices] = preds
        self.epoch_loss.append(fold_loss)

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
    training_data: Dataset,
    test_data: Dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    collate_fn: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
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


def _prediction_from_output(output: nn.Module | Tensor | tuple | list) -> Tensor:
    """Unpack model outputs, returning the tensor used for loss/pred storage."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        first = output[0]
        if isinstance(first, Tensor):
            return first
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _pred_to_numpy(pred: Tensor) -> ArrayF32:
    arr = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    while arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def _prepare_batch(
    batch: tuple | list,
    device: torch.device,
    model: Model,
) -> tuple[tuple[Tensor, ...], Tensor, torch.Tensor | None]:
    if len(batch) == 2:
        spectra, target = batch
        inputs = (spectra.to(device, non_blocking=True),)
        return inputs, target.to(device, non_blocking=True), None
    if len(batch) == 4:
        feats, mask, target, scene_idx = batch
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if isinstance(model, MILMeanHead):
            inputs = (feats, mask)
        else:
            inputs = (feats,)
        return inputs, target, scene_idx
    raise TypeError(f"Unsupported batch structure of length {len(batch)}")


def train_epoch(
    dataloader: DataLoader,
    model: Model,
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

    for batch in dataloader:
        inputs, target, _ = _prepare_batch(batch, device, model)
        optimizer.zero_grad(set_to_none=True)
        pred = _prediction_from_output(model(*inputs))
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
    model: Model,
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
    for batch in loader:
        inputs, target, _ = _prepare_batch(batch, device, model)
        pred = _prediction_from_output(model(*inputs))
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
    model: Model,
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
    model_type: str | SpecModelType = SpecModelType.LRSM,
    rank: int = 8,
    band_penalty: BandPenalty | None = None,
    verbose: bool = False,
) -> OOFStats:
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
    band_penalty = BandPenalty() if band_penalty is None else band_penalty
    device = pick_device()
    model_type = (
        model_type
        if isinstance(model_type, SpecModelType)
        else SpecModelType(model_type)
    )

    # out of fold metrics container
    oof_stats = OOFStats(prc_spectra=spectral_pairs.Y_proc, artifacts=arts)

    for fold, (tr_dsi, te_dsi, fold_stat) in tqdm(
        enumerate(build_spec_datasets(n_splits, spectral_pairs, arts))
    ):
        train_ds, _ = tr_dsi
        test_ds, te_idx = te_dsi

        train_dl, test_dl = create_dataloader(
            train_ds, test_ds, device, batch_size=batch_size
        )
        c_in = train_ds[0][0].shape[0]
        c_out = train_ds[0][1].shape[0]

        # setup model
        cfg = SpectraCfg(fold_stat, model_type, band_penalty=band_penalty, rank=rank)
        train_setup = get_train_setup(c_in, device, lr=lr, wd=wd, goal=cfg)

        if fold == 0:
            details = f"model type: {train_setup.model} | channels in -> channels out: {c_in} -> {c_out}\n"
            details += f"loss function: {train_setup.loss_fn} | optimizer: {train_setup.optimizer}\n"
            logger.debug(details)

        fold_loss, best_preds = run_epochs(
            epochs,
            train_dl,
            test_dl,
            train_setup.model,
            train_setup.loss_fn,
            train_setup.optimizer,
            device,
            train_setup.penalty,
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
) -> ClassOOFStats:
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
    ppt = int(h / 2)

    region_set = RegionSet()
    oof_stats = ClassOOFStats(class_pair.arts)

    for fold, (tr_dsi, te_dsi, _) in tqdm(
        enumerate(build_scene_datasets(n_splits, class_pair, region_set))
    ):
        tr_ds, _ = tr_dsi
        te_ds, te_i = te_dsi

        # must recreate model, datasets & loaders each fold
        train_dl, test_dl = create_dataloader(
            tr_ds, te_ds, device, batch_size, collate_fn=scene_collate
        )
        c_in = tr_ds[0][0].shape[0]
        # c_out = tr_ds[0][1].shape[0]

        # setup model
        cfg = ClassCfg(h, w, ClassModelType.MIL)
        train_setup = get_train_setup(c_in, device, lr=lr, wd=wd, goal=cfg)

        if fold == 0:
            details = f"model type: {train_setup.model} \n"
            details += f"loss function: {train_setup.loss_fn} | optimizer: {train_setup.optimizer}\n"
            logger.debug(details)

        fold_loss, best_preds = run_epochs(
            epochs,
            train_dl,
            test_dl,
            train_setup.model,
            train_setup.loss_fn,
            train_setup.optimizer,
            device,
            train_setup.penalty,
        )

        if verbose:
            out = f"Fold {fold + 1} |" + fold_loss.repr()
            tqdm.write(out)

        # store metrics for held-out scenes
        oof_stats.store(te_i, best_preds, fold_loss)

    return oof_stats
