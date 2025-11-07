from dataclasses import dataclass, field
from typing import Any, Literal, override

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
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

type Alignment = None | Tensor | LinearSpectralMapper


def get_tensor(maybe_tens: Tensor | Any) -> Tensor:
    """Check that input is a tensor, return said tensor or raise TypeError."""
    if isinstance(maybe_tens, Tensor):
        return maybe_tens
    else:
        raise TypeError(f"Input is not a tensor: {type(maybe_tens)}")


@dataclass
class BandPenalty:
    """
    id: LRSM Identity regularization strength or ridge prior for warm starts.
    off_diag: LRSM Off-diagonal penalty weight for linear spectral mappers.
    band: LSM Bandwidth of the diagonal neighborhood preserved by the penalty.
    """

    id: float = 1e-2
    off_diag: float = 1e-5  # 1e3 >= range >= 1e-5
    band: int = 1
    bias: float = 0.0


@dataclass
class TestResult:
    test_mse: float
    test_rmse: float
    fraction_improved: float
    predictions: list[ArrayF32]


@dataclass
class EpochLoss:
    train: ArrayF  # shape: (Epoch count)
    test: ArrayF


@dataclass
class FoldLoss:
    tr_loss_store: ArrayF  # shape: (folds, Epoch count)
    te_loss_store: ArrayF


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
            tr_all_folds.append(fold.tr_loss_store / folds)
            te_all_folds.append(fold.te_loss_store / folds)

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

    @override
    def __repr__(self) -> str:
        """Display metrics stored."""
        rmse_diag_std = float(np.sqrt(((self.diag_std - self.true_std) ** 2).mean()))
        rmse_diag_orig = float(np.sqrt(((self.diag_orig - self._prc) ** 2).mean()))
        rmse_std = float(np.sqrt(((self.pred_std - self.true_std) ** 2).mean()))
        rmse_orig = float(np.sqrt(((self.pred_orig - self._prc) ** 2).mean()))
        out = f"OOF RMSE (standardized space): {rmse_std:.6f}\n"
        out += f"Diagonal affine OOF RMSE (std):  {rmse_diag_std:.6f}\n"
        out += f"Diagonal affine OOF RMSE (orig): {rmse_diag_orig:.6f}\n"
        out += f"OOF RMSE (original units):    {rmse_orig:.6f}"
        return out


# tensor dimension checks
type BatchTensor = Float[Tensor, "B c1 H W"]
type ClassTensor = Float[Tensor, "B F"]
type OneDimTensor = Float[Tensor, "1"]
type ZeroDimTensor = Float[Tensor, ""]

# -- conv model


def calc_layer_adj(k: int, stride: int, pad: int, h: int, w: int):
    """Adjusts the height/width based on parameters of the pool/convulution layer."""
    num = (2 * pad) - (k - 1) - 1
    frac = num / stride
    h += np.floor(frac + 1)
    w += np.floor(frac + 1)


class ConvSpectralClassifier(nn.Module):
    def __init__(self, c_in: int, c_out: int, h: int, w: int):
        super().__init__()
        # TODO: calcuate height/width?, currently single pixels, thus 1x1
        k_conv: int = 3
        s_conv: int = 1
        pad_conv = 2
        k_pool: int = 3  # TODO: what is the desired pool kernel size ?
        s_pool: int = 1
        pad_pool = 0

        conv1_out = 128  # TODO: decide the output size
        conv2_out = 64  # TODO: decide the output size
        linear1_out = 512  # TODO: decide on the output size
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(c_in, conv1_out, kernel_size=k_conv, padding=pad_conv),
            nn.Tanh(),  # Kurz et al. uses ReLU
            nn.MaxPool2d(k_pool, stride=s_pool),
            nn.Conv2d(conv1_out, conv2_out, kernel_size=k_conv, padding=pad_conv),
            nn.Tanh(),
            nn.MaxPool2d(k_pool, stride=s_pool),
        )
        self.features.compile(backend="cudagraphs")
        calc_layer_adj(k_conv, s_conv, pad_conv, h, w)  # conv1: ...,...,H+2,W+2
        calc_layer_adj(k_pool, s_pool, pad_pool, h, w)  # maxpool1
        calc_layer_adj(k_conv, s_conv, pad_conv, h, w)  # conv2: ...,...,H+2,W+2
        calc_layer_adj(k_pool, s_pool, pad_pool, h, w)  # maxpool2
        flatten_out = (h * w) * conv2_out
        self.head: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_out, linear1_out),
            nn.Tanh(),
            nn.Linear(linear1_out, c_out),
        )

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, x: BatchTensor) -> ClassTensor:
        """
        (B, C, H, W) -> returns (B, C_out)
        """
        features: Float[Tensor, "B C2 H2 W2"] = self.features(x)
        y: ClassTensor = self.head(features)
        return y

    def get_loss_managers(
        self, lr: float = 5e-4, wd: float = 1e-4
    ) -> tuple[nn.CrossEntropyLoss, torch.optim.AdamW]:
        """Create the optimizer and loss function directly from object."""
        return (
            nn.CrossEntropyLoss(),
            torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd),
        )


# -- Low Rank, less complexity less memory


class LowRankSpectralMapper(nn.Module):
    """y = x + (x @ V) @ U^T  (near-identity low-rank correction)"""

    def __init__(self, c_in: int, rank: int = 64):
        super().__init__()
        self.V: nn.Parameter = nn.Parameter(torch.zeros(c_in, rank))
        self.U: nn.Parameter = nn.Parameter(torch.zeros(c_in, rank))
        # normal distribution initialization
        nn.init.normal_(self.V, std=1e-3)
        nn.init.normal_(self.U, std=1e-3)
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(c_in))

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, x: BatchTensor) -> BatchTensor:
        """
        (B, C, H, W)         -> returns (B, C, H, W)
        """
        B, C = x.shape[:2]
        x_flat: Float[Tensor, "batch c1"] = x.view(B, C)
        # sanity check channel dimension
        if x_flat.shape[1] != self.V.shape[0]:
            raise RuntimeError(
                f"Channel mismatch: x has C={x.shape[1]} but mapper was built with C={self.V.shape[0]}"
            )

        # (B,C) @ (C,r) -> (B,r) -> (B,C)
        corr: Float[Tensor, "batch r"] = (x_flat @ self.V) @ self.U.t()
        # broadcast bias (C,)
        y_flat: Float[Tensor, "batch c1"] = x_flat + corr + self.bias
        y: BatchTensor = y_flat.unsqueeze(-1).unsqueeze(-1)

        return y

    def get_loss_managers(
        self, lr: float = 5e-4, wd: float = 1e-4
    ) -> tuple[nn.MSELoss, torch.optim.AdamW]:
        """Create the optimizer and loss function directly from object."""
        return (
            nn.MSELoss(),
            torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd),
        )


class LowRankIdentityPenalty(nn.Module):
    """λ * (||U||_F^2 + ||V||_F^2). Optionally, bias L2, too."""

    def __init__(self, band_penalty: BandPenalty):
        super().__init__()
        self.lam_id: float = band_penalty.id
        self.lam_bias: float = band_penalty.bias

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, mapper: LowRankSpectralMapper) -> ZeroDimTensor:
        loss: ZeroDimTensor = self.lam_id * (
            mapper.U.pow(2).sum() + mapper.V.pow(2).sum()
        )
        if self.lam_bias:
            loss = loss + self.lam_bias * mapper.bias.pow(2).sum()
        return loss


# -- Higher rank, more complexity, more memory


class LinearSpectralMapper(nn.Module):
    """
    Per-pixel spectral map: y = W x + b, with W in R^{C×C} via 1×1 conv.
    """

    def __init__(self, c: int):
        super().__init__()
        self.proj: nn.Conv2d = nn.Conv2d(c, c, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj_bias: Tensor = get_tensor(self.proj.bias)
            # init near-identity to help small amount of data
            nn.init.zeros_(self.proj.weight)
            for i in range(c):
                self.proj.weight[i, i, 0, 0] = 1.0
            nn.init.zeros_(self.proj_bias)

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, x: BatchTensor) -> BatchTensor:
        return get_tensor(self.proj(x))

    def get_loss_managers(
        self, lr: float = 5e-4, wd: float = 1e-4, type: str = "AdamW"
    ) -> tuple[nn.MSELoss, torch.optim.Optimizer]:
        """Create the optimizer and loss function directly from object."""
        if type == "LBFGS":
            return (
                nn.MSELoss(),
                torch.optim.LBFGS(
                    self.parameters(), lr=1e-2, max_iter=100, history_size=10
                ),
            )
        elif type == "AdamW":
            return (
                nn.MSELoss(),
                torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd),
            )
        else:
            raise RuntimeError


class IdentityBandedPenalty(nn.Module):
    """λ||W−I||_F^2 on full matrix + μ * energy outside a ±band around diagonal"""

    def __init__(self, c: int, band_penalty: BandPenalty):
        super().__init__()
        self.register_buffer("eye", torch.eye(c))  # (C,C)
        self.band: int = band_penalty.band
        self.lam_id: float = band_penalty.id
        self.off_diag_penalty: float = band_penalty.off_diag

    def _sum_sq_diagonals(self, W: Tensor, kmax: int) -> ZeroDimTensor:
        """sum_{k=-kmax..kmax} ||diag_k(W)||_2^2"""
        s: ZeroDimTensor = torch.sum(torch.diagonal(W, offset=0).pow(2))
        for k in range(1, kmax + 1):
            s = s + torch.sum(torch.diagonal(W, offset=k).pow(2))
            s = s + torch.sum(torch.diagonal(W, offset=-k).pow(2))
        return s

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, conv1x1: nn.Conv2d) -> ZeroDimTensor:
        W: Float[Tensor, "c1 c1"] = conv1x1.weight.squeeze(-1).squeeze(-1)  # (C,C)
        # (C,C) -> (1)
        if isinstance(self.eye, Tensor):
            loss_id: ZeroDimTensor = torch.sum((W - self.eye) ** 2) * self.lam_id
            frob_2: ZeroDimTensor = torch.sum(W * W)  # Frobenius solution r1 == r2
            band_energy: ZeroDimTensor = self._sum_sq_diagonals(W, self.band)
            offband_energy: ZeroDimTensor = frob_2 - band_energy
            return loss_id + self.off_diag_penalty * offband_energy  # (1)
        else:
            raise TypeError(f"self.eye is not tensor: {type(self.eye)=}")


# -- Training Functions

type Penalty = IdentityBandedPenalty | LowRankIdentityPenalty
type SpectralMapper = LinearSpectralMapper | LowRankSpectralMapper


def init_linear_map(
    raw: ArrayF32, prc: ArrayF32, lam_l2: float, model: LinearSpectralMapper
) -> None:
    """Warm start a linear mapper via ridge regression solved in numpy space.

    Args:
        raw: Training spectra prior to preprocessing, shape (N, C).
        prc: Processed spectra used as the regression targets, shape (n_samples, C).
        lam_l2: Ridge penalty weight applied to stabilize the linear system.
    """
    XT = raw.T  # (n_train, C)
    a = (XT @ raw) + (lam_l2 * np.eye(raw.shape[1]))
    b = XT @ prc
    W0 = np.linalg.solve(a, b)  # (C,C)
    b0 = prc.mean(axis=0) - (raw.mean(axis=0) @ W0)  # (C,)
    with torch.no_grad():
        model.proj.weight.copy_(
            torch.from_numpy(W0.T).float().unsqueeze(-1).unsqueeze(-1)
        )
        model.proj_bias.copy_(torch.from_numpy(b0).float())


def get_model(
    model_type: ModelType,
    fold_stat: FoldStat,
    band_penalty: BandPenalty,
    c_in: int,
    rank: int,
    device: torch.device,
) -> tuple[SpectralMapper, Penalty]:
    """Instantiate the spectral mapper and its regularizer for the requested model type.

    Args:
        model_type: Configuration value specifying the mapper architecture.
        fold_stat: Fold-level statistics used to warm start linear models.
        c_in: Number of spectral channels available in the input tensors.
        rank: Rank used for low-rank models (ignored for linear CNN variant).
        device: Torch device to move the model and penalty onto.
        band: Half-width of the diagonal band for `IdentityBandedPenalty`.
        off_diag_penalty: Regularization strength for energy outside the diagonal band.
        lam_id: Identity penalty weight (also used as ridge prior for warm start).

    Returns:
        Tuple comprised of the spectral mapper module and an accompanying penalty.
    """
    model: SpectralMapper
    penalty: Penalty
    match model_type:
        case ModelType.LSM:
            model = LinearSpectralMapper(c_in).to(device)
            # initialize linear map for CNN only
            init_linear_map(
                fold_stat.train_raw_z, fold_stat.train_prc_z, band_penalty.id, model
            )
            penalty = IdentityBandedPenalty(c_in, band_penalty).to(device)
        case ModelType.LRSM:
            model = LowRankSpectralMapper(c_in, rank).to(device)
            penalty = LowRankIdentityPenalty(band_penalty).to(device)
    return model, penalty


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


def create_dataloader(
    training_data: Dataset[SpectralData],
    test_data: Dataset[SpectralData],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[DataLoader[SpectralData], DataLoader[SpectralData]]:
    """Build train and test dataloaders with convenience logging of sample shapes.

    Args:
        training_data: Dataset yielding raw and processed spectra for fitting.
        test_data: Held-out dataset used for validation loss computation.
        device: Torch device used to decide whether to pin loader memory.
        batch_size: Number of samples per stochastic batch.

    Returns:
        Tuple with the training and test dataloaders.
    """
    # Create data loaders.
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

    for x, y in test_dataloader:
        logger.debug(f"Shape of spectra [N, C, H, W]: {x.shape}")
        logger.debug(f"Shape of processed spectra: {y.shape} {y.dtype}")
        break

    return (train_dataloader, test_dataloader)


def apply_penalty(
    model: SpectralMapper, penalty: Penalty | None, batch_loss: Tensor
) -> Tensor:
    # determine penalty type
    if penalty is None:
        return batch_loss
    elif hasattr(model, "proj"):
        return get_tensor(batch_loss + penalty(model.proj))
    else:
        return get_tensor(batch_loss + penalty(model))


def train_epoch(
    dataloader: DataLoader[SpectralData],
    model: SpectralMapper,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    penalty: Penalty | None = None,
) -> float:
    """Run a single training epoch and return the average batch loss.

    Args:
        dataloader: Iterator over mini-batches of raw and processed spectra.
        model: Spectral mapper being optimized.
        loss_fn: Criterion measuring reconstruction error in processed space.
        optimizer: Optimizer compatible with the mapper (AdamW or LBFGS).
        device: Device onto which inputs are streamed during the epoch.
        penalty: Optional regularizer applied to the mapper parameters per batch.

    Returns:
        Mean loss across all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for spectra, spectra_prc in dataloader:
        spectra, spectra_prc = (
            spectra.to(device, non_blocking=True),
            spectra_prc.to(device, non_blocking=True),
        )
        optimizer.zero_grad(set_to_none=True)
        pred = model(spectra)
        batch_loss = loss_fn(pred, spectra_prc)
        if isinstance(optimizer, torch.optim.AdamW):
            batch_loss = apply_penalty(model, penalty, batch_loss)
            # backpropagation
            batch_loss.backward()
            optimizer.step()
        elif isinstance(optimizer, torch.optim.LBFGS):

            def closure() -> Tensor:
                optimizer.zero_grad()
                pred = model(spectra.to(device))
                loss = loss_fn(pred, spectra_prc.to(device))
                loss = apply_penalty(model, penalty, loss)
                loss.backward()
                return loss.detach()

            optimizer.step(closure)
        else:
            raise RuntimeError("Failed to find proper optimizer")

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
    vl_rmse = vl_mse = total_improv = 0.0
    n_total = 0
    preds_fold = []
    for spectra, spectra_prc in loader:
        spectra, spectra_prc = spectra.to(device), spectra_prc.to(device)
        pred = model(spectra)
        preds_fold.append(pred.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32))

        # weight loss by batch size
        batches = spectra.shape[0]
        vl_intermediate_rmse = float(rmse(spectra_prc, pred))
        vl_mse += float(loss_fn(pred, spectra_prc)) * batches
        vl_rmse += vl_intermediate_rmse * batches

        total_improv += float((vl_intermediate_rmse < rmse(spectra_prc, spectra)))
        n_total += batches
    frac_improved = total_improv / max(1, n_total)
    vl_mse /= max(1, n_total)
    vl_rmse /= max(1, n_total)
    return TestResult(vl_mse, vl_rmse, frac_improved, preds_fold)


# -- training function


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
        csv_path: Path to annotations or metadata describing the spectra pairs.
        base_dir: Root directory for locating spectra assets on disk.
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
    model_type = model_type if model_type is ModelType else ModelType(model_type)
    raw = spectral_pairs.X_raw.astype(np.float32)  # (N,C)
    prc = spectral_pairs.Y_proc.astype(np.float32)  # (N,C)
    if groups:
        h = w = 8
        group_ints = KFolds.create_groups(h, w)
        cv, split_iter = KFolds(n_splits, raw, group_ints).get_splits()
    else:
        cv, split_iter = KFolds(n_splits, raw).get_splits()

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

        # setup model
        model, penalty = get_model(
            model_type, fold_stat, band_penalty, c_in, rank, device
        )

        loss_fn, optimizer = model.get_loss_managers(lr=lr, wd=wd)
        if fold == 0:
            c_out = train_ds[0][1].shape[0]
            details = f"model type: {model} | channels in -> channels out: {c_in} -> {c_out}\n"
            details += f"loss function: {loss_fn} | optimizer: {optimizer}\n"
            details += f"Fold option: {type(cv)}"
            logger.debug(details)

        # metrics
        best_vl_rmse, best_preds, tr_loss_fold, vl_loss_fold = (
            float("inf"),
            None,
            [],
            [],
        )

        for ep in range(1, epochs + 1):
            tr_loss = train_epoch(train_dl, model, loss_fn, optimizer, device, penalty)
            test_result = test_epoch(test_dl, model, loss_fn, device)

            tr_loss_fold.append(tr_loss)
            vl_loss_fold.append(test_result.test_rmse)
            if test_result.test_rmse < best_vl_rmse:
                best_vl_rmse = test_result.test_rmse
                best_preds = test_result.predictions

            if verbose and (ep == (epochs)):
                tqdm.write(
                    f"Fold {fold + 1}/{total_folds} | train loss={tr_loss:.3g} | "
                    + f"test MSE={test_result.test_mse:.3g} | best test RMSE={best_vl_rmse:.3g} | "
                    + f"frac improved vs identity={test_result.fraction_improved:.3f}"
                )

        # store metrics
        oof_stats.store(
            fold_stat,
            best_preds,
            yhat_te_std,
            yhat_te_orig,
            te_idx,
            FoldLoss(np.asarray(tr_loss_fold), np.asarray(vl_loss_fold)),
        )

    return oof_stats
