from dataclasses import dataclass
from typing import Any, override

from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
import numpy as np
import torch
from torch import Tensor, nn

from pyspectral.config import ArrayF32, ClassModelType, SpecModelType
from pyspectral.data.features import FoldStat


def get_tensor(maybe_tens: Tensor | Any) -> Tensor:
    """Check that input is a tensor, return said tensor or raise TypeError."""
    if isinstance(maybe_tens, Tensor):
        return maybe_tens
    else:
        raise TypeError(f"Input is not a tensor: {type(maybe_tens)}")


# tensor dimension checks
type BatchTensor = Float[Tensor, "B C"]
type ClassTensor = Float[Tensor, "B F"]
type OneDimTensor = Float[Tensor, "1"]
type ZeroDimTensor = Float[Tensor, ""]

type FeatTensor = Float[Tensor, "B T D"]
type MaskTensor = Bool[Tensor, "B T"]

# -- multiple-instance learning model


class MILMeanHead(nn.Module):
    def __init__(self, c_in: int, hidden: int = 64):
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # per-pixel logit
        )

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, X: FeatTensor, mask: MaskTensor):
        """
        Args:
            X: (B,T,D)
            mask: (B,T)
        """

        z = self.net(X).squeeze(-1)  # (B,T)
        p = torch.sigmoid(z)
        # masked mean pooling -> map probability
        denom = mask.float().sum(dim=1).clamp_min(1.0)
        prob = (p * mask).sum(dim=1) / denom  # (B,)
        return prob, z, p


# -- conv model


def calc_layer_adj(k: int, stride: int, pad: int, h: int, w: int) -> None:
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
        # valid backends: ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        # self.features.compile() # WARN: hard to get working
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


type ClassMapper = ConvSpectralClassifier | MILMeanHead
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
        (B, C) -> returns (B, C)
        """
        # sanity check channel dimension
        if x.shape[1] != self.V.shape[0]:
            raise RuntimeError(
                f"Channel mismatch: x has C={x.shape[1]} but mapper was "
                + f"built with C={self.V.shape[0]}"
            )

        # (B,C) @ (C,r) -> (B,r) -> (B,C)
        xcorr: Float[Tensor, "B r"] = x @ self.V
        corr: Float[Tensor, "B u"] = xcorr @ self.U.t()
        # broadcast bias (C,)
        y_flat: Float[Tensor, "B u 1 1"] = x + corr + self.bias
        return y_flat.squeeze(-1).squeeze(-1)


class LowRankIdentityPenalty(nn.Module):
    """λ * (||U||_F^2 + ||V||_F^2)."""

    def __init__(self, lam_id: float, bias: float):
        super().__init__()
        self.lam_id: float = lam_id
        self.lam_bias: float = bias

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


class IdentityBandedPenalty(nn.Module):
    """λ||W−I||_F^2 on full matrix + μ * energy outside a ±band around diagonal"""

    def __init__(self, c: int, band: int, lam_id: float, off_diag_penalty: float):
        super().__init__()
        self.register_buffer("eye", torch.eye(c))  # (C,C)
        self.band: int = band
        self.lam_id: float = lam_id
        self.off_diag_penalty: float = off_diag_penalty

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


type SpectralMapper = LinearSpectralMapper | LowRankSpectralMapper
type Model = SpectralMapper | ClassMapper

# -- penalties


@dataclass(frozen=True, slots=True)
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

    def make_penalty(self, model: SpectralMapper, c_in: int):
        if isinstance(model, LinearSpectralMapper):
            return Penalty(
                _penalty=IdentityBandedPenalty(c_in, self.band, self.id, self.off_diag)
            )
        else:
            return Penalty(_penalty=LowRankIdentityPenalty(self.id, self.bias))


@dataclass(frozen=True, slots=True)
class Penalty:
    _penalty: None | IdentityBandedPenalty | LowRankIdentityPenalty

    def apply_penalty(self, model: Model, batch_loss: Tensor) -> Tensor:
        # determine penalty type
        if self._penalty is None:
            return batch_loss
        elif hasattr(model, "proj"):
            return get_tensor(batch_loss + self._penalty(model.proj))
        else:
            return get_tensor(batch_loss + self._penalty(model))


# -- Training Functions


@dataclass
class SpectraCfg:
    fold_stat: FoldStat
    model_type: SpecModelType = SpecModelType.LRSM
    band_penalty: BandPenalty = BandPenalty()
    rank: int = 12


@dataclass
class ClassCfg:
    H: int
    W: int
    model_type: ClassModelType = ClassModelType.MIL
    hidden: int = 64
    c_out: int = 2


type Goal = SpectraCfg | ClassCfg


def get_model(
    goal: Goal,
    c_in: int,
    device: torch.device,
) -> tuple[Model, Penalty]:
    """Instantiate the spectral mapper and its regularizer for the requested model type.

    Args:
        c_in: Number of spectral channels available in the input tensors.
        rank: Rank used for low-rank models (ignored for linear CNN variant).
        device: Torch device to move the model and penalty onto.

    Returns:
        Tuple comprised of the spectral mapper module and an accompanying penalty.
    """
    model: Model
    if isinstance(goal, SpectraCfg):
        rank = goal.rank
        band_penalty = goal.band_penalty
        fold_stat = goal.fold_stat
        match goal.model_type:
            case SpecModelType.LSM:
                model = LinearSpectralMapper(c_in).to(device)
                # initialize linear map for CNN only
                init_linear_map(
                    fold_stat.train_raw_z, fold_stat.train_prc_z, band_penalty.id, model
                )
                penalty = band_penalty.make_penalty(model, c_in)
            case SpecModelType.LRSM:
                model = LowRankSpectralMapper(c_in, rank).to(device)
                penalty = band_penalty.make_penalty(model, c_in)
    elif isinstance(goal, ClassCfg):  # pyright: ignore[reportUnnecessaryIsInstance]
        match goal.model_type:
            case ClassModelType.CONV:
                model = ConvSpectralClassifier(c_in, goal.c_out, goal.H, goal.W).to(
                    device
                )
                penalty = Penalty(None)
            case ClassModelType.MIL:
                model = MILMeanHead(c_in, goal.hidden).to(device)
                penalty = Penalty(None)
    else:
        raise NotImplementedError()  # pyright: ignore[reportUnreachable]

    return model, penalty


def get_loss_managers(
    model: nn.Module,
    lr: float = 5e-4,
    wd: float = 1e-4,
) -> tuple[nn.CrossEntropyLoss | nn.MSELoss, torch.optim.Optimizer]:
    """Create the optimizer and loss function directly from object."""
    if isinstance(model, ConvSpectralClassifier):
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    return (
        loss_fn,
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
    )


@dataclass
class TrainSetup:
    loss_fn: nn.CrossEntropyLoss | nn.MSELoss
    optimizer: torch.optim.Optimizer
    model: Model
    penalty: Penalty


def get_train_setup(
    c_in: int,
    device: torch.device,
    lr: float,
    wd: float,
    goal: ClassCfg | SpectraCfg,
) -> TrainSetup:
    model, penalty = get_model(goal, c_in, device)
    loss_fn, optimizer = get_loss_managers(model, lr, wd)
    return TrainSetup(loss_fn, optimizer, model, penalty)
