from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, override

from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from loguru import logger
import numpy as np
import torch
from torch import Tensor, nn

from pyspectral.config import MODELS_DIR
from pyspectral.data.features import FoldStat
from pyspectral.types import ArrayF32, ClassModelType, SpecModelType

# tensor dimension checks
type BatchTensor = Float[Tensor, "B C"]
type ClassTensor = Float[Tensor, "B F"]
type OneDimTensor = Float[Tensor, "1"]
type ZeroDimTensor = Float[Tensor, ""]

type FeatTensor = Float[Tensor, "B T D"]
type ReFeatTensor = Float[Tensor, "B D T"]
type MaskTensor = Bool[Tensor, "B T"]

type BatchPrep = tuple[tuple[Tensor, ...], Tensor, Tensor | None]
type ModelOutput = nn.Module | Tensor | tuple | list


def _get_pt_file(directory: Path = MODELS_DIR, fold_number: int = 0) -> Path:
    models = list(directory.glob("*.pt"))
    options = len(models)
    mod_list = filter(lambda x: str(fold_number) in x.as_posix(), models)
    if options == 0:
        raise FileNotFoundError(f"No '.pt' files found in {MODELS_DIR}")
    elif options == 1:
        model = models[0]
        logger.debug(f"Only one model found, loading model at: {model}")
        return model
    else:
        while True:
            for m in mod_list:
                print(f"Use the model at: {m}")
                inp = input("Enter (y/n): ").lower()
                if inp == "y":
                    return m
                else:
                    continue

            inp2 = input("No model selected, go through options again? (y/n):").lower()
            if inp2 == "y":
                continue
            else:
                sys.exit()


def _prepare_batch(
    batch: tuple | list,
    device: torch.device,
    model: Model,
) -> BatchPrep:
    if len(batch) == 2:
        spectra, target = batch
        inputs = (spectra.to(device, non_blocking=True),)
        return inputs, target.to(device, non_blocking=True), None
    if len(batch) == 4:
        feats, mask, target, scene_idx = batch
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if isinstance(model, MILMeanHead) or isinstance(model, MILMeanHeadMulti):
            inputs = (feats, mask)
        else:
            inputs = (feats,)
        return inputs, target, scene_idx
    raise TypeError(f"Unsupported batch structure of length {len(batch)}")


def _prediction_from_output(output: ModelOutput) -> Tensor:
    """Unpack model outputs, returning the tensor used for loss/pred storage."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        first = output[0]
        if isinstance(first, Tensor):
            return first
    raise TypeError(f"Unsupported model output type: {type(output)}")


def get_tensor(maybe_tens: Tensor | Any) -> Tensor:
    """Check that input is a tensor, return said tensor or raise TypeError."""
    if isinstance(maybe_tens, Tensor):
        return maybe_tens
    else:
        raise TypeError(f"Input is not a tensor: {type(maybe_tens)}")


# -- multiple-instance learning model
# replace with max/noisy-OR/attention if they are better


class MILMeanHeadMulti(nn.Module):
    """Predict each pixel, yes, no, maybe."""

    def __init__(self, d_in: int, hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.n_classes: int = n_classes
        self.ff: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),  # per-pixel logits for C classes
        )

    def prepare(self, batch: tuple | list, device: torch.device) -> BatchPrep:
        return _prepare_batch(batch, device, self)

    @jaxtyped(typechecker=beartype)
    @override
    def forward(
        self,
        X: FeatTensor,  # (B, T, D)
        mask: MaskTensor,  # (B, T), bool
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            bag_logits: (B, C)    # for CrossEntropyLoss
            pix_logits: (B, T, C) # per-pixel logits
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("Non-finite features in X")

        pix_logits: Float[Tensor, "B T C"] = self.ff(X)  # raw logits

        if torch.isnan(pix_logits).any() or torch.isinf(pix_logits).any():
            raise ValueError("Non-finite logits pix_logits")

        # masked mean pooling over T -> bag-level logits
        mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
        denom: Float[Tensor, "B 1"] = mask_f.sum(dim=1).clamp_min(1.0)
        bag_logits: Float[Tensor, "B C"] = (pix_logits * mask_f).sum(dim=1) / denom

        return bag_logits, pix_logits

    def pred(self, input: tuple[Tensor, ...]) -> Tensor:
        return _prediction_from_output(self(*input))

    @classmethod
    def load(
        cls, d_in: int, *, hidden: int = 64, n_classes: int = 3, fold_number: int = 0
    ) -> MILMeanHeadMulti:
        model_path = _get_pt_file(fold_number=fold_number)
        model = cls(d_in, hidden=hidden, n_classes=n_classes)
        model.load_state_dict(torch.load(model_path))
        return model


class MILMeanHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.ff: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # per-pixel logit
        )

    def prepare(self, batch: tuple | list, device: torch.device) -> BatchPrep:
        return _prepare_batch(batch, device, self)

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, X: FeatTensor, mask: MaskTensor):
        """
        Args:
            X: (B,T,D)
            mask: (B,T)
        """

        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("Non-finite features in X")
        z: Float[Tensor, "B T"] = self.ff(X).squeeze(-1)
        if torch.isnan(z).any() or torch.isinf(z).any():
            raise ValueError("Non-finite logits z")
        p: Float[Tensor, "B T"] = torch.sigmoid(z)
        if torch.isnan(p).any():
            raise ValueError("Non-finite probabilities p")
        # masked mean pooling -> map probability
        denom: Float[Tensor, "B"] = mask.float().sum(dim=1).clamp_min(1.0)
        xnum = p * mask
        num = xnum.sum(dim=1)
        prob: Float[Tensor, "B"] = num / denom  # (B,)
        return prob, z, p

    def pred(self, input: tuple[Tensor, ...]) -> Tensor:
        return _prediction_from_output(self(*input))

    @classmethod
    def load(cls, d_in: int, *, hidden: int = 64, fold_number: int = 0) -> MILMeanHead:
        model_path = _get_pt_file(fold_number=fold_number)
        model = cls(d_in, hidden=hidden)
        model.load_state_dict(torch.load(model_path))
        return model


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

    def prepare(self, batch: tuple | list, device: torch.device) -> BatchPrep:
        return _prepare_batch(batch, device, self)

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, x: BatchTensor) -> ClassTensor:
        """
        (B, C, H, W) -> returns (B, C_out)
        """
        features: Float[Tensor, "B C2 H2 W2"] = self.features(x)
        y: ClassTensor = self.head(features)
        return y

    def pred(self, input: tuple[Tensor, ...]) -> Tensor:
        return _prediction_from_output(self(*input))


type ClassMapper = ConvSpectralClassifier | MILMeanHead | MILMeanHeadMulti
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

    def prepare(self, batch: tuple | list, device: torch.device) -> BatchPrep:
        return _prepare_batch(batch, device, self)

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

    def pred(self, input: tuple[Tensor, ...]) -> Tensor:
        return _prediction_from_output(self(*input))


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

    def prepare(self, batch: tuple | list, device: torch.device) -> BatchPrep:
        return _prepare_batch(batch, device, self)

    @jaxtyped(typechecker=beartype)
    @override
    def forward(self, x: BatchTensor) -> BatchTensor:
        return get_tensor(self.proj(x))

    def pred(self, input: tuple[Tensor, ...]) -> Tensor:
        return _prediction_from_output(self(*input))


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
    d_in: int
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
                    fold_stat.tr_x_znorm.z,
                    fold_stat.tr_y_znorm.z,
                    band_penalty.id,
                    model,
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
            case ClassModelType.MILMULTI:
                d_in = goal.d_in
                model = MILMeanHeadMulti(d_in, goal.hidden).to(device)
                penalty = Penalty(None)
            case ClassModelType.MIL:
                d_in = goal.d_in
                model = MILMeanHead(d_in, goal.hidden).to(device)
                penalty = Penalty(None)
    else:
        raise NotImplementedError(f"{type(goal)}")  # pyright: ignore[reportUnreachable]

    return model, penalty


def get_loss_managers(
    model: nn.Module,
    lr: float = 5e-4,
    wd: float = 1e-4,
) -> tuple[nn.CrossEntropyLoss | nn.BCELoss | nn.MSELoss, torch.optim.Optimizer]:
    """Create the optimizer and loss function."""
    loss_fn: nn.CrossEntropyLoss | nn.BCELoss | nn.MSELoss
    if isinstance(model, ConvSpectralClassifier) or isinstance(model, MILMeanHeadMulti):
        loss_fn = nn.CrossEntropyLoss()
    elif isinstance(model, MILMeanHead):
        loss_fn = nn.BCELoss()
    else:
        loss_fn = nn.MSELoss()
    return (
        loss_fn,
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
    )


@dataclass
class TrainSetup:
    loss_fn: nn.CrossEntropyLoss | nn.MSELoss | nn.BCELoss
    optimizer: torch.optim.Optimizer
    model: Model
    penalty: Penalty

    def log_init(self, fold: int) -> None:
        if fold == 1:
            details = f"model type: {self.model} \n"
            details += f"loss function: {self.loss_fn} | optimizer: {self.optimizer}\n"
            logger.debug(details)

    @classmethod
    def get_train_setup(
        cls,
        c_in: int,
        device: torch.device,
        lr: float,
        wd: float,
        goal: ClassCfg | SpectraCfg,
    ) -> TrainSetup:
        model, penalty = get_model(goal=goal, c_in=c_in, device=device)
        loss_fn, optimizer = get_loss_managers(model, lr, wd)
        return cls(loss_fn, optimizer, model, penalty)
