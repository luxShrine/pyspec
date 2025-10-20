from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, NewType

from dotenv import load_dotenv
from loguru import logger
import numpy as np
import numpy.typing as npt

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJ_ROOT / "models"

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# -- types & defaults ----------------------------------------------------


type ArrayF = npt.NDArray[np.float64]
type ArrayF32 = npt.NDArray[np.float32]
Std = NewType("Std", float)
Mean = NewType("Mean", float)
StdArray = NewType("StdArray", npt.NDArray[np.float32])
MeanArray = NewType("MeanArray", npt.NDArray[np.float32])

RNG = np.random.default_rng(42)


class PlotType(StrEnum):
    TRAIN = auto()
    TEST = auto()
    BOTH = auto()


class ModelType(StrEnum):
    LSM = auto()
    LRSM = auto()


# -- physics types -------------------------------------------------------


class SetType(StrEnum):
    PROCESSED = auto()
    RAW = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class FlatMap:
    """A (N,) grid with (C) values representing the number spectrum recordings.

    This is the *flattened* representation of the Cube, where (N == H*W) and
    the number of channels is the number of bands (C == M).
    """

    _array: ArrayF32
    N: int
    C: int

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.ArrayLike:
        return np.asarray(self._array, dtype=dtype)  # enables np.asarray(Cube)

    def __len__(self) -> int:
        return self.N

    def __repr__(self) -> str:
        return f"Cube(shape={self.shape}, dtype={self._array.dtype})"

    def __getitem__(self, key: slice | int) -> Any | ArrayF32:
        if isinstance(key, int):
            return self._array[key]
        elif isinstance(key, slice):
            return self._array[key]
        else:
            raise TypeError("Invalid key type")

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the map."""
        return (self.N, self.C)

    def get(self) -> ArrayF32:
        """Get the underlying flat array explicitly."""
        return self._array

    @classmethod
    def make(cls, array: ArrayF32) -> FlatMap:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Must be a numpy array, got {type(array)=}")
        if array.ndim != 2:
            raise ValueError(f"Cube must be 3D (H, W, M). Got shape={array.shape!r}")
        N, C = array.shape
        return cls(_array=array, N=N, C=C)


@dataclass(frozen=True, slots=True)
class Cube:
    """A (H,W) grid with (M) values representing the number spectrum recordings."""

    _cube: ArrayF32
    H: int
    W: int
    M: int

    def __getitem__(self, key: slice | int) -> Any | ArrayF32:
        if isinstance(key, int):
            return self._cube[key]
        elif isinstance(key, slice):
            return self._cube[key]
        else:
            raise TypeError("Invalid key type")

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.ArrayLike:
        return np.asarray(self._cube, dtype=dtype)  # enables np.asarray(Cube)

    def __len__(self) -> int:
        return self.H

    def __repr__(self) -> str:
        return f"Cube(shape={self.shape}, dtype={self._cube.dtype})"

    def get(self) -> ArrayF32:
        """Get the underlying cube explicitly."""
        return self._cube

    def flatten(self) -> FlatMap:
        """Reshape the cube to flat array: from (H, W, M) -> (H*W, M)."""
        return FlatMap.make(self._cube.reshape(-1, self.M).astype(np.float32, copy=False))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the cube."""
        return (self.H, self.W, self.M)

    @classmethod
    def make(cls, cube: ArrayF32) -> Cube:
        """Create a spectral Cube instance from a proper numpy array."""
        if not isinstance(cube, np.ndarray):
            raise TypeError(f"Cube must be a numpy array, got {type(cube)=}")
        if cube.ndim != 3:
            raise ValueError(f"Cube must be 3D (H, W, M). Got shape={cube.shape!r}")
        # grab the three dimensions
        H, W, M = cube.shape
        return cls(_cube=cube, H=H, W=W, M=M)

    @classmethod
    def from_flat(cls, flat_cube: ArrayF32, height: int, width: int, spec_bands: int) -> Cube:
        """Reshape flat array to cube: from (H*W, M) -> (H, W, M)."""
        return cls.make(flat_cube.reshape(height, width, spec_bands).astype(np.float32))


# with tqdm installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True, diagnose=False)
    logger.add("app.log", rotation="1 week", level="DEBUG", enqueue=True, serialize=True)  # file
except ModuleNotFoundError:
    pass
