from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from pyspectral.types import (
    Arr1DF,
    Arr2DF,
    Arr3DF32,
    ArrayF,
    ArrayF32,
    MeanArray,
    StdArray,
)


def apply_z(y: np.ndarray, mean: MeanArray, std: StdArray) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def assert_same_grid(a: ArrayF, b: ArrayF, tolerance: float = 1e-9) -> None | str:
    """Check that array 'a' and 'b' are the same shape and each element is within a tolerance."""
    # add to a single string so that we can see both issues potentially
    output: str = ""
    if a.shape != b.shape:
        output += f"Grid {a.shape=} not equal to {b.shape=}.\n"
    if not np.allclose(a, b, atol=tolerance):
        # np.allclose checks if the element by element array are the samef within tolerance
        output += f"Grids differ by supplied tolerance: {tolerance}."
    if output != "":
        return output
    return None


@dataclass
class TruePredPair:
    true: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True, slots=True)
class TestResult:
    test_loss: float
    fraction_improved: float
    predictions: list[ArrayF32]


@dataclass(frozen=True, slots=True)
class EpochLoss:
    train: Arr1DF  # shape: (Epoch count)
    test: Arr1DF


@dataclass(frozen=True, slots=True)
class FlatMap:
    """A (N,) grid with (C) values representing the number spectrum recordings.

    This is the *flattened* representation of the Cube, where (N == H*W) and
    the number of channels is the number of bands (C == M).
    """

    _array: Arr2DF
    N: int
    C: int

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.ArrayLike:
        return np.asarray(self._array, dtype=dtype)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, key: slice | int) -> Any | ArrayF32:
        return self._array[key]

    def __iter__(self):
        yield from self._array

    def get_pixels(self) -> list[Arr1DF]:
        pixel_indices = range(self.N)
        return [self._array[i] for i in pixel_indices]

    @classmethod
    def from_dict(cls, d: dict) -> FlatMap:
        return cls(_array=np.array(d["_array"]), N=d["N"], C=d["C"])

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the map."""
        return (self.N, self.C)

    def get(self) -> Arr2DF:
        """Get the underlying flat array explicitly."""
        return self._array

    @classmethod
    def make(cls, array: np.ndarray) -> FlatMap:
        if array.ndim != 2:
            raise ValueError(f"Cube must be 3D (H, W, M). Got shape={array.shape!r}")
        N, C = array.shape
        return cls(_array=array, N=N, C=C)


class SpectraByClass(TypedDict):
    negatives: list[Arr1DF]
    maybe: list[Arr1DF]
    samples: list[Arr1DF]


@dataclass(frozen=True, slots=True)
class Cube:
    """A (H,W) grid with (M) values representing the number spectrum recordings."""

    _cube: Arr3DF32
    H: int
    W: int
    M: int

    def __getitem__(self, key: slice | int) -> Any | ArrayF32:
        return self._cube[key]

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.ArrayLike:
        return np.asarray(self._cube, dtype=dtype)  # enables np.asarray(Cube)

    def __sub__(self, other: Cube) -> Cube:
        diff = np.asarray(self) - np.asarray(other)
        return self.make(diff)

    def __add__(self, other: Cube) -> Cube:
        sum = np.asarray(self) + np.asarray(other)
        return self.make(sum)

    def __len__(self) -> int:
        return self.H

    def get(self) -> Arr3DF32:
        """Get the underlying cube explicitly."""
        return self._cube

    def flatten(self) -> FlatMap:
        """Reshape the cube to flat array: from (H, W, M) -> (H*W, M)."""
        return FlatMap.make(
            self._cube.reshape(-1, self.M).astype(np.float64, copy=False)
        )

    def resample_cube(self, wl_src: ArrayF, wl_dst: ArrayF) -> Cube:
        """Resample (H,W,M_src) -> (H,W,M_dst) by 1D linear interp per pixel."""
        flat = self.get().reshape(-1, self.M)
        out = np.empty((flat.shape[0], wl_dst.size), dtype=np.float32)
        for i in range(flat.shape[0]):  # NOTE: could vectorize this process
            out[i] = np.interp(wl_dst, wl_src, flat[i])
        return self.from_flat(
            flat_cube=out, height=self.H, width=self.W, spec_bands=wl_dst.size
        )

    def split_cube_by_label(
        self,
        labels: np.ndarray[tuple[int, int], np.dtype[np.uint]],
    ) -> SpectraByClass:
        """Flatten cube per class.

        Args:
            labels: 2D array, (H, W), of int values ∈ {0,3}
        """
        H, W, _ = self.shape
        if (c := labels.shape) != (H, W):
            raise ValueError(
                f"Shape of labels {c} is not equal to shape of cube {H=}, {W=}"
            )

        # Flatten spatial dims
        flat = self.flatten()
        flat_labels = labels.reshape(-1)  # (H*W,)

        class_names = {0: "negatives", 1: "maybe", 2: "samples"}
        spectra_by_class: SpectraByClass = {"negatives": [], "maybe": [], "samples": []}

        for spec, lab in zip(flat, flat_labels):
            spectra_by_class[class_names[int(lab)]].append(spec)

        return spectra_by_class

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the cube."""
        return (self.H, self.W, self.M)

    @classmethod
    def from_dict(cls, d: dict) -> Cube:
        return cls(_cube=np.array(d["_cube"]), H=d["H"], W=d["W"], M=d["M"])

    @classmethod
    def make(cls, cube: np.ndarray) -> Cube:
        """Create a spectral Cube instance from a 3D (H, W, M) numpy array."""
        if cube.ndim != 3:
            raise ValueError(f"Cube must be 3D (H, W, M). Got shape={cube.shape!r}")
        # grab the three dimensions
        H, W, M = cube.shape
        return cls(_cube=cube, H=H, W=W, M=M)

    @classmethod
    def from_flat(
        cls,
        flat_cube: np.ndarray[tuple[int, int]],
        height: int,
        width: int,
        spec_bands: int,
    ) -> Cube:
        """Reshape flat array to cube: from (H*W, M) -> (H, W, M)."""
        return cls.make(flat_cube.reshape(height, width, spec_bands).astype(np.float32))
