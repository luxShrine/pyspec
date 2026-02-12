from __future__ import annotations

from collections.abc import Generator
from enum import StrEnum, auto
from typing import NewType

import numpy as np
import numpy.typing as npt

# -- arrays ----------------------------------------------------


type ArrayF = npt.NDArray[np.float64]
type ArrayF32 = npt.NDArray[np.float32]
type ArrayI = npt.NDArray[np.int64]
type IndexArray = npt.NDArray[np.intp]
# shaped type arrays
type Arr1DF32 = np.ndarray[tuple[int], np.dtype[np.float32]]
type Arr1DF = np.ndarray[tuple[int], np.dtype[np.float64]]
type Arr2DF32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type Arr2DF = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Arr3DF32 = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type Arr3DF = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


# -- stats ----------------------------------------------------

Std = NewType("Std", float)
Mean = NewType("Mean", float)
StdArray = NewType("StdArray", npt.NDArray[np.float32])
MeanArray = NewType("MeanArray", npt.NDArray[np.float32])

type SplitIter = Generator[tuple[IndexArray, IndexArray]]
type ClassPixIndices = np.ndarray[tuple[int], np.dtype[np.int64]]
type KfoldPixIndices = tuple[
    ClassPixIndices, ClassPixIndices, ClassPixIndices, ClassPixIndices
]


type NPSigned = (
    np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.float32
    | np.float64
    | np.float96
    | np.float128
)

type ThirdCoeff = tuple[float, float, float]
type Numeric = int | float | NPSigned


class UnitFloat(float):
    """Numeric value within zero and one."""

    def __new__(cls, value: Numeric):
        if not (0 < value < 1):
            raise ValueError("UnitFloat must be strictly between 0 and 1.")
        return super().__new__(cls, value)


class SameFirstDimensionArrays:
    def __init__(self, array_1: np.ndarray, array_2: np.ndarray):
        if not (array_1.shape[0] == array_2.shape[0]):
            raise ValueError(
                f"SameFirstDimensionArrays must hold the same first dimension shape: {array_1.shape=}, {array_2.shape=}."
            )
        self.arr_1: np.ndarray = array_1
        self.arr_2: np.ndarray = array_2

    def tup(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.arr_1, self.arr_2)


class SameDimensionArrays:
    def __init__(self, array_1: np.ndarray, array_2: np.ndarray):
        if not (array_1.shape == array_2.shape):
            raise ValueError(
                f"SameFirstDimensionArrays must hold the same shapes: {array_1.shape=}, {array_2.shape=}."
            )
        self.arr_1: np.ndarray = array_1
        self.arr_2: np.ndarray = array_2

    def tup(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.arr_1, self.arr_2)


# -- physics/processing types -------------------------------------------------------


BaselinePolynomialDegree = NewType("BaselinePolynomialDegree", int)


class SpectralMode(StrEnum):
    RAMAN = auto()
    REFLECTANCE = auto()  # alternative method not used


class PlotType(StrEnum):
    TRAIN = auto()
    TEST = auto()
    BOTH = auto()


class SpecModelType(StrEnum):
    LSM = auto()
    LRSM = auto()


class ClassModelType(StrEnum):
    MILMULTI = auto()
    MIL = auto()
    CONV = auto()


class SetType(StrEnum):
    PROCESSED = auto()
    RAW = auto()
    UNKNOWN = auto()
