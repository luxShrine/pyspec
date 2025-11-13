from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import NewType

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
READY_DATA_DIR: Path = DATA_DIR / "ready"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJ_ROOT / "models"

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# -- types & defaults ----------------------------------------------------


type ArrayF = npt.NDArray[np.float64]
type ArrayF32 = npt.NDArray[np.float32]
type ArrayI = npt.NDArray[np.int64]
type IndexArray = npt.NDArray[np.intp]
Std = NewType("Std", float)
Mean = NewType("Mean", float)
StdArray = NewType("StdArray", npt.NDArray[np.float32])
MeanArray = NewType("MeanArray", npt.NDArray[np.float32])

RNG = np.random.default_rng(42)


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
    MIL = auto()
    CONV = auto()


# -- physics types -------------------------------------------------------

REF_PHE = 1003.0


class SetType(StrEnum):
    PROCESSED = auto()
    RAW = auto()
    UNKNOWN = auto()


# with tqdm installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(
        lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True, diagnose=False
    )
    logger.add(
        "app.log", rotation="1 week", level="DEBUG", enqueue=True, serialize=True
    )  # file
except ModuleNotFoundError:
    pass
