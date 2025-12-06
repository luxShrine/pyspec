from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import numpy as np

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

RNG = np.random.default_rng(42)
REF_PHE = 1003.0

# with tqdm installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(
        lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True, diagnose=False
    )
    # file
    logger.add(
        f"{PROJ_ROOT}/app.log",
        rotation="1 week",
        retention="10 days",
        level="DEBUG",
        enqueue=True,
        serialize=True,
    )
except ModuleNotFoundError:
    pass
