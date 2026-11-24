"""Filesystem locations used by the RCBR-YOLO scripts.

The repository deliberately contains no machine-specific absolute path.  Every
location can be overridden with an environment variable, which is the only
supported way to point the scripts at a local dataset or checkpoint:

    RCBR_ROOT          repository root (default: parent directory of ``src``)
    RCBR_DATA_YAML     dataset yaml used by Ultralytics (default: <root>/data/data.yaml)
    RCBR_ANN_DIR       directory with COCO ``instances_{train,val,test}.json``
                       (default: <root>/data/annotations)
    RCBR_WEIGHTS       starting checkpoint (default: <root>/weights/yolo26m.pt)
    RCBR_RUNS          output directory for trained runs (default: <root>/runs)
    RCBR_REPORTS       output directory for generated reports (default: <root>/reports)
    RCBR_STATE         output directory for reload artefacts (default: <root>/state)
    RCBR_TRAIN_IMAGES  training image directory
    RCBR_VAL_IMAGES    validation image directory
    RCBR_TEST_IMAGES   test image directory

The defaults follow the layout described in ``README.md``; they are created on
demand by the scripts, so a fresh clone only needs ``data/`` and ``weights/``
filled in.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RCBR_ROOT", Path(__file__).resolve().parents[1])).resolve()

DATA_YAML = Path(os.environ.get("RCBR_DATA_YAML", ROOT / "data" / "data.yaml"))
ANN_DIR = Path(os.environ.get("RCBR_ANN_DIR", ROOT / "data" / "annotations"))
WEIGHTS = Path(os.environ.get("RCBR_WEIGHTS", ROOT / "weights" / "yolo26m.pt"))

TRAIN_IMAGES = Path(os.environ.get("RCBR_TRAIN_IMAGES", ROOT / "data" / "images" / "train"))
VAL_IMAGES = Path(os.environ.get("RCBR_VAL_IMAGES", ROOT / "data" / "images" / "val"))
TEST_IMAGES = Path(os.environ.get("RCBR_TEST_IMAGES", ROOT / "data" / "images" / "test"))

RUNS = Path(os.environ.get("RCBR_RUNS", ROOT / "runs"))
REPORTS = Path(os.environ.get("RCBR_REPORTS", ROOT / "reports"))
STATE = Path(os.environ.get("RCBR_STATE", ROOT / "state"))


def require(path: Path, what: str) -> Path:
    """Raise a readable error instead of a bare ``FileNotFoundError``."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{what} not found: {path}\n"
            "Set the corresponding RCBR_* environment variable or place the file "
            "in the default location (see README.md)."
        )
    return path


def ensure_output_dirs() -> None:
    for directory in (RUNS, REPORTS, STATE):
        directory.mkdir(parents=True, exist_ok=True)
