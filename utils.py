"""Shared constants and small helpers used across the repo."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent

# Fixed orders (SPEC): never reorder these.
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 100  # sampling rate in Hz
N_SAMPLES = 1000  # 10 s at 100 Hz


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_auroc(y: np.ndarray, prob: np.ndarray) -> float:
    """Mean AUROC over classes; classes with only one label value in y are skipped. y, prob: (N, 5)."""
    from sklearn.metrics import roc_auc_score

    scores = [roc_auc_score(y[:, k], prob[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    return float(np.mean(scores)) if scores else float("nan")


def atomic_save(obj, path: Path) -> None:
    """torch.save to a temp file, then rename over the target, so an interrupted write never leaves a
    half-written checkpoint behind (important on Colab / Google Drive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
