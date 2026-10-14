"""Shared constants and small helpers used across the repo."""
from __future__ import annotations

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
