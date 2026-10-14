"""Data contract (SPEC section 1). The only module that reads raw dataset files.

load_dataset(name, root=None, debug=False, return_raw=False)
    -> (X, y, fold)            or (X, y, fold, X_raw) with return_raw=True
X:     float32 (N, 12, 1000), per-lead z-score per record, lead order utils.LEADS
y:     float32 (N, 5) multi-hot, class order utils.CLASSES
fold:  int64 (N,), train 1-8, validation 9, test 10
X_raw: float32 (N, 12, 1000), un-normalized signal in mV
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from utils import CLASSES, FS, LEADS, N_SAMPLES, REPO_DIR

DEBUG_N = 500
Z_EPS = 1e-6


def zscore(x_raw: np.ndarray) -> np.ndarray:
    """Per-lead, per-record z-score. x_raw: (N, 12, T). Flat leads (std ~ 0) become all zeros."""
    mean = x_raw.mean(axis=-1, keepdims=True)
    std = x_raw.std(axis=-1, keepdims=True)
    out = (x_raw - mean) / np.maximum(std, Z_EPS)
    out[np.broadcast_to(std < Z_EPS, out.shape)] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------- synthetic

# Beat components: (name, offset from R peak in s, amplitude in mV, width in s).
_COMPONENTS = [("P", -0.20, 0.15, 0.025), ("Q", -0.035, -0.10, 0.010), ("R", 0.0, 1.0, 0.012),
               ("S", 0.035, -0.25, 0.012), ("T", 0.30, 0.30, 0.050)]
# Per-lead gain, roughly mimicking the real projection of the heart's axis (aVR inverted, V1 mostly negative).
_LEAD_GAIN = np.array([0.8, 1.0, 0.5, -0.9, 0.4, 0.7, -0.6, -0.3, 0.4, 0.9, 1.0, 0.8])


def _idx(names: list[str]) -> list[int]:
    return [LEADS.index(n) for n in names]


def _synthetic(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECG-like synthetic records whose labels change the waveform on specific leads.

    So that a working pipeline scores above chance on CPU and losing the right leads hurts
    (each effect has a random per-record strength, so weak positives resemble normal records):
      MI   -> deeper Q wave on V1-V4
      STTC -> flattened / inverted T wave on I, aVL, V5, V6
      CD   -> widened QRS on II, III, V1, V2
      HYP  -> taller R wave on aVL, V4-V6
    Returns X_raw (n, 12, 1000) in mV, y (n, 5), fold (n,).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS

    # Labels: NORM alone (40%), otherwise 1+ pathologies, each with p=0.4.
    y = np.zeros((n, len(CLASSES)), dtype=np.float32)
    is_norm = rng.random(n) < 0.4
    y[is_norm, 0] = 1
    path = (rng.random((n, 4)) < 0.4).astype(np.float32)
    empty = path.sum(1) == 0
    path[empty, rng.integers(0, 4, empty.sum())] = 1
    y[~is_norm, 1:] = path[~is_norm]

    x_raw = np.zeros((n, 12, N_SAMPLES), dtype=np.float32)
    for i in range(n):
        rr = 60.0 / rng.uniform(55, 100)
        beats = np.arange(rng.uniform(0, rr), t[-1] + 0.5, rr) + rng.normal(0, 0.01, 1)
        dt = t[None, :] - beats[:, None]  # (n_beats, T)
        gain = _LEAD_GAIN * rng.uniform(0.7, 1.3, 12)
        # Effect strength per class in [0, 1): weak cases look almost normal, so the task is not trivial.
        s = rng.uniform(0, 1, 5) * y[i]
        sig = np.zeros((12, N_SAMPLES))
        for name, mu, amp, width in _COMPONENTS:
            a = np.full(12, amp) * rng.uniform(0.8, 1.2, 12)
            w = np.full(12, width)
            if name == "Q":
                j = _idx(["V1", "V2", "V3", "V4"])
                a[j] = a[j] * (1 - s[1]) - 0.6 * s[1]
            if name == "T":
                a[_idx(["I", "aVL", "V5", "V6"])] *= 1 - 2 * s[2]
            if name in ("Q", "R", "S"):
                w[_idx(["II", "III", "V1", "V2"])] *= 1 + 1.5 * s[3]
            if name == "R":
                a[_idx(["aVL", "V4", "V5", "V6"])] *= 1 + s[4]
            g = np.exp(-((dt[None] - mu) ** 2) / (2 * w[:, None, None] ** 2)).sum(1)  # (12, T)
            sig += a[:, None] * g
        sig *= gain[:, None]
        wander = 0.05 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 2 * np.pi))
        sig += wander[None] + rng.normal(0, 0.03, sig.shape)
        x_raw[i] = sig

    fold = (np.arange(n) % 10 + 1).astype(np.int64)
    return x_raw, y, fold


# -------------------------------------------------------------------- PTB-XL

def _ptbxl_labels(root: Path):
    """Read metadata; return (filenames, y, fold) for records with at least one superclass."""
    import pandas as pd

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    stm = pd.read_csv(root / "scp_statements.csv", index_col=0)
    stm = stm[stm["diagnostic"] == 1]
    code_to_class = stm["diagnostic_class"].to_dict()

    y = np.zeros((len(db), len(CLASSES)), dtype=np.float32)
    for r, codes in enumerate(db["scp_codes"]):
        for code in ast.literal_eval(codes):
            cls = code_to_class.get(code)
            if cls in CLASSES:
                y[r, CLASSES.index(cls)] = 1
    keep = y.sum(1) > 0
    return db["filename_lr"].to_numpy()[keep], y[keep], db["strat_fold"].to_numpy()[keep].astype(np.int64)


def _ptbxl(root: Path, debug: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read PTB-XL 100 Hz waveforms. Returns X_raw (N, 12, 1000) mV, y, fold."""
    import wfdb
    from tqdm import tqdm

    files, y, fold = _ptbxl_labels(root)
    if debug:
        files, y, fold = files[:DEBUG_N], y[:DEBUG_N], fold[:DEBUG_N]
    upper = [l.upper() for l in LEADS]
    x_raw = np.zeros((len(files), 12, N_SAMPLES), dtype=np.float32)
    for i, f in enumerate(tqdm(files, desc="reading PTB-XL", mininterval=5)):
        sig, meta = wfdb.rdsamp(str(root / f))
        order = [[s.upper() for s in meta["sig_name"]].index(l) for l in upper]
        x_raw[i] = np.nan_to_num(sig[:N_SAMPLES, order].T)
    return x_raw, y, fold


# ------------------------------------------------------------------- public

def load_dataset(name: str, root: str | Path | None = None, debug: bool = False,
                 return_raw: bool = False, cache_dir: str | Path | None = None):
    """Load a dataset following the contract in the module docstring.

    The cache stores the raw mV signal; the z-scored copy is computed on load. Only with
    return_raw=True are two full copies held in memory (meant for the demo / small sets).
    """
    if name == "synthetic":
        x_raw, y, fold = _synthetic()
        if debug:
            x_raw, y, fold = x_raw[:DEBUG_N], y[:DEBUG_N], fold[:DEBUG_N]
    elif name == "ptbxl":
        cache_dir = Path(cache_dir) if cache_dir else REPO_DIR / "cache"
        cache = cache_dir / ("ptbxl_debug.npz" if debug else "ptbxl.npz")
        full_cache = cache_dir / "ptbxl.npz"
        if debug and not cache.exists() and full_cache.exists():
            cache = full_cache  # slice the full cache instead of re-reading raw files
        if cache.exists():
            d = np.load(cache)
            x_raw, y, fold = d["X_raw"], d["y"], d["fold"]
            if debug:
                x_raw, y, fold = x_raw[:DEBUG_N], y[:DEBUG_N], fold[:DEBUG_N]
        else:
            if root is None:
                raise ValueError("ptbxl needs --root (folder containing ptbxl_database.csv) when no cache exists")
            x_raw, y, fold = _ptbxl(Path(root), debug)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, X_raw=x_raw, y=y, fold=fold)
            print(f"cached to {cache}")
    else:
        raise ValueError(f"unknown dataset: {name}")

    x = zscore(x_raw)
    if return_raw:
        return x, y, fold, x_raw.astype(np.float32)
    del x_raw
    return x, y, fold


def _summary(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> None:
    print(f"X: {x.shape} {x.dtype}  y: {y.shape} {y.dtype}  fold: {fold.shape} {fold.dtype}")
    print(f"X finite: {bool(np.isfinite(x).all())}  mean {x.mean():.3f}  std {x.std():.3f}")
    print("label counts: " + ", ".join(f"{c}={int(n)}" for c, n in zip(CLASSES, y.sum(0))))
    print(f"labels per record: mean {y.sum(1).mean():.2f}, records with none: {int((y.sum(1) == 0).sum())}")
    f, n = np.unique(fold, return_counts=True)
    print("records per fold: " + ", ".join(f"{a}:{b}" for a, b in zip(f, n)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Load a dataset and print a summary.")
    p.add_argument("--name", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--root", default=None, help="PTB-XL folder (with ptbxl_database.csv)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    _summary(*load_dataset(a.name, a.root, a.debug, cache_dir=a.cache_dir))
