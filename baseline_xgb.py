"""M1: XGBoost on per-lead statistics (SPEC section 3, model list).

Features per lead: mean, std, min, max, skew, kurtosis, dominant frequency, peak-to-peak -> 12 x 8 = 96.
A lead is missing when it is flat (std < 1e-6, detected from the signal): all its features are NaN,
which XGBoost handles natively. One binary classifier per class, early-stopped on fold 9.

predict_xgb returns {"prob": (N, 5)}, the same key and shape as the neural models' output.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data import load_dataset
from utils import CLASSES, FS, REPO_DIR

FEATURES = ["mean", "std", "min", "max", "skew", "kurtosis", "dom_freq", "ptp"]
FLAT_STD = 1e-6


def lead_features(x: np.ndarray) -> np.ndarray:
    """x: (N, 12, T) -> features (N, 96), ordered lead-major (lead 0's 8 features first). NaN for flat leads."""
    x = x.astype(np.float64)
    std = x.std(-1)
    flat = std < FLAT_STD
    spec = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1))
    dom = spec[..., 1:].argmax(-1) + 1  # skip DC
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.stack([x.mean(-1), std, x.min(-1), x.max(-1), skew(x, axis=-1), kurtosis(x, axis=-1),
                      dom * FS / x.shape[-1], x.max(-1) - x.min(-1)], axis=-1)  # (N, 12, 8)
    f[flat] = np.nan
    return f.reshape(len(x), -1).astype(np.float32)


def train_xgb(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
              seed: int = 0) -> list[XGBClassifier]:
    """Fit one binary classifier per class. x: (N, 12, T), y: (N, 5)."""
    f_tr, f_va = lead_features(x_tr), lead_features(x_va)
    models = []
    for k in range(y_tr.shape[1]):
        m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                          early_stopping_rounds=30, random_state=seed, n_jobs=-1)
        m.fit(f_tr, y_tr[:, k], eval_set=[(f_va, y_va[:, k])], verbose=False)
        models.append(m)
    return models


def predict_xgb(models: list[XGBClassifier], x: np.ndarray) -> dict:
    """x: (N, 12, T) -> {"prob": (N, 5)}."""
    f = lead_features(x)
    return {"prob": np.stack([m.predict_proba(f)[:, 1] for m in models], axis=1).astype(np.float32)}


def save_xgb(models: list[XGBClassifier], path: Path, meta: dict) -> None:
    """Atomic write (temp file + rename). meta records the data setting, e.g. {"data": ..., "debug": ...}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"models": models, **meta}, fh)
    os.replace(tmp, path)


def load_xgb(path: Path) -> tuple[list[XGBClassifier], dict]:
    """Returns (models, meta)."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return d.pop("models"), d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()

    X, y, fold = load_dataset(a.data, a.data_root, debug=a.debug)
    tr, va, te = fold <= 8, fold == 9, fold == 10
    t0 = time.time()
    models = train_xgb(X[tr], y[tr], X[va], y[va], seed=a.seed)
    print(f"trained {len(models)} classifiers on {tr.sum()} records in {time.time() - t0:.1f}s "
          f"(trees per class: {[m.best_iteration + 1 for m in models]})")

    x_drop = X[te].copy()
    x_drop[:, 6:] = 0  # precordial leads missing -> NaN features
    nan_frac = np.isnan(lead_features(x_drop)).mean()
    for name, xs, ys in [("val clean", X[va], y[va]), ("test clean", X[te], y[te]),
                         ("test V1-V6 zeroed", x_drop, y[te])]:
        prob = predict_xgb(models, xs)["prob"]
        per = [roc_auc_score(ys[:, k], prob[:, k]) for k in range(5)]
        print(f"{name:18s} prob {prob.shape} | macro-AUROC {np.mean(per):.3f} | "
              + " ".join(f"{c}={v:.2f}" for c, v in zip(CLASSES, per)))
    print(f"NaN fraction of features with V1-V6 zeroed: {nan_frac:.2f} (expected 0.50)")

    path = Path(a.ckpt_dir) / f"xgb_s{a.seed}.pkl"
    save_xgb(models, path, {"data": a.data, "debug": a.debug, "seed": a.seed})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
