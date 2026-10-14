"""Save figures/corruption_examples.png: one record before and after each corruption (SPEC section 2)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import corrupt
from data import load_dataset
from utils import FS, LEADS, REPO_DIR

# Okabe-Ito (colorblind-safe): clean, mildly corrupted (severity < 0.5), bad (>= 0.5)
C_CLEAN, C_MILD, C_BAD = "#222222", "#E69F00", "#D55E00"
OFFSET = 6.0


def _panels(x: torch.Tensor, seed: int):
    g = lambda: torch.Generator().manual_seed(seed)
    three = torch.zeros(1, 12, dtype=torch.bool)
    three[0, [1, 6, 10]] = True  # II, V1, V5
    return [
        ("Clean", x.clone(), torch.zeros(1, 12)),
        ("Random drop (k = 4)", *corrupt.drop_random(x, 4, g())),
        ("Group drop: precordial (V1-V6)", *corrupt.drop_group(x, "precordial")),
        ("Flatline (II, V1, V5)", *corrupt.flatline(x, three)),
        ("EMG noise 10 dB, 20-45 Hz (II, V1, V5)", *corrupt.add_emg_noise(x, three, 10.0, g())),
        ("Baseline wander (II, V1, V5)", *corrupt.add_baseline_wander(x, three, gen=g())),
        ("Amplitude clipping (II, V1, V5)", *corrupt.clip_amplitude(x, three, gen=g())),
        ("Training mix (random draw)", *corrupt.training_corruption(x, g())),
    ]


def plot(x: np.ndarray, out: Path, seed: int = 0) -> None:
    """x: (12, T) z-scored record."""
    xt = torch.from_numpy(x)[None]
    panels = _panels(xt, seed)
    t = np.arange(x.shape[-1]) / FS
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharex=True, sharey=True)
    for ax, (title, xc, sev) in zip(axes.ravel(), panels):
        xc, sev = xc[0].numpy(), sev[0].numpy()
        for j in range(12):
            s = sev[j]
            color = C_CLEAN if s == 0 else (C_MILD if s < 0.5 else C_BAD)
            ax.plot(t, xc[j] - j * OFFSET, color=color, lw=0.7)
            if s > 0:
                ax.text(t[-1] + 0.1, -j * OFFSET, f"{s:.2f}", color=color, va="center", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_yticks([-j * OFFSET for j in range(12)], LEADS)
        ax.set_xlim(0, t[-1] + 0.9)
        ax.set_ylim(-11.8 * OFFSET, OFFSET)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in (C_CLEAN, C_MILD, C_BAD)]
    fig.legend(handles, ["untouched (severity 0)", "corrupted, severity < 0.5", "bad lead, severity >= 0.5"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Simulated lead corruptions on one z-scored record (numbers = severity)", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None, help="folder with ptbxl.npz (default: <repo>/cache)")
    p.add_argument("--index", type=int, default=0, help="record index to plot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_DIR / "figures" / "corruption_examples.png"))
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    X, y, fold = load_dataset(a.data, a.data_root, debug=a.debug, cache_dir=a.cache_dir)
    plot(X[a.index], Path(a.out), a.seed)
    print(f"saved {a.out} (record {a.index}, labels {y[a.index].astype(int).tolist()})")
