"""Figures 1-4 and Tables 1-2 (SPEC section 7). Reads only results/*.csv and results/attention.npz.

python figures.py [--results-dir results] [--fig-dir figures]
Every number shown comes from those files; seeds are aggregated as mean and std (std only where > 1 seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import LEADS, REPO_DIR

DPI = 300
# Variants in a fixed order; categorical hues assigned in that order (validated 8-hue palette).
# The two ResNets share a hue and differ by line style / hatch (composite encoding, 9 variants on 8 hues).
VARIANTS = [  # (key, label, color, linestyle, marker, hatch)
    ("lg_edl", "LG-EDL", "#2a78d6", "-", "o", ""),
    ("resnet_c1", "ResNet (corruption-trained)", "#eb6834", "-", "s", ""),
    ("resnet_c0", "ResNet (clean-trained)", "#eb6834", "--", "s", "//"),
    ("lg_soft", "LG-soft", "#1baf7a", "-", "^", ""),
    ("lg_hard", "LG-hard", "#eda100", "-", "v", ""),
    ("resnet_attn", "ResNet + lead attention", "#e87ba4", "-", "D", ""),
    ("transformer", "Patch Transformer", "#008300", "-", "P", ""),
    ("mlp", "MLP", "#4a3aa7", "-", "X", ""),
    ("xgb", "XGBoost", "#e34948", "-", "*", ""),
]
STYLE = {k: dict(label=l, color=c, ls=ls, marker=m, hatch=h) for k, l, c, ls, m, h in VARIANTS}
ABLATION = ["resnet_attn", "lg_hard", "lg_soft", "lg_edl"]  # M5-M8
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def variant(df: pd.DataFrame) -> pd.Series:
    """Model name, with ResNet split by its corrupt-train flag (taken from the run name, e.g. resnet_s0_c1)."""
    return np.where(df.model == "resnet", "resnet_" + df.run.str.rsplit("_", n=1).str[-1], df.model)


def per_seed(df: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    """Average over draws within each run, then mean / std / n over seeds per variant."""
    df = df.assign(variant=variant(df))
    run_level = df.groupby(["variant", "seed"] + by, as_index=False)[value].mean()
    g = run_level.groupby(["variant"] + by)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_seeds": g.size()}).reset_index()


PAPER_MODE = False  # True: leave out the "Figure N." titles, the LaTeX captions replace them


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    if not (PAPER_MODE and title.startswith("Figure")):
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def fig1(res: Path, out: Path) -> None:
    """Macro-AUROC vs number of missing leads."""
    s = per_seed(pd.read_csv(res / "random_loss.csv"), ["n_missing"], "macro_auroc")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for key in STYLE:
        d = s[s.variant == key].sort_values("n_missing")
        if d.empty:
            continue
        st = STYLE[key]
        lab = st["label"] + (f" (mean ± std, {d.n_seeds.max()} seeds)" if d.n_seeds.max() > 1 else "")
        ax.plot(d.n_missing, d["mean"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=1.8,
                label=lab, zorder=3 if key == "lg_edl" else 2)
        if d.n_seeds.max() > 1:
            ax.fill_between(d.n_missing, d["mean"] - d["std"], d["mean"] + d["std"], color=st["color"],
                            alpha=0.15, lw=0)
    _style(ax, "Figure 1. Macro-AUROC under random lead loss (test fold, 3 draws per k)",
           "number of missing leads (k)", "macro-AUROC")
    ax.set_xticks(sorted(s.n_missing.unique()))
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig1_auroc_vs_missing.png", dpi=DPI)
    plt.close(fig)


def fig2(res: Path, out: Path) -> None:
    """Attention heatmaps: clean vs V1 disconnected vs II disconnected."""
    path = res / "attention.npz"
    if not path.exists():
        print("fig2 skipped: attention.npz missing")
        return
    d = np.load(path)
    attn, names = d["attn"], [str(n) for n in d["names"]]
    vmax = float(attn.max())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4), constrained_layout=True)
    for ax, a, n in zip(axes, attn, names):
        im = ax.imshow(a, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(12), LEADS, rotation=90, fontsize=8)
        ax.set_yticks(range(12), LEADS, fontsize=8)
        ax.set_xlabel("key (attended lead)", color=MUTED)
        ax.set_ylabel("query lead", color=MUTED)
        ax.set_title(n, fontsize=11, color=TEXT)
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    PAPER_MODE or fig.suptitle(f"Figure 2. {d['run']} attention (last layer, mean over heads and test records)",
                 fontsize=11, color=TEXT)
    fig.savefig(out / "fig2_attention.png", dpi=DPI)
    plt.close(fig)


def fig3(res: Path, out: Path) -> None:
    """Group-loss bars."""
    s = per_seed(pd.read_csv(res / "group_loss.csv"), ["level"], "macro_auroc")
    groups = [g for g in ("precordial", "limb", "v1_v3") if g in set(s.level)]
    keys = [k for k in STYLE if k in set(s.variant)]
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, key in enumerate(keys):
        st = STYLE[key]
        d = s[s.variant == key].set_index("level").reindex(groups)
        x = np.arange(len(groups)) - 0.4 + (i + 0.5) * w
        err = d["std"].where(d.n_seeds > 1)
        ax.bar(x, d["mean"], w, color=st["color"], hatch=st["hatch"], edgecolor="white", lw=1,
               label=st["label"], yerr=err, ecolor=TEXT, capsize=2, error_kw={"lw": 0.8})
    _style(ax, "Figure 3. Macro-AUROC with a lead group removed (error bars: std over 3 seeds)",
           "removed lead group", "macro-AUROC")
    ax.set_xticks(range(len(groups)), [{"precordial": "Precordial (V1-V6)", "limb": "Limb (I-aVF)",
                                        "v1_v3": "V1-V3"}[g] for g in groups])
    lo = np.nanmin(s["mean"] - s["std"].fillna(0))
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out / "fig3_group_loss.png", dpi=DPI)
    plt.close(fig)


def fig4(res: Path, out: Path) -> None:
    """Risk-coverage curves (clean, 6 random leads missing) and reliability diagram (clean)."""
    rc = pd.read_csv(res / "risk_coverage.csv")
    rc = rc.assign(variant=variant(rc))
    curves = [("lg_edl", "edl_U", "LG-EDL, uncertainty U", "-"),
              ("lg_edl", "sigmoid_style", "LG-EDL, sigmoid-style score", ":"),
              ("resnet_c1", "sigmoid_style", "ResNet (corruption-trained)", "-"),
              ("lg_soft", "sigmoid_style", "LG-soft", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cond, title in [(axes[0], "clean", "(a) risk-coverage, clean"),
                            (axes[1], "random_6", "(b) risk-coverage, 6 random leads missing")]:
        d0 = rc[rc.condition == cond]
        for key, score, lab, ls in curves:
            d = d0[(d0.variant == key) & (d0.score == score)]
            if d.empty:
                continue
            g = d.groupby(["seed", "coverage"], as_index=False).risk.mean().groupby("coverage").risk
            mean, std, n = g.mean(), g.std(ddof=1), g.size().max()
            st = STYLE[key]
            ax.plot(mean.index, mean.values, color=st["color"], ls=ls, lw=1.8, label=lab + (f" ({n} seeds)" if n > 1 else ""))
            if n > 1:
                ax.fill_between(mean.index, mean - std, mean + std, color=st["color"], alpha=0.12, lw=0)
        _style(ax, title, "coverage (fraction of records kept)", "risk (mean per-record label error)")
        ax.legend(fontsize=8, frameon=False)

    cal = pd.read_csv(res / "calibration.csv")
    cal = cal[cal.q_mode != "oracle"].assign(variant=lambda d: variant(d))
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for key in ("lg_edl", "resnet_c1"):
        d = cal[cal.variant == key]
        if d.empty:
            continue
        d = d.assign(wp=d.mean_pred * d["count"], wf=d.frac_positive * d["count"])
        b = d.groupby("bin")[["wp", "wf", "count"]].sum()
        b = b[b["count"] > 0]
        ece = d.groupby("seed").ece.first()
        lab = f"{STYLE[key]['label']}, ECE {ece.mean():.3f}" + (f" ± {ece.std(ddof=1):.3f}" if len(ece) > 1 else "")
        ax.plot(b.wp / b["count"], b.wf / b["count"], color=STYLE[key]["color"], marker=STYLE[key]["marker"],
                ms=6, lw=1.8, label=lab)
    _style(ax, "(c) reliability, clean (15 bins, all classes)", "mean predicted probability", "fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    PAPER_MODE or fig.suptitle("Figure 4. Selective prediction and calibration on the test fold", fontsize=11, color=TEXT, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig4_risk_coverage_calibration.png", dpi=DPI)
    plt.close(fig)


def _fmt(mean: float, std: float, n: int) -> str:
    return f"{mean:.3f} ± {std:.3f}" if n > 1 else f"{mean:.3f}"


def table1(res: Path, out: Path) -> pd.DataFrame:
    """Clean results, all variants."""
    m = pd.read_csv(res / "main_clean.csv")
    rows = []
    for key in STYLE:
        cols = {}
        for metric in ("macro_auroc", "macro_f1", "ece"):
            s = per_seed(m, [], metric)
            s = s[s.variant == key]
            if s.empty:
                break
            r = s.iloc[0]
            cols.update({f"{metric}_mean": r["mean"], f"{metric}_std": r["std"], "n_seeds": int(r.n_seeds),
                         metric: _fmt(r["mean"], r["std"], int(r.n_seeds))})
        if cols:
            rows.append({"variant": key, "label": STYLE[key]["label"], **cols,
                         "note": "single seed" if cols["n_seeds"] == 1 else f"mean ± std over {cols['n_seeds']} seeds"})
    t = pd.DataFrame(rows)
    t.to_csv(out / "table1.csv", index=False)
    return t


def table2(res: Path, out: Path) -> pd.DataFrame:
    """Ablation M5-M8: clean, random lead loss (k = 6 and 10, mean over draws), precordial loss, clean ECE."""
    rl, gl, mc = (pd.read_csv(res / f) for f in ("random_loss.csv", "group_loss.csv", "main_clean.csv"))
    pieces = {
        "clean_auroc": per_seed(mc, [], "macro_auroc"),
        "random6_auroc": per_seed(rl[rl.n_missing == 6], [], "macro_auroc"),
        "random10_auroc": per_seed(rl[rl.n_missing == 10], [], "macro_auroc"),
        "precordial_auroc": per_seed(gl[gl.level == "precordial"], [], "macro_auroc"),
        "clean_ece": per_seed(mc, [], "ece"),
    }
    rows = []
    for key in ABLATION:
        row = {"variant": key, "label": STYLE[key]["label"]}
        for name, s in pieces.items():
            s = s[s.variant == key]
            if not s.empty:
                r = s.iloc[0]
                row.update({name: _fmt(r["mean"], r["std"], int(r.n_seeds)), f"{name}_mean": r["mean"],
                            "n_seeds": int(r.n_seeds)})
        row["note"] = "single seed" if row.get("n_seeds", 1) == 1 else f"mean ± std over {row['n_seeds']} seeds"
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "table2.csv", index=False)
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(REPO_DIR / "results"))
    p.add_argument("--fig-dir", default=str(REPO_DIR / "figures"))
    a = p.parse_args()
    res, fig_dir = Path(a.results_dir), Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f(res, fig_dir)
    t1, t2 = table1(res, res), table2(res, res)
    print(f"figures in {fig_dir}: " + ", ".join(sorted(p.name for p in fig_dir.glob("fig*.png"))))
    print(f"tables in {res}: table1.csv ({len(t1)} rows), table2.csv ({len(t2)} rows)")


if __name__ == "__main__":
    main()
