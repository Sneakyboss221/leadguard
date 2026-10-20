"""Clinical vignette figure: one demo patient with all 12 leads and with V1-V6 unplugged.

python paper/make_vignette.py [--sample 4] [--ckpt-dir checkpoints] [--samples-dir demo_samples]

Runs the real models through app.Demo (the same code path as the demo), so the probabilities in the figure are
exactly what the demo shows. Writes paper/figures/fig_vignette.pdf and paper/generated/vignette.tex (macros).
The demo samples were selected so that both models are fully correct with all 12 leads: this is an illustration,
not evidence; the aggregate results are in the tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(REPO))

from app import Demo  # noqa: E402
from utils import CLASSES, FS, LEADS  # noqa: E402

CHEST = list(range(6, 12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=4, help="index into demo_samples (4 = MI 1)")
    p.add_argument("--ckpt-dir", default=str(REPO / "checkpoints"))
    p.add_argument("--samples-dir", default=str(REPO / "demo_samples"))
    a = p.parse_args()

    demo = Demo(Path(a.ckpt_dir), Path(a.samples_dir), REPO / "results")
    on_all = [True] * 12
    on_cut = [j not in CHEST for j in range(12)]
    req = {"sample": a.sample, "noisy": [], "quality": "sqa", "baseline": "resnet_c1", "seed": 0}
    full = demo.predict(req | {"on": on_all})
    cut = demo.predict(req | {"on": on_cut})
    truth = full["truth"]
    name = demo.meta()["samples"][a.sample]
    same_class = [s for s in demo.meta()["samples"] if s["target"] == name["target"]]
    label = f"{name['target']} {[s['index'] for s in same_class].index(a.sample) + 1}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    blue, orange, ink, off = "#2a6fca", "#d9622b", "#16171a", "#b3b2ac"
    grid_minor, grid_major = "#f6dedb", "#ebb9b4"

    fig = plt.figure(figsize=(7.16, 2.9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1.0], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    raw = np.array(full["raw"])
    t = np.arange(raw.shape[1]) / FS
    offset = 3.0
    for x in np.arange(0, 10.01, 0.2):
        ax.axvline(x, color=grid_major if abs(x - round(x)) < 1e-9 else grid_minor, lw=0.4, zorder=0)
    for y in np.arange(-12 * offset, 1.5, 0.5):
        ax.axhline(y, color=grid_minor, lw=0.4, zorder=0)
    for j in range(12):
        y0 = -j * offset
        if j in CHEST:
            ax.plot(t, raw[j] + y0, color=off, lw=0.6, zorder=1)
            ax.plot([0, 10], [y0, y0], color=off, lw=0.6, ls=(0, (3, 3)), zorder=1)
        else:
            ax.plot(t, raw[j] + y0, color=ink, lw=0.6, zorder=2)
        ax.text(-0.15, y0, LEADS[j], ha="right", va="center", fontsize=7.5,
                color=off if j in CHEST else ink, fontweight="bold")
    ax.plot([10.25, 10.25], [0.2, 1.2], color=ink, lw=1.2)
    ax.text(10.35, 0.7, "1 mV", fontsize=7, va="center")
    ax.set_xlim(-0.9, 10.9)
    ax.set_ylim(-11 * offset - 2, 2.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("time (s)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"(a) Patient {label} (true label: {' + '.join(truth)}); grey: V1-V6 unplugged in (c)",
                 loc="left")

    def bars(axp, res, title):
        y = np.arange(len(CLASSES))
        h = 0.36
        axp.barh(y - h / 2, res["base"]["prob"], height=h, color=orange, label="ResNet (corruption-trained)")
        axp.barh(y + h / 2, res["lg"]["prob"], height=h, color=blue, label="LG-EDL")
        for i in range(len(CLASSES)):
            axp.plot([res["base"]["thr"][i]] * 2, [i - h, i], color=ink, lw=1.1)
            axp.plot([res["lg"]["thr"][i]] * 2, [i, i + h], color=ink, lw=1.1)
        axp.set_yticks(y, [c + (" (true)" if c in truth else "") for c in CLASSES])
        axp.invert_yaxis()
        axp.set_xlim(0, 1)
        axp.set_xticks([0, 0.5, 1])
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
        pb = [c for c, v in zip(CLASSES, res["base"]["pred"]) if v]
        pl = [c for c, v in zip(CLASSES, res["lg"]["pred"]) if v]
        axp.set_title(f"{title}\nResNet: {' + '.join(pb) or 'none'};  LG-EDL: {' + '.join(pl) or 'none'}",
                      loc="left", fontsize=8)

    axb = fig.add_subplot(gs[0, 1])
    bars(axb, full, "(b) All 12 leads")
    axc = fig.add_subplot(gs[1, 1])
    bars(axc, cut, "(c) V1-V6 unplugged")
    axc.set_xlabel("predicted probability (tick: decision threshold)")
    h, l = axb.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="outside lower right", ncol=2, fontsize=7)
    fig.savefig(PAPER / "figures" / "fig_vignette.pdf")
    plt.close(fig)

    def f3(x: float) -> str:
        return f"{x:.2f}"

    k_mi, k_st = CLASSES.index("MI"), CLASSES.index("STTC")
    lines = [f"% generated by paper/make_vignette.py (demo sample {a.sample}, models via app.Demo)",
             f"\\newcommand{{\\vigPatient}}{{{label}}}",
             f"\\newcommand{{\\vigBaseMiFull}}{{{f3(full['base']['prob'][k_mi])}}}",
             f"\\newcommand{{\\vigBaseMiCut}}{{{f3(cut['base']['prob'][k_mi])}}}",
             f"\\newcommand{{\\vigBaseStFull}}{{{f3(full['base']['prob'][k_st])}}}",
             f"\\newcommand{{\\vigBaseStCut}}{{{f3(cut['base']['prob'][k_st])}}}",
             f"\\newcommand{{\\vigBaseStThr}}{{{f3(cut['base']['thr'][k_st])}}}",
             f"\\newcommand{{\\vigLgMiFull}}{{{f3(full['lg']['prob'][k_mi])}}}",
             f"\\newcommand{{\\vigLgMiCut}}{{{f3(cut['lg']['prob'][k_mi])}}}",
             f"\\newcommand{{\\vigLgStCut}}{{{f3(cut['lg']['prob'][k_st])}}}",
             f"\\newcommand{{\\vigLgUFull}}{{{full['U']:.3f}}}",
             f"\\newcommand{{\\vigLgUCut}}{{{cut['U']:.3f}}}"]
    (PAPER / "generated" / "vignette.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("full: base", [round(v, 3) for v in full["base"]["prob"]], "lg", [round(v, 3) for v in full["lg"]["prob"]],
          "U", round(full["U"], 3))
    print("cut:  base", [round(v, 3) for v in cut["base"]["prob"]], "lg", [round(v, 3) for v in cut["lg"]["prob"]],
          "U", round(cut["U"], 3), "q", cut["q"])
    print("wrote figures/fig_vignette.pdf and generated/vignette.tex")


if __name__ == "__main__":
    main()
