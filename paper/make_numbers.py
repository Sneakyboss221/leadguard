"""Turn results/*.csv into LaTeX macros and tables for the paper, so no number is typed by hand.

python paper/make_numbers.py [--results-dir results] [--fig-dir figures]

Writes paper/generated/numbers.tex (one \\newcommand per number), paper/generated/tab_*.tex (tables) and copies
the figures into paper/figures/. Aggregation matches figures.py: mean over the draws of a condition within each
run, then mean (and sample std where there are 3 seeds) over seeds.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent

# Variant keys (as in figures.py) -> macro stem (LaTeX macro names cannot contain digits) and table label.
VARIANTS = {
    "lg_edl": ("LgEdl", "LG-EDL"),
    "lg_soft": ("LgSoft", "LG-soft"),
    "lg_hard": ("LgHard", "LG-hard"),
    "resnet_attn": ("ResAttn", "ResNet + lead attention"),
    "resnet_c1": ("ResDrop", "ResNet, corruption-trained"),
    "resnet_c0": ("ResClean", "ResNet, clean-trained"),
    "transformer": ("Trans", "Patch Transformer"),
    "mlp": ("Mlp", "MLP"),
    "xgb": ("Xgb", "XGBoost"),
}
K_WORD = {0: "Zero", 2: "Two", 4: "Four", 6: "Six", 8: "Eight", 10: "Ten"}
COND_WORD = {"precordial": "Prec", "limb": "Limb", "v1_v3": "VOneThree"}
ART_WORD = {"emg": "Emg", "wander": "Wander", "clip": "Clip"}
N_WORD = {2: "Two", 4: "Four", 6: "Six"}
COV_WORD = {0.5: "Half", 0.8: "Eighty", 1.0: "Full"}

macros: list[tuple[str, str, str]] = []  # (name, value, provenance)


def m(name: str, value: str, src: str) -> None:
    assert name.isalpha(), name
    macros.append((name, value, src))


def f3(x: float) -> str:
    return f"{x:.3f}"


def f4(x: float) -> str:
    """Signed 4-decimal value; values that round to zero print as 0.0000 (no misleading sign)."""
    if round(x, 4) == 0:
        return "0.0000"
    return f"{x:+.4f}".replace("-", "$-$") if x < 0 else f"{x:+.4f}"


def variant(df: pd.DataFrame) -> pd.Series:
    return np.where(df.model == "resnet", "resnet_" + df.run.str.rsplit("_", n=1).str[-1], df.model)


def per_seed(df: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    df = df.assign(variant=variant(df))
    run = df.groupby(["variant", "seed"] + by, as_index=False)[value].mean()
    g = run.groupby(["variant"] + by)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n": g.size()}).reset_index()


def pm(mean: float, std: float, n: int) -> str:
    return f"{mean:.3f}" if n < 2 else f"{mean:.3f}\\,$\\pm$\\,{std:.3f}"


# ------------------------------------------------------------------ paper figures (IEEE two-column sizes)
TEXT_W = 7.16  # in, IEEE two-column text width; every figure is drawn at its printed size, so 8 pt prints as 8 pt
GRID = "#e6e4df"
LABEL = {"lg_edl": "LG-EDL", "lg_soft": "LG-soft", "lg_hard": "LG-hard", "resnet_attn": "Un-gated lead attention",
         "resnet_c1": "ResNet, corruption-trained", "resnet_c0": "ResNet, clean-trained",
         "transformer": "Patch Transformer", "mlp": "MLP", "xgb": "XGBoost"}


def paper_style():
    """Matplotlib with the paper's typography (Times, 7-8 pt) and vector output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42})
    return plt


def _axes_style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_random(R: Path, out: Path, style: dict) -> None:
    """Macro-AUROC versus number of randomly removed leads: (a) all nine variants, (b) the five corruption-trained
    ResNet / lead-attention models, which overlap in (a). Mean over draws, then over seeds; bands: std over seeds."""
    plt = paper_style()
    s = per_seed(pd.read_csv(R / "random_loss.csv"), ["n_missing"], "macro_auroc")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.35), layout="constrained")
    close = ["lg_edl", "lg_soft", "lg_hard", "resnet_attn", "resnet_c1"]
    for ax, keys, title in [(axes[0], list(LABEL), "(a) All models"),
                            (axes[1], close, "(b) Corruption-trained ResNet and lead-attention models")]:
        for key in keys:
            d = s[s.variant == key].sort_values("n_missing")
            st = style[key]
            ax.plot(d.n_missing, d["mean"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5,
                    label=LABEL[key] + (" (3 seeds)" if d.n.max() > 1 else ""), zorder=3 if key == "lg_edl" else 2)
            if d.n.max() > 1:
                ax.fill_between(d.n_missing, d["mean"] - d["std"], d["mean"] + d["std"], color=st["color"],
                                alpha=0.18, lw=0)
        _axes_style(ax)
        ax.set_xticks(sorted(s.n_missing.unique()))
        ax.set_xlabel("leads removed at random ($k$)")
        ax.set_ylabel("macro-AUROC")
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(out / "fig_random.pdf")
    plt.close(fig)


def fig_riskcov(R: Path, out: Path, style: dict) -> None:
    """(a, b) Risk-coverage on the clean test fold and with six random leads removed; (c) reliability diagram on
    the clean test fold. Sources: risk_coverage.csv, calibration.csv."""
    plt = paper_style()
    rc = pd.read_csv(R / "risk_coverage.csv")
    rc = rc.assign(variant=variant(rc))
    curves = [("lg_edl", "edl_U", "LG-EDL ranked by $U$", "-"),
              ("lg_edl", "sigmoid_style", "LG-EDL ranked by $s_p$", ":"),
              ("resnet_c1", "sigmoid_style", "ResNet, corruption-trained, by $s_p$", "-"),
              ("lg_soft", "sigmoid_style", "LG-soft by $s_p$", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), layout="constrained")
    for ax, cond, title in [(axes[0], "clean", "(a) Clean"), (axes[1], "random_6", "(b) Six random leads removed")]:
        d0 = rc[rc.condition == cond]
        for key, score, lab, ls in curves:
            d = d0[(d0.variant == key) & (d0.score == score)]
            g = d.groupby(["seed", "coverage"], as_index=False).risk.mean().groupby("coverage").risk
            mean, std, n = g.mean(), g.std(ddof=1), g.size().max()
            ax.plot(mean.index, mean.values, color=style[key]["color"], ls=ls, label=lab)
            if n > 1:
                ax.fill_between(mean.index, mean - std, mean + std, color=style[key]["color"], alpha=0.15, lw=0)
        _axes_style(ax)
        ax.set_xlabel("coverage (fraction of records kept)")
        ax.set_ylabel("risk (mean label error)")
        ax.set_title(title, loc="left")
    cal = pd.read_csv(R / "calibration.csv")
    cal = cal[cal.q_mode != "oracle"].assign(variant=lambda d: variant(d))
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color="#7a7b80", lw=0.8, ls="--")
    for key in ("lg_edl", "resnet_c1"):
        d = cal[cal.variant == key]
        d = d.assign(wp=d.mean_pred * d["count"], wf=d.frac_positive * d["count"])
        b = d.groupby("bin")[["wp", "wf", "count"]].sum()
        b = b[b["count"] > 0]
        ax.plot(b.wp / b["count"], b.wf / b["count"], color=style[key]["color"], marker=style[key]["marker"], ms=3,
                label=LABEL[key])
    _axes_style(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction positive")
    ax.set_title("(c) Reliability, clean", loc="left")
    ax.legend(frameon=False, loc="upper left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(out / "fig_riskcov.pdf")
    plt.close(fig)


def fig_corruptions(out: Path, sample: Path) -> None:
    """The simulated failure types on one PTB-XL test record (a released demo sample, already standardised), drawn
    with corrupt.py exactly as in plot_corruptions.py, at the printed size."""
    import sys
    sys.path.insert(0, str(REPO))
    import torch
    import plot_corruptions
    from utils import FS, LEADS

    plt = paper_style()
    x = np.load(sample)["x"].astype(np.float32)
    panels = plot_corruptions._panels(torch.from_numpy(x)[None], seed=0)
    titles = ["Clean", "Random removal ($k=4$)", "Precordial leads removed", "Flatline (II, V1, V5)",
              "EMG 10 dB (II, V1, V5)", "Baseline wander (II, V1, V5)", "Clipping (II, V1, V5)", "Training mixture"]
    t = np.arange(x.shape[-1]) / FS
    off = 6.0
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_W, 4.3), sharex=True, sharey=True, layout="constrained")
    colors = (plot_corruptions.C_CLEAN, plot_corruptions.C_MILD, plot_corruptions.C_BAD)
    for ax, title, (_, xc, sev) in zip(axes.ravel(), titles, panels):
        xc, sev = xc[0].numpy(), sev[0].numpy()
        for j in range(12):
            c = colors[0] if sev[j] == 0 else (colors[1] if sev[j] < 0.5 else colors[2])
            ax.plot(t, xc[j] - j * off, color=c, lw=0.45)
            if sev[j] > 0:
                ax.text(10.15, -j * off, f"{sev[j]:.2f}", color=c, va="center", fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_yticks([-j * off for j in range(12)], LEADS)
        ax.set_xlim(0, 11.6)
        ax.set_ylim(-11.8 * off, off)
        ax.set_xticks([0, 5, 10])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    handles = [plt.Line2D([], [], color=c, lw=1.5) for c in colors]
    fig.legend(handles, ["untouched ($s_j=0$)", "corrupted, reliable ($s_j<0.5$)", "unreliable ($s_j\\geq0.5$)"],
               loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "fig_corruptions.pdf")
    plt.close(fig)


def fig_certainty(R: Path, out: Path) -> None:
    """Paper figure: LG-EDL accuracy, evidential uncertainty and abstention as leads are removed at random.

    (a) macro-AUROC, (b) mean U with the abstention threshold, (c) abstention rate by rule. Mean over the 3 draws
    within each seed, then mean (line) and std (band / error bar) over the 3 seeds. Sources: random_loss.csv,
    abstain_rate.csv.
    """
    plt = paper_style()
    blue, red, ink, muted, grid = "#2a6fca", "#b3263a", "#16171a", "#7a7b80", "#e6e4df"

    rl = pd.read_csv(R / "random_loss.csv")
    rl = rl[rl.model == "lg_edl"].groupby(["seed", "n_missing"], as_index=False).macro_auroc.mean()
    auc = rl.groupby("n_missing").macro_auroc.agg(["mean", "std"])
    ab = pd.read_csv(R / "abstain_rate.csv")
    ab = ab[(ab.family == "random")].groupby(["seed", "n_missing"], as_index=False)[
        ["mean_U", "rate_U_above_threshold", "rate_qsum_below_3", "abstain_rate"]].mean()
    g = ab.groupby("n_missing").agg(["mean", "std"])
    tau = pd.read_csv(R / "abstain_rate.csv").groupby("run").u_threshold.first()
    k = auc.index.values

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.1), layout="constrained")
    for ax in axes:
        ax.grid(True, color=grid, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(k)
        ax.set_xlabel("leads removed at random ($k$)")

    ax = axes[0]
    ax.fill_between(k, auc["mean"] - auc["std"], auc["mean"] + auc["std"], color=blue, alpha=0.15, lw=0)
    ax.plot(k, auc["mean"], "-o", color=blue, ms=3.5, lw=1.4)
    ax.set_ylabel("macro-AUROC")
    ax.set_title("(a) Discrimination", loc="left")

    ax = axes[1]
    mu, sd = g[("mean_U", "mean")], g[("mean_U", "std")]
    ax.fill_between(k, mu - sd, mu + sd, color=blue, alpha=0.15, lw=0)
    ax.plot(k, mu, "-o", color=blue, ms=3.5, lw=1.4, label="mean $U$")
    ax.axhline(tau.mean(), color=ink, lw=1.0, ls="--", label=r"abstention threshold $\tau$")
    ax.set_ylim(0, max(0.6, tau.max() + 0.05))
    ax.set_ylabel("evidential uncertainty $U$")
    ax.set_title("(b) Evidential uncertainty", loc="left")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]
    w = 0.7
    ax.bar(k - w / 2, 100 * g[("rate_U_above_threshold", "mean")], width=w, color=blue, label=r"$U>\tau$")
    ax.bar(k + w / 2, 100 * g[("rate_qsum_below_3", "mean")], width=w, color=red, label=r"$\Sigma q<3$")
    ax.set_ylabel("records abstained (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(c) Abstention by rule", loc="left")
    ax.legend(frameon=False, loc="upper left")

    fig.savefig(out / "fig_certainty.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--fig-dir", default=str(REPO / "figures"))
    a = p.parse_args()
    R = Path(a.results_dir)
    out = PAPER / "generated"
    out.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- dataset / setup
    sqa = pd.read_csv(R / "sqa_agreement.csv")
    n_test = int(sqa.loc[sqa.corruption == "clean", "n_leads"].iloc[0]) // 12
    m("nTestRecords", f"{n_test:,}".replace(",", "{,}"), "sqa_agreement.csv: n_leads(clean)/12, test fold")
    m("sqaGate", f3(sqa.agreement_excl_band.min()), "sqa_agreement.csv: min agreement_excl_band")
    m("sqaGateType", sqa.loc[sqa.agreement_excl_band.idxmin(), "corruption"].replace("_", "\\_"), "sqa_agreement.csv")
    m("sqaMeanQClean", f3(sqa.loc[sqa.corruption == "clean", "mean_q_untouched"].iloc[0]), "sqa_agreement.csv")
    m("sqaEmgTestCorrupted", f3(sqa.loc[sqa.corruption == "emg_10db_test", "agreement_corrupted_leads"].iloc[0]),
      "sqa_agreement.csv: emg_10db_test agreement_corrupted_leads")
    m("sqaEmgTestBand", f3(sqa.loc[sqa.corruption == "emg_10db_test", "frac_in_band"].iloc[0]), "sqa_agreement.csv")
    timing = pd.read_csv(R / "timing.csv")
    m("gpuMinutes", f"{timing.seconds.sum() / 60:.0f}", "timing.csv: sum of seconds / 60")
    first = timing[~timing.resumed.astype(bool)]  # first invocation of each run (a resumed run adds a second row)
    sec = lambda model, c: first[(first.model == model) & (first.corrupt_train == c)].seconds.mean()
    m("secLgEdlRun", f"{sec('lg_edl', 1):.0f}", "timing.csv: mean seconds per lg_edl run (first invocation)")
    m("secResDropRun", f"{sec('resnet', 1):.0f}", "timing.csv: mean seconds per resnet c1 run")
    m("timeRatioLgRes", f"{sec('lg_edl', 1) / sec('resnet', 1):.1f}", "timing.csv: lg_edl / resnet c1 seconds per run")

    # ---------------------------------------------------------------- clean (Table I)
    t1 = pd.read_csv(R / "table1.csv").set_index("variant")
    for key, (stem, _) in VARIANTS.items():
        r = t1.loc[key]
        n = int(r.n_seeds)
        m(f"auc{stem}Clean", f3(r.macro_auroc_mean), "table1.csv macro_auroc_mean")
        m(f"auc{stem}CleanPM", pm(r.macro_auroc_mean, r.macro_auroc_std, n), "table1.csv")
        m(f"fone{stem}Clean", f3(r.macro_f1_mean), "table1.csv macro_f1_mean")
        m(f"ece{stem}Clean", f3(r.ece_mean), "table1.csv ece_mean")
        m(f"ece{stem}CleanPM", pm(r.ece_mean, r.ece_std, n), "table1.csv")
    rows = []
    for key, (_, label) in VARIANTS.items():
        r = t1.loc[key]
        n = int(r.n_seeds)
        rows.append(f"{label} & {n} & {pm(r.macro_auroc_mean, r.macro_auroc_std, n)} & "
                    f"{pm(r.macro_f1_mean, r.macro_f1_std, n)} & {pm(r.ece_mean, r.ece_std, n)} \\\\")
    (out / "tab_clean.tex").write_text(
        "% generated by make_numbers.py from results/table1.csv\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModel & Seeds & Macro-AUROC & Macro-F1 & ECE \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- random lead loss
    rl = per_seed(pd.read_csv(R / "random_loss.csv"), ["n_missing"], "macro_auroc")
    for key, (stem, _) in VARIANTS.items():
        for k, w in K_WORD.items():
            r = rl[(rl.variant == key) & (rl.n_missing == k)].iloc[0]
            m(f"auc{stem}K{w}", f3(r["mean"]), "random_loss.csv, mean over draws then seeds")
            m(f"auc{stem}K{w}PM", pm(r["mean"], r["std"], int(r.n)), "random_loss.csv")
    drop = lambda key, k: rl[(rl.variant == key) & (rl.n_missing == k)]["mean"].iloc[0]
    m("gainDropKTen", f3(drop("resnet_c1", 10) - drop("resnet_c0", 10)), "random_loss.csv: resnet_c1 - resnet_c0 at k=10")
    m("lossResCleanKTen", f3(drop("resnet_c0", 0) - drop("resnet_c0", 10)), "random_loss.csv")
    m("lossResDropKTen", f3(drop("resnet_c1", 0) - drop("resnet_c1", 10)), "random_loss.csv")
    m("lossLgEdlKTen", f3(drop("lg_edl", 0) - drop("lg_edl", 10)), "random_loss.csv")
    attn_k10 = [drop(k, 10) for k in ["lg_edl", "lg_soft", "lg_hard", "resnet_attn"]]
    m("gapAttnResKTen", f3(max(attn_k10) - drop("resnet_c1", 10)),
      "random_loss.csv: best attention variant minus resnet_c1 at k=10")
    m("seedStdLgEdlMaxLoss", f3(rl[(rl.variant == "lg_edl") & (rl.n_missing > 0)]["std"].max()),
      "random_loss.csv: largest LG-EDL seed std over k=2..10")
    # Decision-level metrics under random lead loss: macro-F1 at the validation-tuned thresholds, and ECE.
    rlf = pd.read_csv(R / "random_loss.csv")
    for metric, pre in [("macro_f1", "fone"), ("ece", "ece")]:
        sf = per_seed(rlf, ["n_missing"], metric)
        for key, (stem, _) in VARIANTS.items():
            for k, w in K_WORD.items():
                r = sf[(sf.variant == key) & (sf.n_missing == k)].iloc[0]
                m(f"{pre}{stem}K{w}", f3(r["mean"]), f"random_loss.csv {metric}, mean over draws then seeds")
    keys = list(VARIANTS)
    rows = []
    for key in keys:
        cells = []
        for k in K_WORD:
            r = rl[(rl.variant == key) & (rl.n_missing == k)].iloc[0]
            cells.append(f"{r['mean']:.3f}")
        rows.append(f"{VARIANTS[key][1]} & " + " & ".join(cells) + " \\\\")
    (out / "tab_random.tex").write_text(
        "% generated by make_numbers.py from results/random_loss.csv (mean over 3 draws, then over seeds)\n"
        "\\begin{tabular}{@{}l" + "c" * len(K_WORD) + "@{}}\n\\toprule\n"
        "& \\multicolumn{" + str(len(K_WORD)) + "}{c}{Leads removed at random ($k$)} \\\\\n"
        "\\cmidrule(l){2-" + str(len(K_WORD) + 1) + "}\nModel & " + " & ".join(str(k) for k in K_WORD) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- group loss + artifacts
    gl = per_seed(pd.read_csv(R / "group_loss.csv"), ["level"], "macro_auroc")
    for key, (stem, _) in VARIANTS.items():
        for lvl, w in COND_WORD.items():
            r = gl[(gl.variant == key) & (gl.level == lvl)].iloc[0]
            m(f"auc{stem}{w}", f3(r["mean"]), "group_loss.csv")
    art = pd.read_csv(R / "artifact.csv")
    ar = per_seed(art, ["condition"], "macro_auroc")
    for key, (stem, _) in VARIANTS.items():
        for kind, w in ART_WORD.items():
            for n, nw in N_WORD.items():
                r = ar[(ar.variant == key) & (ar.condition == f"{kind}_{n}")].iloc[0]
                m(f"auc{stem}{w}{nw}", f3(r["mean"]), "artifact.csv")
    rows = []
    for key in keys:
        g = [gl[(gl.variant == key) & (gl.level == lvl)]["mean"].iloc[0] for lvl in COND_WORD]
        s = [ar[(ar.variant == key) & (ar.condition == f"{kind}_6")]["mean"].iloc[0] for kind in ART_WORD]
        rows.append(f"{VARIANTS[key][1]} & " + " & ".join(f"{v:.3f}" for v in g + s) + " \\\\")
    (out / "tab_group_artifact.tex").write_text(
        "% generated by make_numbers.py from results/group_loss.csv and results/artifact.csv\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "& \\multicolumn{3}{c}{Lead group removed} & \\multicolumn{3}{c}{Artifact on 6 leads} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
        "Model & V1--V6 & Limb & V1--V3 & EMG & Wander & Clip \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- paired bootstrap (LG-EDL - ResNet c1)
    bs = pd.read_csv(R / "bootstrap_ci.csv")
    bs = bs[bs.condition != "random_0"]  # identical to clean (no lead removed)
    excl = bs[(bs.ci_low > 0) | (bs.ci_high < 0)]
    m("bootNRows", str(len(bs)), "bootstrap_ci.csv rows excluding random_0 (= clean)")
    m("bootNSig", str(len(excl)), "bootstrap_ci.csv: CI excludes 0")
    m("bootNSigNeg", str(int((excl["diff"] < 0).sum())), "bootstrap_ci.csv: CI entirely below 0")
    m("bootNSigPos", str(int((excl["diff"] > 0).sum())), "bootstrap_ci.csv: CI entirely above 0")
    m("bootNBoot", str(int(bs.n_boot.iloc[0])), "bootstrap_ci.csv n_boot")
    for _, r in bs.iterrows():
        cond, draw = r.condition, int(r.draw)
        kind, _, arg = cond.partition("_")
        if cond == "clean":
            name = "Clean"
        elif kind == "random":
            name = f"K{K_WORD[int(arg)]}D{['A', 'B', 'C'][draw]}"
        elif kind == "group":
            name = COND_WORD[arg]
        else:
            name = ART_WORD[kind] + N_WORD[int(arg)]
        m(f"boot{name}Diff", f4(r["diff"]), "bootstrap_ci.csv diff")
        m(f"boot{name}CI", f"[{f4(r.ci_low)}, {f4(r.ci_high)}]", "bootstrap_ci.csv ci_low/ci_high")
    labels = {"clean": "Clean", **{f"group_{g}": f"Group: {l}" for g, l in
              [("precordial", "V1--V6"), ("limb", "limb"), ("v1_v3", "V1--V3")]}}
    rows = []
    for _, r in bs.iterrows():
        cond = r.condition
        kind, _, arg = cond.partition("_")
        if kind == "random":
            lab = f"Random, $k={arg}$ (draw {int(r.draw) + 1})"
        elif cond in labels:
            lab = labels[cond]
        else:
            lab = {"emg": "EMG 10\\,dB", "wander": "Baseline wander", "clip": "Clipping"}[kind] + f", {arg} leads"
        mark = "$^{*}$" if (r.ci_low > 0 or r.ci_high < 0) else ""
        rows.append(f"{lab} & {f4(r['diff'])}{mark} & [{f4(r.ci_low)}, {f4(r.ci_high)}] \\\\")
    (out / "tab_bootstrap.tex").write_text(
        "% generated by make_numbers.py from results/bootstrap_ci.csv (random_0 omitted: identical to clean)\n"
        "\\begin{tabular}{@{}lcc@{}}\n\\toprule\nCondition & $\\Delta$ macro-AUROC & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- ablation (Table II)
    t2 = pd.read_csv(R / "table2.csv").set_index("variant")
    rows = []
    for key in ["resnet_attn", "lg_hard", "lg_soft", "lg_edl"]:
        r = t2.loc[key]
        rows.append(f"{VARIANTS[key][1]} & {int(r.n_seeds)} & " + " & ".join(
            str(r[c]).replace("±", "$\\pm$") for c in
            ["clean_auroc", "random6_auroc", "random10_auroc", "precordial_auroc", "clean_ece"]) + " \\\\")
    (out / "tab_ablation.tex").write_text(
        "% generated by make_numbers.py from results/table2.csv\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Variant & Seeds & Clean & $k{=}6$ & $k{=}10$ & V1--V6 & ECE \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    abl = [t2.loc[k] for k in ["resnet_attn", "lg_hard", "lg_soft", "lg_edl"]]
    for col, w in [("clean_auroc_mean", "Clean"), ("random10_auroc_mean", "KTen")]:
        vals = [float(r[col]) for r in abl]
        m(f"ablRange{w}", f3(max(vals) - min(vals)), f"table2.csv {col}: max - min over M5-M8")

    # ---------------------------------------------------------------- oracle vs estimated q
    o = pd.read_csv(R / "oracle_vs_estimated.csv")
    piv = o.pivot_table(index=["run", "condition", "draw"], columns="q_mode", values="macro_auroc")
    m("oracleMaxAbsDiff", f3((piv["oracle"] - piv["estimated"]).abs().max()), "oracle_vs_estimated.csv: max |oracle - estimated|")

    # ---------------------------------------------------------------- selective prediction
    rc = pd.read_csv(R / "risk_coverage.csv")
    rc = rc.assign(variant=variant(rc))
    sel_rows = []
    combos = [("lg_edl", "edl_U", "LG-EDL, $U$"), ("lg_edl", "sigmoid_style", "LG-EDL, confidence"),
              ("lg_soft", "sigmoid_style", "LG-soft, confidence"), ("resnet_attn", "sigmoid_style", "ResNet + attn., conf."),
              ("resnet_c1", "sigmoid_style", "ResNet (corr.), conf.")]
    for cond, cw in [("clean", "Clean"), ("random_6", "KSix")]:
        d0 = rc[rc.condition == cond]
        for key, score, lab in combos:
            d = d0[(d0.variant == key) & (d0.score == score)]
            # mean over draws within a seed, then over seeds
            per = d.groupby(["seed", "coverage"], as_index=False).risk.mean().groupby("coverage").risk.mean()
            sw = "U" if score == "edl_U" else "Conf"
            for cov, covw in COV_WORD.items():
                m(f"risk{VARIANTS[key][0]}{sw}{cw}{covw}", f3(per.loc[cov]), "risk_coverage.csv")
            m(f"aurc{VARIANTS[key][0]}{sw}{cw}", f3(per.mean()), "risk_coverage.csv: mean risk over the 20 coverage levels")
            if cond in ("clean", "random_6"):
                sel_rows.append((cond, lab, per.loc[0.5], per.loc[0.8], per.loc[1.0], per.mean()))
    rows, last = [], None
    for cond, lab, r50, r80, r100, au in sel_rows:
        if cond != last:
            if last is not None:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{'Clean' if cond == 'clean' else '6 random leads removed'}}}}} \\\\")
            last = cond
        rows.append(f"{lab} & {r50:.3f} & {r80:.3f} & {r100:.3f} & {au:.3f} \\\\")
    (out / "tab_selective.tex").write_text(
        "% generated by make_numbers.py from results/risk_coverage.csv\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Model, score & 50\\% & 80\\% & 100\\% & Mean \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- abstention vs lead loss (LG-EDL)
    ab = pd.read_csv(R / "abstain_rate.csv")
    m("uThresholdMean", f3(ab.groupby("run").u_threshold.first().mean()), "abstain_rate.csv u_threshold, mean over seeds")
    abr = ab[ab.family.isin(["clean", "random"])]
    abr = abr[abr.condition != "clean"]
    g = abr.groupby(["seed", "n_missing"], as_index=False)[
        ["abstain_rate", "rate_U_above_threshold", "rate_qsum_below_3", "mean_U", "mean_qsum"]].mean().groupby("n_missing").mean()
    for k, w in K_WORD.items():
        m(f"abstainK{w}", f"{100 * g.loc[k, 'abstain_rate']:.1f}", "abstain_rate.csv, %")
        m(f"abstainUK{w}", f"{100 * g.loc[k, 'rate_U_above_threshold']:.1f}", "abstain_rate.csv, %")
        m(f"abstainQK{w}", f"{100 * g.loc[k, 'rate_qsum_below_3']:.1f}", "abstain_rate.csv, %")
        m(f"meanUK{w}", f3(g.loc[k, "mean_U"]), "abstain_rate.csv")
        m(f"meanQsumK{w}", f"{g.loc[k, 'mean_qsum']:.2f}", "abstain_rate.csv")
    rows = [f"{k} & {100 * g.loc[k, 'abstain_rate']:.1f} & {100 * g.loc[k, 'rate_U_above_threshold']:.1f} & "
            f"{100 * g.loc[k, 'rate_qsum_below_3']:.1f} & {g.loc[k, 'mean_U']:.3f} & {g.loc[k, 'mean_qsum']:.2f} \\\\"
            for k in K_WORD]
    (out / "tab_abstain.tex").write_text(
        "% generated by make_numbers.py from results/abstain_rate.csv (LG-EDL, mean over draws then seeds)\n"
        "\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n"
        "$k$ & Abstain (\\%) & $U{>}\\tau$ (\\%) & $\\Sigma q{<}3$ (\\%) & Mean $U$ & Mean $\\Sigma q$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---------------------------------------------------------------- per-class AUROC (clean)
    pc = pd.read_csv(R / "per_class_auroc.csv")
    pc = pc[pc.q_mode != "oracle"]
    for cond, suffix in [("clean", ""), ("group_precordial", "Prec")]:
        pcs = per_seed(pc[pc.condition == cond], ["class"], "auroc")
        for key in ["lg_edl", "resnet_c1", "resnet_c0", "resnet_attn"]:
            for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
                r = pcs[(pcs.variant == key) & (pcs["class"] == cls)].iloc[0]
                m(f"pc{VARIANTS[key][0]}{cls.capitalize()}{suffix}", f3(r["mean"]), f"per_class_auroc.csv, {cond}")

    # ---------------------------------------------------------------- parameter counts (from the model code)
    import sys
    sys.path.insert(0, str(REPO))
    from models import build_model, count_params
    from quality import LeadQualityNet

    def mio(n: int) -> str:
        return f"{n / 1e6:.2f}\\,M" if n >= 1e6 else f"{n / 1e3:.0f}\\,k"

    for key, stem in [("mlp", "Mlp"), ("resnet", "Res"), ("transformer", "Trans"), ("resnet_attn", "ResAttn"),
                      ("lg_hard", "LgHard"), ("lg_soft", "LgSoft"), ("lg_edl", "LgEdl")]:
        m(f"params{stem}", mio(count_params(build_model(key))), "models.py count_params")
    lg = build_model("lg_edl")
    m("paramsLgEncoder", mio(count_params(lg.encoder)), "models.py: LeadEncoder")
    m("paramsLgAttention", mio(sum(count_params(l) for l in lg.layers)), "models.py: 2 MALA layers")
    m("paramsSqa", mio(count_params(LeadQualityNet())), "quality.py LeadQualityNet")

    # ---------------------------------------------------------------- write macros + figures
    lines = ["% generated by paper/make_numbers.py from results/*.csv -- do not edit by hand",
             "% each line: macro, value, source"]
    seen = set()
    for name, val, src in macros:
        assert name not in seen, f"duplicate macro {name}"
        seen.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}} % {src}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figures: drawn here at the printed IEEE size (vector PDF, Times, 7-8 pt), from the same CSVs and with the same
    # colours and markers as figures.py. The corruption figure uses a released demo sample (PTB-XL test record).
    import sys
    sys.path.insert(0, str(REPO))
    import figures
    fig_out = PAPER / "figures"
    fig_out.mkdir(exist_ok=True)
    for old in ("fig1_auroc_vs_missing.png", "fig2_attention.png", "fig3_group_loss.png",
                "fig4_risk_coverage_calibration.png", "corruption_examples.png"):
        (fig_out / old).unlink(missing_ok=True)
    fig_random(R, fig_out, figures.STYLE)
    fig_riskcov(R, fig_out, figures.STYLE)
    fig_corruptions(fig_out, REPO / "demo_samples" / "sample_04.npz")
    fig_certainty(R, fig_out)
    print(f"{len(macros)} macros -> {out / 'numbers.tex'}; tables: "
          + ", ".join(sorted(p.name for p in out.glob("tab_*.tex"))) + f"; figures redrawn in {fig_out}")


if __name__ == "__main__":
    main()
