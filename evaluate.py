"""Evaluation (SPEC section 6). Writes results/*.csv and results/attention.npz.

Every trained run from the run list is evaluated on the clean test fold and on every condition in
corrupt.TEST_CONDITIONS. Each condition corrupts the whole test fold once, so all models see identical data.
lg_hard / lg_soft / lg_edl are evaluated with SQA-estimated q and with oracle q = 1 - severity.

Fitted on clean validation predictions (fold 9): one temperature per sigmoid model (all models except
lg_edl), per-class F1 thresholds, and the lg_edl abstain threshold on U.

python evaluate.py --data ptbxl --data-root PATH --ckpt-dir DIR --out-dir DIR [--cache-dir DIR] [--strict] [--debug]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

import corrupt
from baseline_xgb import load_xgb, predict_xgb
from data import load_dataset
from models import abstain, build_model
from quality import estimate_q
from run_all import BATCH, EPOCHS, RUNS
from train import USES_Q, check_status, ckpt_paths, load_sqa_checked, run_config, run_name
from utils import CLASSES, REPO_DIR, get_device, macro_auroc

N_BINS = 15
ABSTAIN_QUANTILE = 0.95  # U threshold = 95th percentile of U on clean validation (5% abstention there)
COVERAGES = np.round(np.linspace(0.05, 1.0, 20), 2)
EPS = 1e-6


# ------------------------------------------------------------------ loading

def load_runs(ckpt_dir: Path, data: str, debug: bool, device, strict: bool = False) -> list[dict]:
    """Finished runs from the run list, with their model loaded.

    Missing or unfinished runs are reported and skipped; with strict=True they stop the evaluation instead.
    """
    runs, missing = [], []
    for model, c, seed in RUNS:
        if model == "xgb":
            path = ckpt_dir / f"xgb_s{seed}.pkl"
            if not path.exists():
                print(f"WARNING: {path.name} missing, not evaluated")
                missing.append(path.stem)
                continue
            obj, meta = load_xgb(path)
            if (meta.get("data"), meta.get("debug")) != (data, debug):
                raise RuntimeError(f"{path} is from data={meta.get('data')}, debug={meta.get('debug')}")
            runs.append({"run": f"xgb_s{seed}", "model": model, "seed": seed, "corrupt_train": "n/a", "obj": obj})
            continue
        status, _ = check_status(ckpt_dir, run_config(model, seed, c, data, debug, EPOCHS, BATCH))
        if status != "finished":
            print(f"WARNING: {run_name(model, seed, c)} is {status}, not evaluated")
            missing.append(run_name(model, seed, c))
            continue
        best, _ = ckpt_paths(ckpt_dir, model, seed, c)
        net = build_model(model).to(device)
        net.load_state_dict(torch.load(best, map_location=device, weights_only=False)["state_dict"])
        runs.append({"run": run_name(model, seed, c), "model": model, "seed": seed, "corrupt_train": c,
                     "obj": net.eval()})
    if strict and missing:
        raise SystemExit(f"--strict: {len(missing)} of {len(RUNS)} runs are missing or unfinished: {missing}. "
                         "Finish them with run_all.py (same arguments), then evaluate again.")
    return runs


@torch.no_grad()
def predict(r: dict, x: torch.Tensor, q_mode: str, sev: torch.Tensor | None, sqa, device,
            batch: int = 256, attn: bool = False) -> dict:
    """x: (N, 12, T) on CPU. q_mode: "estimated" (SQA), "oracle" (1 - severity) or "n/a" (baselines).
    Returns numpy arrays: prob (N, 5); for SQA models q (N, 12); for lg_edl U (N,); attention if asked."""
    if r["model"] == "xgb":
        return {"prob": predict_xgb(r["obj"], x.numpy())["prob"]}
    out = {"prob": [], "q": [], "U": [], "attn": []}
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        if r["model"] not in USES_Q:
            q = torch.ones(len(xb), 12, device=device)
        elif q_mode == "oracle":
            q = (1 - sev[i:i + batch]).to(device)
        else:
            q = estimate_q(sqa, xb)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            o = r["obj"](xb, q)
        out["prob"].append(o["prob"].float().cpu())
        out["q"].append(q.float().cpu())
        if "U" in o:
            out["U"].append(o["U"].float().cpu())
        if attn:
            out["attn"].append(o["attn"][-1].float().mean(1).sum(0).cpu())  # last layer, mean over heads
    res = {"prob": torch.cat(out["prob"]).numpy()}
    if r["model"] in USES_Q:
        res["q"] = torch.cat(out["q"]).numpy()
    if out["U"]:
        res["U"] = torch.cat(out["U"]).numpy()
    if attn:
        res["attn"] = (torch.stack(out["attn"]).sum(0) / len(x)).numpy()
    return res


# ------------------------------------------------------------------ metrics

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(prob: np.ndarray, y: np.ndarray) -> float:
    """One temperature T minimizing BCE of sigmoid(logit / T) on validation, over all 5 classes."""
    z = _logit(prob)

    def nll(log_t):
        p = np.clip(1 / (1 + np.exp(-z / np.exp(log_t))), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(prob: np.ndarray, t: float) -> np.ndarray:
    return (1 / (1 + np.exp(-_logit(prob) / t))).astype(np.float32)


def f1_per_class(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """y, pred: (N, K) binary -> (K,) F1 (0 when undefined)."""
    tp = (pred * y).sum(0)
    denom = pred.sum(0) + y.sum(0)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def tune_thresholds(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-class threshold in {0.01, ..., 0.99} maximizing F1 on validation. Returns (5,)."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    scores = np.stack([f1_per_class(y, (prob >= t).astype(float)) for t in grid])  # (G, K)
    return grid[scores.argmax(0)]


def reliability(prob: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    """Flattened over all records and classes. Returns (ece, bins DataFrame)."""
    p, t = prob.ravel(), y.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        conf = float(p[m].mean()) if n else np.nan
        acc = float(t[m].mean()) if n else np.nan
        if n:
            ece += n / len(p) * abs(acc - conf)
        rows.append({"bin": b, "bin_lower": edges[b], "bin_upper": edges[b + 1], "count": n,
                     "mean_pred": conf, "frac_positive": acc})
    return float(ece), pd.DataFrame(rows)


def per_class_auroc(y: np.ndarray, prob: np.ndarray) -> list[float]:
    return [roc_auc_score(y[:, k], prob[:, k]) if 0 < y[:, k].sum() < len(y) else np.nan for k in range(y.shape[1])]


def sigmoid_uncertainty(prob: np.ndarray) -> np.ndarray:
    """1 - mean_k max(p_k, 1 - p_k). (N,)"""
    return 1 - np.maximum(prob, 1 - prob).mean(1)


def risk_coverage(err: np.ndarray, unc: np.ndarray) -> list[tuple[float, float]]:
    """Keep the most certain records first; risk = mean per-record label error among the kept ones."""
    order = np.argsort(unc, kind="stable")
    return [(c, float(err[order[:max(1, int(np.ceil(c * len(err))))]].mean())) for c in COVERAGES]


def paired_bootstrap(y: np.ndarray, probs_a: list, probs_b: list, n_boot: int, seed: int = 0) -> dict:
    """Macro-AUROC difference (a - b), each side averaged over its seeds, on shared resamples of the test fold."""
    rng = np.random.default_rng(seed)
    point = np.mean([macro_auroc(y, p) for p in probs_a]) - np.mean([macro_auroc(y, p) for p in probs_b])
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean([macro_auroc(y[i], p[i]) for p in probs_a])
                     - np.mean([macro_auroc(y[i], p[i]) for p in probs_b]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot,
            "frac_boot_le_0": float(np.mean(np.array(diffs) <= 0))}


def condition_info(cond: str) -> dict:
    kind, _, arg = cond.partition("_")
    if cond == "clean":
        return {"family": "clean", "level": "", "n_missing": 0}
    if kind == "random":
        return {"family": "random", "level": arg, "n_missing": int(arg)}
    if kind == "group":
        return {"family": "group", "level": arg, "n_missing": len(corrupt.LEAD_GROUPS[arg])}
    return {"family": kind, "level": arg, "n_missing": 0}  # artifacts: leads are noisy, not removed


# --------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None, help="folder with ptbxl.npz (default: <repo>/cache)")
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--out-dir", default=str(REPO_DIR / "results"))
    p.add_argument("--strict", action="store_true", help="stop if any run of the run list is missing or unfinished")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--debug", action="store_true", help="debug data and checkpoints; 100 bootstrap samples")
    a = p.parse_args()
    t_start = time.time()
    ckpt_dir, out_dir = Path(a.ckpt_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    n_boot = 100 if a.debug else a.n_boot

    runs = load_runs(ckpt_dir, a.data, a.debug, device, strict=a.strict)  # before loading data: fail fast
    X, y, fold = load_dataset(a.data, a.data_root, debug=a.debug, cache_dir=a.cache_dir)
    x_va, y_va = torch.from_numpy(X[fold == 9]), y[fold == 9]
    x_te, y_te = torch.from_numpy(X[fold == 10]), y[fold == 10]
    del X
    sqa = load_sqa_checked(ckpt_dir / "sqa.pt", a.data, device) if any(r["model"] in USES_Q for r in runs) else None
    print(f"device {device} | val {len(y_va)} | test {len(y_te)} | runs {len(runs)}: {[r['run'] for r in runs]}")

    # Fitted on clean validation predictions.
    for r in runs:
        pv = predict(r, x_va, "estimated", None, sqa, device)
        r["T"] = 1.0 if r["model"] == "lg_edl" else fit_temperature(pv["prob"], y_va)
        if not np.exp(-2.99) < r["T"] < np.exp(2.99):
            print(f"WARNING: {r['run']} temperature {r['T']:.3f} is at the search bound [e^-3, e^3]")
        r["thr"] = tune_thresholds(apply_temperature(pv["prob"], r["T"]), y_va)
        if "U" in pv:
            r["u_thr"] = float(np.quantile(pv["U"], ABSTAIN_QUANTILE))

    metric_rows, pc_rows, rc_rows, ab_rows, cal_rows = [], [], [], [], []
    boot_preds: dict[str, dict[str, list]] = {}
    conditions = [("clean", 0)] + list(corrupt.TEST_CONDITIONS)
    for ci, (cond, draw) in enumerate(conditions):
        xc, sev = corrupt.fixed_test_corruption(x_te, cond, draw)
        info = condition_info(cond)
        n_bad = float((sev >= 0.5).sum(1).float().mean())
        for r in runs:
            for q_mode in (["estimated", "oracle"] if r["model"] in USES_Q else ["n/a"]):
                o = predict(r, xc, q_mode, sev, sqa, device)
                prob = apply_temperature(o["prob"], r["T"])
                pred = (prob >= r["thr"]).astype(float)
                ece, bins = reliability(prob, y_te)
                pca = per_class_auroc(y_te, prob)
                base = {"run": r["run"], "model": r["model"], "seed": r["seed"], "corrupt_train": r["corrupt_train"],
                        "condition": cond, "draw": draw, **info, "q_mode": q_mode}
                metric_rows.append({**base, "macro_auroc": macro_auroc(y_te, prob),
                                    "macro_f1": float(f1_per_class(y_te, pred).mean()), "ece": ece,
                                    "mean_bad_leads": n_bad, "temperature": r["T"]})
                pc_rows += [{**base, "class": c, "auroc": v} for c, v in zip(CLASSES, pca)]
                if cond == "clean":
                    cal_rows += [{**base, "temperature": r["T"], "ece": ece, **b} for b in bins.to_dict("records")]
                if q_mode == "oracle":
                    continue
                err = (pred != y_te).mean(1)
                scores = {"sigmoid_style": sigmoid_uncertainty(prob)}
                if "U" in o:
                    scores = {"edl_U": o["U"], **scores}
                for s_name, unc in scores.items():
                    rc_rows += [{**base, "score": s_name, "coverage": c, "risk": rk} for c, rk in risk_coverage(err, unc)]
                if r["model"] == "lg_edl":
                    by_u, by_q = o["U"] > r["u_thr"], o["q"].sum(1) < 3
                    ab = abstain(torch.from_numpy(o["U"]), torch.from_numpy(o["q"]), r["u_thr"]).numpy()
                    ab_rows.append({**base, "u_threshold": r["u_thr"], "abstain_rate": float(ab.mean()),
                                    "rate_U_above_threshold": float(by_u.mean()), "rate_qsum_below_3": float(by_q.mean()),
                                    "mean_U": float(o["U"].mean()), "mean_qsum": float(o["q"].sum(1).mean()),
                                    "mean_bad_leads": n_bad})
                if (r["model"], r["corrupt_train"]) in (("lg_edl", 1), ("resnet", 1)):
                    boot_preds.setdefault(f"{cond}|{draw}", {"lg_edl": [], "resnet": []})[r["model"]].append(prob)
        print(f"[{ci + 1:2d}/{len(conditions)}] {cond} (draw {draw}) | mean bad leads {n_bad:.2f} | "
              f"{time.time() - t_start:.0f}s", flush=True)

    boot_rows = []
    for key, d in boot_preds.items():
        if d["lg_edl"] and d["resnet"]:
            cond, draw = key.split("|")
            boot_rows.append({"condition": cond, "draw": int(draw), **condition_info(cond),
                              "n_seeds_lg_edl": len(d["lg_edl"]), "n_seeds_resnet": len(d["resnet"]),
                              **paired_bootstrap(y_te, d["lg_edl"], d["resnet"], n_boot)})

    # Attention for Figure 2: first lg_edl run, estimated q, clean vs V1 disconnected vs II disconnected.
    lg = next((r for r in runs if r["model"] == "lg_edl"), None)
    if lg is not None:
        maps, qs, names = [], [], ["clean", "V1 disconnected", "II disconnected"]
        for leads in ([], [6], [1]):
            xa = x_te.clone()
            xa[:, leads] = 0.0
            o = predict(lg, xa, "estimated", None, sqa, device, attn=True)
            maps.append(o["attn"])
            qs.append(o["q"].mean(0))
        np.savez(out_dir / "attention.npz", names=np.array(names), attn=np.stack(maps), q=np.stack(qs),
                 run=lg["run"])

    m = pd.DataFrame(metric_rows)
    est = m[m.q_mode != "oracle"]
    keep = ["run", "model", "seed", "corrupt_train", "condition", "draw", "family", "level", "n_missing",
            "q_mode", "macro_auroc", "macro_f1", "ece", "mean_bad_leads"]
    outputs = {
        "main_clean.csv": est[est.family == "clean"][keep + ["temperature"]],
        "random_loss.csv": est[est.family == "random"][keep],
        "group_loss.csv": est[est.family == "group"][keep],
        "artifact.csv": est[est.family.isin(["emg", "wander", "clip"])][keep],
        "oracle_vs_estimated.csv": m[m.model.isin(USES_Q)][keep],
        "calibration.csv": pd.DataFrame(cal_rows),
        "risk_coverage.csv": pd.DataFrame(rc_rows),
        "per_class_auroc.csv": pd.DataFrame(pc_rows),
        "abstain_rate.csv": pd.DataFrame(ab_rows),
        "bootstrap_ci.csv": pd.DataFrame(boot_rows),
    }
    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)
    print(f"wrote {len(outputs)} CSVs + attention.npz to {out_dir} in {time.time() - t_start:.0f}s:")
    for name, df in outputs.items():
        print(f"  {name:26s} {len(df):6d} rows")


if __name__ == "__main__":
    main()
