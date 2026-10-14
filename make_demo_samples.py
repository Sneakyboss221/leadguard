"""Export demo samples for app.py (SPEC section 8).

Picks test-fold records (4 per class, 20 in total) that both resnet (corruption-trained, seed 0) and lg_edl
(seed 0) classify fully correctly with all 12 leads: every one of the 5 labels right at the validation-tuned
thresholds. If fewer qualify (e.g. undertrained debug models), the remaining slots are filled with the records
having the most correct labels, and flagged with strict_correct = False.

Writes <out-dir>/sample_XX.npz (x: z-scored (12, 1000), x_raw: mV (12, 1000), y: (5,), metadata) and
<out-dir>/demo_config.json (validation-fitted temperatures, F1 thresholds and the lg_edl U threshold, computed
with the same functions as evaluate.py).

python make_demo_samples.py --data ptbxl --data-root PATH --ckpt-dir DIR [--debug]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_dataset
from evaluate import (ABSTAIN_QUANTILE, apply_temperature, fit_temperature, predict, tune_thresholds)
from models import build_model
from run_all import BATCH, EPOCHS
from train import USES_Q, check_status, ckpt_paths, load_sqa_checked, run_config, run_name
from utils import CLASSES, REPO_DIR, get_device

DEMO_RUNS = {"resnet_c0": ("resnet", 0, 0), "resnet_c1": ("resnet", 1, 0), "lg_edl": ("lg_edl", 1, 0)}
PER_CLASS = 4


def load_run(ckpt_dir: Path, model: str, c: int, seed: int, data: str, debug: bool, device) -> dict:
    status, _ = check_status(ckpt_dir, run_config(model, seed, c, data, debug, EPOCHS, BATCH))
    if status != "finished":
        raise RuntimeError(f"{run_name(model, seed, c)} is {status}; run run_all.py first")
    net = build_model(model).to(device)
    best, _ = ckpt_paths(ckpt_dir, model, seed, c)
    net.load_state_dict(torch.load(best, map_location=device, weights_only=False)["state_dict"])
    return {"run": run_name(model, seed, c), "model": model, "seed": seed, "corrupt_train": c, "obj": net.eval()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--results-dir", default=str(REPO_DIR / "results"), help="to cross-check the U threshold")
    p.add_argument("--out-dir", default=str(REPO_DIR / "demo_samples"))
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    ckpt_dir, out_dir = Path(a.ckpt_dir), Path(a.out_dir)
    device = get_device()

    X, y, fold, X_raw = load_dataset(a.data, a.data_root, debug=a.debug, return_raw=True)
    idx_va, idx_te = np.where(fold == 9)[0], np.where(fold == 10)[0]
    x_va, y_va = torch.from_numpy(X[idx_va]), y[idx_va]
    x_te, y_te = torch.from_numpy(X[idx_te]), y[idx_te]
    sqa = load_sqa_checked(ckpt_dir / "sqa.pt", a.data, device)

    config, probs, correct = {"data": a.data, "debug": a.debug, "u_quantile": ABSTAIN_QUANTILE,
                              "abstain_q_sum": 3.0, "classes": CLASSES, "runs": {}}, {}, {}
    for key, (model, c, seed) in DEMO_RUNS.items():
        r = load_run(ckpt_dir, model, c, seed, a.data, a.debug, device)
        pv = predict(r, x_va, "estimated", None, sqa, device)
        t = 1.0 if model == "lg_edl" else fit_temperature(pv["prob"], y_va)
        thr = tune_thresholds(apply_temperature(pv["prob"], t), y_va)
        entry = {"run": r["run"], "temperature": t, "thresholds": thr.tolist()}
        if "U" in pv:
            entry["u_threshold"] = float(np.quantile(pv["U"], ABSTAIN_QUANTILE))
        config["runs"][key] = entry
        probs[key] = apply_temperature(predict(r, x_te, "estimated", None, sqa, device)["prob"], t)
        correct[key] = ((probs[key] >= thr) == (y_te > 0.5)).sum(1)  # labels right, 0-5

    ab_csv = Path(a.results_dir) / "abstain_rate.csv"
    if ab_csv.exists():
        ab = pd.read_csv(ab_csv)
        ref = ab[(ab.run == config["runs"]["lg_edl"]["run"]) & (ab.condition == "clean")].u_threshold
        if len(ref):
            same = np.isclose(ref.iloc[0], config["runs"]["lg_edl"]["u_threshold"], atol=1e-5)
            print(f"U threshold {config['runs']['lg_edl']['u_threshold']:.4f} vs abstain_rate.csv "
                  f"{ref.iloc[0]:.4f}: {'match' if same else 'MISMATCH'}")

    both = correct["resnet_c1"] + correct["lg_edl"]  # 0-10
    strict = (correct["resnet_c1"] == 5) & (correct["lg_edl"] == 5)
    chosen: list[tuple[int, str]] = []
    for k, cls in enumerate(CLASSES):
        cand = [i for i in np.where(y_te[:, k] > 0.5)[0] if i not in {j for j, _ in chosen}]
        cand.sort(key=lambda i: (-int(strict[i]), -int(both[i]), -float(probs["lg_edl"][i, k])))
        chosen += [(i, cls) for i in cand[:PER_CLASS]]

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sample_*.npz"):
        old.unlink()
    for n, (i, cls) in enumerate(chosen):
        rec = int(idx_te[i])
        np.savez(out_dir / f"sample_{n:02d}.npz", x=X[rec], x_raw=X_raw[rec], y=y[rec], record_index=rec,
                 fold=int(fold[rec]), target_class=cls, strict_correct=bool(strict[i]),
                 data=a.data, labels_correct_resnet=int(correct["resnet_c1"][i]),
                 labels_correct_lg_edl=int(correct["lg_edl"][i]))
    with open(out_dir / "demo_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"test records {len(y_te)} | fully correct for both models: {int(strict.sum())}")
    for n, (i, cls) in enumerate(chosen):
        labels = "+".join(c for c, v in zip(CLASSES, y_te[i]) if v > 0.5)
        print(f"  sample_{n:02d}  target {cls:4s}  labels {labels:14s}  correct labels resnet "
              f"{correct['resnet_c1'][i]}/5, lg_edl {correct['lg_edl'][i]}/5  {'' if strict[i] else '(fallback)'}")
    print(f"saved {len(chosen)} samples and demo_config.json to {out_dir}")


if __name__ == "__main__":
    main()
