"""Run the lean run list (SPEC section 4) one after another.

Before each run its final checkpoint is checked: finished runs are skipped, unfinished ones resume (train.py
picks up its _last.pt), new ones start. Each run executes as its own process; its wall-clock time is
appended to <out-dir>/timing.csv. Stops at the first failing run.

python run_all.py --data ptbxl --data-root PATH --ckpt-dir DIR --out-dir DIR [--cache-dir DIR] [--only RUN ...] [--debug]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


from baseline_xgb import load_xgb
from train import USES_Q, check_status, load_sqa_checked, run_config, run_name
from utils import REPO_DIR, get_device

# (model, corrupt_train, seed); corrupt_train None = not applicable (XGBoost)
RUNS = (
    [("xgb", None, 0), ("mlp", 1, 0), ("resnet", 0, 0)]
    + [("resnet", 1, s) for s in (0, 1, 2)]
    + [("transformer", 1, 0), ("resnet_attn", 1, 0), ("lg_hard", 1, 0), ("lg_soft", 1, 0)]
    + [("lg_edl", 1, s) for s in (0, 1, 2)]
)
TIMING_FIELDS = ["run", "model", "seed", "corrupt_train", "status", "resumed", "seconds", "device", "data",
                 "debug", "finished_at"]
EPOCHS, BATCH = 15, 128


def xgb_status(ckpt_dir: Path, seed: int, data: str, debug: bool) -> str:
    path = ckpt_dir / f"xgb_s{seed}.pkl"
    if not path.exists():
        return "new"
    _, meta = load_xgb(path)
    if (meta.get("data"), meta.get("debug")) != (data, debug):
        raise RuntimeError(f"{path} was trained with data={meta.get('data')}, debug={meta.get('debug')}; "
                           f"requested data={data}, debug={debug}. Use another --ckpt-dir or delete it.")
    return "finished"


def append_timing(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None, help="folder with ptbxl.npz (default: <repo>/cache)")
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--out-dir", default=str(REPO_DIR / "results"))
    p.add_argument("--only", nargs="+", default=None, metavar="RUN",
                   help="run only these entries of the run list, e.g. --only resnet_s0_c1 lg_edl_s0_c1")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    ckpt_dir, out_dir = Path(a.ckpt_dir), Path(a.out_dir)
    timing = out_dir / "timing.csv"
    device = get_device().type

    runs = RUNS
    if a.only:
        names = {(f"xgb_s{s}" if m == "xgb" else run_name(m, s, c)): (m, c, s) for m, c, s in RUNS}
        unknown = sorted(set(a.only) - set(names))
        if unknown:
            sys.exit(f"--only: not in the run list: {unknown}; valid names: {list(names)}")
        runs = [names[n] for n in names if n in a.only]

    # Check every run first, so a bad setup fails now rather than an hour in.
    plan = []
    for model, c, seed in runs:
        if model == "xgb":
            status = xgb_status(ckpt_dir, seed, a.data, a.debug)
        else:
            status, _ = check_status(ckpt_dir, run_config(model, seed, c, a.data, a.debug, EPOCHS, BATCH))
        plan.append((model, c, seed, status))
    if any(m in USES_Q and s != "finished" for m, _, _, s in plan):
        load_sqa_checked(ckpt_dir / "sqa.pt", a.data, "cpu")
    todo = [r for r in plan if r[3] != "finished"]
    print(f"run list: {len(plan)} runs | finished {len(plan) - len(todo)} | "
          f"to resume {sum(r[3] == 'partial' for r in todo)} | new {sum(r[3] == 'new' for r in todo)}")

    for model, c, seed, status in plan:
        name = f"xgb_s{seed}" if model == "xgb" else run_name(model, seed, c)
        if status == "finished":
            print(f"[skip] {name}: final checkpoint exists")
            continue
        print(f"[{'resume' if status == 'partial' else 'start'}] {name}", flush=True)
        common = ["--data", a.data, "--ckpt-dir", str(ckpt_dir), "--seed", str(seed)]
        common += ["--data-root", a.data_root] if a.data_root else []
        common += ["--cache-dir", a.cache_dir] if a.cache_dir else []
        common += ["--debug"] if a.debug else []
        if model == "xgb":
            cmd = [sys.executable, "-u", str(REPO_DIR / "baseline_xgb.py")] + common
        else:
            cmd = [sys.executable, "-u", str(REPO_DIR / "train.py"), "--model", model, "--corrupt-train", str(c),
                   "--out-dir", str(out_dir), "--epochs", str(EPOCHS), "--batch", str(BATCH)] + common
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO_DIR).returncode
        row = {"run": name, "model": model, "seed": seed, "corrupt_train": "n/a" if c is None else c,
               "status": "ok" if rc == 0 else f"failed (exit {rc})", "resumed": status == "partial",
               "seconds": round(time.time() - t0, 1), "device": device, "data": a.data, "debug": a.debug,
               "finished_at": datetime.now().isoformat(timespec="seconds")}
        append_timing(timing, row)
        if rc != 0:
            sys.exit(f"{name} failed with exit code {rc}; fix it and re-run the same command (finished runs are skipped)")
    print(f"all {len(plan)} runs finished; timing in {timing}")


if __name__ == "__main__":
    main()
