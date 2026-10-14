"""Train one model (SPEC section 4).

python train.py --model lg_edl --seed 0 --corrupt-train 1 --data ptbxl --data-root PATH --ckpt-dir DIR
                --out-dir DIR [--cache-dir DIR] [--epochs 15] [--batch 128] [--debug]

Checkpoints in --ckpt-dir, written atomically after every epoch:
  {model}_s{seed}_c{c}.pt       best weights so far (by validation macro-AUROC), used for evaluation
  {model}_s{seed}_c{c}_last.pt  full training state (optimizer, scheduler, scaler, RNGs, early-stopping
                                counter, history) plus a `finished` flag
Before doing anything, the _last.pt checkpoint is checked: finished -> skip; unfinished -> resume at the next
epoch with identical state; written under a different data/debug/epochs/batch setting -> refuse.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import corrupt
from data import load_dataset
from models import build_model, edl_loss, kl_anneal
from quality import estimate_q, load_sqa
from utils import REPO_DIR, atomic_save, get_device, macro_auroc, set_seed

USES_Q = {"lg_hard", "lg_soft", "lg_edl"}  # models that receive q from the frozen SQA
PATIENCE = 4
DEBUG_EPOCHS = 2
CONFIG_KEYS = ("model", "seed", "corrupt_train", "data", "debug", "epochs", "batch")


def run_name(model: str, seed: int, corrupt_train: int) -> str:
    return f"{model}_s{seed}_c{corrupt_train}"


def ckpt_paths(ckpt_dir, model: str, seed: int, corrupt_train: int) -> tuple[Path, Path]:
    """(best, last) checkpoint paths."""
    name = run_name(model, seed, corrupt_train)
    return Path(ckpt_dir) / f"{name}.pt", Path(ckpt_dir) / f"{name}_last.pt"


def run_config(model, seed, corrupt_train, data, debug, epochs, batch) -> dict:
    return {"model": model, "seed": seed, "corrupt_train": corrupt_train, "data": data, "debug": bool(debug),
            "epochs": DEBUG_EPOCHS if debug else epochs, "batch": batch}


def check_status(ckpt_dir, cfg: dict) -> tuple[str, dict | None]:
    """Inspect the _last.pt checkpoint of a run. Returns (status, state):
    "new" (nothing to resume), "partial" (resume from state) or "finished".
    Raises if a checkpoint exists that was written under a different configuration.
    """
    best, last = ckpt_paths(ckpt_dir, cfg["model"], cfg["seed"], cfg["corrupt_train"])
    if not last.exists():
        return "new", None
    try:
        state = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as e:  # unreadable file: keep it for inspection, start over
        bad = last.with_name(last.name + ".unreadable")
        last.replace(bad)
        print(f"WARNING: could not read {last} ({e}); moved to {bad.name}, starting this run from scratch")
        return "new", None
    diff = {k: (state["config"].get(k), cfg[k]) for k in CONFIG_KEYS if state["config"].get(k) != cfg[k]}
    if diff:
        raise RuntimeError(f"{last} was written with a different setting {diff} (checkpoint, requested). "
                           "Use another --ckpt-dir or delete the old checkpoints.")
    if state.get("finished") and best.exists():
        return "finished", state
    return "partial", state


def load_sqa_checked(path: Path, data: str, device) -> torch.nn.Module:
    """Load the frozen SQA, refusing one pre-trained on a different dataset."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run pretrain_sqa.py --data {data} first")
    trained_on = torch.load(path, map_location="cpu", weights_only=False).get("data")
    if trained_on != data:
        raise RuntimeError(f"{path} was pre-trained on '{trained_on}', but this run uses '{data}'")
    return load_sqa(path, device)


def _rng_state(gen: torch.Generator) -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
         "gen": gen.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict, gen: torch.Generator) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    gen.set_state(s["gen"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@torch.no_grad()
def predict(model, x: torch.Tensor, sqa=None, batch: int = 512) -> np.ndarray:
    """x: (N, 12, T) on the model's device -> prob (N, 5) as numpy. q from the SQA if given, else ones."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        q = estimate_q(sqa, xb) if sqa is not None else torch.ones(len(xb), 12, device=xb.device)
        with torch.autocast(xb.device.type, enabled=xb.device.type == "cuda"):
            out.append(model(xb, q)["prob"].float().cpu())
    return torch.cat(out).numpy()


def train(a: argparse.Namespace) -> None:
    cfg = run_config(a.model, a.seed, a.corrupt_train, a.data, a.debug, a.epochs, a.batch)
    name = run_name(a.model, a.seed, a.corrupt_train)
    best_path, last_path = ckpt_paths(a.ckpt_dir, a.model, a.seed, a.corrupt_train)
    status, state = check_status(a.ckpt_dir, cfg)
    if status == "finished":
        print(f"{name}: already finished (best val macro-AUROC {state['best_auroc']:.4f}), skipping")
        return

    set_seed(a.seed)
    device = get_device()
    X, y, fold = load_dataset(a.data, a.data_root, debug=a.debug, cache_dir=a.cache_dir)
    x_tr = torch.from_numpy(X[fold <= 8]).to(device)
    y_tr = torch.from_numpy(y[fold <= 8]).to(device)
    x_va, y_va = torch.from_numpy(X[fold == 9]).to(device), y[fold == 9]
    del X

    model = build_model(a.model).to(device)
    sqa = load_sqa_checked(Path(a.sqa or Path(a.ckpt_dir) / "sqa.pt"), a.data, device) if a.model in USES_Q else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps_per_epoch = (len(x_tr) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * steps_per_epoch)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device).manual_seed(a.seed)

    start_epoch, best_auroc, bad_epochs, history, elapsed = 0, -float("inf"), 0, [], 0.0
    if status == "partial":
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _set_rng_state(state["rng"], gen)
        start_epoch, best_auroc, bad_epochs = state["epoch"] + 1, state["best_auroc"], state["bad_epochs"]
        history, elapsed = state["history"], state["elapsed"]
        print(f"{name}: resuming at epoch {start_epoch + 1}/{cfg['epochs']} (best so far {best_auroc:.4f})")
    print(f"{name}: device {device} | train {len(x_tr)} | val {len(x_va)} | epochs {cfg['epochs']} | "
          f"corrupt-train {a.corrupt_train} | SQA q {'yes' if sqa is not None else 'no'}")

    def save_last(epoch: int, finished: bool) -> None:
        atomic_save({"config": cfg, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(gen),
                     "epoch": epoch, "best_auroc": best_auroc, "bad_epochs": bad_epochs, "history": history,
                     "elapsed": elapsed, "finished": finished}, last_path)

    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        if bad_epochs >= PATIENCE:
            break
        t0, total, n = time.time(), 0.0, 0
        model.train()
        perm = torch.randperm(len(x_tr), device=device, generator=gen)
        for i in range(0, len(x_tr), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:  # BatchNorm cannot train on a single sample
                continue
            xb, yb = x_tr[idx], y_tr[idx]
            if a.corrupt_train:
                xb, _ = corrupt.training_corruption(xb, gen)  # true severities never reach the classifier
            q = estimate_q(sqa, xb) if sqa is not None else torch.ones(len(xb), 12, device=device)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(xb, q)
            if a.model == "lg_edl":
                loss = edl_loss(out["alpha"].float(), out["beta"].float(), yb, kl_anneal(epoch))
            else:
                loss = F.binary_cross_entropy_with_logits(out["logits"].float(), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss {loss.item()} at epoch {epoch + 1}, step {i // a.batch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)

        val_auroc = macro_auroc(y_va, predict(model, x_va, sqa))
        improved = val_auroc > best_auroc or not best_path.exists()
        if improved:
            best_auroc, bad_epochs = val_auroc, 0
            atomic_save({"config": cfg, "state_dict": model.state_dict(), "epoch": epoch,
                         "val_auroc": val_auroc}, best_path)
        else:
            bad_epochs += 1
        dt = time.time() - t0
        elapsed += dt
        history.append({"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_auroc": val_auroc, "seconds": dt})
        done = epoch == cfg["epochs"] - 1 or bad_epochs >= PATIENCE
        save_last(epoch, finished=done)
        print(f"{name} | epoch {epoch + 1:2d} | loss {total / max(n, 1):.4f} | val macro-AUROC {val_auroc:.4f}"
              f"{' *' if improved else ''} | {dt:.1f}s")
        if done:
            break

    save_last(epoch, finished=True)  # also covers resuming a run that had already met its stop condition
    print(f"{name}: finished | best val macro-AUROC {best_auroc:.4f} | training time {elapsed:.1f}s | {best_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corrupt-train", type=int, default=1, choices=[0, 1])
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None, help="folder with ptbxl.npz (default: <repo>/cache)")
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--out-dir", default=str(REPO_DIR / "results"))
    p.add_argument("--sqa", default=None, help="SQA checkpoint (default: <ckpt-dir>/sqa.pt)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--debug", action="store_true", help=f"500 records, {DEBUG_EPOCHS} epochs")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
