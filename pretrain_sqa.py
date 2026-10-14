"""Pre-train the Signal Quality Assessor (SPEC section 3A) and check its lead-level agreement.

Trains on the training folds (1-8) with fresh `training_corruption` every batch, BCE against the soft
target q_target = 1 - severity. Model selection on fold 9; the agreement check is reported on fold 10,
per corruption type, and written to <out-dir>/sqa_agreement.csv. Saves <ckpt-dir>/sqa.pt.

Agreement = accuracy of (q >= 0.5) against true_mask = (severity < 0.5), reported:
  all leads | excluding the ambiguous band |severity - 0.5| < 0.1 (the gate, see DECISIONS.md) |
  corrupted leads only (the harder subset).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import corrupt
from data import load_dataset
from quality import LeadQualityNet
from utils import REPO_DIR, get_device, set_seed

BAND = 0.1
GATE = 0.95


def eval_sets(x: torch.Tensor, seed: int = 0) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic corrupted copies of x, one per corruption type (3 random leads unless stated)."""
    def gen(i):
        return torch.Generator().manual_seed(seed * 101 + i)

    b = x.shape[0]
    three = lambda i: corrupt.random_lead_mask(b, 3, gen(i), x.device)
    snr = 20 * torch.rand(b, 12, generator=gen(99)).to(x.device)
    return {
        "clean": (x.clone(), torch.zeros(b, 12, device=x.device)),
        "drop_random_4": corrupt.drop_random(x, 4, gen(1)),
        "drop_group": corrupt.drop_group(x, "v1_v3"),
        "flatline": corrupt.flatline(x, three(2)),
        "emg_0_20db": corrupt.add_emg_noise(x, three(3), snr, gen(4)),
        "emg_10db_test": corrupt.add_emg_noise(x, three(5), 10.0, gen(6)),
        "wander": corrupt.add_baseline_wander(x, three(7), gen=gen(8)),
        "clip": corrupt.clip_amplitude(x, three(9), gen=gen(10)),
        "training_mix": corrupt.training_corruption(x, gen(11)),
    }


@torch.no_grad()
def predict(model: LeadQualityNet, x: torch.Tensor, batch: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def agreement_table(model, sets) -> pd.DataFrame:
    rows = []
    for name, (xc, sev) in sets.items():
        q = predict(model, xc)
        correct = (q >= 0.5) == (sev < 0.5)
        keep = (sev - 0.5).abs() >= BAND
        bad = sev > 0
        clean = sev == 0
        rows.append({
            "corruption": name,
            "n_leads": correct.numel(),
            "agreement_all": correct.float().mean().item(),
            "agreement_excl_band": correct[keep].float().mean().item() if keep.any() else float("nan"),
            "frac_in_band": 1 - keep.float().mean().item(),
            "agreement_corrupted_leads": correct[bad].float().mean().item() if bad.any() else float("nan"),
            "mean_q_untouched": q[clean].mean().item(),
            "frac_untouched_q_gt_0.8": (q[clean] > 0.8).float().mean().item(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None, help="folder with ptbxl.npz (default: <repo>/cache)")
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--out-dir", default=str(REPO_DIR / "results"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="500 records, 2 epochs")
    a = p.parse_args()

    set_seed(a.seed)
    device = get_device()
    epochs = 2 if a.debug else a.epochs
    X, _, fold = load_dataset(a.data, a.data_root, debug=a.debug, cache_dir=a.cache_dir)
    x_tr = torch.from_numpy(X[fold <= 8]).to(device)
    val_sets = eval_sets(torch.from_numpy(X[fold == 9]).to(device), seed=1)
    test_sets = eval_sets(torch.from_numpy(X[fold == 10]).to(device), seed=2)
    del X
    print(f"device {device} | train records {len(x_tr)} | epochs {epochs}")

    model = LeadQualityNet().to(device)
    print(f"SQA parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = epochs * ((len(x_tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device).manual_seed(a.seed)

    ckpt_dir = Path(a.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        t0, total = time.time(), 0.0
        model.train()
        perm = torch.randperm(len(x_tr), device=device, generator=gen)
        for i in range(0, len(x_tr), a.batch):
            xb = x_tr[perm[i:i + a.batch]]
            xc, sev = corrupt.training_corruption(xb, gen)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model.logits(xc)
            loss = F.binary_cross_entropy_with_logits(logits.float(), 1 - sev)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(xb)

        with torch.no_grad():
            val_loss = sum(F.binary_cross_entropy(predict(model, xc).clamp(1e-6, 1 - 1e-6), 1 - s).item()
                           for xc, s in val_sets.values()) / len(val_sets)
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d} | train BCE {total / len(x_tr):.4f} | val BCE {val_loss:.4f} | {time.time() - t0:.1f}s")

    model.load_state_dict(best_state)
    table = agreement_table(model, test_sets)
    gate_value = table["agreement_excl_band"].min()
    torch.save({"state_dict": best_state, "config": {}, "data": a.data,
                "val_bce": best, "agreement": table.to_dict("records")}, ckpt_dir / "sqa.pt")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "sqa_agreement.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table.to_string(index=False))
    print(f"gate (min agreement_excl_band over types) = {gate_value:.3f} -> "
          f"{'PASS' if gate_value > GATE else 'FAIL'} (needs > {GATE})")
    print(f"saved {ckpt_dir / 'sqa.pt'} and {out_dir / 'sqa_agreement.csv'}")


if __name__ == "__main__":
    main()
