"""SPEC section 5, tests 1-8. Nothing trains until these pass: python -m pytest -q tests.py
(Test 9, the SQA accuracy check, runs on real data via pretrain_sqa.py.)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

import corrupt
from baseline_xgb import FEATURES, lead_features
from data import load_dataset
from models import MODELS, Q_FLOOR, build_model, count_params, edl_loss, kl_anneal
from utils import set_seed

B, T = 4, 1000
LG_MODELS = ["lg_hard", "lg_soft", "lg_edl"]
BASELINES = ["mlp", "resnet", "transformer", "resnet_attn"]
ATOL = 1e-5


def _model(name: str) -> torch.nn.Module:
    set_seed(0)
    return build_model(name).eval()


def _x() -> torch.Tensor:
    set_seed(1)
    return torch.randn(B, 12, T)


def _replace_with_noise(x: torch.Tensor, leads: list[int]) -> torch.Tensor:
    x2 = x.clone()
    x2[:, leads] = torch.randn(B, len(leads), T, generator=torch.Generator().manual_seed(7)) * 5
    return x2


# 1. Hard-mask invariance
def test_hard_mask_invariance():
    m, x = _model("lg_hard"), _x()
    mask = torch.ones(B, 12)
    leads = [0, 6, 7]
    mask[:, leads] = 0
    with torch.no_grad():
        a = m(x, mask)["logits"]
        b = m(_replace_with_noise(x, leads), mask)["logits"]
    assert torch.allclose(a, b, atol=ATOL), (a - b).abs().max()


# 2. Soft invariance (SQA bypassed, q_j = 0 injected)
@pytest.mark.parametrize("name", ["lg_soft", "lg_edl"])
def test_soft_invariance(name):
    m, x = _model(name), _x()
    q = torch.rand(B, 12, generator=torch.Generator().manual_seed(3)) * 0.9 + 0.1
    leads = [1, 4, 9, 11]
    q[:, leads] = 0
    with torch.no_grad():
        a = m(x, q)["logits"]
        b = m(_replace_with_noise(x, leads), q)["logits"]
    assert torch.allclose(a, b, atol=ATOL), (a - b).abs().max()


# 3. Attention on floored keys is exactly 0 and rows sum to 1
@pytest.mark.parametrize("name", LG_MODELS)
def test_attention_floored_keys(name):
    m, x = _model(name), _x()
    q = torch.full((B, 12), 0.9)
    floored = [2, 5, 8]
    q[:, floored] = Q_FLOOR / 2
    with torch.no_grad():
        attn = m(x, q)["attn"]
    assert len(attn) == 2
    for a in attn:
        assert a.shape == (B, 4, 12, 12)
        assert (a[..., floored] == 0).all()
        assert torch.allclose(a.sum(-1), torch.ones(B, 4, 12), atol=ATOL)


# 4. Degenerate inputs: single active lead, or all q near 0
@pytest.mark.parametrize("name", LG_MODELS)
def test_degenerate_inputs(name):
    m, x = _model(name), _x()
    single = torch.zeros(B, 12)
    single[:, 3] = 1.0
    near_zero = torch.full((B, 12), 1e-4)
    all_zero = torch.zeros(B, 12)
    with torch.no_grad():
        for q in (single, near_zero, all_zero):
            out = m(x, q)
            for key in ("logits", "prob"):
                assert torch.isfinite(out[key]).all(), (key, q[0])
            for a in out["attn"]:
                assert torch.isfinite(a).all()


# 5. Shapes
@pytest.mark.parametrize("name", list(MODELS))
def test_shapes(name):
    m, x = _model(name), _x()
    with torch.no_grad():
        out = m(x, torch.ones(B, 12))
    assert out["prob"].shape == (B, 5)
    assert out["logits"].shape == ((B, 10) if name == "lg_edl" else (B, 5))
    assert torch.isfinite(out["prob"]).all()
    if name == "lg_edl":
        for key in ("alpha", "beta", "u"):
            assert out[key].shape == (B, 5)
        assert out["U"].shape == (B,)
        assert ((out["u"] > 0) & (out["u"] <= 1)).all()
    if name in LG_MODELS + ["resnet_attn"]:
        assert [a.shape for a in out["attn"]] == [(B, 4, 12, 12)] * 2
        assert count_params(m.encoder) + sum(count_params(l) for l in m.layers) < 1_000_000
    if name in BASELINES:  # baselines ignore the second argument
        with torch.no_grad():
            out0 = m(x, torch.zeros(B, 12))
        assert torch.equal(out["logits"], out0["logits"])


def test_xgb_features():
    x = _x().numpy()
    x[0, [6, 7]] = 0.0  # two missing leads in record 0
    f = lead_features(x).reshape(B, 12, len(FEATURES))
    assert f.shape == (B, 12, 8)
    assert np.isnan(f[0, [6, 7]]).all()
    assert np.isfinite(f[0, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]).all() and np.isfinite(f[1:]).all()


# 6. Corruption masks: severity exactly 0 on untouched leads
def test_corruption_severity_zero_on_untouched():
    x = _x()
    g = lambda: torch.Generator().manual_seed(0)
    leads = torch.zeros(B, 12, dtype=torch.bool)
    leads[:, [1, 6, 10]] = True
    cases = {
        "drop_random": corrupt.drop_random(x, 4, g()),
        "drop_group": corrupt.drop_group(x, "v1_v3"),
        "flatline": corrupt.flatline(x, leads),
        "emg": corrupt.add_emg_noise(x, leads, 10.0, g()),
        "wander": corrupt.add_baseline_wander(x, leads, gen=g()),
        "clip": corrupt.clip_amplitude(x, leads, gen=g()),
        "training": corrupt.training_corruption(x, g()),
    }
    for name, (xc, sev) in cases.items():
        assert sev.shape == (B, 12), name
        assert ((sev >= 0) & (sev <= 1)).all(), name
        untouched = ~(xc != x).any(-1)
        assert (sev[untouched] == 0).all(), name
        assert (untouched | (sev > 0)).all(), name  # every changed lead reports severity > 0


# 7. Determinism of the fixed test corruption
def test_fixed_test_corruption_deterministic():
    x = _x()
    for cond, seed in corrupt.TEST_CONDITIONS:
        a, sa = corrupt.fixed_test_corruption(x, cond, seed)
        b, sb = corrupt.fixed_test_corruption(x, cond, seed)
        assert torch.equal(a, b) and torch.equal(sa, sb), cond


# 8. Overfit: lg_edl on 200 samples for 50 epochs reaches training macro-AUROC > 0.95
def test_overfit_lg_edl():
    set_seed(0)
    X, y, _ = load_dataset("synthetic")
    x, yt = torch.from_numpy(X[:200]), torch.from_numpy(y[:200])
    q = torch.ones(len(x), 12)
    m = build_model("lg_edl")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(50):
        m.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i:i + 32]
            out = m(x[idx], q[idx])
            loss = edl_loss(out["alpha"], out["beta"], yt[idx], kl_anneal(epoch))
            assert torch.isfinite(loss), epoch
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        prob = torch.cat([m(x[i:i + 50], q[i:i + 50])["prob"] for i in range(0, len(x), 50)]).numpy()
    auroc = np.mean([roc_auc_score(y[:200, k], prob[:, k]) for k in range(5)])
    print(f"overfit training macro-AUROC = {auroc:.4f}")
    assert auroc > 0.95
