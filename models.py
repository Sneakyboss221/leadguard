"""LeadGuard models (SPEC sections 3B, 3C, 3D).

Every model: forward(x: (B, 12, T), q: (B, 12)) -> dict with at least
  "logits": raw head output ((B, 5) sigmoid head, (B, 10) Beta-EDL head)
  "prob":   (B, 5) per-class probability
LeadGuard models also return "attn": list (one per layer) of (B, heads, 12, 12) attention weights,
and the Beta-EDL head adds "alpha", "beta", "u" (B, 5) and "U" (B,).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEADS, N_CLASSES, D_MODEL = 12, 5, 128
Q_FLOOR = 0.02
MASK_BIAS = -1e9


# ------------------------------------------------------------- lead encoder

class ResBlock1d(nn.Module):
    """Basic residual block: two k=7 convs, BatchNorm, GELU; strided 1x1 shortcut when shape changes."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.short = nn.Identity() if (c_in == c_out and stride == 1) else nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(self.bn2(self.conv2(h)) + self.short(x))


class LeadEncoder(nn.Module):
    """Shared ResNet-1D applied to each lead separately, plus a learned per-lead-position embedding.

    x: (B, 12, T) -> h: (B, 12, 128)
    """

    def __init__(self, d: int = D_MODEL, channels: tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels[0], 7, 2, 3, bias=False),
                                  nn.BatchNorm1d(channels[0]), nn.GELU())
        stages, c_in = [], channels[0]
        for c in channels:
            stages.append(ResBlock1d(c_in, c, stride=2))
            c_in = c
        self.stages = nn.Sequential(*stages)
        self.proj = nn.Linear(c_in, d)
        self.lead_emb = nn.Parameter(torch.randn(N_LEADS, d) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h = self.stages(self.stem(x.reshape(b * c, 1, t))).mean(-1)
        return self.proj(h).view(b, c, -1) + self.lead_emb


# --------------------------------------------------------------------- MALA

def gate_weights(q: torch.Tensor, guard_q: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the q floor and the all-floored guard.

    q: (B, 12) -> (key_bias (B, 12), pool_w (B, 12)).
    Leads with q < 0.02 get key bias -1e9 and pooling weight 0. If every lead of a record is below the
    floor, the lead with the highest q is kept as fully active (key bias 0, pooling weight 1).
    guard_q: scores used to pick that lead (the hard variant passes the soft q, since its 0/1 q has ties).
    """
    guard_q = q if guard_q is None else guard_q
    active = q >= Q_FLOOR
    none = ~active.any(1)
    keep = torch.zeros_like(active)
    if none.any():
        keep[none] = F.one_hot(guard_q[none].argmax(1), q.shape[1]).bool()
    bias = torch.where(active, torch.log(q.clamp(min=1e-6)), torch.full_like(q, MASK_BIAS))
    bias = torch.where(keep, torch.zeros_like(q), bias)
    pool_w = torch.where(active, q, torch.zeros_like(q))
    pool_w = torch.where(keep, torch.ones_like(q), pool_w)
    return bias, pool_w


class MaskedLeadAttention(nn.Module):
    """Multi-head self-attention over leads with an additive per-key bias, written by hand.

    softmax(Q K^T / sqrt(d_k) + bias_j) over keys j. h: (B, 12, d), key_bias: (B, 12).
    Returns (out (B, 12, d), attn (B, heads, 12, 12)).
    """

    def __init__(self, d: int = D_MODEL, heads: int = 4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, h: torch.Tensor, key_bias: torch.Tensor):
        b, n, d = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each (B, H, n, dk)
        scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dk) + key_bias[:, None, None, :].float()
        attn = scores.softmax(-1)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(out), attn


class MALALayer(nn.Module):
    """Pre-norm transformer layer: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d: int = D_MODEL, heads: int = 4, ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MaskedLeadAttention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, key_bias: torch.Tensor):
        a, attn = self.attn(self.ln1(h), key_bias)
        h = h + self.drop(a)
        h = h + self.drop(self.ff(self.ln2(h)))
        return h, attn


# -------------------------------------------------------------------- heads

class SigmoidHead(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.fc = nn.Linear(d, N_CLASSES)

    def forward(self, z: torch.Tensor) -> dict:
        logits = self.fc(z)
        return {"logits": logits, "prob": torch.sigmoid(logits)}


class BetaEDLHead(nn.Module):
    """Linear to 10 outputs, softplus -> evidence e_plus, e_minus >= 0 per class.

    alpha = e_plus + 1, beta = e_minus + 1, p = alpha / (alpha + beta), u = 2 / (alpha + beta), U = mean_k u_k.
    """

    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.fc = nn.Linear(d, 2 * N_CLASSES)

    def forward(self, z: torch.Tensor) -> dict:
        logits = self.fc(z)
        ev = F.softplus(logits.float())
        alpha, beta = ev[:, :N_CLASSES] + 1, ev[:, N_CLASSES:] + 1
        s = alpha + beta
        u = 2 / s
        return {"logits": logits, "alpha": alpha, "beta": beta, "prob": alpha / s, "u": u, "U": u.mean(1)}


def _kl_beta_uniform(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """KL(Beta(a, b) || Beta(1, 1)), elementwise."""
    return (torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
            + (a - 1) * torch.digamma(a) + (b - 1) * torch.digamma(b) - (a + b - 2) * torch.digamma(a + b))


def edl_loss(alpha: torch.Tensor, beta: torch.Tensor, y: torch.Tensor, kl_weight: float) -> torch.Tensor:
    """Beta-EDL loss (adapted from Sensoy et al. 2018). alpha, beta, y: (B, 5). Returns a scalar.

    Expected BCE under Beta(alpha, beta): y=1 -> digamma(a+b) - digamma(a); y=0 -> digamma(a+b) - digamma(b).
    KL toward Beta(1, 1) on misleading evidence only: y=1 -> Beta(1, beta); y=0 -> Beta(alpha, 1).
    Summed over classes, averaged over the batch.
    """
    s = alpha + beta
    nll = y * (torch.digamma(s) - torch.digamma(alpha)) + (1 - y) * (torch.digamma(s) - torch.digamma(beta))
    a_t = torch.where(y > 0.5, torch.ones_like(alpha), alpha)
    b_t = torch.where(y > 0.5, beta, torch.ones_like(beta))
    kl = _kl_beta_uniform(a_t, b_t)
    return (nll + kl_weight * kl).sum(1).mean()


def kl_anneal(epoch: int, anneal_epochs: int = 10) -> float:
    """KL weight: 0 at epoch 0, rising linearly to 1 at `anneal_epochs` (0-based epochs)."""
    return min(1.0, epoch / anneal_epochs)


def abstain(U: torch.Tensor, q: torch.Tensor, threshold: float) -> torch.Tensor:
    """Record-level abstention: U > threshold or sum_j q_j < 3. U: (B,), q: (B, 12) -> bool (B,)."""
    return (U > threshold) | (q.sum(1) < 3)


# ---------------------------------------------------------------- LeadGuard

class LeadGuard(nn.Module):
    """Lead encoder -> MALA (2 layers, 4 heads, d=128) -> q-weighted pooling -> head.

    gating: "soft" uses q directly; "hard" replaces q by (q >= 0.5) as 0/1 (same attention and pooling code);
    "none" ignores q (all ones: unmasked lead attention and mean pooling), used by the resnet_attn baseline.
    forward(x: (B, 12, T), q: (B, 12)) -> dict (see module docstring).
    """

    def __init__(self, gating: str = "soft", head: str = "sigmoid", layers: int = 2, heads: int = 4):
        super().__init__()
        assert gating in ("soft", "hard", "none") and head in ("sigmoid", "edl")
        self.gating = gating
        self.encoder = LeadEncoder()
        self.layers = nn.ModuleList([MALALayer(D_MODEL, heads) for _ in range(layers)])
        self.head = SigmoidHead() if head == "sigmoid" else BetaEDLHead()

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> dict:
        q_soft = torch.ones_like(x[:, :, 0]) if self.gating == "none" else q.to(x.dtype)
        q = (q_soft >= 0.5).to(x.dtype) if self.gating == "hard" else q_soft
        key_bias, pool_w = gate_weights(q, guard_q=q_soft)
        h = self.encoder(x)
        attn = []
        for layer in self.layers:
            h, a = layer(h, key_bias)
            attn.append(a)
        z = (pool_w[..., None] * h).sum(1) / (pool_w.sum(1, keepdim=True) + 1e-6)
        out = self.head(z)
        out["attn"] = attn
        return out


# ---------------------------------------------------------------- baselines
# Baselines ignore q and see the corrupted input as is (dropped leads are zero-filled).

class MLP(nn.Module):
    """M2: flatten -> 12000-512-128-5 with dropout."""

    def __init__(self, in_dim: int = N_LEADS * 1000, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout))
        self.head = SigmoidHead()

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None) -> dict:
        return self.head(self.net(x))


class ResNet1d(nn.Module):
    """M3: ResNet-1D over all 12 leads as channels; stem + 4 residual stages + global average pool."""

    def __init__(self, channels: tuple[int, ...] = (64, 64, 128, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(N_LEADS, channels[0], 7, 2, 3, bias=False),
                                  nn.BatchNorm1d(channels[0]), nn.GELU())
        stages, c_in = [], channels[0]
        for c in channels:
            stages.append(ResBlock1d(c_in, c, stride=2))
            c_in = c
        self.stages = nn.Sequential(*stages)
        self.proj = nn.Linear(c_in, D_MODEL)
        self.head = SigmoidHead()

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None) -> dict:
        return self.head(self.proj(self.stages(self.stem(x)).mean(-1)))


class PatchTransformer(nn.Module):
    """M4: 1D Transformer over time patches (patch 25 -> 40 tokens of 12*25), unmasked, mean pooled."""

    def __init__(self, patch: int = 25, t: int = 1000, d: int = D_MODEL, layers: int = 2, heads: int = 4):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(N_LEADS * patch, d)
        self.pos = nn.Parameter(torch.randn(t // patch, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=256, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = SigmoidHead()

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None) -> dict:
        b, c, t = x.shape
        tokens = x.view(b, c, t // self.patch, self.patch).permute(0, 2, 1, 3).reshape(b, t // self.patch, -1)
        h = self.encoder(self.embed(tokens) + self.pos)
        return self.head(self.norm(h).mean(1))


MODELS = {
    "mlp": MLP,
    "resnet": ResNet1d,
    "transformer": PatchTransformer,
    "resnet_attn": lambda: LeadGuard(gating="none", head="sigmoid"),
    "lg_hard": lambda: LeadGuard(gating="hard", head="sigmoid"),
    "lg_soft": lambda: LeadGuard(gating="soft", head="sigmoid"),
    "lg_edl": lambda: LeadGuard(gating="soft", head="edl"),
}


def build_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name}; choose from {list(MODELS)}")
    return MODELS[name]()


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
