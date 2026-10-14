"""Lead-failure simulation (SPEC section 2). Vectorized torch on x: (B, 12, T).

Every corruption returns (x_corrupt, severity) with severity: (B, 12) in [0, 1]
(0 = untouched, exactly; 1 = dead). Derived: true_mask = severity < 0.5, q_target = 1 - severity.

`leads` arguments accept: None (all leads), a list of lead indices (same for every record),
or a bool tensor of shape (12,) or (B, 12) for per-record selections.
`gen` is an optional torch.Generator; random numbers are drawn on gen's device, then moved to x's device.
"""
from __future__ import annotations

import zlib

import torch

from utils import FS

LEAD_GROUPS = {
    "precordial": [6, 7, 8, 9, 10, 11],  # V1-V6
    "limb": [0, 1, 2, 3, 4, 5],  # I, II, III, aVR, aVL, aVF
    "v1_v3": [6, 7, 8],
}
EMG_BAND_HZ = (20.0, 45.0)  # never 50/60 Hz: not representable at 100 Hz sampling
CLIP_SEVERITY_SCALE = 0.2  # severity = clipped fraction / 0.2, capped at 1


# ------------------------------------------------------------------ helpers

def _rand(shape, gen, device) -> torch.Tensor:
    if gen is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=gen, device=gen.device).to(device)


def _randn(shape, gen, device) -> torch.Tensor:
    if gen is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=gen, device=gen.device).to(device)


def _uniform(shape, lo, hi, gen, device) -> torch.Tensor:
    return lo + (hi - lo) * _rand(shape, gen, device)


def _randint(lo, hi, shape, gen, device) -> torch.Tensor:
    """Integers in [lo, hi)."""
    return (lo + (_rand(shape, gen, device) * (hi - lo)).floor()).long().clamp(max=hi - 1)


def _lead_mask(leads, x: torch.Tensor) -> torch.Tensor:
    """Normalize a `leads` argument to a bool mask (B, 12)."""
    b, c = x.shape[:2]
    if leads is None:
        return torch.ones(b, c, dtype=torch.bool, device=x.device)
    if isinstance(leads, torch.Tensor) and leads.dtype == torch.bool:
        return leads.to(x.device).expand(b, c)
    m = torch.zeros(b, c, dtype=torch.bool, device=x.device)
    m[:, list(leads)] = True
    return m


def random_lead_mask(b: int, k, gen=None, device="cpu") -> torch.Tensor:
    """Pick k distinct random leads per record. k: int or (B,) tensor. Returns bool (B, 12)."""
    k = torch.as_tensor(k, device=device).expand(b)
    rank = _rand((b, 12), gen, device).argsort(1).argsort(1)
    return rank < k[:, None]


def _drop(x: torch.Tensor, mask: torch.Tensor):
    """Zero the selected leads (severity 1)."""
    return x.masked_fill(mask[..., None], 0.0), mask.to(x.dtype)


# -------------------------------------------------------------- corruptions

def drop_random(x: torch.Tensor, k, gen=None):
    """Zero k random leads per record. x: (B, 12, T); k: int or (B,) tensor."""
    return _drop(x, random_lead_mask(x.shape[0], k, gen, x.device))


def drop_group(x: torch.Tensor, group: str):
    """Zero a lead group: 'precordial', 'limb' or 'v1_v3'."""
    return _drop(x, _lead_mask(LEAD_GROUPS[group], x))


def flatline(x: torch.Tensor, leads=None):
    """Stuck amplifier: the selected leads hold their first sample value (severity 1)."""
    m = _lead_mask(leads, x)
    out = torch.where(m[..., None], x[..., :1].expand_as(x), x)
    return out, m.to(x.dtype)


def add_emg_noise(x: torch.Tensor, leads=None, snr_db=10.0, gen=None):
    """Add broadband noise band-limited to 20-45 Hz at the given SNR (relative to each lead's power).

    snr_db: float, (B,) or (B, 12) tensor. Severity = 1 - clip(snr_db / 20, 0, 1) on affected leads.
    """
    b, c, t = x.shape
    m = _lead_mask(leads, x)
    snr = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device)
    snr = snr[:, None] if snr.dim() == 1 else snr
    snr = snr.expand(b, c)

    spec = torch.fft.rfft(_randn((b, c, t), gen, x.device), dim=-1)
    f = torch.fft.rfftfreq(t, d=1.0 / FS).to(x.device)
    spec = spec * ((f >= EMG_BAND_HZ[0]) & (f <= EMG_BAND_HZ[1]))
    noise = torch.fft.irfft(spec, n=t, dim=-1).to(x.dtype)
    noise = noise / noise.std(-1, keepdim=True).clamp_min(1e-8)
    p_sig = x.var(-1)
    noise = noise * torch.sqrt(p_sig / 10 ** (snr / 10))[..., None]

    out = torch.where(m[..., None], x + noise, x)
    sev = m.to(x.dtype) * (1 - (snr / 20).clamp(0, 1))
    return out, sev


def add_baseline_wander(x: torch.Tensor, leads=None, amp_ratio=None, freq_hz=None, gen=None):
    """Add a 0.1-0.5 Hz sinusoid with amplitude 0.5-3x the lead std (sampled per lead unless given).

    Severity = clip(amp_ratio / 2, 0, 1) on affected leads.
    """
    b, c, t = x.shape
    m = _lead_mask(leads, x)
    ratio = _uniform((b, c), 0.5, 3.0, gen, x.device) if amp_ratio is None else \
        torch.as_tensor(amp_ratio, dtype=x.dtype, device=x.device).expand(b, c)
    freq = _uniform((b, c), 0.1, 0.5, gen, x.device) if freq_hz is None else \
        torch.as_tensor(freq_hz, dtype=x.dtype, device=x.device).expand(b, c)
    phase = _uniform((b, c), 0.0, 2 * torch.pi, gen, x.device)
    time = torch.arange(t, device=x.device, dtype=x.dtype) / FS
    wave = torch.sin(2 * torch.pi * freq[..., None] * time + phase[..., None])
    wave = wave * (ratio * x.std(-1))[..., None]

    out = torch.where(m[..., None], x + wave, x)
    sev = m.to(x.dtype) * (ratio / 2).clamp(0, 1)
    return out, sev


def clip_amplitude(x: torch.Tensor, leads=None, percentile=None, gen=None):
    """Saturate each selected lead at the given percentile of |x| (sampled in [70, 95] unless given).

    Severity = fraction of samples clipped / 0.2, capped at 1, on affected leads.
    """
    b, c, t = x.shape
    m = _lead_mask(leads, x)
    pct = _uniform((b, c), 70.0, 95.0, gen, x.device) if percentile is None else \
        torch.as_tensor(percentile, dtype=x.dtype, device=x.device).expand(b, c)
    mag = x.abs()
    idx = (pct / 100 * (t - 1)).round().long().clamp(0, t - 1)
    thr = mag.sort(-1).values.gather(-1, idx[..., None])  # (B, 12, 1)
    clipped = torch.maximum(torch.minimum(x, thr), -thr)
    frac = (mag > thr).to(x.dtype).mean(-1)

    out = torch.where(m[..., None], clipped, x)
    sev = m.to(x.dtype) * (frac / CLIP_SEVERITY_SCALE).clamp(0, 1)
    return out, sev


# ------------------------------------------------------------------ training

def training_corruption(x: torch.Tensor, gen=None):
    """Random corruption mix, sampled independently per record.

    Per record: 30% artifact (EMG 0-20 dB / wander / clipping) on 1-3 leads; 50% random drop of
    0-6 leads (a quarter of dropped leads flatline instead of zeroing); 15% group drop. These can
    combine; the remaining records stay clean. At least 2 good leads (severity < 0.5) are always kept.
    x: (B, 12, T) -> (x_corrupt (B, 12, T), severity (B, 12))
    """
    b, dev = x.shape[0], x.device
    out, sev = x.clone(), torch.zeros(b, 12, dtype=x.dtype, device=dev)

    # Artifacts first, so a later drop overrides them.
    has_art = _rand(b, gen, dev) < 0.30
    art_mask = random_lead_mask(b, _randint(1, 4, b, gen, dev), gen, dev) & has_art[:, None]
    kind = _randint(0, 3, b, gen, dev)
    snr = _uniform((b, 12), 0.0, 20.0, gen, dev)
    for k, fn in enumerate([lambda z, m: add_emg_noise(z, m, snr, gen),
                            lambda z, m: add_baseline_wander(z, m, gen=gen),
                            lambda z, m: clip_amplitude(z, m, gen=gen)]):
        m = art_mask & (kind == k)[:, None]
        if m.any():
            out, s = fn(out, m)
            sev = torch.maximum(sev, s)

    has_drop = _rand(b, gen, dev) < 0.50
    n_drop = _randint(0, 7, b, gen, dev) * has_drop
    drop_mask = random_lead_mask(b, n_drop, gen, dev)
    flat = drop_mask & (_rand((b, 12), gen, dev) < 0.25)
    out, s1 = _drop(out, drop_mask & ~flat)
    out, s2 = flatline(out, flat)
    sev = torch.maximum(sev, torch.maximum(s1, s2))

    has_group = _rand(b, gen, dev) < 0.15
    table = torch.stack([_lead_mask(v, x[:1])[0] for v in LEAD_GROUPS.values()])  # (3, 12)
    group_mask = table[_randint(0, len(LEAD_GROUPS), b, gen, dev)] & has_group[:, None]
    out, s = _drop(out, group_mask)
    sev = torch.maximum(sev, s)

    # Guarantee >= 2 good leads: restore the least-corrupted leads of records that fell below.
    need = (sev < 0.5).sum(1) < 2
    if need.any():
        order = sev.argsort(1, stable=True)[:, :2]
        restore = torch.zeros(b, 12, dtype=torch.bool, device=dev)
        restore.scatter_(1, order, True)
        restore &= need[:, None]
        out = torch.where(restore[..., None], x, out)
        sev = sev.masked_fill(restore, 0.0)
    return out, sev


# ------------------------------------------------------------------ test set

EMG_TEST_SNR_DB = 10.0

TEST_CONDITIONS = (
    [(f"random_{k}", d) for k in (0, 2, 4, 6, 8, 10) for d in range(3)]
    + [(f"group_{g}", 0) for g in LEAD_GROUPS]
    + [(f"{a}_{n}", 0) for a in ("emg", "wander", "clip") for n in (2, 4, 6)]
)


def fixed_test_corruption(x: torch.Tensor, condition: str, seed: int = 0):
    """Deterministic corruption for evaluation: same (x, condition, seed) -> identical output.

    Apply to the whole test set at once (not per batch), so every model sees identical data.
    condition: 'clean', 'random_{k}', 'group_{precordial|limb|v1_v3}', '{emg|wander|clip}_{n_leads}'.
    """
    gen = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(condition.encode()))
    b = x.shape[0]
    kind, _, arg = condition.partition("_")
    if condition == "clean":
        return x.clone(), torch.zeros(b, 12, dtype=x.dtype, device=x.device)
    if kind == "random":
        return drop_random(x, int(arg), gen)
    if kind == "group":
        return drop_group(x, arg)
    leads = random_lead_mask(b, int(arg), gen, x.device)
    if kind == "emg":
        return add_emg_noise(x, leads, EMG_TEST_SNR_DB, gen)
    if kind == "wander":
        return add_baseline_wander(x, leads, gen=gen)
    if kind == "clip":
        return clip_amplitude(x, leads, gen=gen)
    raise ValueError(f"unknown condition: {condition}")
