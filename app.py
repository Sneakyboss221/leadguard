"""LeadGuard demo (SPEC section 8). Streamlit, CPU, offline, single file.

streamlit run app.py [-- --ckpt-dir checkpoints --samples-dir demo_samples]

Loads the frozen SQA, resnet (clean-trained and corruption-trained, seed 0), lg_edl (seed 0), the demo samples
and demo_config.json (validation-fitted temperatures, thresholds and the U threshold) from make_demo_samples.py.
Research prototype, not for clinical use.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

import corrupt
from evaluate import apply_temperature
from models import build_model
from quality import load_sqa
from train import ckpt_paths
from utils import CLASSES, FS, LEADS, REPO_DIR

# ------------------------------------------------------------------ settings

EMG_SNR_DB = 10.0
OFFSET_MV = 3.0  # vertical distance between stacked leads
CLASS_NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
               "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
BASELINES = {"resnet_c1": "ResNet · corruption-trained", "resnet_c0": "ResNet · clean-trained"}

# Palette: the paper figures' hues for the two models; ink / surfaces neutral; status colors only for state.
INK, INK_2, MUTED, LINE, SURFACE = "#16161a", "#52514e", "#8a8984", "#e6e4df", "#fbfbfa"
C_LG, C_BASE = "#2a78d6", "#eb6834"
C_TRACE, C_NOISY, C_OFF = "#1c1c1f", "#c98500", "#b9b7b1"
C_CRIT, C_CRIT_BG = "#c22f45", "#fcedef"
PAPER_BG, PAPER_MINOR, PAPER_MAJOR = "#fffcfb", "#fbe8e8", "#f2c9c9"
FONT = "Inter, ui-sans-serif, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, 'Liberation Mono', monospace"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--samples-dir", default=str(REPO_DIR / "demo_samples"))
    return p.parse_known_args(sys.argv[1:])[0]


# --------------------------------------------------------------------- loading

@st.cache_resource(show_spinner=False)
def load_assets(ckpt_dir: str, samples_dir: str):
    cfg = json.loads((Path(samples_dir) / "demo_config.json").read_text())
    models = {}
    for key, entry in cfg["runs"].items():
        model_name, seed_part, c_part = entry["run"].rsplit("_", 2)
        net = build_model(model_name)
        best, _ = ckpt_paths(ckpt_dir, model_name, int(seed_part[1:]), int(c_part[1:]))
        net.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
        models[key] = net.eval()
    sqa = load_sqa(Path(ckpt_dir) / "sqa.pt", "cpu")
    samples = []
    for f in sorted(Path(samples_dir).glob("sample_*.npz")):
        d = np.load(f)
        samples.append({k: d[k] for k in d.files} | {"name": f.stem})
    return cfg, models, sqa, samples


# ------------------------------------------------------------------- state

def _init_state() -> None:
    ss = st.session_state
    for j in range(12):
        ss.setdefault(f"lead_{j}", True)
    ss.setdefault("noisy", [])
    ss.setdefault("q_auto", False)
    ss.setdefault("draw", 0)


def _set_leads(off: set[int]) -> None:
    for j in range(12):
        st.session_state[f"lead_{j}"] = j not in off
    st.session_state.noisy = []


def preset_drop_precordial():
    _set_leads(set(range(6, 12)))


def preset_drop_limb():
    _set_leads(set(range(6)))


def preset_random6():
    st.session_state.draw += 1
    _set_leads(set(np.random.default_rng(st.session_state.draw).choice(12, 6, replace=False).tolist()))


def preset_emg3():
    st.session_state.draw += 1
    on = [j for j in range(12) if st.session_state[f"lead_{j}"]]
    rng = np.random.default_rng(1000 + st.session_state.draw)
    st.session_state.noisy = sorted(rng.choice(on, min(3, len(on)), replace=False).tolist())


def preset_reset():
    _set_leads(set())
    st.session_state.q_auto = False


def auto_detect():
    st.session_state.q_auto = True


def use_switches():
    st.session_state.q_auto = False


# --------------------------------------------------------------- inference

def simulate(sample: dict, on: np.ndarray, noisy: list[int], seed: int):
    """Apply the current lead state to both the z-scored and the raw (mV) copy.

    EMG noise is generated on the z-scored signal and mapped back to mV with each lead's own mean and std,
    so both copies carry the same corruption. Disconnected leads are zero in both.
    """
    z = torch.from_numpy(sample["x"].astype(np.float32))[None]
    raw = sample["x_raw"].astype(np.float32)
    mean, std = raw.mean(-1, keepdims=True), raw.std(-1, keepdims=True)
    noisy_on = [j for j in noisy if on[j]]
    if noisy_on:
        m = torch.zeros(1, 12, dtype=torch.bool)
        m[0, noisy_on] = True
        z, _ = corrupt.add_emg_noise(z, m, EMG_SNR_DB, torch.Generator().manual_seed(seed))
        raw = np.where(m[0, :, None].numpy(), mean + z[0].numpy() * std, raw)
    off = torch.from_numpy(~on)
    z = z.masked_fill(off[None, :, None], 0.0)
    raw = np.where(on[:, None], raw, 0.0)
    return z, raw, noisy_on


@torch.no_grad()
def infer(cfg, models, sqa, z: torch.Tensor, on: np.ndarray, q_auto: bool, baseline: str) -> dict:
    q = sqa(z)[0] if q_auto else torch.from_numpy(on.astype(np.float32))
    out = models["lg_edl"](z, q[None])
    base = models[baseline](z, torch.ones(1, 12))
    b_cfg, lg_cfg = cfg["runs"][baseline], cfg["runs"]["lg_edl"]
    U, q_sum = float(out["U"][0]), float(q.sum())
    reasons = []
    if U > lg_cfg["u_threshold"]:
        reasons.append(f"uncertainty U = {U:.3f} is above the validation threshold {lg_cfg['u_threshold']:.3f}")
    if q_sum < cfg["abstain_q_sum"]:
        reasons.append(f"only Σq = {q_sum:.2f} of usable signal remains (needs ≥ {cfg['abstain_q_sum']:.0f})")
    return {
        "q": q.numpy(), "q_sum": q_sum, "U": U, "u_thr": lg_cfg["u_threshold"], "abstain": bool(reasons),
        "reasons": reasons, "attn": out["attn"][-1][0].mean(0).numpy(),
        "p_lg": out["prob"][0].numpy(), "thr_lg": np.array(lg_cfg["thresholds"]),
        "p_base": apply_temperature(base["prob"].numpy(), b_cfg["temperature"])[0],
        "thr_base": np.array(b_cfg["thresholds"]),
    }


# ------------------------------------------------------------------ charts

def _theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_2, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
                            titleFontWeight=500, domainColor=LINE, tickColor=LINE, gridColor="#efeeea")
            .configure_legend(labelColor=INK_2, titleColor=MUTED, labelFontSize=11, orient="top",
                              symbolStrokeWidth=0))


def ecg_chart(raw: np.ndarray, on: np.ndarray, noisy: list[int]) -> alt.Chart:
    t = np.arange(raw.shape[1]) / FS
    rows = []
    for j in range(12):
        state = "disconnected" if not on[j] else ("noisy" if j in noisy else "ok")
        rows.append(pd.DataFrame({"t": t, "mv": raw[j] - j * OFFSET_MV, "lead": LEADS[j], "state": state}))
    df = pd.concat(rows)
    y_dom = [-11 * OFFSET_MV - OFFSET_MV * 0.75, OFFSET_MV * 0.75]
    x = alt.X("t:Q", scale=alt.Scale(domain=[-0.55, 10.45], nice=False),
              axis=alt.Axis(values=list(range(11)), title="time (s)", grid=False, labelColor=INK_2))
    y = alt.Y("mv:Q", scale=alt.Scale(domain=y_dom, nice=False), axis=None)
    minor = pd.DataFrame({"t": np.arange(0, 10.01, 0.04)})
    major = pd.DataFrame({"t": np.arange(0, 10.01, 0.2)})
    h_minor = pd.DataFrame({"mv": np.arange(math.ceil(y_dom[0] / 0.1) * 0.1, y_dom[1], 0.1)})
    h_major = pd.DataFrame({"mv": np.arange(math.ceil(y_dom[0] / 0.5) * 0.5, y_dom[1], 0.5)})
    grid = [
        alt.Chart(minor).mark_rule(color=PAPER_MINOR, strokeWidth=0.5).encode(x=x),
        alt.Chart(h_minor).mark_rule(color=PAPER_MINOR, strokeWidth=0.5).encode(y=y),
        alt.Chart(major).mark_rule(color=PAPER_MAJOR, strokeWidth=0.8).encode(x=x),
        alt.Chart(h_major).mark_rule(color=PAPER_MAJOR, strokeWidth=0.8).encode(y=y),
    ]
    traces = alt.Chart(df).mark_line(strokeWidth=1.15, strokeJoin="round").encode(
        x=x, y=y, detail="lead:N",
        color=alt.Color("state:N", scale=alt.Scale(domain=["ok", "noisy", "disconnected"],
                                                   range=[C_TRACE, C_NOISY, C_OFF]), legend=None))
    labels = pd.DataFrame({"t": -0.12, "mv": [-j * OFFSET_MV + 0.9 for j in range(12)], "lead": LEADS,
                           "state": ["disconnected" if not on[j] else ("noisy" if j in noisy else "ok")
                                     for j in range(12)]})
    lab = alt.Chart(labels).mark_text(align="right", fontWeight=600, fontSize=12, font=FONT).encode(
        x=x, y=y, text="lead:N",
        color=alt.Color("state:N", scale=alt.Scale(domain=["ok", "noisy", "disconnected"],
                                                   range=[INK, C_NOISY, C_OFF]), legend=None))
    bar = pd.DataFrame({"t": [10.2], "y0": [0.25 - 1.0], "y1": [0.25]})
    scale = alt.Chart(bar).mark_rule(color=INK, strokeWidth=2).encode(
        x=x, y=alt.Y("y0:Q", scale=alt.Scale(domain=y_dom, nice=False), axis=None), y2="y1:Q")
    scale_txt = alt.Chart(pd.DataFrame({"t": [10.2], "mv": [0.55], "txt": ["1 mV"]})).mark_text(
        fontSize=10, color=INK_2, font=FONT).encode(x=x, y=y, text="txt:N")
    chart = alt.layer(*grid, traces, lab, scale, scale_txt).properties(
        height=640, background=PAPER_BG, padding={"left": 6, "right": 6, "top": 8, "bottom": 4})
    return _theme(chart)


def quality_chart(q: np.ndarray) -> alt.Chart:
    df = pd.DataFrame({"lead": LEADS, "q": q,
                       "band": np.where(q < 0.02, "masked", np.where(q < 0.5, "low", "usable")),
                       "label": [f"{v:.2f}" for v in q]})
    y = alt.Y("lead:N", sort=LEADS, title=None, axis=alt.Axis(labelFontWeight=600, labelColor=INK, ticks=False,
                                                            domain=False, labelPadding=8))
    color = alt.Color("band:N", scale=alt.Scale(domain=["usable", "low", "masked"], range=[C_LG, C_NOISY, C_OFF]),
                      legend=alt.Legend(title=None, symbolType="square", symbolSize=80))
    track = alt.Chart(df).mark_bar(color="#f1f0ec", cornerRadius=3, height=12).encode(y=y, x=alt.datum(1.0))
    bars = alt.Chart(df).mark_bar(cornerRadius=3, height=12).encode(
        y=y, x=alt.X("q:Q", scale=alt.Scale(domain=[0, 1]), title="reliability q",
                     axis=alt.Axis(values=[0, 0.5, 1], grid=False)), color=color,
        tooltip=[alt.Tooltip("lead:N"), alt.Tooltip("q:Q", format=".3f"), alt.Tooltip("band:N", title="status")])
    txt = alt.Chart(df).mark_text(align="left", dx=4, fontSize=10, font=MONO, color=INK_2).encode(
        y=y, x=alt.datum(1.0), text="label:N")
    half = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(x="x:Q")
    return _theme(alt.layer(track, bars, half, txt).properties(height=640, padding={"right": 34}))


def prob_chart(res: dict, baseline_label: str, y_true: np.ndarray) -> alt.Chart:
    rows = []
    for k, c in enumerate(CLASSES):
        tag = f"{c}  ●" if y_true[k] > 0.5 else c
        rows.append({"cls": tag, "model": baseline_label, "p": float(res["p_base"][k]), "faded": False,
                     "pred": res["p_base"][k] >= res["thr_base"][k], "thr": float(res["thr_base"][k])})
        rows.append({"cls": tag, "model": "LeadGuard-EDL", "p": float(res["p_lg"][k]), "faded": res["abstain"],
                     "pred": res["p_lg"][k] >= res["thr_lg"][k], "thr": float(res["thr_lg"][k])})
    df = pd.DataFrame(rows)
    order = [r["cls"] for r in rows[::2]]
    models = [baseline_label, rows[1]["model"]]
    y = alt.Y("cls:N", sort=order, title=None,
              axis=alt.Axis(labelFontWeight=600, labelColor=INK, ticks=False, domain=False, labelPadding=8))
    y_off = alt.YOffset("model:N", sort=models)
    color = alt.Color("model:N", scale=alt.Scale(domain=models, range=[C_BASE, C_LG]), legend=alt.Legend(title=None))
    bars = alt.Chart(df).mark_bar(cornerRadius=3, height=11).encode(
        y=y, yOffset=y_off, color=color,
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), title="probability", axis=alt.Axis(values=[0, 0.5, 1])),
        opacity=alt.condition(alt.datum.faded, alt.value(0.35), alt.value(1.0)),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("cls:N", title="class"), alt.Tooltip("p:Q", format=".3f"),
                 alt.Tooltip("thr:Q", title="decision threshold", format=".2f"), alt.Tooltip("pred:N", title="positive")])
    ticks = alt.Chart(df).mark_tick(color=INK, thickness=1.5, size=11).encode(y=y, yOffset=y_off, x="thr:Q")
    return _theme(alt.layer(bars, ticks).properties(height=300))


def attention_chart(attn: np.ndarray) -> alt.Chart:
    df = pd.DataFrame([{"query": LEADS[i], "key": LEADS[j], "w": float(attn[i, j])}
                       for i in range(12) for j in range(12)])
    ax = dict(labelFontSize=10, ticks=False, domain=False, labelColor=INK_2, labelOverlap=False)
    return _theme(alt.Chart(df).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("key:N", sort=LEADS, title="attends to", axis=alt.Axis(labelAngle=-90, **ax)),
        y=alt.Y("query:N", sort=LEADS, title=None, axis=alt.Axis(**ax)),
        color=alt.Color("w:Q", scale=alt.Scale(scheme="blues", domain=[0, max(0.25, float(attn.max()))]),
                        legend=alt.Legend(title="weight", orient="right", gradientLength=150)),
        tooltip=[alt.Tooltip("query:N"), alt.Tooltip("key:N"), alt.Tooltip("w:Q", format=".3f")],
    ).properties(height=300))


def gauge_chart(U: float, thr: float) -> alt.Chart:
    a = lambda v: -math.pi / 2 + math.pi * min(max(v, 0.0), 1.0)
    over = U > thr
    arcs = pd.DataFrame([
        {"s": a(0), "e": a(1), "part": "track"},
        {"s": a(0), "e": a(U), "part": "value"},
        {"s": a(thr) - 0.012, "e": a(thr) + 0.012, "part": "threshold"},
    ])
    ring = alt.Chart(arcs).mark_arc(innerRadius=78, outerRadius=100, cornerRadius=2).encode(
        theta=alt.Theta("s:Q", scale=None), theta2="e:Q",
        color=alt.Color("part:N", scale=alt.Scale(domain=["track", "value", "threshold"],
                                                  range=["#efeee9", C_CRIT if over else C_LG, INK]), legend=None))
    val = alt.Chart(pd.DataFrame({"t": [f"{U:.3f}"]})).mark_text(
        fontSize=30, fontWeight=600, dy=-22, color=INK, font=FONT).encode(text="t:N")
    sub = alt.Chart(pd.DataFrame({"t": [f"threshold {thr:.3f}"]})).mark_text(
        fontSize=11, dy=4, color=MUTED, font=FONT).encode(text="t:N")
    return _theme(alt.layer(ring, val, sub).properties(height=300))


# --------------------------------------------------------------------- page

CSS = f"""
<style>
html, body, [class*="css"], .stApp {{ font-family: {FONT}; color: {INK}; }}
.stApp {{ background: {SURFACE}; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1480px; }}
[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.lg-eyebrow {{ font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED};
  font-weight: 600; margin: 18px 0 6px; }}
.lg-head {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  padding-bottom: 14px; border-bottom: 1px solid {LINE}; margin-bottom: 16px; }}
.lg-title {{ font-size: 22px; font-weight: 650; letter-spacing: -.01em; }}
.lg-title span {{ color: {MUTED}; font-weight: 450; margin-left: 10px; font-size: 15px; letter-spacing: 0; }}
.lg-tags {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.lg-tag {{ font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid {LINE};
  color: {INK_2}; background: #fff; white-space: nowrap; }}
.lg-tag.warn {{ border-color: #ead9b3; background: #fdf8ec; color: #7a5a12; }}
.lg-cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
.lg-card {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 14px; }}
.lg-card .k {{ font-size: 12px; color: {MUTED}; }}
.lg-card .v {{ font-size: 22px; font-weight: 620; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.lg-card .s {{ font-size: 12px; color: {INK_2}; margin-top: 2px; }}
.lg-card.crit {{ border-color: #efc2c9; background: {C_CRIT_BG}; }}
.lg-card.crit .v {{ color: {C_CRIT}; }}
.lg-banner {{ display: flex; gap: 14px; align-items: flex-start; border: 1px solid #efc2c9;
  border-left: 4px solid {C_CRIT}; background: {C_CRIT_BG}; border-radius: 10px; padding: 12px 16px;
  margin-bottom: 14px; }}
.lg-banner .badge {{ font-weight: 700; letter-spacing: .1em; color: #fff; background: {C_CRIT};
  border-radius: 6px; padding: 3px 9px; font-size: 13px; }}
.lg-banner .txt {{ font-size: 14px; color: {INK}; line-height: 1.45; }}
.lg-banner .txt small {{ color: {INK_2}; font-size: 13px; }}
.lg-panel-title {{ font-size: 13px; font-weight: 620; color: {INK}; margin: 2px 0 2px; }}
.lg-panel-sub {{ font-size: 12px; color: {MUTED}; margin-bottom: 6px; }}
.lg-legend {{ display: flex; gap: 14px; font-size: 12px; color: {INK_2}; margin: 2px 0 6px; flex-wrap: wrap; }}
.lg-legend i {{ display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin-right: 6px; }}
.lg-foot {{ font-size: 12px; color: {MUTED}; border-top: 1px solid {LINE}; padding-top: 10px; margin-top: 18px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-color: {LINE} !important;
  border-radius: 12px !important; }}
.stButton > button {{ border-radius: 8px; border: 1px solid {LINE}; font-size: 13px; padding: 4px 10px;
  width: 100%; color: {INK}; background: #fff; }}
.stButton > button:hover {{ border-color: {C_LG}; color: {C_LG}; }}
.stButton > button[kind="primary"] {{ background: {INK}; border-color: {INK}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #000; color: #fff; }}
[data-testid="stCheckbox"] label p {{ font-size: 13px; font-weight: 550; }}
</style>
"""


def _card(k: str, v: str, s: str, kind: str = "") -> str:
    size = ' style="font-size:17px;line-height:1.35"' if len(v) > 14 and "<" not in v else ""
    return (f'<div class="lg-card {kind}"><div class="k">{k}</div><div class="v"{size}>{v}</div>'
            f'<div class="s">{s}</div></div>')


def _panel(title: str, sub: str) -> None:
    st.markdown(f'<div class="lg-panel-title">{title}</div><div class="lg-panel-sub">{sub}</div>',
                unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="LeadGuard · ECG lead-failure demo", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    a = _args()
    try:
        cfg, models, sqa, samples = load_assets(a.ckpt_dir, a.samples_dir)
    except FileNotFoundError as e:
        st.error(f"Missing demo assets ({e.filename}). Run make_demo_samples.py first.")
        st.stop()
    if not samples:
        st.error("No demo samples found. Run make_demo_samples.py first.")
        st.stop()
    _init_state()
    ss = st.session_state

    # ---------------------------------------------------------- sidebar
    with st.sidebar:
        st.markdown('<div class="lg-title" style="font-size:18px">LeadGuard</div>', unsafe_allow_html=True)
        st.markdown('<div class="lg-eyebrow">Record</div>', unsafe_allow_html=True)
        labels = [f"{s['name'][-2:]} · {str(s['target_class'])} · "
                  + "+".join(c for c, v in zip(CLASSES, s["y"]) if v > 0.5) for s in samples]
        idx = st.selectbox("Sample", range(len(samples)), format_func=lambda i: labels[i],
                           label_visibility="collapsed")
        st.markdown('<div class="lg-eyebrow">Baseline</div>', unsafe_allow_html=True)
        baseline = st.selectbox("Baseline", list(BASELINES), format_func=BASELINES.get, label_visibility="collapsed")

        st.markdown('<div class="lg-eyebrow">Scenarios</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("Drop V1–V6", on_click=preset_drop_precordial)
        c2.button("Drop limb", on_click=preset_drop_limb)
        c1.button("Random 6 off", on_click=preset_random6)
        c2.button("EMG on 3 leads", on_click=preset_emg3)
        st.button("Reset", on_click=preset_reset)

        st.markdown('<div class="lg-eyebrow">Leads</div>', unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        for j in range(6):
            l1.checkbox(LEADS[j], key=f"lead_{j}")
            l2.checkbox(LEADS[j + 6], key=f"lead_{j + 6}")

        st.markdown('<div class="lg-eyebrow">Lead quality</div>', unsafe_allow_html=True)
        if ss.q_auto:
            st.button("Use lead switches", on_click=use_switches)
        else:
            st.button("Auto-detect quality", on_click=auto_detect, type="primary")
        st.caption("Source: signal quality assessor (SQA)" if ss.q_auto else
                   "Source: lead switches (on = 1, off = 0)")

    # ----------------------------------------------------------- compute
    sample = samples[idx]
    on = np.array([ss[f"lead_{j}"] for j in range(12)])
    z, raw, noisy = simulate(sample, on, ss.noisy, seed=1000 + ss.draw)
    res = infer(cfg, models, sqa, z, on, ss.q_auto, baseline)
    y_true = sample["y"]
    true_txt = ", ".join(c for c, v in zip(CLASSES, y_true) if v > 0.5)
    pred_lg = ", ".join(c for c, p, t in zip(CLASSES, res["p_lg"], res["thr_lg"]) if p >= t) or "none"
    pred_base = ", ".join(c for c, p, t in zip(CLASSES, res["p_base"], res["thr_base"]) if p >= t) or "none"

    # ------------------------------------------------------------ header
    tags = [f'<span class="lg-tag">{html.escape(str(cfg["data"]))} data'
            + (" · debug models" if cfg["debug"] else "") + "</span>",
            '<span class="lg-tag warn">Research prototype · not for clinical use</span>']
    st.markdown(f'<div class="lg-head"><div class="lg-title">LeadGuard'
                f'<span>Quality-gated ECG classification under lead failure</span></div>'
                f'<div class="lg-tags">{"".join(tags)}</div></div>', unsafe_allow_html=True)

    if res["abstain"]:
        st.markdown(f'<div class="lg-banner"><span class="badge">ABSTAIN</span><div class="txt">'
                    f'LeadGuard-EDL declines to classify this record.<br><small>'
                    f'{html.escape("; ".join(res["reasons"]))}.</small></div></div>', unsafe_allow_html=True)

    n_on = int(on.sum())
    cards = [
        _card("Leads connected", f"{n_on}<span style='color:{MUTED};font-size:15px'> / 12</span>",
              f"{len(noisy)} with simulated EMG noise" if noisy else "no simulated noise"),
        _card("Usable signal Σq", f"{res['q_sum']:.2f}", f"abstain below {cfg['abstain_q_sum']:.0f} · "
              + ("SQA estimate" if ss.q_auto else "from lead switches"), "crit" if res["q_sum"] < cfg["abstain_q_sum"] else ""),
        _card("Uncertainty U", f"{res['U']:.3f}", f"threshold {res['u_thr']:.3f} (validation)",
              "crit" if res["U"] > res["u_thr"] else ""),
        _card("LeadGuard-EDL", "Abstained" if res["abstain"] else html.escape(pred_lg),
              f"true: {html.escape(true_txt)} · baseline: {html.escape(pred_base)}",
              "crit" if res["abstain"] else ""),
    ]
    st.markdown(f'<div class="lg-cards">{"".join(cards)}</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------ panels
    left, right = st.columns([2.6, 1], gap="medium")
    with left, st.container(border=True):
        _panel("12-lead ECG", f"{sample['name'].replace('_', ' ')} · raw signal in mV · 25 mm/s-style grid "
               "(0.2 s × 0.5 mV major)")
        st.markdown(f'<div class="lg-legend"><span><i style="background:{C_TRACE}"></i>connected</span>'
                    f'<span><i style="background:{C_NOISY}"></i>EMG noise ({EMG_SNR_DB:.0f} dB)</span>'
                    f'<span><i style="background:{C_OFF}"></i>disconnected</span></div>', unsafe_allow_html=True)
        st.altair_chart(ecg_chart(raw, on, noisy), width="stretch")
    with right, st.container(border=True):
        _panel("Lead reliability q", ("SQA estimate" if ss.q_auto else "from lead switches · try Auto-detect")
               + " · low < 0.5 · masked < 0.02")
        st.altair_chart(quality_chart(res["q"]), width="stretch")

    c1, c2, c3 = st.columns([1.35, 1.1, 0.85], gap="medium")
    with c1, st.container(border=True):
        _panel("Class probabilities", "● = true label · tick = validation-tuned decision threshold"
               + (" · faded = abstained" if res["abstain"] else ""))
        st.altair_chart(prob_chart(res, BASELINES[baseline], y_true), width="stretch")
    with c2, st.container(border=True):
        _panel("Lead attention", "LeadGuard-EDL, last layer, mean over heads")
        st.altair_chart(attention_chart(res["attn"]), width="stretch")
    with c3, st.container(border=True):
        _panel("Uncertainty", "U = mean over classes of 2 / (α + β)")
        st.altair_chart(gauge_chart(res["U"], res["u_thr"]), width="stretch")

    st.markdown('<div class="lg-foot">Research prototype, not for clinical use. Lead failures are simulated; '
                'models were trained on retrospective data and are not clinically validated.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":  # `streamlit run app.py` runs the script as __main__
    main()
