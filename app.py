"""LeadGuard demo (SPEC section 8). CPU, offline. The stack changed from Streamlit, see DECISIONS.md.

python app.py [--ckpt-dir checkpoints] [--samples-dir demo_samples] [--results-dir results] [--port 8501]
              [--no-browser]

A small standard-library HTTP server: it serves demo/index.html and a JSON API, so the page needs no internet
and no extra packages. Loads the frozen SQA, resnet (clean-trained and corruption-trained, seed 0), lg_edl
(seed 0), the demo samples and demo_config.json (validation-fitted temperatures, thresholds and the U threshold)
from make_demo_samples.py. Research prototype, not for clinical use.

API
  GET  /api/meta     data setting, class and lead names, thresholds, the list of samples,
                     whole-test-set macro-AUROC vs unplugged leads (from results/random_loss.csv)
  POST /api/predict  {sample, on: [12 bools], noisy: [lead idx], quality: "switches"|"sqa",
                      baseline: "resnet_c1"|"resnet_c0", seed}
                     -> corrupted raw signal (mV), q, U, abstain decision, both models' probabilities and
                        predictions (also with all 12 leads, as a reference), last-layer attention of lg_edl
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

import corrupt
from evaluate import apply_temperature
from models import build_model
from quality import load_sqa
from train import ckpt_paths
from utils import CLASSES, FS, LEADS, REPO_DIR

UI_FILE = REPO_DIR / "demo" / "index.html"
EMG_SNR_DB = 10.0  # same SNR as the EMG test condition
CLASS_NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
               "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
BASELINES = {"resnet_c1": "ResNet, corruption-trained", "resnet_c0": "ResNet, clean-trained"}


def testset_curves(results_dir: Path) -> dict | None:
    """Macro-AUROC vs number of randomly unplugged leads on the whole test fold, from results/random_loss.csv,
    for the three demo models: mean over the 3 draws per k, then mean over seeds. None if the CSV is missing."""
    path = results_dir / "random_loss.csv"
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    df = df.assign(variant=np.where(df.model == "resnet", "resnet_" + df.run.str.rsplit("_", n=1).str[-1], df.model))
    df = df[df.variant.isin(["lg_edl", "resnet_c1", "resnet_c0"])]
    per_seed = df.groupby(["variant", "seed", "n_missing"], as_index=False).macro_auroc.mean()
    curves = {}
    for v, d in per_seed.groupby("variant"):
        g = d.groupby("n_missing").macro_auroc
        curves[v] = {"mean": g.mean().round(4).tolist(), "n_seeds": int(d.seed.nunique())}
        ks = g.mean().index.astype(int).tolist()
    return {"k": ks, "curves": curves}


class Demo:
    """Models, samples and the inference for one request."""

    def __init__(self, ckpt_dir: Path, samples_dir: Path, results_dir: Path):
        self.testset = testset_curves(results_dir)
        self.cfg = json.loads((samples_dir / "demo_config.json").read_text())
        self.models = {}
        for key, entry in self.cfg["runs"].items():
            model_name, seed_part, c_part = entry["run"].rsplit("_", 2)
            net = build_model(model_name)
            best, _ = ckpt_paths(ckpt_dir, model_name, int(seed_part[1:]), int(c_part[1:]))
            net.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
            self.models[key] = net.eval()
        self.sqa = load_sqa(ckpt_dir / "sqa.pt", "cpu")
        self.samples = []
        for f in sorted(samples_dir.glob("sample_*.npz")):
            with np.load(f) as d:
                self.samples.append({k: d[k] for k in d.files} | {"name": f.stem})
        if not self.samples:
            raise FileNotFoundError(f"no sample_*.npz in {samples_dir}; run make_demo_samples.py first")
        self.lock = threading.Lock()  # one inference at a time

    def meta(self) -> dict:
        return {
            "data": self.cfg["data"], "debug": self.cfg["debug"], "classes": CLASSES, "class_names": CLASS_NAMES,
            "leads": LEADS, "fs": FS, "emg_snr_db": EMG_SNR_DB, "baselines": BASELINES,
            "u_threshold": self.cfg["runs"]["lg_edl"]["u_threshold"], "abstain_q_sum": self.cfg["abstain_q_sum"],
            "testset": self.testset,
            "samples": [{"index": i, "name": s["name"], "target": str(s["target_class"]),
                         "labels": [c for c, v in zip(CLASSES, s["y"]) if v > 0.5]}
                        for i, s in enumerate(self.samples)],
        }

    @staticmethod
    def simulate(sample: dict, on: np.ndarray, noisy: list[int], seed: int):
        """Apply the lead state to both the z-scored and the raw (mV) copy.

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
        z = z.masked_fill(torch.from_numpy(~on)[None, :, None], 0.0)
        raw = np.where(on[:, None], raw, 0.0)
        return z, raw, noisy_on

    @torch.no_grad()
    def predict(self, req: dict) -> dict:
        sample = self.samples[int(req["sample"])]
        on = np.array(req["on"], dtype=bool)
        if on.shape != (12,):
            raise ValueError("'on' needs 12 values")
        noisy = sorted({int(j) for j in req.get("noisy", []) if 0 <= int(j) < 12})
        use_sqa = req.get("quality") == "sqa"
        baseline = req.get("baseline") if req.get("baseline") in BASELINES else "resnet_c1"

        z, raw, noisy_on = self.simulate(sample, on, noisy, int(req.get("seed", 0)))
        z_full = torch.from_numpy(sample["x"].astype(np.float32))[None]  # reference: all 12 leads, no noise
        with self.lock:
            q = self.sqa(z)[0] if use_sqa else torch.from_numpy(on.astype(np.float32))
            out = self.models["lg_edl"](z, q[None])
            base = self.models[baseline](z, torch.ones(1, 12))
            q_full = self.sqa(z_full) if use_sqa else torch.ones(1, 12)
            p_lg_full = self.models["lg_edl"](z_full, q_full)["prob"][0].numpy()
            p_base_full = self.models[baseline](z_full, torch.ones(1, 12))["prob"].numpy()

        lg_cfg, b_cfg = self.cfg["runs"]["lg_edl"], self.cfg["runs"][baseline]
        p_base_full = apply_temperature(p_base_full, b_cfg["temperature"])[0]
        U, q_sum = float(out["U"][0]), float(q.sum())
        u_thr, q_min = float(lg_cfg["u_threshold"]), float(self.cfg["abstain_q_sum"])
        reasons = []
        if q_sum < q_min:
            reasons.append({"kind": "signal", "value": q_sum, "limit": q_min})
        if U > u_thr:
            reasons.append({"kind": "uncertainty", "value": U, "limit": u_thr})
        p_lg = out["prob"][0].numpy()
        p_base = apply_temperature(base["prob"].numpy(), b_cfg["temperature"])[0]
        thr_lg, thr_base = np.array(lg_cfg["thresholds"]), np.array(b_cfg["thresholds"])
        return {
            "raw": np.round(raw, 3).tolist(), "on": on.tolist(), "noisy": noisy_on,
            "q": [round(float(v), 4) for v in q], "q_sum": q_sum, "q_min": q_min,
            "U": U, "u_threshold": u_thr, "abstain": bool(reasons), "reasons": reasons,
            "truth": [c for c, v in zip(CLASSES, sample["y"]) if v > 0.5],
            "lg": {"prob": p_lg.tolist(), "thr": thr_lg.tolist(), "pred": (p_lg >= thr_lg).tolist(),
                   "pred_full": (p_lg_full >= thr_lg).tolist()},
            "base": {"prob": p_base.tolist(), "thr": thr_base.tolist(), "pred": (p_base >= thr_base).tolist(),
                     "pred_full": (p_base_full >= thr_base).tolist(), "label": BASELINES[baseline]},
            "attn": np.round(out["attn"][-1][0].mean(0).numpy(), 4).tolist(),
        }


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console quiet during a presentation
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/meta":
                self._json(demo.meta())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                return self._json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._json(demo.predict(req))
            except (KeyError, ValueError, IndexError, TypeError) as e:
                self._json({"error": f"bad request: {e}"}, 400)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=str(REPO_DIR / "checkpoints"))
    p.add_argument("--samples-dir", default=str(REPO_DIR / "demo_samples"))
    p.add_argument("--results-dir", default=str(REPO_DIR / "results"), help="for the whole-test-set curve")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = p.parse_args()

    print("loading models and samples ...", flush=True)
    demo = Demo(Path(a.ckpt_dir), Path(a.samples_dir), Path(a.results_dir))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(demo))
    url = f"http://localhost:{a.port}"
    print(f"LeadGuard demo: {url}  ({demo.cfg['data']} data, {len(demo.samples)} samples)  Ctrl+C to stop", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
