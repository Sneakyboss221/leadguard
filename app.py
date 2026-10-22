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
                     sample is a demo sample index or the id of an uploaded ECG ("u1", "u2", ...)
  POST /api/upload   {files: [{name, data (base64)}], fs}: a WFDB record (.hea + .dat) or a CSV file
                     (one column per lead, mV; fs in Hz for CSV) -> id, name, missing leads, notes
  GET  /examples/F   example files from demo/examples (PTB-XL test-fold records not among the demo samples)
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import tempfile
import threading
import webbrowser
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

import corrupt
from data import zscore
from evaluate import apply_temperature
from models import build_model
from quality import load_sqa
from train import ckpt_paths
from utils import CLASSES, FS, LEADS, N_SAMPLES, REPO_DIR

UI_FILE = REPO_DIR / "demo" / "index.html"
EMG_SNR_DB = 10.0  # same SNR as the EMG test condition
MAX_BODY_BYTES = 100_000  # a predict request is well under 1 kB
MAX_UPLOAD_BYTES = 12_000_000  # base64 JSON; a 10 s 500 Hz WFDB record is about 0.2 MB
MAX_UPLOADS = 10  # uploaded ECGs kept in memory; the oldest is dropped
EXAMPLES_DIR = REPO_DIR / "demo" / "examples"
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


def _lead_index(name: str) -> int | None:
    """Position of a lead name in utils.LEADS ("aVR", "AVR", "Lead aVR" ...), None if it is not a standard lead."""
    key = re.sub(r"[^A-Z0-9]", "", name.upper()).removeprefix("LEAD")
    upper = [l.upper() for l in LEADS]
    return upper.index(key) if key in upper else None


def _parse_csv(text: str) -> tuple[np.ndarray, list[str]]:
    """CSV/TXT with one column per lead (optional header with lead names, optional leading time column).
    Returns (channels, samples) and the channel names."""
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        raise ValueError("the CSV file is empty")
    delim = next((d for d in (",", ";", "\t") if d in lines[0]), None)
    split = (lambda l: l.split(delim)) if delim else (lambda l: l.split())

    def is_num(c: str) -> bool:
        try:
            float(c)
            return True
        except ValueError:
            return False

    first = [c.strip().strip('"') for c in split(lines[0])]
    header = None if all(is_num(c) for c in first) else first
    try:
        arr = np.array([[float(c) for c in split(l)] for l in (lines[1:] if header else lines)], dtype=np.float64)
    except ValueError as e:
        raise ValueError("below the header row the CSV must contain only numbers, with the same number of "
                         "columns on every row") from e
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("could not read a table of numbers from the CSV file")
    if header is None:
        if arr.shape[0] in (12, 13) and arr.shape[1] > 13:  # leads stored as rows
            arr = arr.T
        if arr.shape[1] == 13:  # first column is time or sample number
            arr = arr[:, 1:]
        if arr.shape[1] != 12:
            raise ValueError(f"without a header row the CSV needs 12 columns in the order {', '.join(LEADS)} "
                             f"(found {arr.shape[1]})")
        return arr.T, list(LEADS)
    if len(header) != arr.shape[1]:
        raise ValueError(f"the header has {len(header)} names but the rows have {arr.shape[1]} values")
    return arr.T, header


def read_upload(files: list[dict], fs_csv: float) -> tuple[np.ndarray, list[bool], dict]:
    """Uploaded files -> raw signal (12, N_SAMPLES) in mV at FS Hz in utils.LEADS order, the leads found, notes.

    WFDB records are read with wfdb (sampling rate and units from the header); CSV files are taken as mV at
    fs_csv Hz. Signals are resampled to FS, cut to the first 10 s, and leads that are absent or flat stay zero.
    """
    by_ext: dict[str, list[tuple[str, bytes]]] = {}
    for f in files:
        name = Path(str(f.get("name", ""))).name
        if not re.fullmatch(r"[\w.\- ]{1,120}", name):
            raise ValueError(f"unsupported file name: {name!r}")
        try:
            data = base64.b64decode(str(f.get("data", "")), validate=True)
        except ValueError as e:
            raise ValueError(f"could not decode {name}") from e
        by_ext.setdefault(Path(name).suffix.lower(), []).append((name, data))

    notes = []
    if ".hea" in by_ext:
        import wfdb

        record = Path(by_ext[".hea"][0][0]).stem
        with tempfile.TemporaryDirectory() as tmp:
            for group in by_ext.values():
                for name, data in group:
                    (Path(tmp) / name).write_bytes(data)
            try:
                sig, meta = wfdb.rdsamp(str(Path(tmp) / record))
            except Exception as e:
                raise ValueError(f"could not read the WFDB record '{record}'. Select the .hea and the .dat file "
                                 "together, with their original names") from e
        sig, names, fs = np.asarray(sig, dtype=np.float64).T, list(meta["sig_name"]), float(meta["fs"])
        for k, unit in enumerate(meta.get("units") or []):
            if str(unit).lower() in ("uv", "\u00b5v", "microvolt"):
                sig[k] /= 1000.0
        source = "WFDB record"
    elif ".csv" in by_ext or ".txt" in by_ext:
        name, data = (by_ext.get(".csv") or by_ext[".txt"])[0]
        record = Path(name).stem
        sig, names = _parse_csv(data.decode("utf-8-sig", errors="replace"))
        fs = float(fs_csv)
        if not 50 <= fs <= 10_000:
            raise ValueError("the sampling rate must be between 50 and 10000 Hz")
        source = "CSV file"
    else:
        raise ValueError("upload a CSV file, or a WFDB record (the .hea and the .dat file together)")

    sig = np.nan_to_num(sig)
    raw, found = np.zeros((12, sig.shape[1])), [False] * 12
    for ch, name in enumerate(names):
        j = _lead_index(str(name))
        if j is not None and not found[j]:
            raw[j], found[j] = sig[ch], True
    if not any(found):
        raise ValueError(f"no standard lead names found ({', '.join(map(str, names[:12]))}); expected "
                         f"{', '.join(LEADS)}")
    if source == "CSV file" and np.percentile(np.abs(raw[found]), 99) > 20:
        raw /= 1000.0
        notes.append("values looked like microvolts and were divided by 1000")
    seconds = raw.shape[1] / fs
    if seconds < N_SAMPLES / FS:
        raise ValueError(f"the recording is {seconds:.1f} s long; the models need 10 s")
    if abs(fs - FS) > 1e-6:
        from scipy.signal import resample_poly

        frac = Fraction(FS / fs).limit_denominator(1000)
        raw = resample_poly(raw, frac.numerator, frac.denominator, axis=-1)
        notes.append(f"resampled from {fs:g} Hz to {FS} Hz")
    if seconds > N_SAMPLES / FS + 0.05:
        notes.append(f"first 10 s of {seconds:.1f} s used")
    raw = raw[:, :N_SAMPLES].astype(np.float32)
    found = [bool(f and raw[j].std() >= 1e-6) for j, f in enumerate(found)]
    raw[~np.array(found)] = 0.0
    info = {"source": source, "record": record, "fs": fs, "seconds": round(seconds, 2), "notes": notes,
            "missing": [LEADS[j] for j in range(12) if not found[j]]}
    return raw, found, info


def load_examples(examples_dir: Path) -> dict:
    """demo/examples/labels.json: example records with their PTB-XL labels (empty if absent)."""
    path = examples_dir / "labels.json"
    return json.loads(path.read_text()) if path.exists() else {"records": []}


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
        self.examples = load_examples(EXAMPLES_DIR)
        self.uploads: dict[str, dict] = {}
        self.n_uploads = 0

    def meta(self) -> dict:
        return {
            "data": self.cfg["data"], "debug": self.cfg["debug"], "classes": CLASSES, "class_names": CLASS_NAMES,
            "leads": LEADS, "fs": FS, "emg_snr_db": EMG_SNR_DB, "baselines": BASELINES,
            "u_threshold": self.cfg["runs"]["lg_edl"]["u_threshold"], "abstain_q_sum": self.cfg["abstain_q_sum"],
            "testset": self.testset, "examples": self.examples["records"],
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

    def upload(self, req: dict) -> dict:
        """Read an uploaded ECG and keep it under a new id. Labels are known only for the example files."""
        files = req.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("no file received")
        raw, found, info = read_upload(files, float(req.get("fs", FS)))
        known = {r["record"]: r["labels"] for r in self.examples["records"]}
        labels = known.get(info["record"])
        with self.lock:
            self.n_uploads += 1
            uid = f"u{self.n_uploads}"
            self.uploads[uid] = {"x": zscore(raw[None])[0], "x_raw": raw, "present": np.array(found),
                                 "y": None if labels is None else np.array([c in labels for c in CLASSES], float),
                                 "name": info["record"]}
            while len(self.uploads) > MAX_UPLOADS:
                self.uploads.pop(next(iter(self.uploads)))
        return {"id": uid, "name": info["record"], "labels": labels, "info": info}

    def _sample(self, key) -> dict:
        if isinstance(key, str) and key in self.uploads:
            return self.uploads[key]
        if isinstance(key, str) and key.startswith("u"):
            raise ValueError("this uploaded ECG is no longer in memory; upload it again")
        i = int(key)
        if not 0 <= i < len(self.samples):
            raise ValueError(f"'sample' must be in 0..{len(self.samples) - 1}")
        return self.samples[i]

    @torch.no_grad()
    def predict(self, req: dict) -> dict:
        sample = self._sample(req["sample"])
        on = np.array(req["on"], dtype=bool)
        if on.shape != (12,):
            raise ValueError("'on' needs 12 values")
        noisy = sorted({int(j) for j in req.get("noisy", []) if 0 <= int(j) < 12})
        use_sqa = req.get("quality") == "sqa"
        baseline = req.get("baseline") if req.get("baseline") in BASELINES else "resnet_c1"

        z, raw, noisy_on = self.simulate(sample, on, noisy, int(req.get("seed", 0)))
        z_full = torch.from_numpy(sample["x"].astype(np.float32))[None]  # reference: all 12 leads, no noise
        present = sample.get("present", np.ones(12, dtype=bool))  # an upload may lack some leads
        with self.lock:
            q = self.sqa(z)[0] if use_sqa else torch.from_numpy(on.astype(np.float32))
            out = self.models["lg_edl"](z, q[None])
            base = self.models[baseline](z, torch.ones(1, 12))
            q_full = self.sqa(z_full) if use_sqa else torch.from_numpy(present.astype(np.float32))[None]
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
            "truth": None if sample["y"] is None else [c for c, v in zip(CLASSES, sample["y"]) if v > 0.5],
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
            elif path.startswith("/examples/"):
                name = path.removeprefix("/examples/")
                allowed = {f for r in demo.examples["records"] for f in r["files"]}
                if name not in allowed or not (EXAMPLES_DIR / name).is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = "text/csv" if name.endswith(".csv") else "application/octet-stream"
                body = (EXAMPLES_DIR / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            routes = {"/api/predict": (demo.predict, MAX_BODY_BYTES), "/api/upload": (demo.upload, MAX_UPLOAD_BYTES)}
            if self.path not in routes:
                return self._json({"error": "not found"}, 404)
            handle, limit = routes[self.path]
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > limit:
                    return self._json({"error": "request body too large"}, 413)
                req = json.loads(self.rfile.read(n))
                self._json(handle(req))
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
