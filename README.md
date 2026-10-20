# LeadGuard: do quality gating and evidential abstention help ECG classifiers survive lead failure?

Code, results, paper and interactive demo for a controlled ablation on PTB-XL. A frozen signal-quality assessor
scores every lead, a mask-aware lead attention uses those scores, and a per-class Beta-evidential head decides
when to abstain. The study compares this system with three ablations and five baselines under simulated lead
failure. In short:

- **Training with simulated lead failure does the heavy lifting**: +0.187 macro-AUROC for the same ResNet with
  ten of twelve leads missing, at no cost on complete recordings.
- **Quality gating adds no measurable benefit** over un-gated lead attention.
- **Evidential uncertainty shows an illusion of certainty**: it stays flat as leads disappear while accuracy
  falls; all abstention under lead loss comes from counting usable leads.

All numbers come from `results/*.csv`; the paper (`paper/`) is generated from them.

## Try the demo (CPU, offline)

1. Install the environment (see Quick start below).
2. Download `leadguard_checkpoints_v1.zip` from the
   [v1.0 release](https://github.com/Sneakyboss221/leadguard/releases/tag/v1.0) and unzip it in the repository
   folder, so that the files land in `checkpoints/`.
3. Run `python app.py`; the page opens in your browser.

## Rebuild the paper

```
python paper/make_numbers.py     # numbers, tables and figures from results/*.csv
python paper/make_vignette.py    # the illustrative case (needs the checkpoints)
```
Then compile `paper/main.tex` with pdflatex and bibtex (or upload `paper/` to Overleaf).

## Data

PTB-XL v1.0.3 (Wagner et al., Scientific Data 2020, doi:10.1038/s41597-020-0495-6; PhysioNet,
doi:10.13026/kfzx-aw45). The raw data are not included; download them from PhysioNet under its terms.
`demo_samples/` contains 20 test-fold records derived from PTB-XL.

## Design documents

`SPEC.md` is the specification (data contract, corruption model, architectures, tests, evaluation protocol).
`DECISIONS.md` records every interpretation of, and deviation from, the specification.

## Reproduce on PTB-XL (GPU)

The paper's runs used one GPU; the full run list took about 25 minutes. Download PTB-XL v1.0.3 from PhysioNet
(about 2 GB), then:
```
python data.py --name ptbxl --root PATH/TO/ptb-xl-1.0.3      # builds cache/ptbxl.npz once (about 1 GB)
python -m pytest -q tests.py                                 # must pass before training
python pretrain_sqa.py --data ptbxl --ckpt-dir OUT/checkpoints --out-dir OUT/results
python run_all.py --data ptbxl --ckpt-dir OUT/checkpoints --out-dir OUT/results
python evaluate.py --data ptbxl --strict --ckpt-dir OUT/checkpoints --out-dir OUT/results
python figures.py --results-dir OUT/results --fig-dir OUT/figures
python make_demo_samples.py --data ptbxl --ckpt-dir OUT/checkpoints --results-dir OUT/results --out-dir OUT/demo_samples
```
`pretrain_sqa.py` must end with `PASS` (lead-level agreement above 0.95 outside the ambiguous severity band).
`run_all.py` skips finished runs and resumes interrupted ones, so after a disconnect the same command continues;
`--only RUN ...` runs a subset (e.g. `--only resnet_s0_c1 lg_edl_s0_c1`). `evaluate.py --strict` refuses to run
unless all 13 runs are finished. Add `--cache-dir DIR` to every command if the cache lives elsewhere.

## License

Code: MIT (see `LICENSE`). PTB-XL is not redistributed here; it is subject to its own PhysioNet terms.

## Quick start (local, CPU, synthetic data)

Windows:
```
python -m venv C:\venvs\leadguard
C:\venvs\leadguard\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests.py
```

Mac/Linux:
```
python3 -m venv ~/venvs/leadguard
source ~/venvs/leadguard/bin/activate
pip install torch
pip install -r requirements.txt
python -m pytest -q tests.py
```

Whole pipeline on synthetic data (about 25 minutes on a laptop CPU):
```
python pretrain_sqa.py --data synthetic --epochs 40
python run_all.py --data synthetic --debug
python evaluate.py --data synthetic --debug --strict
python figures.py
python make_demo_samples.py --data synthetic --debug
python app.py
```
Synthetic numbers only check that the code runs; they never go into the paper. For the real data, see
"Reproduce on PTB-XL" above.

## Layout

```
utils.py              shared constants (lead and class order), set_seed, device, atomic checkpoint save
data.py               data contract: synthetic and PTB-XL loaders, cache
corrupt.py            lead-failure simulation (training mix and fixed test conditions)
plot_corruptions.py   figures/corruption_examples.png
quality.py            signal quality assessor (SQA)
pretrain_sqa.py       SQA pre-training and agreement check
models.py             LeadGuard (lg_hard, lg_soft, lg_edl) and the neural baselines
baseline_xgb.py       XGBoost baseline
train.py              one training run (resumable checkpoints)
run_all.py            the full run list, results/timing.csv
evaluate.py           all test conditions and metrics -> results/*.csv
figures.py            Figures 1-4, Tables 1-2
tests.py              SPEC section 5 tests
make_demo_samples.py  demo samples and demo_config.json
app.py                demo server (standard library HTTP + JSON API), opens the page in a browser
demo/index.html       demo page (plain HTML/CSS/JS, no internet needed)
cache/  checkpoints/  results/  figures/  demo_samples/
```

Common options: `--data synthetic|ptbxl`, `--data-root` (raw PTB-XL, only needed to build the cache), `--cache-dir` (where `ptbxl.npz` lives; default `cache/`), `--ckpt-dir`, `--out-dir`, `--debug`.

Numbers in the paper must come from `results/*.csv`.
