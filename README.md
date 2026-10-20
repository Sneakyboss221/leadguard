# LeadGuard: robust ECG classification under lead failure

Course project. Architecture and rules: `SPEC.md`. Agent rules: `AGENTS.md`. Build order: `PROMPTS.md`. Human walkthrough, including the Colab run: `GUIDE.md`. Every interpretation of or deviation from the spec: `DECISIONS.md`.

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
Synthetic numbers only check that the code runs; they never go into the paper. Real-data commands are in `GUIDE.md` section 9.

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
