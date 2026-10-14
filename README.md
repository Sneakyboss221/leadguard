# LeadGuard: robust ECG classification under lead failure

Course project. Architecture and rules: `SPEC.md`. Agent rules: `AGENTS.md`. Build order: `PROMPTS.md`. Human walkthrough: `GUIDE.md`.

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

## Layout

```
data.py  corrupt.py  plot_corruptions.py  quality.py  pretrain_sqa.py
models.py  baseline_xgb.py  train.py  run_all.py
evaluate.py  figures.py  tests.py  app.py  make_demo_samples.py
cache/  checkpoints/  results/  figures/  demo_samples/
```

Numbers in the paper must come from `results/*.csv`.
