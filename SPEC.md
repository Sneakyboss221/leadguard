# Build Spec: LeadGuard (the complete project, lean version)

Code name for classes and files: `LeadGuard`. Paper title (separate from the code name), suggestion: *Quality-Gated Attention with Evidential Abstention for Robust ECG Classification under Lead Failure*.

This spec is the whole project. Do not add models, experiments, figures or features that are not listed here.

## 0. Overview

An ECG classifier that stays usable when leads fail:

1. A **Signal Quality Assessor (SQA)** gives each lead a reliability score q in [0, 1].
2. A **Mask-Aware Lead Attention (MALA)** transformer uses q to suppress unreliable leads.
3. A **Beta-EDL head** (evidential, per class) outputs probabilities plus uncertainty. The system abstains when uncertainty is high or too little clean signal remains.

We compare it against baselines and ablations under simulated lead failure, then ship a Streamlit demo and a conference-style paper.

Rules for the coding tool: follow file names, signatures, shapes and tests exactly; write clean PyTorch; never invent results; ask when something is ambiguous. Develop and test on CPU with the synthetic dataset and `--debug`; train on a Colab T4 GPU.

Claims rule: masking attention and evidential learning are both known techniques. The contribution is the combination plus the evaluation protocol. Do not write "first". Every number in the paper must come from `results/*.csv`.

---

## 1. Data contract (`data.py`), the only code that touches raw files

`load_dataset(name, root=None, debug=False) -> (X, y, fold)`

- `X`: float32 `(N, 12, 1000)` (10 s at 100 Hz), lead order I, II, III, aVR, aVL, aVF, V1-V6, per-lead z-score per record.
- `y`: float32 `(N, 5)` multi-hot: NORM, MI, STTC, CD, HYP.
- `fold`: ints; train 1-8, validation 9, test 10.
- `name="synthetic"`: random but correctly shaped data (N = 600, folds 1-10 spread evenly), so everything can be built and tested without the real dataset.
- `name="ptbxl"`: PTB-XL waveforms at 100 Hz read with `wfdb` from `root`; `ptbxl_database.csv` (`strat_fold`) and `scp_statements.csv` (rows with `diagnostic == 1`, column `diagnostic_class`) give the 5 superclasses; drop records with no label; cache to `cache/ptbxl.npz`; if the cache exists, load it instead of reading raw files.
- `debug=True`: first 500 records only.
- `python data.py --name synthetic` prints shapes, dtypes, label counts per class and records per fold.
- For the demo, also keep the raw (un-normalized, mV) signal: `load_dataset(..., return_raw=True)` additionally returns `X_raw` with the same shape.

**OPEN, resolve before the real run (do not silently drop this):** the course requires data published 2023-2026. PTB-XL waveforms are from a 2020 release. Using the 2023 PTB-XL+ release is only defensible if the professor accepts its release date. Another dataset can be added by writing one more branch in `data.py` that returns the same three arrays.

---

## 2. Corruption simulation (`corrupt.py`)

Vectorized torch on `(B, 12, T)`. Each function returns `(x_corrupt, severity)` with `severity` of shape `(B, 12)` in [0, 1] (0 untouched, 1 dead). Derived: `true_mask = (severity < 0.5)`, `q_target = 1 - severity`.

- `drop_random(x, k)`: zero k random leads (severity 1).
- `drop_group(x, group)`: `precordial` (V1-V6), `limb` (I, II, III, aVR, aVL, aVF), `v1_v3`.
- `flatline`: constant value (severity 1).
- `add_emg_noise(x, leads, snr_db)`: broadband noise band-limited to 20-45 Hz. Never simulate 50 or 60 Hz: 100 Hz sampling cannot represent them. Severity = `1 - clip(snr_db / 20, 0, 1)`.
- `add_baseline_wander`: sinusoid 0.1-0.5 Hz, amplitude 0.5-3x lead std. Severity = `clip(amplitude_ratio / 2, 0, 1)`.
- `clip_amplitude`: saturate at a percentile threshold. Severity = fraction of samples clipped, scaled to [0, 1].
- `training_corruption(x, gen)`: per batch, 50% random drop of 0-6 leads, 15% group drop, 30% artifact on 1-3 leads (can combine). Always keep at least 2 good leads.
- `fixed_test_corruption(x, condition, seed)`: deterministic, so every model sees identical corrupted test data.
- `plot_corruptions.py` saves `figures/corruption_examples.png` (one sample before and after each corruption).

---

## 3. Models (`quality.py`, `models.py`, `baseline_xgb.py`)

Encoder-side parameters under about 1 million. Every neural model: `forward(x, mask_or_q) -> outputs`; baselines ignore the second argument and receive zero-filled input.

### A. SQA (`LeadQualityNet`)
- Shared small 1D CNN on `(B*12, 1, 1000)`: 4 Conv1D blocks (kernel 7, stride 2, BatchNorm, GELU), global pool, linear, sigmoid, giving q per lead.
- Pre-train in `pretrain_sqa.py` on clean plus synthetically corrupted data, BCE against soft `q_target`. Then freeze it (eval mode, no gradient). Save `checkpoints/sqa.pt`.
- Check: thresholded at 0.5, lead-level agreement with the synthetic `true_mask` must exceed 95%, printed per corruption type. Clean real leads should mostly get q above 0.8.
- `estimate_q(model, x) -> (B, 12)`.

### B. Lead encoder
- Shared ResNet-1D stem per lead (3 residual stages, global pooling) to a 128-d vector `h_j`. Keep it small.
- Add a learned 128-d embedding per lead position (12 of them).

### C. MALA
- 2 transformer layers, 4 heads, d = 128. Write attention by hand and return the attention weights.
- Key bias: `log(clamp(q, min=1e-6))`. **q floor:** any lead with `q < 0.02` is fully masked (key bias `-1e9`, pooling weight 0). This makes invariance exact for dead leads.
- Equation: `softmax(Q K^T / sqrt(d_k) + log(q_j))` over keys j.
- Weighted pooling, applied to the output sequence of the final transformer layer (not the encoder output): `z = sum_j(w_j * h'_j) / (sum_j(w_j) + 1e-6)`, where `h'_j` is the transformer output for lead j and `w_j = q_j` (0 below the floor).
- Guard: if every lead is below the floor, keep the lead with the highest q.
- Hard-mask variant: `q` is replaced by `(q >= 0.5)` as 0/1, with the same attention and pooling code.

### D. Heads
- **Sigmoid head:** linear 128 to 5, BCE-with-logits.
- **Beta-EDL head:** linear to 10 outputs with softplus, giving `e_plus_k, e_minus_k >= 0` per class. `alpha_k = e_plus_k + 1`, `beta_k = e_minus_k + 1`. `p_k = alpha_k / (alpha_k + beta_k)`. `u_k = 2 / (alpha_k + beta_k)`. Record-level uncertainty `U = mean_k u_k`.
- **EDL loss:** per class, expected binary cross-entropy under Beta(alpha, beta): for y = 1 use `digamma(alpha + beta) - digamma(alpha)`; for y = 0 use `digamma(alpha + beta) - digamma(beta)`. Add a KL regularizer toward Beta(1, 1) on the misleading evidence: if y_k = 1, penalize beta_k toward 1; if y_k = 0, penalize alpha_k toward 1. Formally KL(Beta(alpha~_k, beta~_k) || Beta(1, 1)), with alpha~_k = 1, beta~_k = beta_k when y_k = 1 and alpha~_k = alpha_k, beta~_k = 1 when y_k = 0. Sum over the 5 classes; weight annealed from 0 to 1 over the first 10 epochs (adapted from Sensoy et al. 2018; verify against the paper).
- **Abstain rule:** abstain on the whole record when `U` exceeds a threshold tuned on the validation fold, or when `sum_j q_j < 3`.

### Model list (8 models, 9 trained variants)

| ID | Code name | Model | Missing-lead handling | Role |
| --- | --- | --- | --- | --- |
| M1 | `xgb` | XGBoost on per-lead statistics (mean, std, min, max, skew, kurtosis, dominant frequency, peak-to-peak) | NaN for missing leads | Classical baseline |
| M2 | `mlp` | MLP 12000-512-128-5, dropout | Zero-fill | Basic neural baseline |
| M3 | `resnet` | ResNet-1D (12 channels, 4 residual stages), trained clean (M3a) and with corruption training (M3b) | Zero-fill | Standard strong baseline; M3b is the fair baseline |
| M4 | `transformer` | 1D Transformer over time patches (patch 25, unmasked) | Zero-fill | Transformer baseline |
| M5 | `resnet_attn` | Lead encoder + unmasked lead attention | Zero-fill | Attention without quality gating |
| M6 | `lg_hard` | LeadGuard, hard mask, sigmoid head | Binary mask from SQA | Ablation: hard vs soft |
| M7 | `lg_soft` | LeadGuard, soft gating, sigmoid head | Soft q | Ablation: gating without EDL |
| M8 | `lg_edl` | LeadGuard, soft gating, Beta-EDL head | Soft q + evidential abstention | **Proposed** |

---

## 4. Training (`train.py`, `run_all.py`)

`python train.py --model lg_edl --seed 0 --corrupt-train 1 --data ptbxl --data-root PATH --ckpt-dir DIR --out-dir DIR [--epochs 15] [--batch 128] [--debug]`

- AdamW, lr 1e-3, weight decay 1e-2, cosine decay, mixed precision. Max 15 epochs, early stopping patience 4 on validation macro-AUROC. Keep the whole training split on the GPU as one float32 tensor.
- With `--corrupt-train 1`: corrupt each batch with `training_corruption`, then run the **frozen SQA on the corrupted signals** to get q (what the model sees at test time). Never feed true severities to the classifier.
- Sigmoid models: BCE-with-logits. `lg_edl`: EDL loss.
- Save best and last checkpoints every epoch to `--ckpt-dir` as `{model}_s{seed}_c{0|1}.pt`; resumable after a disconnect.
- XGBoost: `baseline_xgb.py` with its own train and predict functions, same prediction format as the neural models.

**Run list (`run_all.py`)**, executed one after another, skipping any whose final checkpoint already exists, logging wall-clock time per run to `results/timing.csv`:

| Model | corrupt-train | Seeds |
| --- | --- | --- |
| `xgb` | n/a | 0 |
| `mlp` | 1 | 0 |
| `resnet` | 0 and 1 | 0; plus 1 and 2 for corrupt-train 1 |
| `transformer` | 1 | 0 |
| `resnet_attn` | 1 | 0 |
| `lg_hard` | 1 | 0 |
| `lg_soft` | 1 | 0 |
| `lg_edl` | 1 | 0, 1, 2 |

That is 12 neural runs plus XGBoost; target about 1-2 GPU hours on a T4 (an estimate; measure it). Report mean and std only for `resnet` (corrupt-train 1) and `lg_edl`; label every other result "single seed".

---

## 5. Tests (`tests.py`); nothing trains until these pass

1. **Hard-mask invariance (`lg_hard`):** eval mode; set a mask entry to 0 and replace that lead with random noise; logits identical within 1e-5.
2. **Soft invariance (`lg_soft`, `lg_edl`):** bypass the SQA and inject `q_j = 0` for chosen leads; replacing those leads with noise leaves logits identical within 1e-5. With the SQA in the loop, noise changes q, so invariance is NOT expected there; do not test it that way.
3. **Attention check:** attention on floored keys is exactly 0 and each row sums to 1.
4. **Degenerate inputs:** no NaN with a single active lead, or with all q near 0.
5. **Shapes:** every model returns the expected shape for `(B, 12, 1000)`.
6. **Corruption masks:** each corruption function reports severity 0 exactly on untouched leads.
7. **Determinism:** `fixed_test_corruption` with one seed returns identical tensors twice.
8. **Overfit check:** `lg_edl` on 200 samples for 50 epochs reaches training macro-AUROC above 0.95.
9. **SQA check:** the accuracy requirement in section 3A (on the real run).

---

## 6. Evaluation (`evaluate.py`), outputs to `results/*.csv`

Conditions (identical corrupted data for every model):
- Random lead loss k in {0, 2, 4, 6, 8, 10}, 3 draws each.
- Group loss: precordial, limb, V1-V3.
- Artifacts: EMG noise at 10 dB SNR, baseline wander, and clipping, each on 2, 4 and 6 leads.
- For `lg_hard`, `lg_soft`, `lg_edl`: also evaluate with the true q (oracle) versus the SQA-estimated q.

Metrics:
- Macro-AUROC (primary), macro-F1 with per-class thresholds tuned on clean validation predictions, per-class AUROC.
- Expected calibration error (15 bins). Temperature scaling fitted on validation for the sigmoid models, so the comparison with EDL is fair.
- Paired bootstrap 95% CI on the test fold for `lg_edl` versus `resnet` (corrupt-train 1).

Abstention (risk-coverage), defined for multi-label:
- Record-level uncertainty. `lg_edl`: `U = mean_k u_k`. Sigmoid models: `1 - mean_k max(p_k, 1 - p_k)`. Also report `lg_edl` with the sigmoid-style score for a like-for-like check.
- Sort records by uncertainty (low first); coverage = fraction kept; risk = mean per-record label error rate (fraction of the 5 labels wrong at tuned thresholds) among kept records.
- Report abstain rate versus number of missing leads. Do not assume uncertainty rises with lead loss; measure it.

CSV files: `main_clean.csv`, `random_loss.csv`, `group_loss.csv`, `artifact.csv`, `oracle_vs_estimated.csv`, `calibration.csv`, `risk_coverage.csv`, `per_class_auroc.csv`, `abstain_rate.csv`, `timing.csv`.

## 7. Figures and tables (`figures.py`); reads only `results/*.csv` plus saved attention; 300 DPI PNG

1. Macro-AUROC vs number of missing leads, one line per model, shaded std where seeds exist.
2. Attention heatmap before and after disconnecting V1 or II.
3. Group-loss bars (precordial, limb, V1-V3).
4. Risk-coverage curves plus reliability diagram.
Tables as CSV: `table1.csv` (clean results, all models), `table2.csv` (ablation: M5, M6, M7, M8). Titles, labeled axes, colorblind-safe palette.

## 8. Demo (`app.py`, Streamlit, CPU, offline, single file)

- Loads one baseline (`resnet`; a dropdown switches between the clean-trained and corruption-trained versions, so the demo is not rigged), `lg_edl`, the frozen SQA, and 20 `.npz` demo samples (raw mV signal plus z-scored signal plus label), exported by `make_demo_samples.py` from cases the results support.
- Sidebar: sample picker; 12 lead checkboxes; presets (Drop V1-V6, Drop limb, Random 6 off, Add EMG noise to 3 leads, Reset); "Auto-detect quality" button.
- Main: 12-lead plot with a fixed vertical offset between leads like a clinical printout, time axis in seconds, 1 mV scale bar; plot the raw mV signal (z-scoring removes real amplitude); disconnected leads grey, noisy leads orange; simulated corruption applied to both raw and z-scored copies. Per-lead q bars. Attention heatmap. Side-by-side probability bars with the true label. Uncertainty gauge and an "ABSTAIN" banner using the validation-tuned threshold.
- On-screen note: "Research prototype, not for clinical use."

## 9. Paper wording rules

State exactly which models were trained, how many seeds, and that results other than `resnet` and `lg_edl` are single-seed. State that lead loss is simulated, the data is retrospective, and nothing is clinically validated. Do not claim the combination or any single technique is "the first".

## 10. Acceptance checklist

- [ ] Dataset date question answered by the professor
- [ ] `python -m pytest -q tests.py` passes
- [ ] SQA accuracy check passed on real data before freezing
- [ ] `run_all.py` finished; `results/timing.csv` exists
- [ ] All CSVs, Figures 1-4 and Tables 1-2 generated from real data
- [ ] Demo runs offline; abstain banner shown only if the results support it
- [ ] Backup screen recording saved
- [ ] Every number in the paper traces to a CSV cell; every reference verified
