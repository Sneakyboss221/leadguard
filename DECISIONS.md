# DECISIONS.md: interpretations of and deviations from SPEC.md

`SPEC.md` stays the source of truth. This file records every place where the spec was ambiguous or where we chose to deviate, with the reason. Mention the relevant ones in the paper's method section.

## Dataset
- **PTB-XL publication date.** The requirement is a dataset published within 5 years of 2026 (2021-2026). We use PTB-XL **v1.0.3**, published on PhysioNet on 9 Nov 2022 (DOI 10.13026/kfzx-aw45). Cite this version and DOI. The original PTB-XL paper is from 2020; confirm with the professor that the release date counts.
- **Labels.** All `scp_codes` whose statement has `diagnostic == 1` map to their `diagnostic_class`, regardless of the likelihood value (the common convention from the PTB-XL benchmark). Records with no superclass are dropped.
- **Cache.** `cache/ptbxl.npz` stores the raw mV signal; the z-scored copy is computed on load. This keeps one full copy in memory, except when `return_raw=True` (demo). `--debug` without a full cache reads only the first 500 records into `cache/ptbxl_debug.npz`.
- **Z-score.** Per lead per record, with a guard: leads with std < 1e-6 (dead) become all zeros instead of NaN.

## Synthetic data (deviation)
- The spec says "random but correctly shaped". We make it **learnable**: ECG-like beats, random multi-hot labels (40% NORM-only), and each class changes the waveform on specific leads with a random per-record strength (MI: Q wave on V1-V4; STTC: T wave on I, aVL, V5, V6; CD: QRS width on II, III, V1, V2; HYP: R height on aVL, V4-V6).
- Why: with pure noise every metric is ~0.5 and wiring bugs (label order, folds, masking) are invisible until the GPU run. Measured with a gradient-boosting model on per-lead features: macro-AUROC 0.93 clean, 0.83 with V1-V6 dropped (MI falls to 0.61), so lead loss visibly matters. Shapes, dtypes and folds follow the contract (N = 600, 60 per fold).
- Synthetic numbers are for debugging only and never go into the paper.

## Corruption
- **Severities exactly at the 0.5 boundary.** EMG at 10 dB gives severity exactly 0.5, so `true_mask = severity < 0.5` labels it bad by a hair; baseline wander with amplitude ratio 1.0 does the same. We keep the spec's 10 dB test condition. The SQA agreement check (SPEC 3A) is reported twice: on all leads, and excluding the ambiguous band |severity - 0.5| < 0.1. The 95% gate uses the second number. Both are reported.
- **Per-record sampling.** `training_corruption` samples its corruption independently for each record, not one corruption per batch (more variety per step).
- **Training mix details.** 30% artifact on 1-3 leads (EMG with SNR uniform in 0-20 dB, baseline wander, or clipping), 50% random drop of 0-6 leads, 15% group drop; these combine, and artifacts are applied before drops. A quarter of dropped leads are flatlined instead of zeroed, so `flatline` is part of training. Severity of combined corruptions is the maximum. If fewer than 2 good leads remain, the least-corrupted leads are restored to clean. Measured on synthetic data: 36% of records stay clean, at least 3 good leads always remain, 2.3 bad leads per record on average.
- **Flatline.** The lead holds its first sample value (stuck amplifier), which differs from a zeroed lead.
- **EMG noise.** White noise, band-pass 20-45 Hz via FFT, scaled to the target SNR relative to that lead's variance.
- **Clipping.** Threshold at a percentile of |x| per lead (uniform 70-95 unless given). Severity = clipped fraction / 0.2, capped at 1.
- **Fixed test corruption.** Conditions: `clean`, `random_{k}` (3 draws = seeds 0-2), `group_{precordial|limb|v1_v3}`, `{emg|wander|clip}_{2|4|6}`; listed in `corrupt.TEST_CONDITIONS`. The generator is seeded from (seed, condition name) on CPU, so results do not depend on the device. It must be applied to the whole test set at once, not per batch.

## SQA (Step 2)
- "Global pool" is implemented as global average + global max pooling concatenated.
- Model selection on fold 9 (validation BCE); the agreement check is reported on fold 10 and saved to `results/sqa_agreement.csv` (so the number quoted in the paper comes from a CSV).
- Gate accepted by the user: min over corruption types of `agreement_excl_band` > 0.95. All-leads and corrupted-leads-only agreement are reported alongside. On synthetic data, 40 epochs pass (0.967); the 2-epoch `--debug` run only checks that the script runs.

## LeadGuard models (Step 3)
- Lead encoder: stem conv (k7, stride 2, 32 ch) + 3 residual stages (32, 64, 128 ch, stride 2, k7) + global average pool + linear to 128. Parameters: encoder 260k, MALA 265k, total 527k (under ~1M).
- MALA layers are pre-norm (LayerNorm before attention and FFN), FFN width 256, dropout 0.1 on the FFN and residual branches only (never on attention weights, so rows always sum to 1).
- All-floored guard: the kept lead gets key bias 0 and pooling weight 1 (with q ~ 0 its q-weight would make the pooled vector ~0). In the hard variant the guard picks the lead with the highest *soft* q, because the 0/1 mask has ties.
- Outputs: every model returns a dict with `logits` (raw head output: (B, 5) sigmoid, (B, 10) Beta-EDL) and `prob` (B, 5); LeadGuard adds `attn`, and the EDL head adds `alpha`, `beta`, `u`, `U`. The invariance tests compare `logits`.
- EDL loss: summed over classes, averaged over the batch; KL weight = min(1, epoch / 10) with 0-based epochs.

## Baselines (Step 4)
- Parameter counts: mlp 6.21M, resnet 574k, transformer 310k, resnet_attn 526k, lg_hard / lg_soft 526k, lg_edl 527k.
- `mlp`: dropout 0.3 after both hidden layers, GELU.
- `resnet`: stem conv (12 -> 64 ch, k7, stride 2) + 4 residual stages (64, 64, 128, 128 ch, stride 2) + global average pool + linear to 128 + sigmoid head. Clean-trained (M3a) vs corruption-trained (M3b) is a training flag, not a different model.
- `transformer`: the spec gives no depth, so it uses the same size as MALA for comparability (2 pre-norm layers, 4 heads, d = 128, FFN 256), learned positional embedding over the 40 patches, mean pooled.
- `resnet_attn`: the LeadGuard encoder and attention code with q replaced by all ones (unmasked attention, mean pooling), so the only difference from `lg_soft` is the quality gating.
- `xgb`: one binary XGBClassifier per class (500 trees max, depth 4, learning rate 0.05, subsample and colsample 0.8, early stopping 30 rounds on fold 9 AUROC). Saved as `checkpoints/xgb_s{seed}.pkl`. Writing predictions to `results/` happens in `evaluate.py` (Step 6), from `predict_xgb`'s `{"prob": (N, 5)}`, the same format as the neural models.

## Training (Step 5)
- Early stopping and best-checkpoint selection use macro-AUROC on the **clean** validation fold (fold 9). For SQA models, q on validation comes from the frozen SQA applied to the clean signal.
- Checkpoints: `{model}_s{seed}_c{c}.pt` = best weights (what evaluation loads); `{model}_s{seed}_c{c}_last.pt` = full training state with a `finished` flag. Both are written every epoch via temp file + rename. "Final checkpoint exists" in `run_all.py` means `_last.pt` has `finished = True` and the best checkpoint exists.
- Resume restores model, optimizer, scheduler, AMP scaler, early-stopping counter, history and all RNG states (python, numpy, torch, CUDA, the corruption generator). Verified: a run killed mid-epoch and resumed reproduces the uninterrupted run exactly (loss 3.6112, val macro-AUROC 0.6546 at epoch 2, lg_edl synthetic debug).
- Safety: a checkpoint written under a different data / debug / epochs / batch / model / seed / corrupt-train setting makes `train.py` and `run_all.py` stop with an error instead of resuming or skipping; the SQA checkpoint must come from the same dataset. An unreadable `_last.pt` is renamed `*.unreadable` and the run restarts. The XGBoost checkpoint `xgb_s{seed}.pkl` stores data/debug too.
- `run_all.py` runs each job as its own process, checks every run (and the SQA) before starting, appends one row per invocation to `timing.csv` (a resumed run gets a second row with `resumed = True`; total time = sum of rows), and stops at the first failure.
- `--debug` uses 2 epochs. `train.py` accepts `--out-dir` for interface consistency; it writes nothing there (results come from `evaluate.py`).
- No gradient clipping (not in the spec). The loss is checked every step and training stops on NaN/inf.

## Models and evaluation (decided now, implemented later)
- The "under ~1M parameters" limit applies to the LeadGuard and attention encoders. The MLP baseline (12000-512-128-5, ~6.2M) is exempt because the spec fixes its size.
- XGBoost "missing" leads are detected from the signal (lead std < 1e-6 → features NaN), never from true severity.
- Temperature scaling: one temperature per sigmoid model, fitted on the validation fold.
- ECE: 15 bins over all 5 x N binary predictions, flattened.
- `utils.py` holds `set_seed`, `get_device`, and the fixed lead and class orders.
- Python: developed locally on 3.14 (the only version installed); Colab uses 3.12. Avoid version-specific features.
