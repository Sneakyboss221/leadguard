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

## Models and evaluation (decided now, implemented later)
- The "under ~1M parameters" limit applies to the LeadGuard and attention encoders. The MLP baseline (12000-512-128-5, ~6.2M) is exempt because the spec fixes its size.
- XGBoost "missing" leads are detected from the signal (lead std < 1e-6 → features NaN), never from true severity.
- Temperature scaling: one temperature per sigmoid model, fitted on the validation fold.
- ECE: 15 bins over all 5 x N binary predictions, flattened.
- `utils.py` holds `set_seed`, `get_device`, and the fixed lead and class orders.
- Python: developed locally on 3.14 (the only version installed); Colab uses 3.12. Avoid version-specific features.
