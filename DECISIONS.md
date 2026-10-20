# DECISIONS.md: interpretations of and deviations from SPEC.md

`SPEC.md` stays the source of truth. This file records every place where the spec was ambiguous or where we chose to deviate, with the reason. Mention the relevant ones in the paper's method section.

## Dataset
- **Dataset and its date.** Signals and labels come from PTB-XL **v1.0.3**, published on PhysioNet on 9 Nov 2022 (DOI 10.13026/kfzx-aw45); the original PTB-XL paper is from 2020. Its companion release **PTB-XL+ v1.0.1** (published 27 Jun 2023, DOI 10.13026/g6h6-7g88; Strodthoff et al., Scientific Data 10, 279, 2023) contains features, median beats, fiducial points and SNOMED-mapped labels, but no full recordings; the pipeline uses none of its files. Describe it exactly this way; never call PTB-XL+ the training data.
- **Why PTB-XL (justification for the paper).** Chosen for suitability, not for results: it is the largest public 12-lead dataset with curated diagnostic labels for the 5 superclasses, it ships official stratified folds (`strat_fold`, used for train 1-8 / val 9 / test 10), and it is the standard benchmark for this task, so results are comparable with published work. Switching datasets to chase better numbers would be selection bias. The SPEC 1 "OPEN" item stays open until the professor accepts this description.
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

## Evaluation and figures (Step 6)
- Conditions: `clean` plus the 30 entries of `corrupt.TEST_CONDITIONS` (`random_0` equals clean and gives the k = 0 point). Each condition corrupts the whole test fold once on CPU; every model sees the same tensor. SQA models are evaluated with estimated q and oracle q = 1 - severity; all other CSVs use estimated q.
- Fitted on clean validation (fold 9) predictions, never on test: one temperature per model for every model except `lg_edl` (XGBoost included, since its outputs are sigmoids of a margin), searched in [e^-3, e^3] with a warning at the bound; per-class F1 thresholds on the temperature-scaled probabilities (grid 0.01-0.99); and the `lg_edl` U threshold.
- **Abstain threshold (spec says only "tuned on the validation fold"):** the 95th percentile of U on clean validation, i.e. 5% abstention on clean validation data. Stored in `abstain_rate.csv` (`u_threshold`) for the demo.
- ECE: 15 equal-width bins over the positive-class probability of all N x 5 predictions (flattened). Reliability bins for the clean condition go to `calibration.csv`.
- Risk-coverage at 20 coverage levels (0.05-1.00); ties in uncertainty keep record order (stable sort). Scores: `edl_U` (lg_edl only) and `sigmoid_style` for every model, including lg_edl for the like-for-like check.
- `abstain_rate.csv` reports the combined rule and each part separately (U above threshold, sum q < 3), plus mean U and mean sum q per condition, so the relation between uncertainty and lead loss is measured, not assumed.
- Paired bootstrap (extra file `bootstrap_ci.csv`; the spec asks for the CI but lists no file): difference in macro-AUROC, `lg_edl` minus `resnet` corruption-trained, each side averaged over its 3 seeds, on the same resampled test records; 1000 resamples (100 with `--debug`), percentile 95% CI, every condition.
- `attention.npz`: first `lg_edl` run, estimated q, last-layer attention averaged over heads and test records, for clean / V1 zeroed / II zeroed.
- Figures: 9 variants on the validated 8-hue categorical palette; the two ResNets share a hue and differ by line style (Fig. 1) or hatch (Fig. 3); every series also has a distinct marker. Shaded bands and error bars = std over seeds, only where there are 3 seeds. Fig. 4 shows risk-coverage for clean and `random_6` and the clean reliability diagram for lg_edl vs resnet (corruption-trained).
- `table2.csv` (ablation M5-M8) columns: clean macro-AUROC, random loss k = 6 and k = 10 (mean over draws), precordial loss, clean ECE.

## Demo (Step 7)
- `make_demo_samples.py` picks 4 test-fold records per class (20 total) on which `resnet` (corruption-trained, seed 0) and `lg_edl` (seed 0) both get all 5 labels right at the validation-tuned thresholds with 12 clean leads. If fewer qualify, the remaining slots are filled with the records with the most correct labels, flagged `strict_correct = False`. With the 2-epoch synthetic debug models no record qualifies, so all 20 are fallbacks; this must be re-run on the real models.
- `demo_samples/demo_config.json` holds the validation-fitted temperatures, F1 thresholds and the lg_edl U threshold, computed with the same functions as `evaluate.py`; the U threshold is cross-checked against `abstain_rate.csv` (matched: 0.6271).
- Lead quality in the demo has two sources: the lead switches (q = 1 on, 0 off; the default) or "Auto-detect quality" (the frozen SQA on the current signal). Presets: Drop V1-V6, Drop limb, Random 6 off, EMG on 3 leads (10 dB, the test SNR), Reset.
- EMG noise is generated on the z-scored signal and mapped back to mV with each lead's own mean and std, so the raw plot and the model input carry the same corruption. Disconnected leads are zero in both.
- The baseline shown is temperature-scaled, like in `evaluate.py`; lg_edl uses its Beta probabilities.
- Styling: theme pinned to light in `.streamlit/config.toml` (otherwise Streamlit follows the OS dark mode and the custom light panels lose contrast). Model hues match the paper figures; status red is used only for the abstain state; no green "correct" state, since the demo cannot know the model is right.

## Freeze review (Step 8)
- Approved by the user: per-record training corruption (D3), the 95th-percentile abstain threshold (D7), and the EDL loss formulation (D15). The EDL loss was later checked against the text of Sensoy et al. 2018 (see "After the real run").
- Added after the review: `--cache-dir` on every script that loads data (passed through by `run_all.py`); `evaluate.py --strict` (stop unless all 13 runs are finished); `run_all.py --only RUN ...` (run a subset of the run list, still logging `timing.csv`), used by `GUIDE.md` 9.4 for the first two real runs.
- `GUIDE.md` section 9 now starts with a synthetic smoke test on the Colab GPU (the laptop never executed the CUDA / mixed-precision paths), sends every output to explicit Drive folders, and builds the demo samples on Colab.
- All synthetic outputs (results, figures including the step-1 corruption figure, demo samples, checkpoints) were deleted before the real run; regenerate the corruption figure with `plot_corruptions.py --data ptbxl`.

## After the real run
- **EDL loss checked against Sensoy et al. 2018** (arXiv 1806.01768, read via ar5iv). `edl_loss` is their Eq. 4 (Bayes risk of cross-entropy, `sum_j y_j (psi(S) - psi(alpha_j))`) applied per class as a two-outcome Dirichlet, i.e. Beta(alpha_k, beta_k). The KL target matches their alpha-tilde = y + (1 - y) * alpha (y_k = 1 -> Beta(1, beta_k); y_k = 0 -> Beta(alpha_k, 1)), the closed-form KL to Beta(1, 1) matches their Dirichlet formula with K = 2, and the weight min(1, t / 10) matches their lambda_t. Numerically, the closed-form KL and expected BCE agree with direct integration under Beta(alpha, beta) to within 2e-11. Wording for the paper: Sensoy et al. used the squared-error variant (their Eq. 5) in their own experiments; we use the cross-entropy variant (Eq. 4), per class (Beta instead of Dirichlet, for multi-label).
- **timing.csv has two rows for `lg_soft_s0_c1`.** The second (15 Oct, `resumed = True`, 41.4 s) is a resume from a `_last.pt` that was not marked finished on Drive (most likely a write not flushed before the runtime ended). Resume restores the full training state, so the run is equivalent to an uninterrupted one; its final `_last.pt` shows 15 epochs. Total time for that run = sum of both rows.

## Models and evaluation (decided now, implemented later)
- The "under ~1M parameters" limit applies to the LeadGuard and attention encoders. The MLP baseline (12000-512-128-5, ~6.2M) is exempt because the spec fixes its size.
- XGBoost "missing" leads are detected from the signal (lead std < 1e-6 → features NaN), never from true severity.
- Temperature scaling: one temperature per sigmoid model, fitted on the validation fold.
- ECE: 15 bins over all 5 x N binary predictions, flattened.
- `utils.py` holds `set_seed`, `get_device`, and the fixed lead and class orders.
- Python: developed locally on 3.14 (the only version installed); Colab uses 3.12. Avoid version-specific features.
