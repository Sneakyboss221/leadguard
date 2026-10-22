# Example files for the demo upload

Four records from the PTB-XL test fold (fold 10) that are not among the 20 demo samples, picked as the first
human-validated records in the database whose only diagnostic statement is NORM, IMI (MI), CLBBB (CD) or LVH (HYP).
They were not selected on how the models classify them. `labels.json` gives each record's superclass, which the
demo shows as the reference diagnosis.

- `00057_lr.csv`: the 100 Hz record converted to CSV (one column per lead, mV), to show the CSV format.
- `01737_lr`, `00618_lr`, `03785_lr`: the original WFDB files (`.hea` + `.dat`).

Source: Wagner et al., PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3), PhysioNet,
https://doi.org/10.13026/kfzx-aw45, licensed under CC BY 4.0.
