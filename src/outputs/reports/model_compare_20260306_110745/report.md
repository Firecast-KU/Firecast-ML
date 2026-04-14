# Model Comparison Report

- Generated at: 2026-03-06T11:07:45
- Data file: `C:\Users\pds20\firecast\src\data\processed\weather_labeled.parquet`
- Date column: `date`
- Label column: `fire_label`
- Holdout days: `240` (cutoff: `2021-05-05`)
- Eval rows: `482` (positives: `23`)
- Threshold: `0.5`

## Metrics

| Metric | Old | New | Delta (New-Old) |
|---|---:|---:|---:|
| ROC-AUC | 0.537084 | 0.537084 | +0.000000 |
| PR-AUC | 0.050684 | 0.050684 | +0.000000 |
| Brier | 0.201438 | 0.201438 | +0.000000 |
| LogLoss | 0.589627 | 0.589627 | +0.000000 |
| Accuracy | 0.707469 | 0.707469 | +0.000000 |
| Precision | 0.053030 | 0.053030 | +0.000000 |
| Recall | 0.304348 | 0.304348 | +0.000000 |
| F1 | 0.090323 | 0.090323 | +0.000000 |
| PositiveRate | 0.273859 | 0.273859 | +0.000000 |
