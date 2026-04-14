# Model Comparison Report

- Generated at: 2026-03-06T17:08:46
- Data file: `C:\Users\pds20\firecast\src\data\processed\weather_labeled.parquet`
- Date column: `date`
- Label column: `fire_label`
- Holdout days: `240` (cutoff: `2021-05-05`)
- Eval rows: `482` (positives: `23`)
- Threshold: `0.5`

## Metrics

| Metric | Old | New | Delta (New-Old) |
|---|---:|---:|---:|
| ROC-AUC | 0.537084 | 0.512267 | -0.024818 |
| PR-AUC | 0.050684 | 0.053757 | +0.003072 |
| Brier | 0.201438 | 0.230687 | +0.029249 |
| LogLoss | 0.589627 | 0.755187 | +0.165560 |
| Accuracy | 0.707469 | 0.692946 | -0.014523 |
| Precision | 0.053030 | 0.050360 | -0.002671 |
| Recall | 0.304348 | 0.304348 | +0.000000 |
| F1 | 0.090323 | 0.086420 | -0.003903 |
| PositiveRate | 0.273859 | 0.288382 | +0.014523 |
