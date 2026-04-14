# Station Winner Selection

- Generated at: 2026-04-14T11:09:26
- Holdout rows: 482
- Models: catboost, lr, tcn
- Primary metric: PR-AUC
- Secondary metric: Recall@Precision>=0.3 then Recall@FAR<=0.05
- Tertiary metric: Brier

## Winners

- Station 104: `catboost` (PR-AUC=0.308267, Recall@P=0.642857, Recall@FAR=0.357143, Brier=0.198332)
- Station 105: `tcn` (PR-AUC=0.152319, Recall@P=0.333333, Recall@FAR=0.333333, Brier=0.200569)