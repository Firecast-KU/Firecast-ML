import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.paths import ROOT
from src.models.common.metrics import classification_metrics
from src.models.selection.station_winners import collect_holdout_predictions


def benchmark_models(
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    output_dir: Path | None = None,
) -> dict:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / "reports" / f"model_benchmark_{timestamp}"

    pred_df = collect_holdout_predictions()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_dir / "holdout_predictions.parquet", index=False)

    rows = []
    for model_name, model_df in pred_df.groupby("model_name", sort=True):
        metrics = classification_metrics(
            y_true=model_df["fire_label"].astype(int).to_numpy(),
            y_prob=model_df["pred_prob"].astype(float).to_numpy(),
            threshold=threshold,
            min_precision=min_precision,
            max_far=max_far,
        )
        rows.append(
            {
                "model_name": model_name,
                "rows": int(len(model_df)),
                "positive": int(model_df["fire_label"].sum()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_alarm_rate": metrics["false_alarm_rate"],
                "positive_rate": metrics["positive_rate"],
                "recall_at_precision": metrics["recall_at_precision"]["recall"],
                "recall_at_precision_threshold": metrics["recall_at_precision"]["threshold"],
                "recall_at_far": metrics["recall_at_far"]["recall"],
                "recall_at_far_threshold": metrics["recall_at_far"]["threshold"],
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values(["pr_auc", "recall_at_precision", "recall_at_far", "brier"], ascending=[False, False, False, True])
    metrics_df.to_csv(output_dir / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# Model Benchmark",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Threshold: {threshold}",
        f"- Fixed precision target: {min_precision}",
        f"- Fixed FAR target: {max_far}",
        "",
        "## Ranking",
        "",
    ]
    for _, row in metrics_df.iterrows():
        report_lines.append(
            f"- `{row['model_name']}`: PR-AUC={row['pr_auc']:.6f}, "
            f"Recall@P={row['recall_at_precision'] if pd.notna(row['recall_at_precision']) else 'NA'}, "
            f"Recall@FAR={row['recall_at_far'] if pd.notna(row['recall_at_far']) else 'NA'}, "
            f"Brier={row['brier']:.6f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "ranking": metrics_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Benchmark LR/CatBoost/TCN on the same holdout predictions.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_models(
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
