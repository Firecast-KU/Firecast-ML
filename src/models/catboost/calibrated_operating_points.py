import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import ROOT
from src.models.common.metrics import classification_metrics, recall_at_fixed_far, recall_at_fixed_precision


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return {
        "threshold": float(threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "positive_rate": metrics["positive_rate"],
        "tp": metrics["confusion_matrix"]["tp"],
        "fp": metrics["confusion_matrix"]["fp"],
        "fn": metrics["confusion_matrix"]["fn"],
        "tn": metrics["confusion_matrix"]["tn"],
    }


def summarize_calibrated_operating_points(
    prediction_path: Path,
    method: str = "platt",
    baseline_threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    max_curve_points: int = 200,
    output_dir: Path | None = None,
) -> dict:
    pred_df = pd.read_parquet(prediction_path)
    if "method" not in pred_df.columns:
        raise KeyError(f"`method` column not found in {prediction_path}")

    method_df = pred_df[pred_df["method"] == method].copy()
    if method_df.empty:
        raise ValueError(f"No rows found for method={method} in {prediction_path}")

    y_true = method_df["fire_label"].astype(int).to_numpy()
    y_prob = method_df["pred_prob"].astype(float).to_numpy()

    default_point = _metrics_at_threshold(y_true, y_prob, threshold=baseline_threshold)
    precision_point = recall_at_fixed_precision(y_true, y_prob, min_precision=min_precision)
    far_point = recall_at_fixed_far(y_true, y_prob, max_far=max_far)

    rows = [default_point]
    if precision_point["threshold"] is not None:
        rows.append(_metrics_at_threshold(y_true, y_prob, threshold=float(precision_point["threshold"])))
    if far_point["threshold"] is not None:
        rows.append(_metrics_at_threshold(y_true, y_prob, threshold=float(far_point["threshold"])))

    threshold_grid = np.unique(np.round(y_prob, 6))
    if len(threshold_grid) > max_curve_points:
        quantiles = np.linspace(0.0, 1.0, max_curve_points)
        threshold_grid = np.unique(np.quantile(threshold_grid, quantiles))
    curve_rows = [_metrics_at_threshold(y_true, y_prob, threshold=float(thr)) for thr in threshold_grid]

    if output_dir is None:
        output_dir = ROOT / "outputs" / "reports" / f"catboost_{method}_operating_points"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).to_csv(
        output_dir / "recommended_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(curve_rows).sort_values("threshold").to_csv(
        output_dir / "threshold_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "prediction_path": str(prediction_path),
        "method": method,
        "rows": int(len(method_df)),
        "positive": int(method_df["fire_label"].sum()),
        "curve_points": int(len(threshold_grid)),
        "default_threshold": default_point,
        "recall_priority_threshold": precision_point,
        "false_alarm_limited_threshold": far_point,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Summarize operating thresholds for calibrated CatBoost probabilities.")
    parser.add_argument(
        "--prediction-path",
        type=str,
        default=str(ROOT / "outputs" / "reports" / "catboost_calibration_check" / "test_predictions.parquet"),
    )
    parser.add_argument("--method", type=str, choices=["base", "platt", "isotonic"], default="platt")
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--max-curve-points", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_calibrated_operating_points(
        prediction_path=Path(args.prediction_path),
        method=args.method,
        baseline_threshold=args.baseline_threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
        max_curve_points=args.max_curve_points,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
