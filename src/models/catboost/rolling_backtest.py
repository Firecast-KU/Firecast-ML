import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.paths import ROOT
from src.models.catboost.train import _load_training_frame, evaluate_split
from src.models.common.metrics import classification_metrics
from src.models.lr.train import FEATURES, LABEL


def _year_windows(df: pd.DataFrame, start_year: int | None, end_year: int | None) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    years = sorted(df["date"].dt.year.unique().tolist())
    if start_year is not None:
        years = [year for year in years if year >= start_year]
    if end_year is not None:
        years = [year for year in years if year <= end_year]

    windows = []
    for year in years:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        windows.append((year, test_start, test_end))
    return windows


def run_rolling_backtest(
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    start_year: int | None = 2020,
    end_year: int | None = 2021,
    min_train_rows: int = 365,
    output_dir: Path | None = None,
) -> dict:
    df = _load_training_frame()
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / "reports" / f"catboost_rolling_backtest_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pred_frames = []
    station_rows = []

    for year, test_start, test_end in _year_windows(df, start_year=start_year, end_year=end_year):
        train_df = df[df["date"] < test_start].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train_df) < min_train_rows or test_df.empty:
            continue
        if train_df[LABEL].nunique() < 2 or test_df[LABEL].nunique() < 2:
            continue

        result = evaluate_split(
            train_df=train_df,
            test_df=test_df,
            threshold=threshold,
            min_precision=min_precision,
            max_far=max_far,
        )
        metrics = result["metrics"]

        fold_rows.append(
            {
                "test_year": year,
                "train_start_date": str(train_df["date"].min().date()),
                "train_end_date": str(train_df["date"].max().date()),
                "test_start_date": str(test_df["date"].min().date()),
                "test_end_date": str(test_df["date"].max().date()),
                "train_rows": result["train_rows"],
                "train_positive": result["train_positive"],
                "test_rows": result["test_rows"],
                "test_positive": result["test_positive"],
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

        pred_df = result["test_pred"].copy()
        pred_df["test_year"] = year
        pred_frames.append(pred_df)

        for station_id, station_df in pred_df.groupby("station_id", sort=True):
            if station_df[LABEL].nunique() < 2:
                continue
            station_metrics = classification_metrics(
                y_true=station_df[LABEL].astype(int).to_numpy(),
                y_prob=station_df["pred_prob"].astype(float).to_numpy(),
                threshold=threshold,
                min_precision=min_precision,
                max_far=max_far,
            )
            station_rows.append(
                {
                    "test_year": year,
                    "station_id": station_id,
                    "rows": int(len(station_df)),
                    "positive": int(station_df[LABEL].sum()),
                    "roc_auc": station_metrics["roc_auc"],
                    "pr_auc": station_metrics["pr_auc"],
                    "brier": station_metrics["brier"],
                    "log_loss": station_metrics["log_loss"],
                    "precision": station_metrics["precision"],
                    "recall": station_metrics["recall"],
                    "f1": station_metrics["f1"],
                    "false_alarm_rate": station_metrics["false_alarm_rate"],
                    "positive_rate": station_metrics["positive_rate"],
                }
            )

    if not fold_rows:
        raise ValueError("No valid rolling windows were generated. Check year range and label availability.")

    fold_df = pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True)
    fold_df.to_csv(output_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")

    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "holdout_predictions.parquet", index=False)

    if station_rows:
        pd.DataFrame(station_rows).sort_values(["test_year", "station_id"]).to_csv(
            output_dir / "station_year_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "output_dir": str(output_dir),
        "feature_count": len(FEATURES),
        "folds": fold_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Run year-wise rolling backtest for CatBoost.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--min-train-rows", type=int, default=365)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rolling_backtest(
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_rows=args.min_train_rows,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
