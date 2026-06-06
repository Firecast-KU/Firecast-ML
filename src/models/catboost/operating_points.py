import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config.paths import CATBOOST_ARTIFACT_DIR, ROOT
from src.models.common.metrics import classification_metrics, recall_at_fixed_far, recall_at_fixed_precision
from src.models.lr.train import DATA_FILE, LABEL, time_split_holdout


def _load_holdout_predictions(
    model_path: Path,
    meta_path: Path,
    holdout_days: int | None = None,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["features"]
    categorical_cols = set(meta.get("categorical_features", []))
    split_holdout = meta.get("split", {}).get("holdout_days", 365)
    holdout_days = split_holdout if holdout_days is None else holdout_days

    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df.dropna(subset=["date", LABEL]).copy()

    for col in feature_cols:
        if col in categorical_cols:
            df[col] = df[col].astype("string").fillna("__MISSING__")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _, test_df, cutoff = time_split_holdout(df, holdout_days=holdout_days)
    model = joblib.load(model_path)
    pred_df = test_df[["station_id", "date", LABEL]].copy()
    pred_df["pred_prob"] = model.predict_proba(test_df[feature_cols])[:, 1]
    pred_df["cutoff"] = cutoff
    return pred_df.reset_index(drop=True)


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


def summarize_operating_points(
    model_path: Path,
    meta_path: Path,
    holdout_days: int | None = None,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    output_dir: Path | None = None,
) -> dict:
    if output_dir is None:
        output_dir = ROOT / "outputs" / "reports" / "catboost_operating_points"

    pred_df = _load_holdout_predictions(model_path=model_path, meta_path=meta_path, holdout_days=holdout_days)
    y_true = pred_df[LABEL].astype(int).to_numpy()
    y_prob = pred_df["pred_prob"].astype(float).to_numpy()

    default_point = _metrics_at_threshold(y_true, y_prob, threshold=0.5)
    precision_point = recall_at_fixed_precision(y_true, y_prob, min_precision=min_precision)
    far_point = recall_at_fixed_far(y_true, y_prob, max_far=max_far)

    rows = [default_point]
    if precision_point["threshold"] is not None:
        rows.append(_metrics_at_threshold(y_true, y_prob, threshold=float(precision_point["threshold"])))
    if far_point["threshold"] is not None:
        rows.append(_metrics_at_threshold(y_true, y_prob, threshold=float(far_point["threshold"])))

    threshold_grid = np.unique(np.round(y_prob, 6))
    curve_rows = [_metrics_at_threshold(y_true, y_prob, threshold=float(thr)) for thr in threshold_grid]

    summary = {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "data_file": str(DATA_FILE),
        "holdout_days": int(holdout_days) if holdout_days is not None else None,
        "cutoff_date": str(pd.to_datetime(pred_df["cutoff"].iloc[0]).date()),
        "rows": int(len(pred_df)),
        "positive": int(pred_df[LABEL].sum()),
        "default_threshold": default_point,
        "recall_priority_threshold": precision_point,
        "false_alarm_limited_threshold": far_point,
    }

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
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Summarize CatBoost operating thresholds on holdout predictions.")
    parser.add_argument("--model-path", type=str, default=str(CATBOOST_ARTIFACT_DIR / "catboost_v1.joblib"))
    parser.add_argument("--meta-path", type=str, default=str(CATBOOST_ARTIFACT_DIR / "catboost_v1_meta.json"))
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_operating_points(
        model_path=Path(args.model_path),
        meta_path=Path(args.meta_path),
        holdout_days=args.holdout_days,
        min_precision=args.min_precision,
        max_far=args.max_far,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
