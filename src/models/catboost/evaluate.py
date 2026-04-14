import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd

from src.config.paths import CATBOOST_ARTIFACT_DIR
from src.models.common.metrics import classification_metrics
from src.models.lr.train import DATA_FILE, LABEL, time_split_holdout


def evaluate_saved_model(
    model_path: Path,
    meta_path: Path,
    holdout_days: int | None = None,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["features"]
    categorical_cols = set(meta.get("categorical_features", []))
    split_holdout = meta.get("split", {}).get("holdout_days", 240)
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
    y_true = test_df[LABEL].astype(int).to_numpy()
    y_prob = model.predict_proba(test_df[feature_cols])[:, 1]

    metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        min_precision=min_precision,
        max_far=max_far,
    )
    summary = {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "data_file": str(DATA_FILE),
        "cutoff_date": str(cutoff.date()),
        "eval_rows": int(len(test_df)),
        "eval_positive": int(y_true.sum()),
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Evaluate saved CatBoost model on catboost_dataset holdout.")
    parser.add_argument("--model-path", type=str, default=str(CATBOOST_ARTIFACT_DIR / "catboost_v1.joblib"))
    parser.add_argument("--meta-path", type=str, default=str(CATBOOST_ARTIFACT_DIR / "catboost_v1_meta.json"))
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_saved_model(
        model_path=Path(args.model_path),
        meta_path=Path(args.meta_path),
        holdout_days=args.holdout_days,
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
    )
