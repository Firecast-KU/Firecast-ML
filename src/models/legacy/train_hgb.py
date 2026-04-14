import json
from argparse import ArgumentParser
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.config.paths import LEGACY_ARTIFACT_DIR, PROC_DIR


FEATURES = [
    "TA",
    # "TA_dtr",  # temporarily excluded (diurnal temperature range)
    "POP",
    "is_precip",
    "WD_sin",
    "WD_cos",
    "SKY",
]
LABEL = "fire_label"

MODEL_NAME = "hgb_v1.joblib"
META_NAME = "hgb_v1_meta.json"


def time_split_holdout(df: pd.DataFrame, holdout_days: int = 240):
    df = df.sort_values(["date"]).reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    return train_df, test_df, cutoff


def train_and_save(holdout_days: int = 240):
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])

    required_cols = FEATURES + [LABEL, "date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in weather_labeled.parquet: {missing_cols}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")

    df = df.dropna(subset=FEATURES + [LABEL])
    if df.empty:
        raise ValueError("No rows available after dropping missing feature/label values.")

    train_df, test_df, cutoff = time_split_holdout(df, holdout_days=holdout_days)
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split (holdout_days={holdout_days}): train_rows={len(train_df)}, test_rows={len(test_df)}"
        )

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL].astype(int)
    X_test = test_df[FEATURES]
    y_test = test_df[LABEL].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training labels have only one class.")
    if y_test.nunique() < 2:
        raise ValueError("Test labels have only one class.")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    LEGACY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = LEGACY_ARTIFACT_DIR / MODEL_NAME
    meta_path = LEGACY_ARTIFACT_DIR / META_NAME

    joblib.dump(model, model_path)
    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(PROC_DIR / "weather_labeled.parquet"),
        "features": FEATURES,
        "label": LABEL,
        "model_type": "HistGradientBoostingClassifier",
        "params": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
        "split": {
            "type": "time_holdout",
            "holdout_days": holdout_days,
            "cutoff": str(cutoff.date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
        },
        "quick_metrics_on_holdout": {
            "roc_auc": roc,
            "pr_auc": pr,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Train rows / positives: {len(train_df)} / {int(y_train.sum())}")
    print(f"Test rows  / positives: {len(test_df)} / {int(y_test.sum())}")
    print(f"Quick ROC-AUC: {roc}")
    print(f"Quick PR-AUC : {pr}")


def parse_args():
    parser = ArgumentParser(description="Train HGB model with time holdout.")
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=240,
        help="Number of most-recent days used as test holdout (default: 240)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(holdout_days=args.holdout_days)
