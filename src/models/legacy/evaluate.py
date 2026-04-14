import json
from argparse import ArgumentParser
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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

MODEL_NAME = "legacy_base_lr.joblib"
META_NAME = "legacy_base_lr_meta.json"


def time_split_holdout(df: pd.DataFrame, holdout_days: int = 90):
    """Time-based holdout split using the latest N days as test."""
    df = df.sort_values(["date"]).reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    return train_df, test_df, cutoff


def train_and_save(holdout_days: int = 90):
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])

    required_cols = FEATURES + [LABEL, "date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in weather_labeled.parquet: {missing_cols}")

    # Numeric safety for model input.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [LABEL])
    if df.empty:
        raise ValueError("No rows available after dropping missing feature/label values.")

    train_df, test_df, cutoff = time_split_holdout(df, holdout_days=holdout_days)
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split (holdout_days={holdout_days}): train_rows={len(train_df)}, test_rows={len(test_df)}"
        )

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]
    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]
    if y_train.nunique() < 2:
        raise ValueError("Training labels have only one class. Cannot train LogisticRegression.")
    if y_test.nunique() < 2:
        raise ValueError(
            "Test labels have only one class for the selected holdout window "
            f"(holdout_days={holdout_days}, test_rows={len(test_df)}, positives={int(y_test.sum())}). "
            "Choose a different holdout window with both classes."
        )

    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    LEGACY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = LEGACY_ARTIFACT_DIR / MODEL_NAME
    joblib.dump(model, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(PROC_DIR / "weather_labeled.parquet"),
        "features": FEATURES,
        "label": LABEL,
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

    meta_path = LEGACY_ARTIFACT_DIR / META_NAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Train rows / positives: {len(train_df)} / {int(y_train.sum())}")
    print(f"Test rows  / positives: {len(test_df)} / {int(y_test.sum())}")
    print(f"Quick ROC-AUC: {roc}")
    print(f"Quick PR-AUC : {pr}")

    return model


def parse_args():
    parser = ArgumentParser(description="Train and evaluate base LR model with time holdout.")
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=90,
        help="Number of most-recent days used as test holdout (default: 90)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # train_and_save(holdout_days=args.holdout_days)
    train_and_save(holdout_days=240)
