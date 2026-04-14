import json
from argparse import ArgumentParser
from datetime import datetime

import joblib
import pandas as pd

from src.config.paths import MODEL_DIR
from src.models.metrics_utils import classification_metrics
from src.models.train_base_model import CATEGORICAL_FEATURES, DATA_FILE, FEATURES, LABEL, time_split_holdout


MODEL_NAME = "catboost_v1.joblib"
META_NAME = "catboost_v1_meta.json"


def _import_catboost():
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "catboost is not installed. Install it with `pip install catboost` before training."
        ) from exc
    return CatBoostClassifier


def _load_training_frame() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )

    df = pd.read_parquet(DATA_FILE)
    required_cols = FEATURES + [LABEL, "date"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in {DATA_FILE.name}: {missing_cols}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df.dropna(subset=["date", LABEL]).copy()

    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("string").fillna("__MISSING__")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def train_and_save(
    holdout_days: int = 240,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
):
    CatBoostClassifier = _import_catboost()

    df = _load_training_frame()
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

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
        auto_class_weights=None,
        scale_pos_weight=scale_pos_weight,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(
        y_true=y_test.to_numpy(),
        y_prob=y_prob,
        threshold=threshold,
        min_precision=min_precision,
        max_far=max_far,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_NAME
    meta_path = MODEL_DIR / META_NAME
    joblib.dump(model, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_FILE),
        "feature_table": "catboost_dataset.parquet",
        "label": LABEL,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_type": "CatBoostClassifier",
        "params": {
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": 42,
            "scale_pos_weight": scale_pos_weight,
        },
        "split": {
            "type": "time_holdout",
            "holdout_days": holdout_days,
            "cutoff": str(cutoff.date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "train_start_date": str(train_df["date"].min().date()),
            "train_end_date": str(train_df["date"].max().date()),
            "test_start_date": str(test_df["date"].min().date()),
            "test_end_date": str(test_df["date"].max().date()),
        },
        "holdout_metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Dataset     : {DATA_FILE}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Train rows / positives: {len(train_df)} / {int(y_train.sum())}")
    print(f"Test rows  / positives: {len(test_df)} / {int(y_test.sum())}")
    print(f"Holdout ROC-AUC: {metrics['roc_auc']}")
    print(f"Holdout PR-AUC : {metrics['pr_auc']}")
    print(f"Holdout Brier  : {metrics['brier']}")
    print(f"Holdout LogLoss: {metrics['log_loss']}")
    return model, meta


def parse_args():
    parser = ArgumentParser(description="Train CatBoost model on catboost_dataset time-holdout.")
    parser.add_argument("--holdout-days", type=int, default=240)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        holdout_days=args.holdout_days,
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
    )
