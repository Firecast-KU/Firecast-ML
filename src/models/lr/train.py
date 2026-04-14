import json
from argparse import ArgumentParser
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import DATASET_DIR, LR_ARTIFACT_DIR


LABEL = "fire_label"
DATA_FILE = DATASET_DIR / "catboost_dataset.parquet"

KEY_COLS = ["station_id", "date"]
STATIC_COLS = ["station_id", "stn_lat", "stn_lon"]
CALENDAR_COLS = ["month", "dayofyear", "doy_sin", "doy_cos"]

CURRENT_DAY_DYNAMIC_COLS = [
    "TA_day",
    "TA_dtr_day",
    "POP_day",
    "is_precip_day",
    "WD_sin_day",
    "WD_cos_day",
    "SKY_day",
    "obs_cnt_sel",
    "has_00_obs",
    "has_12_obs",
]

FEATURES = [
    "station_id",
    "stn_lat",
    "stn_lon",
    "month",
    "dayofyear",
    "doy_sin",
    "doy_cos",
    "TA_day_lag1",
    "TA_dtr_day_lag1",
    "POP_day_lag1",
    "WD_sin_day_lag1",
    "WD_cos_day_lag1",
    "is_precip_day_lag1",
    "has_00_obs_lag1",
    "has_12_obs_lag1",
    "obs_cnt_sel_lag1",
    "SKY_day_lag1",
    "TA_day_lag3",
    "TA_dtr_day_lag3",
    "POP_day_lag3",
    "WD_sin_day_lag3",
    "WD_cos_day_lag3",
    "is_precip_day_lag3",
    "has_00_obs_lag3",
    "has_12_obs_lag3",
    "obs_cnt_sel_lag3",
    "SKY_day_lag3",
    "TA_day_lag7",
    "TA_dtr_day_lag7",
    "POP_day_lag7",
    "WD_sin_day_lag7",
    "WD_cos_day_lag7",
    "is_precip_day_lag7",
    "has_00_obs_lag7",
    "has_12_obs_lag7",
    "obs_cnt_sel_lag7",
    "SKY_day_lag7",
    "TA_day_lag14",
    "TA_dtr_day_lag14",
    "POP_day_lag14",
    "WD_sin_day_lag14",
    "WD_cos_day_lag14",
    "is_precip_day_lag14",
    "has_00_obs_lag14",
    "has_12_obs_lag14",
    "obs_cnt_sel_lag14",
    "SKY_day_lag14",
    "TA_day_rollmean_3",
    "TA_dtr_day_rollmean_3",
    "POP_day_rollmean_3",
    "WD_sin_day_rollmean_3",
    "WD_cos_day_rollmean_3",
    "obs_cnt_sel_rollmean_3",
    "is_precip_day_rollsum_3",
    "has_00_obs_rollsum_3",
    "has_12_obs_rollsum_3",
    "SKY_day_mode_3",
    "TA_day_rollmean_7",
    "TA_dtr_day_rollmean_7",
    "POP_day_rollmean_7",
    "WD_sin_day_rollmean_7",
    "WD_cos_day_rollmean_7",
    "obs_cnt_sel_rollmean_7",
    "is_precip_day_rollsum_7",
    "has_00_obs_rollsum_7",
    "has_12_obs_rollsum_7",
    "SKY_day_mode_7",
    "TA_day_rollmean_14",
    "TA_dtr_day_rollmean_14",
    "POP_day_rollmean_14",
    "WD_sin_day_rollmean_14",
    "WD_cos_day_rollmean_14",
    "obs_cnt_sel_rollmean_14",
    "is_precip_day_rollsum_14",
    "has_00_obs_rollsum_14",
    "has_12_obs_rollsum_14",
    "SKY_day_mode_14",
    "dry_spell_days",
]

CATEGORICAL_FEATURES = [
    "station_id",
    "month",
    "SKY_day_lag1",
    "SKY_day_lag3",
    "SKY_day_lag7",
    "SKY_day_lag14",
    "SKY_day_mode_3",
    "SKY_day_mode_7",
    "SKY_day_mode_14",
]

MODEL_NAME = "base_lr.joblib"
META_NAME = "base_lr_meta.json"


def time_split_holdout(df: pd.DataFrame, holdout_days: int = 240):
    df = df.sort_values(["date", "station_id"]).reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    return train_df, test_df, cutoff


def _build_pipeline() -> Pipeline:
    numeric_features = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def _load_training_frame() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )

    df = pd.read_parquet(DATA_FILE)
    required_cols = FEATURES + [LABEL] + KEY_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in {DATA_FILE.name}: {missing_cols}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df.dropna(subset=["date", LABEL]).copy()

    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def train_and_save(holdout_days: int = 240):
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

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
        raise ValueError("Training labels have only one class. Cannot train LogisticRegression.")
    if y_test.nunique() < 2:
        raise ValueError(
            "Test labels have only one class for the selected holdout window "
            f"(holdout_days={holdout_days}, test_rows={len(test_df)}, positives={int(y_test.sum())})."
        )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    LR_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = LR_ARTIFACT_DIR / MODEL_NAME
    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_FILE),
        "feature_table": "catboost_dataset.parquet",
        "label": LABEL,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_current_day_features": CURRENT_DAY_DYNAMIC_COLS,
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
        "model_type": "Pipeline(LogisticRegression)",
        "model_params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "quick_metrics_on_holdout": {
            "roc_auc": roc,
            "pr_auc": pr,
            "brier": brier,
            "log_loss": ll,
        },
    }

    meta_path = LR_ARTIFACT_DIR / META_NAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Dataset     : {DATA_FILE}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Train rows / positives: {len(train_df)} / {int(y_train.sum())}")
    print(f"Test rows  / positives: {len(test_df)} / {int(y_test.sum())}")
    print(f"Quick ROC-AUC: {roc}")
    print(f"Quick PR-AUC : {pr}")
    print(f"Quick Brier  : {brier}")
    print(f"Quick LogLoss: {ll}")

    return pipeline


def parse_args():
    parser = ArgumentParser(description="Train LR baseline on catboost_dataset time-holdout.")
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
