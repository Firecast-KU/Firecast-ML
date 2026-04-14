import json

import joblib
import pandas as pd

from src.config.paths import DATASET_DIR, LR_ARTIFACT_DIR


MODEL_NAME = "base_lr.joblib"
META_NAME = "base_lr_meta.json"
DATA_FILE = DATASET_DIR / "catboost_dataset.parquet"


def risk_level_from_prob(p: float) -> str:
    if p <= 0.4:
        return "LOW"
    if p <= 0.6:
        return "MODERATE"
    if p <= 0.8:
        return "HIGH"
    return "EXTREME"


def _load_meta() -> dict:
    meta_path = LR_ARTIFACT_DIR / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found: {meta_path}\n"
            "Run `python -m src.models.lr.train` first."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def predict_for_date(target_date: str, save: bool = False):
    model_path = LR_ARTIFACT_DIR / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run `python -m src.models.lr.train` first."
        )
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )

    meta = _load_meta()
    feature_cols = meta["features"]
    categorical_cols = set(meta.get("categorical_features", []))

    model = joblib.load(model_path)

    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    for col in feature_cols:
        if col in categorical_cols:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_dt = pd.to_datetime(target_date)
    day_df = df[df["date"].dt.date == target_dt.date()].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date={target_date} in {DATA_FILE.name}")

    X = day_df[feature_cols]
    day_df["base_prob"] = model.predict_proba(X)[:, 1]
    day_df["risk_level"] = day_df["base_prob"].apply(risk_level_from_prob)

    result_cols = ["station_id", "date", "base_prob", "risk_level", "stn_lat", "stn_lon"]
    result = day_df[[c for c in result_cols if c in day_df.columns]].sort_values(["station_id"]).reset_index(drop=True)

    if save:
        out_path = DATASET_DIR / f"base_lr_predictions_{target_dt.date()}.parquet"
        result.to_parquet(out_path, index=False)
        print(f"Saved predictions: {out_path}")

    return result


if __name__ == "__main__":
    print(predict_for_date("2021-03-25", save=False).head())
