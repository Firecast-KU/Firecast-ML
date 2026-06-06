import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config.paths import CATBOOST_ARTIFACT_DIR, DATASET_DIR, LR_ARTIFACT_DIR, OUTPUT_DIR, TCN_ARTIFACT_DIR
from src.models.tcn.model import FireTCNClassifier


WIDE_DATA_FILE = DATASET_DIR / "catboost_dataset.parquet"
TCN_DATA_FILE = DATASET_DIR / "tcn_dataset.npz"
TCN_INDEX_FILE = DATASET_DIR / "tcn_index.parquet"
STATION_DAY_BASE_FILE = DATASET_DIR / "station_day_base.parquet"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

MODEL_CONFIG = {
    "lr": {
        "model_path": LR_ARTIFACT_DIR / "base_lr.joblib",
        "meta_path": LR_ARTIFACT_DIR / "base_lr_meta.json",
        "type": "wide",
    },
    "catboost": {
        "model_path": CATBOOST_ARTIFACT_DIR / "catboost_v1.joblib",
        "meta_path": CATBOOST_ARTIFACT_DIR / "catboost_v1_meta.json",
        "type": "wide",
    },
    "tcn": {
        "model_path": TCN_ARTIFACT_DIR / "tcn_v1.pt",
        "meta_path": TCN_ARTIFACT_DIR / "tcn_v1_meta.json",
        "type": "tcn",
    },
}


def risk_level_from_prob(p: float) -> str:
    if p <= 0.4:
        return "LOW"
    if p <= 0.6:
        return "MODERATE"
    if p <= 0.8:
        return "HIGH"
    return "EXTREME"


def _resolve_operating_threshold(meta: dict, default_threshold: float = 0.5) -> float:
    configured = meta.get("operating_threshold")
    if configured is not None:
        return float(configured)

    recall_at_far = meta.get("holdout_metrics", {}).get("recall_at_far", {})
    threshold = recall_at_far.get("threshold")
    if threshold is not None:
        return float(threshold)

    threshold = meta.get("holdout_metrics", {}).get("threshold")
    if threshold is not None:
        return float(threshold)

    return float(default_threshold)


def _load_station_metadata() -> pd.DataFrame:
    if not STATION_DAY_BASE_FILE.exists():
        raise FileNotFoundError(
            f"Station-day base file not found: {STATION_DAY_BASE_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )

    base_df = pd.read_parquet(STATION_DAY_BASE_FILE, columns=["station_id", "stn_lat", "stn_lon"])
    base_df["station_id"] = base_df["station_id"].astype("string")
    base_df = base_df.drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    return base_df


def _normalize_output(df: pd.DataFrame, model_name: str, alert_threshold: float) -> pd.DataFrame:
    result = df.copy()
    result["model_name"] = model_name
    result["station_id"] = result["station_id"].astype("string")
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["pred_prob"] = pd.to_numeric(result["pred_prob"], errors="coerce")
    result["risk_level"] = result["pred_prob"].apply(risk_level_from_prob)
    result["alert_threshold"] = float(alert_threshold)
    result["is_alert"] = (result["pred_prob"] >= result["alert_threshold"]).astype(int)

    cols = ["model_name", "station_id", "date", "pred_prob", "risk_level", "alert_threshold", "is_alert", "stn_lat", "stn_lon"]
    return result[cols].sort_values(["date", "station_id"]).reset_index(drop=True)


def _predict_wide_model(model_name: str, target_date: str) -> pd.DataFrame:
    config = MODEL_CONFIG[model_name]
    model_path = config["model_path"]
    meta_path = config["meta_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not WIDE_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {WIDE_DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["features"]
    categorical_cols = set(meta.get("categorical_features", []))
    alert_threshold = _resolve_operating_threshold(meta)

    df = pd.read_parquet(WIDE_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    for col in feature_cols:
        if col in categorical_cols:
            if model_name == "catboost":
                df[col] = df[col].astype("string").fillna("__MISSING__")
            else:
                df[col] = df[col].astype("object")
                df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_dt = pd.to_datetime(target_date).normalize()
    day_df = df[df["date"] == target_dt].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date={target_date} in {WIDE_DATA_FILE.name}")

    model = joblib.load(model_path)
    day_df["pred_prob"] = model.predict_proba(day_df[feature_cols])[:, 1]
    return _normalize_output(day_df, model_name=model_name, alert_threshold=alert_threshold)


def _predict_tcn_model(target_date: str) -> pd.DataFrame:
    config = MODEL_CONFIG["tcn"]
    model_path = config["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not TCN_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {TCN_DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )
    if not TCN_INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {TCN_INDEX_FILE}")

    target_dt = pd.to_datetime(target_date).normalize()
    npz = np.load(TCN_DATA_FILE)
    index_df = pd.read_parquet(TCN_INDEX_FILE)
    index_df["date"] = pd.to_datetime(index_df["date"]).dt.normalize()
    date_mask = (index_df["date"] == target_dt).to_numpy()
    if not date_mask.any():
        raise ValueError(f"No TCN sequence rows found for date={target_date} in {TCN_INDEX_FILE.name}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FireTCNClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_dyn = (npz["X_dyn"][date_mask] - checkpoint["dyn_mean"]) / checkpoint["dyn_std"]
    x_static = (npz["X_static"][date_mask] - checkpoint["static_mean"]) / checkpoint["static_std"]

    with torch.no_grad():
        pred_prob = torch.sigmoid(
            model(
                torch.tensor(x_dyn, dtype=torch.float32),
                torch.tensor(x_static, dtype=torch.float32),
            )
        ).numpy()

    station_meta = _load_station_metadata()
    result = index_df.loc[date_mask, ["station_id", "date"]].copy().reset_index(drop=True)
    result["station_id"] = result["station_id"].astype("string")
    result["pred_prob"] = pred_prob
    result = result.merge(station_meta, on="station_id", how="left", validate="many_to_one")
    return _normalize_output(result, model_name="tcn", alert_threshold=0.5)


def predict_for_date(target_date: str, model_name: str = "all") -> pd.DataFrame:
    if model_name == "all":
        frames = [
            _predict_wide_model("lr", target_date=target_date),
            _predict_wide_model("catboost", target_date=target_date),
            _predict_tcn_model(target_date=target_date),
        ]
        return pd.concat(frames, ignore_index=True).sort_values(["model_name", "station_id"]).reset_index(drop=True)

    if model_name in {"lr", "catboost"}:
        return _predict_wide_model(model_name=model_name, target_date=target_date)

    if model_name == "tcn":
        return _predict_tcn_model(target_date=target_date)

    raise ValueError(f"Unsupported model_name: {model_name}")


def _save_predictions(df: pd.DataFrame, target_date: str, model_name: str, fmt: str) -> Path:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    file_stem = f"{model_name}_predictions_{pd.to_datetime(target_date).date()}"

    if fmt == "parquet":
        out_path = PREDICTION_DIR / f"{file_stem}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    if fmt == "csv":
        out_path = PREDICTION_DIR / f"{file_stem}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    raise ValueError(f"Unsupported save format: {fmt}")


def parse_args():
    parser = ArgumentParser(description="Unified daily prediction for LR, CatBoost, and TCN models.")
    parser.add_argument("--target-date", type=str, required=True, help="Prediction date in YYYY-MM-DD format")
    parser.add_argument("--model", type=str, choices=["all", "lr", "catboost", "tcn"], default="all")
    parser.add_argument("--save", action="store_true", help="Save predictions to outputs/predictions")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_df = predict_for_date(target_date=args.target_date, model_name=args.model)

    if args.save:
        out_path = _save_predictions(pred_df, target_date=args.target_date, model_name=args.model, fmt=args.format)
    else:
        out_path = None

    summary = {
        "target_date": str(pd.to_datetime(args.target_date).date()),
        "model": args.model,
        "rows": int(len(pred_df)),
        "schema": list(pred_df.columns),
        "predictions": pred_df.to_dict(orient="records"),
        "saved_path": str(out_path) if out_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
