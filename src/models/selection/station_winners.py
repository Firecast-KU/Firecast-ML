import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config.paths import CATBOOST_ARTIFACT_DIR, DATASET_DIR, LR_ARTIFACT_DIR, ROOT, TCN_ARTIFACT_DIR
from src.models.common.metrics import classification_metrics
from src.models.tcn.model import FireTCNClassifier


WIDE_DATA_FILE = DATASET_DIR / "catboost_dataset.parquet"
TCN_DATA_FILE = DATASET_DIR / "tcn_dataset.npz"
TCN_INDEX_FILE = DATASET_DIR / "tcn_index.parquet"

DEFAULT_MODELS = {
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


def _safe_station_metrics(
    station_df: pd.DataFrame,
    threshold: float,
    min_precision: float,
    max_far: float,
) -> dict:
    y_true = station_df["fire_label"].astype(int).to_numpy()
    y_prob = station_df["pred_prob"].astype(float).to_numpy()

    base = {
        "rows": int(len(station_df)),
        "positive": int(y_true.sum()),
        "positive_rate_true": float(y_true.mean()),
    }

    if len(np.unique(y_true)) < 2:
        base.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "brier": None,
                "log_loss": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "positive_rate": None,
                "false_alarm_rate": None,
                "threshold": threshold,
                "confusion_matrix": None,
                "recall_at_precision_threshold": None,
                "recall_at_precision_precision": None,
                "recall_at_precision": None,
                "recall_at_far_threshold": None,
                "recall_at_far_precision": None,
                "recall_at_far": None,
                "recall_at_far_far": None,
            }
        )
        return base

    metrics = classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        min_precision=min_precision,
        max_far=max_far,
    )
    base.update(
        {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "positive_rate": metrics["positive_rate"],
            "false_alarm_rate": metrics["false_alarm_rate"],
            "threshold": metrics["threshold"],
            "confusion_matrix": metrics["confusion_matrix"],
            "recall_at_precision_threshold": metrics["recall_at_precision"]["threshold"],
            "recall_at_precision_precision": metrics["recall_at_precision"]["precision"],
            "recall_at_precision": metrics["recall_at_precision"]["recall"],
            "recall_at_far_threshold": metrics["recall_at_far"]["threshold"],
            "recall_at_far_precision": metrics["recall_at_far"]["precision"],
            "recall_at_far": metrics["recall_at_far"]["recall"],
            "recall_at_far_far": metrics["recall_at_far"]["far"],
        }
    )
    return base


def _load_wide_holdout_predictions(model_name: str) -> pd.DataFrame:
    config = DEFAULT_MODELS[model_name]
    model_path = config["model_path"]
    meta_path = config["meta_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not WIDE_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {WIDE_DATA_FILE}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["features"]
    categorical_cols = set(meta.get("categorical_features", []))
    holdout_days = int(meta["split"]["holdout_days"])

    df = pd.read_parquet(WIDE_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["fire_label"] = pd.to_numeric(df["fire_label"], errors="coerce")
    df = df.dropna(subset=["date", "fire_label"]).copy()

    for col in feature_cols:
        if col in categorical_cols:
            if model_name == "catboost":
                df[col] = df[col].astype("string").fillna("__MISSING__")
            else:
                df[col] = df[col].astype("object")
                df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    holdout_df = df[df["date"] >= cutoff].copy().sort_values(["station_id", "date"]).reset_index(drop=True)

    model = joblib.load(model_path)
    holdout_df["pred_prob"] = model.predict_proba(holdout_df[feature_cols])[:, 1]
    holdout_df["model_name"] = model_name
    return holdout_df[["station_id", "date", "fire_label", "pred_prob", "model_name"]]


def _load_tcn_holdout_predictions() -> pd.DataFrame:
    config = DEFAULT_MODELS["tcn"]
    model_path = config["model_path"]
    meta_path = config["meta_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not TCN_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {TCN_DATA_FILE}")
    if not TCN_INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {TCN_INDEX_FILE}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    holdout_days = int(meta["split"]["holdout_days"])

    npz = np.load(TCN_DATA_FILE)
    index_df = pd.read_parquet(TCN_INDEX_FILE)
    index_df["date"] = pd.to_datetime(index_df["date"])
    cutoff = index_df["date"].max() - pd.Timedelta(days=holdout_days)
    test_mask = (index_df["date"] >= cutoff).to_numpy()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FireTCNClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_dyn = (npz["X_dyn"][test_mask] - checkpoint["dyn_mean"]) / checkpoint["dyn_std"]
    x_static = (npz["X_static"][test_mask] - checkpoint["static_mean"]) / checkpoint["static_std"]

    x_dyn_tensor = torch.tensor(x_dyn, dtype=torch.float32)
    x_static_tensor = torch.tensor(x_static, dtype=torch.float32)
    with torch.no_grad():
        pred_prob = torch.sigmoid(model(x_dyn_tensor, x_static_tensor)).numpy()

    holdout_df = index_df.loc[test_mask].copy().reset_index(drop=True)
    holdout_df["pred_prob"] = pred_prob
    holdout_df["model_name"] = "tcn"
    return holdout_df[["station_id", "date", "fire_label", "pred_prob", "model_name"]]


def collect_holdout_predictions() -> pd.DataFrame:
    frames = [
        _load_wide_holdout_predictions("lr"),
        _load_wide_holdout_predictions("catboost"),
        _load_tcn_holdout_predictions(),
    ]
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df["station_id"] = pred_df["station_id"].astype("string")
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["fire_label"] = pred_df["fire_label"].astype(int)
    pred_df = pred_df.sort_values(["station_id", "date", "model_name"]).reset_index(drop=True)
    return pred_df


def select_station_winners(
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    output_dir: Path | None = None,
) -> dict:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / "reports" / f"station_winners_{timestamp}"

    pred_df = collect_holdout_predictions()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_dir / "holdout_predictions.parquet", index=False)

    rows = []
    for (station_id, model_name), station_df in pred_df.groupby(["station_id", "model_name"], sort=True):
        metrics = _safe_station_metrics(
            station_df=station_df,
            threshold=threshold,
            min_precision=min_precision,
            max_far=max_far,
        )
        rows.append(
            {
                "station_id": station_id,
                "model_name": model_name,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "station_model_metrics.csv", index=False, encoding="utf-8-sig")

    sortable = metrics_df.copy()
    sortable["pr_auc_sort"] = sortable["pr_auc"].fillna(-np.inf)
    sortable["recall_at_precision_sort"] = sortable["recall_at_precision"].fillna(-np.inf)
    sortable["recall_at_far_sort"] = sortable["recall_at_far"].fillna(-np.inf)
    sortable["brier_sort"] = sortable["brier"].fillna(np.inf)

    winner_df = (
        sortable.sort_values(
            [
                "station_id",
                "pr_auc_sort",
                "recall_at_precision_sort",
                "recall_at_far_sort",
                "brier_sort",
                "model_name",
            ],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("station_id", as_index=False)
        .head(1)
        .drop(columns=["pr_auc_sort", "recall_at_precision_sort", "recall_at_far_sort", "brier_sort"])
        .reset_index(drop=True)
    )
    winner_df.to_csv(output_dir / "station_winners.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# Station Winner Selection",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Holdout rows: {len(pred_df) // pred_df['model_name'].nunique()}",
        f"- Models: {', '.join(sorted(pred_df['model_name'].unique().tolist()))}",
        f"- Primary metric: PR-AUC",
        f"- Secondary metric: Recall@Precision>={min_precision} then Recall@FAR<={max_far}",
        f"- Tertiary metric: Brier",
        "",
        "## Winners",
        "",
    ]
    for _, row in winner_df.iterrows():
        pr_text = f"{row['pr_auc']:.6f}" if pd.notna(row["pr_auc"]) else "NA"
        rap_text = f"{row['recall_at_precision']:.6f}" if pd.notna(row["recall_at_precision"]) else "NA"
        raf_text = f"{row['recall_at_far']:.6f}" if pd.notna(row["recall_at_far"]) else "NA"
        brier_text = f"{row['brier']:.6f}" if pd.notna(row["brier"]) else "NA"
        report_lines.append(
            f"- Station {row['station_id']}: `{row['model_name']}` "
            f"(PR-AUC={pr_text}, Recall@P={rap_text}, Recall@FAR={raf_text}, Brier={brier_text})"
        )

    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "winners": winner_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Select station-wise winner models from holdout predictions.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    select_station_winners(
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
