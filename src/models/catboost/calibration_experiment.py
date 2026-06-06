import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config.paths import ROOT
from src.models.catboost.train import _load_training_frame, build_model
from src.models.common.metrics import classification_metrics
from src.models.lr.train import FEATURES, LABEL


def _split_train_calibration_test(
    df: pd.DataFrame,
    holdout_days: int,
    calibration_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    ordered = df.sort_values(["date", "station_id"]).reset_index(drop=True)
    test_cutoff = ordered["date"].max() - pd.Timedelta(days=holdout_days)
    calibration_cutoff = test_cutoff - pd.Timedelta(days=calibration_days)

    train_df = ordered[ordered["date"] < calibration_cutoff].copy()
    calibration_df = ordered[(ordered["date"] >= calibration_cutoff) & (ordered["date"] < test_cutoff)].copy()
    test_df = ordered[ordered["date"] >= test_cutoff].copy()
    return train_df, calibration_df, test_df, calibration_cutoff, test_cutoff


def _clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)


def _fit_platt_scaler(calibration_prob: np.ndarray, y_calibration: np.ndarray) -> LogisticRegression:
    logit_x = np.log(_clip_prob(calibration_prob) / (1.0 - _clip_prob(calibration_prob))).reshape(-1, 1)
    scaler = LogisticRegression(random_state=42)
    scaler.fit(logit_x, y_calibration)
    return scaler


def _apply_platt_scaler(scaler: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    logit_x = np.log(_clip_prob(prob) / (1.0 - _clip_prob(prob))).reshape(-1, 1)
    return scaler.predict_proba(logit_x)[:, 1]


def _fit_isotonic_scaler(calibration_prob: np.ndarray, y_calibration: np.ndarray) -> IsotonicRegression:
    scaler = IsotonicRegression(out_of_bounds="clip")
    scaler.fit(calibration_prob, y_calibration)
    return scaler


def _build_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    df = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob.astype(float)})
    df["prob_bin"] = pd.cut(df["y_prob"], bins=bins, include_lowest=True, right=True)

    rows = []
    for prob_bin, bdf in df.groupby("prob_bin", observed=False):
        if len(bdf) == 0:
            continue
        rows.append(
            {
                "method": method_name,
                "prob_bin": str(prob_bin),
                "rows": int(len(bdf)),
                "avg_pred_prob": float(bdf["y_prob"].mean()),
                "empirical_fire_rate": float(bdf["y_true"].mean()),
                "gap_pred_minus_actual": float(bdf["y_prob"].mean() - bdf["y_true"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_calibration_experiment(
    holdout_days: int = 365,
    calibration_days: int = 365,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
    output_dir: Path | None = None,
) -> dict:
    df = _load_training_frame()
    train_df, calibration_df, test_df, calibration_cutoff, test_cutoff = _split_train_calibration_test(
        df=df,
        holdout_days=holdout_days,
        calibration_days=calibration_days,
    )

    if train_df.empty or calibration_df.empty or test_df.empty:
        raise ValueError("Invalid time split for calibration experiment.")
    if train_df[LABEL].nunique() < 2 or calibration_df[LABEL].nunique() < 2 or test_df[LABEL].nunique() < 2:
        raise ValueError("One of train/calibration/test splits has only one class.")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / "reports" / f"catboost_calibration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, scale_pos_weight = build_model(
        X_train=train_df[FEATURES],
        y_train=train_df[LABEL].astype(int),
        X_valid=calibration_df[FEATURES],
        y_valid=calibration_df[LABEL].astype(int),
    )

    y_calibration = calibration_df[LABEL].astype(int).to_numpy()
    y_test = test_df[LABEL].astype(int).to_numpy()

    calibration_prob_base = model.predict_proba(calibration_df[FEATURES])[:, 1]
    test_prob_base = model.predict_proba(test_df[FEATURES])[:, 1]

    platt_scaler = _fit_platt_scaler(calibration_prob_base, y_calibration)
    test_prob_platt = _apply_platt_scaler(platt_scaler, test_prob_base)

    isotonic_scaler = _fit_isotonic_scaler(calibration_prob_base, y_calibration)
    test_prob_isotonic = isotonic_scaler.predict(test_prob_base)

    method_probs = {
        "base": test_prob_base,
        "platt": test_prob_platt,
        "isotonic": test_prob_isotonic,
    }

    metric_rows = []
    bin_frames = []
    pred_frames = []
    for method_name, y_prob in method_probs.items():
        metrics = classification_metrics(
            y_true=y_test,
            y_prob=y_prob,
            threshold=threshold,
            min_precision=min_precision,
            max_far=max_far,
        )
        metric_rows.append(
            {
                "method": method_name,
                "rows": int(len(y_test)),
                "positive": int(y_test.sum()),
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
        bin_frames.append(_build_calibration_bins(y_true=y_test, y_prob=y_prob, method_name=method_name))

        method_pred = test_df[["station_id", "date", LABEL]].copy()
        method_pred["method"] = method_name
        method_pred["pred_prob"] = y_prob
        pred_frames.append(method_pred)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["brier", "log_loss", "pr_auc"], ascending=[True, True, False])
    metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False, encoding="utf-8-sig")

    bins_df = pd.concat(bin_frames, ignore_index=True)
    bins_df.to_csv(output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df.to_parquet(output_dir / "test_predictions.parquet", index=False)

    summary = {
        "output_dir": str(output_dir),
        "holdout_days": holdout_days,
        "calibration_days": calibration_days,
        "threshold": threshold,
        "split": {
            "train_start_date": str(train_df["date"].min().date()),
            "train_end_date": str(train_df["date"].max().date()),
            "calibration_start_date": str(calibration_df["date"].min().date()),
            "calibration_end_date": str(calibration_df["date"].max().date()),
            "test_start_date": str(test_df["date"].min().date()),
            "test_end_date": str(test_df["date"].max().date()),
            "calibration_cutoff": str(calibration_cutoff.date()),
            "test_cutoff": str(test_cutoff.date()),
        },
        "rows": {
            "train_rows": int(len(train_df)),
            "train_positive": int(train_df[LABEL].sum()),
            "calibration_rows": int(len(calibration_df)),
            "calibration_positive": int(calibration_df[LABEL].sum()),
            "test_rows": int(len(test_df)),
            "test_positive": int(test_df[LABEL].sum()),
        },
        "scale_pos_weight": scale_pos_weight,
        "ranking": metrics_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Compare Platt and Isotonic calibration for CatBoost.")
    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--calibration-days", type=int, default=365)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration_experiment(
        holdout_days=args.holdout_days,
        calibration_days=args.calibration_days,
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
