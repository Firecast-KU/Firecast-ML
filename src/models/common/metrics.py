from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def time_split_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    holdout_days: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

    ordered = df.sort_values([date_col]).reset_index(drop=True)
    cutoff = ordered[date_col].max() - pd.Timedelta(days=holdout_days)
    train_df = ordered[ordered[date_col] < cutoff].copy()
    test_df = ordered[ordered[date_col] >= cutoff].copy()
    return train_df, test_df, cutoff


def recall_at_fixed_precision(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    min_precision: float,
) -> dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {"threshold": None, "precision": None, "recall": None}

    precision = precision[:-1]
    recall = recall[:-1]
    valid = precision >= min_precision
    if not np.any(valid):
        return {"threshold": None, "precision": None, "recall": None}

    best_idx = np.argmax(recall[valid])
    selected = np.where(valid)[0][best_idx]
    return {
        "threshold": float(thresholds[selected]),
        "precision": float(precision[selected]),
        "recall": float(recall[selected]),
    }


def recall_at_fixed_far(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    max_far: float,
) -> dict[str, float | None]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    thresholds = np.unique(y_prob_arr)[::-1]
    best: dict[str, float | None] = {"threshold": None, "far": None, "recall": None, "precision": None}

    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        if far > max_far:
            continue

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        if best["recall"] is None or recall > best["recall"]:
            best = {
                "threshold": float(threshold),
                "far": far,
                "recall": recall,
                "precision": precision,
            }

    return best


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "log_loss": float(log_loss(y_true_arr, y_prob_arr)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "false_alarm_rate": far,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "recall_at_precision": recall_at_fixed_precision(y_true_arr, y_prob_arr, min_precision=min_precision),
        "recall_at_far": recall_at_fixed_far(y_true_arr, y_prob_arr, max_far=max_far),
    }
    return metrics
