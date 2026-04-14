import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.paths import LEGACY_ARTIFACT_DIR, PROC_DIR, ROOT


DEFAULT_DATA_FILE = PROC_DIR / "weather_labeled.parquet"


def _infer_meta_path(model_path: Path) -> Path:
    if model_path.name.endswith(".joblib"):
        return model_path.with_name(model_path.name.replace(".joblib", "_meta.json"))
    return model_path.with_suffix(".json")


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_data_file(cli_data_file: str | None, old_meta: dict, new_meta: dict) -> Path:
    if cli_data_file:
        return Path(cli_data_file)
    for meta in (old_meta, new_meta):
        if meta.get("data_file"):
            return Path(meta["data_file"])
    return DEFAULT_DATA_FILE


def _resolve_label(cli_label: str | None, old_meta: dict, new_meta: dict) -> str:
    if cli_label:
        return cli_label
    labels = [m.get("label") for m in (old_meta, new_meta) if m.get("label")]
    if labels and len(set(labels)) == 1:
        return labels[0]
    return "fire_label"


def _resolve_features(
    cli_features: list[str] | None,
    old_meta: dict,
    new_meta: dict,
) -> tuple[list[str], list[str]]:
    if cli_features:
        return cli_features, cli_features

    old_features = old_meta.get("features")
    new_features = new_meta.get("features")

    if not old_features or not new_features:
        raise ValueError(
            "Cannot infer features from meta files. "
            "Pass --features or provide meta files that include a 'features' list."
        )
    return old_features, new_features


def _resolve_holdout_days(cli_holdout_days: int | None, old_meta: dict, new_meta: dict) -> int:
    if cli_holdout_days is not None:
        return cli_holdout_days
    for meta in (old_meta, new_meta):
        split = meta.get("split", {})
        holdout_days = split.get("holdout_days")
        if isinstance(holdout_days, int) and holdout_days > 0:
            return holdout_days
    return 240


def _prepare_eval_data(
    data_file: Path,
    date_col: str,
    label_col: str,
    old_features: list[str],
    new_features: list[str],
    holdout_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_parquet(data_file)
    required_cols = {date_col, label_col, *old_features, *new_features}
    missing = sorted([c for c in required_cols if c not in df.columns])
    if missing:
        raise KeyError(f"Missing required columns in data: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # Use the same rows for both models for fair comparison.
    drop_cols = list(set(old_features + new_features + [label_col]))
    df = df.dropna(subset=drop_cols).copy()
    if df.empty:
        raise ValueError("No rows available after dropping missing values.")

    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    test_df = df[df[date_col] >= cutoff].copy()
    if test_df.empty:
        raise ValueError(f"No test rows found for holdout_days={holdout_days}.")
    if test_df[label_col].nunique() < 2:
        raise ValueError("Test labels contain only one class; metrics like ROC-AUC are undefined.")

    return test_df, cutoff


def _predict_prob(model, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    y_pred = model.predict(X)
    return pd.Series(y_pred, index=X.index).astype(float)


def _metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }


def _build_report_markdown(summary: dict) -> str:
    old = summary["old"]
    new = summary["new"]
    delta = summary["delta"]
    meta = summary["meta"]
    return (
        "# Model Comparison Report\n\n"
        f"- Generated at: {meta['generated_at']}\n"
        f"- Data file: `{meta['data_file']}`\n"
        f"- Date column: `{meta['date_col']}`\n"
        f"- Label column: `{meta['label_col']}`\n"
        f"- Holdout days: `{meta['holdout_days']}` (cutoff: `{meta['cutoff_date']}`)\n"
        f"- Eval rows: `{meta['eval_rows']}` (positives: `{meta['eval_positive']}`)\n"
        f"- Threshold: `{meta['threshold']}`\n\n"
        "## Metrics\n\n"
        "| Metric | Old | New | Delta (New-Old) |\n"
        "|---|---:|---:|---:|\n"
        f"| ROC-AUC | {old['roc_auc']:.6f} | {new['roc_auc']:.6f} | {delta['roc_auc']:+.6f} |\n"
        f"| PR-AUC | {old['pr_auc']:.6f} | {new['pr_auc']:.6f} | {delta['pr_auc']:+.6f} |\n"
        f"| Brier | {old['brier']:.6f} | {new['brier']:.6f} | {delta['brier']:+.6f} |\n"
        f"| LogLoss | {old['log_loss']:.6f} | {new['log_loss']:.6f} | {delta['log_loss']:+.6f} |\n"
        f"| Accuracy | {old['accuracy']:.6f} | {new['accuracy']:.6f} | {delta['accuracy']:+.6f} |\n"
        f"| Precision | {old['precision']:.6f} | {new['precision']:.6f} | {delta['precision']:+.6f} |\n"
        f"| Recall | {old['recall']:.6f} | {new['recall']:.6f} | {delta['recall']:+.6f} |\n"
        f"| F1 | {old['f1']:.6f} | {new['f1']:.6f} | {delta['f1']:+.6f} |\n"
        f"| PositiveRate | {old['positive_rate']:.6f} | {new['positive_rate']:.6f} | {delta['positive_rate']:+.6f} |\n"
    )


def compare_and_save(
    old_model_path: Path,
    new_model_path: Path,
    old_meta_path: Path,
    new_meta_path: Path,
    data_file: Path,
    date_col: str,
    label_col: str,
    old_features: list[str],
    new_features: list[str],
    holdout_days: int,
    threshold: float,
) -> dict:
    old_model = joblib.load(old_model_path)
    new_model = joblib.load(new_model_path)

    test_df, cutoff = _prepare_eval_data(
        data_file=data_file,
        date_col=date_col,
        label_col=label_col,
        old_features=old_features,
        new_features=new_features,
        holdout_days=holdout_days,
    )

    y_true = test_df[label_col].astype(int)
    old_prob = _predict_prob(old_model, test_df[old_features])
    new_prob = _predict_prob(new_model, test_df[new_features])

    old_metrics = _metrics(y_true, old_prob, threshold=threshold)
    new_metrics = _metrics(y_true, new_prob, threshold=threshold)

    delta = {k: float(new_metrics[k] - old_metrics[k]) for k in old_metrics.keys() if k != "threshold"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "reports" / f"model_compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "old_model_path": str(old_model_path),
            "new_model_path": str(new_model_path),
            "old_meta_path": str(old_meta_path),
            "new_meta_path": str(new_meta_path),
            "data_file": str(data_file),
            "date_col": date_col,
            "label_col": label_col,
            "old_features": old_features,
            "new_features": new_features,
            "holdout_days": holdout_days,
            "cutoff_date": str(cutoff.date()),
            "eval_rows": int(len(test_df)),
            "eval_positive": int(y_true.sum()),
            "threshold": threshold,
        },
        "old": old_metrics,
        "new": new_metrics,
        "delta": delta,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(
        [
            {"model": "old", **old_metrics},
            {"model": "new", **new_metrics},
            {"model": "delta_new_minus_old", **{**delta, "threshold": threshold}},
        ]
    )
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    report_md = _build_report_markdown(summary)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Comparison completed.\nReport directory: {out_dir}")
    print("Generated files:")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'metrics.csv'}")
    print(f"- {out_dir / 'summary.json'}")
    print("\nKey deltas (new - old):")
    print(f"ROC-AUC: {delta['roc_auc']:+.6f}")
    print(f"PR-AUC : {delta['pr_auc']:+.6f}")
    print(f"Recall : {delta['recall']:+.6f}")
    print(f"F1     : {delta['f1']:+.6f}")

    return summary


def parse_args():
    parser = ArgumentParser(description="Compare old vs new model on the same time-holdout evaluation set.")
    parser.add_argument("--old-model", type=str, default=str(LEGACY_ARTIFACT_DIR / "legacy_base_lr.joblib"))
    parser.add_argument("--new-model", type=str, required=True)
    parser.add_argument("--old-meta", type=str, default=None)
    parser.add_argument("--new-meta", type=str, default=None)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Force same feature list for both models. If omitted, uses each model meta features.",
    )
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    old_model_path = Path(args.old_model)
    new_model_path = Path(args.new_model)
    old_meta_path = Path(args.old_meta) if args.old_meta else _infer_meta_path(old_model_path)
    new_meta_path = Path(args.new_meta) if args.new_meta else _infer_meta_path(new_model_path)

    if not old_model_path.exists():
        raise FileNotFoundError(f"Old model file not found: {old_model_path}")
    if not new_model_path.exists():
        raise FileNotFoundError(f"New model file not found: {new_model_path}")

    old_meta = _load_meta(old_meta_path)
    new_meta = _load_meta(new_meta_path)

    data_file = _resolve_data_file(args.data_file, old_meta, new_meta)
    label_col = _resolve_label(args.label_col, old_meta, new_meta)
    old_features, new_features = _resolve_features(args.features, old_meta, new_meta)
    holdout_days = _resolve_holdout_days(args.holdout_days, old_meta, new_meta)

    compare_and_save(
        old_model_path=old_model_path,
        new_model_path=new_model_path,
        old_meta_path=old_meta_path,
        new_meta_path=new_meta_path,
        data_file=data_file,
        date_col=args.date_col,
        label_col=label_col,
        old_features=old_features,
        new_features=new_features,
        holdout_days=holdout_days,
        threshold=args.threshold,
    )
