import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.paths import DATASET_DIR, MODEL_DIR
from src.models.metrics_utils import classification_metrics
from src.models.tcn_model import FireTCNClassifier


DATA_FILE = DATASET_DIR / "tcn_dataset.npz"
INDEX_FILE = DATASET_DIR / "tcn_index.parquet"


def _build_loader(x_dyn: np.ndarray, x_static: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x_dyn, dtype=torch.float32),
        torch.tensor(x_static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_saved_model(
    model_path: Path,
    meta_path: Path,
    holdout_days: int | None = None,
    batch_size: int = 128,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    holdout_days = meta.get("split", {}).get("holdout_days", 240) if holdout_days is None else holdout_days

    npz = np.load(DATA_FILE)
    X_dyn = npz["X_dyn"]
    X_static = npz["X_static"]
    y = npz["y"]
    index_df = pd.read_parquet(INDEX_FILE)
    index_df["date"] = pd.to_datetime(index_df["date"])

    cutoff = index_df["date"].max() - pd.Timedelta(days=holdout_days)
    test_mask = (index_df["date"] >= cutoff).to_numpy()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    x_dyn_test = (X_dyn[test_mask] - checkpoint["dyn_mean"]) / checkpoint["dyn_std"]
    x_static_test = (X_static[test_mask] - checkpoint["static_mean"]) / checkpoint["static_std"]
    y_test = y[test_mask]

    loader = _build_loader(x_dyn_test, x_static_test, y_test, batch_size=batch_size)
    model = FireTCNClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probs = []
    with torch.no_grad():
        for batch_dyn, batch_static, _ in loader:
            logits = model(batch_dyn, batch_static)
            probs.append(torch.sigmoid(logits).numpy())
    y_prob = np.concatenate(probs, axis=0)

    metrics = classification_metrics(
        y_true=y_test,
        y_prob=y_prob,
        threshold=threshold,
        min_precision=min_precision,
        max_far=max_far,
    )
    summary = {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "data_file": str(DATA_FILE),
        "index_file": str(INDEX_FILE),
        "cutoff_date": str(cutoff.date()),
        "eval_rows": int(len(y_test)),
        "eval_positive": int(y_test.sum()),
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = ArgumentParser(description="Evaluate saved TCN model on tcn_dataset holdout.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_DIR / "tcn_v1.pt"))
    parser.add_argument("--meta-path", type=str, default=str(MODEL_DIR / "tcn_v1_meta.json"))
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_saved_model(
        model_path=Path(args.model_path),
        meta_path=Path(args.meta_path),
        holdout_days=args.holdout_days,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
    )
