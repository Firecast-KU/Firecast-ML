import json
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.paths import DATASET_DIR, MODEL_DIR
from src.models.metrics_utils import classification_metrics
from src.models.tcn_model import FireTCNClassifier


DATA_FILE = DATASET_DIR / "tcn_dataset.npz"
INDEX_FILE = DATASET_DIR / "tcn_index.parquet"
META_FILE = DATASET_DIR / "tcn_dataset_meta.json"

MODEL_NAME = "tcn_v1.pt"
META_NAME = "tcn_v1_meta.json"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_tcn_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}\n"
            "Run `python -m src.pipelines.build_model_datasets` first."
        )
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Meta file not found: {META_FILE}")

    npz = np.load(DATA_FILE)
    index_df = pd.read_parquet(INDEX_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))

    index_df["date"] = pd.to_datetime(index_df["date"])
    return npz["X_dyn"], npz["X_static"], npz["y"], index_df, meta


def _time_split_indices(index_df: pd.DataFrame, holdout_days: int):
    cutoff = index_df["date"].max() - pd.Timedelta(days=holdout_days)
    train_mask = index_df["date"] < cutoff
    test_mask = index_df["date"] >= cutoff
    return train_mask.to_numpy(), test_mask.to_numpy(), cutoff


def _standardize_from_train(
    x_dyn_train: np.ndarray,
    x_dyn_test: np.ndarray,
    x_static_train: np.ndarray,
    x_static_test: np.ndarray,
):
    dyn_mean = x_dyn_train.mean(axis=(0, 1), keepdims=True)
    dyn_std = x_dyn_train.std(axis=(0, 1), keepdims=True)
    dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std)

    static_mean = x_static_train.mean(axis=0, keepdims=True)
    static_std = x_static_train.std(axis=0, keepdims=True)
    static_std = np.where(static_std < 1e-6, 1.0, static_std)

    x_dyn_train_std = (x_dyn_train - dyn_mean) / dyn_std
    x_dyn_test_std = (x_dyn_test - dyn_mean) / dyn_std
    x_static_train_std = (x_static_train - static_mean) / static_std
    x_static_test_std = (x_static_test - static_mean) / static_std

    stats = {
        "dyn_mean": dyn_mean.astype(np.float32),
        "dyn_std": dyn_std.astype(np.float32),
        "static_mean": static_mean.astype(np.float32),
        "static_std": static_std.astype(np.float32),
    }
    return x_dyn_train_std, x_dyn_test_std, x_static_train_std, x_static_test_std, stats


def _build_loader(x_dyn: np.ndarray, x_static: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x_dyn, dtype=torch.float32),
        torch.tensor(x_static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    probs = []
    model.eval()
    with torch.no_grad():
        for x_dyn, x_static, _ in loader:
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)
            logits = model(x_dyn, x_static)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_and_save(
    holdout_days: int = 240,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold: float = 0.5,
    min_precision: float = 0.3,
    max_far: float = 0.05,
):
    set_seed(42)

    X_dyn, X_static, y, index_df, dataset_meta = _load_tcn_dataset()
    train_mask, test_mask, cutoff = _time_split_indices(index_df, holdout_days=holdout_days)

    x_dyn_train = X_dyn[train_mask]
    x_dyn_test = X_dyn[test_mask]
    x_static_train = X_static[train_mask]
    x_static_test = X_static[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    index_test = index_df.loc[test_mask].reset_index(drop=True)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Invalid TCN split: empty train or test set.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels have only one class.")
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test labels have only one class.")

    x_dyn_train, x_dyn_test, x_static_train, x_static_test, stats = _standardize_from_train(
        x_dyn_train, x_dyn_test, x_static_train, x_static_test
    )

    train_loader = _build_loader(x_dyn_train, x_static_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = _build_loader(x_dyn_test, x_static_test, y_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireTCNClassifier(
        dyn_features=x_dyn_train.shape[2],
        static_features=x_static_train.shape[1],
        channels=(32, 32, 32),
        kernel_size=3,
        dropout=0.1,
        static_hidden=16,
    ).to(device)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_pr_auc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_dyn, batch_static, batch_y in train_loader:
            batch_dyn = batch_dyn.to(device)
            batch_static = batch_static.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_dyn, batch_static)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)

        epoch_loss = running_loss / len(y_train)
        val_prob = _predict_prob(model, test_loader, device=device)
        val_metrics = classification_metrics(
            y_true=y_test,
            y_prob=val_prob,
            threshold=threshold,
            min_precision=min_precision,
            max_far=max_far,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_brier": val_metrics["brier"],
            }
        )

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("TCN training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_prob = _predict_prob(model, test_loader, device=device)
    metrics = classification_metrics(
        y_true=y_test,
        y_prob=test_prob,
        threshold=threshold,
        min_precision=min_precision,
        max_far=max_far,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_NAME
    meta_path = MODEL_DIR / META_NAME

    checkpoint = {
        "state_dict": model.state_dict(),
        "dyn_mean": stats["dyn_mean"],
        "dyn_std": stats["dyn_std"],
        "static_mean": stats["static_mean"],
        "static_std": stats["static_std"],
        "model_config": {
            "dyn_features": int(x_dyn_train.shape[2]),
            "static_features": int(x_static_train.shape[1]),
            "channels": [32, 32, 32],
            "kernel_size": 3,
            "dropout": 0.1,
            "static_hidden": 16,
        },
    }
    torch.save(checkpoint, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_FILE),
        "index_file": str(INDEX_FILE),
        "dataset_meta_file": str(META_FILE),
        "model_type": "FireTCNClassifier",
        "sequence_length": int(dataset_meta["sequence_length"]),
        "dynamic_features": dataset_meta["dynamic_features"],
        "static_features": dataset_meta["static_saved_columns"],
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "pos_weight": float(pos_weight.item()),
        },
        "split": {
            "type": "time_holdout",
            "holdout_days": holdout_days,
            "cutoff": str(cutoff.date()),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "train_start_date": str(index_df.loc[train_mask, "date"].min().date()),
            "train_end_date": str(index_df.loc[train_mask, "date"].max().date()),
            "test_start_date": str(index_df.loc[test_mask, "date"].min().date()),
            "test_end_date": str(index_df.loc[test_mask, "date"].max().date()),
        },
        "holdout_metrics": metrics,
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = MODEL_DIR / "tcn_v1_holdout_predictions.parquet"
    pred_df = index_test.copy()
    pred_df["pred_prob"] = test_prob
    pred_df.to_parquet(pred_path, index=False)

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Pred saved : {pred_path}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Train rows / positives: {len(y_train)} / {int(y_train.sum())}")
    print(f"Test rows  / positives: {len(y_test)} / {int(y_test.sum())}")
    print(f"Holdout ROC-AUC: {metrics['roc_auc']}")
    print(f"Holdout PR-AUC : {metrics['pr_auc']}")
    print(f"Holdout Brier  : {metrics['brier']}")
    print(f"Holdout LogLoss: {metrics['log_loss']}")

    return model, meta


def parse_args():
    parser = ArgumentParser(description="Train TCN model on tcn_dataset time-holdout.")
    parser.add_argument("--holdout-days", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--max-far", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        holdout_days=args.holdout_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        min_precision=args.min_precision,
        max_far=args.max_far,
    )
