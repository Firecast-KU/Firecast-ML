from argparse import ArgumentParser
import json
from pathlib import Path
from shutil import copy2

import pandas as pd

from src.config.paths import DATASET_DIR
from src.core.model_dataset import build_catboost_dataset, build_tcn_dataset


DEFAULT_V2_SOURCE = DATASET_DIR / "v2" / "features" / "datasets" / "station_day_base.parquet"


def apply_v2_dataset(
    source_path: Path | None = None,
    backup: bool = True,
) -> dict:
    if source_path is None:
        source_path = DEFAULT_V2_SOURCE

    source_path = Path(source_path)
    target_path = DATASET_DIR / "station_day_base.parquet"
    backup_path = DATASET_DIR / "station_day_base.backup.parquet"

    if not source_path.exists():
        raise FileNotFoundError(f"v2 source file not found: {source_path}")

    if backup and target_path.exists():
        copy2(target_path, backup_path)

    copy2(source_path, target_path)

    station_day_base = pd.read_parquet(target_path)
    build_catboost_dataset(station_day_base=station_day_base)
    tcn_meta = build_tcn_dataset(station_day_base=station_day_base)

    summary = {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "backup_path": str(backup_path) if backup and backup_path.exists() else None,
        "station_day_base_rows": int(len(station_day_base)),
        "station_day_base_positive": int(station_day_base["fire_label"].sum()),
        "catboost_rows": int(len(pd.read_parquet(DATASET_DIR / "catboost_dataset.parquet"))),
        "catboost_path": str(DATASET_DIR / "catboost_dataset.parquet"),
        "tcn_path": str(DATASET_DIR / "tcn_dataset.npz"),
        "tcn_samples": int(tcn_meta["num_samples"]),
        "tcn_positive": int(tcn_meta["positive_samples"]),
        "dataset_dir": str(DATASET_DIR),
    }
    summary_path = DATASET_DIR / "dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = ArgumentParser(
        description="Apply downloaded v2 station_day_base parquet and rebuild derived model datasets."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_V2_SOURCE,
        help="Path to v2 station_day_base.parquet",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup of the current station_day_base.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_v2_dataset(
        source_path=args.source,
        backup=not args.no_backup,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
