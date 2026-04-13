from pathlib import Path

# firecast/ 폴더 기준 루트
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
FEAT_DIR = DATA_DIR / "features"
DATASET_DIR = FEAT_DIR / "datasets"

# 세부 경로
FIRE_RAW_DIR = RAW_DIR / "fires" / "FRT000102_42"
WEATHER_RAW_DIR = RAW_DIR / "weather"

TRAIN_TEST_DIR = FEAT_DIR / "train_test_split"

MODEL_DIR = ROOT / "models"

# 필요한 디렉토리는 미리 만들어두기
PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
