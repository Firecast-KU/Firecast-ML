from pathlib import Path

# project/ 기준 루트
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

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
MODEL_ARTIFACT_DIR = MODEL_DIR / "artifacts"
LR_ARTIFACT_DIR = MODEL_ARTIFACT_DIR / "lr"
CATBOOST_ARTIFACT_DIR = MODEL_ARTIFACT_DIR / "catboost"
TCN_ARTIFACT_DIR = MODEL_ARTIFACT_DIR / "tcn"
LEGACY_ARTIFACT_DIR = MODEL_ARTIFACT_DIR / "legacy"

OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"

# 필요한 디렉토리는 미리 만들어두기
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LR_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CATBOOST_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TCN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
