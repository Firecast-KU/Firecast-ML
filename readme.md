# 🔥 Firecast
**기상 데이터 기반 산불 발생 예측 파이프라인**

Firecast는 산불 발생 이력과 기상 관측 데이터를 결합하여  
산불 발생 가능성을 예측하기 위한 데이터 파이프라인 및  
Baseline 예측 모델을 구현한 프로젝트입니다.

---

## 📁 Project Structure
```
firecast/
│
├── src/ # 핵심 파이썬 소스 코드
│ ├── config/ # 전역 설정 및 경로 관리
│ │ ├── init.py
│ │ └── paths.py # 프로젝트 경로 일괄 관리
│ │
│ ├── core/ # 핵심 도메인 로직
│ │ ├── init.py
│ │ ├── fire_events.py # 산불 발생 데이터 처리
│ │ ├── stations.py # 기상 관측소 정보 처리
│ │ ├── weather_daily.py# 일별 기상 데이터 전처리
│ │ └── labeling.py # 산불 발생 라벨링 로직
│ │
│ ├── eda/ # 탐색적 데이터 분석
│ │ ├── init.py
│ │ ├── data_exploration.py
│ │ └── inspect_fire_gangneung.py
│ │
│ ├── pipelines/ # 재현 가능한 데이터 파이프라인
│ │ ├── init.py
│ │ ├── build_fire_events.py
│ │ ├── match_fire_station.py
│ │ └── merge_fire_weather.py
│ │
│ ├── experiments/ # 초기/레거시 모델 실험 코드
│ │ ├── init.py
│ │ └── lr_baseline_legacy.py
│ │
│ ├── models/ # 모델 학습·평가·예측 로직
│ │ ├── init.py
│ │ ├── common/ # 공통 metric / 유틸
│ │ ├── lr/ # LR 학습 / 예측
│ │ ├── catboost/ # CatBoost 학습 / 평가
│ │ ├── tcn/ # TCN 모델 / 학습 / 평가
│ │ ├── selection/ # benchmark / station winner 선택
│ │ ├── legacy/ # 과거 weather_labeled 기반 코드 보존
│ │ └── artifacts/ # 저장된 모델 아티팩트
│ │
│ └── validation/ # 데이터 및 결과 검증
│ ├── init.py
│ ├── validate_fire_weather.py
│ └── RN_histogram.py
│
├── data/
│ ├── raw/ # 🔒 원본 데이터 (절대 수정 금지)
│ │ ├── fires/ # 산불 발생 shapefile
│ │ ├── weather/ # ASOS 일별 기상 데이터
│ │ └── meta/ # 관측소 메타데이터
│ │
│ ├── processed/ # 전처리 완료 데이터
│ │ ├── fire_events.parquet
│ │ ├── weather_daily.parquet
│ │ ├── fire_weather_merged.parquet
│ │ └── weather_labeled.parquet
│ │
│ └── features/ # 모델 입력용 데이터
│ └── train_test_split/
│
├── notebooks/ # 분석/실험용 노트북
│ ├── 01_data_exploration.ipynb
│ ├── 02_match_fire_station.ipynb
│ ├── 03_join_fire_weather.ipynb
│ └── LR_Firecast.ipynb
│
├── outputs/ # 결과물 (모델, 그래프 등)
│
├── docs/ # 문서
│ └── analysis.md
│
├── readme.md
└── .gitignore
```


---

## 🔄 Data Processing Pipeline

### 1️⃣ 산불 발생 데이터 처리
- Shapefile 로딩 및 좌표계 통일 (EPSG)
- 중복 이벤트 제거
- 발생 날짜 기준 정제

→ `core/fire_events.py`  
→ `pipelines/build_fire_events.py`

---

### 2️⃣ 기상 관측 데이터 전처리
- ASOS 일별 기상 데이터 로딩
- 컬럼 정규화 및 결측치 처리
- 날짜 단위 집계

→ `core/weather_daily.py`

---

### 3️⃣ 산불–관측소 매칭
- 산불 발생 좌표 기준 최근접 관측소 탐색
- 거리 기반 매칭 (미터 단위)

→ `core/stations.py`  
→ `pipelines/match_fire_station.py`

---

### 4️⃣ 산불–기상 데이터 병합
- 산불 발생일 기준 기상 데이터 Join
- 학습용 테이블 생성

→ `pipelines/merge_fire_weather.py`

---

### 5️⃣ Feature Engineering & Labeling
- 주요 기상 변수 선택
- 산불 발생 여부 이진 라벨 생성

→ `core/labeling.py`

---

## 🤖 Modeling

### Base Model (일 단위 예측)
- **Logistic Regression**
- Time-based split 적용 (최근 N일 holdout)
- 클래스 불균형 고려 (Recall 중심 평가)

#### 역할 분리
- 학습: `models/lr/train.py`
- 예측(서빙): `models/lr/predict.py`

**출력**
- 날짜별·관측소별 산불 위험 확률
- 위험도 레벨 (LOW / MODERATE / HIGH / EXTREME)

---

## 📊 Validation & Analysis

- 기상 변수 분포 검증
- 산불 발생 / 비발생 비교 시각화
- 데이터 이상치 점검

→ `validation/validate_fire_weather.py`  
→ `validation/RN_histogram.py`

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- GeoPandas
- Scikit-learn
- Matplotlib

---

## 📌 Design Principles

- `data/raw`는 **절대 수정하지 않음**
- 모든 경로는 `src/config/paths.py`에서 관리
- Notebook은 **탐색/실험용**,  
  실제 로직은 **pipeline & src 코드로 재현 가능하게 구현**

---

## 🚀 Future Work

- 시계열 기반 모델 (LSTM / Transformer)
- 공간적 특성 반영 (Grid / Spatial Encoding)
- 기상 예보 데이터 기반 **미래 산불 위험도 예측**
- 비선형 모델(RandomForest / XGBoost) 고도화
- **일 단위 예측 + 3시간 단위 위험도 갱신 구조**
- 관측 기반 Δ feature 보정 모델
- 기상 예보 데이터 활용한 선제적 위험 예측
---

## Server Integration Notes

Firecast 예측 결과에는 백엔드 연동을 위한 명시적인 Alert 판단 필드가 포함되어 있습니다.

- `pred_prob`: 모델이 예측한 산불 발생 확률
- `risk_level`: 화면 표시용 위험도 구간 (`LOW`, `MODERATE`, `HIGH`, `EXTREME`)
- `alert_threshold`: Alert 발생 여부를 결정하는 운영 기준 임계값
- `is_alert`: 최종 Alert 플래그 (`1` = Alert 발생, `0` = Alert 없음)

백엔드 개발자는 다음 원칙에 따라 해당 필드를 사용해야 합니다.

- 최종 운영 Alert 여부는 `is_alert` 값을 기준으로 판단합니다.
- `risk_level`은 시각화 또는 사용자 설명 용도로만 사용하며, Alert 라우팅 기준으로 사용해서는 안 됩니다.
- Firecast가 이미 `is_alert` 값을 반환하는 경우, 서버에서 별도의 임계값(threshold)을 하드코딩하여 다시 판단하지 않습니다.
- 응답 DTO, Serializer, API Schema에 `alert_threshold: float` 및 `is_alert: int` 필드를 포함하도록 업데이트합니다.

### Examples

- `pred_prob = 0.80`, `alert_threshold = 0.7165` → `is_alert = 1`
- `pred_prob = 0.57`, `alert_threshold = 0.7165` → `is_alert = 0`
