# Firecast 서버 데이터 계약서

이 문서는 서버 개발자가 Firecast 학습 및 예측에 필요한 데이터를 구축할 수 있도록 입력 데이터 형식, 정규화 규칙, 모델 입출력 형식을 정리한 문서다.

핵심 원칙은 다음과 같다.

- 서버는 모델별 입력을 따로 만들 필요가 없다.
- 서버는 공통 피처 테이블을 만들면 된다.
- Firecast 내부에서 공통 피처를 LR, CatBoost, TCN 입력 형태로 분기한다.

## 1. 서버가 준비해야 하는 데이터

서버는 아래 4종 데이터를 준비하면 된다.

1. 관측소 메타 데이터
2. 시간별 지상관측 데이터
3. 단기예보 육상 데이터
4. 화재 라벨 데이터

이 데이터들은 최종적으로 아래 기본 키를 갖는 공통 일 단위 피처 테이블로 정규화된다.

```text
station_id + date
```

이 일 단위 테이블이 서버와 Firecast 사이의 핵심 데이터 계약이다.

## 2. 전체 처리 흐름

```text
관측소 메타
  + 시간별 지상관측
  + 단기예보 육상
  + 화재 라벨
  -> 공통 일 단위 피처 테이블 생성
  -> LR / CatBoost용 wide dataset 생성
  -> TCN용 sequence dataset 생성
  -> 공통 예측 출력 생성
```

## 3. 원천 데이터 계약

### 3.1 관측소 메타 데이터

필수 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `STN_ID` | int | 관측소 번호 |
| `LON` | float | 경도 |
| `LAT` | float | 위도 |
| `FCT_ID` | string | 예보구역코드 |

권장 추가 컬럼:

- `STN_KO`
- `STN_EN`
- `HT`
- `HT_PA`
- `HT_TA`
- `HT_WD`
- `HT_RN`
- `LAW_ID`
- `LAW_ADDR`

내부 매핑:

| 원천 컬럼 | 내부 컬럼 |
|---|---|
| `STN_ID` | `station_id` |
| `LON` | `stn_lon` |
| `LAT` | `stn_lat` |
| `FCT_ID` | 예보 조인 키 |

예시:

```json
[
  {
    "STN_ID": 104,
    "LON": 128.85535,
    "LAT": 37.80456,
    "FCT_ID": "11D20501",
    "STN_KO": "북강릉"
  },
  {
    "STN_ID": 105,
    "LON": 128.89098,
    "LAT": 37.75146,
    "FCT_ID": "11D20501",
    "STN_KO": "강릉"
  }
]
```

### 3.2 시간별 지상관측 데이터

필수 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `STN` | int | 관측소 번호 |
| `TM` | string | 관측 시각, KST |
| `TA` | float | 기온 |
| `WD` | int 또는 string | 풍향 |
| `RN` | float | 강수량 |

권장 추가 컬럼:

- `WS`
- `HM`
- `TD`
- `PA`
- `PS`
- `IR`
- `WW`
- `WP`
- `CA_TOT`

내부 정규화 컬럼:

| 내부 컬럼 | 생성 규칙 |
|---|---|
| `station_id` | `STN` |
| `obs_ts` | `TM` 파싱 |
| `date` | `obs_ts`에서 날짜 추출 |
| `TA` | `TA` |
| `WD_sin` | `WD`를 각도로 변환 후 `sin` |
| `WD_cos` | `WD`를 각도로 변환 후 `cos` |
| `is_precip` | `RN > 0` 또는 날씨 코드 기반 판단 |

권장 시각 포맷:

```text
YYYYMMDDHHMM
```

예시:

```json
[
  {
    "STN": 104,
    "TM": "202103250000",
    "TA": 12.3,
    "WD": 18,
    "RN": 0.0,
    "HM": 41.2,
    "WS": 2.4
  },
  {
    "STN": 104,
    "TM": "202103251200",
    "TA": 18.7,
    "WD": 20,
    "RN": 0.0,
    "HM": 33.1,
    "WS": 3.1
  }
]
```

### 3.3 단기예보 육상 데이터

필수 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `REG_ID` | string | 예보구역코드 |
| `TM_FC` | string | 발표시각 |
| `TM_EF` | string | 발효시각 |
| `ST` | float | 강수확률 |
| `SKY` | string | 하늘상태코드 |
| `PREP` | int | 강수유무코드 |

권장 추가 컬럼:

- `TA`
- `WF`
- `W1`
- `W2`

내부 정규화 컬럼:

| 내부 컬럼 | 생성 규칙 |
|---|---|
| `forecast_region_id` | `REG_ID` |
| `forecast_issue_ts` | `TM_FC` |
| `forecast_effective_ts` | `TM_EF` |
| `POP` | `ST` |
| `SKY` | `SKY` |
| `forecast_precip_code` | `PREP` |

예시:

```json
[
  {
    "REG_ID": "11D20501",
    "TM_FC": "202103241800",
    "TM_EF": "202103250000",
    "ST": 20,
    "SKY": "DB02",
    "PREP": 0,
    "TA": 13
  },
  {
    "REG_ID": "11D20501",
    "TM_FC": "202103241800",
    "TM_EF": "202103251200",
    "ST": 10,
    "SKY": "DB01",
    "PREP": 0,
    "TA": 19
  }
]
```

### 3.4 화재 라벨 데이터

학습용으로만 필요하다.

필수 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `station_id` | int 또는 string | 관측소 ID |
| `fire_date` | date 또는 datetime | 화재 발생일 |

예시:

```json
[
  {
    "station_id": 104,
    "fire_date": "2021-03-25"
  },
  {
    "station_id": 105,
    "fire_date": "2021-04-02"
  }
]
```

## 4. 조인 규칙

### 관측소 메타 + 시간별 관측

```text
STN_ID = STN
```

### 관측소 메타 + 단기예보

```text
FCT_ID = REG_ID
```

즉 관측소와 예보구역의 연결 관계가 정확해야 한다.

## 5. 예보 선택 규칙

예보 데이터는 `발표시각(TM_FC)`과 `발효시각(TM_EF)`가 다르므로, 학습과 서빙에서 같은 규칙으로 선택해야 한다.

권장 규칙:

1. 예측 대상 날짜 `D`를 정한다.
2. `TM_EF`가 날짜 `D`에 해당하는 예보만 남긴다.
3. 그중 `TM_FC`가 가장 최신인 레코드를 선택한다.
4. 선택된 예보를 `FCT_ID = REG_ID` 기준으로 관측소에 붙인다.

이 규칙을 쓰면 학습과 예측 시점의 데이터 선택 방식이 일관된다.

## 6. 공통 일 단위 피처 테이블

서버가 최종적으로 제공하거나 생성 가능해야 하는 테이블이다.

기본 키:

```text
station_id + date
```

권장 스키마:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `station_id` | string 또는 int | 관측소 ID |
| `date` | date | 기준 날짜 |
| `stn_lat` | float | 위도 |
| `stn_lon` | float | 경도 |
| `TA_day` | float | 대표 일 기온 |
| `TA_dtr_day` | float | 일교차 proxy |
| `POP_day` | float | 일 강수확률 |
| `is_precip_day` | int | 강수 여부 0/1 |
| `WD_sin_day` | float | 풍향 sin |
| `WD_cos_day` | float | 풍향 cos |
| `SKY_day` | string | `DB01`~`DB04` |
| `obs_cnt_sel` | int | 선택된 관측 수 |
| `has_00_obs` | int | 00시 관측 존재 여부 |
| `has_12_obs` | int | 12시 관측 존재 여부 |
| `fire_label` | int | 화재 라벨 |

현재 Firecast 코드는 00시와 12시 관측을 우선 사용한다.

일 단위 집계 규칙:

- `TA_day`: 00시와 12시 기온 평균
- `TA_dtr_day`: `|TA(12) - TA(00)|`
- `POP_day`: 해당 날짜 예보 `ST`
- `is_precip_day`: 강수 여부 0/1
- `WD_sin_day`, `WD_cos_day`: 선택된 관측 평균
- `SKY_day`: 대표 하늘상태코드

예시:

```json
[
  {
    "station_id": "104",
    "date": "2021-03-25",
    "stn_lat": 37.80456,
    "stn_lon": 128.85535,
    "TA_day": 15.5,
    "TA_dtr_day": 6.4,
    "POP_day": 15.0,
    "is_precip_day": 0,
    "WD_sin_day": 0.28,
    "WD_cos_day": -0.91,
    "SKY_day": "DB02",
    "obs_cnt_sel": 2,
    "has_00_obs": 1,
    "has_12_obs": 1,
    "fire_label": 1
  }
]
```

## 7. 파생 피처 생성 규칙

### 풍향 변환

모델은 원본 `WD`를 직접 쓰지 않고 `WD_sin`, `WD_cos`를 사용한다.

예시:

```python
import math

degree = wd_index_to_degree(WD)
wd_sin = math.sin(math.radians(degree))
wd_cos = math.cos(math.radians(degree))
```

주의:

- 원천 데이터가 16방위인지 36방위인지에 따라 각도 매핑 테이블이 달라진다.
- 결측 풍향은 결측으로 유지한다.

### 강수 여부 생성

권장 우선순위:

1. `RN > 0` 이면 `is_precip = 1`
2. 아니면 `PREP in {1, 2, 3, 4}` 이면 `1`
3. 아니면 `IR`, `WW`, `WP`가 비/눈 계열이면 `1`
4. 아니면 `0`

### SKY 코드 생성

우선 사용 데이터:

- 단기예보의 `SKY`

허용값:

- `DB01`
- `DB02`
- `DB03`
- `DB04`

관측 기반 대체 규칙:

| `CA_TOT` | `SKY` |
|---|---|
| 0 ~ 2 | `DB01` |
| 3 ~ 5 | `DB02` |
| 6 ~ 8 | `DB03` |
| 9 ~ 10 | `DB04` |

## 8. Firecast 내부 산출 파일

현재 Firecast가 사용하는 내부 파일은 다음과 같다.

| 파일 | 설명 |
|---|---|
| `data/features/datasets/station_day_base.parquet` | 공통 일 단위 테이블 |
| `data/features/datasets/catboost_dataset.parquet` | LR/CatBoost용 wide dataset |
| `data/features/datasets/tcn_dataset.npz` | TCN용 sequence dataset |
| `data/features/datasets/tcn_index.parquet` | TCN 샘플 인덱스 |

## 9. 모델별 입력 형식

### LR

- 입력 파일: `catboost_dataset.parquet`
- 한 행 = 한 관측소의 하루
- lag, rolling feature 기반 분류

### CatBoost

- 입력 파일: `catboost_dataset.parquet`
- LR와 동일한 wide table 사용

### TCN

- 입력 파일: `tcn_dataset.npz`
- 한 샘플 = 최근 14일 시계열 + 정적 피처

## 10. 공통 예측 출력 계약

공통 예측 실행:

```bash
python -m src.models.predict --target-date YYYY-MM-DD --model all
```

출력 스키마:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `model_name` | string | `lr`, `catboost`, `tcn` |
| `station_id` | string | 관측소 ID |
| `date` | datetime | 예측 대상 날짜 |
| `pred_prob` | float | 화재 확률 |
| `risk_level` | string | `LOW`, `MODERATE`, `HIGH`, `EXTREME` |
| `alert_threshold` | float | 운영 경보 판정 기준 (`pred_prob >= alert_threshold`) |
| `is_alert` | int | 운영 경보 여부 (`1`=경보, `0`=비경보) |
| `stn_lat` | float | 위도 |
| `stn_lon` | float | 경도 |

예시 응답:

```json
{
  "target_date": "2021-03-25",
  "model": "all",
  "rows": 6,
  "predictions": [
    {
      "model_name": "catboost",
      "station_id": "104",
      "date": "2021-03-25 00:00:00",
      "pred_prob": 0.595367531142382,
      "risk_level": "MODERATE",
      "alert_threshold": 0.7164710376269543,
      "is_alert": 0,
      "stn_lat": 37.80456,
      "stn_lon": 128.85535
    },
    {
      "model_name": "lr",
      "station_id": "104",
      "date": "2021-03-25 00:00:00",
      "pred_prob": 0.862471844173419,
      "risk_level": "EXTREME",
      "alert_threshold": 0.5,
      "is_alert": 1,
      "stn_lat": 37.80456,
      "stn_lon": 128.85535
    }
  ]
}
```

위험도 규칙:

| `pred_prob` 구간 | `risk_level` |
|---|---|
| `<= 0.4` | `LOW` |
| `<= 0.6` | `MODERATE` |
| `<= 0.8` | `HIGH` |
| `> 0.8` | `EXTREME` |

## 11. 서버가 실제로 넘겨야 하는 형태

연동 방식은 두 가지가 가능하다.

### 방식 A: 권장 방식

서버가 공통 일 단위 피처 테이블을 만들어서 전달한다.

이 방식의 장점:

- 모델 공통 계약이 단순하다
- 서버와 모델 코드의 책임이 명확하다
- 운영 시 가장 안정적이다

### 방식 B: 원천데이터 전달 방식

서버가 메타, 관측, 예보, 라벨 원천데이터를 전달하고 Firecast가 정규화를 담당한다.

이 방식의 장점:

- Firecast가 정규화 로직을 완전히 소유할 수 있다

이 방식의 단점:

- 서버와 모델 사이의 결합도가 높아진다
- 구현 복잡도가 높다

운영 환경에서는 `방식 A`, 즉 공통 일 단위 피처 테이블 전달을 권장한다.

## 12. 서버 개발 체크리스트

- `STN_ID <-> FCT_ID` 매핑 테이블 확보
- `TM`, `TM_FC`, `TM_EF` 파싱 규칙 통일
- 예보 선택 규칙 고정
- 풍향 각도 변환 테이블 구현
- 강수 여부 생성 규칙 구현
- `SKY`를 `DB01 ~ DB04`로 정규화
- `station_id + date` 유니크 보장
- 학습과 예측에서 동일한 정규화 로직 사용
- 미래 정보를 학습 데이터에 섞지 않도록 검증

## 13. 데이터 검증 규칙

Firecast에 데이터를 넘기기 전에 아래를 확인해야 한다.

1. `station_id + date`가 유일해야 한다.
2. `POP_day`는 숫자여야 한다.
3. `is_precip_day`는 `0` 또는 `1`만 허용한다.
4. `SKY_day`는 `DB01`, `DB02`, `DB03`, `DB04`, 또는 null이어야 한다.
5. 모든 관측소에 대해 `stn_lat`, `stn_lon`이 존재해야 한다.
6. 모든 시각 정보는 KST 기준으로 일관되어야 한다.
7. 예측 대상 날짜 이후의 미래 정보가 학습 행에 포함되면 안 된다.

## 14. 현재 Firecast 구현 파일

- 공통 dataset 생성: [src/core/model_dataset.py](../core/model_dataset.py)
- 공통 예측 실행: [src/models/predict.py](../models/predict.py)
- LR 학습: [src/models/lr/train.py](../models/lr/train.py)
- CatBoost 학습: [src/models/catboost/train.py](../models/catboost/train.py)
- TCN 학습: [src/models/tcn/train.py](../models/tcn/train.py)
## 15. 서버 구현 주의사항

- Firecast 공통 예측 출력에는 `alert_threshold`, `is_alert`가 포함된다.
- `risk_level`은 시각화와 설명용 등급이다.
- 실제 경보 발송, 알림 라우팅, 대시보드 강조 여부는 `is_alert`를 기준으로 처리한다.
- 서버가 별도의 threshold를 hard-code 하지 말고 Firecast가 반환한 `alert_threshold`, `is_alert`를 그대로 사용한다.
- 기존 응답 스키마를 사용 중이면 DTO / serializer / OpenAPI schema에 `alert_threshold`, `is_alert`를 추가해야 한다.
- 예시:
  - `pred_prob = 0.80`, `alert_threshold = 0.7165` -> `is_alert = 1`
  - `pred_prob = 0.57`, `alert_threshold = 0.7165` -> `is_alert = 0`
