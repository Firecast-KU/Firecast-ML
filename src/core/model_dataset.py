import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.paths import DATASET_DIR, PROC_DIR
from src.core.stations import WeatherStationRegistry
from src.core.weather_daily import (
    TARGET_STATIONS,
    load_weather_raw,
    parse_obs_datetime,
    pick_column,
)


K_CAT = (1, 3, 7, 14)
W_CAT = (3, 7, 14)
TCN_SEQUENCE_LENGTH = 14
TCN_SKY_LEVELS = ("DB01", "DB02", "DB03", "DB04")


def _series_or_empty(df: pd.DataFrame, col: str, dtype: str = "float64") -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=dtype)
    return df[col]


def _mode_or_na(series: pd.Series):
    # 범주형 rolling 집계에서 최빈값을 일관되게 반환한다.
    s = series.dropna().astype("string")
    if s.empty:
        return pd.NA
    mode = s.mode()
    if mode.empty:
        return s.iloc[-1]
    return mode.iloc[0]


def ensure_station_day_base_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()

    if "month" not in normalized.columns:
        normalized["month"] = normalized["date"].dt.month.astype("Int64")
    if "dayofyear" not in normalized.columns:
        normalized["dayofyear"] = normalized["date"].dt.dayofyear.astype("Int64")

    if "doy_sin" not in normalized.columns or "doy_cos" not in normalized.columns:
        radians = 2 * np.pi * normalized["dayofyear"].astype(float) / 365.0
        if "doy_sin" not in normalized.columns:
            normalized["doy_sin"] = np.sin(radians)
        if "doy_cos" not in normalized.columns:
            normalized["doy_cos"] = np.cos(radians)

    return normalized.sort_values(["station_id", "date"]).reset_index(drop=True)


def _resolve_station_metadata() -> pd.DataFrame:
    # 현재 저장소에서는 기본 registry를 우선 사용한다.
    registry = WeatherStationRegistry.default_kma_gangneung()
    station_meta = pd.DataFrame(
        [
            {
                "station_id": station.station_id,
                "stn_lat": station.lat,
                "stn_lon": station.lon,
            }
            for station in registry.stations
        ]
    )
    station_meta["station_id"] = pd.to_numeric(station_meta["station_id"], errors="coerce").astype("Int64")
    return station_meta.sort_values("station_id").reset_index(drop=True)


def _load_fire_label_table() -> pd.DataFrame:
    fire_path = PROC_DIR / "fire_events.parquet"
    fires = pd.read_parquet(fire_path)

    required_cols = {"station_id", "fire_date"}
    missing = sorted(required_cols - set(fires.columns))
    if missing:
        raise KeyError(f"fire_events.parquet 에 필요한 컬럼이 없습니다: {missing}")

    fires = fires.copy()
    fires["station_id"] = pd.to_numeric(fires["station_id"], errors="coerce").astype("Int64")
    fires["date"] = pd.to_datetime(fires["fire_date"]).dt.normalize()
    fires["fire_label"] = 1

    labels = (
        fires.groupby(["station_id", "date"], as_index=False)["fire_label"]
        .max()
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    labels["fire_label"] = labels["fire_label"].astype("Int64")
    return labels


def _load_weather_hourly() -> pd.DataFrame:
    weather_raw = load_weather_raw()

    stn_col = pick_column(weather_raw.columns, ["STN", "stn"])
    tm_col = pick_column(weather_raw.columns, ["TM", "datetime", "date"])
    ta_col = pick_column(weather_raw.columns, ["TA"])
    pop_col = pick_column(weather_raw.columns, ["POP"])
    precip_col = pick_column(weather_raw.columns, ["is_precip", "IS_PRECIP"])
    wd_sin_col = pick_column(weather_raw.columns, ["WD_sin", "wd_sin"])
    wd_cos_col = pick_column(weather_raw.columns, ["WD_cos", "wd_cos"])
    sky_col = pick_column(weather_raw.columns, ["SKY", "sky"])

    if stn_col is None or tm_col is None:
        raise ValueError("기상 원천 데이터에서 STN/TM 컬럼을 찾지 못했습니다.")

    keep_cols = [
        c
        for c in [stn_col, tm_col, ta_col, pop_col, precip_col, wd_sin_col, wd_cos_col, sky_col]
        if c is not None
    ]
    weather = weather_raw[keep_cols].copy()

    rename_map = {
        stn_col: "station_id",
        tm_col: "obs_ts",
    }
    optional_map = {
        ta_col: "TA",
        pop_col: "POP",
        precip_col: "is_precip",
        wd_sin_col: "WD_sin",
        wd_cos_col: "WD_cos",
        sky_col: "SKY",
    }
    rename_map.update({src: dst for src, dst in optional_map.items() if src is not None})
    weather = weather.rename(columns=rename_map)

    weather["station_id"] = pd.to_numeric(weather["station_id"], errors="coerce").astype("Int64")
    weather["obs_ts"] = parse_obs_datetime(weather["obs_ts"])
    weather = weather[weather["station_id"].isin(TARGET_STATIONS)].copy()
    weather = weather[weather["obs_ts"].notna()].copy()

    for col in ["TA", "POP", "is_precip", "WD_sin", "WD_cos"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
            weather[col] = weather[col].replace([-99, -999, -9999], np.nan)

    if "SKY" in weather.columns:
        weather["SKY"] = weather["SKY"].astype("string").str.strip().str.upper()

    weather["date"] = weather["obs_ts"].dt.normalize()
    weather["hour"] = weather["obs_ts"].dt.hour
    weather = weather.sort_values(["station_id", "obs_ts"]).reset_index(drop=True)
    return weather


def _build_daily_weather_base(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (station_id, date), day_df in weather_hourly.groupby(["station_id", "date"], sort=True):
        preferred = day_df[day_df["hour"].isin([0, 12])].copy()
        selected = preferred if not preferred.empty else day_df.copy()

        ta_00 = day_df.loc[day_df["hour"] == 0, "TA"] if "TA" in day_df.columns else pd.Series(dtype=float)
        ta_12 = day_df.loc[day_df["hour"] == 12, "TA"] if "TA" in day_df.columns else pd.Series(dtype=float)

        row = {
            "station_id": station_id,
            "date": pd.Timestamp(date).normalize(),
            "TA_day": pd.to_numeric(_series_or_empty(selected, "TA"), errors="coerce").mean(),
            "POP_day": pd.to_numeric(_series_or_empty(selected, "POP"), errors="coerce").mean(),
            "is_precip_day": pd.to_numeric(_series_or_empty(selected, "is_precip"), errors="coerce").max(),
            "WD_sin_day": pd.to_numeric(_series_or_empty(selected, "WD_sin"), errors="coerce").mean(),
            "WD_cos_day": pd.to_numeric(_series_or_empty(selected, "WD_cos"), errors="coerce").mean(),
            "SKY_day": _mode_or_na(selected["SKY"]) if "SKY" in selected.columns else pd.NA,
            "obs_cnt_sel": int(len(selected)),
            "has_00_obs": int((day_df["hour"] == 0).any()),
            "has_12_obs": int((day_df["hour"] == 12).any()),
        }

        if not ta_00.dropna().empty and not ta_12.dropna().empty:
            row["TA_dtr_day"] = float(abs(ta_12.mean() - ta_00.mean()))
        else:
            row["TA_dtr_day"] = np.nan

        rows.append(row)

    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("일 단위 기상 집계 결과가 비어 있습니다.")

    daily["station_id"] = pd.to_numeric(daily["station_id"], errors="coerce").astype("Int64")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["is_precip_day"] = pd.to_numeric(daily["is_precip_day"], errors="coerce").fillna(0).astype("Int64")
    daily["obs_cnt_sel"] = pd.to_numeric(daily["obs_cnt_sel"], errors="coerce").fillna(0).astype("Int64")
    daily["has_00_obs"] = pd.to_numeric(daily["has_00_obs"], errors="coerce").fillna(0).astype("Int64")
    daily["has_12_obs"] = pd.to_numeric(daily["has_12_obs"], errors="coerce").fillna(0).astype("Int64")
    daily["SKY_day"] = daily["SKY_day"].astype("string")
    return daily.sort_values(["station_id", "date"]).reset_index(drop=True)


def _build_station_day_calendar(station_ids: Iterable[int], min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    # 관측소별로 동일한 날짜 축을 갖는 wide/sequence 입력의 기준 테이블을 만든다.
    dates = pd.date_range(min_date, max_date, freq="D")
    calendar = pd.MultiIndex.from_product(
        [list(station_ids), dates],
        names=["station_id", "date"],
    ).to_frame(index=False)
    calendar["station_id"] = pd.to_numeric(calendar["station_id"], errors="coerce").astype("Int64")
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    return calendar.sort_values(["station_id", "date"]).reset_index(drop=True)


def validate_station_day_base(df: pd.DataFrame) -> None:
    # 데이터 엔지니어링 규칙을 코드로 강제한다.
    if df.duplicated(["station_id", "date"]).any():
        dup_count = int(df.duplicated(["station_id", "date"]).sum())
        raise ValueError(f"(station_id, date) 중복이 존재합니다: {dup_count}건")

    fire_values = set(df["fire_label"].dropna().astype(int).unique().tolist())
    if not fire_values.issubset({0, 1}):
        raise ValueError(f"fire_label 값이 이진이 아닙니다: {sorted(fire_values)}")

    station_meta = df[["station_id", "stn_lat", "stn_lon"]].drop_duplicates()
    if station_meta["station_id"].duplicated().any():
        raise ValueError("station metadata join 이 1:1 이 아닙니다.")

    if not df.sort_values(["station_id", "date"]).index.equals(df.index):
        raise ValueError("station_day_base 가 station_id, date 로 정렬되어 있지 않습니다.")


def build_station_day_base(output_path: Path | None = None) -> pd.DataFrame:
    if output_path is None:
        output_path = DATASET_DIR / "station_day_base.parquet"

    weather_hourly = _load_weather_hourly()
    daily_weather = _build_daily_weather_base(weather_hourly)
    station_meta = _resolve_station_metadata()
    labels = _load_fire_label_table()

    min_date = daily_weather["date"].min()
    max_date = daily_weather["date"].max()
    calendar = _build_station_day_calendar(station_meta["station_id"].tolist(), min_date, max_date)

    base = calendar.merge(station_meta, on="station_id", how="left", validate="many_to_one")
    base = base.merge(daily_weather, on=["station_id", "date"], how="left", validate="one_to_one")
    base = base.merge(labels, on=["station_id", "date"], how="left", validate="one_to_one")
    base["fire_label"] = base["fire_label"].fillna(0).astype("Int64")

    base["month"] = base["date"].dt.month.astype("Int64")
    base["dayofyear"] = base["date"].dt.dayofyear.astype("Int64")
    radians = 2 * np.pi * base["dayofyear"].astype(float) / 365.0
    base["doy_sin"] = np.sin(radians)
    base["doy_cos"] = np.cos(radians)

    base = base.sort_values(["station_id", "date"]).reset_index(drop=True)
    validate_station_day_base(base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)
    return base


def _build_dry_spell_days(series: pd.Series) -> pd.Series:
    # 현재 날짜를 제외한 직전 연속 무강수 일수를 계산한다.
    shifted = series.shift(1)
    result = []
    streak = 0
    for value in shifted:
        if pd.isna(value):
            streak = 0
            result.append(np.nan)
            continue
        if int(value) == 0:
            streak += 1
        else:
            streak = 0
        result.append(float(streak))
    return pd.Series(result, index=series.index)


def _add_lag_feature(df: pd.DataFrame, group_col: str, source_col: str, lag: int, out_col: str) -> None:
    df[out_col] = df.groupby(group_col, sort=False)[source_col].shift(lag)


def _add_roll_feature(
    df: pd.DataFrame,
    group_col: str,
    source_col: str,
    window: int,
    out_col: str,
    agg: str,
) -> None:
    grouped = df.groupby(group_col, sort=False)[source_col]
    shifted = grouped.shift(1)

    if agg == "mean":
        df[out_col] = shifted.groupby(df[group_col], sort=False).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        return

    if agg == "sum":
        df[out_col] = shifted.groupby(df[group_col], sort=False).rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)
        return

    raise ValueError(f"지원하지 않는 rolling 집계입니다: {agg}")


def build_catboost_dataset(
    station_day_base: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if output_path is None:
        output_path = DATASET_DIR / "catboost_dataset.parquet"

    if station_day_base is None:
        station_day_base = pd.read_parquet(DATASET_DIR / "station_day_base.parquet")

    df = ensure_station_day_base_schema(station_day_base)

    continuous_cols = ["TA_day", "TA_dtr_day", "POP_day", "WD_sin_day", "WD_cos_day"]
    binary_cols = ["is_precip_day", "has_00_obs", "has_12_obs"]
    quality_cols = ["obs_cnt_sel"]

    for col in continuous_cols + binary_cols + quality_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for lag in K_CAT:
        for col in continuous_cols + binary_cols + quality_cols + ["SKY_day"]:
            _add_lag_feature(df, "station_id", col, lag, f"{col}_lag{lag}")

    for window in W_CAT:
        for col in continuous_cols + quality_cols:
            _add_roll_feature(df, "station_id", col, window, f"{col}_rollmean_{window}", agg="mean")
        for col in binary_cols:
            _add_roll_feature(df, "station_id", col, window, f"{col}_rollsum_{window}", agg="sum")

        values = []
        for _, station_df in df.groupby("station_id", sort=False):
            shifted = station_df["SKY_day"].shift(1)
            for end_pos in range(len(station_df)):
                hist = shifted.iloc[max(0, end_pos - window + 1): end_pos + 1]
                hist = hist.dropna()
                values.append(_mode_or_na(hist) if not hist.empty else pd.NA)
        df[f"SKY_day_mode_{window}"] = pd.Series(values, index=df.index, dtype="string")

    df["dry_spell_days"] = (
        df.groupby("station_id", sort=False)["is_precip_day"]
        .apply(_build_dry_spell_days)
        .reset_index(level=0, drop=True)
    )

    df["station_id"] = df["station_id"].astype("string")
    df["fire_label"] = pd.to_numeric(df["fire_label"], errors="coerce").astype("Int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def build_tcn_dataset(
    station_day_base: pd.DataFrame | None = None,
    output_npz_path: Path | None = None,
) -> dict:
    if output_npz_path is None:
        output_npz_path = DATASET_DIR / "tcn_dataset.npz"

    if station_day_base is None:
        station_day_base = pd.read_parquet(DATASET_DIR / "station_day_base.parquet")

    df = ensure_station_day_base_schema(station_day_base)

    for sky_code in TCN_SKY_LEVELS:
        # TCN 입력에서는 SKY를 one-hot 으로 고정된 채널에 넣는다.
        df[f"SKY_{sky_code}"] = (df["SKY_day"].astype("string") == sky_code).astype("int64")

    dyn_cols = [
        "TA_day",
        "TA_dtr_day",
        "POP_day",
        "is_precip_day",
        "WD_sin_day",
        "WD_cos_day",
        "SKY_DB01",
        "SKY_DB02",
        "SKY_DB03",
        "SKY_DB04",
        "obs_cnt_sel",
        "has_00_obs",
        "has_12_obs",
    ]
    static_cols = ["station_id", "stn_lat", "stn_lon"]
    generated_dyn_cols = [
        "TA_day",
        "TA_dtr_day",
        "POP_day",
        "is_precip_day",
        "WD_sin_day",
        "WD_cos_day",
        "SKY_DB01",
        "SKY_DB02",
        "SKY_DB03",
        "SKY_DB04",
        "obs_cnt_sel",
        "has_00_obs",
        "has_12_obs",
    ]

    df["station_id_encoded"] = pd.factorize(df["station_id"], sort=True)[0].astype("int64")
    static_value_cols = ["station_id_encoded", "stn_lat", "stn_lon"]

    for col in ["TA_day", "TA_dtr_day", "POP_day", "is_precip_day", "WD_sin_day", "WD_cos_day", "obs_cnt_sel", "has_00_obs", "has_12_obs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sequences = []
    static_rows = []
    labels = []
    index_rows = []

    for station_id, station_df in df.groupby("station_id", sort=True):
        station_df = station_df.sort_values("date").reset_index(drop=True)
        dyn_values = station_df[generated_dyn_cols].astype(float)
        static_values = station_df[static_value_cols].astype(float)

        for end_idx in range(TCN_SEQUENCE_LENGTH, len(station_df)):
            history = dyn_values.iloc[end_idx - TCN_SEQUENCE_LENGTH:end_idx]
            if history.isna().any().any():
                continue

            target_row = station_df.iloc[end_idx]
            if pd.isna(target_row["fire_label"]):
                continue

            sequences.append(history.to_numpy(dtype=np.float32))
            static_rows.append(static_values.iloc[end_idx].to_numpy(dtype=np.float32))
            labels.append(int(target_row["fire_label"]))
            index_rows.append(
                {
                    "station_id": station_id,
                    "date": target_row["date"],
                    "fire_label": int(target_row["fire_label"]),
                }
            )

    if not sequences:
        raise ValueError("TCN 학습용 시퀀스를 생성하지 못했습니다. 결측치 또는 기간 길이를 확인하세요.")

    X_dyn = np.stack(sequences).astype(np.float32)
    X_static = np.stack(static_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, X_dyn=X_dyn, X_static=X_static, y=y)

    index_path = output_npz_path.with_name("tcn_index.parquet")
    pd.DataFrame(index_rows).to_parquet(index_path, index=False)

    meta = {
        "sequence_length": TCN_SEQUENCE_LENGTH,
        "dynamic_features": dyn_cols,
        "static_features": static_cols,
        "static_saved_columns": ["station_id_encoded", "stn_lat", "stn_lon"],
        "num_samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "npz_path": str(output_npz_path),
        "index_path": str(index_path),
    }
    meta_path = output_npz_path.with_name("tcn_dataset_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return meta


def build_all_model_datasets() -> dict:
    station_day_base = build_station_day_base()
    catboost_df = build_catboost_dataset(station_day_base=station_day_base)
    tcn_meta = build_tcn_dataset(station_day_base=station_day_base)

    summary = {
        "station_day_base_rows": int(len(station_day_base)),
        "station_day_base_positive": int(station_day_base["fire_label"].sum()),
        "catboost_rows": int(len(catboost_df)),
        "tcn_samples": int(tcn_meta["num_samples"]),
        "tcn_positive": int(tcn_meta["positive_samples"]),
        "dataset_dir": str(DATASET_DIR),
    }

    summary_path = DATASET_DIR / "dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
