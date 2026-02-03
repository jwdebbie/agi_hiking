# meta_models/features.py
# total_hiking_data의 원본 컬럼에서 학습용 입력 X(feature)를 만든다.
# 가설에서 중요했던 “거리/참여/개선”을 AI 입력으로 재표현하는 파트

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance in km.
    Inputs can be numpy arrays / pandas series.
    """
    R = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complete_date"] = pd.to_datetime(df["complete_date"], errors="coerce")
    df["weekday"] = df["complete_date"].dt.weekday.fillna(0).astype(int)  # 0~6
    df["month"] = df["complete_date"].dt.month.fillna(1).astype(int)      # 1~12
    return df


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses columns from total_hiking_data view:
    - user_key, complete_date, total_distance_m, total_duration_sec,
      stamp_latitude/longitude, member_latitude/longitude, birth_year (optional)
    """
    df = df.copy()

    # Distance between home and stamp start point
    df["home_to_stamp_km"] = haversine_km(
        df["member_latitude"], df["member_longitude"],
        df["stamp_latitude"], df["stamp_longitude"]
    )

    # Hiking speed (m/s)
    dur = df["total_duration_sec"].replace(0, np.nan).astype(float)
    df["avg_speed_mps"] = (df["total_distance_m"].astype(float) / dur).fillna(0.0)

    return df


def add_user_history_features(df: pd.DataFrame, window_days: int = 30, trend_n: int = 5) -> pd.DataFrame:
    """
    Builds user-level rolling features:
    - hikes_last_30d
    - days_since_last_hike
    - total_hike_count (cumulative)
    - speed_trend (slope over last n records)
    - speed_std (std over last n records)
    """
    df = df.copy()
    df["complete_date"] = pd.to_datetime(df["complete_date"], errors="coerce")

    df = df.sort_values(["user_key", "complete_date"])

    # cumulative count
    df["total_hike_count"] = df.groupby("user_key").cumcount() + 1

    # days since last hike
    prev_date = df.groupby("user_key")["complete_date"].shift(1)
    df["days_since_last_hike"] = (df["complete_date"] - prev_date).dt.total_seconds() / (3600 * 24)
    df["days_since_last_hike"] = df["days_since_last_hike"].fillna(window_days).clip(0, window_days)

    # hikes_last_30d (rolling count by time window)
    # We'll approximate using expanding window with a time-based filter per row (vectorized per user is heavier).
    # For assignment/demo, this is acceptable. If you want, we can optimize later.
    hikes_last = []
    for u, g in df.groupby("user_key", sort=False):
        dates = g["complete_date"].values.astype("datetime64[ns]")
        cnts = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            start = dates[i] - np.timedelta64(window_days, "D")
            cnts[i] = int(((dates >= start) & (dates <= dates[i])).sum())
        hikes_last.extend(list(cnts))
    df["hikes_last_30d"] = hikes_last

    # speed trend/std over last trend_n records per user
    def _trend(vals: np.ndarray) -> float:
        # slope of y over x=0..n-1
        n = len(vals)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = vals.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    speed_trend = []
    speed_std = []
    for u, g in df.groupby("user_key", sort=False):
        speeds = g["avg_speed_mps"].astype(float).values
        t = np.zeros(len(g), dtype=float)
        s = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            start = max(0, i - trend_n + 1)
            window = speeds[start:i+1]
            t[i] = _trend(window)
            s[i] = float(np.std(window)) if len(window) > 1 else 0.0
        speed_trend.extend(list(t))
        speed_std.extend(list(s))

    df["speed_trend"] = speed_trend
    df["speed_std"] = speed_std

    return df


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      X: (N, D) float32
      cols: actually used cols
    """
    X = df[feature_cols].copy()

    # Fill missing numeric values
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X.values.astype(np.float32), feature_cols
