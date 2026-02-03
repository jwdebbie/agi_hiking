# meta_models/pseudo_labels.py
# 학습 타깃 y(정답)를 만든다.(진짜 라벨이 없으니, 기존 모델의 점수나 간단한 점수로 “학습 가능” 상태를 만든다)
# “가설 기반 점수(통계/규칙)”를 AI의 학습 목표로 만들어주는 다리

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_0_100(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(x, dtype=float)
    return 100.0 * (x - lo) / (hi - lo)


def try_generate_labels_from_existing_modules(df: pd.DataFrame):
    """
    If your existing Model1/Model2 calculators are importable and expose a clean API,
    you can wire them here. This function tries but falls back safely.

    Return:
      motivation_score (np.ndarray) or None
      fitness_score (np.ndarray) or None
      trend_score (np.ndarray) or None
    """
    # NOTE: We keep this robust: if import fails, return None(s).
    try:
        # Example (only works if your modules provide callable classes/functions):
        # from model1.motivation_distance import MotivationDistanceCalculator
        # from model1.motivation_stamp import MotivationStampCalculator
        # from model2.healthscore import HealthScoreCalculator
        # from model2.trendscore import TrendScoreCalculator
        #
        # dist_calc = MotivationDistanceCalculator(...)
        # stamp_calc = MotivationStampCalculator(...)
        # motivation_score = combine(...)
        #
        # fitness_calc = HealthScoreCalculator(...)
        # trend_calc = TrendScoreCalculator(...)
        #
        # return motivation_score, fitness_score, trend_score
        return None, None, None
    except Exception:
        return None, None, None


def fallback_motivation_label(df: pd.DataFrame) -> np.ndarray:
    """
    Simple pseudo-label that mimics:
      - more hikes recently -> higher
      - farther home_to_stamp_km (top range) -> higher "challenge"
    """
    recent = df["hikes_last_30d"].to_numpy(dtype=float)
    dist = df["home_to_stamp_km"].to_numpy(dtype=float)

    # challenge: emphasize long distances (sqrt to soften)
    challenge = np.sqrt(np.clip(dist, 0, None))

    score = 0.6 * minmax_0_100(recent) + 0.4 * minmax_0_100(challenge)
    return np.clip(score, 0, 100)


def fallback_fitness_trend_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    fitness_score ~ higher avg_speed_mps
    trend_score ~ higher speed_trend (improving)
    """
    speed = df["avg_speed_mps"].to_numpy(dtype=float)
    trend = df["speed_trend"].to_numpy(dtype=float)

    fitness = minmax_0_100(speed)
    trend_s = minmax_0_100(trend)
    return np.clip(fitness, 0, 100), np.clip(trend_s, 0, 100)


def make_pseudo_labels(df: pd.DataFrame, which: str):
    """
    which: "motivation" or "fitness_trend"
    """
    mot, fit, tr = try_generate_labels_from_existing_modules(df)

    if which == "motivation":
        if mot is not None:
            return mot.astype(np.float32)
        return fallback_motivation_label(df).astype(np.float32)

    if which == "fitness_trend":
        if fit is not None and tr is not None:
            return fit.astype(np.float32), tr.astype(np.float32)
        return tuple(x.astype(np.float32) for x in fallback_fitness_trend_labels(df))

    raise ValueError(f"Unknown which={which}")
