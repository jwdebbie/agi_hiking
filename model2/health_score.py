# model2/healthscore.py
"""
SW2 - 체력 점수 (health score)

입력: total_distance_m, total_duration_sec
계산:
  각 투어 속도 = 거리(km) / 시간(hour)
  전체 평균 속도 = 모든 투어 속도의 합 / 투어 개수
출력: 0~100 체력 점수
"""

# model2/health_score.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.data_loader import load_total_hiking_data



class HealthScoreCalculator:
    """신체적 건강 점수 계산 (SW2)"""

    def __init__(self, max_score: float = 100.0):
        self.max_score = max_score

    @staticmethod
    def _add_speed_cols(df: pd.DataFrame) -> pd.DataFrame:
        """거리/시간에서 km/h 속도 계산."""
        df = df.copy()
        df["distance_km"] = df["total_distance_m"] / 1000.0
        df["duration_hr"] = df["total_duration_sec"] / 3600.0
        df["speed_kmh"] = df["distance_km"] / df["duration_hr"].replace(0, np.nan)
        df["speed_kmh"] = df["speed_kmh"].fillna(0.0)
        return df

    def compute_user_avg_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 user별 전체 평균 속도(km/h) 계산.

        input df: total_hiking_data (user_key, total_distance_m, total_duration_sec, complete_date ...)
        return: DataFrame[user_key, avg_speed_kmh]
        """
        df = self._add_speed_cols(df)
        grouped = df.groupby("user_key")["speed_kmh"].mean().reset_index()
        grouped = grouped.rename(columns={"speed_kmh": "avg_speed_kmh"})
        grouped["raw_value"] = grouped["avg_speed_kmh"]
        return grouped

    def normalize_to_score(self, df_speed: pd.DataFrame) -> pd.DataFrame:
        """
        평균 속도 → 0~max_score 점수로 변환 (간단 min-max).
        """
        df = df_speed.copy()
        s = df["avg_speed_kmh"]
        min_s, max_s = s.min(), s.max()

        if max_s == min_s:
            df["sw2_score"] = self.max_score / 2.0
        else:
            df["sw2_score"] = (s - min_s) / (max_s - min_s) * self.max_score

        return df[["user_key", "avg_speed_kmh", "sw2_score"]]

    def run(self, limit: int | None = None) -> pd.DataFrame:
        """
        외부에서 호출하는 메인 함수.
        """
        raw = load_total_hiking_data(limit=limit)
        # 필요한 컬럼만 사용
        raw = raw[["user_key", "total_distance_m", "total_duration_sec", "complete_date"]]
        speed_df = self.compute_user_avg_speed(raw)
        scored_df = self.normalize_to_score(speed_df)
        return scored_df


if __name__ == "__main__":
    calc = HealthScoreCalculator()
    out = calc.run(limit=None) 
    print("users in sw2:", len(out))
    print(out.head())
