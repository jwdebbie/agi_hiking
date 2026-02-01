# model2/trendscore.py
"""
SW6 - 개선율 점수 (trend score)

입력: SW2에서 계산한 개별 속도들 (각 투어 speed_kmh)
계산:
  초기 2회 평균 속도
  최근 3회 평균 속도
  개선율 = (최근 - 초기) / 초기
출력: 0~100 개선율 점수
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.data_loader import load_total_hiking_data


class TrendScoreCalculator:
    """최근 추세 점수 계산 (SW6)"""

    def __init__(self, initial_count: int = 2, recent_count: int = 3, max_score: float = 100.0):
        self.initial_count = initial_count
        self.recent_count = recent_count
        self.max_score = max_score

    @staticmethod
    def _prepare_speed_df(df: pd.DataFrame) -> pd.DataFrame:
        """각 투어별 km/h 속도 및 시간순 정렬."""
        df = df.copy()
        df["distance_km"] = df["total_distance_m"] / 1000.0
        df["duration_hr"] = df["total_duration_sec"] / 3600.0
        df["speed_kmh"] = df["distance_km"] / df["duration_hr"].replace(0, np.nan)
        df["speed_kmh"] = df["speed_kmh"].fillna(0.0)
        df["complete_date"] = pd.to_datetime(df["complete_date"])
        df = df.sort_values(["user_key", "complete_date"])
        return df

    def _calc_improvement_for_user(self, g: pd.DataFrame) -> float:
        """한 user의 속도 개선율 계산."""
        g = g.sort_values("complete_date")

        if len(g) < self.initial_count + 1:
            # 투어 개수가 너무 적으면 개선율 0
            return 0.0

        initial = g.head(self.initial_count)["speed_kmh"].mean()
        recent = g.tail(self.recent_count)["speed_kmh"].mean()

        if initial <= 0:
            return 0.0

        improvement = (recent - initial) / initial
        return float(round(improvement, 4))

    def compute_improvement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        user별 개선율 계산.

        input df: total_hiking_data (user_key, total_distance_m, total_duration_sec, complete_date ...)
        return: DataFrame[user_key, improvement_rate, tour_count]
        """
        df = self._prepare_speed_df(df)

        results = []
        for user, g in df.groupby("user_key"):
            rate = self._calc_improvement_for_user(g)
            results.append(
                {
                    "user_key": user,
                    "improvement_rate": rate,
                    "tour_count": len(g),
                    "raw_value": rate,
                }
            )
        return pd.DataFrame(results)

    def normalize_to_score(self, df_imp: pd.DataFrame) -> pd.DataFrame:
        """
        개선율 → 0~max_score 점수로 변환.

        예시:
          -50% (‑0.5) ~ +50% (+0.5) 구간을 0~100으로 매핑.
        """
        df = df_imp.copy()
        imp = df["improvement_rate"].clip(-0.5, 0.5)
        df["sw6_score"] = (imp + 0.5) * self.max_score
        return df[["user_key", "improvement_rate", "tour_count", "sw6_score"]]

    def run(self, limit: int | None = None) -> pd.DataFrame:
        raw = load_total_hiking_data(limit=limit)
        raw = raw[["user_key", "total_distance_m", "total_duration_sec", "complete_date"]]
        imp_df = self.compute_improvement(raw)
        scored_df = self.normalize_to_score(imp_df)
        return scored_df


if __name__ == "__main__":
    calc = TrendScoreCalculator()
    out = calc.run(limit=None)
    print("users in sw2:", len(out))
    print(out.head())
