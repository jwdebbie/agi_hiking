# model1/motivationstamp.py
"""
SW3 - 참여율 점수 (motivation_stamp)

입력:
  Member: join_date
  MemberTourStampCourse: complete_date

계산:
  활동한 주수 / 가입 후 경과 주수 → 참여율(0~1)
출력:
  0~100 참여율 점수
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 상위 폴더에서 data_loader import
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from data.data_loader import load_total_hiking_data


class MotivationStampCalculator:
    def __init__(self, max_score: float = 100.0):
        self.max_score = max_score

    @staticmethod
    def _weeks_since_join(join_date, current_date=None) -> int:
        if isinstance(join_date, str):
            join_dt = datetime.fromisoformat(str(join_date))
        else:
            join_dt = join_date

        if current_date is None:
            current_dt = datetime.now()
        elif isinstance(current_date, str):
            current_dt = datetime.fromisoformat(str(current_date))
        else:
            current_dt = current_date

        days = (current_dt - join_dt).days
        return max(1, days // 7)

    @staticmethod
    def _week_id(dt: pd.Timestamp):
        iso = dt.isocalendar()
        return (iso.year, iso.week)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        total_hiking_data에서 참여율 계산에 필요한 컬럼만 사용:
        user_key, join_date, complete_date
        """
        df = df[["user_key", "join_date", "complete_date"]].copy()
        df["join_date"] = pd.to_datetime(df["join_date"])
        df["complete_date"] = pd.to_datetime(df["complete_date"])
        return df

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        user별 참여율(0~1) 계산.
        participation_rate = 활동한 주수 / 가입 후 경과 주수
        """
        df = self._prepare_data(df)
        results = []

        for user, g in df.groupby("user_key"):
            g = g.sort_values("complete_date")
            join_date = g["join_date"].iloc[0]

            stamp_weeks = {self._week_id(t) for t in g["complete_date"]}
            active_weeks = len(stamp_weeks)

            total_weeks = self._weeks_since_join(join_date, g["complete_date"].max())

            rate = active_weeks / total_weeks if total_weeks > 0 else 0.0
            results.append(
                {
                    "user_key": user,
                    "participation_rate": round(rate, 4),
                    "active_weeks": active_weeks,
                    "total_weeks": total_weeks,
                    "raw_value": rate,
                }
            )

        return pd.DataFrame(results)

    def normalize_to_score(self, df_rate: pd.DataFrame) -> pd.DataFrame:
        """
        참여율(0~1)을 0~max_score 점수로 변환.
        """
        df = df_rate.copy()
        df["sw3_score"] = df["participation_rate"].clip(0, 1) * self.max_score
        return df[["user_key", "participation_rate", "sw3_score", "active_weeks", "total_weeks"]]

    def run(self, limit: int | None = None) -> pd.DataFrame:
        raw = load_total_hiking_data(limit=limit)
        rate_df = self.calculate_batch(raw)
        scored_df = self.normalize_to_score(rate_df)
        return scored_df


if __name__ == "__main__":
    calc = MotivationStampCalculator()
    out = calc.run(limit=None)
    print(out.head())
    print("users in SW3:", len(out))
