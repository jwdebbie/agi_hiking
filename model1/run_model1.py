# model1/run_model1.py
"""
모델1: SW3(참여율) + SW4(도전 의지) 종합 점수
"""

import pandas as pd

from .motivation_stamp import MotivationStampCalculator
from .motivation_distance import MotivationDistanceCalculator


def run_model1(limit: int | None = None) -> pd.DataFrame:
    sw3_calc = MotivationStampCalculator()
    sw4_calc = MotivationDistanceCalculator()

    sw3_df = sw3_calc.run(limit=limit)   # user_key, participation_rate, sw3_score, ...
    sw4_df = sw4_calc.run(limit=limit)   # user_key, avg_top_distance_km, sw4_score

    merged = sw3_df.merge(sw4_df, on="user_key", how="inner")

    # 1번 모델 결과: SW3 결과 + SW4 결과 100점 만점 환산 (단순 평균, 나중에 가중치 조절 가능)
    merged["model1_score"] = (merged["sw3_score"] + merged["sw4_score"]) / 2.0

    return merged[
        [
            "user_key",
            "participation_rate",
            "sw3_score",
            "avg_top_distance_km",
            "sw4_score",
            "model1_score",
            "active_weeks",
            "total_weeks",
        ]
    ]


if __name__ == "__main__":
    out = run_model1(limit=None)
    print(out.head())
    print("users in model1:", len(out))
