# model2/run_model2.py
"""
모델2: health_score(SW2) + trend_score(SW6) 종합 점수
"""

from .health_score import HealthScoreCalculator
from .trend_score import TrendScoreCalculator
import pandas as pd


def run_model2(limit: int | None = None) -> pd.DataFrame:
    health_calc = HealthScoreCalculator()
    trend_calc = TrendScoreCalculator()

    sw2_df = health_calc.run(limit=limit)        # user_key, avg_speed_kmh, sw2_score
    sw6_df = trend_calc.run(limit=limit)         # user_key, improvement_rate, tour_count, sw6_score

    merged = sw2_df.merge(sw6_df, on="user_key", how="inner")

    # 2번 모델 결과: SW2 + SW6 결과 100점 만점 환산 (단순 평균, 나중에 가중치 조정 가능)
    merged["model2_score"] = (merged["sw2_score"] + merged["sw6_score"]) / 2.0

    return merged[
        [
            "user_key",
            "avg_speed_kmh",
            "sw2_score",
            "improvement_rate",
            "sw6_score",
            "model2_score",
            "tour_count",
        ]
    ]


if __name__ == "__main__":
    out = run_model2(limit=1000)
    print(out.head())
