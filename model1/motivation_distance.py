# model1/motivationdistance.py
"""
SW4 - 거리 기반 도전 의지 점수 (motivation_distance)

입력:
  Member: member_latitude, member_longitude (회원 집 좌표)
  Stamp: stamp_latitude, stamp_longitude (코스 시작점)

계산:
  집 → 코스 시작점 거리 (km)
  각 user의 상위 N개 거리 평균
출력:
  0~100 도전 의지 점수
"""

import sys
from pathlib import Path
from math import radians, cos, sin, sqrt, atan2

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from data.data_loader import load_total_hiking_data


class MotivationDistanceCalculator:
    def __init__(self, top_n: int = 5, max_score: float = 100.0):
        self.top_n = top_n
        self.max_score = max_score

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        total_hiking_data에서 거리 계산에 필요한 컬럼만 사용:
        user_key, member_latitude, member_longitude, stamp_latitude, stamp_longitude
        """
        df = df[["user_key", "member_latitude", "member_longitude", "stamp_latitude", "stamp_longitude"]].copy()
        
        # 숫자 변환(문자열로 들어왔을 때 대비)
        for c in ["member_latitude", "member_longitude", "stamp_latitude", "stamp_longitude"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        user별 상위 N개 거리 평균(km) 계산.
        """
        df = self._prepare_data(df)
        results = []

        for user, g in df.groupby("user_key"):
            home_lat = g["member_latitude"].iloc[0]
            home_lon = g["member_longitude"].iloc[0]

            if pd.isna(home_lat) or pd.isna(home_lon):
                avg_dist = 0.0
            else:
                dists = []
                for _, row in g.iterrows():
                    lat, lon = row["stamp_latitude"], row["stamp_longitude"]
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    dists.append(self._haversine(home_lat, home_lon, lat, lon))

                if not dists:
                    avg_dist = 0.0
                else:
                    top = sorted(dists, reverse=True)[: self.top_n]
                    avg_dist = float(np.mean(top))

            results.append(
                {
                    "user_key": user,
                    "avg_top_distance_km": round(avg_dist, 2),
                    "raw_value": avg_dist,
                }
            )

        return pd.DataFrame(results)

    def normalize_to_score(self, df_dist: pd.DataFrame) -> pd.DataFrame:
        """
        거리 → 0~max_score 점수로 변환.
        먼 거리일수록 도전 의지 ↑
        """
        df = df_dist.copy()
        s = df["avg_top_distance_km"]
        min_s, max_s = s.min(), s.max()
        if max_s == min_s:
            df["sw4_score"] = self.max_score / 2.0
        else:
            df["sw4_score"] = (s - min_s) / (max_s - min_s) * self.max_score

        return df[["user_key", "avg_top_distance_km", "sw4_score"]]

    def run(self, limit: int | None = None) -> pd.DataFrame:
        raw = load_total_hiking_data(limit=limit)
        dist_df = self.calculate_batch(raw)
        scored_df = self.normalize_to_score(dist_df)
        return scored_df


if __name__ == "__main__":
    calc = MotivationDistanceCalculator()
    out = calc.run(limit=None)
    print(out.head())
    print("users in SW4:", len(out))
