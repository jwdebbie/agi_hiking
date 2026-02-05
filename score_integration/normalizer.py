# model3/normalizer.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

# 상위 폴더 경로 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.data_loader import load_total_hiking_data
from model1.run_model1 import run_model1
from model2.run_model2 import run_model2


class ScoreNormalizer:
    """점수 정규화 + 등급 부여"""

    def __init__(self):
        pass

    # ----- 기존 Poisson 기반 정규화 함수들 -----

    def normalize_by_poisson(self, value, cohort_values):
        if len(cohort_values) == 0:
            return 50.0
        lambda_param = np.mean(cohort_values)
        if lambda_param <= 0:
            return 50.0
        cumulative_prob = poisson.cdf(value, lambda_param)
        score = cumulative_prob * 100
        return round(score, 2)

    def normalize_module_scores(self, df, module_col, cohort_col="cohort", max_score=100):
        df = df.copy()
        score_col = f"{module_col}_norm"  # 혼동 피하려고 _norm으로 이름 변경

        for cohort in df[cohort_col].unique():
            cohort_mask = df[cohort_col] == cohort
            cohort_values = df.loc[cohort_mask, module_col].values

            for idx in df[cohort_mask].index:
                user_value = df.loc[idx, module_col]
                normalized = self.normalize_by_poisson(user_value, cohort_values)
                scaled_score = (normalized / 100) * max_score
                df.loc[idx, score_col] = scaled_score

        return df

    def calculate_total_score(self, df, score_columns):
        df = df.copy()
        df["total_score"] = df[score_columns].sum(axis=1)
        max_total = len(score_columns) * 100
        df["percentage"] = (df["total_score"] / max_total * 100).round(2)
        return df

    def assign_grade(self, total_score, max_score):
        percentage = (total_score / max_score) * 100

        if percentage >= 90:
            return "S"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B"
        elif percentage >= 60:
            return "C"
        else:
            return "D"

    def add_grades(self, df, total_col="total_score"):
        df = df.copy()
        max_score = 4 * 100  # 모듈 네 개
        df["grade"] = df[total_col].apply(lambda x: self.assign_grade(x, max_score))
        return df

    # ----- 여기서부터 이 프로젝트용 파이프라인 -----

    @staticmethod
    def _make_cohort(birth_year: int) -> str:
        """출생연도로 나이대(cohort) 만들기 예시: 20대/30대/40대/50대+."""
        if pd.isna(birth_year):
            return "기타"

        age = 2025 - int(birth_year)  # 기준 연도는 데이터 기준 연도에 맞춰서 조정
        if age < 20:
            return "10대 이하"
        elif age < 30:
            return "20대"
        elif age < 40:
            return "30대"
        elif age < 50:
            return "40대"
        else:
            return "50대 이상"

    def build_base_df(self, limit=None) -> pd.DataFrame:
        """total_hiking_data에서 user_key, birth_year만 뽑아 cohort 생성."""
        raw = load_total_hiking_data(limit=limit)
        base = (
            raw[["user_key", "birth_year"]]
            .drop_duplicates("user_key")
            .reset_index(drop=True)
        )
        base["cohort"] = base["birth_year"].apply(self._make_cohort)
        return base
    def run(self, limit=None) -> pd.DataFrame:
        """
        모델1/2 결과 + 나이대를 합쳐서
        네 모듈 점수(SW3, SW4, SW2, SW6)를 코호트별 정규화 → 총점 → S/A/B/C/D 등급.
        """
        # 1) 기본 user + cohort
        base = self.build_base_df(limit=limit)

        # 2) 모델1, 모델2 결과 가져오기
        m1 = run_model1(limit=limit)  # user_key, sw3_score, sw4_score, model1_score ...
        m2 = run_model2(limit=limit)  # user_key, sw2_score, sw6_score, model2_score ...

        # 3) user_key 기준으로 병합
        df = (
            base.merge(
                m1[["user_key", "sw3_score", "sw4_score"]],
                on="user_key",
                how="left",
            )
            .merge(
                m2[["user_key", "sw2_score", "sw6_score"]],
                on="user_key",
                how="left",
            )
        )

        # 결측치는 0으로
        for col in ["sw3_score", "sw4_score", "sw2_score", "sw6_score"]:
            df[col] = df[col].fillna(0.0)

        # 4) 나이대(cohort)별로 각 모듈 점수 정규화
        module_cols = ["sw3_score", "sw4_score", "sw2_score", "sw6_score"]
        for col in module_cols:
            df = self.normalize_module_scores(
                df,
                module_col=col,
                cohort_col="cohort",
                max_score=100,
            )

        # _norm 컬럼들만 사용해서 총점 계산
        norm_cols = [f"{c}_norm" for c in module_cols]
        df = self.calculate_total_score(df, norm_cols)

        # 5) 등급 부여
        df = self.add_grades(df, total_col="total_score")

        return df[
            [
                "user_key",
                "birth_year",
                "cohort",
                "sw3_score_norm",
                "sw4_score_norm",
                "sw2_score_norm",
                "sw6_score_norm",
                "total_score",
                "percentage",
                "grade",
            ]
        ]

        
if __name__ == "__main__":
    normalizer = ScoreNormalizer()
    out = normalizer.run(limit=None)
    print(out.head())
    print("rows:", len(out))
