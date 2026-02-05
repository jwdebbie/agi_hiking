"""
pseudo_labels.py: Motivation 점수 의사 라벨 생성 (motivation_maml용)

역할:
  딥러닝 모델 학습을 위한 의사 라벨(pseudo-label)을 생성.
  기존 통계 모델(motivation_stat)의 Calculator 클래스를 활용하여
  라벨을 생성하고, 불가능한 경우 휴리스틱 기반 fallback을 사용.

Input:
  - 피처가 생성된 DataFrame (features.py 출력)
  - 필수 컬럼: user_key, hikes_last_30d, home_to_stamp_km

Output:
  - motivation_score 컬럼이 추가된 DataFrame
  - 점수 범위: 0~100
"""

import numpy as np
import pandas as pd
from typing import Optional


def minmax_normalize(series: pd.Series, min_val: float = 0, max_val: float = 100) -> pd.Series:
    """
    Min-max 정규화를 수행.
    
    Args:
        series: 정규화할 Series
        min_val: 최소값
        max_val: 최대값
    
    Returns:
        [min_val, max_val] 범위로 정규화된 Series
    """
    s_min = series.min()
    s_max = series.max()
    
    if s_max == s_min:
        return pd.Series([min_val] * len(series), index=series.index)
    
    normalized = (series - s_min) / (s_max - s_min)
    scaled = normalized * (max_val - min_val) + min_val
    return scaled


def generate_motivation_score_from_stat_model(df: pd.DataFrame) -> pd.Series:
    """
    기존 통계 모델(motivation_stat)의 Calculator 클래스를 사용하여 motivation 점수를 생성.
    
    Args:
        df: 원본 데이터가 포함된 DataFrame
        
    Returns:
        motivation_score (0-100) Series
    """
    try:
        # 기존 motivation_stat 모델의 클래스 임포트 시도
        import sys
        import os
        
        # 프로젝트 루트를 sys.path에 추가
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # motivation_stat 폴더의 Calculator 클래스 임포트
        from motivation_stat.motivation_stamp import MotivationStampCalculator
        from motivation_stat.motivation_distance import MotivationDistanceCalculator
        
        print("✓ motivation_stat 모델의 Calculator 클래스를 사용합니다")
        
        # 원본 데이터에서 필요한 컬럼만 추출
        # SW3: user_key, join_date, complete_date 필요
        # SW4: user_key, member_latitude, member_longitude, stamp_latitude, stamp_longitude 필요
        
        # SW3 계산
        sw3_calc = MotivationStampCalculator()
        sw3_df = sw3_calc.calculate_batch(df)  # user_key, participation_rate, sw3_score 반환
        sw3_df = sw3_calc.normalize_to_score(sw3_df)  # sw3_score 계산
        
        # SW4 계산
        sw4_calc = MotivationDistanceCalculator()
        sw4_df = sw4_calc.calculate_batch(df)  # user_key, avg_top_distance_km 반환
        sw4_df = sw4_calc.normalize_to_score(sw4_df)  # sw4_score 계산
        
        # user_key 기준으로 병합
        score_df = sw3_df[['user_key', 'sw3_score']].merge(
            sw4_df[['user_key', 'sw4_score']], 
            on='user_key', 
            how='outer'
        )
        
        # 결측값 처리
        score_df['sw3_score'] = score_df['sw3_score'].fillna(50.0)
        score_df['sw4_score'] = score_df['sw4_score'].fillna(50.0)
        
        # 두 점수의 평균으로 motivation 계산
        score_df['motivation_score'] = (score_df['sw3_score'] + score_df['sw4_score']) / 2.0
        
        # 원본 DataFrame과 병합 (user_key 기준)
        result = df.merge(
            score_df[['user_key', 'motivation_score']], 
            on='user_key', 
            how='left'
        )
        
        # 병합 후에도 결측값이 있으면 fallback 사용
        if result['motivation_score'].isnull().any():
            print(f"⚠ {result['motivation_score'].isnull().sum()}개 행에 대해 fallback 사용")
            mask = result['motivation_score'].isnull()
            result.loc[mask, 'motivation_score'] = generate_motivation_score_fallback(result[mask])
        
        return result['motivation_score']
        
    except (ImportError, ModuleNotFoundError, AttributeError, KeyError) as e:
        print(f"⚠ motivation_stat 모델을 사용할 수 없습니다: {e}")
        print("→ Fallback 휴리스틱을 사용하여 motivation_score를 생성합니다")
        return generate_motivation_score_fallback(df)


def generate_motivation_score_fallback(df: pd.DataFrame) -> pd.Series:
    """
    Fallback: 휴리스틱 기반 motivation 점수 생성
    
    로직:
      - 60% 가중치: 최근 활동 빈도 (hikes_last_30d)
        → 자주 등산하는 사람일수록 동기가 높음
      - 40% 가중치: 탐험 거리 (sqrt of home_to_stamp_km)
        → 집에서 먼 곳까지 가는 사람일수록 동기가 높음
        → sqrt를 사용하여 극단적으로 먼 거리에 대한 보상을 완화
    
    Args:
        df: 피처가 포함된 DataFrame
    
    Returns:
        motivation_score (0-100) Series
    """
    # 활동 빈도 컴포넌트 (최근 30일 등산 횟수)
    if 'hikes_last_30d' in df.columns:
        activity_component = minmax_normalize(df['hikes_last_30d'], 0, 1)
    else:
        activity_component = pd.Series([0.5] * len(df), index=df.index)
    
    # 탐험 의지 컴포넌트 (집-스탬프 거리)
    if 'home_to_stamp_km' in df.columns:
        # sqrt를 사용하여 먼 거리에 대한 보상을 dampening
        exploration_component = minmax_normalize(
            np.sqrt(df['home_to_stamp_km'].clip(lower=0)), 0, 1
        )
    else:
        exploration_component = pd.Series([0.5] * len(df), index=df.index)
    
    # 가중 평균으로 결합
    motivation = 0.6 * activity_component + 0.4 * exploration_component
    
    # 0~100 범위로 스케일링
    return motivation * 100


def add_motivation_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame에 motivation_score 라벨을 추가합니다.
    
    Args:
        df: 피처가 포함된 DataFrame
        
    Returns:
        motivation_score 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    # 기존 통계 모델 사용 시도
    df['motivation_score'] = generate_motivation_score_from_stat_model(df)
    
    # 점수가 유효한 범위(0-100)인지 확인
    df['motivation_score'] = df['motivation_score'].clip(0, 100)
    
    return df