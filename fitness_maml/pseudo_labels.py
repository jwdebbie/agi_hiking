"""
pseudo_labels.py: Fitness & Trend 점수 의사 라벨 생성 (fitness_maml용)

역할:
  딥러닝 모델 학습을 위한 의사 라벨(pseudo-label)을 생성.
  기존 통계 모델(fitness_stat)의 Calculator 클래스를 활용하여
  라벨을 생성하고, 불가능한 경우 휴리스틱 기반 fallback을 사용.

Input:
  - 피처가 생성된 DataFrame (features.py 출력, for_fitness=True)
  - 필수 컬럼: user_key, avg_speed_mps, speed_trend

Output:
  - fitness_score, trend_score 컬럼이 추가된 DataFrame
  - 점수 범위: 각각 0~100
"""

import numpy as np
import pandas as pd
from typing import Tuple


def minmax_normalize(series: pd.Series, min_val: float = 0, max_val: float = 100) -> pd.Series:
    """
    Min-max 정규화를 수행합니다.
    
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


def generate_scores_from_stat_model(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    기존 통계 모델(fitness_stat)의 Calculator 클래스를 사용하여
    fitness, trend 점수를 생성합니다.
    
    Args:
        df: 원본 데이터가 포함된 DataFrame
        
    Returns:
        (fitness_score, trend_score) 튜플 (각각 0-100)
    """
    try:
        # 기존 fitness_stat 모델의 클래스 임포트 시도
        import sys
        import os
        
        # 프로젝트 루트를 sys.path에 추가
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # fitness_stat 폴더의 Calculator 클래스 임포트
        from fitness_stat.health_score import HealthScoreCalculator
        from fitness_stat.trend_score import TrendScoreCalculator
        
        print("✓ fitness_stat 모델의 Calculator 클래스를 사용합니다")
        
        # 원본 데이터에서 필요한 컬럼만 추출
        # SW2: user_key, total_distance_m, total_duration_sec, complete_date 필요
        # SW6: user_key, total_distance_m, total_duration_sec, complete_date 필요
        
        # SW2 계산 (체력 점수)
        health_calc = HealthScoreCalculator()
        sw2_df = health_calc.compute_user_avg_speed(df)  # user_key, avg_speed_kmh 반환
        sw2_df = health_calc.normalize_to_score(sw2_df)  # sw2_score 계산
        
        # SW6 계산 (트렌드 점수)
        trend_calc = TrendScoreCalculator()
        sw6_df = trend_calc.compute_improvement(df)  # user_key, improvement_rate 반환
        sw6_df = trend_calc.normalize_to_score(sw6_df)  # sw6_score 계산
        
        # user_key 기준으로 병합
        score_df = sw2_df[['user_key', 'sw2_score']].merge(
            sw6_df[['user_key', 'sw6_score']], 
            on='user_key', 
            how='outer'
        )
        
        # 결측값 처리
        score_df['sw2_score'] = score_df['sw2_score'].fillna(50.0)
        score_df['sw6_score'] = score_df['sw6_score'].fillna(50.0)
        
        # 점수 이름 변경 (fitness_score, trend_score)
        score_df = score_df.rename(columns={
            'sw2_score': 'fitness_score',
            'sw6_score': 'trend_score'
        })
        
        # 원본 DataFrame과 병합 (user_key 기준)
        result = df.merge(
            score_df[['user_key', 'fitness_score', 'trend_score']], 
            on='user_key', 
            how='left'
        )
        
        # 병합 후에도 결측값이 있으면 fallback 사용
        if result['fitness_score'].isnull().any() or result['trend_score'].isnull().any():
            n_missing = result['fitness_score'].isnull().sum()
            print(f"⚠ {n_missing}개 행에 대해 fallback 사용")
            
            mask = result['fitness_score'].isnull() | result['trend_score'].isnull()
            fallback_fitness, fallback_trend = generate_scores_fallback(result[mask])
            
            result.loc[mask, 'fitness_score'] = fallback_fitness
            result.loc[mask, 'trend_score'] = fallback_trend
        
        return result['fitness_score'], result['trend_score']
        
    except (ImportError, ModuleNotFoundError, AttributeError, KeyError) as e:
        print(f"⚠ fitness_stat 모델을 사용할 수 없습니다: {e}")
        print("→ Fallback 휴리스틱을 사용하여 fitness/trend 점수를 생성합니다")
        return generate_scores_fallback(df)


def generate_scores_fallback(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Fallback: 휴리스틱 기반 fitness, trend 점수 생성
    
    Fitness 로직:
      - 평균 속도(avg_speed_mps)를 정규화하여 체력 점수 산정
      - 빠를수록 높은 점수
    
    Trend 로직:
      - 속도 변화 추세(speed_trend)를 정규화하여 개선율 점수 산정
      - 양수(개선)는 높은 점수, 음수(악화)는 낮은 점수
    
    Args:
        df: 피처가 포함된 DataFrame
    
    Returns:
        (fitness_score, trend_score) 튜플 (각각 0-100)
    """
    # Fitness 점수: 평균 속도를 0~100으로 정규화
    if 'avg_speed_mps' in df.columns:
        fitness = minmax_normalize(df['avg_speed_mps'], 0, 100)
    else:
        fitness = pd.Series([50.0] * len(df), index=df.index)
    
    # Trend 점수: 속도 트렌드를 0~100으로 정규화
    if 'speed_trend' in df.columns:
        trend = df['speed_trend'].copy()
        
        # 모든 값이 같으면 중간 값(50) 부여
        if trend.max() == trend.min():
            trend = pd.Series([50] * len(trend), index=trend.index)
        else:
            trend = minmax_normalize(trend, 0, 100)
    else:
        trend = pd.Series([50.0] * len(df), index=df.index)
    
    return fitness, trend


def add_fitness_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame에 fitness_score와 trend_score 라벨을 추가합니다.
    
    Args:
        df: 피처가 포함된 DataFrame
        
    Returns:
        fitness_score, trend_score 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    # 기존 통계 모델 사용 시도
    df['fitness_score'], df['trend_score'] = generate_scores_from_stat_model(df)
    
    # 점수가 유효한 범위(0-100)인지 확인
    df['fitness_score'] = df['fitness_score'].clip(0, 100)
    df['trend_score'] = df['trend_score'].clip(0, 100)
    
    return df