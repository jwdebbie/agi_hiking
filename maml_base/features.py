"""
features.py: 등산 데이터 피처 엔지니어링

역할: 
  등산 데이터로부터 MAML 메타러닝 모델 학습에 필요한 피처들을 생성.

Input:
  - PostgreSQL에서 로드한 total_hiking_data DataFrame
  - 컬럼: user_key, complete_date, total_distance_m, total_duration_sec,
         stamp_latitude, stamp_longitude, member_latitude, member_longitude

Output:
  - 피처가 추가된 DataFrame
  - 추가 피처: home_to_stamp_km, avg_speed_mps, hikes_last_30d, 
              days_since_last_hike, total_hike_count, weekday, month,
              speed_trend(for_fitness=True인 경우), speed_std
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from typing import Tuple


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 지점 간의 Haversine 거리를 계산.
    
    Args:
        lat1, lon1: 첫 번째 지점의 위도, 경도
        lat2, lon2: 두 번째 지점의 위도, 경도
    
    Returns:
        두 지점 간의 거리(킬로미터)
    """
    # 십진법 각도를 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine 공식
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c  # 지구 반지름 (킬로미터)
    return km


def compute_home_to_stamp_km(df: pd.DataFrame) -> pd.Series:
    """
    집(member)에서 스탬프(stamp) 위치까지의 거리를 계산.
    
    Args:
        df: member_latitude/longitude, stamp_latitude/longitude 컬럼이 있는 DataFrame
    
    Returns:
        거리(km) Series
    """
    return df.apply(
        lambda row: haversine(
            row['member_latitude'], row['member_longitude'],
            row['stamp_latitude'], row['stamp_longitude']
        ),
        axis=1
    )


def compute_avg_speed_mps(df: pd.DataFrame) -> pd.Series:
    """
    평균 속도를 계산합니다 (m/s).
    
    Args:
        df: total_distance_m, total_duration_sec 컬럼이 있는 DataFrame
    
    Returns:
        평균 속도(m/s) Series
    """
    # 0으로 나누기 방지
    duration = df['total_duration_sec'].replace(0, np.nan)
    speed = df['total_distance_m'] / duration
    return speed.fillna(0)


def compute_user_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    사용자별 히스토리 기반 피처를 생성.
    
    생성 피처:
      - hikes_last_30d: 최근 30일 내 등산 횟수
      - days_since_last_hike: 마지막 등산 이후 일수
      - total_hike_count: 누적 등산 횟수
    
    Args:
        df: user_key, complete_date 컬럼이 있는 DataFrame
    
    Returns:
        히스토리 피처가 추가된 DataFrame
    """
    df = df.sort_values(['user_key', 'complete_date']).copy()
    
    result_rows = []
    
    for user_key, user_df in df.groupby('user_key'):
        user_df = user_df.sort_values('complete_date').reset_index(drop=True)
        
        hikes_last_30d = []
        days_since_last = []
        total_count = []
        
        for idx, row in user_df.iterrows():
            current_date = row['complete_date']
            
            # 현재 등산 이전의 데이터
            past_df = user_df.iloc[:idx]
            
            if len(past_df) > 0:
                # 최근 30일 내 등산 횟수
                past_30d = past_df[
                    past_df['complete_date'] >= (current_date - pd.Timedelta(days=30))
                ]
                hikes_last_30d.append(len(past_30d))
                
                # 마지막 등산 이후 일수
                if idx > 0:
                    last_date = user_df.iloc[idx - 1]['complete_date']
                    days_diff = (current_date - last_date).total_seconds() / 86400
                    days_since_last.append(days_diff)
                else:
                    days_since_last.append(999)  # 첫 등산은 큰 값
            else:
                hikes_last_30d.append(0)
                days_since_last.append(999)
            
            # 누적 등산 횟수 (현재 포함)
            total_count.append(idx + 1)
        
        user_df['hikes_last_30d'] = hikes_last_30d
        user_df['days_since_last_hike'] = days_since_last
        user_df['total_hike_count'] = total_count
        
        result_rows.append(user_df)
    
    return pd.concat(result_rows, ignore_index=True)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 기반 피처를 생성.
    
    생성 피처:
      - weekday: 요일 (0=월요일, 6=일요일)
      - month: 월 (1-12)
    
    Args:
        df: complete_date 컬럼이 있는 DataFrame
    
    Returns:
        시간 피처가 추가된 DataFrame
    """
    df = df.copy()
    df['weekday'] = df['complete_date'].dt.weekday
    df['month'] = df['complete_date'].dt.month
    return df


def compute_speed_trend_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    속도 변화 트렌드 피처를 생성.
    
    생성 피처:
      - speed_trend: 최근 n회 등산의 속도 변화 추세(선형 회귀 기울기)
      - speed_std: 최근 n회 등산의 속도 표준편차
    
    Args:
        df: user_key, complete_date, avg_speed_mps 컬럼이 있는 DataFrame
        window: 트렌드 계산에 사용할 최근 등산 횟수
    
    Returns:
        트렌드 피처가 추가된 DataFrame
    """
    df = df.sort_values(['user_key', 'complete_date']).copy()
    
    speed_trends = []
    speed_stds = []
    
    for user_key, user_df in df.groupby('user_key'):
        user_df = user_df.sort_values('complete_date').reset_index(drop=True)
        
        for idx in range(len(user_df)):
            # 최근 window개 레코드 가져오기 (현재 포함)
            start_idx = max(0, idx - window + 1)
            window_df = user_df.iloc[start_idx:idx+1]
            
            if len(window_df) >= 2:
                speeds = window_df['avg_speed_mps'].values
                x = np.arange(len(speeds))
                
                # 선형 회귀 기울기 (트렌드)
                if len(speeds) > 1:
                    slope = np.polyfit(x, speeds, 1)[0]
                    std = np.std(speeds)
                else:
                    slope = 0
                    std = 0
                
                speed_trends.append(slope)
                speed_stds.append(std)
            else:
                # 데이터 부족 시 0
                speed_trends.append(0)
                speed_stds.append(0)
    
    df['speed_trend'] = speed_trends
    df['speed_std'] = speed_stds
    
    return df


def create_feature_dataframe(df: pd.DataFrame, for_fitness: bool = False) -> pd.DataFrame:
    """
    전체 피처 엔지니어링 파이프라인을 실행.
    
    Args:
        df: 원시 데이터 DataFrame
        for_fitness: True인 경우 속도 트렌드 피처 추가 (fitness_maml용)
    
    Returns:
        모든 피처가 추가된 DataFrame
    """
    # 기본 피처
    df = df.copy()
    df['home_to_stamp_km'] = compute_home_to_stamp_km(df)
    df['avg_speed_mps'] = compute_avg_speed_mps(df)
    
    # 사용자 히스토리 피처
    df = compute_user_history_features(df)
    
    # 시간 피처
    df = compute_time_features(df)
    
    # 속도 트렌드 피처 (fitness 모델용)
    if for_fitness:
        df = compute_speed_trend_features(df)
    
    return df