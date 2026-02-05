# data/impute_location.py
"""
회원 위치 결측값 추정: 방문한 산들의 중앙값 사용
실행하면 member_imputed 테이블이 DB에 생성됨
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from .db_config import DB_CONFIG


def load_member_with_hiking_locations():
    """회원 정보와 방문한 산 위치 정보를 함께 로드"""
    query = """
    SELECT 
        m.user_key,
        m.birth_year,
        m.address,
        m.join_date,
        m.longitude AS member_longitude,
        m.latitude AS member_latitude,
        m.geocode_status,
        th.stamp_longitude,
        th.stamp_latitude
    FROM member m
    LEFT JOIN total_hiking_data th ON m.user_key = th.user_key
    """
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(query, conn)
    
    return df


def impute_member_locations(df):
    """
    회원별로 방문한 산들의 중앙값으로 집 위치 추정
    
    Args:
        df: member + hiking 정보가 결합된 DataFrame
    
    Returns:
        결측값이 채워진 회원 DataFrame
        (원본 위치가 있거나, 방문 기록으로 추정 가능한 회원만 포함)
    """
    imputed_locations = []
    skipped_count = 0
    
    for user_key, group in df.groupby('user_key'):
        member_info = group.iloc[0].copy()
        
        # 회원의 집 위치가 이미 있는 경우 - 원본 유지
        if pd.notna(member_info['member_latitude']) and pd.notna(member_info['member_longitude']):
            imputed_locations.append({
                'user_key': user_key,
                'birth_year': member_info['birth_year'],
                'address': member_info['address'],
                'join_date': member_info['join_date'],
                'longitude': member_info['member_longitude'],
                'latitude': member_info['member_latitude'],
                'geocode_status': member_info['geocode_status'],
                'imputation_method': 'original'
            })
        else:
            # 결측값인 경우 - 방문한 산들의 위치로 추정
            visited_stamps = group[
                (group['stamp_latitude'].notna()) & 
                (group['stamp_longitude'].notna())
            ]
            
            if len(visited_stamps) > 0:
                # 방문한 산들의 중앙값으로 추정
                imputed_lat = visited_stamps['stamp_latitude'].median()
                imputed_lon = visited_stamps['stamp_longitude'].median()
                
                imputed_locations.append({
                    'user_key': user_key,
                    'birth_year': member_info['birth_year'],
                    'address': member_info['address'] if pd.notna(member_info['address']) else '위치추정',
                    'join_date': member_info['join_date'],
                    'longitude': imputed_lon,
                    'latitude': imputed_lat,
                    'geocode_status': 'imputed_from_stamps',
                    'imputation_method': f'median_of_{len(visited_stamps)}_stamps'
                })
            else:
                # 방문 기록이 없으면 버림
                skipped_count += 1
    
    print(f"   방문 기록이 없어 제외된 회원: {skipped_count}명")
    
    return pd.DataFrame(imputed_locations)


def create_member_cleaned_table():
    """
    결측값을 삭제한 member_cleaned 테이블을 DB에 생성
    """
    print("\n결측값 삭제 버전 테이블 생성 시작...")
    print("-" * 60)
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        cursor = conn.cursor()
        
        # 기존 테이블 삭제
        cursor.execute("DROP TABLE IF EXISTS member_cleaned")
        
        # 결측값이 없는 레코드만 선택하여 새 테이블 생성
        cursor.execute("""
            CREATE TABLE member_cleaned AS
            SELECT * FROM member
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL 
              AND geocode_status = 'ok'
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX idx_member_cleaned_user_key ON member_cleaned(user_key)")
        
        # 커밋
        conn.commit()
        
        # 결과 확인
        cursor.execute("SELECT COUNT(*) FROM member_cleaned")
        count = cursor.fetchone()[0]
        print(f"   member_cleaned 테이블 생성 완료 ({count} 레코드)")
    
    print("-" * 60 + "\n")


def create_member_imputed_table():
    """
    member_imputed 테이블을 DB에 생성
    """
    print("\n회원 위치 결측값 추정 시작...")
    print("-" * 60)
    
    # 1. 데이터 로드
    print("1. 데이터 로딩 중...")
    df = load_member_with_hiking_locations()
    total_users = df['user_key'].nunique()
    print(f"   {len(df)} 레코드 로드 (회원 수: {total_users}명)")
    
    # 2. 결측값 추정
    print("\n2. 위치 추정 중...")
    imputed_df = impute_member_locations(df)
    print(f"   {len(imputed_df)}명 회원 데이터 생성 완료")
    
    # 3. 추정 방법별 통계
    print("\n3. 추정 방법별 통계:")
    method_counts = imputed_df['imputation_method'].value_counts()
    for method, count in method_counts.items():
        print(f"   - {method}: {count}명")
    
    # 4. DB에 테이블 생성 및 저장
    print("\n4. DB에 저장 중...")
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            cursor = conn.cursor()
            
            # 기존 테이블 삭제
            cursor.execute("DROP TABLE IF EXISTS member_imputed")
            
            # 새 테이블 생성
            cursor.execute("""
                CREATE TABLE member_imputed (
                    user_key INTEGER PRIMARY KEY,
                    birth_year INTEGER,
                    address TEXT,
                    join_date TIMESTAMP,
                    longitude DOUBLE PRECISION,
                    latitude DOUBLE PRECISION,
                    geocode_status VARCHAR(100),
                    imputation_method VARCHAR(100)
                )
            """)
            
            # 데이터 삽입
            insert_query = """
                INSERT INTO member_imputed 
                (user_key, birth_year, address, join_date, longitude, latitude, 
                 geocode_status, imputation_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            execute_batch(cursor, insert_query, imputed_df.values.tolist())
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX idx_member_imputed_user_key ON member_imputed(user_key)")
            
            # 커밋
            conn.commit()
            print(f"   member_imputed 테이블 생성 완료 ({len(imputed_df)} 레코드)")
            
    except Exception as e:
        print(f"   오류 발생: {e}")
        conn.rollback()
        raise
    
    # 5. DB 반영 확인
    print("\n5. DB 반영 확인:")
    with psycopg2.connect(**DB_CONFIG) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM member_imputed")
        count = cursor.fetchone()[0]
        print(f"   member_imputed 테이블: {count} 레코드")
        
        cursor.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE latitude IS NULL) as null_lat,
                COUNT(*) FILTER (WHERE longitude IS NULL) as null_lon
            FROM member_imputed
        """)
        null_lat, null_lon = cursor.fetchone()
        print(f"   결측값: 위도 {null_lat}개, 경도 {null_lon}개")
    
    print("\n" + "-" * 60)
    print("완료! member_imputed 테이블이 DB에 생성되었습니다.")
    print("-" * 60 + "\n")
    
    return imputed_df


if __name__ == "__main__":
    # 두 테이블 모두 생성
    create_member_cleaned_table()
    create_member_imputed_table()