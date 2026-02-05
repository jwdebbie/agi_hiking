import os
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor

# .env 파일에서 환경변수 로드 (API 키, DB 정보 등)
load_dotenv()

# API 키
REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
if not REST_API_KEY:
    raise RuntimeError("환경변수 KAKAO_REST_API_KEY 를 설정하세요.")

# DB 설정 (.env 파일에서 읽기)
DB_CONFIG = {
    'host': os.environ.get("DB_HOST"),
    'port': int(os.environ.get("DB_PORT")),
    'user': os.environ.get("DB_USER"),
    'password': os.environ.get("DB_PASSWORD"),
    'database': os.environ.get("DB_NAME")
}

# 테이블 및 컬럼 설정
TABLE_NAME = "member"           # 테이블명
ADDRESS_COLUMN = "address"      # 주소 컬럼명
LONGITUDE_COLUMN = "longitude"  # 경도 컬럼명 (이미 존재)
LATITUDE_COLUMN = "latitude"    # 위도 컬럼명 (이미 존재)
STATUS_COLUMN = "geocode_status"  # 상태 컬럼명 (자동 추가됨)
ID_COLUMN = "user_key"          # PK 컬럼명

def setup_database(conn):
    """필요한 컬럼이 없으면 추가"""
    with conn.cursor() as cursor:
        # status 컬럼 확인 및 추가
        cursor.execute(f"""
            SELECT COUNT(*) as cnt 
            FROM information_schema.columns 
            WHERE table_schema = 'public'
            AND table_name = '{TABLE_NAME}' 
            AND column_name = '{STATUS_COLUMN}'
        """)
        result = cursor.fetchone()
        if result['cnt'] == 0:
            cursor.execute(f"""
                ALTER TABLE {TABLE_NAME} 
                ADD COLUMN {STATUS_COLUMN} VARCHAR(100)
            """)
            print(f"{STATUS_COLUMN} 컬럼 추가됨")
        
        conn.commit()

def main():
    # DB 연결
    print("DB 연결 중...")
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    
    try:
        # 필요한 컬럼 추가 (status만 추가, longitude/latitude는 이미 있음)
        setup_database(conn)
        
        # API 세션 설정
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        session = requests.Session()
        session.headers.update({"Authorization": f"KakaoAK {REST_API_KEY}"})
        
        # 처리할 데이터 가져오기 (아직 처리 안 된 것만)
        with conn.cursor() as cursor:
            query = f"""
                SELECT {ID_COLUMN}, {ADDRESS_COLUMN}
                FROM {TABLE_NAME}
                WHERE ({STATUS_COLUMN} IS NULL OR {STATUS_COLUMN} != 'ok')
                AND {ADDRESS_COLUMN} IS NOT NULL
                ORDER BY {ID_COLUMN}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
        
        total = len(rows)
        print(f"처리할 레코드: {total}개")
        
        if total == 0:
            print("모든 데이터가 이미 처리되었습니다!")
            return
        
        success_count = 0
        
        # 지오코딩 처리
        for idx, row in enumerate(tqdm(rows, desc="Geocoding")):
            record_id = row[ID_COLUMN]
            addr = row[ADDRESS_COLUMN]
            
            # null 체크
            if addr is None:
                update_status(conn, record_id, None, None, "null")
                continue
            
            addr = addr.strip()
            
            # 빈 문자열 체크
            if not addr:
                update_status(conn, record_id, None, None, "empty")
                continue
            
            try:
                r = session.get(url, params={"query": addr}, timeout=10)
                r.raise_for_status()
                data = r.json()
                docs = data.get("documents", [])
                
                if not docs:
                    update_status(conn, record_id, None, None, "no_result")
                else:
                    x = docs[0].get("x")  # longitude
                    y = docs[0].get("y")  # latitude
                    
                    if x and y:
                        update_status(conn, record_id, float(x), float(y), "ok")
                        success_count += 1
                    else:
                        update_status(conn, record_id, None, None, "no_coords")
            
            except requests.exceptions.RequestException as e:
                error_msg = f"error:{type(e).__name__}"
                update_status(conn, record_id, None, None, error_msg)
                print(f"\n❌ ID {record_id} 실패: {addr} - {e}")
            
            except Exception as e:
                error_msg = f"error:{type(e).__name__}"
                update_status(conn, record_id, None, None, error_msg)
                print(f"\n⚠️ ID {record_id} 예외: {addr} - {e}")
            
            # 100개마다 commit
            if (idx + 1) % 100 == 0:
                conn.commit()
                print(f"\n 중간 커밋: {idx + 1}/{total}")
            
            # API 호출 제한
            time.sleep(0.1)
        
        # 최종 커밋
        conn.commit()
        print(f"\n 완료! 총 {total}개 중 {success_count}개 성공")
        
        # 결과 통계
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT {STATUS_COLUMN}, COUNT(*) as cnt 
                FROM {TABLE_NAME} 
                GROUP BY {STATUS_COLUMN}
                ORDER BY cnt DESC
            """)
            print("\n 상태별 결과:")
            for row in cursor.fetchall():
                status = row[STATUS_COLUMN] if row[STATUS_COLUMN] else 'NULL'
                print(f"  {status}: {row['cnt']}개")
    
    finally:
        conn.close()
        print("\nDB 연결 종료")

def update_status(conn, record_id, longitude, latitude, status):
    """레코드 업데이트"""
    with conn.cursor() as cursor:
        query = f"""
            UPDATE {TABLE_NAME}
            SET {LONGITUDE_COLUMN} = %s,
                {LATITUDE_COLUMN} = %s,
                {STATUS_COLUMN} = %s
            WHERE {ID_COLUMN} = %s
        """
        cursor.execute(query, (longitude, latitude, status, record_id))

if __name__ == "__main__":
    main()