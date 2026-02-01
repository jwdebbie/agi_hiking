import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # .env 파일 읽기

REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
if not REST_API_KEY:
    raise RuntimeError("환경변수 KAKAO_REST_API_KEY 를 설정해줘 (카카오 REST API 키).")

INPUT_CSV = "data/csv/address_area.csv"
OUTPUT_CSV = "data/csv/geocoded.csv"

df = pd.read_csv(INPUT_CSV)
if "address_area" not in df.columns:
    raise ValueError("CSV에 address_area 컬럼이 필요해.")

df["longitude"] = None
df["latitude"] = None
df["status"] = None  # 성공/실패 메모

url = "https://dapi.kakao.com/v2/local/search/address.json"
headers = {"Authorization": f"KakaoAK {REST_API_KEY}"}

for i, addr in enumerate(df["address_area"].astype(str)):
    addr = addr.strip()
    if not addr:
        df.loc[i, "status"] = "empty"
        continue

    try:
        r = requests.get(url, headers=headers, params={"query": addr}, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = data.get("documents", [])

        if not docs:
            df.loc[i, "status"] = "no_result"
        else:
            # 1순위 결과 사용
            x = docs[0].get("x")  # longitude
            y = docs[0].get("y")  # latitude
            df.loc[i, "longitude"] = float(x) if x is not None else None
            df.loc[i, "latitude"] = float(y) if y is not None else None
            df.loc[i, "status"] = "ok"

    except Exception as e:
        df.loc[i, "status"] = f"error:{type(e).__name__}"

    # 너무 빠른 연속 호출 방지(안전)
    time.sleep(0.1)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("saved:", OUTPUT_CSV)
print(df["status"].value_counts(dropna=False))
