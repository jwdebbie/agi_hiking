import os
from dotenv import load_dotenv

load_dotenv()  # 프로젝트 루트의 .env 파일에서 접속 정보를 읽음

DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
}
