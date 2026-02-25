# agi_hiking 프로젝트 구조 및 사용법

## 1. 폴더 구조

```text
agi_hiking/
├── data/
│   ├── data_loader.py          # PostgreSQL view 데이터 로드
│   ├── db_config.py            # DB 접속 설정
│   ├── impute_location.py      # 위치 정보 보정
│   └── kakao_geocode_db.py     # 주소 → 좌표 변환
│
├── maml_base/                  # MAML 공통 모듈
│   ├── features.py             # 피처 엔지니어링
│   ├── task_dataset.py         # 사용자 단위 Task 데이터셋 구성
│   ├── nets.py                 # 신경망 모델 정의
│   └── maml_trainer.py         # MAML 학습 및 평가 로직
│
├── motivation_maml/            # 동기 점수 MAML 모델
│   ├── pseudo_labels.py        # Motivation 의사 라벨 생성
│   └── run_motivation_maml.py  # MAML 실행 스크립트
│
├── fitness_maml/               # 체력/트렌드 MAML 모델
│   ├── pseudo_labels.py        # Fitness/Trend 의사 라벨 생성
│   └── run_fitness_maml.py     # MAML 실행 스크립트
│
├── motivation_stat/            # 통계 기반 동기 점수 모델 (의사 라벨용)
│   ├── motivation_stamp.py     # SW3: 참여율 점수
│   ├── motivation_distance.py  # SW4: 거리 기반 도전 의지
│   └── run_model1.py           # 통계 모델 실행
│
├── fitness_stat/               # 통계 기반 체력/트렌드 점수 모델 (의사 라벨용)
│   ├── health_score.py         # SW2: 체력 점수
│   ├── trend_score.py          # SW6: 개선 트렌드 점수
│   └── run_model2.py           # 통계 모델 실행
│
├── score_integration/          # 점수 통합 및 정규화
│   └── normalizer.py
│
├── requirements.txt
└── README.md

```

---

## 2. 사전 준비

### 2.1 가상환경 생성
cd orda_smu
python -m venv .venv
.\.venv\Scripts\activate      # Windows


### 2.2 패키지 설치
pip install -r requirements.txt

(처음 세팅할 때 requirements.txt가 비어 있거나 부족하면:)
pip install psycopg2-binary pandas numpy scipy
pip freeze > requirements.txt




## 3. DB 설정
`data/db_config.py` 실제 서버 관련 정보 변경 시 접속 정보도 수정해주세요.

DB_CONFIG = {
    "host": "203.153.148.28",
    "port": 5433,
    "dbname": "orda_stamp",
    "user": "postgres",
    "password": "509509",
}



## 4. 데이터 로딩 테스트
cd orda_smu
.\.venv\Scripts\activate

python -m data.data_loader
