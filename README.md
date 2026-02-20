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



## 5. 모델 실행 순서

### 5.1 모델1: 운동 의지 (참여율 + 도전 의지)

- SW3: 가입 후 경과 주수 대비 **실제로 등산한 주수** → 참여율.
- SW4: 집 주소–코스 시작점 거리 기반 도전 의지 (현재는 주소 구조에 맞게 추가 작업 필요).

실행:
python -m model1.motivation_stamp      # SW3만
python -m model1.motivation_distance   # SW4만
python -m model1.run_model1           # 모델1 최종 (SW3+SW4)

출력:  
`user_key, participation_rate, sw3_score, avg_top_distance_km, sw4_score, model1_score, ...`

---

### 5.2 모델2: 운동 능력 (체력 + 개선율)

- SW2: 각 투어 속도(거리/시간) → 사용자 평균 속도 → 0~100 체력 점수.
- SW6: 초기 2회 vs 최근 3회 속도 평균의 **개선율** → 0~100 개선율 점수.

실행:
python -m model2.health_score    # SW2만
python -m model2.trend_score     # SW6만
python -m model2.run_model2      # 모델2 최종 (SW2+SW6)


출력:  
`user_key, avg_speed_kmh, sw2_score, improvement_rate, sw6_score, model2_score, tour_count, ...`

---

### 5.3 모델3: 그룹화 분석 및 등급 부여

입력:
- 모델1 점수: SW3(참여율), SW4(도전 의지)
- 모델2 점수: SW2(체력), SW6(개선율)
- Member의 출생연도(birth_year) → 나이대(cohort)

역할:
- 나이대(cohort)별로 네 모듈 점수를 Poisson 기반으로 정규화.
- 정규화된 네 점수를 합산해 total_score(최대 400) 계산.
- total_score 기준으로 S/A/B/C/D 등급 부여.

실행:
python -m model3.normalizer

출력 컬럼 예:

- `user_key`
- `birth_year`
- `cohort` (20대, 30대, 50대 이상 등)
- `sw3_score_norm` (참여율 정규화 점수)
- `sw4_score_norm` (도전 의지 정규화 점수)
- `sw2_score_norm` (체력 정규화 점수)
- `sw6_score_norm` (개선율 정규화 점수)
- `total_score` (네 모듈 합산, 0~400)
- `percentage` (0~100 백분율)
- `grade` (S, A, B, C, D)

