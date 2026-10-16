# 🌄 메타러닝 기반 개인별 등산 활동 패턴 분석 연구

> 등산 기록 데이터로 사용자의 **운동 의지**와 **운동 능력**을 정량화하는 베이스라인 시스템,
> 그리고 이를 바탕으로 한 **MAML 기반 Few-shot 개인화** 연구


<br/>

## 프로젝트 개요

기존 등산 활동 분석은 모든 사용자에게 같은 기준을 적용해서, 개인의 체력 수준이나 생활 반경 차이를 반영하지 못했습니다.
이 프로젝트는 두 단계로 이 문제를 다룹니다.

1. **베이스라인 구축** — 참여율, 도전 의지, 체력, 개선율 4가지 지표로 사용자 상태를 점수화하고, 연령대별로 공정하게 비교해 S~D 등급을 부여합니다.
2. **메타러닝 개인화** — 베이스라인 결과를 학습 레이블(Pseudo-label)로 활용해, 신규 사용자도 **단 몇 번의 기록만으로** 개인화된 예측이 가능한지 검증합니다.


<br/>

## 베이스라인 구조

PostgreSQL에서 등산 기록을 불러와 최종 등급(S~D)까지 산출하는 3단계 파이프라인입니다.

```
등산 기록 (PostgreSQL)
        │
   ┌────┴─────┐
   ▼          ▼
Model 1     Model 2
운동 의지     운동 능력
(SW3+SW4)   (SW2+SW6)
   └────┬─────┘
        ▼
     Model 3
 연령대별 정규화 → 등급 (S~D)
```

### Model 1 · 운동 의지

| 지표 | 의미 | 계산 방식 |
|---|---|---|
| **SW3** 참여율 | 얼마나 꾸준히 등산하는가 | 가입 후 경과 주수 대비 실제 등산한 주수 |
| **SW4** 도전 의지 | 얼마나 먼 곳까지 찾아가는가 | Haversine 공식으로 계산한 집–코스 시작점 거리 중 상위 5개 평균 |

### Model 2 · 운동 능력

| 지표 | 의미 | 계산 방식 |
|---|---|---|
| **SW2** 체력 | 기본 체력과 지구력 | 투어별 속도(거리 / 시간)의 평균 |
| **SW6** 개선율 | 체력이 나아지고 있는가 | 초기 2회 대비 최근 3회 평균 속도의 변화율 |

### Model 3 · 연령대별 정규화 및 등급

- 출생연도로 10세 단위 연령대(cohort)를 나누고, 4개 지표를 **연령대 안에서 Poisson 분포 기반으로 정규화**합니다. 세대 간 신체 차이를 보정해 공정하게 비교하기 위해서입니다.
- 정규화된 4개 점수를 합산(최대 400점)한 뒤 백분율 기준으로 등급을 부여합니다.

| 등급 | S | A | B | C | D |
|---|---|---|---|---|---|
| 백분율 | 90% 이상 | 80% 이상 | 70% 이상 | 60% 이상 | 60% 미만 |

<br/>

## 메타러닝 개인화 연구

베이스라인은 연령대 평균을 기준으로 하기 때문에, 개인의 특수한 패턴이 '노이즈'로 묻히는 한계가 있습니다.

**가설** — 사용자의 초기 산행 기록 몇 개(Few-shot)만으로도 개인에게 빠르게 적응하는 예측이 가능할 것이다.

- **Task 정의**: 사용자 1명을 하나의 Task로 보고, 공통 지식을 학습한 뒤 개인별로 빠르게 적응하는 **FO-MAML** 구조를 설계했습니다.
- **Pseudo-label**: 실제 정답 데이터가 없는 문제를 해결하기 위해, 베이스라인 결과를 학습 레이블로 활용했습니다.

<br/>

## 연구 결과

**Motivation MAML** — 적응에 쓰는 기록이 늘어날수록 오차가 크게 줄었습니다. 1-shot 대비 10-shot에서 **MAE 약 73% 감소 (0.64 → 0.17)**.

| k-shot | MAE (적응 전) | MAE (적응 후) | 개선율 |
|---|---|---|---|
| 1 | 0.6591 | 0.6433 | 2.4% |
| 5 | 0.4468 | 0.4030 | 9.8% |
| 10 | 0.2126 | 0.1732 | 18.5% |

**Fitness MAML** — 체력과 개선율을 함께 예측하는 멀티태스크 학습에서, 모든 shot 설정에서 적응 후 오차가 일관되게 감소했습니다.

| k-shot | MAE (적응 전) | MAE (적응 후) | 개선율 |
|---|---|---|---|
| 1 | 0.1149 | 0.1037 | 9.7% |
| 5 | 0.1016 | 0.0907 | 10.8% |
| 10 | 0.0656 | 0.0582 | 11.3% |

**Ablation** — Feature / Label 정규화 단계를 추가할 때마다 오차가 크게 줄어, 데이터 전처리가 성능에 결정적이라는 점을 확인했습니다.

<br/>

## 폴더 구조

```
agi_hiking/
├── data/
│   ├── db_config.py              # DB 접속 정보 (.env에서 읽음)
│   ├── data_loader.py            # PostgreSQL → total_hiking_data 로딩
│   └── csv/
│       └── geocode_address_area.py  # 주소 → 행정구역 변환 (카카오 API)
├── model1/
│   ├── motivation_stamp.py       # SW3 참여율
│   ├── motivation_distance.py    # SW4 도전 의지
│   └── run_model1.py             # Model 1 실행
├── model2/
│   ├── health_score.py           # SW2 체력
│   ├── trend_score.py            # SW6 개선율
│   └── run_model2.py             # Model 2 실행
├── model3/
│   └── normalizer.py             # 연령대별 정규화 + 등급 부여
├── .env.example
└── requirements.txt
```

<br/>

## 실행 방법

### 1. 환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. 접속 정보 설정

`.env.example`을 복사해 `.env` 파일을 만들고 DB 접속 정보를 입력합니다. `.env`는 Git에 올라가지 않습니다.

```bash
cp .env.example .env
```

### 3. 실행

```bash
python -m data.data_loader        # 데이터 로딩 테스트
python -m model1.run_model1       # Model 1: 운동 의지
python -m model2.run_model2       # Model 2: 운동 능력
python -m model3.normalizer       # Model 3: 최종 등급 산출
```

각 지표만 따로 확인하려면 `python -m model1.motivation_stamp`처럼 개별 모듈을 실행하면 됩니다.

<br/>

## 🔗 관련 연구

- 환경 변화에 강건한 CCTV 칼 인식 연구 — [Robustness-Research](https://github.com/jwdebbie/Robustness-Research)
