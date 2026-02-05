"""
run_fitness_maml.py: MAML 기반 체력/트렌드 점수 예측 모델 실행 스크립트 (멀티태스크)
Version 1: 이상치 처리 전 기본 버전

역할:
  메타러닝을 사용하여 사용자별로 빠르게 적응하는 체력 및 트렌드 점수 예측 모델을 학습.
  하나의 모델이 두 가지 점수를 동시에 예측 (멀티태스크 학습).
  
Input:
  - PostgreSQL의 total_hiking_data view (data_loader를 통해 로드)
  
Output:
  - 학습된 모델: saved_models/fitness_maml_v1_TIMESTAMP.pt
  - 평가 결과: saved_models/fitness_maml_v1_results_TIMESTAMP.csv
  - K-shot별 성능 비교 (K=1, 5, 10)

실행:
  python -m fitness_maml.run_fitness_maml_v1
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.data_loader import load_total_hiking_data
from maml_base.features import create_feature_dataframe
from fitness_maml.pseudo_labels import add_fitness_trend_labels
from maml_base.task_dataset import UserTaskDataset, TaskSampler, create_train_test_split
from maml_base.nets import create_model
from maml_base.maml_trainer import MAMLTrainer


def main():
    print("=" * 80)
    print("MAML 기반 체력/트렌드 점수(Fitness & Trend) 예측 모델 (멀티태스크)")
    print("Version 1: 이상치 처리 전 기본 버전")
    print("=" * 80)
    
    # 설정
    config = {
        'random_seed': 42,
        'test_user_ratio': 0.2,         # 테스트 사용자 비율
        'min_samples_per_user': 5,      # 사용자당 최소 샘플 수
        'k_shot_train': 5,               # 학습 시 Support set 크기
        'k_shot_eval': [1, 5, 10],       # 평가 시 K-shot 설정들
        'hidden_dim': 64,                # 은닉층 차원
        'hidden_layers': 2,              # 은닉층 개수
        'dropout': 0.1,                  # Dropout 비율
        'inner_lr': 0.01,                # Inner loop 학습률
        'outer_lr': 0.001,               # Outer loop 학습률
        'inner_steps': 3,                # Inner loop 반복 횟수
        'n_epochs': 50,                  # 메타 학습 에폭 수
        'tasks_per_batch': 4,            # 배치당 Task(사용자) 수
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n설정")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 랜덤 시드 설정
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    # 1. 데이터 로드
    print("\n" + "=" * 80)
    print("1️⃣  PostgreSQL에서 데이터 로딩...")
    print("=" * 80)
    df = load_total_hiking_data(filter_missing_address=True)
    print(f"✓ {len(df):,}개 레코드 로드 완료 (사용자 수: {df['user_key'].nunique()}명)")
    
    # 2. 피처 생성 (속도 트렌드 포함)
    print("\n" + "=" * 80)
    print("2️⃣  피처 엔지니어링 (속도 트렌드 분석 포함)...")
    print("=" * 80)
    
    # 피처 생성
    df = create_feature_dataframe(df, for_fitness=True)
    print(f"✓ 총 {len(df.columns)}개 컬럼 생성 완료")
    
    # 3. 의사 라벨 생성
    print("\n" + "=" * 80)
    print("3️⃣  의사 라벨 생성...")
    print("=" * 80)
    
    df = add_fitness_trend_labels(df)
    
    print(f"✓ Fitness 점수 - 평균: {df['fitness_score'].mean():.2f}, 표준편차: {df['fitness_score'].std():.2f}")
    print(f"✓ Trend 점수 - 평균: {df['trend_score'].mean():.2f}, 표준편차: {df['trend_score'].std():.2f}")
    
    # 4. 피처 정의
    feature_cols = [
        'total_distance_m',           # 총 거리
        'total_duration_sec',         # 총 소요시간
        'avg_speed_mps',              # 평균 속도
        'hikes_last_30d',             # 최근 30일 등산 횟수
        'days_since_last_hike',       # 마지막 등산 이후 일수
        'total_hike_count',           # 누적 등산 횟수
        'speed_trend',                # 속도 변화 추세 (기울기)
        'speed_std',                  # 속도 표준편차
        'weekday',                    # 요일
        'month'                       # 월
    ]
    label_cols = ['fitness_score', 'trend_score']
    
    print(f"\n피처 ({len(feature_cols)}개)")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    print(f"\n라벨 ({len(label_cols)}개)")
    for i, col in enumerate(label_cols, 1):
        print(f"  {i}. {col}")
    
    # 5. 결측값/무한값 처리 (최소한의 처리만)
    print("\n결측값 및 무한값 처리...")
    for col in feature_cols + label_cols:
        if df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            print(f"⚠ {col}: {n_missing}개 결측값 → 0으로 채움")
            df[col] = df[col].fillna(0)
        
        if np.isinf(df[col]).any():
            n_inf = np.isinf(df[col]).sum()
            print(f"⚠ {col}: {n_inf}개 무한값 → 0으로 대체")
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    print("\n데이터 통계:")
    print(f"  총 샘플 수: {len(df)}")
    print(f"  총 사용자 수: {df['user_key'].nunique()}")
    print(f"  사용자당 평균 샘플 수: {len(df) / df['user_key'].nunique():.1f}")

    # 6. Train/Test 분할
    print("\n" + "=" * 80)
    print("4️⃣  Train/Test 사용자 분할...")
    print("=" * 80)
    train_df, test_df = create_train_test_split(
        df,
        test_user_ratio=config['test_user_ratio'],
        random_seed=config['random_seed']
    )
    
    # 7. Task 데이터셋 생성
    print("\n" + "=" * 80)
    print("5️⃣  Task 데이터셋 생성...")
    print("=" * 80)
    train_dataset = UserTaskDataset(
        train_df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        min_samples_per_user=config['min_samples_per_user']
    )
    
    test_dataset = UserTaskDataset(
        test_df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        min_samples_per_user=config['min_samples_per_user']
    )
    
    # 8. 멀티태스크 모델 생성
    print("\n" + "=" * 80)
    print("6️⃣  멀티태스크 신경망 모델 생성...")
    print("=" * 80)
    model = create_model(
        model_type='multi',
        input_dim=len(feature_cols),
        hidden_dim=config['hidden_dim'],
        hidden_layers=config['hidden_layers'],
        n_outputs=len(label_cols),
        dropout=config['dropout']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 모델 생성 완료 ({n_params:,}개 파라미터)")
    print(f"✓ 구조: 공유 Backbone + {len(label_cols)}개 Task Head")
    print(f"✓ 디바이스: {config['device']}")
    
    # 9. Trainer 생성
    trainer = MAMLTrainer(
        model=model,
        inner_lr=config['inner_lr'],
        outer_lr=config['outer_lr'],
        inner_steps=config['inner_steps'],
        device=config['device']
    )
    
    # 10. 메타 학습
    print("\n" + "=" * 80)
    print("7️⃣  MAML 메타 학습 시작...")
    print("=" * 80)
    train_sampler = TaskSampler(
        train_dataset,
        k_shot=config['k_shot_train']
    )
    
    trainer.train(
        dataset=train_dataset,
        sampler=train_sampler,
        n_epochs=config['n_epochs'],
        tasks_per_batch=config['tasks_per_batch'],
        verbose=True
    )
    
    # 11. 모델 저장
    model_dir = os.path.join(project_root, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'fitness_maml_v1_{timestamp}.pt')
    trainer.save_model(model_path)
    
    # 12. 평가
    print("\n" + "=" * 80)
    print("8️⃣  테스트 사용자에 대한 평가...")
    print("=" * 80)
    
    results_table = []
    
    for k_shot in config['k_shot_eval']:
        print(f"\nK-shot = {k_shot}")
        results = trainer.evaluate(test_dataset, k_shot=k_shot)
        
        print(f"  0-step MAE (적응 전): {results['mae_before_adaptation']:.4f}")
        print(f"  Adapted MAE ({config['inner_steps']}단계 적응 후): {results['mae_after_adaptation']:.4f}")
        print(f"  개선: {results['improvement']:.4f} ({results['improvement']/results['mae_before_adaptation']*100:.1f}%)")
        print(f"  테스트 Task 수: {results['n_tasks']}")
        
        # 멀티태스크의 경우 타겟별 결과도 출력
        if 'target_0_mae_before' in results:
            print(f"\n  [Target 0 - Fitness Score]")
            print(f"    적응 전 MAE: {results['target_0_mae_before']:.4f}")
            print(f"    적응 후 MAE: {results['target_0_mae_after']:.4f}")
            print(f"    개선: {results['target_0_improvement']:.4f} ({results['target_0_improvement_pct']:.1f}%)")
        
        if 'target_1_mae_before' in results:
            print(f"\n  [Target 1 - Trend Score]")
            print(f"    적응 전 MAE: {results['target_1_mae_before']:.4f}")
            print(f"    적응 후 MAE: {results['target_1_mae_after']:.4f}")
            print(f"    개선: {results['target_1_improvement']:.4f} ({results['target_1_improvement_pct']:.1f}%)")
        
        results_table.append({
            'k_shot': k_shot,
            'mae_before': results['mae_before_adaptation'],
            'mae_after': results['mae_after_adaptation'],
            'improvement': results['improvement'],
            'improvement_pct': results['improvement']/results['mae_before_adaptation']*100,
            'n_tasks': results['n_tasks']
        })
    
    # 13. 결과 요약
    print("\n" + "=" * 80)
    print("평가 결과 요약")
    print("=" * 80)
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # 결과 저장
    results_path = os.path.join(model_dir, f'fitness_maml_v1_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ 결과 저장: {results_path}")
    print(f"✓ 모델 저장: {model_path}")
    
    print("\n" + "=" * 80)
    print("Fitness & Trend 모델 학습 완료 (Version 1)")
    print("=" * 80)


if __name__ == '__main__':
    main()