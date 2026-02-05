"""
run_motivation_maml.py: MAML 기반 동기 점수 예측 모델 실행 스크립트

역할:
  메타러닝을 사용하여 사용자별로 빠르게 적응하는 동기 점수 예측 모델을 학습합니다.
  
Input:
  - PostgreSQL의 total_hiking_data view (data_loader를 통해 로드)
  
Output:
  - 학습된 모델: saved_models/motivation_maml_TIMESTAMP.pt
  - 평가 결과: saved_models/motivation_maml_results_TIMESTAMP.csv
  - K-shot별 성능 비교 (K=1, 5, 10)

실행:
  python -m motivation_maml.run_motivation_maml
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
from motivation_maml.pseudo_labels import add_motivation_label
from maml_base.task_dataset import UserTaskDataset, TaskSampler, create_train_test_split
from maml_base.nets import create_model
from maml_base.maml_trainer import MAMLTrainer


def main():
    print("=" * 80)
    print("MAML 기반 동기 점수(Motivation) 예측 모델")
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
    
    # 2. 피처 생성
    print("\n" + "=" * 80)
    print("2️⃣  피처 엔지니어링...")
    print("=" * 80)
    
    # 원본 데이터 백업 (통계 모델용)
    df_original = df.copy()
    
    # 피처 생성
    df = create_feature_dataframe(df, for_fitness=False)
    print(f"✓ 총 {len(df.columns)}개 컬럼 생성 완료")
    
    # 3. 의사 라벨 생성 (motivation_stat 모델 활용 시도)
    print("\n" + "=" * 80)
    print("3️⃣  의사 라벨 생성...")
    print("=" * 80)
    
    # 통계 모델이 원본 컬럼을 필요로 하므로 병합
    # 필요한 원본 컬럼: join_date, member_latitude, member_longitude, stamp_latitude, stamp_longitude
    original_cols = ['user_key', 'join_date', 'complete_date', 
                     'member_latitude', 'member_longitude', 
                     'stamp_latitude', 'stamp_longitude']
    
    if all(col in df.columns for col in original_cols):
        # df에 이미 join_date가 있으므로 merge 불필요
        print("✓ 모든 필요한 원본 컬럼이 이미 존재합니다")
        df = add_motivation_label(df)
    elif all(col in df_original.columns for col in original_cols):
        # merge 필요 시 suffixes 명시
        print("⚠ 일부 컬럼 추가를 위해 merge 수행")
        df_with_original = df.merge(
            df_original[original_cols],
            on=['user_key', 'complete_date'],
            how='left',
            suffixes=('', '_orig')  # suffix 명시
        )
            
        # ✅ suffix 처리 추가!
        for col in original_cols:
            if col + '_orig' in df_with_original.columns:
                if col in df_with_original.columns:
                    # 양쪽에 모두 있는 경우, _orig 값으로 대체
                    df_with_original[col] = df_with_original[col + '_orig'].fillna(df_with_original[col])
                else:
                    # _orig만 있는 경우, rename
                    df_with_original[col] = df_with_original[col + '_orig']
                df_with_original.drop(columns=[col + '_orig'], inplace=True)
            
        # ✅ df_with_original을 사용!
        df = add_motivation_label(df_with_original)
    else:
        print("⚠ 일부 원본 컬럼이 없어 fallback 사용")
        df = add_motivation_label(df)
    
    print(f"✓ Motivation 점수 범위: [{df['motivation_score'].min():.2f}, {df['motivation_score'].max():.2f}]")
    print(f"   평균: {df['motivation_score'].mean():.2f}, 표준편차: {df['motivation_score'].std():.2f}")
    
    # 4. 피처 정의
    feature_cols = [
        'home_to_stamp_km',          # 집-스탬프 거리
        'total_distance_m',           # 총 거리
        'total_duration_sec',         # 총 소요시간
        'avg_speed_mps',              # 평균 속도
        'hikes_last_30d',             # 최근 30일 등산 횟수
        'days_since_last_hike',       # 마지막 등산 이후 일수
        'total_hike_count',           # 누적 등산 횟수
        'weekday',                    # 요일
        'month'                       # 월
    ]
    label_cols = ['motivation_score']
    
    print(f"\n피처 ({len(feature_cols)}개)")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    # 결측값/무한값 처리
    for col in feature_cols + label_cols:
        if df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            print(f"⚠ {col}: {n_missing}개 결측값 → 0으로 채움")
            df[col] = df[col].fillna(0)
        
        if np.isinf(df[col]).any():
            n_inf = np.isinf(df[col]).sum()
            print(f"⚠ {col}: {n_inf}개 무한값 → 0으로 대체")
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    # ✅ 피처 정규화 추가 (NaN 방지)
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0

    # 라벨 정규화 추가
    print("\n라벨 정규화...")
    label_mean = df['motivation_score'].mean()
    label_std = df['motivation_score'].std()
    if label_std > 0:
        df['motivation_score'] = (df['motivation_score'] - label_mean) / label_std
        print(f"  motivation_score: mean={label_mean:.2f}, std={label_std:.2f}")
        print(f"  정규화 후 범위: [{df['motivation_score'].min():.2f}, {df['motivation_score'].max():.2f}]")
    else:
        print("  ⚠️ 라벨 표준편차가 0입니다")

    # 5. Train/Test 분할 (사용자 단위)
    print("\n" + "=" * 80)
    print("4️⃣  Train/Test 사용자 분할...")
    print("=" * 80)
    train_df, test_df = create_train_test_split(
        df,
        test_user_ratio=config['test_user_ratio'],
        random_seed=config['random_seed']
    )
    
    # 6. Task 데이터셋 생성
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
    
    # 7. 모델 생성
    print("\n" + "=" * 80)
    print("6️⃣  신경망 모델 생성...")
    print("=" * 80)
    model = create_model(
        model_type='single',
        input_dim=len(feature_cols),
        hidden_dim=config['hidden_dim'],
        hidden_layers=config['hidden_layers'],
        dropout=config['dropout']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 모델 생성 완료 ({n_params:,}개 파라미터)")
    print(f"✓ 디바이스: {config['device']}")
    
    # 8. Trainer 생성
    trainer = MAMLTrainer(
        model=model,
        inner_lr=config['inner_lr'],
        outer_lr=config['outer_lr'],
        inner_steps=config['inner_steps'],
        device=config['device']
    )
    
    # 9. 메타 학습
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
    
    # 10. 모델 저장
    model_dir = os.path.join(project_root, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'motivation_maml_{timestamp}.pt')
    trainer.save_model(model_path)
    
    # 11. 평가
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
        
        results_table.append({
            'k_shot': k_shot,
            'mae_before': results['mae_before_adaptation'],
            'mae_after': results['mae_after_adaptation'],
            'improvement': results['improvement'],
            'improvement_pct': results['improvement']/results['mae_before_adaptation']*100,
            'n_tasks': results['n_tasks']
        })
    
    # 12. 결과 요약
    print("\n" + "=" * 80)
    print("평가 결과 요약")
    print("=" * 80)
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # 결과 저장
    results_path = os.path.join(model_dir, f'motivation_maml_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ 결과 저장: {results_path}")
    print(f"✓ 모델 저장: {model_path}")
    
    print("\n" + "=" * 80)
    print("Motivation 모델 학습 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()