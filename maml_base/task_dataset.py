"""
task_dataset.py: MAML 메타러닝을 위한 Task 데이터셋

역할:
  각 사용자를 하나의 Task로 취급하는 메타러닝용 데이터셋을 생성.
  시간 순서를 고려하여 support set(학습용)과 query set(평가용)을 분할.

Input:
  - 피처와 라벨이 포함된 DataFrame
  - feature_cols: 피처 컬럼 리스트
  - label_cols: 라벨 컬럼 리스트
  - k_shot: Support set 크기 (과거 k개 샘플)

Output:
  - UserTaskDataset: 사용자별 데이터를 반환하는 Dataset
  - TaskSampler: Support/Query set을 샘플링하는 Sampler
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional


class UserTaskDataset(Dataset):
    """
    각 사용자를 별도의 Task로 취급하는 데이터셋
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_cols: List[str],
        min_samples_per_user: int = 5
    ):
        """
        Args:
            df: 피처와 라벨이 포함된 DataFrame
            feature_cols: 피처 컬럼명 리스트
            label_cols: 라벨 컬럼명 리스트 (단일 또는 다중)
            min_samples_per_user: 사용자당 최소 샘플 수 (이 미만이면 제외)
        """
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        
        # 충분한 샘플이 있는 사용자만 필터링
        user_counts = df.groupby('user_key').size()
        valid_users = user_counts[user_counts >= min_samples_per_user].index.tolist()
        
        self.df = df[df['user_key'].isin(valid_users)].copy()
        self.users = valid_users
        
        # 사용자별, 시간순으로 정렬
        self.df = self.df.sort_values(['user_key', 'complete_date']).reset_index(drop=True)
        
        # 사용자 → 인덱스 매핑 생성
        self.user_to_indices = {}
        for user in self.users:
            indices = self.df[self.df['user_key'] == user].index.tolist()
            self.user_to_indices[user] = indices
        
        print(f"UserTaskDataset 생성 완료: {len(self.users)}명의 사용자")
        print(f"  전체 샘플 수: {len(self.df)}")
        print(f"  피처 컬럼: {feature_cols}")
        print(f"  라벨 컬럼: {label_cols}")
    
    def __len__(self) -> int:
        """Task(사용자) 수 반환"""
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        특정 사용자(Task)의 모든 데이터를 반환합니다.
        
        Args:
            idx: Task 인덱스
        
        Returns:
            'X'(피처), 'y'(라벨), 'user_key', 'n_samples'를 포함하는 딕셔너리
        """
        user_key = self.users[idx]
        indices = self.user_to_indices[user_key]
        
        user_df = self.df.loc[indices]
        
        # 피처를 텐서로 변환
        X = torch.tensor(
            user_df[self.feature_cols].values,
            dtype=torch.float32
        )
        
        # 라벨을 텐서로 변환
        y = torch.tensor(
            user_df[self.label_cols].values,
            dtype=torch.float32
        )
        
        # 라벨이 1개면 차원 축소 (batch_size,) 형태로
        if len(self.label_cols) == 1:
            y = y.squeeze(-1)
        
        return {
            'X': X,
            'y': y,
            'user_key': user_key,
            'n_samples': len(indices)
        }


class TaskSampler:
    """
    MAML 학습을 위해 각 Task를 Support set과 Query set으로 분할하는 샘플러
    """
    
    def __init__(
        self,
        dataset: UserTaskDataset,
        k_shot: int,
        query_size: Optional[int] = None
    ):
        """
        Args:
            dataset: UserTaskDataset 인스턴스
            k_shot: Support set 크기 (과거 k개 샘플)
            query_size: Query set 크기 (None이면 나머지 전부 사용)
        """
        self.dataset = dataset
        self.k_shot = k_shot
        self.query_size = query_size
    
    def sample_task(self, task_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        하나의 Task를 Support set과 Query set으로 분할합니다.
        시간 순서를 유지하여 과거 데이터는 support, 최근 데이터는 query로 사용합니다.
        
        Args:
            task_idx: Task 인덱스
        
        Returns:
            (support_dict, query_dict) 튜플
            각 딕셔너리는 'X'(피처), 'y'(라벨) 포함
        """
        task_data = self.dataset[task_idx]
        X = task_data['X']
        y = task_data['y']
        n_samples = task_data['n_samples']
        
        # 시간순 분할: 앞쪽 k_shot개를 support, 나머지를 query로
        if n_samples <= self.k_shot:
            # 샘플이 부족하면 전부 support로, query는 비움
            support_X = X
            support_y = y
            query_X = torch.empty((0, X.shape[1]), dtype=torch.float32)
            query_y = torch.empty((0,) if y.dim() == 1 else (0, y.shape[1]), dtype=torch.float32)
        else:
            support_X = X[:self.k_shot]
            support_y = y[:self.k_shot]
            
            if self.query_size is not None:
                # Query size가 지정된 경우
                query_end = min(self.k_shot + self.query_size, n_samples)
                query_X = X[self.k_shot:query_end]
                query_y = y[self.k_shot:query_end]
            else:
                # Query size 미지정 시 나머지 전부
                query_X = X[self.k_shot:]
                query_y = y[self.k_shot:]
        
        support = {'X': support_X, 'y': support_y}
        query = {'X': query_X, 'y': query_y}
        
        return support, query
    
    def sample_batch(self, task_indices: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        여러 Task를 배치로 샘플링합니다.
        
        Args:
            task_indices: Task 인덱스 리스트
        
        Returns:
            (support_batch, query_batch) 튜플
            각각 딕셔너리의 리스트
        """
        support_batch = []
        query_batch = []
        
        for task_idx in task_indices:
            support, query = self.sample_task(task_idx)
            support_batch.append(support)
            query_batch.append(query)
        
        return support_batch, query_batch


def create_train_test_split(
    df: pd.DataFrame,
    test_user_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    사용자를 train과 test로 분할.
    
    중요: 같은 사용자의 데이터가 train과 test에 동시에 포함되지 않도록 합니다.
    이를 통해 새로운 사용자에 대한 일반화 성능을 평가할 수 있습니다.
    
    Args:
        df: 전체 DataFrame
        test_user_ratio: Test set에 포함할 사용자 비율
        random_seed: 랜덤 시드
    
    Returns:
        (train_df, test_df) 튜플
    """
    np.random.seed(random_seed)
    
    users = df['user_key'].unique()
    n_test_users = max(1, int(len(users) * test_user_ratio))
    
    # 랜덤하게 test 사용자 선택
    test_users = np.random.choice(users, size=n_test_users, replace=False)
    train_users = [u for u in users if u not in test_users]
    
    train_df = df[df['user_key'].isin(train_users)].copy()
    test_df = df[df['user_key'].isin(test_users)].copy()
    
    print(f"Train/Test Split 완료:")
    print(f"  Train: {len(train_users)}명, {len(train_df)} 샘플")
    print(f"  Test: {len(test_users)}명, {len(test_df)} 샘플")
    
    return train_df, test_df