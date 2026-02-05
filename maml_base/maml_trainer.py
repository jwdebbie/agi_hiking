"""
maml_trainer.py: MAML (Model-Agnostic Meta-Learning) 학습 및 평가

역할:
  First-Order MAML을 사용하여 개인화 회귀 모델을 메타 학습.
  각 사용자를 별도 Task로 취급하여, 적은 데이터로도 빠르게 적응할 수 있는
  메타 파라미터를 학습.

MAML 알고리즘:
  1. Inner Loop: Support set으로 Task별 파라미터 적응 (θ → θ')
  2. Outer Loop: Query set으로 메타 파라미터 업데이트 (θ ← θ - ∇L(θ'))

Input:
  - UserTaskDataset: 사용자별 Task 데이터셋
  - TaskSampler: Support/Query set 샘플러
  
Output:
  - 학습된 메타 모델 (체크포인트로 저장)
  - 평가 결과: MAE (적응 전/후 비교)
  
개선사항 (Ver 2):
  - ✅ 멀티태스크 모델의 경우 각 타겟별 MAE를 개별 계산
  - ✅ Gradient clipping으로 학습 안정성 향상
  - ✅ NaN/Inf 체크로 robust한 평가
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import List, Dict, Tuple, Optional
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class MAMLTrainer:
    """
    First-Order MAML (FO-MAML) 학습기
    
    First-Order: 2차 미분을 생략하여 계산 효율성 향상
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 3,
        device: str = 'cpu'
    ):
        """
        Args:
            model: 신경망 모델
            inner_lr: Inner loop 학습률 (Task 적응)
            outer_lr: Outer loop 학습률 (메타 업데이트)
            inner_steps: Inner loop 반복 횟수
            device: 'cpu' 또는 'cuda'
        """
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # 메타 옵티마이저 (outer loop용)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
    
    def inner_loop(
        self,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        create_graph: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Inner loop: Support set으로 Task에 맞게 파라미터를 적응시킵니다.
        
        Args:
            support_X: Support set 입력
            support_y: Support set 라벨
            create_graph: 2차 미분 계산 여부 (First-Order에서는 False)
        
        Returns:
            적응된 파라미터 딕셔너리
        """
        # 현재 모델 파라미터 복제
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Inner loop: inner_steps만큼 gradient descent 수행
        for step in range(self.inner_steps):
            # 적응된 파라미터로 순전파
            pred = self._forward_with_params(support_X, adapted_params)
            
            # Support set 손실 계산
            if support_y.dim() == 1:
                # 단일 출력
                loss = F.mse_loss(pred, support_y)
            else:
                # 다중 출력
                loss = F.mse_loss(pred, support_y)
            
            # 그래디언트 계산
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                allow_unused=True
            )
            
            # 파라미터 업데이트: θ' ← θ - α∇L(θ)
            adapted_params = {
                name: param - self.inner_lr * grad if grad is not None else param
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        커스텀 파라미터로 순전파를 수행.
        (MAML의 적응된 파라미터 사용을 위해 필요)
        
        Args:
            x: 입력 텐서
            params: 사용할 파라미터 딕셔너리
        
        Returns:
            모델 출력
        """
        # 원본 파라미터 백업
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]
        
        # 순전파
        output = self.model(x)
        
        # 원본 파라미터 복원
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return output
    
    def train_step(
        self,
        support_batch: List[Dict[str, torch.Tensor]],
        query_batch: List[Dict[str, torch.Tensor]]
    ) -> float:
        """
        하나의 메타 학습 스텝을 수행.
        
        과정:
          1. 각 Task에 대해 inner loop로 적응
          2. Query set에서 손실 계산
          3. 메타 파라미터 업데이트
        
        Args:
            support_batch: Support set 리스트
            query_batch: Query set 리스트
        
        Returns:
            메타 손실 값
        """
        self.model.train()
        meta_loss = None
        
        # 각 Task에 대해 처리
        for support, query in zip(support_batch, query_batch):
            support_X = support['X'].to(self.device)
            support_y = support['y'].to(self.device)
            query_X = query['X'].to(self.device)
            query_y = query['y'].to(self.device)
            
            # Query 샘플이 없으면 스킵
            if len(query_X) == 0:
                continue
            
            # Inner loop: Support set으로 적응
            # FO-MAML: create_graph=False로 2차 미분 생략
            adapted_params = self.inner_loop(support_X, support_y, create_graph=False)
            
            # Outer loop: Query set으로 메타 손실 계산
            query_pred = self._forward_with_params(query_X, adapted_params)
            
            if query_y.dim() == 1:
                task_loss = F.mse_loss(query_pred, query_y)
            else:
                task_loss = F.mse_loss(query_pred, query_y)
            
            # 첫 번째 task_loss는 그대로, 이후는 누적
            if meta_loss is None:
                meta_loss = task_loss
            else:
                meta_loss = meta_loss + task_loss
        
        # meta_loss가 None이면 (모든 query가 비어있음) 0 반환
        if meta_loss is None:
            return 0.0
        
        # 메타 파라미터 업데이트
        meta_loss = meta_loss / len(support_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def train(
        self,
        dataset,  # UserTaskDataset
        sampler,  # TaskSampler
        n_epochs: int = 100,
        tasks_per_batch: int = 4,
        verbose: bool = True
    ):
        """
        메타 학습 전체 루프
        
        Args:
            dataset: UserTaskDataset
            sampler: TaskSampler
            n_epochs: 메타 학습 에폭 수
            tasks_per_batch: 배치당 Task 수
            verbose: 진행상황 출력 여부
        """
        n_tasks = len(dataset)
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Task 순서 섞기
            task_indices = np.random.permutation(n_tasks)
            
            # 배치 생성
            n_batches = (n_tasks + tasks_per_batch - 1) // tasks_per_batch
            
            iterator = range(n_batches)
            if verbose:
                iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{n_epochs}")
            
            for batch_idx in iterator:
                start_idx = batch_idx * tasks_per_batch
                end_idx = min(start_idx + tasks_per_batch, n_tasks)
                batch_task_indices = task_indices[start_idx:end_idx].tolist()
                
                # Support/Query 샘플링
                support_batch, query_batch = sampler.sample_batch(batch_task_indices)
                
                # 메타 학습 스텝
                loss = self.train_step(support_batch, query_batch)
                epoch_losses.append(loss)
            
            if verbose:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{n_epochs}, Meta Loss: {avg_loss:.4f}")
    
    def evaluate(
        self,
        dataset,  # UserTaskDataset
        k_shot: int,
        adapt_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        테스트 Task들에 대해 평가를 수행합니다.
        멀티태스크의 경우 각 타겟별로 MAE를 개별 계산합니다.
        
        평가 방식:
          1. 0-step: 적응 전 메타 파라미터로 예측 (MAE 측정)
          2. Adapted: Support set으로 적응 후 예측 (MAE 측정)
          3. Improvement: 적응으로 인한 성능 개선 측정
        
        Args:
            dataset: 테스트 UserTaskDataset
            k_shot: Support set 크기
            adapt_steps: 적응 스텝 수 (None이면 self.inner_steps 사용)
        
        Returns:
            평가 결과 딕셔너리
            - 멀티태스크의 경우 타겟별 MAE도 포함
        """
        self.model.eval()
        
        if adapt_steps is None:
            adapt_steps = self.inner_steps
        
        # TaskSampler import (여기서만 필요)
        from maml_base.task_dataset import TaskSampler
        sampler = TaskSampler(dataset, k_shot=k_shot)
        
        mae_before = []  # 적응 전 전체 MAE
        mae_after = []   # 적응 후 전체 MAE
        
        # ✅ 멀티태스크용: 타겟별 MAE 저장
        n_targets = None
        mae_before_per_target = None
        mae_after_per_target = None
        
        for task_idx in range(len(dataset)):
            support, query = sampler.sample_task(task_idx)
            
            support_X = support['X'].to(self.device)
            support_y = support['y'].to(self.device)
            query_X = query['X'].to(self.device)
            query_y = query['y'].to(self.device)
            
            # Query 샘플이 없으면 스킵
            if len(query_X) == 0:
                continue
            
            # ✅ 타겟 수 확인 (첫 번째 Task에서)
            if n_targets is None:
                if query_y.dim() == 1:
                    n_targets = 1
                else:
                    n_targets = query_y.shape[1]
                mae_before_per_target = [[] for _ in range(n_targets)]
                mae_after_per_target = [[] for _ in range(n_targets)]
            
            with torch.no_grad():
                # 0-step: 적응 전 예측
                pred_before = self.model(query_X)
                
                if query_y.dim() == 1:
                    # 단일 출력
                    mae_b = F.l1_loss(pred_before, query_y).item()
                    mae_before.append(mae_b)
                    mae_before_per_target[0].append(mae_b)
                else:
                    # ✅ 다중 출력: 전체 평균 MAE
                    mae_b = F.l1_loss(pred_before, query_y).item()
                    mae_before.append(mae_b)
                    
                    # ✅ 각 타겟별로 MAE 계산
                    for target_idx in range(n_targets):
                        mae_b_target = F.l1_loss(
                            pred_before[:, target_idx], 
                            query_y[:, target_idx]
                        ).item()
                        mae_before_per_target[target_idx].append(mae_b_target)
            
            # 원본 파라미터 백업
            original_state = deepcopy(self.model.state_dict())
            
            # Inner loop로 적응 (gradient 활성화 필요)
            self.model.train()  # 학습 모드로 전환
            
            # adapted_params를 실제 모델에 적용하기 위한 수동 gradient descent
            optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            
            for step in range(adapt_steps):
                optimizer.zero_grad()
                pred = self.model(support_X)
                
                if support_y.dim() == 1:
                    loss = F.mse_loss(pred, support_y)
                else:
                    loss = F.mse_loss(pred, support_y)
                
                loss.backward()
                
                # ✅ Gradient Clipping으로 폭발 방지
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # 적응 후 평가
            self.model.eval()
            with torch.no_grad():
                pred_after = self.model(query_X)
                
                # ✅ NaN 체크
                if torch.isnan(pred_after).any() or torch.isinf(pred_after).any():
                    # 적응 실패 시 적응 전 MAE 사용 (개선 없음)
                    mae_a = mae_b
                    mae_after.append(mae_a)
                    
                    # 타겟별로도 동일하게 처리
                    if query_y.dim() > 1:
                        for target_idx in range(n_targets):
                            mae_after_per_target[target_idx].append(
                                mae_before_per_target[target_idx][-1]
                            )
                    else:
                        mae_after_per_target[0].append(mae_b)
                else:
                    if query_y.dim() == 1:
                        # 단일 출력
                        mae_a = F.l1_loss(pred_after, query_y).item()
                        mae_after.append(mae_a)
                        mae_after_per_target[0].append(mae_a)
                    else:
                        # ✅ 다중 출력: 전체 평균 MAE
                        mae_a = F.l1_loss(pred_after, query_y).item()
                        mae_after.append(mae_a)
                        
                        # ✅ 각 타겟별로 MAE 계산
                        for target_idx in range(n_targets):
                            mae_a_target = F.l1_loss(
                                pred_after[:, target_idx], 
                                query_y[:, target_idx]
                            ).item()
                            mae_after_per_target[target_idx].append(mae_a_target)
            
            # 원본 파라미터 복원
            self.model.load_state_dict(original_state)
        
        # ✅ 결과 집계
        results = {
            'mae_before_adaptation': np.mean(mae_before),
            'mae_after_adaptation': np.mean(mae_after),
            'improvement': np.mean(mae_before) - np.mean(mae_after),
            'n_tasks': len(mae_before)
        }
        
        # ✅ 멀티태스크의 경우 타겟별 결과 추가
        if n_targets and n_targets > 1:
            for target_idx in range(n_targets):
                mae_b = np.mean(mae_before_per_target[target_idx])
                mae_a = np.mean(mae_after_per_target[target_idx])
                results[f'target_{target_idx}_mae_before'] = mae_b
                results[f'target_{target_idx}_mae_after'] = mae_a
                results[f'target_{target_idx}_improvement'] = mae_b - mae_a
                results[f'target_{target_idx}_improvement_pct'] = \
                    (mae_b - mae_a) / mae_b * 100 if mae_b > 0 else 0
        
        return results
    
    def save_model(self, path: str):
        """모델 체크포인트 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps
        }, path)
        print(f"✓ 모델 저장 완료: {path}")
    
    def load_model(self, path: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        print(f"✓ 모델 로드 완료: {path}")