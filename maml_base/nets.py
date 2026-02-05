"""
nets.py: MAML 메타러닝을 위한 신경망 모델

역할:
  회귀 태스크를 위한 MLP(Multi-Layer Perceptron) 모델을 제공.
  motivation_maml 및 fitness_maml 모델의 출력(단일/다중)을 지원

Input:
  - 피처 벡터 (batch_size, input_dim)

Output:
  - 단일 출력: (batch_size,) - motivation_score
  - 다중 출력: (batch_size, n_outputs) - fitness_score, trend_score
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPRegressor(nn.Module):
    """
    단일 출력 회귀를 위한 MLP 모델
    
    구조:
      Input → Linear → ReLU → Dropout →
      Linear → ReLU → Dropout → ... →
      Linear → Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 피처 차원
            hidden_dim: 은닉층 차원
            hidden_layers: 은닉층 개수
            dropout: Dropout 확률
        """
        super().__init__()
        
        layers = []
        
        # 입력층 → 첫 번째 은닉층
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 추가 은닉층들
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 출력층 (회귀이므로 활성화 함수 없음)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, input_dim)
        
        Returns:
            출력 텐서 (batch_size,)
        """
        # (batch_size, 1) → (batch_size,)로 차원 축소
        return self.model(x).squeeze(-1)


class MLPMultiRegressor(nn.Module):
    """
    다중 출력 회귀를 위한 MLP 모델
    
    구조:
      Input → Shared Backbone (여러 은닉층) →
      ├─ Head 1 → Output 1
      ├─ Head 2 → Output 2
      └─ ...
    
    Backbone을 공유하여 피처를 함께 학습하고,
    각 출력마다 별도의 head를 사용합니다.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        n_outputs: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 피처 차원
            hidden_dim: 은닉층 차원
            hidden_layers: Backbone의 은닉층 개수
            n_outputs: 출력 개수
            dropout: Dropout 확률
        """
        super().__init__()
        
        self.n_outputs = n_outputs
        
        # 공유 Backbone 구성
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, hidden_dim))
        backbone_layers.append(nn.ReLU())
        backbone_layers.append(nn.Dropout(dropout))
        
        for _ in range(hidden_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 출력별 독립적인 head들
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(n_outputs)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, input_dim)
        
        Returns:
            출력 텐서 (batch_size, n_outputs)
        """
        # Backbone을 통과하여 공유 피처 추출
        features = self.backbone(x)
        
        # 각 head를 통과하여 개별 출력 생성
        outputs = torch.cat([
            head(features) for head in self.heads
        ], dim=1)
        
        return outputs


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    n_outputs: int = 1,
    dropout: float = 0.1
) -> nn.Module:
    """
    모델 팩토리 함수
    
    Args:
        model_type: 'single' (단일 출력) 또는 'multi' (다중 출력)
        input_dim: 입력 피처 차원
        hidden_dim: 은닉층 차원
        hidden_layers: 은닉층 개수
        n_outputs: 출력 개수 (multi-task용)
        dropout: Dropout 확률
    
    Returns:
        생성된 모델 인스턴스
    """
    if model_type == 'single':
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
    elif model_type == 'multi':
        return MLPMultiRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            n_outputs=n_outputs,
            dropout=dropout
        )
    else:
        raise ValueError(f"알 수 없는 model_type: {model_type}. 'single' 또는 'multi'를 사용하세요.")