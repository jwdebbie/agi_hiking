# meta_models/nets.py
# 실제 예측을 하는 신경망(모델 구조)을 정의한다.
# “AI 모델”의 함수 근사기(learnable model) 역할

from __future__ import annotations
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPMultiRegressor(nn.Module):
    """
    Outputs two heads: y1 (fitness), y2 (trend)
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(hidden, 1)
        self.head2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        y1 = self.head1(h).squeeze(-1)
        y2 = self.head2(h).squeeze(-1)
        return y1, y2
