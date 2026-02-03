# meta_models/task_dataset.py
# MAML을 위한 Task 구성을 만든다.
# “개인화” 의도를 메타러닝 구조로 구현하는 핵심 파일

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TaskBatch:
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


@dataclass
class MultiTaskBatch:
    support_x: np.ndarray
    support_y1: np.ndarray
    support_y2: np.ndarray
    query_x: np.ndarray
    query_y1: np.ndarray
    query_y2: np.ndarray


class UserTaskDataset:
    """
    Holds features/labels grouped by user_key.
    """
    def __init__(self, user_ids: np.ndarray, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray):
        self.user_ids = user_ids
        self.X = X
        self.y = y
        self.timestamps = timestamps

        self.users = np.unique(user_ids)
        self.user_to_idx = {u: np.where(user_ids == u)[0] for u in self.users}

    def sample_task(self, k_shot: int, q_size: int | None = None, time_ordered: bool = True) -> TaskBatch:
        u = np.random.choice(self.users)
        idx = self.user_to_idx[u]

        if time_ordered:
            idx = idx[np.argsort(self.timestamps[idx])]
        else:
            idx = np.random.permutation(idx)

        if len(idx) < k_shot + 1:
            # too small, resample
            return self.sample_task(k_shot, q_size=q_size, time_ordered=time_ordered)

        support_idx = idx[:k_shot]
        remain = idx[k_shot:]

        if q_size is None:
            query_idx = remain
        else:
            if len(remain) < q_size:
                return self.sample_task(k_shot, q_size=q_size, time_ordered=time_ordered)
            query_idx = remain[:q_size]

        return TaskBatch(
            support_x=self.X[support_idx],
            support_y=self.y[support_idx],
            query_x=self.X[query_idx],
            query_y=self.y[query_idx],
        )


class UserTaskDatasetMulti:
    def __init__(self, user_ids: np.ndarray, X: np.ndarray, y1: np.ndarray, y2: np.ndarray, timestamps: np.ndarray):
        self.user_ids = user_ids
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.timestamps = timestamps

        self.users = np.unique(user_ids)
        self.user_to_idx = {u: np.where(user_ids == u)[0] for u in self.users}

    def sample_task(self, k_shot: int, q_size: int | None = None, time_ordered: bool = True) -> MultiTaskBatch:
        u = np.random.choice(self.users)
        idx = self.user_to_idx[u]

        if time_ordered:
            idx = idx[np.argsort(self.timestamps[idx])]
        else:
            idx = np.random.permutation(idx)

        if len(idx) < k_shot + 1:
            return self.sample_task(k_shot, q_size=q_size, time_ordered=time_ordered)

        support_idx = idx[:k_shot]
        remain = idx[k_shot:]

        if q_size is None:
            query_idx = remain
        else:
            if len(remain) < q_size:
                return self.sample_task(k_shot, q_size=q_size, time_ordered=time_ordered)
            query_idx = remain[:q_size]

        return MultiTaskBatch(
            support_x=self.X[support_idx],
            support_y1=self.y1[support_idx],
            support_y2=self.y2[support_idx],
            query_x=self.X[query_idx],
            query_y1=self.y1[query_idx],
            query_y2=self.y2[query_idx],
        )
