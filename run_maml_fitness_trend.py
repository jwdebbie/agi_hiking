# run_maml_fitness_trend.py
# 모델2(체력/개선) 파이프라인 실행 스크립트
# AI 모델 1건 완성 + 성능 측정 한 번에 실행

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.data_loader import load_total_hiking_data

from meta_models.features import (
    add_time_features, add_core_features, add_user_history_features, build_feature_matrix
)
from meta_models.pseudo_labels import make_pseudo_labels
from meta_models.task_dataset import UserTaskDatasetMulti
from meta_models.nets import MLPMultiRegressor
from meta_models.maml_trainer import train_maml_multi, evaluate_maml_multi


def split_users(user_ids: np.ndarray, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    users = np.unique(user_ids)
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_u = users[:n_train]
    val_u = users[n_train:n_train+n_val]
    test_u = users[n_train+n_val:]
    return train_u, val_u, test_u


def filter_by_users(df: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    return df[df["user_key"].isin(users)].copy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = load_total_hiking_data()
    required = [
        "user_key", "complete_date",
        "total_distance_m", "total_duration_sec",
        "stamp_latitude", "stamp_longitude",
        "member_latitude", "member_longitude",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in total_hiking_data: {missing}")

    df = add_time_features(df)
    df = add_core_features(df)
    df = add_user_history_features(df, window_days=30, trend_n=5)

    # Pseudo-labels (fitness_score, trend_score)
    y1, y2 = make_pseudo_labels(df, which="fitness_trend")

    feature_cols = [
        "total_distance_m",
        "total_duration_sec",
        "avg_speed_mps",
        "hikes_last_30d",
        "days_since_last_hike",
        "total_hike_count",
        "speed_trend",
        "speed_std",
        "weekday",
        "month",
    ]
    X, _ = build_feature_matrix(df, feature_cols)

    user_ids = df["user_key"].astype(str).to_numpy()
    t = pd.to_datetime(df["complete_date"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    timestamps = t.view("int64").to_numpy()

    train_u, val_u, test_u = split_users(user_ids, seed=42)
    df_train = filter_by_users(df, train_u)
    df_val = filter_by_users(df, val_u)
    df_test = filter_by_users(df, test_u)

    def pack(dff):
        idx = dff.index.to_numpy()
        return user_ids[idx], X[idx], y1[idx], y2[idx], timestamps[idx]

    tr_user, tr_X, tr_y1, tr_y2, tr_t = pack(df_train)
    va_user, va_X, va_y1, va_y2, va_t = pack(df_val)
    te_user, te_X, te_y1, te_y2, te_t = pack(df_test)

    train_ds = UserTaskDatasetMulti(tr_user, tr_X, tr_y1, tr_y2, tr_t)
    val_ds = UserTaskDatasetMulti(va_user, va_X, va_y1, va_y2, va_t)
    test_ds = UserTaskDatasetMulti(te_user, te_X, te_y1, te_y2, te_t)

    model = MLPMultiRegressor(in_dim=tr_X.shape[1], hidden=64)

    print("\n=== Train MAML (Fitness & Trend) ===")
    model = train_maml_multi(
        model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=30,
        meta_batch=8,
        k_shot=5,
        q_size=10,
        inner_lr=0.01,
        inner_steps=3,
        outer_lr=1e-3,
        device=device,
    )

    print("\n=== Meta-test (Fitness & Trend) ===")
    for k in [1, 5, 10]:
        mae0, maeA = evaluate_maml_multi(model, test_ds, k_shot=k, inner_lr=0.01, inner_steps=3, device=device, tasks=50)
        print(f"K={k:2d} | MAE(0-step)={mae0:.3f}  MAE(adapt)={maeA:.3f}")

    torch.save(model.state_dict(), "maml_fitness_trend.pt")
    print("Saved: maml_fitness_trend.pt")


if __name__ == "__main__":
    main()
