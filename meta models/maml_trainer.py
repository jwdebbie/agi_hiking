# meta_models/maml_trainer.py
# MAML 학습/평가를 수행한다. (과제에서 “학습되는 모습”을 보여주는 핵심)
# “새 사용자에서 few-shot 적응으로 성능이 좋아짐”을 측정 가능하게 만든 파트

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call


@dataclass
class MAMLMetrics:
    mae_0: float
    mae_adapt: float


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def _mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def maml_inner_update(
    model: nn.Module,
    params: dict,
    x_s: torch.Tensor,
    y_s: torch.Tensor,
    inner_lr: float,
    steps: int,
    loss_fn,
):
    """
    First-order style inner loop: compute grads w.r.t params, do SGD updates.
    """
    for _ in range(steps):
        pred = functional_call(model, params, (x_s,))
        loss = loss_fn(pred, y_s)
        grads = torch.autograd.grad(loss, params.values(), create_graph=False)
        params = {k: v - inner_lr * g for (k, v), g in zip(params.items(), grads)}
    return params


def train_maml_single(
    model: nn.Module,
    train_ds,
    val_ds,
    epochs: int = 30,
    meta_batch: int = 8,
    k_shot: int = 5,
    q_size: int = 10,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    outer_lr: float = 1e-3,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = _mse

    for ep in range(1, epochs + 1):
        model.train()
        meta_loss = 0.0

        for _ in range(meta_batch):
            task = train_ds.sample_task(k_shot=k_shot, q_size=q_size, time_ordered=True)

            x_s = torch.tensor(task.support_x, device=device)
            y_s = torch.tensor(task.support_y, device=device)
            x_q = torch.tensor(task.query_x, device=device)
            y_q = torch.tensor(task.query_y, device=device)

            base_params = {k: v for k, v in model.named_parameters()}
            adapted_params = maml_inner_update(model, base_params, x_s, y_s, inner_lr, inner_steps, loss_fn)

            # Query loss for meta-update (FO-MAML)
            pred_q = functional_call(model, adapted_params, (x_q,))
            loss_q = loss_fn(pred_q, y_q)
            meta_loss += loss_q

        meta_loss = meta_loss / meta_batch
        opt.zero_grad()
        meta_loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            m = evaluate_maml_single(model, val_ds, k_shot=k_shot, inner_lr=inner_lr, inner_steps=inner_steps, device=device)
            print(f"[Epoch {ep:03d}] meta_loss={meta_loss.item():.4f} | val_mae_0={m.mae_0:.3f} val_mae_adapt={m.mae_adapt:.3f}")

    return model


@torch.no_grad()
def evaluate_maml_single(
    model: nn.Module,
    test_ds,
    k_shot: int = 5,
    q_size: int | None = None,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    device: str = "cpu",
    tasks: int = 50,
) -> MAMLMetrics:
    model.eval()
    loss_fn = _mse

    maes_0 = []
    maes_adapt = []

    for _ in range(tasks):
        task = test_ds.sample_task(k_shot=k_shot, q_size=q_size, time_ordered=True)
        x_s = torch.tensor(task.support_x, device=device)
        y_s = torch.tensor(task.support_y, device=device)
        x_q = torch.tensor(task.query_x, device=device)
        y_q = torch.tensor(task.query_y, device=device)

        # 0-step
        base_params = {k: v for k, v in model.named_parameters()}
        pred0 = functional_call(model, base_params, (x_q,))
        mae0 = _mae(pred0, y_q).item()
        maes_0.append(mae0)

        # adapt (need grad, so temporarily enable)
        with torch.enable_grad():
            base_params2 = {k: v.clone().detach().requires_grad_(True) for k, v in model.named_parameters()}
            adapted_params = maml_inner_update(model, base_params2, x_s, y_s, inner_lr, inner_steps, loss_fn)
            pred1 = functional_call(model, adapted_params, (x_q,))
            mae1 = _mae(pred1, y_q).item()
            maes_adapt.append(mae1)

    return MAMLMetrics(mae_0=float(np.mean(maes_0)), mae_adapt=float(np.mean(maes_adapt)))


# ---- Multi-task (fitness + trend) ----

def maml_inner_update_multi(
    model: nn.Module,
    params: dict,
    x_s: torch.Tensor,
    y1_s: torch.Tensor,
    y2_s: torch.Tensor,
    inner_lr: float,
    steps: int,
):
    for _ in range(steps):
        p1, p2 = functional_call(model, params, (x_s,))
        loss = torch.mean((p1 - y1_s) ** 2) + torch.mean((p2 - y2_s) ** 2)
        grads = torch.autograd.grad(loss, params.values(), create_graph=False)
        params = {k: v - inner_lr * g for (k, v), g in zip(params.items(), grads)}
    return params


def train_maml_multi(
    model: nn.Module,
    train_ds,
    val_ds,
    epochs: int = 30,
    meta_batch: int = 8,
    k_shot: int = 5,
    q_size: int = 10,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    outer_lr: float = 1e-3,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)

    for ep in range(1, epochs + 1):
        model.train()
        meta_loss = 0.0

        for _ in range(meta_batch):
            task = train_ds.sample_task(k_shot=k_shot, q_size=q_size, time_ordered=True)

            x_s = torch.tensor(task.support_x, device=device)
            y1_s = torch.tensor(task.support_y1, device=device)
            y2_s = torch.tensor(task.support_y2, device=device)

            x_q = torch.tensor(task.query_x, device=device)
            y1_q = torch.tensor(task.query_y1, device=device)
            y2_q = torch.tensor(task.query_y2, device=device)

            base_params = {k: v for k, v in model.named_parameters()}
            adapted_params = maml_inner_update_multi(model, base_params, x_s, y1_s, y2_s, inner_lr, inner_steps)

            p1, p2 = functional_call(model, adapted_params, (x_q,))
            loss_q = torch.mean((p1 - y1_q) ** 2) + torch.mean((p2 - y2_q) ** 2)
            meta_loss += loss_q

        meta_loss = meta_loss / meta_batch
        opt.zero_grad()
        meta_loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            mae0, maeA = evaluate_maml_multi(model, val_ds, k_shot=k_shot, inner_lr=inner_lr, inner_steps=inner_steps, device=device)
            print(f"[Epoch {ep:03d}] meta_loss={meta_loss.item():.4f} | val_mae0={mae0:.3f} val_maeAdapt={maeA:.3f}")

    return model


@torch.no_grad()
def evaluate_maml_multi(
    model: nn.Module,
    test_ds,
    k_shot: int = 5,
    q_size: int | None = None,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    device: str = "cpu",
    tasks: int = 50,
):
    model.eval()
    maes0 = []
    maesA = []

    for _ in range(tasks):
        task = test_ds.sample_task(k_shot=k_shot, q_size=q_size, time_ordered=True)

        x_s = torch.tensor(task.support_x, device=device)
        y1_s = torch.tensor(task.support_y1, device=device)
        y2_s = torch.tensor(task.support_y2, device=device)

        x_q = torch.tensor(task.query_x, device=device)
        y1_q = torch.tensor(task.query_y1, device=device)
        y2_q = torch.tensor(task.query_y2, device=device)

        # 0-step
        base_params = {k: v for k, v in model.named_parameters()}
        p1_0, p2_0 = functional_call(model, base_params, (x_q,))
        mae0 = (torch.mean(torch.abs(p1_0 - y1_q)) + torch.mean(torch.abs(p2_0 - y2_q))) / 2
        maes0.append(mae0.item())

        # adapt
        with torch.enable_grad():
            base_params2 = {k: v.clone().detach().requires_grad_(True) for k, v in model.named_parameters()}
            adapted_params = maml_inner_update_multi(model, base_params2, x_s, y1_s, y2_s, inner_lr, inner_steps)
            p1_1, p2_1 = functional_call(model, adapted_params, (x_q,))
            maeA = (torch.mean(torch.abs(p1_1 - y1_q)) + torch.mean(torch.abs(p2_1 - y2_q))) / 2
            maesA.append(maeA.item())

    return float(np.mean(maes0)), float(np.mean(maesA))
