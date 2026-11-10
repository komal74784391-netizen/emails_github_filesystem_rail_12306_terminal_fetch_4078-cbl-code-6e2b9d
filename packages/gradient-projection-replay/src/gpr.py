"""Gradient projection replay for continual learning."""

import torch
from torch import nn


class GradientProjectionReplay(nn.Module):
    """Wrap a model and project gradients away from old task directions."""

    def __init__(self, model: nn.Module, rank: int = 4):
        super().__init__()
        self.model = model
        self.bases = {}
        self.rank = rank

    def register_task(self, task_id: int, features: torch.Tensor):
        with torch.no_grad():
            u, s, _v = torch.linalg.svd(features, full_matrices=False)
            self.bases[task_id] = u[:, : self.rank] * s[: self.rank]

    def project(self, grad: torch.Tensor, task_id: int) -> torch.Tensor:
        basis = self.bases.get(task_id)
        if basis is None:
            return grad
        return grad - basis @ (basis.T @ grad)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def project_gradients(module: GradientProjectionReplay, task_id: int):
    for param in module.model.parameters():
        if param.grad is None:
            continue
        flat = param.grad.reshape(param.grad.shape[0], -1)
        flat = module.project(flat, task_id)
        param.grad = flat.reshape_as(param.grad)
