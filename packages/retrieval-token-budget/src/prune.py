"""Adaptive token pruning for image-text retrieval."""

import torch
from torch import nn


class BudgetPredictor(nn.Module):
    """Predict how many tokens each modality should keep."""

    def __init__(self, dim: int, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        logits = self.head(pooled).squeeze(-1)
        return (logits.sigmoid() * self.max_tokens).round().long()


def prune_tokens(tokens: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Select the top-`keep` tokens per example by norm."""
    scores = tokens.norm(dim=-1)
    order = scores.argsort(dim=-1, descending=True)
    out = []
    for i, k in enumerate(keep.tolist()):
        out.append(tokens[i, order[i, : int(k)]])
    return torch.stack(out).mean(dim=1)
