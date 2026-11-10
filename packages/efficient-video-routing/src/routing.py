"""Learned spatiotemporal token routing for video transformers."""

import torch
from torch import nn
from torch.nn import functional as F


class TokenRouter(nn.Module):
    """Route video tokens into a small set of learned slots."""

    def __init__(self, dim: int, num_routes: int = 8, top_k: int = 2):
        super().__init__()
        self.score = nn.Linear(dim, num_routes)
        self.merge = nn.Linear(dim, dim)
        self.top_k = top_k

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.score(tokens).softmax(dim=-1)
        weights, _routes = scores.topk(self.top_k, dim=-1)
        merged = torch.zeros_like(self.merge(tokens))
        for slot in range(self.top_k):
            slot_w = weights[..., slot].unsqueeze(-1)
            merged = merged + slot_w * self.merge(tokens)
        return F.layer_norm(merged + tokens, tokens.shape[-1:])


def route_attention(query, key, value, router):
    """Apply routing before the attention softmax."""
    q = router(query)
    k = router(key)
    v = router(value)
    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    return attn.softmax(dim=-1) @ v
