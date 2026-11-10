"""Token pruning utilities for multimodal transformers."""

import torch
from torch import nn


class TokenPruner(nn.Module):
    """Score and drop low-salience tokens before fusion."""

    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.scorer = nn.Linear(1, 1)

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        b, n, d = token_states.shape
        k = max(1, int(n * self.keep_ratio))
        salience = token_states.norm(dim=-1)
        index = salience.topk(k, dim=-1).indices
        gathered = torch.gather(
            token_states, 1, index.unsqueeze(-1).expand(-1, -1, d)
        )
        return gathered.mean(dim=1)


def apply_pruning(model: nn.Module, images, texts, pruner: TokenPruner):
    image_tokens = model.encode_image(images)
    text_tokens = model.encode_text(texts)
    image_pooled = pruner(image_tokens)
    text_pooled = pruner(text_tokens)
    return image_pooled, text_pooled
