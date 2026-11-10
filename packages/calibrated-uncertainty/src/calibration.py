"""Structure-aware calibration for segmentation models."""

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Learn a single temperature per anatomical structure."""

    def __init__(self, num_structures: int):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_structures))

    def forward(self, logits: torch.Tensor, structures: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature[structures].exp()
        while temperature.dim() < logits.dim():
            temperature = temperature.unsqueeze(-1)
        return logits / temperature


def expected_calibration_error(confidence, correct, bins: int = 15):
    edges = torch.linspace(0, 1, bins + 1)
    ece = torch.tensor(0.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].float().mean()
        conf = confidence[mask].mean()
        ece += (acc - conf).abs() * mask.float().mean()
    return ece.item()
