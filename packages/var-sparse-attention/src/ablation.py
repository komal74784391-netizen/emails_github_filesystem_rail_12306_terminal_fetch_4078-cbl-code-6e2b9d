"""Ablation runner for sparse video attention routing."""

import argparse
import json
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class AblationConfig:
    route_count: int = 8
    top_k: int = 2
    frames: int = 16
    resolution: int = 224


def run_ablation(model, loader: DataLoader, config: AblationConfig):
    device = next(model.parameters()).device
    totals = {"loss": 0.0, "correct": 0, "examples": 0}
    model.eval()
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            logits = model(frames)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            totals["loss"] += float(loss) * frames.shape[0]
            totals["correct"] += int((logits.argmax(-1) == labels).sum())
            totals["examples"] += frames.shape[0]
    return {
        "loss": totals["loss"] / max(totals["examples"], 1),
        "accuracy": totals["correct"] / max(totals["examples"], 1),
        "config": config.__dict__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="{}")
    args = parser.parse_args()
    config = AblationConfig(**json.loads(args.config))
    print(json.dumps(config.__dict__, indent=2))


if __name__ == "__main__":
    main()
