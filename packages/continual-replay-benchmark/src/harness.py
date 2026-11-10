"""Benchmark harness for continual learning experiments."""

import json
from pathlib import Path

import torch


TASKS = ["permuted_mnist", "split_cifar", "split_miniimagenet"]


def build_sequence(root: Path, task: str, batch_size: int = 64):
    if task == "permuted_mnist":
        return torch.utils.data.DataLoader(
            torchvision_mnist(root), batch_size=batch_size, shuffle=True
        )
    raise ValueError(f"unknown task {task}")


def torchvision_mnist(root: Path):
    import torchvision

    return torchvision.datasets.MNIST(root=str(root), download=True, transform=None)


def summarize(history):
    out = {}
    for task, metrics in history.items():
        out[task] = {
            "final_accuracy": metrics[-1]["accuracy"],
            "forgetting": max(m["accuracy"] for m in metrics) - metrics[-1]["accuracy"],
        }
    return out


def save_report(history, path: Path):
    path.write_text(json.dumps(summarize(history), indent=2))
