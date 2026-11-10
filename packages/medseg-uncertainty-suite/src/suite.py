"""Evaluation suite for segmentation uncertainty."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationMetrics:
    dice: float
    boundary_f1: float
    ece: float


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def boundary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pred_edge = pred ^ np.roll(pred, 1, axis=0) ^ np.roll(pred, 1, axis=1)
    target_edge = target ^ np.roll(target, 1, axis=0) ^ np.roll(target, 1, axis=1)
    tp = np.logical_and(pred_edge, target_edge).sum()
    return float(2 * tp / max(pred_edge.sum() + target_edge.sum(), 1))


def summarize(pred: np.ndarray, target: np.ndarray, ece: float) -> SegmentationMetrics:
    return SegmentationMetrics(dice_score(pred, target), boundary_f1(pred, target), ece)
