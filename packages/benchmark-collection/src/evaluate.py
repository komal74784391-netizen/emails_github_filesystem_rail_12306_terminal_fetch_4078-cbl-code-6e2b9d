"""Unified evaluation entry point for the lab benchmark collection."""

import csv
import json
from pathlib import Path


def load_predictions(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"id": row["id"], "pred": row["pred"], "gold": row["gold"]})
    return rows


def evaluate(rows):
    correct = sum(1 for row in rows if row["pred"] == row["gold"])
    return {"accuracy": correct / max(len(rows), 1), "count": len(rows)}


def export(result: dict, path: Path):
    path.write_text(json.dumps(result, indent=2))
