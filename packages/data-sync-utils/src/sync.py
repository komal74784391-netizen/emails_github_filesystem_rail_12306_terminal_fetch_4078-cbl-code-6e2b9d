"""Synchronise dataset metadata across storage mirrors."""

import hashlib
import json
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_manifest(source: Path, target: Path) -> dict:
    manifest = json.loads(source.read_text())
    updated = []
    for entry in manifest["files"]:
        path = target / entry["name"]
        if path.exists() and checksum(path) == entry["sha256"]:
            continue
        updated.append(entry["name"])
    return {"updated": updated, "total": len(manifest["files"])}
