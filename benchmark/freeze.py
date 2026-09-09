from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tasks import repository_root

LOCK_NAME = "fixtures.lock.json"
INCLUDED_DIRECTORIES = ("fixtures", "tasks/prompts")


def _is_hashable_fixture(path: Path) -> bool:
    """Exclude interpreter cache artifacts from the source fixture lock."""
    return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_hashes(root: Path | None = None) -> dict[str, str]:
    root = root or repository_root()
    hashes: dict[str, str] = {}
    for directory in INCLUDED_DIRECTORIES:
        base = root / directory
        for path in sorted(p for p in base.rglob("*") if p.is_file() and _is_hashable_fixture(p)):
            hashes[path.relative_to(root).as_posix()] = file_hash(path)
    return hashes


def write_lock(root: Path | None = None) -> Path:
    root = root or repository_root()
    lock = root / LOCK_NAME
    lock.write_text(json.dumps({"algorithm": "sha256", "files": collect_hashes(root)}, indent=2) + "\n")
    return lock


def verify_lock(root: Path | None = None) -> list[str]:
    root = root or repository_root()
    expected = json.loads((root / LOCK_NAME).read_text())["files"]
    actual = collect_hashes(root)
    problems: list[str] = []
    for path in sorted(set(expected) | set(actual)):
        if expected.get(path) != actual.get(path):
            problems.append(f"frozen fixture differs: {path}")
    return problems
