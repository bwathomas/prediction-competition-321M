"""Canonical, cross-agent OOF split manifest.

Item->fold assignment is a deterministic hash of (seed, item_key) so all three
agents (llama/qwen/mistral) reproduce BYTE-IDENTICAL folds without sharing state.
``assert_identical`` is the cross-agent leakage guard.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def item_fold(item_key: str, n_folds: int, seed: int) -> int:
    """Deterministic fold id for one item_key. Stable across machines/Python runs."""
    h = hashlib.sha256(f"{seed}:{item_key}".encode("utf-8")).hexdigest()
    return int(h, 16) % n_folds


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    n_folds: int
    assignment: dict  # item_key (str) -> fold id (int)

    def fold_of(self, item_key: str) -> int:
        return self.assignment[str(item_key)]

    def save(self, path) -> None:
        payload = {"seed": self.seed, "n_folds": self.n_folds, "assignment": self.assignment}
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path) -> "SplitManifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return SplitManifest(seed=int(d["seed"]), n_folds=int(d["n_folds"]),
                             assignment={str(k): int(v) for k, v in d["assignment"].items()})


def build_manifest(item_keys, n_folds: int = 3, seed: int = 0) -> SplitManifest:
    uniq = sorted(set(str(k) for k in item_keys))
    assignment = {k: item_fold(k, n_folds, seed) for k in uniq}
    return SplitManifest(seed=seed, n_folds=n_folds, assignment=assignment)


def assert_identical(a: SplitManifest, b: SplitManifest) -> None:
    if (a.seed, a.n_folds, a.assignment) != (b.seed, b.n_folds, b.assignment):
        raise AssertionError("SplitManifest mismatch across agents — cross-agent leakage risk")
