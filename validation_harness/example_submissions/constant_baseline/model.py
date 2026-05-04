"""Trivial baseline: always predict 0.5.

Useful as a smoke test and as a log-likelihood floor (mean LL = log(0.5)
~ -0.6931 regardless of label distribution).
"""

from __future__ import annotations


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    return 0.5
