"""Slightly smarter baseline: average the labels we were shown adaptively.

If `labeled` is non-empty, predict the mean of the revealed labels (clipped
to [eps, 1-eps]). Otherwise predict 0.5. Uses no global state.
"""

from __future__ import annotations

EPS = 1e-3


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if not labeled:
        return 0.5
    vals = [float(x["label"]) for x in labeled if "label" in x]
    if not vals:
        return 0.5
    p = sum(vals) / len(vals)
    if p < EPS:
        p = EPS
    if p > 1.0 - EPS:
        p = 1.0 - EPS
    return p
