"""Tiny calibration utilities for the runtime `predict()`.

The active platform reveals K=5 labels per round per data category. Anything
more elaborate than a 1- or 2-parameter calibrator on 5*15 = 75 labels
overfits and degrades the score. We therefore implement only:

- ``IdentityCalibrator``: no change. Always the safe fallback.
- ``InterceptShiftCalibrator``: shift logits by a single scalar (1 free param).
- ``TemperatureInterceptCalibrator``: temperature + intercept (2 free params).

The fitter is closed-form-ish: a few Newton steps on the BCE objective.
If the fit ever fails (singular, infinite loss, fewer than N_MIN labels) we
return ``IdentityCalibrator`` -- the caller never has to worry about NaNs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


EPS = 1e-7


def _logit(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


@dataclass
class IdentityCalibrator:
    """Calibrator with no parameters. Always safe."""

    def apply(self, p: float) -> float:
        if not math.isfinite(p):
            return 0.5
        return min(max(p, EPS), 1.0 - EPS)

    def to_dict(self) -> dict:
        return {"kind": "identity"}


@dataclass
class InterceptShiftCalibrator:
    """logit(p') = logit(p) + b, 1 free parameter."""

    b: float = 0.0

    def apply(self, p: float) -> float:
        if not math.isfinite(p):
            return 0.5
        z = _logit(p) + self.b
        return min(max(_sigmoid(z), EPS), 1.0 - EPS)

    def to_dict(self) -> dict:
        return {"kind": "intercept", "b": float(self.b)}


@dataclass
class TemperatureInterceptCalibrator:
    """logit(p') = logit(p) / T + b, 2 free parameters.

    Requires more labels than the intercept-only variant; we only fall back
    to this when at least ``N_MIN_TEMP`` labels are available.
    """

    T: float = 1.0
    b: float = 0.0

    def apply(self, p: float) -> float:
        if not math.isfinite(p):
            return 0.5
        z = _logit(p) / max(0.1, self.T) + self.b
        return min(max(_sigmoid(z), EPS), 1.0 - EPS)

    def to_dict(self) -> dict:
        return {"kind": "temp_intercept", "T": float(self.T), "b": float(self.b)}


Calibrator = (
    IdentityCalibrator | InterceptShiftCalibrator | TemperatureInterceptCalibrator
)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


N_MIN_INTERCEPT = 5
N_MIN_TEMP = 30


def fit_intercept(p: Sequence[float], y: Sequence[float]) -> InterceptShiftCalibrator:
    """Fit a single-bias logistic regression on (logit(p), y)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size < N_MIN_INTERCEPT or p.size != y.size:
        return InterceptShiftCalibrator(0.0)
    z = np.log(np.clip(p, EPS, 1 - EPS)) - np.log(
        np.clip(1.0 - p, EPS, 1 - EPS)
    )
    b = 0.0
    for _ in range(50):
        pred = 1.0 / (1.0 + np.exp(-(z + b)))
        grad = float((pred - y).mean())
        hess = float((pred * (1.0 - pred)).mean()) + 1e-6
        step = grad / hess
        b -= step
        if abs(step) < 1e-6:
            break
    if not math.isfinite(b):
        return InterceptShiftCalibrator(0.0)
    return InterceptShiftCalibrator(float(b))


def fit_temperature_intercept(
    p: Sequence[float], y: Sequence[float]
) -> TemperatureInterceptCalibrator:
    """Fit (T, b) by 2D Newton on BCE. Falls back to identity on failure."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size < N_MIN_TEMP or p.size != y.size:
        return TemperatureInterceptCalibrator(1.0, 0.0)
    z = np.log(np.clip(p, EPS, 1 - EPS)) - np.log(
        np.clip(1.0 - p, EPS, 1 - EPS)
    )
    T = 1.0
    b = 0.0
    for _ in range(100):
        zt = z / T + b
        pred = 1.0 / (1.0 + np.exp(-zt))
        err = pred - y
        # gradients wrt (b, alpha=1/T)
        d_b = float(err.mean())
        d_alpha = float((err * z).mean())  # alpha = 1/T
        # diagonal-ish Hessian (cheap, robust)
        h_b = float((pred * (1.0 - pred)).mean()) + 1e-6
        h_alpha = float((pred * (1.0 - pred) * z * z).mean()) + 1e-6
        new_b = b - d_b / h_b
        new_alpha = (1.0 / T) - d_alpha / h_alpha
        new_T = 1.0 / max(0.1, new_alpha)
        if not (math.isfinite(new_b) and math.isfinite(new_T)):
            return TemperatureInterceptCalibrator(1.0, 0.0)
        if abs(new_b - b) + abs(new_T - T) < 1e-6:
            b, T = new_b, new_T
            break
        b, T = new_b, new_T
    return TemperatureInterceptCalibrator(float(T), float(b))


def best_effort_fit(
    p: Sequence[float], y: Sequence[float]
) -> Calibrator:
    """Pick the simplest calibrator the label budget supports, with fallback."""
    try:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        if p.size == 0 or p.size != y.size:
            return IdentityCalibrator()
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y))):
            return IdentityCalibrator()
        # Need at least one positive and one negative label or the fit is degenerate.
        yb = (y >= 0.5).astype(int)
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            return IdentityCalibrator()
        if p.size >= N_MIN_TEMP:
            cal = fit_temperature_intercept(p, y)
            # Sanity-check the fitted output.
            out = np.array([cal.apply(float(x)) for x in p])
            if not np.all(np.isfinite(out)):
                return IdentityCalibrator()
            return cal
        return fit_intercept(p, y)
    except Exception:
        return IdentityCalibrator()


# ---------------------------------------------------------------------------
# (De)serialize for embedding in submission/model.py
# ---------------------------------------------------------------------------


def calibrator_from_dict(d: dict) -> Calibrator:
    kind = (d or {}).get("kind", "identity")
    if kind == "intercept":
        return InterceptShiftCalibrator(b=float(d.get("b", 0.0)))
    if kind == "temp_intercept":
        return TemperatureInterceptCalibrator(
            T=float(d.get("T", 1.0)), b=float(d.get("b", 0.0))
        )
    return IdentityCalibrator()


__all__ = [
    "Calibrator",
    "IdentityCalibrator",
    "InterceptShiftCalibrator",
    "TemperatureInterceptCalibrator",
    "best_effort_fit",
    "calibrator_from_dict",
    "fit_intercept",
    "fit_temperature_intercept",
]
