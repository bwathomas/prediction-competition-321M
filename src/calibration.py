"""Tiny calibration utilities for the runtime `predict()`.

The active platform reveals K=5 labels per round per data category. Anything
more elaborate than a 3-parameter calibrator on 5*15 = 75 labels overfits
and degrades the score. We therefore implement only:

- ``IdentityCalibrator``: no change. Always the safe fallback.
- ``InterceptShiftCalibrator``: shift logits by a single scalar (1 free param).
- ``TemperatureInterceptCalibrator``: temperature + intercept (2 free params).
- ``BetaCalibrator``: Kull/Filho/Flach 2017 beta calibration (3 free params,
  handles asymmetric reliability curves the symmetric Platt sigmoid can't).

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


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull/Filho/Flach 2017), 3 free parameters.

    ``logit(p') = a * log(p) - b * log(1 - p) + c``

    The Kweon et al. 2022 RecSys paper finds that beta calibration
    consistently beats plain Platt sigmoid scaling on ranking-style scores
    because reliability curves are rarely symmetric around 0.5. The extra
    parameter handles asymmetric over/under-confidence (e.g. well-calibrated
    near 0, overconfident near 1).

    Monotonicity requires ``a, b > 0``; we clamp at 0.01 after each Newton
    step so the fit cannot produce a non-monotone calibrator that hurts
    pairwise AUC. ``a = 1, b = 1, c = 0`` is the identity (sigmoid(logit(p))
    = p), so the Newton loop starts from a sane warm point.
    """

    a: float = 1.0
    b: float = 1.0
    c: float = 0.0

    def apply(self, p: float) -> float:
        if not math.isfinite(p):
            return 0.5
        p_safe = min(max(float(p), EPS), 1.0 - EPS)
        z = (
            max(0.01, self.a) * math.log(p_safe)
            - max(0.01, self.b) * math.log(1.0 - p_safe)
            + self.c
        )
        return min(max(_sigmoid(z), EPS), 1.0 - EPS)

    def to_dict(self) -> dict:
        return {
            "kind": "beta",
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
        }


Calibrator = (
    IdentityCalibrator
    | InterceptShiftCalibrator
    | TemperatureInterceptCalibrator
    | BetaCalibrator
)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


N_MIN_INTERCEPT = 5
N_MIN_TEMP = 30
N_MIN_BETA = 60


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


def _beta_loss(
    a: float, b: float, c: float, lp: np.ndarray, lq: np.ndarray, y: np.ndarray
) -> float:
    """Mean BCE under the beta calibration parameterization."""
    z = a * lp - b * lq + c
    # Stable log-sigmoid / log-(1-sigmoid)
    log_p = -np.logaddexp(0.0, -z)
    log_q = -np.logaddexp(0.0, z)
    return float(-(y * log_p + (1.0 - y) * log_q).mean())


def fit_beta(p: Sequence[float], y: Sequence[float]) -> BetaCalibrator:
    """Fit ``(a, b, c)`` by full-Hessian Newton with backtracking.

    The diagonal Hessian approximation that works for the 1- and 2-parameter
    fits diverges here because the (a, b) cross-derivative ``-lp*lq*h`` is
    non-trivial near moderate ``p`` and the diagonal step systematically
    overshoots. We solve the full 3x3 system with a small ridge and accept
    the step only if BCE actually decreased (poor-man's line search). ``a``
    and ``b`` are projected onto ``[0.01, 100]`` so the calibrator stays
    monotone increasing; ``c`` is clamped to ``[-20, 20]``.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size < N_MIN_BETA or p.size != y.size:
        return BetaCalibrator(1.0, 1.0, 0.0)
    p_safe = np.clip(p, EPS, 1.0 - EPS)
    lp = np.log(p_safe)
    lq = np.log(1.0 - p_safe)
    a, b, c = 1.0, 1.0, 0.0
    loss = _beta_loss(a, b, c, lp, lq, y)
    for _ in range(100):
        z = a * lp - b * lq + c
        pred = 1.0 / (1.0 + np.exp(-z))
        err = pred - y
        h = pred * (1.0 - pred)
        # Gradient (mean over batch).
        g = np.array(
            [float((err * lp).mean()), float((-err * lq).mean()), float(err.mean())],
            dtype=np.float64,
        )
        # Full 3x3 Hessian H[i,j] = mean(h * dz/d_i * dz/d_j).
        # (dz/da, dz/db, dz/dc) = (lp, -lq, 1).
        H = np.empty((3, 3), dtype=np.float64)
        H[0, 0] = float((h * lp * lp).mean())
        H[1, 1] = float((h * lq * lq).mean())
        H[2, 2] = float(h.mean())
        H[0, 1] = H[1, 0] = float((h * lp * (-lq)).mean())
        H[0, 2] = H[2, 0] = float((h * lp).mean())
        H[1, 2] = H[2, 1] = float((h * (-lq)).mean())
        # Ridge for numerical stability when the batch is small or extreme.
        H += 1e-6 * np.eye(3)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        # Backtracking: try the full step, then halve up to 6 times.
        step = 1.0
        accepted = False
        for _bt in range(7):
            new_a = min(100.0, max(0.01, a - step * float(delta[0])))
            new_b = min(100.0, max(0.01, b - step * float(delta[1])))
            new_c = min(20.0, max(-20.0, c - step * float(delta[2])))
            if not (
                math.isfinite(new_a) and math.isfinite(new_b) and math.isfinite(new_c)
            ):
                step *= 0.5
                continue
            new_loss = _beta_loss(new_a, new_b, new_c, lp, lq, y)
            if math.isfinite(new_loss) and new_loss <= loss + 1e-12:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        if abs(new_a - a) + abs(new_b - b) + abs(new_c - c) < 1e-7:
            a, b, c, loss = new_a, new_b, new_c, new_loss
            break
        a, b, c, loss = new_a, new_b, new_c, new_loss
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return BetaCalibrator(1.0, 1.0, 0.0)
    return BetaCalibrator(float(a), float(b), float(c))


def best_effort_fit(
    p: Sequence[float], y: Sequence[float]
) -> Calibrator:
    """Pick the simplest calibrator the label budget supports, with fallback.

    Tier order (richest first):
      - ``N >= N_MIN_BETA``: 3-param beta calibration; fall back to
        ``temp_intercept`` if the fitted output is non-finite.
      - ``N >= N_MIN_TEMP``: 2-param temperature + intercept; fall back to
        identity if the fitted output is non-finite.
      - ``N >= N_MIN_INTERCEPT``: 1-param intercept only.
      - Otherwise: identity.
    """
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
        if p.size >= N_MIN_BETA:
            cal: Calibrator = fit_beta(p, y)
            out = np.array([cal.apply(float(x)) for x in p])
            if not np.all(np.isfinite(out)):
                cal = fit_temperature_intercept(p, y)
                out = np.array([cal.apply(float(x)) for x in p])
                if not np.all(np.isfinite(out)):
                    return IdentityCalibrator()
            return cal
        if p.size >= N_MIN_TEMP:
            cal = fit_temperature_intercept(p, y)
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
    if kind == "beta":
        return BetaCalibrator(
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 1.0)),
            c=float(d.get("c", 0.0)),
        )
    return IdentityCalibrator()


__all__ = [
    "BetaCalibrator",
    "Calibrator",
    "IdentityCalibrator",
    "InterceptShiftCalibrator",
    "TemperatureInterceptCalibrator",
    "best_effort_fit",
    "calibrator_from_dict",
    "fit_beta",
    "fit_intercept",
    "fit_temperature_intercept",
]
