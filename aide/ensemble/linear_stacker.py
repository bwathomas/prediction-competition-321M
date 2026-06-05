"""LinearStacker — the linear stacker used at BOTH ensemble layers.

Logistic regression over the per-member prediction columns, fit in LOGIT space (the
natural scale for blending probabilities) on standardized features. ``nonneg=True``
constrains member weights to be non-negative — a sensible prior for a probability blend
and a guard against a member entering with a large negative weight off noise.
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LinearStacker:
    def __init__(self, l2: float = 1e-3, iters: int = 500, lr: float = 0.3, nonneg: bool = False):
        self.l2, self.iters, self.lr, self.nonneg = l2, iters, lr, nonneg
        self.w = None
        self.mu = None
        self.sd = None

    def _design(self, P):
        Z = _logit(P)
        Zs = (Z - self.mu) / self.sd
        return np.column_stack([np.ones(len(Zs)), Zs])

    def fit(self, P, y):
        P = np.asarray(P, dtype=float)
        y = np.asarray(y, dtype=float)
        Z = _logit(P)
        self.mu = Z.mean(axis=0)
        self.sd = Z.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0  # near-constant columns too, not just exact 0 (M1)
        Xb = self._design(P)
        w = np.zeros(Xb.shape[1])
        for _ in range(self.iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / len(Xb) + self.l2 * w
            w = w - self.lr * grad
            if self.nonneg:
                w[1:] = np.maximum(w[1:], 0.0)  # member weights >= 0 (bias free)
        self.w = w
        return self

    def predict(self, P):
        return _sigmoid(self._design(np.asarray(P, dtype=float)) @ self.w)
