"""Architecture canon (spec §6.1). The pure-numpy members (LogisticArchitecture,
MLPArchitecture) are tested locally; the heavy Kaggle libraries are registered behind
lazy loaders that raise a clear error when the lib is absent, so local CI stays numpy-only
while the same registry entry trains for real on the Colab A100.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _Standardizer:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def __call__(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd


class LogisticArchitecture:
    """Standardized L2 logistic regression (numpy GD)."""

    def __init__(self, lr: float = 0.5, iters: int = 400, l2: float = 1e-3):
        self.lr, self.iters, self.l2 = lr, iters, l2
        self.std = _Standardizer()
        self.w = None

    def _design(self, X):
        Xs = self.std(X)
        return np.column_stack([np.ones(len(Xs)), Xs])

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.std.fit(X)
        Xb = self._design(X)
        w = np.zeros(Xb.shape[1])
        for _ in range(self.iters):
            p = _sigmoid(Xb @ w)
            w = w - self.lr * (Xb.T @ (p - y) / len(Xb) + self.l2 * w)
        self.w = w
        return self

    def predict(self, X):
        return _sigmoid(self._design(X) @ self.w)


class MLPArchitecture:
    """One hidden ReLU layer, He init, standardized inputs, seeded (numpy)."""

    def __init__(self, hidden: int = 16, iters: int = 400, lr: float = 0.1,
                 l2: float = 1e-4, seed: int = 0):
        self.hidden, self.iters, self.lr, self.l2, self.seed = hidden, iters, lr, l2, seed
        self.std = _Standardizer()

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.std.fit(X)
        Xs = self.std(X)
        n, d = Xs.shape
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden), (self.hidden,))
        self.b2 = 0.0
        for _ in range(self.iters):
            H = np.maximum(Xs @ self.W1 + self.b1, 0.0)
            p = _sigmoid(H @ self.W2 + self.b2)
            do = (p - y) / n
            dW2 = H.T @ do + self.l2 * self.W2
            db2 = do.sum()
            dH = np.outer(do, self.W2) * (H > 0)
            dW1 = Xs.T @ dH + self.l2 * self.W1
            db1 = dH.sum(axis=0)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
        return self

    def predict(self, X):
        Xs = self.std(X)
        H = np.maximum(Xs @ self.W1 + self.b1, 0.0)
        return _sigmoid(H @ self.W2 + self.b2)


def _lazy(lib_name, builder):
    """Return a factory that builds a heavy model, or raises a clear error if the
    library is unavailable (it is present on the Colab A100, not on local CI)."""
    def factory(**kw):
        try:
            __import__(lib_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"architecture requires {lib_name!r}, which is available on the Colab "
                f"A100 runtime but not in this environment") from exc
        return builder(**kw)
    return factory


# Heavy Kaggle-canon entries (built lazily on Colab; see registry.REGISTRY).
def _build_lightgbm(**kw):  # pragma: no cover - exercised only where lightgbm is installed
    import lightgbm as lgb  # noqa: F401
    return _LightGBMWrapper(**kw)


class _LightGBMWrapper:  # pragma: no cover - Colab-only
    def __init__(self, num_leaves=31, n_estimators=200, learning_rate=0.05, **kw):
        import lightgbm as lgb
        self._clf = lgb.LGBMClassifier(
            num_leaves=num_leaves, n_estimators=n_estimators,
            learning_rate=learning_rate, **kw)

    def fit(self, X, y):
        self._clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float) > 0.5)
        return self

    def predict(self, X):
        return self._clf.predict_proba(np.asarray(X, dtype=float))[:, 1]


lazy_lightgbm = _lazy("lightgbm", _build_lightgbm)
