"""Toy models + synthetic data + fixture-cache writer for harness tests.

Models honor the drop-in contract: ``fit(X, y)`` then ``predict(X) -> probs in [0,1]``.
"""
from __future__ import annotations

import numpy as np

from aide.harness.eval import Dataset


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    """Tiny L2 logistic regression by gradient descent (numpy only)."""

    def __init__(self, lr=0.5, iters=400, l2=1e-3):
        self.lr, self.iters, self.l2 = lr, iters, l2
        self.w = None
        self.mu = None
        self.sd = None

    def _std(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # standardize on TRAIN stats (a real logistic member preprocesses its own
        # features; without this, raw-scale identity columns blow up the logits at OOF)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Xb = np.column_stack([np.ones(len(X)), self._std(X)])
        w = np.zeros(Xb.shape[1])
        for _ in range(self.iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / len(Xb) + self.l2 * w
            w = w - self.lr * grad
        self.w = w
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(np.asarray(X))), self._std(X)])
        return _sigmoid(Xb @ self.w)


class MemorizerModel:
    """Leakage detector. Memorizes the values of one feature column during fit; predicts
    1.0 for rows whose value was seen in training, else 0.5. Under correct OOF the value
    is unseen -> 0.5, so a 1.0 anywhere proves a train/predict leak."""

    def __init__(self, key_col=0):
        self.key_col = key_col
        self.seen = set()

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.seen = set(np.round(X[:, self.key_col], 6).tolist())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        keys = np.round(X[:, self.key_col], 6)
        return np.where(np.isin(keys, list(self.seen)), 1.0, 0.5)


class CaptureModel:
    """Records the X it was last fit on (to inspect dropout masking); predicts 0.5."""

    def __init__(self, sink):
        self.sink = sink

    def fit(self, X, y):
        self.fit_X = np.asarray(X, dtype=float).copy()
        self.sink.append(self)
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), 0.5)


# columns: [item_id (neutral), subject_key (subject proxy), benchmark (benchmark proxy),
#           sig0, sig1 (neutral signal)]
COLUMNS = ["item_id", "subject_key", "benchmark", "sig0", "sig1"]


def make_dataset(n_items=30, rows_per_item=4, n_subjects=6, n_benchmarks=4, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    item_ids = np.repeat(np.arange(n_items), rows_per_item)
    n = len(item_ids)
    subj = rng.integers(0, n_subjects, n)
    bench = rng.integers(0, n_benchmarks, n)
    sig = rng.normal(size=(n, 2))
    logit = 1.5 * sig[:, 0] - 1.0 * sig[:, 1]
    y = (rng.random(n) < _sigmoid(logit)).astype(float)
    X = np.column_stack([item_ids.astype(float), subj.astype(float),
                         bench.astype(float), sig]).astype(np.float32)
    return Dataset(
        X=X, feature_columns=list(COLUMNS), y=y,
        item_keys=np.array([f"item{i}" for i in item_ids]),
        subjects=np.array([f"s{i}" for i in subj]),
        benchmarks=np.array([f"b{i}" for i in bench]))


def write_fixture_cache(root, group, X, columns, row_ids):
    root = __import__("pathlib").Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / f"{group}.npz",
             X=np.asarray(X, dtype=np.float32),
             columns=np.asarray(columns, dtype=str),
             row_ids=np.asarray(row_ids, dtype=str))
