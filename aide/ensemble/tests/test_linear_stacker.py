import numpy as np
from aide.ensemble.linear_stacker import LinearStacker
from aide.harness.metrics import log_loss


def _noisy_member(y, seed, noise):
    r = np.random.default_rng(seed)
    p = 0.5 + (y - 0.5) * 0.6 + r.normal(0, noise, len(y))
    return np.clip(p, 0.01, 0.99)


def test_stacker_is_no_worse_than_its_best_member():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(float)
    P = np.column_stack([_noisy_member(y, 1, 0.2), _noisy_member(y, 2, 0.25)])
    s = LinearStacker().fit(P, y)
    pred = s.predict(P)
    assert log_loss(y, pred) <= min(log_loss(y, P[:, 0]), log_loss(y, P[:, 1])) + 1e-3
    assert (pred >= 0).all() and (pred <= 1).all()


def test_nonneg_keeps_member_weights_nonnegative():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.5).astype(float)
    good = _noisy_member(y, 3, 0.2)
    anti = 1.0 - good  # anti-correlated -> would earn a negative weight unconstrained
    P = np.column_stack([good, anti])
    s = LinearStacker(nonneg=True).fit(P, y)
    assert np.all(s.w[1:] >= -1e-9)  # member weights non-negative (bias unconstrained)
