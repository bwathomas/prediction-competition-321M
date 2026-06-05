import math
import numpy as np
import pytest
from aide.ensemble.registry import get, smoke_test
from aide.hygiene.manifest import build_manifest
from aide.harness.tests._toy import make_dataset


def _manifest(ds):
    return build_manifest(ds.item_keys, n_folds=3, seed=0)


def test_get_and_smoke_numpy_archs_pass():
    ds = make_dataset(seed=2)
    m = _manifest(ds)
    for name in ["logistic", "mlp"]:
        r = smoke_test(get(name), ds, m)
        assert r.ok and math.isfinite(r.nll)


def test_smoke_catches_broken_factory():
    class Broken:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(np.asarray(X)), np.nan)

    ds = make_dataset(seed=2)
    r = smoke_test(lambda: Broken(), ds, _manifest(ds))
    assert r.ok is False and r.error is not None


def test_unknown_arch_raises():
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_lightgbm_lazy_requires_colab_when_absent():
    f = get("gbdt_lightgbm")
    try:
        import lightgbm  # noqa: F401
        have = True
    except Exception:
        have = False
    if have:
        assert f() is not None
    else:
        with pytest.raises(RuntimeError):
            f()
