import math
import numpy as np
from aide.ensemble.architectures import LogisticArchitecture, MLPArchitecture
from aide.hygiene.manifest import build_manifest
from aide.harness.eval import Dataset, evaluate
from aide.harness.tests._toy import make_dataset


def _manifest(ds):
    return build_manifest(ds.item_keys, n_folds=3, seed=0)


def test_logistic_beats_chance_on_linear_signal():
    ds = make_dataset(seed=3)
    res = evaluate(lambda: LogisticArchitecture(), ds, _manifest(ds))
    assert res.nll < math.log(2)


def test_mlp_beats_chance_on_linear_signal():
    ds = make_dataset(seed=3)
    res = evaluate(lambda: MLPArchitecture(hidden=16, iters=500), ds, _manifest(ds))
    assert res.nll < math.log(2)


def _xor_dataset(n_items=60, rows_per_item=4, seed=0):
    rng = np.random.default_rng(seed)
    item_ids = np.repeat(np.arange(n_items), rows_per_item)
    n = len(item_ids)
    sig = rng.normal(size=(n, 2))
    y = ((sig[:, 0] > 0) ^ (sig[:, 1] > 0)).astype(float)  # nonlinear: XOR
    X = np.column_stack([item_ids.astype(float), sig]).astype(np.float32)
    return Dataset(X=X, feature_columns=["item_id", "sig0", "sig1"], y=y,
                   item_keys=np.array([f"i{i}" for i in item_ids]),
                   subjects=np.array(["s0"] * n), benchmarks=np.array(["b0"] * n))


def test_mlp_beats_logistic_on_nonlinear_signal():
    ds = _xor_dataset(seed=1)
    m = _manifest(ds)
    mlp = evaluate(lambda: MLPArchitecture(hidden=32, iters=1500, lr=0.3), ds, m)
    log = evaluate(lambda: LogisticArchitecture(), ds, m)
    assert mlp.auc is not None and mlp.auc > 0.8   # MLP learns XOR
    assert log.auc is not None and log.auc < 0.65  # logistic cannot
    assert mlp.nll < log.nll
