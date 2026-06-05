import math
import numpy as np
from aide.hygiene.manifest import build_manifest
from aide.harness.eval import (
    DropoutConfig, oof_predict, evaluate, build_oof_meta, recursive_evaluate)
from aide.harness.tests._toy import (
    LogisticModel, MemorizerModel, CaptureModel, make_dataset)


def _manifest(ds, seed=0):
    return build_manifest(ds.item_keys, n_folds=3, seed=seed)


def test_oof_predict_fills_one_prediction_per_row():
    ds = make_dataset(seed=1)
    oof = oof_predict(lambda: LogisticModel(), ds, _manifest(ds))
    assert oof.shape == (len(ds.y),)
    assert not np.isnan(oof).any()


def test_oof_is_item_leakage_free_memorizer_scores_half():
    # MemorizerModel keyed on item_id (col 0): under item-uniform OOF, every OOF row's
    # item was never in its trainer's fold -> 0.5 everywhere. A 1.0 would prove leakage.
    ds = make_dataset(seed=2)
    oof = oof_predict(lambda: MemorizerModel(key_col=0), ds, _manifest(ds))
    assert np.allclose(oof, 0.5)


def test_evaluate_beats_chance_on_learnable_signal():
    ds = make_dataset(seed=3)
    res = evaluate(lambda: LogisticModel(), ds, _manifest(ds))
    assert res.nll < math.log(2)  # better than always-0.5
    assert res.auc is not None and res.auc > 0.6


def test_subject_dropout_zeros_subject_proxy_in_training_X():
    ds = make_dataset(seed=4)
    captured = []
    oof_predict(lambda: CaptureModel(captured), ds, _manifest(ds),
                dropout=DropoutConfig(subject_rate=1.0), seed=0)
    j = ds.feature_columns.index("subject_key")
    assert captured  # models were fit
    for m in captured:
        assert np.all(m.fit_X[:, j] == 0.0)


def test_evaluate_is_reproducible_under_partial_dropout():
    # C1 regression: a fold-deterministic dropped set (not an advancing rng) makes the
    # score independent of evaluation order — two runs with the same seed must match.
    ds = make_dataset(seed=7)
    m = _manifest(ds)
    cfg = DropoutConfig(subject_rate=0.5, benchmark_rate=0.3)
    r1 = evaluate(lambda: LogisticModel(), ds, m, dropout=cfg, seed=0)
    r2 = evaluate(lambda: LogisticModel(), ds, m, dropout=cfg, seed=0)
    assert r1.nll == r2.nll


def test_column_coverage_passes_with_neutral_prefixes():
    ds = make_dataset(seed=8)
    # subject_key/benchmark are proxies (covered by the tree); the rest are neutral
    evaluate(lambda: LogisticModel(), ds, _manifest(ds),
             neutral_prefixes=["item_id", "sig0", "sig1"])


def test_column_coverage_raises_on_unclassified_column():
    import pytest
    ds = make_dataset(seed=8)
    with pytest.raises(AssertionError):
        evaluate(lambda: LogisticModel(), ds, _manifest(ds),
                 neutral_prefixes=["item_id"])  # sig0/sig1 unclassified


def test_build_oof_meta_is_nested_leakage_free():
    # The recursion guard: stacker training meta-features come from inner-OOF members
    # that never trained on that row's item -> memorizer scores 0.5 everywhere.
    ds = make_dataset(seed=5)
    meta = build_oof_meta(
        [lambda: MemorizerModel(key_col=0)],
        ds.X, ds.y, ds.item_keys, n_folds=3, seed=0, outer_index=0)
    assert meta.shape == (len(ds.y), 1)
    assert np.allclose(meta, 0.5)


def test_recursive_evaluate_returns_one_pred_per_row_and_beats_chance():
    ds = make_dataset(seed=6)
    res = recursive_evaluate(
        [lambda: LogisticModel(), lambda: LogisticModel()],
        lambda: LogisticModel(), ds, _manifest(ds))
    assert res.oof.shape == (len(ds.y),)
    assert not np.isnan(res.oof).any()
    assert res.nll < math.log(2)
