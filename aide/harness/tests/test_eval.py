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
                dropout=DropoutConfig(subject_rate=1.0), rng=np.random.default_rng(0))
    j = ds.feature_columns.index("subject_key")
    assert captured  # models were fit
    for m in captured:
        assert np.all(m.fit_X[:, j] == 0.0)


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
