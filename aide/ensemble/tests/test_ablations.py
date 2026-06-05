import numpy as np
import pytest
from aide.ensemble.ablations import make_ablated_factory
from aide.harness.tests._toy import CaptureModel


def test_ablated_model_only_sees_kept_columns():
    cols = ["a", "b", "c"]
    X = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    y = np.array([0.0, 1.0])
    captured = []
    f = make_ablated_factory(lambda: CaptureModel(captured), cols, ["b", "c"])
    f().fit(X, y)
    assert captured[0].fit_X.shape == (2, 2)          # only 2 kept columns reach the model
    assert captured[0].fit_X[:, 0].tolist() == [10.0, 20.0]   # first kept col is 'b'
    assert captured[0].fit_X[:, 1].tolist() == [100.0, 200.0]  # second kept col is 'c'


def test_ablation_preserves_keep_column_order():
    cols = ["a", "b", "c"]
    X = np.array([[1.0, 10.0, 100.0]])
    captured = []
    f = make_ablated_factory(lambda: CaptureModel(captured), cols, ["c", "a"])
    f().fit(X, np.array([0.0]))
    assert captured[0].fit_X[0].tolist() == [100.0, 1.0]  # order = keep order, not col order


def test_missing_keep_column_raises():
    with pytest.raises(KeyError):
        make_ablated_factory(lambda: CaptureModel([]), ["a", "b"], ["a", "z"])
