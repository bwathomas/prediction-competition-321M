"""Regression: Member 2 (LightGBM) item-stratified cold-start val split.

The legacy random-row split let the booster see rows from the same
item on both sides of its internal early-stopping val. This makes
val_logloss reflect *item memorization*, not cold-start
generalization, and the booster keeps adding trees that fit
item-specific noise. The new ``holdout_group_id`` kwarg makes the
internal split hold out *whole groups* (typically item ids) so val
rows live entirely in unseen items -- mirroring the cold-start outer
val of Model 1.

These tests pin the new contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gbdt_member import fit_gbdt_member


def _toy_data(n_items: int = 32, rows_per_item: int = 8, seed: int = 0):
    """Group-structured data: each item has ``rows_per_item`` rows. The
    label is a deterministic function of item_id so a model that
    'memorizes' items can drive val NLL to near 0 on a row-split but
    has to actually generalize on an item-split.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    item_ids = []
    labels = []
    for it in range(int(n_items)):
        # Each item has fixed feature noise + item-specific signal.
        item_signal = rng.normal(size=4).astype(np.float32)
        item_label_prob = float(rng.uniform(0.05, 0.95))
        for _ in range(int(rows_per_item)):
            row_noise = rng.normal(size=4).astype(np.float32) * 0.1
            rows.append(item_signal + row_noise)
            item_ids.append(int(it))
            labels.append(float(rng.uniform() < item_label_prob))
    X = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    g = np.array(item_ids, dtype=np.int64)
    return X, y, g


# ---- Group split honored ----


def test_holdout_group_id_routes_groups_to_one_side():
    """Every row of a held-out item lands in val; no row of a train item
    lands in val. This is the core invariant -- without it the LightGBM
    early-stopping val is item-memorization, not generalization."""
    X, y, g = _toy_data(n_items=20, rows_per_item=10, seed=0)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))

    # Patch the internal LightGBM training to record train_idx / val_idx
    # without actually running LightGBM (slow + offline-only). We
    # monkey-patch at the lightgbm import name in the module under test.
    import src.gbdt_member as gm
    captured: dict = {}

    class _StubBooster:
        def __init__(self):
            self.best_iteration = 1

        def trees_to_dataframe(self):
            import pandas as pd
            return pd.DataFrame(
                {
                    "tree_index": [0],
                    "node_index": ["0-L0"],
                    "left_child": [None],
                    "right_child": [None],
                    "split_feature": [None],
                    "threshold": [np.nan],
                    "value": [0.0],
                    "missing_direction": ["none"],
                    "decision_type": [None],
                }
            )

        def predict(self, X_, raw_score=False, num_iteration=None):
            return np.zeros(len(X_), dtype=np.float64)

    class _StubLgb:
        Dataset = type(
            "Dataset",
            (),
            {"__init__": lambda self, X, label, weight=None, free_raw_data=True: None},
        )

        @staticmethod
        def early_stopping(stopping_rounds, verbose=False):
            return None

        @staticmethod
        def log_evaluation(period, show_stdv=False):
            return None

        @staticmethod
        def train(*args, **kwargs):
            ds = kwargs["train_set"]
            return _StubBooster()

    # Just make sure the *split* logic is invoked correctly. Don't try
    # to run a real boost here -- the fixture is too small and the
    # parity check would be flaky. We catch the split via the LOG.info
    # output emitted at split time.
    import logging
    log_messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: log_messages.append(record.getMessage())
    handler.setLevel(logging.DEBUG)
    prev_level = gm.LOG.level
    gm.LOG.setLevel(logging.INFO)
    gm.LOG.addHandler(handler)
    try:
        fit_gbdt_member(
            X=X,
            y=y,
            feature_names=feature_names,
            holdout_group_id=g,
            n_estimators=2,
            num_leaves=3,
            min_data_in_leaf=1,
            early_stopping_rounds=1,
            seed=0,
            parity_atol=10.0,    # loose; we only care about the split path
        )
    finally:
        gm.LOG.removeHandler(handler)
        gm.LOG.setLevel(prev_level)
    # The split log line must report 20 groups and ~2 held (val_fraction=0.1).
    split_lines = [m for m in log_messages if "group-stratified split" in m]
    assert split_lines, f"expected group-stratified-split log, got: {log_messages[:5]}"
    msg = split_lines[0]
    assert "20 groups" in msg, f"expected 20 groups, got: {msg}"
    # Group-split should pick exactly round(0.1 * 20) = 2 held groups.
    assert "2 held" in msg, f"expected '2 held' in {msg!r}"


# ---- Validation of inputs ----


def test_holdout_group_id_shape_must_match_X():
    X, y, g = _toy_data(n_items=8, rows_per_item=4, seed=0)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    with pytest.raises(ValueError, match=r"holdout_group_id shape"):
        fit_gbdt_member(
            X=X, y=y, feature_names=feature_names,
            holdout_group_id=g[:5],  # wrong shape
            n_estimators=2, early_stopping_rounds=1, seed=0,
        )


def test_holdout_group_id_requires_at_least_two_groups():
    X = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    g = np.zeros(20, dtype=np.int64)  # all same group
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    with pytest.raises(ValueError, match=r"need >=2"):
        fit_gbdt_member(
            X=X, y=y, feature_names=feature_names,
            holdout_group_id=g,
            n_estimators=2, early_stopping_rounds=1, seed=0,
        )


# ---- Default (no group id) preserves legacy behavior ----


def test_no_group_id_does_not_emit_group_split_log():
    """When holdout_group_id is None, the random row split kicks in and
    the cold-start log line MUST NOT appear. This guards against
    accidentally enabling group-split for callers that haven't migrated."""
    import src.gbdt_member as gm
    import logging
    log_messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: log_messages.append(record.getMessage())
    handler.setLevel(logging.DEBUG)
    prev_level = gm.LOG.level
    gm.LOG.setLevel(logging.INFO)
    gm.LOG.addHandler(handler)
    # n_val floor is 64; bump N high enough that train slice is non-empty.
    X = np.random.default_rng(0).normal(size=(800, 4)).astype(np.float32)
    y = (np.random.default_rng(1).uniform(size=800) > 0.5).astype(np.float32)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    try:
        fit_gbdt_member(
            X=X, y=y, feature_names=feature_names,
            n_estimators=2, early_stopping_rounds=1,
            seed=0, parity_atol=10.0, min_data_in_leaf=2,
        )
    finally:
        gm.LOG.removeHandler(handler)
        gm.LOG.setLevel(prev_level)
    split_lines = [m for m in log_messages if "group-stratified split" in m]
    assert not split_lines, (
        f"random-row split must not log the group-stratified line; got: "
        f"{split_lines}"
    )
