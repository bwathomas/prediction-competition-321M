"""OOF-correctness tests for the tabular target-encoding codec.

The leakage-critical contract is that fold ``f``'s rows are encoded from the OTHER folds'
labels only. The math is exercised here in numpy (the oracle); polars is only the Colab
groupby accelerator over the same definition.
"""
import numpy as np

from aide.features.derive_tabular import derive_tabular, target_encode_oof


def test_oof_excludes_own_fold():
    keys = np.array(["a", "a", "a", "a"])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    folds = np.array([0, 0, 1, 1])
    enc = target_encode_oof(keys, y, folds, m=0.0)  # no smoothing
    # fold-0 rows see only fold-1 labels {0,0} -> 0.0; fold-1 rows see {1,1} -> 1.0
    assert enc[0] == 0.0 and enc[1] == 0.0
    assert enc[2] == 1.0 and enc[3] == 1.0


def test_changing_own_fold_labels_does_not_change_encoding():
    keys = np.array(["a", "a", "a", "a"])
    folds = np.array([0, 0, 1, 1])
    base = target_encode_oof(keys, np.array([1.0, 1.0, 0.0, 0.0]), folds, m=0.0)
    # flip fold-0's own labels; fold-0's encoding (from fold-1) must be unchanged
    flipped = target_encode_oof(keys, np.array([0.0, 0.0, 0.0, 0.0]), folds, m=0.0)
    assert base[0] == flipped[0] and base[1] == flipped[1]


def test_smoothing_pulls_toward_global_mean():
    keys = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    folds = np.array([0, 1, 0, 1])
    gm = float(np.mean(y))  # 0.5
    low = target_encode_oof(keys, y, folds, m=0.0)
    high = target_encode_oof(keys, y, folds, m=100.0)
    # 'a' fold-0 row: low encoding = fold-1 'a' label = 1.0; high ~ global 0.5
    i = 0
    assert abs(low[i] - 1.0) < 1e-9
    assert abs(high[i] - gm) < abs(low[i] - gm)


def test_unseen_key_falls_back_to_global():
    keys = np.array(["a", "b"])
    y = np.array([1.0, 0.0])
    folds = np.array([0, 1])
    enc = target_encode_oof(keys, y, folds, m=0.0)
    # 'a' only appears in fold 0; from other folds it is unseen -> global mean 0.5
    assert enc[0] == 0.5


def test_std_channel_zero_for_constant_group():
    keys = np.array(["a", "a", "a", "a"])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    folds = np.array([0, 0, 1, 1])
    std = target_encode_oof(keys, y, folds, m=0.0, stat="std")
    assert np.allclose(std, 0.0)


def test_nan_labels_are_ignored():
    keys = np.array(["a", "a", "a"])
    y = np.array([1.0, np.nan, 0.0])
    folds = np.array([0, 1, 1])
    enc = target_encode_oof(keys, y, folds, m=0.0)
    # fold-0 sees fold-1 observed labels {0.0} (nan ignored) -> 0.0
    assert enc[0] == 0.0


def test_derive_tabular_blocks_and_columns():
    n = 6
    folds = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    subject_keys = np.array(["s0", "s1", "s0", "s1", "s0", "s1"])
    subject_meta = {"organization": np.array(["o0", "o1", "o0", "o1", "o0", "o1"])}
    benchmark_meta = {"topic": np.array(["t0", "t0", "t1", "t1", "t0", "t1"])}
    parents = {"subject_mean": np.linspace(0, 1, n), "cluster_difficulty": np.linspace(1, 0, n),
               "geo__lid_estimate": np.ones(n), "nn__coverage_K8": np.full(n, 0.5)}
    out = derive_tabular(row_ids=[f"r{i}" for i in range(n)], fold_ids=folds, y=y,
                         subject_keys=subject_keys, subject_meta=subject_meta,
                         benchmark_meta=benchmark_meta, parents=parents)
    assert "groupby_subject_metadata" in out and "groupby_benchmark_metadata" in out
    assert "mean_encoded_subject" in out and "interactions_subject" in out
    assert all(c.startswith("grp_subj__") for c in out["groupby_subject_metadata"].columns)
    assert all(c.startswith("grp_bench__") for c in out["groupby_benchmark_metadata"].columns)
    assert all(c.startswith("m2_subj") for c in out["mean_encoded_subject"].columns)
    assert all(c.startswith(("int__", "ratio__")) for c in out["interactions_subject"].columns)
    assert any("organization_passrate_mean" in c for c in out["groupby_subject_metadata"].columns)
    assert any("organization_passrate_std" in c for c in out["groupby_subject_metadata"].columns)
    for blk in out.values():
        assert blk.X.shape[0] == n
        assert np.all(np.isfinite(blk.X))   # no nan/inf reaches a model


def test_two_smoothings_emitted_for_subject_encoding():
    n = 4
    out = derive_tabular(row_ids=[f"r{i}" for i in range(n)],
                         fold_ids=np.array([0, 1, 0, 1]), y=np.array([1.0, 0.0, 1.0, 0.0]),
                         subject_keys=np.array(["s0", "s0", "s1", "s1"]),
                         subject_meta={}, benchmark_meta={}, parents=None)
    cols = out["mean_encoded_subject"].columns
    assert len(cols) == 2 and len(set(cols)) == 2  # high/low smoothing, distinct columns
