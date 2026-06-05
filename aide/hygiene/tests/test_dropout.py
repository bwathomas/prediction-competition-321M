import numpy as np
from aide.hygiene.dropout import apply_proxy_dropout


def _toy():
    cols = ["subject_key", "meta:family", "feat:nn_passrate__mean",
            "benchmark", "condition", "item_emb__0"]
    X = np.ones((4, len(cols)), dtype=np.float32)
    subjects = ["s1", "s1", "s2", "s2"]
    benchmarks = ["b1", "b2", "b1", "b2"]
    return X, cols, subjects, benchmarks


def test_dropping_all_subjects_zeros_every_subject_proxy_column_for_all_rows():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(0)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=1.0, benchmark_rate=0.0)
    for c in ("subject_key", "meta:family", "feat:nn_passrate__mean"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 0.0), f"{c} must be fully masked"
    for c in ("benchmark", "condition", "item_emb__0"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 1.0)
    assert info["dropped_subjects"] == {"s1", "s2"}


def test_dropout_is_entity_consistent_all_rows_of_a_dropped_subject_masked():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(3)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=0.5, benchmark_rate=0.0)
    j = cols.index("subject_key")
    for i, s in enumerate(subjects):
        if s in info["dropped_subjects"]:
            assert Xd[i, j] == 0.0
        else:
            assert Xd[i, j] == 1.0


def test_benchmark_dropout_masks_condition_proxy():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(0)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=0.0, benchmark_rate=1.0)
    for c in ("benchmark", "condition"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 0.0)
    assert np.all(Xd[:, cols.index("subject_key")] == 1.0)


def test_input_is_not_mutated_in_place():
    X, cols, subjects, benchmarks = _toy()
    X0 = X.copy()
    apply_proxy_dropout(X, cols, subjects=subjects, benchmarks=benchmarks,
                        rng=np.random.default_rng(0), subject_rate=1.0, benchmark_rate=1.0)
    assert np.array_equal(X, X0)
