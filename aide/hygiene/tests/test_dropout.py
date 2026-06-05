import numpy as np
from aide.hygiene.dropout import apply_proxy_dropout, choose_dropped, mask_dropped


def test_one_chosen_set_masks_train_and_test_consistently():
    # C1 foundation: a single dropped set applied to two disjoint row groups masks the
    # SAME entities in both — proving train/test can share one set.
    cols = ["subject_key", "benchmark"]
    rng = np.random.default_rng(0)
    dsubj, dbench = choose_dropped(subjects=["s1", "s2", "s3"], benchmarks=["b1"],
                                   subject_rate=0.5, benchmark_rate=0.0, rng=rng)
    subj_tr, subj_te = ["s1", "s2", "s3"], ["s1", "s3", "s2"]
    Xtr, _ = mask_dropped(np.ones((3, 2), np.float32), cols, subjects=subj_tr,
                          benchmarks=["b1"] * 3, dropped_subjects=dsubj, dropped_benchmarks=dbench)
    Xte, _ = mask_dropped(np.ones((3, 2), np.float32), cols, subjects=subj_te,
                          benchmarks=["b1"] * 3, dropped_subjects=dsubj, dropped_benchmarks=dbench)
    j = cols.index("subject_key")
    for X, subs in [(Xtr, subj_tr), (Xte, subj_te)]:
        for i, s in enumerate(subs):
            assert X[i, j] == (0.0 if s in dsubj else 1.0)


def _toy():
    # real catalog column names
    cols = ["subject_key", "subj_cat__family__001", "nn__passrate_mean",
            "benchmark", "condition", "nn__passrate_benchmark_conditional", "item_emb__0"]
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
    for c in ("subject_key", "subj_cat__family__001", "nn__passrate_mean",
              "nn__passrate_benchmark_conditional"):  # nn is a subject proxy
        assert np.all(Xd[:, cols.index(c)] == 0.0), c
    for c in ("benchmark", "condition", "item_emb__0"):  # not subject proxies
        assert np.all(Xd[:, cols.index(c)] == 1.0), c
    assert set(info["dropped_subjects"]) == {"s1", "s2"}


def test_benchmark_dropout_masks_condition_and_only_the_conditional_nn_cell():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(0)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=0.0, benchmark_rate=1.0)
    for c in ("benchmark", "condition", "nn__passrate_benchmark_conditional"):
        assert np.all(Xd[:, cols.index(c)] == 0.0), c
    # subject-only nn + subject metadata untouched under benchmark dropout
    assert np.all(Xd[:, cols.index("nn__passrate_mean")] == 1.0)
    assert np.all(Xd[:, cols.index("subject_key")] == 1.0)


def test_duplicate_columns_rejected():
    import pytest
    X = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_proxy_dropout(X, ["benchmark", "benchmark"], subjects=["s", "s"],
                            benchmarks=["b", "b"], rng=np.random.default_rng(0),
                            subject_rate=1.0, benchmark_rate=1.0)


def test_input_is_not_mutated_in_place():
    X, cols, subjects, benchmarks = _toy()
    X0 = X.copy()
    apply_proxy_dropout(X, cols, subjects=subjects, benchmarks=benchmarks,
                        rng=np.random.default_rng(0), subject_rate=1.0, benchmark_rate=1.0)
    assert np.array_equal(X, X0)
