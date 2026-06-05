from aide.hygiene.manifest import build_manifest, item_fold, assert_identical


def test_item_fold_is_deterministic_and_in_range():
    f1 = item_fold("benchA\nnone\nq1", n_folds=3, seed=0)
    f2 = item_fold("benchA\nnone\nq1", n_folds=3, seed=0)
    assert f1 == f2
    assert 0 <= f1 < 3


def test_build_manifest_assigns_every_unique_item_once():
    keys = ["a", "a", "b", "c", "c", "c"]
    m = build_manifest(keys, n_folds=3, seed=7)
    assert set(m.assignment) == {"a", "b", "c"}
    assert all(0 <= v < 3 for v in m.assignment.values())


def test_two_agents_same_seed_produce_identical_manifest():
    keys = ["a", "b", "c", "d", "e"]
    m_llama = build_manifest(keys, n_folds=3, seed=0)
    m_qwen = build_manifest(list(reversed(keys)), n_folds=3, seed=0)  # different order
    assert_identical(m_llama, m_qwen)  # must NOT raise — order-independent


def test_save_load_roundtrip(tmp_path):
    from aide.hygiene.manifest import SplitManifest
    m = build_manifest(["a", "b", "c"], n_folds=3, seed=1)
    p = tmp_path / "manifest.json"
    m.save(p)
    m2 = SplitManifest.load(p)
    assert_identical(m, m2)


def test_assert_identical_raises_on_mismatch():
    import pytest
    a = build_manifest(["x", "y"], n_folds=3, seed=0)
    b = build_manifest(["x", "y"], n_folds=3, seed=1)
    with pytest.raises(AssertionError):
        assert_identical(a, b)
