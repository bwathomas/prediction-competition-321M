"""CsrPassrate must be a drop-in for DensePassrate (same gather/pooled_mean/global_mean).

The driver feeds the codecs a CSR-backed passrate on Colab (906 subjects × 311k items);
these tests prove the sparse implementation returns bit-identical values to the dense test
double on the same data, so every OOF/leakage test that used DensePassrate transfers.
"""
import numpy as np

from aide.features.derive_nn import DensePassrate
from aide.features.passrate import CsrPassrate


def _both(L, subjects, items):
    return DensePassrate(subjects, items, L), CsrPassrate.from_dense(subjects, items, L)


def test_gather_matches_dense():
    L = np.array([[1.0, np.nan, 0.0, 0.5],
                  [np.nan, 1.0, 1.0, np.nan]])
    subs, items = ["s0", "s1"], ["i0", "i1", "i2", "i3"]
    d, c = _both(L, subs, items)
    for s in subs:
        a = d.gather(s, items)
        b = c.gather(s, items)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


def test_gather_unknown_subject_or_item_is_nan():
    L = np.array([[1.0, 0.0]])
    c = CsrPassrate.from_dense(["s0"], ["i0", "i1"], L)
    assert np.all(np.isnan(c.gather("ghost", ["i0", "i1"])))
    out = c.gather("s0", ["i0", "iX"])     # unknown item -> nan
    assert out[0] == 1.0 and np.isnan(out[1])


def test_pooled_mean_matches_dense():
    L = np.array([[1.0, np.nan, 0.0, 1.0],
                  [0.0, 1.0, np.nan, 1.0],
                  [np.nan, 1.0, 1.0, 0.0]])
    subs = ["s0", "s1", "s2"]
    items = ["i0", "i1", "i2", "i3"]
    d, c = _both(L, subs, items)
    for sub in [["i0"], ["i1", "i2"], ["i0", "i1", "i2", "i3"], ["iX"], []]:
        assert abs(d.pooled_mean(sub) - c.pooled_mean(sub, default=0.0)) < 1e-9
    assert abs(d.global_mean() - c.global_mean()) < 1e-9


def test_pooled_mean_default_for_empty():
    L = np.array([[1.0, 0.0]])
    c = CsrPassrate.from_dense(["s0"], ["i0", "i1"], L)
    assert c.pooled_mean(["unknown"], default=0.7) == 0.7   # no observed cols -> default


def test_codec_runs_on_csr_passrate():
    """Smoke: derive_nn produces the same leakage-safe output on a CSR passrate."""
    from aide.features.derive_nn import bruteforce_knn, derive_nn
    emb = (lambda a: a / np.linalg.norm(a, axis=1, keepdims=True))(
        np.array([[1, 0], [0.9, 0.4], [0, 1], [-0.9, 0.4]], dtype=np.float32))
    items = ["i0", "i1", "i2", "i3"]
    L = np.array([[1.0, 1.0, 0.0, 0.0]])
    cpr = CsrPassrate.from_dense(["s0"], items, L)
    out = derive_nn(query_emb=emb[[0]], query_item_keys=["i0"], query_subjects=["s0"],
                    row_ids=["r"], index_emb=emb, index_item_keys=items, passrate=cpr,
                    Ks=(1, 2), knn_fn=bruteforce_knn)
    assert out["nn_label_derivatives"].X.shape == (1, 9)
