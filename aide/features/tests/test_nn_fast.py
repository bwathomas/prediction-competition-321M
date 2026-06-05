"""nn_fast.derive_nn_labels_fast must equal derive_nn's label blocks in the production
regime (every row has >= maxK valid neighbours, so no short-neighbour padding)."""
import numpy as np

from aide.features.derive_nn import bruteforce_knn, derive_nn
from aide.features.nn_fast import derive_nn_labels_fast
from aide.features.passrate import CsrPassrate


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    items = [f"i{j}" for j in range(20)]
    emb = rng.normal(size=(20, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    subs = ["s0", "s1", "s2"]
    L = rng.integers(0, 2, size=(3, 20)).astype(float)
    L[rng.random((3, 20)) < 0.3] = np.nan          # ~30% unobserved
    return items, emb, subs, CsrPassrate.from_dense(subs, items, L)


def test_labels_match_codec_no_padding():
    items, emb, subs, pr = _fixture()
    q_items = ["i0", "i5", "i10", "i15", "i3"]
    q_subj = ["s0", "s1", "s2", "s0", "s1"]
    q_emb = emb[[items.index(i) for i in q_items]]
    rids = [f"{s}|{i}" for s, i in zip(q_subj, q_items)]
    common = dict(query_emb=q_emb, query_item_keys=q_items, query_subjects=q_subj,
                  row_ids=rids, index_emb=emb, index_item_keys=items, passrate=pr,
                  Ks=(2, 4), knn_fn=bruteforce_knn)
    codec = derive_nn(**common)
    fast = derive_nn_labels_fast(**common)
    for g in ("nn_label_derivatives", "counts_subject"):
        assert fast[g].columns == codec[g].columns, g
        assert np.allclose(fast[g].X, codec[g].X, atol=1e-5), g


def test_gather_pairs_matches_gather():
    items, emb, subs, pr = _fixture(1)
    # parallel-array gather equals per-call gather
    s_rows = np.array([pr.s_idx["s0"], pr.s_idx["s1"], pr.s_idx["s0"]])
    cols = np.array([pr.i_idx["i0"], pr.i_idx["i7"], pr.i_idx["i7"]])
    paired = pr.gather_pairs(s_rows, cols)
    expected = np.array([pr.gather("s0", ["i0"])[0], pr.gather("s1", ["i7"])[0],
                         pr.gather("s0", ["i7"])[0]])
    assert np.array_equal(np.isnan(paired), np.isnan(expected))
    assert np.allclose(paired[~np.isnan(paired)], expected[~np.isnan(expected)])


def test_gather_pairs_unknown_indices_are_nan():
    items, emb, subs, pr = _fixture(2)
    out = pr.gather_pairs(np.array([-1, 0]), np.array([0, -1]))
    assert np.all(np.isnan(out))
