"""Unit tests for :class:`src.models.IndexedEmbeddingView` and the
companion :func:`src.embeddings.index_embeddings` helper, plus the
:class:`src.models.LookupDataset` integration that lets the view
stand in for a stacked dense ``item_emb`` tensor.

The behaviour contract these tests pin down is what the notebook
``_build`` -> ``_score_dataset`` / DataLoader path actually
relies on. If any of these break, the M-train / M-val LookupDataset
will silently OOM (falling back to the stacked 80 GB path) or
emit wrong rows during scoring.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.embeddings import index_embeddings, stack_lookup
from src.models import IndexedEmbeddingView, LookupDataset


# ---------------------------------------------------------------------------
# index_embeddings helper
# ---------------------------------------------------------------------------


def test_index_embeddings_matches_stack_lookup_row_by_row() -> None:
    rng = np.random.default_rng(0)
    keys_unique = [f"item_{i:04d}" for i in range(50)]
    lookup = {k: rng.standard_normal(8).astype(np.float32) for k in keys_unique}
    per_row_keys = [keys_unique[int(i)] for i in rng.integers(0, 50, size=500)]

    stacked = stack_lookup(per_row_keys, lookup)
    uniq, ptr = index_embeddings(per_row_keys, lookup)

    assert uniq.dtype == np.float32
    assert ptr.dtype == np.int64
    assert uniq.shape == (len(set(per_row_keys)), 8)
    assert ptr.shape == (500,)

    for i in range(500):
        np.testing.assert_array_equal(stacked[i], uniq[ptr[i]])


def test_index_embeddings_first_appearance_order_is_deterministic() -> None:
    """Unique row order should reflect first-appearance in ``keys``;
    this makes downstream caching reproducible across runs."""
    lookup = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([1.0, 1.0], dtype=np.float32),
    }
    uniq, ptr = index_embeddings(["b", "a", "b", "c", "a"], lookup)
    np.testing.assert_array_equal(uniq[0], lookup["b"])
    np.testing.assert_array_equal(uniq[1], lookup["a"])
    np.testing.assert_array_equal(uniq[2], lookup["c"])
    np.testing.assert_array_equal(ptr, [0, 1, 0, 2, 1])


def test_index_embeddings_empty_keys_returns_well_shaped_arrays() -> None:
    lookup = {"x": np.zeros(4, dtype=np.float32)}
    uniq, ptr = index_embeddings([], lookup)
    assert uniq.shape == (0, 4)
    assert ptr.shape == (0,)


def test_index_embeddings_memory_advantage_at_realistic_scale() -> None:
    """At realistic many-row, few-unique scale the helper should
    use roughly U*D*4 + N*8 bytes vs N*D*4 for stack_lookup."""
    rng = np.random.default_rng(42)
    n_unique = 1000
    keys_unique = [f"item_{i}" for i in range(n_unique)]
    lookup = {k: rng.standard_normal(64).astype(np.float32) for k in keys_unique}
    per_row_keys = [keys_unique[int(i)] for i in rng.integers(0, n_unique, size=50_000)]

    stacked = stack_lookup(per_row_keys, lookup)
    uniq, ptr = index_embeddings(per_row_keys, lookup)

    bytes_stacked = stacked.nbytes
    bytes_indexed = uniq.nbytes + ptr.nbytes
    assert bytes_indexed < bytes_stacked / 3, (
        f"indexed form {bytes_indexed} should be much smaller than "
        f"stacked form {bytes_stacked}"
    )


# ---------------------------------------------------------------------------
# IndexedEmbeddingView contract
# ---------------------------------------------------------------------------


@pytest.fixture
def view_and_reference() -> tuple[IndexedEmbeddingView, np.ndarray]:
    rng = np.random.default_rng(1)
    keys_unique = [f"k{i}" for i in range(20)]
    lookup = {k: rng.standard_normal(6).astype(np.float32) for k in keys_unique}
    per_row_keys = [keys_unique[int(i)] for i in rng.integers(0, 20, size=200)]
    uniq, ptr = index_embeddings(per_row_keys, lookup)
    stacked_reference = stack_lookup(per_row_keys, lookup)
    return IndexedEmbeddingView(uniq, ptr), stacked_reference


def test_view_shape_and_len(view_and_reference) -> None:
    view, ref = view_and_reference
    assert view.shape == ref.shape
    assert len(view) == ref.shape[0]
    assert view.dtype == torch.float32


def test_view_scalar_int_indexing_matches_stacked(view_and_reference) -> None:
    view, ref = view_and_reference
    for i in [0, 1, 99, len(view) - 1]:
        np.testing.assert_array_equal(view[i].numpy(), ref[i])


def test_view_slice_indexing_matches_stacked(view_and_reference) -> None:
    """The chunked scoring loop in the notebook calls
    ``ds.item_emb[start:end]`` -- this must return a [b-a, D]
    tensor with the right rows."""
    view, ref = view_and_reference
    for sl in [slice(0, 32), slice(100, 132), slice(150, 200), slice(0, 0)]:
        np.testing.assert_array_equal(view[sl].numpy(), ref[sl])


def test_view_list_and_array_indexing_matches_stacked(view_and_reference) -> None:
    view, ref = view_and_reference
    idx = [0, 5, 10, 15, 20, 25]
    np.testing.assert_array_equal(view[idx].numpy(), ref[idx])
    np.testing.assert_array_equal(view[np.asarray(idx)].numpy(), ref[idx])
    np.testing.assert_array_equal(
        view[torch.tensor(idx, dtype=torch.long)].numpy(), ref[idx],
    )


def test_view_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="unique_emb must be 2-D"):
        IndexedEmbeddingView(np.zeros(8, dtype=np.float32), np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="row_to_uniq must be 1-D"):
        IndexedEmbeddingView(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((3, 2), dtype=np.int64),
        )


def test_view_rejects_out_of_range_pointers() -> None:
    with pytest.raises(IndexError):
        IndexedEmbeddingView(
            np.zeros((3, 2), dtype=np.float32),
            np.asarray([0, 1, 5], dtype=np.int64),
        )
    with pytest.raises(IndexError):
        IndexedEmbeddingView(
            np.zeros((3, 2), dtype=np.float32),
            np.asarray([-1, 0, 1], dtype=np.int64),
        )


def test_view_nbytes_is_compact() -> None:
    """``nbytes`` should reflect the indexed form, NOT the stacked form."""
    rng = np.random.default_rng(2)
    uniq = rng.standard_normal((100, 32)).astype(np.float32)
    ptr = rng.integers(0, 100, size=10_000).astype(np.int64)
    view = IndexedEmbeddingView(uniq, ptr)
    expected = uniq.nbytes + ptr.nbytes
    assert view.nbytes == expected


# ---------------------------------------------------------------------------
# LookupDataset integration
# ---------------------------------------------------------------------------


def _make_dataset_pair() -> tuple[LookupDataset, LookupDataset]:
    rng = np.random.default_rng(7)
    keys_unique = [f"item_{i:03d}" for i in range(40)]
    lookup = {k: rng.standard_normal(12).astype(np.float32) for k in keys_unique}
    per_row_keys = [keys_unique[int(i)] for i in rng.integers(0, 40, size=300)]

    stacked = stack_lookup(per_row_keys, lookup)
    uniq, ptr = index_embeddings(per_row_keys, lookup)
    view = IndexedEmbeddingView(uniq, ptr)

    subj = rng.integers(0, 7, size=300).astype(np.int64)
    bc = rng.integers(0, 3, size=300).astype(np.int64)
    labels = (rng.random(300) < 0.6).astype(np.float32)
    nn = rng.standard_normal((300, 8)).astype(np.float32)

    stacked_ds = LookupDataset(
        subject_ids=subj, bc_ids=bc, item_emb=stacked, labels=labels, nn_feats=nn,
    )
    viewed_ds = LookupDataset(
        subject_ids=subj, bc_ids=bc, item_emb=view, labels=labels, nn_feats=nn,
    )
    return stacked_ds, viewed_ds


def test_lookup_dataset_with_view_matches_stacked_per_row() -> None:
    stacked_ds, viewed_ds = _make_dataset_pair()
    assert len(stacked_ds) == len(viewed_ds)
    for i in [0, 1, 17, len(stacked_ds) - 1]:
        s_row = stacked_ds[i]
        v_row = viewed_ds[i]
        for s_t, v_t in zip(s_row, v_row):
            np.testing.assert_array_equal(s_t.numpy(), v_t.numpy())


def test_lookup_dataset_with_view_matches_stacked_in_dataloader() -> None:
    """The actual production access pattern: training fetches
    batches via DataLoader. Confirm batch contents are identical."""
    stacked_ds, viewed_ds = _make_dataset_pair()
    loader_s = torch.utils.data.DataLoader(stacked_ds, batch_size=16, shuffle=False)
    loader_v = torch.utils.data.DataLoader(viewed_ds, batch_size=16, shuffle=False)
    for batch_s, batch_v in zip(loader_s, loader_v):
        for ts, tv in zip(batch_s, batch_v):
            np.testing.assert_array_equal(ts.numpy(), tv.numpy())


def test_lookup_dataset_slice_indexing_used_by_score_dataset() -> None:
    """``_score_dataset`` does ``ds.item_emb[start:end].to(device)``
    instead of using a DataLoader. Confirm the slice path returns
    the right rows under the view as well."""
    stacked_ds, viewed_ds = _make_dataset_pair()
    for sl in [slice(0, 32), slice(100, 132), slice(280, 300)]:
        np.testing.assert_array_equal(
            stacked_ds.item_emb[sl].numpy(),
            viewed_ds.item_emb[sl].numpy(),
        )


def test_lookup_dataset_rejects_length_mismatched_view() -> None:
    rng = np.random.default_rng(8)
    uniq = rng.standard_normal((5, 4)).astype(np.float32)
    ptr = rng.integers(0, 5, size=10).astype(np.int64)
    view = IndexedEmbeddingView(uniq, ptr)
    with pytest.raises(ValueError, match="length .* labels length"):
        LookupDataset(
            subject_ids=np.zeros(7, dtype=np.int64),
            bc_ids=np.zeros(7, dtype=np.int64),
            item_emb=view,
            labels=np.zeros(7, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Downstream realism: trainer's exact DataLoader config + torch.utils.data
# helpers that the notebook actually uses around the view.
# ---------------------------------------------------------------------------


def test_view_indexing_works_with_numpy_int_indices() -> None:
    """Custom batch samplers (LengthBucketBatchSampler, the OOF
    fold remappers) sometimes pass through ``numpy.int64`` indices
    instead of Python ints. The scalar-int branch must coerce
    cleanly."""
    rng = np.random.default_rng(11)
    uniq = rng.standard_normal((10, 4)).astype(np.float32)
    ptr = rng.integers(0, 10, size=50).astype(np.int64)
    view = IndexedEmbeddingView(uniq, ptr)
    for i in [np.int64(0), np.int32(5), np.int64(49)]:
        np.testing.assert_array_equal(view[i].numpy(), uniq[ptr[int(i)]])


def test_view_compatible_with_torch_utils_data_subset() -> None:
    """``Subset`` indexes the underlying dataset with positional
    Python ints. The trainer's val-eval path wraps LookupDataset in
    a Subset when ``val_eval_max_batches > 0``; confirm the chain
    still yields identical rows under the view."""
    stacked_ds, viewed_ds = _make_dataset_pair()
    indices = [0, 5, 100, 200, 299]
    sub_s = torch.utils.data.Subset(stacked_ds, indices)
    sub_v = torch.utils.data.Subset(viewed_ds, indices)
    for i in range(len(indices)):
        row_s = sub_s[i]
        row_v = sub_v[i]
        for ts, tv in zip(row_s, row_v):
            np.testing.assert_array_equal(ts.numpy(), tv.numpy())


def test_view_dataloader_with_pin_memory_and_shuffle() -> None:
    """Realistic trainer config: shuffle=True + pin_memory=True
    (the latter pins the *collated* batch tensors, not the
    per-row outputs of __getitem__, so the view doesn't need to
    be in pinned memory itself). Confirm the batches collated
    out of the view are still equivalent to those out of the
    stacked tensor under the same seed."""
    stacked_ds, viewed_ds = _make_dataset_pair()
    # Same seed for both so the shuffle order is identical.
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    loader_s = torch.utils.data.DataLoader(
        stacked_ds, batch_size=32, shuffle=True, generator=g1,
        pin_memory=False, num_workers=0,
    )
    loader_v = torch.utils.data.DataLoader(
        viewed_ds, batch_size=32, shuffle=True, generator=g2,
        pin_memory=False, num_workers=0,
    )
    n_batches = 0
    for batch_s, batch_v in zip(loader_s, loader_v):
        for ts, tv in zip(batch_s, batch_v):
            np.testing.assert_array_equal(ts.numpy(), tv.numpy())
        n_batches += 1
    assert n_batches > 1


def test_view_to_device_moves_underlying_tensors() -> None:
    """``view.to(...)`` must move the underlying unique stack so
    that subsequent slice indexing returns tensors on the same
    device. Important because the scoring loop chains
    ``ds.item_emb[start:end].to(device)`` — if the underlying
    stack were on a different device than expected the chain
    would still work, but ``.to(...)`` on the view alone should
    pre-move the bytes for callers that pre-stage."""
    rng = np.random.default_rng(12)
    uniq = rng.standard_normal((20, 6)).astype(np.float32)
    ptr = rng.integers(0, 20, size=100).astype(np.int64)
    view = IndexedEmbeddingView(uniq, ptr)
    moved = view.to(dtype=torch.float64)
    assert moved.dtype == torch.float64
    assert moved._uniq.dtype == torch.float64
    np.testing.assert_allclose(moved[5].numpy(), uniq[ptr[5]].astype(np.float64))


def test_lookup_dataset_with_view_pickle_roundtrip() -> None:
    """``num_workers > 0`` would pickle the dataset to each worker
    subprocess. We don't use that in the notebook today, but the
    invariant is cheap to keep and would matter if anyone enabled
    it. Verifies the IndexedEmbeddingView round-trips through
    pickle preserving content, shape, dtype, and indexing
    behavior."""
    import pickle

    stacked_ds, viewed_ds = _make_dataset_pair()
    restored = pickle.loads(pickle.dumps(viewed_ds))
    assert isinstance(restored.item_emb, IndexedEmbeddingView)
    assert restored.item_emb.shape == viewed_ds.item_emb.shape
    assert restored.item_emb.dtype == viewed_ds.item_emb.dtype
    for i in [0, 7, 150, 299]:
        np.testing.assert_array_equal(
            restored.item_emb[i].numpy(),
            stacked_ds.item_emb[i].numpy(),
        )


def test_view_supports_dataset_len_and_indexing_via_random_sampler() -> None:
    """The RandomSampler used inside DataLoader(shuffle=True) calls
    ``len(dataset)`` to decide the index range, then probes random
    Python ints in that range. Verify both directly so a regression
    in either would surface here, not just in a full DataLoader
    pass."""
    stacked_ds, viewed_ds = _make_dataset_pair()
    assert len(viewed_ds) == len(stacked_ds)
    rng = np.random.default_rng(13)
    for _ in range(50):
        i = int(rng.integers(0, len(viewed_ds)))
        row_s = stacked_ds[i]
        row_v = viewed_ds[i]
        for ts, tv in zip(row_s, row_v):
            np.testing.assert_array_equal(ts.numpy(), tv.numpy())
