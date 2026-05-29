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
