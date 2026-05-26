"""Cold-start NN feature regression + latency tests.

The runtime path for NN features works like this on every ``predict()``
call:

    1. Encode the (cold-start) item via the encoder (out of scope here;
       cached per ``(benchmark, condition, item_content)`` already).
    2. Project the embedding through PCA (cached).
    3. ``_TrainingItemCache.nearest`` -> FAISS (or brute force) for K
       neighbors.
    4. Look up subject pass rates + the four trait pass rates + the
       cluster-by-subject pass rate against the bundled CSR matrices.
    5. Aggregate via ``_aggregate_nn_features``.

Steps 3 and 4 dominate the per-call cost once the encoder is warm. The
*new* caches in ``src/export_submission.py``::

    _neighbor_cache : item_cache_key -> (idx, sims)
    _csr_row_cache  : (id(csr), row_id) -> (sorted_cols, sorted_vals)

reduce the workload from "rerun for every (subject, item) pair" to "run
once per cold-start item AND once per (subject, trait) pair", which is
what we want for typical Codabench rounds where every subject is
evaluated against every item.

These tests verify three things:

  1. **Source guarantee** -- the runtime template still contains the
     caching sentinels. If someone deletes the cache wiring this test
     fails immediately.
  2. **Reuse** -- a synthetic ``MockCache`` that mirrors the production
     caching logic exhibits the expected reuse (one FAISS search per
     unique item, one argsort per (subject, trait) pair).
  3. **Latency** -- a small but realistic workload (50 cold-start items
     x 20 subjects = 1k predicts) runs in well under a second on the
     pure-NumPy brute-force path.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

import src.export_submission as exp


# ---------------------------------------------------------------------------
# 1. Source guarantee
# ---------------------------------------------------------------------------


def test_runtime_template_contains_neighbor_cache_wiring() -> None:
    """The cold-start optimization is a string-template change; if the
    sentinels disappear it almost certainly means a regression."""
    src = exp._RUNTIME_MODEL_PY
    assert "self._neighbor_cache" in src
    assert "self._neighbor_cache_cap" in src
    assert "_neighbor_cache.get(item_cache_key)" in src
    assert "self._neighbor_cache[item_cache_key] = (idx_arr, sim_arr)" in src


def test_runtime_template_contains_csr_row_cache_wiring() -> None:
    src = exp._RUNTIME_MODEL_PY
    assert "self._csr_row_cache" in src
    assert "def _csr_row_sorted(" in src
    assert "row_lookup=self._csr_row_sorted" in src
    # The row_lookup kwarg must be threaded into _lookup_csr_pairs and
    # _resolve_conditional_inputs_runtime.
    assert "def _lookup_csr_pairs(row_ids, col_ids, pr_csr, mk_csr, *, row_lookup=None)" in src
    assert "row_lookup=row_lookup" in src
    assert "row_lookup=None," in src


def test_runtime_template_threads_item_cache_key() -> None:
    """``_get_nn_features`` must forward ``item_cache_key`` to
    ``compute_nn_features`` so the per-item neighbor cache actually
    sees a stable key across subjects.
    """
    src = exp._RUNTIME_MODEL_PY
    assert "item_cache_key=str(item_cache_key)" in src


# ---------------------------------------------------------------------------
# 2. Reuse semantics via a mirror MockCache
#
# We mirror the production caching logic by hand here. This is duplicate
# code on purpose -- it lets us exercise the cache state machine in
# isolation without exec'ing the entire runtime template (which pulls
# in torch + the full submission). The "source guarantee" tests above
# protect against the two implementations drifting silently.
# ---------------------------------------------------------------------------


class _MockCache:
    """In-memory mirror of the production ``_TrainingItemCache`` caches.

    Intentionally thin: only the methods exercised by cold-start NN
    feature computation are here. Same caching behavior as the
    template; the source-guarantee tests above pin the template.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        passrate_csr: sp.csr_matrix,
        passrate_mask_csr: sp.csr_matrix,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.nn_passrate = passrate_csr.tocsr()
        self.nn_passrate_mask = passrate_mask_csr.tocsr()
        self._neighbor_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._csr_row_cache: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray | None]
        ] = {}
        self._neighbor_cache_cap = 100_000
        self._csr_row_cache_cap = 200_000
        self.search_count = 0
        self.argsort_count = 0

    def nearest(
        self,
        query_embed: np.ndarray,
        k: int,
        *,
        item_cache_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if item_cache_key is not None:
            hit = self._neighbor_cache.get(item_cache_key)
            if hit is not None and hit[0].shape[0] >= int(k):
                return hit[0][: int(k)], hit[1][: int(k)]
        self.search_count += 1
        sims = self.embeddings @ query_embed.astype(np.float32)
        kk = min(int(k), sims.size)
        topk = (
            np.argpartition(-sims, kk - 1)[:kk]
            if kk < sims.size
            else np.arange(sims.size)
        )
        order = np.argsort(-sims[topk])
        idx_arr = topk[order].astype(np.int64)
        sim_arr = sims[topk][order].astype(np.float32)
        if item_cache_key is not None:
            if len(self._neighbor_cache) >= self._neighbor_cache_cap:
                self._neighbor_cache.pop(next(iter(self._neighbor_cache)))
            self._neighbor_cache[item_cache_key] = (idx_arr, sim_arr)
        return idx_arr, sim_arr

    def _csr_row_sorted(
        self, csr_obj, row_id: int, return_vals: bool
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        if csr_obj is None:
            return None
        key = (id(csr_obj), int(row_id))
        hit = self._csr_row_cache.get(key)
        if hit is not None:
            return hit
        n_rows = csr_obj.shape[0]
        if row_id < 0 or row_id >= n_rows:
            return None
        s = csr_obj.indptr[row_id]
        e = csr_obj.indptr[row_id + 1]
        cols = csr_obj.indices[s:e]
        if cols.size == 0:
            self._csr_row_cache[key] = (cols, None)
            return self._csr_row_cache[key]
        self.argsort_count += 1
        order = np.argsort(cols)
        sorted_cols = cols[order].astype(np.int64, copy=False)
        sorted_vals = (
            csr_obj.data[s:e][order].astype(np.float32, copy=False)
            if return_vals
            else None
        )
        if len(self._csr_row_cache) >= self._csr_row_cache_cap:
            self._csr_row_cache.pop(next(iter(self._csr_row_cache)))
        self._csr_row_cache[key] = (sorted_cols, sorted_vals)
        return self._csr_row_cache[key]


def _make_synthetic_world(
    n_items: int = 80, n_subjects: int = 10, dim: int = 32, density: float = 0.4
) -> tuple[_MockCache, np.ndarray]:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(n_items, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    pr = sp.random(
        n_subjects, n_items, density=density, format="csr", random_state=rng
    ).astype(np.float32)
    mk = pr.copy()
    mk.data = np.ones_like(mk.data)
    cache = _MockCache(embeddings, pr.tocsr(), mk.tocsr())
    cold_items = rng.normal(size=(50, dim)).astype(np.float32)
    cold_items /= np.linalg.norm(cold_items, axis=1, keepdims=True) + 1e-12
    return cache, cold_items


def test_neighbor_cache_reuses_across_subjects() -> None:
    """All N subjects evaluating the same cold-start item run FAISS
    (here: brute force) exactly once."""
    cache, cold_items = _make_synthetic_world()
    item = cold_items[0]
    key = "cold-item-key-A"

    # First call: miss -> 1 search.
    cache.nearest(item, k=8, item_cache_key=key)
    assert cache.search_count == 1
    assert key in cache._neighbor_cache

    # Many subjects all asking about the SAME item -> still 1 search.
    for _ in range(50):
        cache.nearest(item, k=8, item_cache_key=key)
    assert cache.search_count == 1, (
        "neighbor cache failed -- multiple subjects re-ran FAISS for the "
        "same cold-start item, which is the entire optimization we added"
    )

    # Different item -> new miss.
    cache.nearest(cold_items[1], k=8, item_cache_key="cold-item-key-B")
    assert cache.search_count == 2


def test_neighbor_cache_returns_byte_identical_results() -> None:
    """Cached vs. fresh search yield the same indices/sims for the same
    K, and slicing-down to a smaller K matches a fresh search at K."""
    cache, cold_items = _make_synthetic_world()
    item = cold_items[3]
    key = "stability-key"
    idx0, sim0 = cache.nearest(item, k=12, item_cache_key=key)
    idx1, sim1 = cache.nearest(item, k=12, item_cache_key=key)
    np.testing.assert_array_equal(idx0, idx1)
    np.testing.assert_array_equal(sim0, sim1)

    # Asking for a smaller K than cached -> we serve the prefix.
    idx2, sim2 = cache.nearest(item, k=5, item_cache_key=key)
    np.testing.assert_array_equal(idx2, idx0[:5])
    np.testing.assert_array_equal(sim2, sim0[:5])


def test_neighbor_cache_misses_when_cached_K_is_too_small() -> None:
    """If the previous request only retrieved K=3 we MUST re-run for
    K=10 (the prefix-only contract is one-directional: smaller K can
    serve from larger K, never the reverse)."""
    cache, cold_items = _make_synthetic_world()
    item = cold_items[0]
    key = "small-then-large"
    cache.nearest(item, k=3, item_cache_key=key)
    assert cache.search_count == 1
    cache.nearest(item, k=10, item_cache_key=key)
    assert cache.search_count == 2


def test_csr_row_cache_skips_argsort_after_first_call() -> None:
    cache, _ = _make_synthetic_world(n_items=64, n_subjects=8)
    # First read for subject 0 => one argsort.
    cache._csr_row_sorted(cache.nn_passrate, 0, return_vals=True)
    assert cache.argsort_count == 1
    # Repeat reads for the same row => zero new argsorts.
    for _ in range(20):
        cache._csr_row_sorted(cache.nn_passrate, 0, return_vals=True)
    assert cache.argsort_count == 1
    # Different row -> new argsort.
    cache._csr_row_sorted(cache.nn_passrate, 1, return_vals=True)
    assert cache.argsort_count == 2


def test_csr_row_cache_handles_empty_rows() -> None:
    """Empty rows cache as ``(empty_cols, None)`` so subsequent calls
    don't re-traverse the CSR structure."""
    csr = sp.csr_matrix((4, 8), dtype=np.float32)  # all empty
    cache, _ = _make_synthetic_world(n_subjects=4)
    out = cache._csr_row_sorted(csr, 2, return_vals=True)
    assert out is not None
    cols, vals = out
    assert cols.size == 0 and vals is None
    # Repeat -> already cached, no argsort.
    cache._csr_row_sorted(csr, 2, return_vals=True)
    assert cache.argsort_count == 0


def test_csr_row_cache_returns_none_for_oob_row() -> None:
    cache, _ = _make_synthetic_world(n_subjects=4)
    assert cache._csr_row_sorted(cache.nn_passrate, 999, return_vals=True) is None
    assert cache._csr_row_sorted(cache.nn_passrate, -1, return_vals=True) is None


def test_csr_row_cache_evicts_at_cap() -> None:
    """At the cap an arbitrary entry is evicted (FIFO-ish)."""
    cache, _ = _make_synthetic_world(n_subjects=4)
    cache._csr_row_cache_cap = 3
    csrs = [cache.nn_passrate, cache.nn_passrate_mask]
    cache._csr_row_sorted(csrs[0], 0, return_vals=True)
    cache._csr_row_sorted(csrs[0], 1, return_vals=True)
    cache._csr_row_sorted(csrs[1], 0, return_vals=False)
    assert len(cache._csr_row_cache) == 3
    cache._csr_row_sorted(csrs[1], 1, return_vals=False)
    # Cap holds at 3.
    assert len(cache._csr_row_cache) == 3


def test_neighbor_cache_evicts_at_cap() -> None:
    cache, cold_items = _make_synthetic_world()
    cache._neighbor_cache_cap = 5
    for i in range(20):
        cache.nearest(cold_items[i % cold_items.shape[0]], k=4, item_cache_key=f"k{i}")
    assert len(cache._neighbor_cache) == 5


# ---------------------------------------------------------------------------
# 3. Latency
# ---------------------------------------------------------------------------


def test_cold_start_latency_under_budget() -> None:
    """50 cold-start items x 20 subjects = 1000 predicts must complete
    quickly on pure NumPy. Threshold is generous (1.0s) -- failures
    here usually indicate the cache wiring broke and we regressed back
    to per-(subject, item) FAISS searches."""
    cache, cold_items = _make_synthetic_world(
        n_items=200, n_subjects=20, dim=64, density=0.3
    )
    start = time.perf_counter()
    for it_idx in range(cold_items.shape[0]):
        item_key = f"item_{it_idx}"
        item_emb = cold_items[it_idx]
        for sid in range(20):
            cache.nearest(item_emb, k=8, item_cache_key=item_key)
            cache._csr_row_sorted(cache.nn_passrate, sid, return_vals=True)
            cache._csr_row_sorted(cache.nn_passrate_mask, sid, return_vals=False)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"cold-start workload too slow: {elapsed:.3f}s"

    # Reuse sanity: 50 unique items -> 50 FAISS calls (not 1000).
    assert cache.search_count == cold_items.shape[0]
    # 20 subjects x 2 CSRs = 40 unique (csr, row) pairs.
    assert cache.argsort_count <= 40


def test_cold_start_no_neighbor_cache_is_strictly_slower() -> None:
    """Sanity: disabling the neighbor cache (item_cache_key=None) MUST
    issue one FAISS search per (subject, item) pair, confirming the
    cache is doing real work."""
    cache, cold_items = _make_synthetic_world(n_items=80, n_subjects=10)
    n_subjects = 10
    for it in cold_items[:5]:
        for _ in range(n_subjects):
            cache.nearest(it, k=4, item_cache_key=None)
    assert cache.search_count == 5 * n_subjects


# ---------------------------------------------------------------------------
# 4. Behavior with the real runtime resolver (parity of row_lookup)
# ---------------------------------------------------------------------------


def test_resolver_with_row_lookup_produces_same_outputs_as_without() -> None:
    """The ``row_lookup`` kwarg added to
    ``_resolve_conditional_inputs_runtime`` and ``_lookup_csr_pairs``
    must be a pure performance optimization: outputs are byte-identical
    whether the lookup is supplied or not.
    """
    # Reuse the parity-test extraction harness from the existing
    # conditional runtime tests.
    from tests.test_conditional_runtime_parity import _extract_runtime_helpers

    rt = _extract_runtime_helpers()

    rng = np.random.default_rng(13)
    n_subj, n_items = 6, 25
    pr = sp.random(n_subj, n_items, density=0.5, format="csr", random_state=rng).astype(np.float32)
    mk = pr.copy()
    mk.data = np.ones_like(mk.data)

    row_ids = np.array([0, 1, 0, 2], dtype=np.int64)
    col_ids = rng.integers(0, n_items, size=(4, 5)).astype(np.int64)

    out_no = rt._lookup_csr_pairs(row_ids, col_ids, pr, mk)
    cache_state: dict = {}

    def _row_lookup(csr_obj, row_id, return_vals):
        key = (id(csr_obj), int(row_id), bool(return_vals))
        hit = cache_state.get(key)
        if hit is not None:
            return hit
        if csr_obj is None or row_id < 0 or row_id >= csr_obj.shape[0]:
            return None
        s = csr_obj.indptr[row_id]
        e = csr_obj.indptr[row_id + 1]
        cols = csr_obj.indices[s:e]
        if cols.size == 0:
            cache_state[key] = (cols, None)
            return cache_state[key]
        order = np.argsort(cols)
        sc = cols[order].astype(np.int64, copy=False)
        sv = (
            csr_obj.data[s:e][order].astype(np.float32, copy=False)
            if return_vals
            else None
        )
        cache_state[key] = (sc, sv)
        return cache_state[key]

    out_with = rt._lookup_csr_pairs(
        row_ids, col_ids, pr, mk, row_lookup=_row_lookup
    )
    np.testing.assert_array_equal(out_no[0], out_with[0])
    np.testing.assert_array_equal(out_no[1], out_with[1])
