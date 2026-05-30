"""Tests for the cache-key digest helper used in the
``qwen8b_four_member_stacked`` notebook to fingerprint the
M2 + FwFM fit inputs.

The notebook is not directly importable (it pulls in heavy
deps and runs side-effecting initialization at module top),
so we instead extract a verbatim copy of the helper here and
test the *behavior contract* against it. Any divergence from
that contract is what we want to catch.

The contract is:

* deterministic for identical inputs.
* changes for *any* single-cell modification anywhere in the
  array, including indices that fall between the stride samples.
* changes when dtype, shape, or arg order changes.
* completes in well under 10 seconds on a (5_000_000, 30)
  float32 -- the working size of the M2 / FwFM fit inputs.

If you change the digest helper in the notebook, copy the new
implementation into ``_content_digest_under_test`` and re-run
this file. Failing tests mean the new helper is weaker; that
in turn means cached fit states could become stale silently
on the next run, so review carefully.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest


def _content_digest_under_test(*arrays, k_rows: int = 4096) -> str:
    """VERBATIM copy of the helper in
    ``notebooks/qwen8b_four_member_stacked.py``.

    Kept here so that the test file is self-contained: importing
    the notebook would trigger heavy initialization side effects
    that are unrelated to the digest contract.

    Chunked aggregates so peak memory stays bounded for huge
    arrays (the previous monolithic ``ac.astype(np.float64)``
    materialized two full-array float64 copies and was the host
    OOM trigger on ``X_train_dense_m4`` (~24 GB float32))."""
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"|None|")
            continue
        ac = np.ascontiguousarray(a)
        h.update(b"|dtype=")
        h.update(str(ac.dtype).encode("ascii"))
        h.update(b"|shape=")
        h.update(str(ac.shape).encode("ascii"))
        if ac.ndim == 0 or ac.size == 0:
            continue
        if ac.shape[0] <= int(k_rows):
            h.update(ac.tobytes())
        else:
            stride = max(int(ac.shape[0]) // int(k_rows), 1)
            h.update(ac[::stride].tobytes())
            h.update(ac[:64].tobytes())
            h.update(ac[-64:].tobytes())
        _CD_CHUNK = 65_536
        n_rows = int(ac.shape[0])
        if ac.dtype.kind == "f":
            s_sum = 0.0
            s_sq = 0.0
            s_abs = 0.0
            mn = float("inf")
            mx = float("-inf")
            for s in range(0, n_rows, _CD_CHUNK):
                e = min(s + _CD_CHUNK, n_rows)
                chunk = ac[s:e].astype(np.float64, copy=False)
                s_sum += float(chunk.sum())
                s_sq += float((chunk * chunk).sum())
                s_abs += float(np.abs(chunk).sum())
                mn = min(mn, float(chunk.min()))
                mx = max(mx, float(chunk.max()))
                chunk = None
            agg = np.asarray([s_sum, s_sq, s_abs, mn, mx], dtype=np.float64)
        else:
            i_sum = 0
            i_sq = 0
            i_abs = 0
            i_mn = int(ac.ravel()[0])
            i_mx = int(ac.ravel()[0])
            for s in range(0, n_rows, _CD_CHUNK):
                e = min(s + _CD_CHUNK, n_rows)
                chunk = ac[s:e]
                c64 = chunk.astype(np.int64, copy=False)
                i_sum += int(c64.sum())
                i_sq += int((c64 * c64).sum())
                i_abs += int(np.abs(c64).sum())
                i_mn = min(i_mn, int(chunk.min()))
                i_mx = max(i_mx, int(chunk.max()))
                chunk = None
                c64 = None
            agg = np.asarray([i_sum, i_sq, i_abs, i_mn, i_mx], dtype=np.int64)
        h.update(agg.tobytes())
    return h.hexdigest()


def test_digest_deterministic_for_same_input() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 30)).astype(np.float32)
    assert _content_digest_under_test(a) == _content_digest_under_test(a)


@pytest.mark.parametrize(
    "idx",
    [
        (0, 0),
        (-1, -1),
        (1_234_567, 11),
        (2_500_000, 15),
        (4_999_999, 29),
    ],
)
def test_digest_detects_any_single_cell_change(idx) -> None:
    """The whole point: even a 1e-6 modification at *any* row
    must propagate into the digest. The stride sample alone
    misses interior cells -- the per-array reduction is what
    plugs that hole."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5_000_000, 30)).astype(np.float32)
    base = _content_digest_under_test(a, k_rows=8192)
    mutated = a.copy()
    mutated[idx] = float(mutated[idx]) + 1e-6
    assert _content_digest_under_test(mutated, k_rows=8192) != base


def test_digest_detects_label_flip_on_int_array() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(1_000_000) < 0.7).astype(np.int64)
    base = _content_digest_under_test(y)
    flipped = y.copy()
    flipped[500_001] ^= 1
    assert _content_digest_under_test(flipped) != base


def test_digest_arg_order_matters() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((100, 5)).astype(np.float32)
    b = rng.standard_normal((100,)).astype(np.float32)
    assert _content_digest_under_test(a, b) != _content_digest_under_test(b, a)


def test_digest_dtype_matters() -> None:
    rng = np.random.default_rng(3)
    a32 = rng.standard_normal((50, 4)).astype(np.float32)
    a64 = a32.astype(np.float64)
    assert _content_digest_under_test(a32) != _content_digest_under_test(a64)


def test_digest_shape_matters() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal((200,)).astype(np.float32)
    assert _content_digest_under_test(a) != _content_digest_under_test(a.reshape(100, 2))


def test_digest_handles_none_argument() -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((10, 3)).astype(np.float32)
    d_with_none = _content_digest_under_test(a, None)
    d_without = _content_digest_under_test(a)
    assert d_with_none != d_without


def test_digest_runs_quickly_on_working_size() -> None:
    """Working size: 5M rows x 30 cols float32, the actual M2
    numerical input width. The helper is invoked at every M2 /
    FwFM cache lookup, so it must stay well below the cost of
    refitting the model."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal((5_000_000, 30)).astype(np.float32)
    t0 = time.time()
    _content_digest_under_test(a, k_rows=8192)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"digest too slow: {elapsed:.2f}s on working size"


def test_digest_peak_memory_bounded_for_huge_array() -> None:
    """Regression test for the host-OOM bug:
    ``ac.astype(np.float64)`` on the M4 hybrid matrix
    (~24 GB float32) allocated two ~48 GB float64 copies plus
    the squared array, killing the kernel before any model
    training started.

    The chunked implementation must hold peak transient memory
    well below ``M.nbytes``. We exercise a 1M x 100 float32
    array (~400 MB resident) and require peak < 0.5 * M.nbytes
    -- comfortable for the chunked path (~105 MB observed)
    and catches any reintroduction of a full f64 materialization
    (which would peak around 3.2 GB on this fixture)."""
    import tracemalloc

    rng = np.random.default_rng(7)
    M = np.ascontiguousarray(
        rng.standard_normal((1_000_000, 100)).astype(np.float32)
    )
    assert M.nbytes >= 100 * 1024 * 1024

    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        _content_digest_under_test(M, k_rows=4096)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    threshold = int(0.5 * M.nbytes)
    assert peak < threshold, (
        f"Peak Python-allocated memory during _content_digest "
        f"({peak / 1e6:.1f} MB) >= threshold "
        f"({threshold / 1e6:.1f} MB). This suggests the function "
        f"is materializing a full float64 copy again. "
        f"M.nbytes={M.nbytes / 1e6:.1f} MB."
    )
