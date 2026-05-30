"""Smoke test for the per-array content digest used in the
notebook cache keys for M2 / FwFM. Verifies (a) determinism,
(b) catches single-cell modifications anywhere in the array,
(c) sensitive to dtype/shape/order, (d) fast on the actual
working size (5M rows x 30 cols float32), and (e) is
peak-memory-bound for huge arrays (the original implementation
materialized full float64 copies via ``ac.astype(np.float64)``
which OOM'd Colab at the M4 hybrid scale of ~24 GB float32)."""
from __future__ import annotations

import hashlib
import time
import tracemalloc

import numpy as np


def _content_digest(*arrays, k_rows: int = 4096) -> str:
    """Chunked, peak-memory-bounded mirror of the notebook helper.

    Holds at most ``CHUNK_ROWS * F * 8 * 2`` bytes transient per
    chunk regardless of the total array size."""
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


def main() -> None:
    rng = np.random.default_rng(0)
    big = rng.standard_normal((5_000_000, 30)).astype(np.float32)

    t = time.time()
    base = _content_digest(big, k_rows=8192)
    print(f"5M x 30 hash: {time.time() - t:.3f}s digest={base}")

    assert base == _content_digest(big, k_rows=8192), "must be deterministic"

    for label, idx in [
        ("first flip", (0, 0)),
        ("middle flip", (2_500_000, 15)),
        ("last flip", (-1, -1)),
        ("offset-stride flip", (1_000_001, 7)),
    ]:
        m = big.copy()
        m[idx] = m[idx] + 1.0
        d = _content_digest(m, k_rows=8192)
        assert d != base, f"{label} not detected"
        print(f"  {label}: {d}  (changed OK)")

    tiny = big.copy()
    tiny[1_234_567, 11] += 1e-6
    d_tiny = _content_digest(tiny, k_rows=8192)
    assert d_tiny != base, "tiny 1e-6 change missed"
    print(f"  tiny 1e-6 flip: {d_tiny}  (changed OK)")

    a = rng.standard_normal((100, 5)).astype(np.float32)
    b = rng.standard_normal((100,)).astype(np.float32)
    assert _content_digest(a, b) != _content_digest(b, a), "arg order must matter"
    assert _content_digest(a.astype(np.float64)) != _content_digest(a), \
        "dtype must matter"
    assert _content_digest(a) != _content_digest(a.reshape(500, 1)), \
        "shape must matter"

    # Peak-memory bound: on a 1M x 100 float32 array (~400 MB
    # resident), the OLD path allocated two full f64 copies
    # (~1.6 GB extra) plus the squared array (another ~800 MB)
    # for total transient ~2.4 GB. The chunked path bounds peak
    # to roughly chunk_rows * F * 8 * 2 ~= 105 MB.
    M = np.ascontiguousarray(rng.standard_normal((1_000_000, 100)).astype(np.float32))
    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        _ = _content_digest(M, k_rows=4096)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak / 1e6
    nbytes_mb = M.nbytes / 1e6
    assert peak < 0.5 * M.nbytes, (
        f"peak ({peak_mb:.1f} MB) >= 0.5 * M.nbytes ({0.5 * nbytes_mb:.1f} MB) "
        f"-- chunked path is materializing a full f64 copy somewhere"
    )
    print(
        f"  peak on 1M x 100 float32: {peak_mb:.1f} MB "
        f"(M.nbytes={nbytes_mb:.1f} MB)  bound OK"
    )

    print("all checks passed")


if __name__ == "__main__":
    main()
