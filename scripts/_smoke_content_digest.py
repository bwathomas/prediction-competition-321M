"""Smoke test for the per-array content digest used in the
notebook cache keys for M2 / FwFM. Verifies (a) determinism,
(b) catches single-cell modifications anywhere in the array,
(c) sensitive to dtype/shape/order, (d) fast on the actual
working size (5M rows x 30 cols float32)."""
from __future__ import annotations

import hashlib
import time

import numpy as np


def _content_digest(*arrays, k_rows: int = 4096) -> str:
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
        if ac.dtype.kind == "f":
            agg = np.asarray(
                [
                    ac.sum(dtype=np.float64),
                    (ac.astype(np.float64) ** 2).sum(),
                    np.abs(ac.astype(np.float64)).sum(),
                    float(ac.min()),
                    float(ac.max()),
                ],
                dtype=np.float64,
            )
        else:
            ac64 = ac.astype(np.int64, copy=False)
            agg = np.asarray(
                [
                    int(ac64.sum()),
                    int((ac64 * ac64).sum()),
                    int(np.abs(ac64).sum()),
                    int(ac.min()),
                    int(ac.max()),
                ],
                dtype=np.int64,
            )
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

    print("all checks passed")


if __name__ == "__main__":
    main()
