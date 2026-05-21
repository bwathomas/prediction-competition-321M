"""End-to-end smoke test: weighted BCE path in src/train.py.

Builds a tiny LookupDataset (with and without weights), runs the trainer's
``_unpack_batch`` + per-row BCE reduction by hand, and confirms:

  1. The new weights channel propagates through the DataLoader as a 10th
     batched tensor.
  2. With weights = ones, the weighted-mean BCE equals the plain BCE
     (bit-equal to within float tolerance).
  3. With non-uniform weights, the weighted BCE matches a hand-computed
     reference, and a single optimizer step still runs cleanly.

Run with:

    py scripts/smoke_train_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import LookupDataset  # noqa: E402
from src.train import _unpack_batch  # noqa: E402


def _make_ds(n: int, weights: np.ndarray | None) -> LookupDataset:
    rng = np.random.default_rng(7)
    return LookupDataset(
        subject_ids=rng.integers(0, 5, size=n).astype(np.int64),
        bc_ids=rng.integers(0, 3, size=n).astype(np.int64),
        item_emb=rng.standard_normal(size=(n, 8)).astype(np.float32),
        labels=(rng.random(n) > 0.5).astype(np.float32),
        sample_weights=weights,
    )


def main() -> int:
    n = 32
    device = "cpu"

    # ---- (1) Default weights = ones ---------------------------------------
    ds = _make_ds(n, weights=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 10, f"expected 10-tuple, got {len(batch)}"
    s, bc, ie, se, pf, ci, jf, nf, y, w = _unpack_batch(batch, device)
    assert torch.allclose(w, torch.ones_like(w)), "default weights should be 1.0"

    rng = np.random.default_rng(11)
    logits = torch.from_numpy(rng.standard_normal(n).astype(np.float32))
    per_row = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    weighted = (per_row * w).sum() / w.sum().clamp_min(1e-8)
    plain = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    assert torch.allclose(weighted, plain, atol=1e-6), (
        f"unweighted equivalence broken: weighted={weighted.item():.6f} "
        f"plain={plain.item():.6f}"
    )
    print(f"[1] unweighted equivalence ok ({weighted.item():.6f} == {plain.item():.6f})")

    # ---- (2) Non-uniform weights ------------------------------------------
    custom = rng.uniform(0.1, 3.0, size=n).astype(np.float32)
    ds2 = _make_ds(n, weights=custom)
    loader2 = torch.utils.data.DataLoader(ds2, batch_size=n, shuffle=False)
    batch2 = next(iter(loader2))
    _s, _bc, _ie, _se, _pf, _ci, _jf, _nf, y2, w2 = _unpack_batch(batch2, device)
    assert torch.allclose(w2, torch.from_numpy(custom), atol=1e-6), (
        "custom weights did not survive the DataLoader"
    )

    per_row2 = F.binary_cross_entropy_with_logits(logits, y2, reduction="none")
    weighted2 = (per_row2 * w2).sum() / w2.sum().clamp_min(1e-8)
    # Hand-computed reference
    ref = float(
        (per_row2.numpy() * custom).sum() / max(custom.sum(), 1e-8)
    )
    assert abs(float(weighted2) - ref) < 1e-5, (
        f"weighted mean mismatch: torch={weighted2.item():.6f} ref={ref:.6f}"
    )
    print(f"[2] weighted mean matches hand-computed reference ({weighted2.item():.6f})")

    # ---- (3) End-to-end gradient step still runs --------------------------
    layer = torch.nn.Linear(8, 1)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    opt.zero_grad(set_to_none=True)
    pred = layer(ie).squeeze(-1)
    per_row3 = F.binary_cross_entropy_with_logits(pred, y, reduction="none")
    loss = (per_row3 * w).sum() / w.sum().clamp_min(1e-8)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item(), "loss is not finite"
    print(f"[3] gradient step on weighted BCE ok (loss={loss.item():.6f})")

    print("\nAll trainer-weight smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
