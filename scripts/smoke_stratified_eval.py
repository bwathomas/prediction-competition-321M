"""Smoke tests for category-stratified validation metrics.

Run from the repo root with::

    py scripts/smoke_stratified_eval.py

Exits non-zero on any failed assertion. Designed to be dependency-light
(numpy + pandas + torch + the new module only).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import LookupDataset  # noqa: E402
from src.semantic_categories import (  # noqa: E402
    CATEGORY_NAMES,
    N_CATEGORIES,
    stratified_eval_metrics,
    format_stratified_eval_report,
)
from src.train import EvalMetrics, evaluate_model  # noqa: E402

FAILURES: list[str] = []


def _check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok  {msg}")
    else:
        print(f"FAIL  {msg}")
        FAILURES.append(msg)


def test_stratified_metrics_basic() -> None:
    print("[1] stratified_eval_metrics: micro vs macro on toy data")
    # Two categories, 4 rows each. Category A is well-predicted (ll~=0),
    # category B is badly-predicted (ll high). The micro mean is closer
    # to A because... actually equal counts here, so micro == macro.
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.float64)
    p = np.array([0.9, 0.8, 0.1, 0.2,  0.4, 0.3, 0.6, 0.7], dtype=np.float64)
    cats = np.array([0, 0, 0, 0,  1, 1, 1, 1], dtype=np.int64)

    stats = stratified_eval_metrics(y, p, cats)
    _check("log_loss_micro" in stats, "result contains log_loss_micro")
    _check("log_loss_macro" in stats, "result contains log_loss_macro")
    _check(
        abs(stats["log_loss_micro"] - stats["log_loss_macro"]) < 1e-6,
        "micro == macro when categories have equal sizes "
        f"(micro={stats['log_loss_micro']:.5f} macro={stats['log_loss_macro']:.5f})",
    )
    _check(
        stats["n_categories_present"] == 2,
        f"n_categories_present == 2 (got {stats['n_categories_present']})",
    )
    _check(
        len(stats["per_category"]) == N_CATEGORIES,
        f"per_category contains all {N_CATEGORIES} names (got {len(stats['per_category'])})",
    )
    pc = stats["per_category"]
    _check(pc[CATEGORY_NAMES[0]]["n_rows"] == 4, "cat 0 has 4 rows")
    _check(pc[CATEGORY_NAMES[1]]["n_rows"] == 4, "cat 1 has 4 rows")
    _check(pc[CATEGORY_NAMES[2]]["n_rows"] == 0, "cat 2 has 0 rows")
    _check(pc[CATEGORY_NAMES[2]]["log_loss"] is None, "empty cat -> log_loss=None")


def test_macro_differs_from_micro_on_unbalanced_sizes() -> None:
    print("[2] macro differs from micro when category sizes differ")
    rng = np.random.default_rng(0)
    # 100 rows in cat 0 (easy: ll near 0), 5 rows in cat 14 (hard: ll near log(2)).
    y0 = (rng.random(100) > 0.5).astype(np.float64)
    p0 = np.where(y0 > 0.5, 0.95, 0.05)
    y1 = np.array([0, 1, 0, 1, 0], dtype=np.float64)
    p1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    y = np.concatenate([y0, y1])
    p = np.concatenate([p0, p1])
    cats = np.concatenate([np.zeros(100, dtype=np.int64), np.full(5, 14, dtype=np.int64)])

    stats = stratified_eval_metrics(y, p, cats)
    # Micro should be ~ll(easy) because 100/105 rows are easy.
    # Macro should be ~mean(ll(easy), ll(hard)) ~ (low + ~log(2))/2.
    _check(
        stats["log_loss_macro"] > stats["log_loss_micro"],
        f"macro ({stats['log_loss_macro']:.5f}) > micro "
        f"({stats['log_loss_micro']:.5f}) when the small bucket is much harder",
    )
    # Macro should be approximately the mean of the two per-cat losses.
    pc = stats["per_category"]
    expected_macro = 0.5 * (
        pc[CATEGORY_NAMES[0]]["log_loss"] + pc[CATEGORY_NAMES[14]]["log_loss"]
    )
    _check(
        abs(stats["log_loss_macro"] - expected_macro) < 1e-9,
        f"macro == mean of populated per-cat log_losses (diff={abs(stats['log_loss_macro']-expected_macro):.2e})",
    )


def test_auc_undefined_skipped() -> None:
    print("[3] per-category AUC handles all-positive / all-negative buckets")
    # Cat 0: mixed labels -> AUC defined. Cat 1: all-positive -> AUC None.
    y = np.array([0, 1, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    p = np.array([0.2, 0.8, 0.3, 0.7,  0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    cats = np.array([0, 0, 0, 0,  1, 1, 1, 1], dtype=np.int64)
    stats = stratified_eval_metrics(y, p, cats)
    pc = stats["per_category"]
    _check(
        pc[CATEGORY_NAMES[0]]["auc"] is not None,
        f"cat0 (mixed) AUC defined (got {pc[CATEGORY_NAMES[0]]['auc']})",
    )
    _check(
        pc[CATEGORY_NAMES[1]]["auc"] is None,
        "cat1 (all-pos) AUC == None",
    )
    _check(
        stats["auc_macro"] is None or abs(
            stats["auc_macro"] - pc[CATEGORY_NAMES[0]]["auc"]
        ) < 1e-9,
        "macro AUC == defined-only mean",
    )


def test_shape_validation() -> None:
    print("[4] shape mismatch raises")
    y = np.zeros(10)
    p = np.zeros(10)
    cats = np.zeros(5, dtype=np.int64)
    try:
        stratified_eval_metrics(y, p, cats)
        ok = False
    except ValueError:
        ok = True
    _check(ok, "mismatched shapes raise ValueError")


def _make_eval_pieces(n: int = 64):
    """Build a tiny linear model + LookupDataset for end-to-end checks."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    item_dim = 8
    ds = LookupDataset(
        subject_ids=rng.integers(0, 4, size=n).astype(np.int64),
        bc_ids=rng.integers(0, 2, size=n).astype(np.int64),
        item_emb=rng.standard_normal(size=(n, item_dim)).astype(np.float32),
        labels=(rng.random(n) > 0.5).astype(np.float32),
    )

    class _ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(item_dim, 1)

        def forward(self, s, bc, ie, se=None, pf=None, ci=None, jf=None, nf=None):
            return self.lin(ie).squeeze(-1)

    return _ToyModel(), ds


def test_evaluate_model_backward_compatible() -> None:
    print("[5] evaluate_model: 5-tuple unpack still works")
    model, ds = _make_eval_pieces(n=32)
    metrics = evaluate_model(
        model, ds, device="cpu", batch_size=32, bf16=False, progress=False,
    )
    # Tuple unpack must still succeed (legacy callers).
    ll, brier, auc, p, y = metrics
    _check(isinstance(ll, float), f"log_loss is float (got {type(ll).__name__})")
    _check(isinstance(brier, float), "brier is float")
    _check(p.shape == y.shape == (32,), f"p/y shape == (32,) (got {p.shape}/{y.shape})")
    _check(metrics.log_loss_macro is None, "macro is None without category_ids")
    _check(metrics.per_category is None, "per_category is None without category_ids")


def test_evaluate_model_with_category_ids() -> None:
    print("[6] evaluate_model: macro metrics populated when category_ids supplied")
    n = 64
    model, ds = _make_eval_pieces(n=n)
    rng = np.random.default_rng(42)
    cats = rng.integers(0, N_CATEGORIES, size=n).astype(np.int64)
    metrics = evaluate_model(
        model, ds, device="cpu", batch_size=16, bf16=False, progress=False,
        val_category_ids=cats,
    )
    _check(metrics.log_loss_macro is not None, "macro log_loss populated")
    _check(metrics.brier_macro is not None, "macro brier populated")
    _check(metrics.per_category is not None, "per_category populated")
    _check(
        metrics.n_categories_present >= 1,
        f"n_categories_present >= 1 (got {metrics.n_categories_present})",
    )

    # Cross-check: macro from evaluate_model matches direct call.
    stats = stratified_eval_metrics(metrics.y, metrics.p, cats)
    _check(
        abs(stats["log_loss_macro"] - metrics.log_loss_macro) < 1e-9,
        f"evaluate_model macro matches stratified_eval_metrics "
        f"({stats['log_loss_macro']:.6f} vs {metrics.log_loss_macro:.6f})",
    )

    # to_log_dict should expose both micro and macro keys.
    log = metrics.to_log_dict()
    for k in ("val_log_loss", "val_brier", "val_log_loss_macro", "val_brier_macro"):
        _check(k in log, f"to_log_dict contains {k}")


def test_evaluate_model_length_mismatch_warns_not_crashes() -> None:
    print("[7] evaluate_model: bad-length category_ids warns and skips macro")
    model, ds = _make_eval_pieces(n=32)
    bad = np.zeros(10, dtype=np.int64)
    metrics = evaluate_model(
        model, ds, device="cpu", batch_size=32, bf16=False, progress=False,
        val_category_ids=bad,
    )
    _check(metrics.log_loss_macro is None, "macro silently skipped on bad length")
    _check(metrics.per_category is None, "per_category not populated on bad length")
    _check(metrics.log_loss == metrics.log_loss, "micro still finite")


def test_format_report_contains_all_categories() -> None:
    print("[8] format_stratified_eval_report includes all 15 categories")
    y = np.array([0, 1, 0, 1], dtype=np.float64)
    p = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
    cats = np.array([0, 1, 2, 3], dtype=np.int64)
    stats = stratified_eval_metrics(y, p, cats)
    txt = format_stratified_eval_report(stats, desc="test")
    for name in CATEGORY_NAMES:
        _check(name in txt, f"report contains {name}")


def main() -> int:
    for fn in [
        test_stratified_metrics_basic,
        test_macro_differs_from_micro_on_unbalanced_sizes,
        test_auc_undefined_skipped,
        test_shape_validation,
        test_evaluate_model_backward_compatible,
        test_evaluate_model_with_category_ids,
        test_evaluate_model_length_mismatch_warns_not_crashes,
        test_format_report_contains_all_categories,
    ]:
        try:
            fn()
        except Exception as exc:
            FAILURES.append(f"{fn.__name__} raised {exc!r}")
            print(f"FAIL  {fn.__name__}: {exc!r}")
            traceback.print_exc()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll stratified-eval smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
