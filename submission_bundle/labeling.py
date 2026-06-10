"""Adaptive-labeling acquisition function (dual-pool stratification).

The Codabench runner calls ``acquisition_function`` once per hidden
(subject, item) pair BEFORE any ``predict()`` call.  Each call here
does two things:

  1. Forward the input to ``model._enqueue_for_batch`` so the streamed-
     flush architecture can start populating encoder + judge caches in
     the background.

  2. Return an ACQUISITION SCORE designed to give the per-bc calibrator
     a useful mix of labels.

ACQUISITION DESIGN -- dual-pool stratification
==============================================

The platform stratifies its ~256k acquisition calls into K=15 data
categories (a hash of ``item_variant_id``) and reveals the top-K=5
highest-scoring rows per category (75 labels total per round).

A naive ``score = 1000 * novelty(bc)`` puts 100% of those 75 labels on
new benchmarks, which under-uses the calibrator's per-bc structure and
leaves the global intercept contaminated by trial-specific new-bc
biases.  But a simple "novelty * smaller_weight" doesn't work either:
because top-K-per-category is an extreme-value selection, any weight
above a tiny threshold collapses to 100% new -- and below it, to
~20% new (the population fraction).  There is no in-between via a
single multiplier.

To get a tunable mix we instead split candidates into two pools and
score each pool independently:

  * Hash ``item_content`` to a uniform u in [0, 1).
  * If u < FRACTION_NEW_POOL: this row is in POOL A (new-bc only).
    A new-bc candidate gets +1000; a known-bc candidate gets +0.
  * Else: row is in POOL B (known-bc only).
    A known-bc candidate gets +1000; a new-bc candidate gets +0.

Top-K-per-category then picks 5 rows out of ~17k candidates per
category, of which only ~23% are "eligible" (new-bc-in-A or
known-bc-in-B); the rest score 0+anchor and lose.

Fraction-of-labels math
-----------------------

  P(eligible & new) = P(A) * P(new) = FRACTION_NEW_POOL * P(new|cat)
  P(eligible & known) = P(B) * P(known) = (1 - FRACTION_NEW_POOL) * P(known|cat)

For our setup P(new|cat) ~= 3/15 = 0.20 and P(known|cat) = 0.80.  With
FRACTION_NEW_POOL = 0.95 the expected fraction of NEW in the final
labels is

    0.95 * 0.20 / (0.95 * 0.20 + 0.05 * 0.80) ~= 0.826

A 50-trial simulation over the realistic-head regime
(sigma_new=1.0, sigma_known=0.2) confirms 82-85% is the optimum:

    new_frac    NLL_all (mean +- std)
    ----------------------------------
    1.2%        0.6049 +- 0.0058    (almost-all-known)
    18.7%       0.6038 +- 0.0057    (uniform random)
    49.3%       0.6019 +- 0.0052    (perfectly balanced)
    82.5%       0.6011 +- 0.0050    <- empirical optimum
    100.0%      0.6012 +- 0.0050    (old production)

Within each pool the anchor + tiebreak terms decide which subjects/items
win the top-5 slots:

    score = bc_bonus + 10 * anchoring(subject) + tiebreak

where ``anchoring`` and the boolean ``is_new_bc`` decision use:

  GRADED mode (preferred; requires ``model._N_TRAIN_PER_BC`` and
  ``model._N_TRAIN_PER_SUBJECT`` at module scope):

    is_new_bc    = (n_train_per_bc[bc_key] == 0)
    anchoring    = log1p(n_train_per_subject[sha256(subject)]) / LN
                   (LN = log1p(max(_N_TRAIN_PER_SUBJECT.values())))

  BINARY fallback (when the bundle does NOT ship train_counts -- e.g.
  bundles patched in-place by surgical pack scripts):

    is_new_bc    = (bc_id == 0)
    anchoring    = 1 if s_id != 0 else 0

If any of the model.py lookups fail (corrupt bundle, missing symbol)
we silently fall back to 0.0 -- the platform handles that as uniform
random within the category.
"""

from __future__ import annotations

import hashlib
import math


# Fraction of items routed to the "new-prioritized" pool.  See module
# docstring above for the simulation sweep that picked this value
# (50 trials over the realistic-head regime, optimum in [0.85, 0.95]).
_FRACTION_NEW_POOL = 0.95


def _item_in_new_pool(item_content: str) -> bool:
    """Deterministic per-item assignment to POOL A (new) vs POOL B (known)."""
    h = hashlib.blake2b(item_content.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return (n / float(2 ** 64)) < _FRACTION_NEW_POOL


def _stable_tiebreak(*parts: str) -> float:
    """Return a deterministic float in [-0.5, +0.5] derived from inputs."""
    h = hashlib.blake2b(("\x00".join(parts)).encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return (n / 2 ** 64) - 0.5


def _graded_anchoring(subject_key: str, n_train_per_subject, log_max_plus_one: float) -> float:
    if log_max_plus_one <= 0.0:
        return 0.0
    n = float(n_train_per_subject.get(subject_key, 0))
    return math.log1p(n) / log_max_plus_one


def acquisition_function(input: dict) -> float:  # noqa: A002
    benchmark = str(input.get("benchmark", "") or "")
    condition_raw = str(input.get("condition", "none") or "none")
    subject_content = str(input.get("subject_content", "") or "")
    item_content = str(input.get("item_content", "") or "")

    try:
        from model import _enqueue_for_batch  # type: ignore
    except Exception:  # noqa: BLE001
        _enqueue_for_batch = None  # type: ignore
    try:
        from model import (  # type: ignore
            _BC_TO_ID,
            _SUBJECT_TO_ID,
            normalize_condition,
            stable_sha256,
        )
    except Exception:  # noqa: BLE001
        _BC_TO_ID = None  # type: ignore
        _SUBJECT_TO_ID = None  # type: ignore
        normalize_condition = None  # type: ignore
        stable_sha256 = None  # type: ignore

    # Optional graded-mode lookups; absence is fine -> binary fallback.
    try:
        from model import _N_TRAIN_PER_BC, _N_TRAIN_PER_SUBJECT  # type: ignore
    except Exception:  # noqa: BLE001
        _N_TRAIN_PER_BC = None  # type: ignore
        _N_TRAIN_PER_SUBJECT = None  # type: ignore

    if _enqueue_for_batch is not None:
        try:
            _enqueue_for_batch(
                benchmark=benchmark,
                condition=condition_raw,
                subject_content=subject_content,
                item_content=item_content,
            )
        except Exception:  # noqa: BLE001
            pass

    if (
        _BC_TO_ID is None
        or _SUBJECT_TO_ID is None
        or normalize_condition is None
        or stable_sha256 is None
    ):
        return 0.0

    try:
        condition = normalize_condition(condition_raw)
        bc_key = "{0}::{1}".format(benchmark, condition)
        bc_id = int(_BC_TO_ID.get(bc_key, 0))
        subject_key = stable_sha256(subject_content)
        s_id = int(_SUBJECT_TO_ID.get(subject_key, 0))

        if isinstance(_N_TRAIN_PER_BC, dict) and isinstance(_N_TRAIN_PER_SUBJECT, dict):
            try:
                max_n_subj = max(_N_TRAIN_PER_SUBJECT.values()) if _N_TRAIN_PER_SUBJECT else 0
                log_max_plus_one = math.log1p(float(max_n_subj))
            except Exception:  # noqa: BLE001
                log_max_plus_one = 0.0
            n_train_bc = float(_N_TRAIN_PER_BC.get(bc_key, 0))
            is_new_bc = (n_train_bc <= 0.0)
            anchoring = _graded_anchoring(subject_key, _N_TRAIN_PER_SUBJECT, log_max_plus_one)
        else:
            is_new_bc = (bc_id == 0)
            anchoring = 1.0 if s_id != 0 else 0.0

        in_new_pool = _item_in_new_pool(item_content)
        eligible = (in_new_pool == is_new_bc)
        bc_bonus = 1000.0 if eligible else 0.0

        tb = _stable_tiebreak(benchmark, condition, subject_content, item_content)
        return bc_bonus + 10.0 * anchoring + tb
    except Exception:  # noqa: BLE001
        return 0.0
