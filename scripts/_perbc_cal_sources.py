"""Shared sources for the per-benchmark calibration + novelty-labeling patch.

Two pack scripts use this module to upgrade an existing submission bundle:

  - ``pack_streamed_encoder_nn_perbc_cal.py``
  - ``pack_item_sample_perbc_cal.py``

The transformation applied to any compatible bundle is:

  1. Replace the calibrator banner-to-next-banner region in ``model.py``
     with a hierarchical, held-out-gated per-benchmark calibrator.
  2. Replace the single ``p = _CALIBRATOR.apply(p)`` line inside
     ``predict()`` with ``p = _CALIBRATOR.apply(p, bc_key)`` where
     ``bc_key`` is the raw ``"benchmark::condition"`` string.
  3. Replace ``labeling.py`` with a benchmark-novelty + anchor-model
     acquisition (1000 * novelty + 10 * anchoring + tiebreak).

A "compatible bundle" is any submission whose ``model.py``:
  * has a banner of the form ``# Calibrator (...)`` that we can find,
  * defines ``_BC_TO_ID``, ``_SUBJECT_TO_ID``, ``normalize_condition``,
    ``stable_sha256`` at module scope BEFORE ``predict()``,
  * has a ``predict()`` function containing exactly one
    ``p = _CALIBRATOR.apply(p)`` line.

All three current per-bc-cal targets (streamed_encoder_nn, item_sample,
k_factor) satisfy these constraints.
"""

from __future__ import annotations

import ast
import re

# ---------------------------------------------------------------------------
# Calibrator (per-benchmark held-out-gated intercept w/ ridge + identity)
# ---------------------------------------------------------------------------

NEW_CALIBRATOR_BLOCK = '''# ---------------------------------------------------------------------------
# Calibrator (per-benchmark partial-pool intercept; ridge-shrunk to global)
# ---------------------------------------------------------------------------
#
# Design notes:
#
#   * One-parameter calibrator per benchmark: an intercept ``b`` on the
#     logit scale.  ``logit(p_cal) = logit(p) + b``.  With N <= 75 labels
#     per round (~5 per benchmark on average) anything richer overfits.
#
#   * PARTIAL POOLING instead of an accept-or-reject gate.  We fit:
#
#       b_global  = argmin_b sum_all BCE(logit(p)+b, y)
#                              + RIDGE_LAMBDA_GLOBAL * b**2
#
#       b_bc[k]   = argmin_b sum_bc=k BCE(logit(p)+b, y)
#                              + RIDGE_LAMBDA_BC * (b - b_global)**2
#
#     The ridge toward ``b_global`` gives continuous shrinkage:
#     benchmarks with many labels move toward their own ``b``;
#     benchmarks with few labels inherit ``b_global``; benchmarks with
#     zero labels just use ``b_global`` at apply time.  No discrete gate
#     to flip; the noise floor that used to push a gated fit back to
#     identity is replaced by a smooth Bayesian-style shrinkage.
#
#   * RIDGE_LAMBDA = 20 was picked from a 50-trial simulation sweep
#     over {2, 5, 10, 15, 20, 25, 30, 40, 60, 80} on the realistic-head
#     regime (sigma_new = 1.0, sigma_known = 0.2).  Lambda in [15, 25]
#     are tied for lowest mean NLL; 20 also has lowest variance, so we
#     pick the safer end of the plateau.  Interpretation: lambda = 20
#     is ~equivalent to twenty fake observations pinned at the prior,
#     which is the right strength when typical per-bc N is in the
#     single digits.
#
#   * Per-bc routing uses the raw ``bc_key`` string
#     ("benchmark::condition") -- NOT the embedding table's bc_id,
#     which would collapse every unseen benchmark to 0 and merge them
#     all into a single bucket (the opposite of what we want for
#     new-benchmark calibration).


_RIDGE_LAMBDA_GLOBAL = 20.0
_RIDGE_LAMBDA_BC = 20.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _fit_intercept_ridge(
    ps,
    ys,
    *,
    target_b: float = 0.0,
    ridge: float = _RIDGE_LAMBDA_GLOBAL,
) -> float:
    """Return ``b`` minimizing sum BCE(logit(p)+b, y) + ridge * (b - target_b)**2."""
    if not ps:
        return float(target_b)
    zs = [_logit(p) for p in ps]
    b = float(target_b)
    for _ in range(80):
        g = 2.0 * ridge * (b - target_b)
        h = 2.0 * ridge
        for z, y in zip(zs, ys):
            q = _sigmoid(z + b)
            g += q - y
            h += q * (1.0 - q)
        if h < 1e-9:
            break
        step = g / h
        new_b = b - step
        if not math.isfinite(new_b):
            break
        if abs(new_b - b) < 1e-8:
            b = new_b
            break
        b = new_b
    if not math.isfinite(b):
        return float(target_b)
    return max(-5.0, min(5.0, float(b)))


class _Calibrator:
    """Hierarchical per-benchmark intercept with continuous ridge shrinkage.

    Routing at apply time:
        ``per_bc[bc_key]`` if present, else ``b_global``.
    ``bc_key`` is the raw "benchmark::condition" string so each unseen
    benchmark gets its own slot (we never collapse new benchmarks to a
    shared bucket).
    """

    def __init__(self, state: dict | None = None) -> None:
        if isinstance(state, dict) and state.get("kind") == "intercept":
            self.b_global: float = float(state.get("b", 0.0))
        else:
            self.b_global = 0.0
        self.per_bc: dict[str, float] = {}

    def fit_from_labeled(self, labeled_list) -> None:
        if not labeled_list:
            return
        ps: list[float] = []
        ys: list[float] = []
        bcs: list[str] = []
        for ex in labeled_list:
            try:
                lbl = ex.get("label")
                if lbl is None:
                    continue
                y = float(lbl)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(y):
                continue
            benchmark = str(ex.get("benchmark", "") or "")
            condition = normalize_condition(ex.get("condition", "none"))
            subject_content = str(ex.get("subject_content", "") or "")
            item_content = str(ex.get("item_content", "") or "")
            try:
                p_base = _predict_uncalibrated(
                    benchmark, condition, subject_content, item_content
                )
            except Exception:
                continue
            if not math.isfinite(p_base):
                continue
            bc_key = "{0}::{1}".format(benchmark, condition)
            ps.append(float(p_base))
            ys.append(min(max(y, 0.0), 1.0))
            bcs.append(bc_key)
        if not ps:
            return
        self.b_global = _fit_intercept_ridge(
            ps, ys, target_b=0.0, ridge=_RIDGE_LAMBDA_GLOBAL
        )
        bc_to_pairs: dict[str, tuple[list[float], list[float]]] = {}
        for p, y, bc in zip(ps, ys, bcs):
            bucket = bc_to_pairs.setdefault(bc, ([], []))
            bucket[0].append(p)
            bucket[1].append(y)
        self.per_bc = {}
        for bc_key, (lp, ly) in bc_to_pairs.items():
            b_bc = _fit_intercept_ridge(
                lp, ly, target_b=self.b_global, ridge=_RIDGE_LAMBDA_BC
            )
            self.per_bc[bc_key] = float(b_bc)
        try:
            n_new_benchmarks = sum(
                1 for k in self.per_bc if _BC_TO_ID.get(k, 0) == 0
            )
            LOG.info(
                "Calibrator fit: N=%d, b_global=%+.4f, per_bc=%d (of which %d are NEW benchmarks)",
                len(ps),
                self.b_global,
                len(self.per_bc),
                n_new_benchmarks,
            )
        except Exception:
            pass

    def apply(self, p: float, bc_key: str = "") -> float:
        if not math.isfinite(p):
            return DEFAULT_PROB
        if isinstance(bc_key, str) and bc_key in self.per_bc:
            b = float(self.per_bc[bc_key])
        else:
            b = float(self.b_global)
        try:
            z = _logit(p) + b
            q = _sigmoid(z)
        except Exception:
            return DEFAULT_PROB
        if not math.isfinite(q):
            return DEFAULT_PROB
        return float(min(max(q, EPS), 1.0 - EPS))


def _calibrator_from_state(state: dict | None) -> "_Calibrator":
    return _Calibrator(state if isinstance(state, dict) else None)


'''


# ---------------------------------------------------------------------------
# labeling.py (benchmark-novelty + anchor-model acquisition)
# ---------------------------------------------------------------------------

NEW_LABELING_PY = '''"""Adaptive-labeling acquisition function (dual-pool stratification).

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
    h = hashlib.blake2b(("\\x00".join(parts)).encode("utf-8"), digest_size=8).digest()
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
'''


# ---------------------------------------------------------------------------
# predict() apply-site patches
# ---------------------------------------------------------------------------

# Pattern A: the bundle is the standard "calibrated" variant -- predict()
# already has the global declarations, the `if labeled: ...` fit block,
# and the bare ``p = _CALIBRATOR.apply(p)`` line.  We only need to swap
# the apply line for the bc-routed form.
_OLD_APPLY_LINE = "        p = _CALIBRATOR.apply(p)\n"
_NEW_APPLY_LINE = (
    '        _bc_key_for_apply = "{0}::{1}".format(benchmark, condition)\n'
    "        p = _CALIBRATOR.apply(p, _bc_key_for_apply)\n"
)

# Pattern B: the bundle is a "nocal" variant -- the apply line is
# commented out and the labeled arg is discarded with a `_ = labeled`
# touch.  We need to re-enable BOTH pieces (the fit + the apply) since
# we are turning calibration back on.
_NOCAL_DISABLED_FIT_RE = re.compile(
    r"        # Calibration intentionally disabled in this variant:[^\n]*\n"
    r"(?:        #[^\n]*\n)+"
    r"        _ = labeled[^\n]*\n"
)
_NOCAL_DISABLED_FIT_REPLACEMENT = (
    "        if labeled:\n"
    "            fp = _labeled_fingerprint(labeled)\n"
    "            if fp != _LAST_LABELED_FINGERPRINT:\n"
    "                _LAST_LABELED_FINGERPRINT = fp\n"
    "                _PROB_CACHE.clear()\n"
    "                _CALIBRATOR = _Calibrator(META.get(\"default_calibrator\"))\n"
    "                _CALIBRATOR.fit_from_labeled(labeled)\n"
)
_NOCAL_COMMENTED_APPLY_LINE = (
    "        # p = _CALIBRATOR.apply(p)  # disabled: uncalibrated variant\n"
)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

_CAL_BANNER_RE = re.compile(r"^# -{50,}\n# Calibrator [^\n]*\n# -{50,}\n", re.M)


def replace_calibrator_block(model_py: str) -> str:
    m = _CAL_BANNER_RE.search(model_py)
    if not m:
        raise RuntimeError("could not find calibrator banner in model.py")
    start = m.start()
    next_banner = re.search(r"\n\n\n# -{50,}\n", model_py[m.end():])
    if not next_banner:
        raise RuntimeError("could not find next banner after calibrator block")
    end = m.end() + next_banner.start() + 1
    return model_py[:start] + NEW_CALIBRATOR_BLOCK + model_py[end:]


def replace_apply_site(model_py: str) -> str:
    """Patch predict()'s calibrator wiring.

    Supports two bundle shapes:

      * "calibrated": predict() already contains
        ``p = _CALIBRATOR.apply(p)`` -- we just swap it for the bc-routed
        equivalent.

      * "nocal": predict() has the apply line commented out AND a
        "Calibration intentionally disabled" block that touches `labeled`
        without using it.  We re-enable BOTH so the patched runtime
        actually fits and applies the calibrator.

    The function detects which shape applies by which markers are
    present, and raises if neither matches.
    """
    has_active = _OLD_APPLY_LINE in model_py
    has_nocal = _NOCAL_COMMENTED_APPLY_LINE in model_py and bool(
        _NOCAL_DISABLED_FIT_RE.search(model_py)
    )
    if has_active and has_nocal:
        raise RuntimeError(
            "model.py contains BOTH active and nocal apply markers -- ambiguous"
        )
    if has_active:
        out = model_py.replace(_OLD_APPLY_LINE, _NEW_APPLY_LINE, 1)
        if _OLD_APPLY_LINE in out:
            raise RuntimeError("found multiple active apply sites; aborting")
        return out
    if has_nocal:
        out = _NOCAL_DISABLED_FIT_RE.sub(
            _NOCAL_DISABLED_FIT_REPLACEMENT, model_py, count=1
        )
        if _NOCAL_DISABLED_FIT_RE.search(out):
            raise RuntimeError("nocal-disabled fit block still present after patch")
        out = out.replace(
            _NOCAL_COMMENTED_APPLY_LINE, _NEW_APPLY_LINE, 1
        )
        if _NOCAL_COMMENTED_APPLY_LINE in out:
            raise RuntimeError("nocal commented apply line still present after patch")
        # The nocal variant relies on these helpers being defined at module
        # scope; if any are missing the patched predict() would NameError.
        for name in (
            "_labeled_fingerprint",
            "_LAST_LABELED_FINGERPRINT",
            "_PROB_CACHE",
            "_CALIBRATOR = _Calibrator",
        ):
            if name not in out:
                raise RuntimeError(
                    "nocal patch requires `{}` to be defined at module scope".format(name)
                )
        return out
    raise RuntimeError(
        "could not find an apply-site pattern to patch in predict() "
        "(neither calibrated nor nocal markers present)"
    )


def sanity_check_model_py(model_py: str) -> None:
    """Validate the patched model.py before shipping."""
    ast.parse(model_py)
    must_have = [
        "class _Calibrator",
        "def fit_from_labeled",
        "def _fit_intercept_ridge",
        "_CALIBRATOR.apply(p, _bc_key_for_apply)",
        "self.per_bc: dict[str, float]",
        "_RIDGE_LAMBDA_GLOBAL",
        "_RIDGE_LAMBDA_BC",
        "target_b",
        "self.b_global",
    ]
    for needle in must_have:
        if needle not in model_py:
            raise RuntimeError("patched model.py missing required symbol: " + needle)
    must_not_have = [
        "def _fit_beta_calibration",
        "def _beta_calibration_loss",
        '"kind": "beta"',
        # The gated-fit machinery is gone now -- partial pooling replaced it.
        "def _gated_fit",
        "def _cv_nll_pair",
        "def _kfold_indices",
        "def _stable_shuffle_order",
        "margin_nats_per_param",
        "require_majority_repeat_wins",
    ]
    for needle in must_not_have:
        if needle in model_py:
            raise RuntimeError(
                "patched model.py still has obsolete symbol: " + needle
            )


def sanity_check_labeling_py(labeling_py: str) -> None:
    ast.parse(labeling_py)
    must_have = [
        "_BC_TO_ID",
        "_SUBJECT_TO_ID",
        "_N_TRAIN_PER_BC",
        "_N_TRAIN_PER_SUBJECT",
        "_FRACTION_NEW_POOL",
        "_item_in_new_pool",
        "in_new_pool",
        "is_new_bc",
        "eligible",
        "anchoring",
        "_enqueue_for_batch",
        "10.0 * anchoring",
    ]
    for needle in must_have:
        if needle not in labeling_py:
            raise RuntimeError("labeling.py missing required symbol: " + needle)
    if "from model import _baseline_logit" in labeling_py:
        raise RuntimeError("labeling.py still imports the old uncertainty fast-path")
    if "1000.0 * novelty" in labeling_py:
        raise RuntimeError(
            "labeling.py still uses the old 1000*novelty score -- expected dual-pool stratification"
        )


def required_model_py_prereqs(model_py: str) -> None:
    """Validate the SOURCE model.py exposes the symbols labeling.py needs."""
    must_have_pre_predict = ["_BC_TO_ID:", "_SUBJECT_TO_ID:", "def normalize_condition", "def stable_sha256"]
    pred_off = model_py.find("def predict(input: dict")
    if pred_off < 0:
        raise RuntimeError("source model.py has no predict() function")
    for needle in must_have_pre_predict:
        off = model_py.find(needle)
        if off < 0:
            raise RuntimeError(
                "source model.py is missing required symbol: " + needle
            )
        if off > pred_off:
            raise RuntimeError(
                "source model.py defines {} AFTER predict() (must come before)".format(needle)
            )


# Validate our own embedded source compiles before any pack script can use it.
ast.parse(NEW_LABELING_PY)
