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
# Calibrator (per-benchmark held-out-gated intercept w/ ridge + identity)
# ---------------------------------------------------------------------------
#
# Design notes:
#
#   * Only one fitted tier -- ridge-regularized intercept-only -- because
#     N<=75 (75 total per round, ~5 per benchmark on average) is way below
#     what is needed to fit a second free parameter reliably.  An earlier
#     experiment with a temp+intercept tier turned out to overfit so hard
#     that CV with random shuffles flipped between accept and reject (T
#     fits ranged from 0.6 to 0.85 on data that was actually well
#     calibrated).
#
#   * Held-out gate uses REPEATED 5-fold CV (5 shuffles) so the gate
#     decision does not depend on the caller's input order, plus an
#     AIC-style complexity margin (4 nats per param per repeat) AND a
#     per-repeat majority gate (calibrator must beat baseline in at least
#     ceil(n_repeats / 2) of the individual shuffles).
#
#   * The fit itself adds an L2 ridge ``lambda * b**2`` so even when
#     accepted, the intercept gets pulled toward zero.  lambda = 1.0,
#     equivalent to one fake observation pinned at b = 0.
#
#   * Per-bc routing uses the raw ``bc_key`` string
#     (``"benchmark::condition"``) -- NOT the embedding table's bc_id,
#     which would collapse every unseen benchmark to 0 and merge them
#     all into a single bucket (the opposite of what we want for
#     new-benchmark calibration).


_RIDGE_LAMBDA = 1.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _fit_intercept_ridge(ps, ys, ridge: float = _RIDGE_LAMBDA):
    """Fit logit(p') = logit(p) + b by 1D Newton on BCE + ridge * b**2."""
    if not ps:
        return {"kind": "identity"}
    zs = [_logit(p) for p in ps]
    b = 0.0
    for _ in range(80):
        g = 2.0 * ridge * b
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
        return {"kind": "identity"}
    b = max(-5.0, min(5.0, float(b)))
    return {"kind": "intercept", "b": b}


def _apply_state(p: float, state: dict) -> float:
    if not math.isfinite(p):
        return DEFAULT_PROB
    kind = state.get("kind", "identity")
    if kind == "intercept":
        z = _logit(p) + float(state.get("b", 0.0))
        return _sigmoid(z)
    return p


def _stable_shuffle_order(ps, ys) -> list:
    """Deterministic permutation invariant to input order."""
    keyed = [(float(p), float(y), j) for j, (p, y) in enumerate(zip(ps, ys))]
    keyed.sort()
    return [k[2] for k in keyed]


def _kfold_indices(perm: list, k: int, seed: int):
    """Apply seed-controlled rotation on ``perm`` then stripe into k folds."""
    n = len(perm)
    if n == 0 or k <= 0:
        return [perm]
    rotated = list(perm)
    rng = __import__("random").Random(seed)
    rng.shuffle(rotated)
    return [rotated[i::k] for i in range(k)]


def _cv_nll_pair(ps, ys, fitter, baseline_state, folds):
    """Return (cal_total_nll, base_total_nll, counted) over `folds`."""
    n = len(ps)
    cal_total = 0.0
    base_total = 0.0
    counted = 0
    for test in folds:
        if not test:
            continue
        train_set = set(test)
        train_p = [ps[j] for j in range(n) if j not in train_set]
        train_y = [ys[j] for j in range(n) if j not in train_set]
        try:
            cal_state = fitter(train_p, train_y)
        except Exception:
            cal_state = {"kind": "identity"}
        for j in test:
            p_cal = _apply_state(ps[j], cal_state)
            p_base = _apply_state(ps[j], baseline_state)
            p_cal = min(max(p_cal, EPS), 1.0 - EPS)
            p_base = min(max(p_base, EPS), 1.0 - EPS)
            y = ys[j]
            cal_total -= y * math.log(p_cal) + (1.0 - y) * math.log(1.0 - p_cal)
            base_total -= y * math.log(p_base) + (1.0 - y) * math.log(1.0 - p_base)
            counted += 1
    return cal_total, base_total, counted


def _gated_fit(
    ps,
    ys,
    baseline_state,
    *,
    min_total: int = 5,
    n_repeats: int = 5,
    margin_nats_per_param: float = 4.0,
    require_majority_repeat_wins: bool = True,
):
    """Return a ridge-fit intercept state if it convincingly beats baseline,
    else return ``baseline_state`` unchanged.
    """
    n = len(ps)
    if n < min_total:
        return baseline_state
    fitter = _fit_intercept_ridge
    k_params = 1
    k = 5 if n >= 10 else n
    perm = _stable_shuffle_order(ps, ys)
    cal_total = 0.0
    base_total = 0.0
    counted = 0
    repeat_wins = 0
    for rep in range(n_repeats):
        folds = _kfold_indices(perm, k, seed=0xC0FFEE + rep * 7919)
        c, b_, n_eval = _cv_nll_pair(ps, ys, fitter, baseline_state, folds)
        cal_total += c
        base_total += b_
        counted += n_eval
        if c < b_:
            repeat_wins += 1
    margin = margin_nats_per_param * k_params * n_repeats
    if counted == 0:
        return baseline_state
    if cal_total >= base_total - margin:
        return baseline_state
    if require_majority_repeat_wins and repeat_wins < (n_repeats + 1) // 2:
        return baseline_state
    try:
        return fitter(ps, ys)
    except Exception:
        return baseline_state


class _Calibrator:
    """Hierarchical per-benchmark calibrator with held-out NLL gating.

    Routing path at apply time:
        local[bc_key] (if accepted) -> global -> identity
    where bc_key is the raw "benchmark::condition" string so each new
    benchmark gets its own slot.
    """

    def __init__(self, state: dict | None = None) -> None:
        self.state: dict = dict(state) if isinstance(state, dict) else {"kind": "identity"}
        self.per_bc: dict[str, dict] = {}

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
        global_state = _gated_fit(ps, ys, baseline_state={"kind": "identity"})
        self.state = global_state
        bc_to_pairs: dict[str, tuple[list[float], list[float]]] = {}
        for p, y, bc in zip(ps, ys, bcs):
            bucket = bc_to_pairs.setdefault(bc, ([], []))
            bucket[0].append(p)
            bucket[1].append(y)
        local: dict[str, dict] = {}
        for bc_key, (lp, ly) in bc_to_pairs.items():
            if len(lp) < 5:
                continue
            cand = _gated_fit(lp, ly, baseline_state=global_state, min_total=5)
            if cand is not global_state and cand.get("kind", "identity") != "identity":
                local[bc_key] = cand
        self.per_bc = local
        try:
            n_new_benchmarks = sum(
                1 for k in self.per_bc if _BC_TO_ID.get(k, 0) == 0
            )
            LOG.info(
                "Calibrator fit: N=%d, global=%s, per_bc=%d (of which %d are NEW benchmarks)",
                len(ps),
                self.state.get("kind", "identity"),
                len(self.per_bc),
                n_new_benchmarks,
            )
        except Exception:
            pass

    def apply(self, p: float, bc_key: str = "") -> float:
        if not math.isfinite(p):
            return DEFAULT_PROB
        if isinstance(bc_key, str) and bc_key in self.per_bc:
            q = _apply_state(p, self.per_bc[bc_key])
        else:
            q = _apply_state(p, self.state)
        if not math.isfinite(q):
            return DEFAULT_PROB
        return float(min(max(q, EPS), 1.0 - EPS))


def _calibrator_from_state(state: dict | None) -> "_Calibrator":
    return _Calibrator(state if isinstance(state, dict) else None)


'''


# ---------------------------------------------------------------------------
# labeling.py (benchmark-novelty + anchor-model acquisition)
# ---------------------------------------------------------------------------

NEW_LABELING_PY = '''"""Adaptive-labeling acquisition function (benchmark-novelty + anchor-model).

The Codabench runner calls ``acquisition_function`` once per hidden
(subject, item) pair BEFORE any ``predict()`` call.  Each call here does
two things:

  1. Forward the input to ``model._enqueue_for_batch`` so the streamed-
     flush architecture can start populating encoder + judge caches in
     the background.

  2. Return an ACQUISITION SCORE biased toward new benchmarks observed
     with well-anchored models. The score has three layers:

       score = 1000 * novelty(B) + 10 * anchoring(M) + tiebreak

     where the SHAPE of novelty/anchoring depends on what the runtime
     model.py exposes:

       GRADED mode (preferred; requires model._N_TRAIN_PER_BC and
       model._N_TRAIN_PER_SUBJECT to be present at module scope):

         novelty(B)   = 1 / sqrt(1 + n_train_per_bc[bc_key])
                        -- 1.0 for an unseen benchmark, decaying smoothly
                           as the benchmark's training-row count grows.
         anchoring(M) = log1p(n_train_per_subject[sha256(subject)]) / LN
                        -- 0.0 for an unseen model, log-scaled toward 1.0
                           as the subject's training-row count grows.
                           LN = log1p(max(_N_TRAIN_PER_SUBJECT.values())).

       BINARY fallback (when the runtime bundle does not ship counts --
       e.g. bundles patched in-place by the surgical pack scripts):

         novelty(B)   = 1 if bc_id == 0 else 0
         anchoring(M) = 1 if s_id  != 0 else 0

     Both modes preserve the same priority ordering -- novelty (max +1000)
     dominates anchoring (max +10) dominates tiebreak (~+/-0.5) -- so the
     platform's top-K-per-category sampler will reliably pick new-benchmark
     anchor-model rows over alternatives.

If any of the model.py lookups fail (corrupt bundle, missing symbol) we
silently fall back to 0.0 -- the platform handles that as uniform random
within the category.
"""

from __future__ import annotations

import hashlib
import math


def _stable_tiebreak(*parts: str) -> float:
    """Return a deterministic float in [-0.5, +0.5] derived from inputs."""
    h = hashlib.blake2b(("\\x00".join(parts)).encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return (n / 2 ** 64) - 0.5


def _graded_novelty(bc_key: str, n_train_per_bc) -> float:
    n = float(n_train_per_bc.get(bc_key, 0))
    return 1.0 / math.sqrt(1.0 + n)


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
            novelty = _graded_novelty(bc_key, _N_TRAIN_PER_BC)
            anchoring = _graded_anchoring(subject_key, _N_TRAIN_PER_SUBJECT, log_max_plus_one)
        else:
            novelty = 1.0 if bc_id == 0 else 0.0
            anchoring = 1.0 if s_id != 0 else 0.0

        tb = _stable_tiebreak(benchmark, condition, subject_content, item_content)
        return 1000.0 * novelty + 10.0 * anchoring + tb
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
        "def _gated_fit",
        "def _apply_state",
        "def _fit_intercept_ridge",
        "_CALIBRATOR.apply(p, _bc_key_for_apply)",
        "self.per_bc: dict[str, dict]",
        "_RIDGE_LAMBDA",
        "n_repeats",
        "margin_nats_per_param",
        "require_majority_repeat_wins",
    ]
    for needle in must_have:
        if needle not in model_py:
            raise RuntimeError("patched model.py missing required symbol: " + needle)
    must_not_have = [
        "def _fit_beta_calibration",
        "def _beta_calibration_loss",
        '"kind": "beta"',
    ]
    for needle in must_not_have:
        if needle in model_py:
            raise RuntimeError(
                "patched model.py still has the beta-calibration tier: " + needle
            )


def sanity_check_labeling_py(labeling_py: str) -> None:
    ast.parse(labeling_py)
    must_have = [
        "_BC_TO_ID",
        "_SUBJECT_TO_ID",
        "_N_TRAIN_PER_BC",
        "_N_TRAIN_PER_SUBJECT",
        "novelty",
        "anchoring",
        "_enqueue_for_batch",
        "1000.0 * novelty",
        "10.0 * anchoring",
    ]
    for needle in must_have:
        if needle not in labeling_py:
            raise RuntimeError("labeling.py missing required symbol: " + needle)
    if "from model import _baseline_logit" in labeling_py:
        raise RuntimeError("labeling.py still imports the old uncertainty fast-path")


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
