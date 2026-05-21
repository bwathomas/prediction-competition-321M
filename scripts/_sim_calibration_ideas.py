"""Head-to-head simulation of calibration ideas.

Goal: empirically settle whether each of the three proposed enhancements
to the current per-bc gated calibrator actually improves held-out NLL on
the platform's actual regime.

Variants tested
---------------
  IDENTITY                  : no calibration (sanity floor / ceiling).
  BASELINE                  : production code -- per-bc gated ridge
                              intercept toward 0, with the same repeated
                              5-fold CV gate + 20-nat AIC margin +
                              majority-wins.
  PARTIAL_POOL              : drop the hard gate; every bc gets a
                              ridge intercept fit shrunk toward the
                              global state with strength tau_bc.
  ANCHOR_WEIGHTED           : partial pool + weight each label by
                              log1p(n_train_per_subject) (normalized).
  PREFIT_PRIOR              : partial pool + initialize the GLOBAL
                              shrinkage target with a calibrator fit on
                              a large training-time labeled set (mimics
                              what export_run() could embed in META).
  ALL_THREE                 : PREFIT_PRIOR + ANCHOR_WEIGHTED partial
                              pool, the "everything we'd actually ship"
                              configuration.

Regimes
-------
We sweep two binary axes and run the full grid:

  anchor_noise: do less-known subjects produce noisier uncalibrated
                probabilities than well-known ones?
                FALSE -> anchor weighting should NOT help (or should
                hurt via lost effective N).
                TRUE  -> anchor weighting SHOULD help (Gauss-Markov
                weighting by inverse noise variance).

  new_bc_bias_matches_training: are the (true) per-bc biases on NEW
                benchmarks drawn from the same distribution as on
                training benchmarks?
                FALSE -> pre-fit prior helps only on known bcs (new
                ones see a wrong prior).
                TRUE  -> pre-fit prior helps everywhere.

For each regime we run 50 trials with different seeds and report mean
NLL, stddev, and win rate vs BASELINE for each variant.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fit_ridge_intercept(
    ps: np.ndarray,
    ys: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    target_b: float = 0.0,
    ridge: float = 1.0,
    max_iter: int = 80,
) -> float:
    """Newton fit of logit(p') = logit(p) + b with ridge toward target_b.

    Weighted BCE: sum_i w_i * BCE(z_i + b, y_i) + ridge * (b - target_b)^2.
    Closed-form gradient + Hessian, clamp |b| <= 5.
    """
    n = len(ps)
    if n == 0:
        return float(target_b)
    zs = logit(ps)
    if weights is None:
        weights = np.ones(n)
    b = float(target_b)
    for _ in range(max_iter):
        q = sigmoid(zs + b)
        g = 2.0 * ridge * (b - target_b) + float(np.sum(weights * (q - ys)))
        h = 2.0 * ridge + float(np.sum(weights * q * (1.0 - q)))
        if h < 1e-9:
            break
        step = g / h
        new_b = b - step
        if not math.isfinite(new_b):
            break
        if abs(new_b - b) < 1e-8:
            b = new_b
            break
        b = float(new_b)
    if not math.isfinite(b):
        return float(target_b)
    return max(-5.0, min(5.0, b))


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


class World:
    """Synthetic dataset mimicking the platform regime.

    * M_known known benchmarks (n_train_per_bc > 0).
    * M_new   new benchmarks (n_train_per_bc == 0).
    * S subjects with log-uniform training row counts.
    * Each (subject, bc) sample draws z_true ~ N(0, 1).
      Uncalibrated head returns sigmoid(z_observed) where z_observed
      may add subject-conditional noise (anchor_noise=True).
      True probability is sigmoid(z_true + b_bc_true[bc]).
      Label y ~ Bernoulli(true probability).
    """

    def __init__(
        self,
        *,
        M_known: int = 12,
        M_new: int = 3,
        S: int = 50,
        anchor_noise: bool = False,
        anchor_noise_scale: float = 1.5,
        b_bc_mu_known: float = -0.5,
        b_bc_sigma_known: float = 0.8,
        b_bc_mu_new: float = -0.5,
        b_bc_sigma_new: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.M_known = M_known
        self.M_new = M_new
        self.M = M_known + M_new
        self.S = S
        # Per-bc training row count -- known bcs log-uniform [30, 5000];
        # new bcs zero.
        self.n_train_per_bc = np.zeros(self.M)
        self.n_train_per_bc[:M_known] = np.exp(
            self.rng.uniform(math.log(30), math.log(5000), M_known)
        ).astype(int)
        # Per-subject training row count -- log-uniform [1, 5000].
        self.n_train_per_subject = np.exp(
            self.rng.uniform(0.0, math.log(5000), S)
        ).astype(int)
        # Per-bc true bias.  Known and new bcs may be drawn from
        # different distributions to test prior robustness.
        self.b_bc_true = np.empty(self.M)
        self.b_bc_true[:M_known] = self.rng.normal(b_bc_mu_known, b_bc_sigma_known, M_known)
        self.b_bc_true[M_known:] = self.rng.normal(b_bc_mu_new, b_bc_sigma_new, M_new)
        self.anchor_noise = anchor_noise
        self.anchor_noise_scale = anchor_noise_scale
        if S > 0 and self.n_train_per_subject.max() > 0:
            self.anchor_normalized = np.log1p(self.n_train_per_subject) / math.log1p(
                self.n_train_per_subject.max()
            )
        else:
            self.anchor_normalized = np.zeros(S)
        self.new_bc_mask = np.zeros(self.M, dtype=bool)
        self.new_bc_mask[M_known:] = True

    def _gen_uncal_and_label(self, subject_ids: np.ndarray, bc_ids: np.ndarray, rng):
        n = len(subject_ids)
        z_true = rng.normal(0.0, 1.0, n)
        if self.anchor_noise:
            sigma = self.anchor_noise_scale * (1.0 - self.anchor_normalized[subject_ids])
            z_observed = z_true + sigma * rng.normal(0.0, 1.0, n)
        else:
            z_observed = z_true.copy()
        p_uncal = sigmoid(z_observed)
        true_p = sigmoid(z_true + self.b_bc_true[bc_ids])
        y = rng.binomial(1, true_p).astype(float)
        return p_uncal, y, true_p


def acquisition_biased_labels(world: World, *, n_labels: int = 75, seed: int, balanced: bool = False):
    """Sample 75 labels under one of two acquisition policies.

    ``balanced=False`` (default): the current production policy --
    1000 * novelty + 10 * anchoring + tiebreak.  In practice this
    saturates: all 75 slots go to new bcs because novelty=1 for them
    beats novelty<<1 for any known bc by 5x or more.

    ``balanced=True``: enforce a 50/50 split between new and known bcs.
    For each half we still sort by novelty + anchoring, so within each
    pool we pick the rarest bc + most-anchored subject; but the calibrator
    now actually SEES known-bc labels and can correct their drift.
    """
    rng = np.random.default_rng(seed)
    bc_novelty = 1.0 / np.sqrt(1.0 + world.n_train_per_bc)
    subject_anchor = world.anchor_normalized

    def _pick_from_pool(bc_pool: np.ndarray, k: int):
        if k <= 0 or len(bc_pool) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        n_pool = max(k * 100, 1000)
        cand_subj = rng.integers(0, world.S, n_pool)
        cand_bc = bc_pool[rng.integers(0, len(bc_pool), n_pool)]
        tb = rng.uniform(-0.5, 0.5, n_pool)
        scores = 1000.0 * bc_novelty[cand_bc] + 10.0 * subject_anchor[cand_subj] + tb
        top = np.argsort(-scores)[:k]
        return cand_subj[top], cand_bc[top]

    all_bc = np.arange(world.M)
    if balanced:
        new_bc = np.where(world.new_bc_mask)[0]
        known_bc = np.where(~world.new_bc_mask)[0]
        k_new = n_labels // 2
        k_known = n_labels - k_new
        s1, b1 = _pick_from_pool(new_bc, k_new)
        s2, b2 = _pick_from_pool(known_bc, k_known)
        subj = np.concatenate([s1, s2])
        bc = np.concatenate([b1, b2])
    else:
        subj, bc = _pick_from_pool(all_bc, n_labels)
    p_uncal, y, true_p = world._gen_uncal_and_label(subj, bc, rng)
    return subj, bc, p_uncal, y, true_p


def uniform_test_set(world: World, *, n: int = 10000, seed: int):
    rng = np.random.default_rng(seed)
    subj = rng.integers(0, world.S, n)
    bc = rng.integers(0, world.M, n)
    p_uncal, y, true_p = world._gen_uncal_and_label(subj, bc, rng)
    return subj, bc, p_uncal, y, true_p


# ---------------------------------------------------------------------------
# Calibrator variants
# ---------------------------------------------------------------------------


def _stable_perm(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    key = ps * 10.0 + ys * 0.1 + np.arange(len(ps)) * 1e-7
    return np.argsort(key)


def cv_gate(
    ps: np.ndarray,
    ys: np.ndarray,
    *,
    candidate_target: float,
    baseline_target: float,
    ridge: float,
    n_repeats: int = 5,
    n_folds: int = 5,
    margin_per_param_per_repeat: float = 4.0,
) -> bool:
    """Production gate: repeated K-fold CV with AIC margin + majority wins.

    Returns True iff the candidate fit beats the baseline state by at
    least k_params * n_repeats * margin_per_param_per_repeat total nats
    on held-out folds AND wins in a majority of individual repeats.
    """
    n = len(ps)
    if n < 5:
        return False
    k = min(5, n) if n < 10 else 5
    perm = _stable_perm(ps, ys)
    cal_total = 0.0
    base_total = 0.0
    repeat_wins = 0
    for r in range(n_repeats):
        rng = np.random.default_rng(0xC0FFEE + r * 7919)
        rotated = rng.permutation(perm)
        folds = [rotated[i::k] for i in range(k)]
        c_tot = 0.0
        b_tot = 0.0
        for fold in folds:
            if len(fold) == 0:
                continue
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            tp = ps[mask]
            ty = ys[mask]
            if len(tp) == 0:
                cal_b = candidate_target
            else:
                cal_b = fit_ridge_intercept(
                    tp, ty, target_b=candidate_target, ridge=ridge
                )
            xp = ps[fold]
            xy = ys[fold]
            p_cal = sigmoid(logit(xp) + cal_b)
            p_base = sigmoid(logit(xp) + baseline_target)
            p_cal = np.clip(p_cal, EPS, 1 - EPS)
            p_base = np.clip(p_base, EPS, 1 - EPS)
            c_tot += float(-np.sum(xy * np.log(p_cal) + (1 - xy) * np.log(1 - p_cal)))
            b_tot += float(-np.sum(xy * np.log(p_base) + (1 - xy) * np.log(1 - p_base)))
        cal_total += c_tot
        base_total += b_tot
        if c_tot < b_tot:
            repeat_wins += 1
    margin = margin_per_param_per_repeat * 1 * n_repeats  # k_params = 1
    if cal_total >= base_total - margin:
        return False
    if repeat_wins < (n_repeats + 1) // 2:
        return False
    return True


def cal_baseline(
    subj: np.ndarray, bc: np.ndarray, ps: np.ndarray, ys: np.ndarray, **_
) -> dict:
    if len(ps) < 5:
        return {"global": 0.0, "per_bc": {}}
    b_global_fit = fit_ridge_intercept(ps, ys, target_b=0.0, ridge=1.0)
    accept_global = cv_gate(
        ps, ys, candidate_target=b_global_fit, baseline_target=0.0, ridge=1.0
    )
    b_global = b_global_fit if accept_global else 0.0
    by_bc = defaultdict(lambda: ([], []))
    for p, y, b_id in zip(ps, ys, bc):
        by_bc[int(b_id)][0].append(float(p))
        by_bc[int(b_id)][1].append(float(y))
    per_bc: dict[int, float] = {}
    for bc_id, (lps, lys) in by_bc.items():
        if len(lps) < 5:
            continue
        ap = np.array(lps)
        ay = np.array(lys)
        b_bc_fit = fit_ridge_intercept(ap, ay, target_b=0.0, ridge=1.0)
        if cv_gate(ap, ay, candidate_target=b_bc_fit, baseline_target=b_global, ridge=1.0):
            per_bc[bc_id] = b_bc_fit
    return {"global": b_global, "per_bc": per_bc}


def cal_partial_pool(
    subj: np.ndarray,
    bc: np.ndarray,
    ps: np.ndarray,
    ys: np.ndarray,
    *,
    prior_b: float = 0.0,
    tau_global: float = 1.0,
    tau_bc: float = 5.0,
    weights: np.ndarray | None = None,
) -> dict:
    """Continuous shrinkage: global -> prior_b, per_bc -> global."""
    if len(ps) < 5:
        return {"global": float(prior_b), "per_bc": {}}
    b_global = fit_ridge_intercept(
        ps, ys, weights=weights, target_b=prior_b, ridge=tau_global
    )
    by_bc: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    if weights is None:
        w_use = np.ones(len(ps))
    else:
        w_use = weights
    for i, b_id in enumerate(bc):
        by_bc[int(b_id)][0].append(float(ps[i]))
        by_bc[int(b_id)][1].append(float(ys[i]))
        by_bc[int(b_id)][2].append(float(w_use[i]))
    per_bc: dict[int, float] = {}
    for bc_id, (lps, lys, lws) in by_bc.items():
        if len(lps) == 0:
            continue
        ap = np.array(lps)
        ay = np.array(lys)
        aw = np.array(lws)
        b_bc = fit_ridge_intercept(
            ap, ay, weights=aw, target_b=b_global, ridge=tau_bc
        )
        per_bc[bc_id] = b_bc
    return {"global": b_global, "per_bc": per_bc}


def cal_anchor_weighted(
    world: World,
    subj: np.ndarray,
    bc: np.ndarray,
    ps: np.ndarray,
    ys: np.ndarray,
    *,
    prior_b: float = 0.0,
    tau_global: float = 1.0,
    tau_bc: float = 5.0,
) -> dict:
    """Partial pool with anchor-strength weights, normalized to mean=1."""
    raw = np.log1p(world.n_train_per_subject[subj]).astype(float)
    if raw.mean() <= 0:
        weights = np.ones_like(raw)
    else:
        weights = raw / raw.mean()
    return cal_partial_pool(
        subj, bc, ps, ys,
        prior_b=prior_b, tau_global=tau_global, tau_bc=tau_bc,
        weights=weights,
    )


def fit_training_time_prior(world: World, *, n_rows: int = 10000, seed: int = 7) -> float:
    """Mimic export_run computing a training-time ridge-intercept prior.

    Samples n_rows training labels uniformly over (subject, KNOWN bc)
    and fits a single global ridge intercept on them.  In production
    this would be embedded in runtime_meta as the default calibrator
    state.
    """
    rng = np.random.default_rng(seed)
    # Sample only over KNOWN bcs (training data has no new bcs by definition)
    subj = rng.integers(0, world.S, n_rows)
    bc = rng.integers(0, world.M_known, n_rows)
    ps, ys, _ = world._gen_uncal_and_label(subj, bc, rng)
    return fit_ridge_intercept(ps, ys, target_b=0.0, ridge=1.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def apply_cal(cal: dict, bc: np.ndarray, ps: np.ndarray) -> np.ndarray:
    bs = np.array([cal["per_bc"].get(int(b), cal["global"]) for b in bc])
    return sigmoid(logit(ps) + bs)


def run_one_trial(
    *,
    trial_seed: int,
    anchor_noise: bool,
    realistic_head: bool,
    balanced_acquisition: bool = False,
    n_labels: int = 75,
    n_test: int = 10000,
) -> dict:
    """Run one round.  Returns per-variant test NLL plus test NLL split
    on new vs known bcs.

    ``realistic_head`` flips between two regimes:

      False ("uniform miscalibration"):
        known and new bcs both have b ~ N(-0.5, 0.8).  Calibration is
        worth the same everywhere; this is the easy case.

      True ("post-trained head"):
        known bcs have b ~ N(0, 0.2)  -- the head was TRAINED on these,
                                         so it's already mostly calibrated
                                         but has small per-bc residual bias.
        new   bcs have b ~ N(0, 1.0)  -- head never saw them, so
                                         miscalibration is much larger
                                         and has random sign.
        This is what the actual deployment looks like.  The training-time
        prior is approximately zero (because training data is well
        calibrated), and per-bc calibration is the lever that actually
        matters -- but the labels we get are biased toward new bcs by
        acquisition.
    """
    if realistic_head:
        world = World(
            M_known=12, M_new=3, S=50,
            anchor_noise=anchor_noise,
            b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
            b_bc_mu_new=0.0,   b_bc_sigma_new=1.0,
            seed=trial_seed,
        )
    else:
        world = World(
            M_known=12, M_new=3, S=50,
            anchor_noise=anchor_noise,
            b_bc_mu_known=-0.5, b_bc_sigma_known=0.8,
            b_bc_mu_new=-0.5,   b_bc_sigma_new=0.8,
            seed=trial_seed,
        )

    subj, bc, ps, ys, _ = acquisition_biased_labels(
        world, n_labels=n_labels, seed=trial_seed + 1, balanced=balanced_acquisition,
    )
    _, test_bc, test_ps, test_ys, _ = uniform_test_set(
        world, n=n_test, seed=trial_seed + 2
    )

    # Pre-fit prior from a "training" simulation on known bcs only.
    prior_b = fit_training_time_prior(world, n_rows=10000, seed=trial_seed + 3)

    variants = {
        "IDENTITY":          {"global": 0.0, "per_bc": {}},
        "BASELINE":          cal_baseline(subj, bc, ps, ys),
        "PARTIAL_POOL":      cal_partial_pool(subj, bc, ps, ys, prior_b=0.0),
        "ANCHOR_WEIGHTED":   cal_anchor_weighted(world, subj, bc, ps, ys, prior_b=0.0),
        "PREFIT_PRIOR":      cal_partial_pool(subj, bc, ps, ys, prior_b=prior_b),
        "PP_CONSERVATIVE":   cal_partial_pool(
            subj, bc, ps, ys, prior_b=0.0, tau_global=10.0, tau_bc=10.0,
        ),
        "ALL_THREE":         cal_anchor_weighted(world, subj, bc, ps, ys, prior_b=prior_b),
    }

    new_mask = world.new_bc_mask[test_bc]
    out: dict = {}
    for name, cal in variants.items():
        p_post = apply_cal(cal, test_bc, test_ps)
        out[name] = {
            "nll_all": nll(p_post, test_ys),
            "nll_known": nll(p_post[~new_mask], test_ys[~new_mask]),
            "nll_new": nll(p_post[new_mask], test_ys[new_mask]) if new_mask.sum() else float("nan"),
            "b_global": cal["global"],
            "n_per_bc": len(cal["per_bc"]),
        }
    out["_meta"] = {
        "prior_b": prior_b,
        "n_known_test": int((~new_mask).sum()),
        "n_new_test": int(new_mask.sum()),
        "true_b_bc_mu_known": float(world.b_bc_true[:world.M_known].mean()),
        "true_b_bc_mu_new": float(world.b_bc_true[world.M_known:].mean()),
        "n_labels_known": int((~world.new_bc_mask[bc]).sum()),
        "n_labels_new": int(world.new_bc_mask[bc].sum()),
        "anchor_min": float(world.anchor_normalized[subj].min()),
        "anchor_max": float(world.anchor_normalized[subj].max()),
        "anchor_mean": float(world.anchor_normalized[subj].mean()),
    }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_regime(name: str, results: list[dict]) -> None:
    print()
    print("=" * 76)
    print("REGIME:", name)
    print("=" * 76)
    variants = [k for k in results[0] if not k.startswith("_")]
    baseline_all = np.array([r["BASELINE"]["nll_all"] for r in results])
    baseline_new = np.array([r["BASELINE"]["nll_new"] for r in results])
    baseline_known = np.array([r["BASELINE"]["nll_known"] for r in results])

    def fmt(label: str, vals_all, vals_known, vals_new) -> str:
        mean_a = vals_all.mean()
        std_a = vals_all.std()
        mean_k = vals_known.mean()
        mean_n = vals_new.mean()
        win_a = (vals_all < baseline_all - 1e-9).mean()
        win_k = (vals_known < baseline_known - 1e-9).mean()
        win_n = (vals_new < baseline_new - 1e-9).mean()
        return "{:<18s}  all {:.4f}+-{:.4f} (win {:>4.0%})  known {:.4f} (win {:>4.0%})  new {:.4f} (win {:>4.0%})".format(
            label, mean_a, std_a, win_a, mean_k, win_k, mean_n, win_n
        )

    for v in variants:
        vals_all = np.array([r[v]["nll_all"] for r in results])
        vals_known = np.array([r[v]["nll_known"] for r in results])
        vals_new = np.array([r[v]["nll_new"] for r in results])
        print(fmt(v, vals_all, vals_known, vals_new))

    # Improvement (relative to baseline NLL) in nats:
    print()
    print("Mean improvement vs BASELINE (positive = better):")
    for v in variants:
        if v == "BASELINE":
            continue
        d_all = baseline_all - np.array([r[v]["nll_all"] for r in results])
        d_known = baseline_known - np.array([r[v]["nll_known"] for r in results])
        d_new = baseline_new - np.array([r[v]["nll_new"] for r in results])
        print(
            "  {:<18s}  all {:+.4f}+-{:.4f}  known {:+.4f}  new {:+.4f}".format(
                v, d_all.mean(), d_all.std(), d_known.mean(), d_new.mean()
            )
        )


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    N_TRIALS = 50

    regimes = [
        # label, anchor_noise, realistic_head, balanced_acquisition
        ("REALISTIC head,    novelty-only acquisition (production)",  False, True, False),
        ("REALISTIC head,    anchor-noise + novelty-only acquisition", True,  True, False),
        ("REALISTIC head,    BALANCED acquisition (50/50 new/known)",  False, True, True),
        ("UNIFORM miscal,    novelty-only acquisition",                False, False, False),
        ("UNIFORM miscal,    BALANCED acquisition",                    False, False, True),
    ]

    for label, anc, realistic, balanced in regimes:
        results = []
        for t in range(N_TRIALS):
            r = run_one_trial(
                trial_seed=10_000 + t * 31,
                anchor_noise=anc,
                realistic_head=realistic,
                balanced_acquisition=balanced,
            )
            results.append(r)
        summarize_regime(label, results)

        # Diagnostics: label distribution + anchor spread + learned b_global
        meta = [r["_meta"] for r in results]
        n_kn = np.mean([m["n_labels_known"] for m in meta])
        n_nw = np.mean([m["n_labels_new"] for m in meta])
        amin = np.mean([m["anchor_min"] for m in meta])
        amax = np.mean([m["anchor_max"] for m in meta])
        amean = np.mean([m["anchor_mean"] for m in meta])
        prior = np.mean([m["prior_b"] for m in meta])
        bmu_k = np.mean([m["true_b_bc_mu_known"] for m in meta])
        bmu_n = np.mean([m["true_b_bc_mu_new"] for m in meta])
        print()
        print(
            "  diagnostics: labels known/new = {:.1f}/{:.1f}   anchor in labels min/mean/max = {:.2f}/{:.2f}/{:.2f}".format(
                n_kn, n_nw, amin, amean, amax
            )
        )
        print(
            "               training prior b = {:+.3f}   true_mu_b (known/new) = {:+.3f} / {:+.3f}".format(
                prior, bmu_k, bmu_n
            )
        )
        # Show what global state each calibrator learned, average across trials
        print("  learned b_global (mean +/- std, mean #per_bc):")
        for v in ["BASELINE", "PARTIAL_POOL", "PREFIT_PRIOR", "PP_CONSERVATIVE", "ALL_THREE"]:
            gs = np.array([r[v]["b_global"] for r in results])
            ns = np.array([r[v]["n_per_bc"] for r in results])
            print(
                "    {:<18s}  b_global = {:+.3f} +/- {:.3f}   #per_bc = {:.1f}".format(
                    v, gs.mean(), gs.std(), ns.mean()
                )
            )


if __name__ == "__main__":
    main()
