"""Red-team simulation for Tier-1 calibration candidates A, C, D.

Each candidate is compared paired-seed against the SHIPPED PP_CONSERVATIVE
baseline (b_global + delta_bc, tau=20).  We run honest + adversarial
regimes for each idea and report mean diff, 95% CI, win rate, and
paired t-stat.  The decision rule is the same as the per-subject /
kNN red-team:

    Ship iff:
      - Wins (positive lower CI on mean diff) on every honest regime.
      - Does NOT lose significantly (t < -1.96) on any adversarial regime.

Candidates
----------
  IW  Importance-weighted calibration (Park et al. AISTATS 2020):
        weight each labeled row by p_test(bc_i) / p_acq(bc_i), clipped.
        Fit b_global + delta_bc with weights.

  EMB Item-embedding as a regularized linear covariate (Aoyama 2023):
        logit(p_cal) = logit(p) + b_global + delta_bc + w . emb_item
        with ||w||^2 heavily ridge-penalized.

  ISO Compositional Platt + Isotonic (Mix-n-Match, Zhang 2020):
        fit PP_CONSERVATIVE first, then a 5-bin isotonic step on its
        outputs, gated on held-out NLL improvement.

Adversarial test for B (IRT) lives in a separate file because it
requires an IRT world.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sim_calibration_ideas as sim  # noqa: E402
import _sim_acquisition_tuning as tune  # noqa: E402
import _sim_per_subject_redteam as psim  # noqa: E402
import _sim_knn_neighbor_redteam as knn_sim  # noqa: E402

N_TRIALS = 60
N_TEST = 5_000
N_ITEMS = 200
D = 32

TAU_GLOBAL = 20.0
TAU_BC = 20.0


# ---------------------------------------------------------------------------
# A. Importance-weighted calibration
# ---------------------------------------------------------------------------


def estimate_p_acq(world, *, fraction_new_pool: float, w_anchor: float,
                   n_samples: int = 20_000, seed: int = 0):
    """Empirically estimate p_acq(bc) from the dual-pool acquisition.

    Returns an array shape [M] of probabilities summing to 1.  We
    average over many acquisition draws to smooth out the top-K
    extreme-value noise.
    """
    M = world.M
    counts = np.zeros(M, dtype=float)
    rng = np.random.default_rng(seed)
    for t in range(40):  # 40 mini-acquisitions of 75 labels each
        _, bc_t, _, _, _ = tune.acquisition_dual_pool(
            world, fraction_new_pool=fraction_new_pool, w_anchor=w_anchor,
            seed=int(rng.integers(1, 10**9)),
        )
        for b in bc_t:
            counts[int(b)] += 1.0
    return counts / counts.sum()


def cal_type_conditional(subj, bc, ps, ys, *, new_bc_mask,
                          tau_global=TAU_GLOBAL, tau_bc=TAU_BC, tau_type=10.0):
    """1-extra-parameter calibrator capturing a systematic mean shift
    between new-bc and known-bc rows:
        logit(p_cal) = logit(p) + b_global + delta_type * 1{bc is new}
                       + delta_bc[bc]
    with delta_type ridge-shrunk toward 0.

    This is a much safer alternative to importance weighting: instead of
    N noisy per-row weights, it adds ONE parameter that collapses to 0
    when there's no shift to correct (so it doesn't hurt the no-shift
    case) but absorbs the mean offset when there is one.
    """
    bc_arr = np.asarray(bc, dtype=np.int64)
    is_new = new_bc_mask[bc_arr].astype(float)
    zs = sim.logit(ps)
    # Initial b_global (treating delta_type as 0).
    b_global = sim.fit_ridge_intercept(ps, ys, target_b=0.0, ridge=tau_global)
    # Fit delta_type with b_global as fixed offset on all rows; only the
    # is_new rows get the shift.
    # BCE(zs + b_global + delta_type * is_new, y) + tau_type * delta_type^2
    delta_type = 0.0
    for _ in range(80):
        q = sim.sigmoid(zs + b_global + delta_type * is_new)
        g = 2.0 * tau_type * delta_type + float(((q - ys) * is_new).sum())
        h = 2.0 * tau_type + float((q * (1.0 - q) * is_new).sum())
        if h < 1e-9:
            break
        step = g / h
        delta_type -= step
        if abs(step) < 1e-8:
            break
    delta_type = max(-5.0, min(5.0, float(delta_type)))
    # Re-fit b_global with delta_type as offset.
    off_per_row = delta_type * is_new
    b_global = 0.0
    for _ in range(80):
        q = sim.sigmoid(zs + off_per_row + b_global)
        g = 2.0 * tau_global * b_global + float((q - ys).sum())
        h = 2.0 * tau_global + float((q * (1.0 - q)).sum())
        if h < 1e-9:
            break
        step = g / h
        b_global -= step
        if abs(step) < 1e-8:
            break
    b_global = max(-5.0, min(5.0, float(b_global)))
    # Per-bc deltas with b_global + delta_type*is_new as offset.
    delta_bc: dict[int, float] = {}
    for bc_id in np.unique(bc_arr):
        mask = bc_arr == bc_id
        off = b_global + (delta_type if is_new[mask][0] > 0.5 else 0.0)
        delta_bc[int(bc_id)] = sim.fit_ridge_intercept(
            ps[mask], ys[mask], target_b=off, ridge=tau_bc,
        ) - off
    return {
        "b_global": b_global,
        "delta_type": delta_type,
        "delta_bc": delta_bc,
        "delta_subj": {},
    }


def apply_type_conditional(state, subj, bc, ps, new_bc_mask):
    bc_arr = np.asarray(bc, dtype=np.int64)
    is_new = new_bc_mask[bc_arr].astype(float)
    zs = sim.logit(ps)
    delta_table = np.array([state["delta_bc"].get(int(b), 0.0) for b in bc_arr])
    return sim.sigmoid(
        zs + state["b_global"] + state["delta_type"] * is_new + delta_table
    )


def cal_iw_bc(subj, bc, ps, ys, *, p_target, p_acq,
              clip_lo=0.1, clip_hi=10.0, tau_global=TAU_GLOBAL, tau_bc=TAU_BC):
    """Importance-weighted partial-pool calibrator."""
    bc_arr = np.asarray(bc, dtype=np.int64)
    w = p_target[bc_arr] / np.clip(p_acq[bc_arr], 1e-6, None)
    w = np.clip(w, clip_lo, clip_hi)
    w /= w.mean()
    b_global = sim.fit_ridge_intercept(ps, ys, weights=w, target_b=0.0,
                                       ridge=tau_global)
    delta_bc: dict[int, float] = {}
    for bc_id in np.unique(bc_arr):
        mask = bc_arr == bc_id
        b_bc = sim.fit_ridge_intercept(
            ps[mask], ys[mask], weights=w[mask],
            target_b=b_global, ridge=tau_bc,
        )
        delta_bc[int(bc_id)] = b_bc - b_global
    return {"b_global": b_global, "delta_bc": delta_bc, "delta_subj": {}}


# ---------------------------------------------------------------------------
# C. Item-embedding as a regularized linear covariate.
# ---------------------------------------------------------------------------


def cal_emb_covariate(subj, bc, ps, ys, item_ids, item_embs,
                      *, tau_global=TAU_GLOBAL, tau_bc=TAU_BC, tau_w=100.0,
                      disable_w: bool = False):
    """Joint coordinate-descent fit of b_global, delta_bc, w_emb.

    Minimizes
        sum_i BCE(logit(p_i) + b_global + delta_bc[bc_i] + w.emb_i, y_i)
      + tau_global * b_global^2 + tau_bc * sum delta_bc^2 + tau_w * ||w||^2
    """
    bc_arr = np.asarray(bc, dtype=np.int64)
    item_arr = np.asarray(item_ids, dtype=np.int64)
    embs = item_embs[item_arr]  # (n, D)
    zs = sim.logit(ps)
    n, dD = embs.shape

    b_global = 0.0
    delta_bc: dict[int, float] = {bc_id: 0.0 for bc_id in np.unique(bc_arr).tolist()}
    w = np.zeros(dD)

    def offset_per_row():
        bc_table = np.array([delta_bc.get(int(b), 0.0) for b in bc_arr])
        return zs + b_global + bc_table + embs @ w

    for _outer in range(6):
        # Update b_global with current delta_bc + w.emb as offset.
        bc_table = np.array([delta_bc.get(int(b), 0.0) for b in bc_arr])
        off = bc_table + embs @ w
        b_global = sim.fit_ridge_intercept(
            sim.sigmoid(zs + off),  # pseudo-ps already include offset
            ys, target_b=0.0, ridge=tau_global,
        )
        # Update delta_bc per bc with b_global + w.emb as offset.
        for bc_id in delta_bc:
            mask = bc_arr == bc_id
            if not mask.any():
                continue
            off = b_global + embs[mask] @ w
            delta_bc[bc_id] = sim.fit_ridge_intercept(
                sim.sigmoid(zs[mask] + off), ys[mask],
                target_b=0.0, ridge=tau_bc,
            )
        if disable_w:
            continue
        # Update w via Newton step on full ridge logistic with offset.
        bc_table = np.array([delta_bc.get(int(b), 0.0) for b in bc_arr])
        off = b_global + bc_table
        for _inner in range(8):
            z = zs + off + embs @ w
            q = sim.sigmoid(z)
            g = embs.T @ (q - ys) + 2.0 * tau_w * w
            S = q * (1.0 - q)
            H = embs.T @ (embs * S[:, None]) + 2.0 * tau_w * np.eye(dD)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            new_w = w - step
            if not np.all(np.isfinite(new_w)):
                break
            if np.linalg.norm(step) < 1e-7:
                w = new_w
                break
            w = new_w
    return {"b_global": float(b_global), "delta_bc": delta_bc, "w_emb": w}


def apply_emb_covariate(state, subj, bc, ps, item_ids, item_embs):
    bc_arr = np.asarray(bc, dtype=np.int64)
    item_arr = np.asarray(item_ids, dtype=np.int64)
    embs = item_embs[item_arr]
    zs = sim.logit(ps)
    bc_table = np.array([state["delta_bc"].get(int(b), 0.0) for b in bc_arr])
    return sim.sigmoid(zs + state["b_global"] + bc_table + embs @ state["w_emb"])


# ---------------------------------------------------------------------------
# D. Compositional Platt + Isotonic (Mix-n-Match).
# ---------------------------------------------------------------------------


def fit_isotonic(ps, ys):
    """Pool Adjacent Violators on (ps, ys); return the step function."""
    order = np.argsort(ps)
    p_sorted = ps[order]
    y_sorted = ys[order].astype(float)
    n = len(p_sorted)
    if n == 0:
        return np.array([0.0]), np.array([0.5])
    weights = np.ones(n)
    values = y_sorted.copy()
    # Standard PAV.
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-12:
            i += 1
            continue
        tot_w = weights[i] + weights[i + 1]
        tot_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / tot_w
        values[i] = tot_v
        weights[i] = tot_w
        values = np.delete(values, i + 1)
        weights = np.delete(weights, i + 1)
        if i > 0:
            i -= 1
    # Map back to step boundaries.
    boundaries = np.cumsum(weights).astype(int)
    p_steps = np.array([p_sorted[b - 1] for b in boundaries])
    return p_steps, values


def apply_isotonic(steps, vals, ps):
    out = np.empty_like(ps)
    for i, p in enumerate(ps):
        idx = np.searchsorted(steps, p, side="left")
        idx = min(idx, len(vals) - 1)
        out[i] = vals[idx]
    return np.clip(out, sim.EPS, 1.0 - sim.EPS)


def cal_iso_composition(subj, bc, ps, ys, *, tau_global=TAU_GLOBAL,
                        tau_bc=TAU_BC, n_folds=3, min_gain=0.005):
    """PP_CONSERVATIVE + isotonic on top, gated on K-fold held-out NLL.

    Only applies the isotonic step if K-fold held-out NLL improves by
    at least ``min_gain`` nats over the parametric-only baseline.
    """
    state_bc = psim.cal_pp_bc(subj, bc, ps, ys, tau_global=tau_global, tau_bc=tau_bc)
    p_after_bc = psim.apply_cal_subj(state_bc, subj, bc, ps)
    # K-fold held-out check.
    n = len(ps)
    if n < n_folds * 3:
        state_bc["iso_steps"] = None
        state_bc["iso_vals"] = None
        return state_bc
    perm = np.argsort(p_after_bc + np.arange(n) * 1e-9)
    folds = np.array_split(perm, n_folds)
    nll_para = 0.0
    nll_iso = 0.0
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        steps, vals = fit_isotonic(p_after_bc[train_idx], ys[train_idx])
        p_iso = apply_isotonic(steps, vals, p_after_bc[test_idx])
        nll_para += sim.nll(p_after_bc[test_idx], ys[test_idx]) * len(test_idx)
        nll_iso  += sim.nll(p_iso, ys[test_idx]) * len(test_idx)
    nll_para /= n; nll_iso /= n
    if nll_para - nll_iso >= min_gain:
        steps, vals = fit_isotonic(p_after_bc, ys)
        state_bc["iso_steps"] = steps
        state_bc["iso_vals"] = vals
    else:
        state_bc["iso_steps"] = None
        state_bc["iso_vals"] = None
    return state_bc


def apply_iso_composition(state, subj, bc, ps):
    p_after_bc = psim.apply_cal_subj(state, subj, bc, ps)
    if state.get("iso_steps") is None:
        return p_after_bc
    return apply_isotonic(state["iso_steps"], state["iso_vals"], p_after_bc)


# ---------------------------------------------------------------------------
# Trial wrapper.
# ---------------------------------------------------------------------------


def run_trial(
    *,
    seed: int,
    candidate: str,  # "IW", "EMB", "ISO", "JOINT"
    b_subj_sigma: float = 0.0,
    item_signal_sigma: float = 0.0,
    high_frequency: bool = False,
    random_y: bool = False,
    b_bc_sigma_new: float = 1.0,
    b_bc_sigma_known: float = 0.2,
    b_bc_mu_new: float = 0.0,
    b_bc_mu_known: float = 0.0,
    fraction_new_pool: float = 0.95,
    w_anchor: float = 10.0,
    # IW hyperparams:
    iw_clip_lo: float = 0.1,
    iw_clip_hi: float = 10.0,
    # EMB hyperparams:
    tau_w: float = 100.0,
    # ISO hyperparams:
    iso_min_gain: float = 0.005,
    # TYPE hyperparams:
    tau_type: float = 10.0,
):
    world = knn_sim.KnnWorld(
        M_known=12, M_new=3, S=50,
        anchor_noise=False,
        b_bc_mu_known=b_bc_mu_known, b_bc_sigma_known=b_bc_sigma_known,
        b_bc_mu_new=b_bc_mu_new, b_bc_sigma_new=b_bc_sigma_new,
        b_subj_sigma=b_subj_sigma,
        item_signal_sigma=item_signal_sigma,
        observed_emb_noise=0.0,
        emb_correlated_with_signal=True,
        high_frequency=high_frequency,
        random_y=random_y,
        n_items=N_ITEMS, D=D,
        seed=seed,
    )
    rng_lbl = np.random.default_rng(seed + 1)
    subj, bc, _, _, _ = tune.acquisition_dual_pool(
        world, fraction_new_pool=fraction_new_pool, w_anchor=w_anchor, seed=seed + 1,
    )
    n_lbl = len(subj)
    item = rng_lbl.integers(0, N_ITEMS, n_lbl)
    ps, ys = world.gen_rows(subj, item, bc, rng_lbl)
    rng_t = np.random.default_rng(seed + 100)
    t_subj = rng_t.integers(0, world.S, N_TEST)
    t_item = rng_t.integers(0, N_ITEMS, N_TEST)
    t_bc = rng_t.integers(0, world.M, N_TEST)
    t_ps, t_ys = world.gen_rows(t_subj, t_item, t_bc, rng_t)

    # Baseline: PP_CONSERVATIVE (b_global + delta_bc only).
    base = psim.cal_pp_bc(subj, bc, ps, ys)
    p_base = psim.apply_cal_subj(base, t_subj, t_bc, t_ps)
    nll_base = sim.nll(p_base, t_ys)

    if candidate == "IW":
        p_acq = estimate_p_acq(
            world, fraction_new_pool=fraction_new_pool, w_anchor=w_anchor,
            seed=seed + 7,
        )
        p_target = np.full(world.M, 1.0 / world.M)
        state = cal_iw_bc(
            subj, bc, ps, ys, p_target=p_target, p_acq=p_acq,
            clip_lo=iw_clip_lo, clip_hi=iw_clip_hi,
        )
        p_cand = psim.apply_cal_subj(state, t_subj, t_bc, t_ps)
    elif candidate == "EMB":
        state = cal_emb_covariate(
            subj, bc, ps, ys, item, world.item_embs_observed,
            tau_w=tau_w,
        )
        p_cand = apply_emb_covariate(
            state, t_subj, t_bc, t_ps, t_item, world.item_embs_observed,
        )
    elif candidate == "TYPE":
        state = cal_type_conditional(
            subj, bc, ps, ys, new_bc_mask=world.new_bc_mask,
            tau_type=tau_type,
        )
        p_cand = apply_type_conditional(
            state, t_subj, t_bc, t_ps, world.new_bc_mask,
        )
    elif candidate == "JOINT":
        # Attribution test: same coordinate-descent as EMB but with
        # w_emb forced to zero.  If this matches EMB's gain, then the
        # 'embedding' is not actually doing anything -- the win is just
        # from joint vs sequential fitting.
        state = cal_emb_covariate(
            subj, bc, ps, ys, item, world.item_embs_observed,
            tau_w=tau_w, disable_w=True,
        )
        p_cand = apply_emb_covariate(
            state, t_subj, t_bc, t_ps, t_item, world.item_embs_observed,
        )
    elif candidate == "ISO":
        state = cal_iso_composition(subj, bc, ps, ys, min_gain=iso_min_gain)
        p_cand = apply_iso_composition(state, t_subj, t_bc, t_ps)
    else:
        raise ValueError(candidate)

    return nll_base, sim.nll(p_cand, t_ys)


def run_regime(label: str, *, candidate: str, **kwargs) -> dict:
    diffs = []
    nll_a, nll_b = [], []
    for t in range(N_TRIALS):
        a, b = run_trial(seed=300_000 + t * 31, candidate=candidate, **kwargs)
        diffs.append(a - b)
        nll_a.append(a); nll_b.append(b)
    diffs = np.array(diffs)
    m = float(diffs.mean()); s = float(diffs.std(ddof=1))
    se = s / math.sqrt(len(diffs))
    return {
        "label": label,
        "candidate": candidate,
        "mean_diff": m,
        "ci95": (m - 1.96 * se, m + 1.96 * se),
        "win_rate": float((diffs > 0).mean()),
        "t_stat": m / se if se > 0 else 0.0,
        "mean_base": float(np.mean(nll_a)),
        "mean_cand": float(np.mean(nll_b)),
    }


def print_row(r: dict) -> None:
    sig = "***" if abs(r["t_stat"]) > 2.58 else ("** " if abs(r["t_stat"]) > 1.96 else "   ")
    print(
        "[{c:<3s}] {lab:<32s} base={mb:.4f} cand={mc:.4f}  diff={d:+.4f}  "
        "CI=[{lo:+.4f},{hi:+.4f}]  win={wr:.0%}  t={t:+.2f} {sig}".format(
            c=r["candidate"], lab=r["label"],
            mb=r["mean_base"], mc=r["mean_cand"], d=r["mean_diff"],
            lo=r["ci95"][0], hi=r["ci95"][1], wr=r["win_rate"],
            t=r["t_stat"], sig=sig,
        )
    )


# ---------------------------------------------------------------------------
# Regime grid.
# ---------------------------------------------------------------------------


HONEST_REGIMES = [
    ("A1 baseline-ish",          dict(b_bc_sigma_new=1.0, b_bc_sigma_known=0.2)),
    ("A2 big new-bc shift 1.5",  dict(b_bc_sigma_new=1.5, b_bc_sigma_known=0.3)),
    ("A3 small new-bc shift 0.5",dict(b_bc_sigma_new=0.5, b_bc_sigma_known=0.1)),
    ("A4 emb signal 0.4",        dict(b_bc_sigma_new=1.0, item_signal_sigma=0.4)),
    ("A5 emb signal 0.8 + subj", dict(b_bc_sigma_new=1.0, item_signal_sigma=0.8, b_subj_sigma=0.3)),
]

ADV_REGIMES = [
    ("Z1 random labels",         dict(b_bc_sigma_new=1.0, random_y=True)),
    ("Z2 high-freq emb 0.5",     dict(b_bc_sigma_new=1.0, item_signal_sigma=0.5, high_frequency=True)),
    ("Z3 zero bc shift",         dict(b_bc_sigma_new=0.1, b_bc_sigma_known=0.05)),
    ("Z4 extreme acq 99/1",      dict(b_bc_sigma_new=1.0, fraction_new_pool=0.99)),
    ("Z5 mild acq 50/50",        dict(b_bc_sigma_new=1.0, fraction_new_pool=0.5)),
    ("Z6 huge bc shift 2.5",     dict(b_bc_sigma_new=2.5, b_bc_sigma_known=0.5)),
]


def main() -> int:
    print("Tier-1 candidates vs PP_CONSERVATIVE baseline (b_global + delta_bc, tau=20)")
    print("  diff > 0 => candidate wins NLL on uniform test set")
    print("  {} trials per regime, paired seeds, {} test rows".format(N_TRIALS, N_TEST))
    print()
    for cand_label, candidate in [
        ("Idea A: importance-weighted calibration", "IW"),
        ("Idea D: PP_CONSERVATIVE + isotonic composition", "ISO"),
        ("Idea C: embedding-as-covariate (tau_w=100)", "EMB"),
    ]:
        print("=" * 78)
        print(cand_label)
        print("=" * 78)
        print("  HONEST regimes (gain must be > 0 with positive lower CI):")
        for label, kwargs in HONEST_REGIMES:
            print_row(run_regime(label, candidate=candidate, **kwargs))
        print("  ADVERSARIAL regimes (must NOT lose with t < -1.96):")
        for label, kwargs in ADV_REGIMES:
            print_row(run_regime(label, candidate=candidate, **kwargs))
        print()

    print("=" * 78)
    print("Hyperparameter sweep on the most-favorable honest regime per candidate")
    print("=" * 78)
    print("[IW] clip range sweep on A2 (big new-bc shift):")
    for clip in [(0.25, 4.0), (0.1, 10.0), (0.05, 20.0), (0.02, 50.0)]:
        r = run_regime("  clip={:.2f},{:.1f}".format(*clip), candidate="IW",
                       b_bc_sigma_new=1.5, b_bc_sigma_known=0.3,
                       iw_clip_lo=clip[0], iw_clip_hi=clip[1])
        print_row(r)
    print("[EMB] tau_w sweep on A5 (emb signal 0.8 + subj):")
    for tw in [30.0, 100.0, 300.0, 1000.0]:
        r = run_regime("  tau_w={:>5.0f}".format(tw), candidate="EMB",
                       b_bc_sigma_new=1.0, item_signal_sigma=0.8,
                       b_subj_sigma=0.3, tau_w=tw)
        print_row(r)
    print("[ISO] min_gain gate sweep on A1 (baseline-ish):")
    for mg in [0.001, 0.005, 0.020, 0.050]:
        r = run_regime("  min_gain={:.3f}".format(mg), candidate="ISO",
                       b_bc_sigma_new=1.0, b_bc_sigma_known=0.2,
                       iso_min_gain=mg)
        print_row(r)

    print()
    print("=" * 78)
    print("FOLLOWUP: IW with SYSTEMATIC mean shifts (fairer test for IW)")
    print("=" * 78)
    print("  Regime where new bcs have mean -1.0 and known bcs have mean +0.5;")
    print("  acquisition oversamples new => raw b_global is biased toward -1.0.")
    print("  IW should correct this; baseline cannot.")
    for label, kwargs in [
        ("F1 mu_new=-1, mu_known=+.5", dict(b_bc_mu_new=-1.0, b_bc_mu_known=0.5,
                                            b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F2 mu_new=+1, mu_known=-.5", dict(b_bc_mu_new=1.0, b_bc_mu_known=-0.5,
                                            b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F3 mu_new=0, mu_known=0 (ctrl)", dict(b_bc_mu_new=0.0, b_bc_mu_known=0.0,
                                                b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
    ]:
        print_row(run_regime(label, candidate="IW", **kwargs))

    print()
    print("=" * 78)
    print("Idea A': TYPE-conditional intercept (cheap, safer alternative to IW)")
    print("=" * 78)
    print("  Adds 1 parameter delta_type shrunk to 0 with ridge tau_type=10.")
    print("  Should match IW on systematic-shift regimes, beat IW elsewhere.")
    print("  HONEST regimes:")
    for label, kwargs in HONEST_REGIMES:
        print_row(run_regime(label, candidate="TYPE", **kwargs))
    print("  ADVERSARIAL regimes:")
    for label, kwargs in ADV_REGIMES:
        print_row(run_regime(label, candidate="TYPE", **kwargs))
    print("  SYSTEMATIC-SHIFT regimes:")
    for label, kwargs in [
        ("F1 mu_new=-1, mu_known=+.5",  dict(b_bc_mu_new=-1.0, b_bc_mu_known=0.5,
                                             b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F2 mu_new=+1, mu_known=-.5",  dict(b_bc_mu_new=1.0, b_bc_mu_known=-0.5,
                                             b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F3 mu_new=0, mu_known=0",     dict(b_bc_mu_new=0.0, b_bc_mu_known=0.0,
                                             b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F4 small shift -.3/+.15",     dict(b_bc_mu_new=-0.3, b_bc_mu_known=0.15,
                                             b_bc_sigma_new=0.5, b_bc_sigma_known=0.3)),
    ]:
        print_row(run_regime(label, candidate="TYPE", **kwargs))

    print()
    print("=" * 78)
    print("ATTRIBUTION: JOINT (coord descent with w forced to 0) vs baseline")
    print("=" * 78)
    print("  If JOINT matches EMB's win, the 'embedding' adds nothing -- the gain")
    print("  is just from joint vs sequential bc fitting.  Then EMB is overkill.")
    for label, kwargs in [
        ("A1 baseline-ish",         dict(b_bc_sigma_new=1.0, b_bc_sigma_known=0.2)),
        ("A4 emb signal 0.4",       dict(b_bc_sigma_new=1.0, item_signal_sigma=0.4)),
        ("A5 emb signal 0.8 + subj",dict(b_bc_sigma_new=1.0, item_signal_sigma=0.8, b_subj_sigma=0.3)),
        ("Z1 random labels",        dict(b_bc_sigma_new=1.0, random_y=True)),
    ]:
        print_row(run_regime(label, candidate="JOINT", **kwargs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
