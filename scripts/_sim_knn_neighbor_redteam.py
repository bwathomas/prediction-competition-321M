"""Red-team simulation for the kNN embedding-neighbor channel.

Test whether a kernel-weighted residual correction
    Delta_z_i = eta * sum_{j in L_s} K(emb_i, emb_j) * r_j / (sum K + tau)
on top of the shipped b_global + delta_bc calibrator improves held-out
NLL across honest and adversarial regimes.

Honest regimes:
    A  no item-level structure (f_s == 0)           -> NOOP expected
    B  weak linear item signal (sigma=0.2)          -> small win
    C  strong linear item signal (sigma=0.6)        -> big win

Adversarial regimes:
    D  random labels                                 -> must NOT degrade
    E  embeddings UNCORRELATED with true item signal -> must NOT degrade
       (subject weights live in a hidden subspace not visible to the kernel)
    F  few labels per subject (subj_only=5, others=0) -> coverage hit
    G  embedding noise (true emb + noise at apply)   -> kernel sees wrong
                                                       neighbors
    H  non-smooth item signal (high-frequency)       -> kernel oversmooths
    I  per-bc + per-subj biases ALSO active          -> kNN must compose
                                                       with the existing
                                                       calibrator

For each regime: 60 trials, paired seeds, uniform-random uniform-bc
uniform-item uniform-subject test set of 5000 rows.  Report mean diff,
95% CI, win rate, t-stat.

Decision rule: ship only if EVERY honest regime shows mean diff >= 0
with a positive lower CI, AND no adversarial regime shows a
statistically significant DEGRADATION (mean diff > 0 with t < -1.96).
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

N_TRIALS = 60
N_TEST = 5_000
N_ITEMS = 200
D = 32


class KnnWorld(psim.SubjWorld):
    """Adds item embeddings + a per-subject linear function in emb space.

    True bias on row (s, i, k) is:
        b_global + b_bc[k] + b_subj[s] + (w_s . emb[i]) + interaction
    """

    def __init__(
        self,
        *,
        n_items: int = N_ITEMS,
        D: int = D,
        item_signal_sigma: float = 0.0,
        observed_emb_noise: float = 0.0,
        emb_correlated_with_signal: bool = True,
        high_frequency: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_items = n_items
        self.D = D
        rng = self.rng
        embs = rng.normal(0.0, 1.0, (n_items, D))
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        self.item_embs_true = embs
        if observed_emb_noise > 0.0:
            noise = rng.normal(0.0, observed_emb_noise, (n_items, D))
            obs = embs + noise
            obs /= np.linalg.norm(obs, axis=1, keepdims=True) + 1e-12
            self.item_embs_observed = obs
        else:
            self.item_embs_observed = embs
        # Per-subject weights for the linear bias function in emb space.
        if emb_correlated_with_signal:
            self.subj_weights = rng.normal(0.0, item_signal_sigma, (self.S, D))
        else:
            # Signal lives in a HIDDEN subspace not aligned with the
            # observed embedding -- the kernel sees random directions.
            self.subj_weights = rng.normal(0.0, item_signal_sigma, (self.S, D))
            # Project out alignment with the embeddings to be extra mean.
            # (No-op for random Gaussian weights; for adversarial we'd
            # need an entirely separate hidden basis.  We instead inject
            # uncorrelated-bias-as-noise via the high_frequency knob.)
        self.high_frequency = high_frequency
        if high_frequency:
            # Non-smooth bias: per-item independent N(0, sigma) instead
            # of a smooth function of emb.  Kernel cannot interpolate.
            self.item_random_bias_per_subj = rng.normal(
                0.0, item_signal_sigma, (self.S, n_items)
            )
        else:
            self.item_random_bias_per_subj = None

    def gen_rows(self, subj_ids, item_ids, bc_ids, rng):
        n = len(subj_ids)
        z_true = rng.normal(0.0, 1.0, n)
        b_bc = self.b_bc_true[bc_ids]
        b_subj = self.b_subj_true[subj_ids]
        if self.high_frequency:
            b_item = self.item_random_bias_per_subj[subj_ids, item_ids]
        else:
            b_item = (self.subj_weights[subj_ids] * self.item_embs_true[item_ids]).sum(axis=1)
        p_uncal = sim.sigmoid(z_true)
        if self.random_y:
            y = (rng.random(n) < 0.5).astype(float)
        else:
            true_p = sim.sigmoid(z_true + b_bc + b_subj + b_item)
            y = rng.binomial(1, true_p).astype(float)
        return p_uncal, y


# ---------------------------------------------------------------------------
# kNN corrector
# ---------------------------------------------------------------------------


def fit_knn(
    *,
    state_bc,
    subj_ids,
    item_ids,
    bc_ids,
    ps,
    ys,
    item_embs,
    y_smooth_eps: float = 0.05,
    return_prob_residuals: bool = False,
) -> dict:
    """Build subj -> (item_ids, residuals_after_bc) from the labeled set.

    residual_after_bc = logit(y_smoothed) - logit(p_after_bc).
    """
    b_global = state_bc["b_global"]
    delta_bc = state_bc["delta_bc"]
    zs = sim.logit(ps)
    n = len(zs)
    bc_table = np.zeros(int(max([int(b) for b in bc_ids]) + 1 if n else 0) + 1)
    for k, v in delta_bc.items():
        if k >= len(bc_table):
            bc_table = np.pad(bc_table, (0, k + 1 - len(bc_table)))
        bc_table[k] = v
    bc_shifts = bc_table[np.asarray(bc_ids, dtype=np.int64)]
    z_after = zs + b_global + bc_shifts
    p_after = sim.sigmoid(z_after)
    y_smoothed = np.clip(ys, y_smooth_eps, 1.0 - y_smooth_eps)
    z_y = sim.logit(y_smoothed)
    resids_logit = z_y - z_after
    resids_prob = np.asarray(ys, dtype=float) - p_after
    by_subj: dict[int, list] = {}
    for i, s in enumerate(subj_ids):
        by_subj.setdefault(int(s), []).append(
            (int(item_ids[i]), float(resids_logit[i]), float(resids_prob[i]))
        )
    out_logit, out_prob = {}, {}
    for s, lst in by_subj.items():
        items = np.array([t[0] for t in lst], dtype=np.int64)
        rs_l = np.array([t[1] for t in lst], dtype=float)
        rs_p = np.array([t[2] for t in lst], dtype=float)
        out_logit[s] = (items, rs_l)
        out_prob[s] = (items, rs_p)
    if return_prob_residuals:
        return out_logit, out_prob
    return out_logit


def apply_knn(
    *,
    state_bc,
    test_subj,
    test_item,
    test_bc,
    test_ps,
    item_embs,
    by_subj_knn,
    eta: float = 1.0,
    tau: float = 2.0,
    k_temp: float = 0.2,
    space: str = "logit",  # "logit" or "prob"
    by_subj_knn_prob=None,  # probability-space residuals if space=="prob"
) -> np.ndarray:
    n = len(test_ps)
    zs = sim.logit(test_ps)
    b_global = state_bc["b_global"]
    delta_bc = state_bc["delta_bc"]
    bc_arr = np.asarray(test_bc, dtype=np.int64)
    if delta_bc:
        max_bc = int(max(bc_arr.max(), max(delta_bc.keys()))) + 1
        bc_table = np.zeros(max_bc)
        for k, v in delta_bc.items():
            bc_table[k] = v
        bc_shifts = bc_table[bc_arr]
    else:
        bc_shifts = np.zeros(n)
    z_out = zs + b_global + bc_shifts
    if space == "logit":
        subj_arr = np.asarray(test_subj, dtype=np.int64)
        item_arr = np.asarray(test_item, dtype=np.int64)
        for s_id in np.unique(subj_arr):
            if int(s_id) not in by_subj_knn:
                continue
            lbl_items, lbl_resids = by_subj_knn[int(s_id)]
            if len(lbl_items) == 0:
                continue
            mask = subj_arr == s_id
            emb_t = item_embs[item_arr[mask]]
            emb_l = item_embs[lbl_items]
            sims = emb_t @ emb_l.T
            weights = np.exp(sims / k_temp)
            num = (weights * lbl_resids[None, :]).sum(axis=1)
            denom = weights.sum(axis=1) + tau
            z_out[mask] += eta * num / denom
        return sim.sigmoid(z_out)
    # Probability-scale correction.
    p_out = sim.sigmoid(z_out)
    if by_subj_knn_prob is None:
        by_subj_knn_prob = by_subj_knn  # caller mistake but don't crash
    subj_arr = np.asarray(test_subj, dtype=np.int64)
    item_arr = np.asarray(test_item, dtype=np.int64)
    for s_id in np.unique(subj_arr):
        if int(s_id) not in by_subj_knn_prob:
            continue
        lbl_items, lbl_resids_p = by_subj_knn_prob[int(s_id)]
        if len(lbl_items) == 0:
            continue
        mask = subj_arr == s_id
        emb_t = item_embs[item_arr[mask]]
        emb_l = item_embs[lbl_items]
        sims = emb_t @ emb_l.T
        weights = np.exp(sims / k_temp)
        num = (weights * lbl_resids_p[None, :]).sum(axis=1)
        denom = weights.sum(axis=1) + tau
        p_out[mask] = np.clip(p_out[mask] + eta * num / denom, sim.EPS, 1.0 - sim.EPS)
    return p_out


# ---------------------------------------------------------------------------
# One-trial comparison
# ---------------------------------------------------------------------------


def run_trial(
    *,
    seed: int,
    item_signal_sigma: float = 0.0,
    b_subj_sigma: float = 0.0,
    observed_emb_noise: float = 0.0,
    emb_correlated_with_signal: bool = True,
    high_frequency: bool = False,
    random_y: bool = False,
    labels_per_subj_cap: int | None = None,
    eta: float = 1.0,
    tau: float = 2.0,
    k_temp: float = 0.2,
    space: str = "logit",
    y_smooth_eps: float = 0.05,
    item_assignment: str = "random",
):
    world = KnnWorld(
        M_known=12, M_new=3, S=50,
        anchor_noise=False,
        b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
        b_bc_mu_new=0.0, b_bc_sigma_new=1.0,
        b_subj_sigma=b_subj_sigma,
        item_signal_sigma=item_signal_sigma,
        observed_emb_noise=observed_emb_noise,
        emb_correlated_with_signal=emb_correlated_with_signal,
        high_frequency=high_frequency,
        random_y=random_y,
        n_items=N_ITEMS, D=D,
        seed=seed,
    )
    rng_lbl = np.random.default_rng(seed + 1)
    # Production acquisition over (subj, bc); add random item per row.
    subj, bc, _ps_uncal, _ys, _ = tune.acquisition_dual_pool(
        world, fraction_new_pool=0.95, w_anchor=10.0, seed=seed + 1,
    )
    n_lbl = len(subj)
    if item_assignment == "random":
        item = rng_lbl.integers(0, N_ITEMS, n_lbl)
    elif item_assignment == "concentrate":
        # Force labels to cluster on first 30 items so kNN has dense
        # local coverage for those items.  This is the BEST possible
        # case for kNN -- many neighbors in the active subspace.
        item = rng_lbl.integers(0, 30, n_lbl)
    else:
        raise ValueError("item_assignment must be 'random' or 'concentrate'")
    if labels_per_subj_cap is not None:
        # Cap to N labels per anchored subject + drop overflow.
        by_s = {}
        keep_idx = []
        for i, s in enumerate(subj):
            by_s.setdefault(int(s), 0)
            if by_s[int(s)] < labels_per_subj_cap:
                keep_idx.append(i)
                by_s[int(s)] += 1
        subj = subj[keep_idx]; bc = bc[keep_idx]; item = item[keep_idx]
        if len(subj) == 0:
            subj = np.array([0]); bc = np.array([0]); item = np.array([0])
    ps, ys = world.gen_rows(subj, item, bc, rng_lbl)
    # Test set: uniform over subj, item, bc.
    rng_t = np.random.default_rng(seed + 100)
    t_subj = rng_t.integers(0, world.S, N_TEST)
    t_item = rng_t.integers(0, N_ITEMS, N_TEST)
    t_bc = rng_t.integers(0, world.M, N_TEST)
    t_ps, t_ys = world.gen_rows(t_subj, t_item, t_bc, rng_t)

    state_bc = psim.cal_pp_bc(subj, bc, ps, ys)
    knn_logit, knn_prob = fit_knn(
        state_bc=state_bc,
        subj_ids=subj, item_ids=item, bc_ids=bc, ps=ps, ys=ys,
        item_embs=world.item_embs_observed,
        y_smooth_eps=y_smooth_eps,
        return_prob_residuals=True,
    )
    p_bc_only = psim.apply_cal_subj(state_bc, t_subj, t_bc, t_ps)
    p_bc_knn = apply_knn(
        state_bc=state_bc,
        test_subj=t_subj, test_item=t_item, test_bc=t_bc, test_ps=t_ps,
        item_embs=world.item_embs_observed,
        by_subj_knn=knn_logit,
        by_subj_knn_prob=knn_prob,
        eta=eta, tau=tau, k_temp=k_temp, space=space,
    )
    return sim.nll(p_bc_only, t_ys), sim.nll(p_bc_knn, t_ys)


def run_regime(label: str, **kwargs) -> dict:
    diffs = []
    nll_a = []; nll_b = []
    for t in range(N_TRIALS):
        a, b = run_trial(seed=200_000 + t * 31, **kwargs)
        diffs.append(a - b)  # > 0 means kNN wins
        nll_a.append(a); nll_b.append(b)
    diffs = np.array(diffs)
    m = float(diffs.mean()); s = float(diffs.std(ddof=1))
    se = s / math.sqrt(len(diffs))
    return {
        "label": label,
        "mean_diff": m,
        "ci95": (m - 1.96 * se, m + 1.96 * se),
        "win_rate": float((diffs > 0).mean()),
        "t_stat": m / se if se > 0 else 0.0,
        "mean_bc": float(np.mean(nll_a)),
        "mean_knn": float(np.mean(nll_b)),
    }


def print_row(r: dict) -> None:
    sig = "***" if abs(r["t_stat"]) > 2.58 else ("** " if abs(r["t_stat"]) > 1.96 else "   ")
    print(
        "{lab:<32s} NLL_bc={mb:.4f} NLL_knn={mk:.4f}  diff={d:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  win={wr:.0%}  t={t:+.2f} {sig}".format(
            lab=r["label"], mb=r["mean_bc"], mk=r["mean_knn"], d=r["mean_diff"],
            lo=r["ci95"][0], hi=r["ci95"][1], wr=r["win_rate"], t=r["t_stat"], sig=sig,
        )
    )


def main() -> int:
    print("Red-team: kNN-neighbor channel vs SHIPPED partial-pool (b_global + delta_bc)")
    print("  diff > 0 => kNN wins on uniform test set NLL")
    print("  {} trials per regime, paired seeds, {} test rows".format(N_TRIALS, N_TEST))
    print("  defaults: eta=1.0, tau=2.0, k_temp=0.2, D={}, n_items={}".format(D, N_ITEMS))
    print()
    print("HONEST regimes (gain must be >= 0 with positive lower CI):")
    for label, kwargs in [
        ("A no item signal",            dict(item_signal_sigma=0.0)),
        ("B weak signal sigma=0.2",     dict(item_signal_sigma=0.2)),
        ("C strong signal sigma=0.6",   dict(item_signal_sigma=0.6)),
    ]:
        print_row(run_regime(label, **kwargs))
    print()
    print("ADVERSARIAL regimes (must NOT degrade with t < -1.96):")
    for label, kwargs in [
        ("D random labels",                dict(item_signal_sigma=0.0, random_y=True)),
        ("E high-freq non-smooth 0.5",     dict(item_signal_sigma=0.5, high_frequency=True)),
        ("F cap 2 labels/subj sig=0.4",    dict(item_signal_sigma=0.4, labels_per_subj_cap=2)),
        ("G emb noise 0.4 sig=0.4",        dict(item_signal_sigma=0.4, observed_emb_noise=0.4)),
        ("H sig=0.4 + subj 0.3 + bc",      dict(item_signal_sigma=0.4, b_subj_sigma=0.3)),
        ("I sig=0 + subj 0.3 (subj-only)", dict(item_signal_sigma=0.0, b_subj_sigma=0.3)),
    ]:
        print_row(run_regime(label, **kwargs))
    print()
    print("PROB-space variant (more conservative, bounded delta):")
    for label, kwargs in [
        ("PA no item signal",          dict(item_signal_sigma=0.0)),
        ("PB weak signal sigma=0.2",   dict(item_signal_sigma=0.2)),
        ("PC strong signal sigma=0.6", dict(item_signal_sigma=0.6)),
    ]:
        print_row(run_regime(label, space="prob", eta=0.3, tau=10.0, k_temp=0.3, **kwargs))
    print()
    print("BEST-CASE: concentrated item assignment, strong signal:")
    for label, kwargs in [
        ("BC1 prob eta=0.3 tau=10", dict(space="prob", eta=0.3, tau=10.0, k_temp=0.3)),
        ("BC2 prob eta=0.1 tau=10", dict(space="prob", eta=0.1, tau=10.0, k_temp=0.3)),
        ("BC3 logit eta=0.5 tau=20",dict(space="logit", eta=0.5, tau=20.0, k_temp=0.3)),
    ]:
        print_row(run_regime(
            label, item_signal_sigma=0.8, item_assignment="concentrate", **kwargs
        ))
    print()
    print("Hyperparam sweep on regime B (item signal 0.2), prob space:")
    for k_temp in [0.1, 0.2, 0.4]:
        for tau in [5.0, 20.0, 50.0]:
            r = run_regime(
                "  k_temp={:.2f} tau={:>4.0f}".format(k_temp, tau),
                item_signal_sigma=0.2, k_temp=k_temp, tau=tau, space="prob", eta=0.3,
            )
            print_row(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
