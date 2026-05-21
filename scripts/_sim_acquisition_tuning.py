"""Sweep acquisition multipliers + calibrator ridge to pick production values.

Reuses the world model from _sim_calibration_ideas.py.  Adds:

  * acquisition_v2(): platform-faithful per-category stratification
    (15 categories x 17000 candidates each x top-5), with tunable
    (w_novelty, w_anchor) multipliers.

  * Sweep 1: w_novelty in {0, 0.3, 0.5, 1, 2, 5, 1000} with w_anchor=10
             and the PP_CONSERVATIVE calibrator (ridge=10, no gate),
             reporting new/known fraction in acquired labels AND
             downstream held-out NLL.

  * Sweep 2: with w_novelty fixed at the empirical optimum from sweep 1,
             sweep tau_global and tau_bc in {2, 5, 10, 20, 50, 100} to
             find the ridge sweet spot.

We run both sweeps on the REALISTIC-HEAD regime (known bcs nearly
calibrated, new bcs broadly miscalibrated) because that's the actual
deployment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sim_calibration_ideas as sim  # noqa: E402

N_TRIALS = 50
N_CATEGORIES = 15
N_PER_CATEGORY = 17000  # 5000 items / 15 cats * 51 subjects
K_PER_CATEGORY = 5      # platform default


def acquisition_v2(
    world: sim.World,
    *,
    w_novelty: float,
    w_anchor: float = 10.0,
    seed: int,
):
    """Platform-faithful per-category top-K acquisition with tunable score."""
    rng = np.random.default_rng(seed)
    bc_novelty_table = 1.0 / np.sqrt(1.0 + world.n_train_per_bc)
    chosen_subj = []
    chosen_bc = []
    for _ in range(N_CATEGORIES):
        cand_subj = rng.integers(0, world.S, N_PER_CATEGORY)
        cand_bc = rng.integers(0, world.M, N_PER_CATEGORY)
        nov = bc_novelty_table[cand_bc]
        anch = world.anchor_normalized[cand_subj]
        tb = rng.uniform(-0.5, 0.5, N_PER_CATEGORY)
        scores = w_novelty * nov + w_anchor * anch + tb
        top = np.argpartition(-scores, K_PER_CATEGORY)[:K_PER_CATEGORY]
        chosen_subj.extend(cand_subj[top])
        chosen_bc.extend(cand_bc[top])
    subj = np.array(chosen_subj)
    bc = np.array(chosen_bc)
    p_uncal, y, true_p = world._gen_uncal_and_label(subj, bc, rng)
    return subj, bc, p_uncal, y, true_p


def acquisition_dual_pool(
    world: sim.World,
    *,
    fraction_new_pool: float,
    w_anchor: float = 10.0,
    seed: int,
):
    """Two-pool stratification (a hash-style split per item, no novelty score).

    Pool A (size fraction_new_pool of items): only candidates with new bcs score
    high; known-bc rows in this pool get -inf.
    Pool B (the rest): only candidates with known bcs score high; new-bc rows in
    this pool get -inf.

    The platform's top-K-per-category then picks a mix.  The fraction of new bc
    in the final acquired set is controlled by fraction_new_pool.
    """
    rng = np.random.default_rng(seed)
    chosen_subj = []
    chosen_bc = []
    new_mask = world.new_bc_mask
    for _ in range(N_CATEGORIES):
        cand_subj = rng.integers(0, world.S, N_PER_CATEGORY)
        cand_bc = rng.integers(0, world.M, N_PER_CATEGORY)
        anch = world.anchor_normalized[cand_subj]
        tb = rng.uniform(-0.5, 0.5, N_PER_CATEGORY)
        item_hash = rng.random(N_PER_CATEGORY)  # surrogate for hash(item_content)
        in_new_pool = item_hash < fraction_new_pool

        is_new_bc = new_mask[cand_bc]
        # Pool A: prefers new bc. Pool B: prefers known bc.
        accept_new_pool_A = in_new_pool & is_new_bc
        accept_known_pool_B = (~in_new_pool) & (~is_new_bc)
        eligible = accept_new_pool_A | accept_known_pool_B
        scores = np.where(eligible, w_anchor * anch + tb, -1e9)

        top = np.argpartition(-scores, K_PER_CATEGORY)[:K_PER_CATEGORY]
        chosen_subj.extend(cand_subj[top])
        chosen_bc.extend(cand_bc[top])
    subj = np.array(chosen_subj)
    bc = np.array(chosen_bc)
    p_uncal, y, true_p = world._gen_uncal_and_label(subj, bc, rng)
    return subj, bc, p_uncal, y, true_p


def run_one(
    *,
    w_novelty: float,
    w_anchor: float = 10.0,
    tau_global: float = 10.0,
    tau_bc: float = 10.0,
    realistic: bool = True,
    anchor_noise: bool = False,
    trial_seed: int,
):
    """One trial: build world, acquire, fit PP_CONSERVATIVE, eval."""
    if realistic:
        world = sim.World(
            M_known=12, M_new=3, S=50,
            anchor_noise=anchor_noise,
            b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
            b_bc_mu_new=0.0,   b_bc_sigma_new=1.0,
            seed=trial_seed,
        )
    else:
        world = sim.World(
            M_known=12, M_new=3, S=50,
            anchor_noise=anchor_noise,
            b_bc_mu_known=-0.5, b_bc_sigma_known=0.8,
            b_bc_mu_new=-0.5,   b_bc_sigma_new=0.8,
            seed=trial_seed,
        )
    subj, bc, ps, ys, _ = acquisition_v2(
        world, w_novelty=w_novelty, w_anchor=w_anchor, seed=trial_seed + 1
    )
    _, t_bc, t_ps, t_ys, _ = sim.uniform_test_set(world, n=10000, seed=trial_seed + 2)
    new_mask_test = world.new_bc_mask[t_bc]
    new_frac_acq = float(world.new_bc_mask[bc].mean())

    # PP_CONSERVATIVE: partial pool with high ridge, no gate.
    cal = sim.cal_partial_pool(
        subj, bc, ps, ys, prior_b=0.0, tau_global=tau_global, tau_bc=tau_bc
    )
    p_post = sim.apply_cal(cal, t_bc, t_ps)
    nll_all = sim.nll(p_post, t_ys)
    nll_known = sim.nll(p_post[~new_mask_test], t_ys[~new_mask_test])
    nll_new = sim.nll(p_post[new_mask_test], t_ys[new_mask_test])
    return {
        "new_frac_acq": new_frac_acq,
        "nll_all": nll_all,
        "nll_known": nll_known,
        "nll_new": nll_new,
        "b_global": cal["global"],
        "n_per_bc": len(cal["per_bc"]),
    }


# ---------------------------------------------------------------------------
# Sweep 1: w_novelty
# ---------------------------------------------------------------------------


def sweep_w_novelty(w_novelty_grid, w_anchor=10.0, tau=10.0):
    print("=" * 80)
    print("Sweep 1: acquisition w_novelty (with w_anchor={}, tau={})".format(w_anchor, tau))
    print("=" * 80)
    print("{:>10s}  {:>12s}  {:>12s}  {:>10s}  {:>10s}  {:>10s}".format(
        "w_novelty", "new_frac", "NLL_all", "NLL_known", "NLL_new", "b_global"
    ))
    rows = []
    for w_n in w_novelty_grid:
        ms = {"new_frac_acq": [], "nll_all": [], "nll_known": [], "nll_new": [], "b_global": []}
        for t in range(N_TRIALS):
            r = run_one(
                w_novelty=w_n, w_anchor=w_anchor,
                tau_global=tau, tau_bc=tau,
                realistic=True, anchor_noise=False,
                trial_seed=10_000 + t * 31,
            )
            for k in ms:
                ms[k].append(r[k])
        agg = {k: float(np.mean(v)) for k, v in ms.items()}
        std_all = float(np.std(ms["nll_all"]))
        print("{:>10.2f}  {:>12.2%}  {:>6.4f}+-{:.4f}  {:>10.4f}  {:>10.4f}  {:>+10.4f}".format(
            w_n, agg["new_frac_acq"], agg["nll_all"], std_all,
            agg["nll_known"], agg["nll_new"], agg["b_global"]
        ))
        rows.append((w_n, agg))
    return rows


# ---------------------------------------------------------------------------
# Sweep 2: tau ridge (with w_novelty fixed at best from sweep 1)
# ---------------------------------------------------------------------------


def sweep_tau(tau_grid, w_novelty, w_anchor=10.0):
    print()
    print("=" * 80)
    print("Sweep 2: calibrator tau (with w_novelty={}, w_anchor={})".format(w_novelty, w_anchor))
    print("=" * 80)
    print("{:>10s}  {:>12s}  {:>12s}  {:>10s}  {:>10s}".format(
        "tau", "new_frac", "NLL_all", "NLL_known", "NLL_new"
    ))
    rows = []
    for tau in tau_grid:
        ms = {"new_frac_acq": [], "nll_all": [], "nll_known": [], "nll_new": []}
        for t in range(N_TRIALS):
            r = run_one(
                w_novelty=w_novelty, w_anchor=w_anchor,
                tau_global=tau, tau_bc=tau,
                realistic=True, anchor_noise=False,
                trial_seed=10_000 + t * 31,
            )
            for k in ms:
                ms[k].append(r[k])
        agg = {k: float(np.mean(v)) for k, v in ms.items()}
        std_all = float(np.std(ms["nll_all"]))
        print("{:>10.2f}  {:>12.2%}  {:>6.4f}+-{:.4f}  {:>10.4f}  {:>10.4f}".format(
            tau, agg["new_frac_acq"], agg["nll_all"], std_all,
            agg["nll_known"], agg["nll_new"]
        ))
        rows.append((tau, agg))
    return rows


# ---------------------------------------------------------------------------
# Joint sweep: (w_novelty x tau) -- find the global optimum
# ---------------------------------------------------------------------------


def joint_sweep(w_novelty_grid, tau_grid, w_anchor=10.0):
    print()
    print("=" * 80)
    print("Joint sweep: NLL_all heatmap (rows = tau, cols = w_novelty, w_anchor={})".format(w_anchor))
    print("=" * 80)
    header = "{:>8s}".format("tau\\w_n") + "".join("{:>8.2f}".format(w) for w in w_novelty_grid)
    print(header)
    best = (math.inf, None)
    grid = {}
    for tau in tau_grid:
        row = []
        for w_n in w_novelty_grid:
            nlls = []
            for t in range(N_TRIALS):
                r = run_one(
                    w_novelty=w_n, w_anchor=w_anchor,
                    tau_global=tau, tau_bc=tau,
                    realistic=True, anchor_noise=False,
                    trial_seed=10_000 + t * 31,
                )
                nlls.append(r["nll_all"])
            m = float(np.mean(nlls))
            row.append(m)
            grid[(tau, w_n)] = m
            if m < best[0]:
                best = (m, (tau, w_n))
        print("{:>8.2f}".format(tau) + "".join("{:>8.4f}".format(v) for v in row))
    print()
    print("Best (NLL_all): {:.4f} at tau={}, w_novelty={}".format(best[0], best[1][0], best[1][1]))
    return best, grid


# ---------------------------------------------------------------------------
# Baseline reference: report what BASELINE / PRODUCTION ACQUISITION gives
# under the same world distribution.  This is the bar we're trying to clear.
# ---------------------------------------------------------------------------


def baseline_reference(*, w_novelty: float = 1000.0):
    print()
    print("=" * 80)
    print("Baseline reference (production acquisition, original gated calibrator)")
    print("=" * 80)
    nlls = []
    for t in range(N_TRIALS):
        world = sim.World(
            M_known=12, M_new=3, S=50,
            anchor_noise=False,
            b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
            b_bc_mu_new=0.0, b_bc_sigma_new=1.0,
            seed=10_000 + t * 31,
        )
        subj, bc, ps, ys, _ = acquisition_v2(
            world, w_novelty=w_novelty, w_anchor=10.0, seed=10_000 + t * 31 + 1
        )
        _, t_bc, t_ps, t_ys, _ = sim.uniform_test_set(world, n=10000, seed=10_000 + t * 31 + 2)
        cal = sim.cal_baseline(subj, bc, ps, ys)
        p_post = sim.apply_cal(cal, t_bc, t_ps)
        nlls.append(sim.nll(p_post, t_ys))
    print("BASELINE under w_novelty={}: NLL = {:.4f} +/- {:.4f}".format(
        w_novelty, float(np.mean(nlls)), float(np.std(nlls))
    ))


def sweep_tau_extended(tau_grid, w_novelty=1000.0, w_anchor=10.0):
    print()
    print("=" * 80)
    print("Sweep 2b (extended): tau in {} with w_novelty={}".format(tau_grid, w_novelty))
    print("=" * 80)
    print("{:>10s}  {:>12s}  {:>14s}  {:>12s}  {:>12s}".format(
        "tau", "new_frac", "NLL_all", "NLL_known", "NLL_new"
    ))
    rows = []
    for tau in tau_grid:
        ms = {"new_frac_acq": [], "nll_all": [], "nll_known": [], "nll_new": []}
        for t in range(N_TRIALS):
            r = run_one(
                w_novelty=w_novelty, w_anchor=w_anchor,
                tau_global=tau, tau_bc=tau,
                realistic=True, anchor_noise=False,
                trial_seed=10_000 + t * 31,
            )
            for k in ms:
                ms[k].append(r[k])
        agg = {k: float(np.mean(v)) for k, v in ms.items()}
        std_all = float(np.std(ms["nll_all"]))
        print("{:>10.2f}  {:>12.2%}  {:>6.4f}+-{:.4f}  {:>12.4f}  {:>12.4f}".format(
            tau, agg["new_frac_acq"], agg["nll_all"], std_all,
            agg["nll_known"], agg["nll_new"]
        ))
        rows.append((tau, agg))
    return rows


def sweep_dual_pool(fractions, tau=20.0, w_anchor=10.0):
    print()
    print("=" * 80)
    print("Sweep 3: dual-pool stratification (varies new-bc fraction in pool A)")
    print("=" * 80)
    print("{:>10s}  {:>12s}  {:>14s}  {:>12s}  {:>12s}".format(
        "frac_pool", "new_frac_acq", "NLL_all", "NLL_known", "NLL_new"
    ))
    rows = []
    for frac in fractions:
        ms = {"new_frac_acq": [], "nll_all": [], "nll_known": [], "nll_new": []}
        for t in range(N_TRIALS):
            world = sim.World(
                M_known=12, M_new=3, S=50,
                anchor_noise=False,
                b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
                b_bc_mu_new=0.0, b_bc_sigma_new=1.0,
                seed=10_000 + t * 31,
            )
            subj, bc, ps, ys, _ = acquisition_dual_pool(
                world, fraction_new_pool=frac, w_anchor=w_anchor,
                seed=10_000 + t * 31 + 1,
            )
            _, t_bc, t_ps, t_ys, _ = sim.uniform_test_set(world, n=10000, seed=10_000 + t * 31 + 2)
            new_mask_test = world.new_bc_mask[t_bc]
            new_frac_acq = float(world.new_bc_mask[bc].mean())
            cal = sim.cal_partial_pool(
                subj, bc, ps, ys, prior_b=0.0, tau_global=tau, tau_bc=tau,
            )
            p_post = sim.apply_cal(cal, t_bc, t_ps)
            ms["new_frac_acq"].append(new_frac_acq)
            ms["nll_all"].append(sim.nll(p_post, t_ys))
            ms["nll_known"].append(sim.nll(p_post[~new_mask_test], t_ys[~new_mask_test]))
            ms["nll_new"].append(sim.nll(p_post[new_mask_test], t_ys[new_mask_test]))
        agg = {k: float(np.mean(v)) for k, v in ms.items()}
        std_all = float(np.std(ms["nll_all"]))
        print("{:>10.2f}  {:>12.2%}  {:>6.4f}+-{:.4f}  {:>12.4f}  {:>12.4f}".format(
            frac, agg["new_frac_acq"], agg["nll_all"], std_all,
            agg["nll_known"], agg["nll_new"]
        ))
        rows.append((frac, agg))
    return rows


def main() -> None:
    np.random.seed(42)

    baseline_reference()

    w_grid = [0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 1000.0]
    rows1 = sweep_w_novelty(w_grid, w_anchor=10.0, tau=10.0)

    tau_grid = [2.0, 5.0, 10.0, 20.0, 50.0]
    # Pick best w_novelty from rows1 by NLL_all.
    best_w = min(rows1, key=lambda r: r[1]["nll_all"])[0]
    sweep_tau(tau_grid, w_novelty=best_w)

    # Extended tau sweep to find the sweet spot.
    sweep_tau_extended([10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0])

    # Dual-pool stratification: does explicit balance help?
    sweep_dual_pool([0.05, 0.20, 0.50, 0.80, 0.95, 1.00])

    # Coarse joint sweep at smaller grid for global view.
    joint_w = [0.0, 0.5, 1.0, 2.0, 5.0]
    joint_tau = [2.0, 5.0, 10.0, 20.0]
    best, _ = joint_sweep(joint_w, joint_tau)


if __name__ == "__main__":
    main()
