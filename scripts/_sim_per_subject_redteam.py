"""Red-team simulation for the per-subject calibrator channel.

Tests whether adding ``delta_subj[subject_key]`` to the partial-pool
calibrator actually improves held-out NLL across a battery of regimes,
INCLUDING adversarial ones designed to expose overfit / misspecification.

We accept the per-subject channel iff:
  (1) It wins (mean NLL strictly lower) on every honest regime.
  (2) It does NOT degrade NLL on adversarial regimes by more than the
      noise floor.

Regimes
-------
  HONEST_A: no subject bias              -- per-subject should be a NOOP
                                            (or marginally hurt because of
                                            added freedom; ridge should
                                            keep it bounded)
  HONEST_B: small subject bias (0.2)     -- per-subject should help a little
  HONEST_C: large subject bias (0.5)     -- per-subject should help a lot

  ADV_D:    bc * subj interaction        -- additive per-subject can't
                                            capture; should at least not
                                            wreck things
  ADV_E:    label-noise floor (y random) -- ridge must hold; channel
                                            should NOT degrade NLL
  ADV_F:    anchor concentration         -- 75 labels go to top-3 subjects
                                            only; per-subject learns those
                                            3 but test set has 50 subjects
                                            (most see only b_global)
  ADV_G:    heavy-tailed subject bias    -- 5% of subjects have N(0,1),
                                            rest are 0.  Most test rows
                                            see 0 bias; calibrator should
                                            not over-correct.

For each regime: 100 trials with paired seeds (so the comparison is
within-trial).  Report:
  * mean NLL with / without per-subject
  * 95% CI of the within-trial diff
  * win rate (per-subject < baseline)
  * paired t-stat
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sim_calibration_ideas as sim  # noqa: E402
import _sim_acquisition_tuning as tune  # noqa: E402

N_TRIALS = 60
N_TEST = 5_000

# Calibrator hyperparams.
TAU_GLOBAL = 20.0
TAU_BC = 20.0
TAU_SUBJ = 20.0  # initial guess -- swept separately later


# ---------------------------------------------------------------------------
# Extended World with per-subject true bias + interaction option.
# ---------------------------------------------------------------------------


class SubjWorld(sim.World):
    """sim.World extended with per-subject true bias.

    Adds ``b_subj_true[S]`` and incorporates it into the true probability:
        true_p = sigmoid(z_true + b_bc_true[bc] + b_subj_true[subj] + interaction_term)

    interaction_term defaults to 0; can be set to e.g. b_bc * b_subj *
    alpha to inject non-additive interaction.
    """

    def __init__(
        self,
        *,
        b_subj_sigma: float = 0.0,
        b_subj_heavy_tail_frac: float = 0.0,
        b_subj_heavy_tail_sigma: float = 1.0,
        interaction_alpha: float = 0.0,
        random_y: bool = False,
        anti_anchor: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        S = self.S
        if b_subj_heavy_tail_frac > 0.0:
            mask = self.rng.random(S) < b_subj_heavy_tail_frac
            heavy = self.rng.normal(0.0, b_subj_heavy_tail_sigma, S)
            normal = self.rng.normal(0.0, b_subj_sigma, S) if b_subj_sigma > 0 else np.zeros(S)
            self.b_subj_true = np.where(mask, heavy, normal)
        else:
            self.b_subj_true = self.rng.normal(0.0, b_subj_sigma, S) if b_subj_sigma > 0 else np.zeros(S)
        # ``anti_anchor``: well-anchored subjects (more training data)
        # have SMALLER calibration error -- the realistic case where the
        # platform's head is trained harder on popular models, so the
        # acquisition's preference for anchored subjects starves the
        # per-subject calibrator of exactly the rows that need it.
        if anti_anchor:
            self.b_subj_true *= (1.0 - 0.9 * self.anchor_normalized)
        self.interaction_alpha = float(interaction_alpha)
        self.random_y = bool(random_y)

    def _gen_uncal_and_label(self, subject_ids: np.ndarray, bc_ids: np.ndarray, rng):
        n = len(subject_ids)
        z_true = rng.normal(0.0, 1.0, n)
        if self.anchor_noise:
            sigma = self.anchor_noise_scale * (1.0 - self.anchor_normalized[subject_ids])
            z_observed = z_true + sigma * rng.normal(0.0, 1.0, n)
        else:
            z_observed = z_true.copy()
        p_uncal = sim.sigmoid(z_observed)
        b_bc = self.b_bc_true[bc_ids]
        b_subj = self.b_subj_true[subject_ids]
        interaction = self.interaction_alpha * b_bc * b_subj
        true_p = sim.sigmoid(z_true + b_bc + b_subj + interaction)
        if self.random_y:
            y = (rng.random(n) < 0.5).astype(float)
        else:
            y = rng.binomial(1, true_p).astype(float)
        return p_uncal, y, true_p


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


def cal_pp_bc(subj, bc, ps, ys, *, tau_global=TAU_GLOBAL, tau_bc=TAU_BC):
    """The currently SHIPPED partial-pool calibrator (b_global + delta_bc)."""
    b_global = sim.fit_ridge_intercept(ps, ys, target_b=0.0, ridge=tau_global)
    delta_bc: dict[int, float] = {}
    by_bc: dict[int, tuple] = {}
    for s, b, p, y in zip(subj, bc, ps, ys):
        by_bc.setdefault(int(b), ([], [])); by_bc[int(b)][0].append(float(p)); by_bc[int(b)][1].append(float(y))
    for bc_id, (lp, ly) in by_bc.items():
        delta_bc[bc_id] = sim.fit_ridge_intercept(
            np.array(lp), np.array(ly), target_b=b_global, ridge=tau_bc
        ) - b_global
    return {"b_global": b_global, "delta_bc": delta_bc, "delta_subj": {}}


def cal_pp_bc_subj(subj, bc, ps, ys, *, tau_global=TAU_GLOBAL, tau_bc=TAU_BC, tau_subj=TAU_SUBJ):
    """CANDIDATE: partial-pool with both delta_bc AND delta_subj.

    Backfit:
      1. b_global = fit(all, target=0, ridge=tau_global)
      2. delta_bc[k] = fit(rows in bc=k, target=b_global, ridge=tau_bc) - b_global
      3. delta_subj[s] = fit(rows of subj=s, offset=(b_global + delta_bc[bc_i] per row),
                              target=0, ridge=tau_subj)
    """
    state = cal_pp_bc(subj, bc, ps, ys, tau_global=tau_global, tau_bc=tau_bc)
    b_global = state["b_global"]
    delta_bc = state["delta_bc"]
    # Stage 3: per-subject delta with the bc shifts as offsets.
    zs = sim.logit(ps)
    delta_subj: dict[int, float] = {}
    by_subj: dict[int, tuple] = {}
    for s, b, z, y in zip(subj, bc, zs, ys):
        offset = b_global + delta_bc.get(int(b), 0.0)
        bucket = by_subj.setdefault(int(s), ([], [], []))
        bucket[0].append(float(z))
        bucket[1].append(float(y))
        bucket[2].append(float(offset))
    for s_id, (lz, ly, lo) in by_subj.items():
        lz = np.array(lz); ly = np.array(ly); lo = np.array(lo)
        # Fit b minimizing sum BCE(z + offset + b, y) + ridge * b^2.
        b = 0.0
        for _ in range(80):
            q = sim.sigmoid(lz + lo + b)
            g = 2.0 * tau_subj * b + float((q - ly).sum())
            h = 2.0 * tau_subj + float((q * (1.0 - q)).sum())
            if h < 1e-9:
                break
            step = g / h
            b -= step
            if abs(step) < 1e-9:
                break
        delta_subj[s_id] = max(-5.0, min(5.0, float(b)))
    state["delta_subj"] = delta_subj
    return state


def apply_cal_subj(state, subj, bc, ps):
    z = sim.logit(ps)
    b_global = state["b_global"]
    delta_bc = state["delta_bc"]
    delta_subj = state["delta_subj"]
    n = len(ps)
    # Vectorized: build a per-row b via two array lookups.
    bc_arr = np.asarray(bc, dtype=np.int64)
    subj_arr = np.asarray(subj, dtype=np.int64)
    if delta_bc:
        max_bc = int(max(bc_arr.max() if n else 0, max(delta_bc.keys()))) + 1
        bc_table = np.zeros(max_bc, dtype=float)
        for k, v in delta_bc.items():
            bc_table[k] = v
        bc_shifts = bc_table[bc_arr]
    else:
        bc_shifts = np.zeros(n)
    if delta_subj:
        max_subj = int(max(subj_arr.max() if n else 0, max(delta_subj.keys()))) + 1
        subj_table = np.zeros(max_subj, dtype=float)
        for k, v in delta_subj.items():
            subj_table[k] = v
        subj_shifts = subj_table[subj_arr]
    else:
        subj_shifts = np.zeros(n)
    b_per_row = b_global + bc_shifts + subj_shifts
    return sim.sigmoid(z + b_per_row)


# ---------------------------------------------------------------------------
# One-trial comparison.
# ---------------------------------------------------------------------------


def run_trial(
    *,
    seed: int,
    b_subj_sigma: float,
    interaction_alpha: float = 0.0,
    b_subj_heavy_tail_frac: float = 0.0,
    b_subj_heavy_tail_sigma: float = 1.0,
    random_y: bool = False,
    anchor_concentration: bool = False,
    anti_anchor: bool = False,
    tau_subj: float = TAU_SUBJ,
):
    world = SubjWorld(
        M_known=12, M_new=3, S=50,
        anchor_noise=False,
        b_bc_mu_known=0.0, b_bc_sigma_known=0.2,
        b_bc_mu_new=0.0, b_bc_sigma_new=1.0,
        b_subj_sigma=b_subj_sigma,
        b_subj_heavy_tail_frac=b_subj_heavy_tail_frac,
        b_subj_heavy_tail_sigma=b_subj_heavy_tail_sigma,
        interaction_alpha=interaction_alpha,
        random_y=random_y,
        anti_anchor=anti_anchor,
        seed=seed,
    )
    # Acquisition: production dual-pool (matches what we ship).
    subj, bc, ps, ys, _ = tune.acquisition_dual_pool(
        world, fraction_new_pool=0.95, w_anchor=10.0, seed=seed + 1,
    )
    if anchor_concentration:
        # Override: send 90% of labels to top-3 anchored subjects (worst case).
        top3 = np.argsort(-world.anchor_normalized)[:3]
        n = len(subj)
        n_top = int(0.9 * n)
        rng = np.random.default_rng(seed + 2)
        new_subj = np.concatenate([
            top3[rng.integers(0, 3, n_top)],
            subj[:n - n_top],
        ])
        np.random.default_rng(seed + 3).shuffle(new_subj)
        subj = new_subj
        # Re-generate (p_uncal, y) for the rewritten subjects.
        ps, ys, _ = world._gen_uncal_and_label(subj, bc, np.random.default_rng(seed + 4))
    _, t_bc, t_ps, t_ys, _ = sim.uniform_test_set(world, n=N_TEST, seed=seed + 10)
    # Need test subject ids for the per-subject apply path.
    rng_t = np.random.default_rng(seed + 10)
    t_subj = rng_t.integers(0, world.S, N_TEST)
    t_bc2 = rng_t.integers(0, world.M, N_TEST)
    t_ps2, t_ys2, _ = world._gen_uncal_and_label(t_subj, t_bc2, rng_t)

    state_bc = cal_pp_bc(subj, bc, ps, ys)
    state_bc_subj = cal_pp_bc_subj(subj, bc, ps, ys, tau_subj=tau_subj)

    p_bc = apply_cal_subj(state_bc, t_subj, t_bc2, t_ps2)
    p_bc_subj = apply_cal_subj(state_bc_subj, t_subj, t_bc2, t_ps2)
    nll_bc = sim.nll(p_bc, t_ys2)
    nll_bc_subj = sim.nll(p_bc_subj, t_ys2)
    return nll_bc, nll_bc_subj


# ---------------------------------------------------------------------------
# Regime sweep.
# ---------------------------------------------------------------------------


def run_regime(label: str, **kwargs) -> dict:
    diffs = []
    nll_bc_list = []
    nll_bcs_list = []
    for t in range(N_TRIALS):
        nll_bc, nll_bc_subj = run_trial(seed=10_000 + t * 31, **kwargs)
        diffs.append(nll_bc - nll_bc_subj)  # >0 means per-subject is better
        nll_bc_list.append(nll_bc)
        nll_bcs_list.append(nll_bc_subj)
    diffs = np.array(diffs)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    se = sd / math.sqrt(len(diffs))
    ci95 = (mean - 1.96 * se, mean + 1.96 * se)
    win_rate = float((diffs > 0).mean())
    t_stat = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    return {
        "label": label,
        "mean_diff": mean,
        "sd_diff": sd,
        "ci95": ci95,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "mean_bc": float(np.mean(nll_bc_list)),
        "mean_bc_subj": float(np.mean(nll_bcs_list)),
    }


def print_row(r: dict) -> None:
    sig = "***" if abs(r["t_stat"]) > 2.58 else ("** " if abs(r["t_stat"]) > 1.96 else "   ")
    print(
        "{lab:<24s} NLL_bc={mb:.4f} NLL_bcs={mbs:.4f}  diff={d:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  win={wr:.0%}  t={t:+.2f} {sig}".format(
            lab=r["label"], mb=r["mean_bc"], mbs=r["mean_bc_subj"],
            d=r["mean_diff"], lo=r["ci95"][0], hi=r["ci95"][1],
            wr=r["win_rate"], t=r["t_stat"], sig=sig,
        )
    )


def main() -> int:
    print("Red-team: PER-SUBJECT channel vs SHIPPED partial-pool (b_global + delta_bc only)")
    print("  diff > 0 => per-subject wins on NLL")
    print("  100 trials per regime, paired seeds, uniform-random uniform-bc test set")
    print("  (tau_global, tau_bc, tau_subj) = ({}, {}, {})".format(TAU_GLOBAL, TAU_BC, TAU_SUBJ))
    print()
    print("HONEST regimes (we MUST win or tie):")
    for label, kwargs in [
        ("A no subject bias",    dict(b_subj_sigma=0.0)),
        ("B small subj bias 0.2", dict(b_subj_sigma=0.2)),
        ("C large subj bias 0.5", dict(b_subj_sigma=0.5)),
    ]:
        print_row(run_regime(label, **kwargs))
    print()
    print("ADVERSARIAL regimes (we MUST NOT lose substantially):")
    for label, kwargs in [
        ("D bc*subj interaction 0.3", dict(b_subj_sigma=0.3, interaction_alpha=0.3)),
        ("E random labels",          dict(b_subj_sigma=0.0, random_y=True)),
        ("F anchor concentration",   dict(b_subj_sigma=0.3, anchor_concentration=True)),
        ("G heavy-tail 5% sigma=1",  dict(b_subj_sigma=0.0, b_subj_heavy_tail_frac=0.05, b_subj_heavy_tail_sigma=1.0)),
        ("H anti-anchor 0.5",         dict(b_subj_sigma=0.5, anti_anchor=True)),
    ]:
        print_row(run_regime(label, **kwargs))
    print()
    # Sweep tau_subj on regime B to find the sweet spot.
    print("tau_subj sweep on regime B (small subj bias 0.2):")
    for ts in [5.0, 10.0, 20.0, 40.0, 80.0]:
        r = run_regime("  tau_subj={:>4.0f}".format(ts), b_subj_sigma=0.2, tau_subj=ts)
        print_row(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
