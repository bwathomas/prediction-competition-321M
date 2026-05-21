"""Tau-type sweep + final adversarial battery for the TYPE-conditional calibrator.

This is the candidate that emerged from the Tier-1 sim: 1 extra parameter
delta_type * 1{bc is new} that captures the systematic new-vs-known mean
shift the dual-pool 95/5 acquisition introduces into b_global.

Decision rule: ship iff every honest regime wins by > 0.001 nats AND no
adversarial regime loses by more than the noise floor (~0.0005 nats)
across the tau_type sweep at tau_type=10.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sim_tier1_candidates as t1  # noqa: E402

N_TRIALS = 80  # Bumped from 60 for tighter CIs on the final decision sweep.


def run_regime(label, **kwargs):
    diffs = []
    nll_a, nll_b = [], []
    for t in range(N_TRIALS):
        a, b = t1.run_trial(seed=400_000 + t * 31, candidate="TYPE", **kwargs)
        diffs.append(a - b); nll_a.append(a); nll_b.append(b)
    diffs = np.array(diffs)
    m = float(diffs.mean()); s = float(diffs.std(ddof=1))
    se = s / math.sqrt(len(diffs))
    return {
        "label": label, "mean_diff": m,
        "ci95": (m - 1.96 * se, m + 1.96 * se),
        "win_rate": float((diffs > 0).mean()),
        "t_stat": m / se if se > 0 else 0.0,
        "mean_base": float(np.mean(nll_a)),
        "mean_cand": float(np.mean(nll_b)),
    }


def pr(r):
    sig = "***" if abs(r["t_stat"]) > 2.58 else ("** " if abs(r["t_stat"]) > 1.96 else "   ")
    print("{lab:<36s} base={mb:.4f} cand={mc:.4f}  diff={d:+.5f}  CI=[{lo:+.5f},{hi:+.5f}]  win={wr:.0%}  t={t:+.2f} {sig}".format(
        lab=r["label"], mb=r["mean_base"], mc=r["mean_cand"], d=r["mean_diff"],
        lo=r["ci95"][0], hi=r["ci95"][1], wr=r["win_rate"], t=r["t_stat"], sig=sig,
    ))


def main():
    print("TYPE-conditional intercept: final decision sweep")
    print("  diff > 0 => TYPE wins NLL vs PP_CONSERVATIVE baseline")
    print("  {} trials per regime, paired seeds, 5000 test rows".format(N_TRIALS))
    print()

    HONEST = [
        ("A1 baseline",                dict(b_bc_sigma_new=1.0, b_bc_sigma_known=0.2)),
        ("A2 big new-bc shift 1.5",    dict(b_bc_sigma_new=1.5, b_bc_sigma_known=0.3)),
        ("A4 emb signal 0.4",          dict(b_bc_sigma_new=1.0, item_signal_sigma=0.4)),
        ("A5 emb sig + subj",          dict(b_bc_sigma_new=1.0, item_signal_sigma=0.8, b_subj_sigma=0.3)),
    ]
    ADV = [
        ("Z1 random labels",           dict(b_bc_sigma_new=1.0, random_y=True)),
        ("Z3 near-zero bc shift",      dict(b_bc_sigma_new=0.1, b_bc_sigma_known=0.05)),
        ("Z4 extreme acq 99/1",        dict(b_bc_sigma_new=1.0, fraction_new_pool=0.99)),
        ("Z5 mild acq 50/50",          dict(b_bc_sigma_new=1.0, fraction_new_pool=0.5)),
    ]
    SYST = [
        ("F1 mu=-1/+.5 sigma=0.3",     dict(b_bc_mu_new=-1.0, b_bc_mu_known=0.5,
                                            b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F3 ctrl mu=0/0 sigma=0.3",   dict(b_bc_mu_new=0.0, b_bc_mu_known=0.0,
                                            b_bc_sigma_new=0.3, b_bc_sigma_known=0.3)),
        ("F4 small shift -.3/+.15",    dict(b_bc_mu_new=-0.3, b_bc_mu_known=0.15,
                                            b_bc_sigma_new=0.5, b_bc_sigma_known=0.3)),
    ]

    print("=" * 88)
    print("Tau-type sweep on a sample of regimes (find robust default):")
    print("=" * 88)
    sample_regimes = [("A1 baseline", HONEST[0][1]),
                      ("A2 big shift", HONEST[1][1]),
                      ("Z1 random", ADV[0][1]),
                      ("Z3 zero shift", ADV[1][1]),
                      ("F1 systematic", SYST[0][1]),
                      ("F3 control", SYST[1][1])]
    for tau in [3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        print("--- tau_type = {:>5.1f} ---".format(tau))
        for lab, kw in sample_regimes:
            pr(run_regime(lab, tau_type=tau, **kw))

    print()
    print("=" * 88)
    print("FINAL battery at tau_type = 10 (default):")
    print("=" * 88)
    print("  HONEST regimes:")
    for lab, kw in HONEST:
        pr(run_regime(lab, tau_type=10.0, **kw))
    print("  ADVERSARIAL regimes:")
    for lab, kw in ADV:
        pr(run_regime(lab, tau_type=10.0, **kw))
    print("  SYSTEMATIC-SHIFT regimes:")
    for lab, kw in SYST:
        pr(run_regime(lab, tau_type=10.0, **kw))


if __name__ == "__main__":
    main()
