# Red-team findings: per-subject and kNN-neighbor channels

Following the suggestion in the external review that we were
"missing higher-leverage things than larger model + ensemble," we ran
adversarial simulations on the two candidate extensions to the shipped
partial-pool calibrator (`b_global` + `delta_bc`):

  1. **Per-subject channel** (`delta_subj`).
  2. **Embedding-neighbor channel** (`Delta_neighbor`).

Both were rejected by the red team.  This document records the
results so future iterations don't relitigate them without new
evidence.

## Verdicts

| Channel | Best honest case | Worst adversarial | Decision |
|---|---|---|---|
| `delta_subj` | +0.0001 nats (regime C, sigma=0.5) | -0.0001 nats, t=-3.02 on random labels | **Do not ship.** |
| `Delta_neighbor` (logit) | -0.0031 nats on every regime | catastrophic everywhere | **Do not ship.** |
| `Delta_neighbor` (prob, conservative) | ~0.0000 (ties), 27-32% win rate | small but consistent losses | **Do not ship.** |

For comparison, the **PP_CONSERVATIVE refactor** (`b_global` +
`delta_bc` with `tau=20`, plus the dual-pool 0.95 acquisition) that
we did ship in commit `6bbcd01` was empirically worth **+0.0082
nats** with a 6x lower variance.  The candidates fail by **80x or
more** in magnitude on the best honest case.

## Why the channels fail (structural, not implementational)

Both extensions share the same root cause.  The platform's acquisition
loop reveals 75 labels per round and we ship a dual-pool acquisition
that hits ~7 anchored subjects with ~10 labels each plus a thin tail.
The test set is roughly uniform over 50 subjects × 200 items.

  - **Per-subject coverage:** only ~16% of test rows are from a
    subject for which we have any labels.  Even when the per-subject
    fit perfectly captures the bias for an anchored subject (it
    doesn't: ridge=20 shrinks the estimate by ~80%), the contribution
    to mean NLL is bounded by `(coverage fraction) * (bias variance)
    / 2`.  With sigma_subj=0.5 that ceiling is ~0.02 nats, which the
    bias-variance tradeoff then erodes to ~0.0001.

  - **Embedding-neighbor coverage:** same as above, plus each
    anchored subject has ~10 labels distributed across 200 items in a
    32-dim embedding.  Random unit vectors in 32 dims have pairwise
    cosine similarity ~ N(0, 0.18), so the kernel weights are nearly
    flat and the "neighbor average" is a noisy average over ~5
    labels.  Binary label noise dominates the residual signal.

The per-subject channel's `tau_subj` sweep on the most-favorable
honest regime (B, sigma=0.2) shows monotonic improvement as we make
the fit weaker (tau=5 -> tau=80), which is the diagnostic signature
of "the signal is below the variance floor".  No hyperparameter
tuning rescues either channel.

## When would these channels start working

  - More labels per subject.  With ~30 labels/subject, per-subject
    pickup is real.  This requires either a different acquisition
    that concentrates on fewer subjects, or a multi-round protocol
    that accumulates labels across rounds (which the platform does
    not give us).

  - Strong embedding signal that the bc channel can't already absorb.
    Today, the bc calibrator captures most of the variance and the
    residual is mostly noise.  A model that gives the bc channel
    less work (e.g. by having a better-calibrated base predictor)
    would leave more room for `Delta_neighbor` to matter.

  - A test set that concentrates on the same subjects the labels
    cover.  The actual platform test distribution is uniform-ish, so
    this doesn't apply unless the holdout policy changes.

## What we shipped instead

  - The submission **audit harness** (`scripts/_audit_bundle.py`),
    which already paid for itself by catching `calibration_disabled:
    True` in the `runtime_meta.json` of two shipped bundles whose
    `model.py` actually calibrates.  This is exactly the silent
    leaderboard/runtime mismatch the review warned about.  The pack
    scripts now strip the stale flag.

  - Both red-team simulations are checked in
    (`scripts/_sim_per_subject_redteam.py`,
    `scripts/_sim_knn_neighbor_redteam.py`) so we can rerun them when
    the regime changes (more labels per subject, better embeddings,
    different test distribution).
