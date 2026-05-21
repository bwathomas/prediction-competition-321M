# findings.md — Calibration report vs. local repository

This note maps the report's claims about calibrating recommender-style
ranking scores (anchored on *Obtaining Calibrated Probabilities with
Personalized Ranking Models*, Kweon et al., RecSys 2022) onto the
current state of this repository. For each idea I summarize the concept,
say whether we are aligned, what it would take to get aligned, the
difficulty, and the realistic size of the win on our 0.444 item-cold-start
NLL baseline.

The big-picture answer first: the report is half-right about the
diagnosis and half-right about the prescription. Our **training** loss
is binary cross-entropy on direct correctness labels, so the model is
already a probabilistic classifier — not a pure ranker like BPR. But our
**runtime calibrator** does behave exactly like the recommender setting
the paper studies: a small parametric head fit on a tiny, non-uniformly
sampled set of revealed labels. That second half is where most of the
addressable gains live.

---

## 1. "Plain temperature scaling is not the whole story"

**Concept.** The argument is that a single global temperature is too
weak a corrector once the score-generating process is anything beyond a
softmax classifier trained on balanced data. The reliability curve can
be S-shaped, asymmetric, or shifted, and a one-parameter scaling cannot
chase those shapes.

1) **Aligned?** Partially. Our runtime calibrator (`src/calibration.py`)
is already richer than pure temperature scaling: we use identity when
N&lt;5 labels, a 1-parameter *intercept* (logit shift) when 5≤N&lt;30, and
a 2-parameter *temperature + intercept* when N≥30. So we already share
the report's premise that a single temperature parameter is not enough.
What we do **not** have is anything richer than a 2-parameter Platt
sigmoid (no beta calibration, no piecewise / spline-based monotonic
transform, no per-slice calibrators).
2) **What to align further.** Add a beta-calibration head (3 params:
`a*logit(p) + b*logit(1-p) + c` passed through a sigmoid) and use it as
a third tier when N≥~100. Optionally add per-`benchmark` calibrators
when the budget allows, falling back upward when not.
3) **Difficulty.** Low. `src/calibration.py` is ~220 lines, and the
fitter is already a closed-form Newton loop; adding beta calibration is
a ~50-line addition plus a tier in `best_effort_fit`. The matching
runtime kind in `src/export_submission.py:_Calibrator` is another
~20 lines. Smoke test already exercises this path.
4) **Size gain.** Small but real. Holding everything else fixed, a
3-parameter calibrator over 75 revealed labels typically buys
0.001–0.005 NLL when the base model is already well-calibrated, and
more like 0.005–0.015 when the base model has reliability curvature.
We are in the former regime today.

## 2. "The model is closer to a recommender / ranking scorer than a multiclass classifier"

**Concept.** Recommender ranking scores from BPR/CML/LightGCN are
trained with pairwise objectives that recover the *order* of items but
not the *level*, so the sigmoid of the score is uncalibrated by
construction. The paper argues that you have to treat the mapping from
score to probability as an explicit, learned object.

1) **Aligned?** Mostly *not* aligned, but for a good reason. We have a
ranking-flavored architecture (subject embedding × item embedding +
Item-IRT 2PL channel: `alpha(item) * (theta_subj - beta(item))`), which
is structurally close to a personalized ranker. **However**, we train
it with `BCEWithLogitsLoss` on direct binary labels (`src/train.py`
line ~461), not with a pairwise / BPR-style ranking loss. So our logits
are already calibrated to the training distribution by construction (up
to the usual neural-net miscalibration drift). The report's framing
is correct about our *head shape* but incorrect about our *loss*.
2) **What to align with.** Two reasonable variants:
   (a) Treat the BCE-trained model as the score generator and lean
   harder on a learned calibrator (covered in items 1 and 3).
   (b) Add an auxiliary pairwise / listwise ranking loss
   (e.g. BPR over pairs of items within the same subject) on top of the
   BCE term and rebalance with a calibrator. This is what the paper's
   pipeline implicitly looks like.
3) **Difficulty.** (a) is essentially free if we already do item 1.
(b) is medium: it requires building a pair sampler within
`LookupDataset` (subject-anchored item pairs), an `BPRLoss` term, and a
weight `lambda_rank` to compose with BCE. Maybe 1–2 days of work plus
ablation grid runs.
4) **Size gain.** (a) ≈ same as item 1 (0.001–0.015 NLL).
(b) speculative: pairwise auxiliary losses on classifiers usually swing
NLL by ±0.002 — they can help AUC noticeably but rarely move the BCE
metric the platform uses. **Not recommended as a primary lever.**

## 3. Parametric calibration families (Platt / beta / gamma / multi-parameter Platt)

**Concept.** Kweon et al. show that parametric calibration families
dominate nonparametric ones (isotonic regression, histogram binning) on
ranking-style outputs, and that the family matters: beta and
gamma-based families consistently beat plain Platt sigmoid scaling.
Table 1 reports 5.21%–25.85% ECE improvements over the strongest
baseline.

1) **Aligned?** Partially. We already use parametric (Platt-style)
calibration only and never use isotonic / histogram binning — that part
the paper would endorse. But we only implement the simplest sigmoid
family (1- and 2-parameter Platt). We do not have beta or gamma
families.
2) **What to align with.** Add `BetaCalibrator` and possibly
`GammaCalibrator` to `src/calibration.py`, plumb a third tier into
`best_effort_fit`, and add the matching `kind` branches into
`_Calibrator.apply()` and `fit_from_labeled()` in
`src/export_submission.py`. Keep identity / 1-param / 2-param as
fallbacks for low-label regimes; beta needs ≥~50 labels to be stable.
3) **Difficulty.** Low. Each new family is a closed-form-ish Newton
fitter plus an apply() function — ~40–80 lines per family. The
serialization protocol (`to_dict`/`from_dict`) and the runtime
mirror in `_Calibrator` are already wired up to accept new kinds.
4) **Size gain.** The paper's headline 5.21%–25.85% is **relative ECE**
on uncalibrated ranking scores. Translated to our setting, where the
base model is already BCE-trained: I would expect the absolute ECE
improvement on 75 revealed labels to be 0.005–0.02, and the
corresponding NLL improvement to be 0.001–0.005 in the regimes where
the reliability curve is actually curved. Not nothing, but well below
the impact of a single feature ablation.

## 4. Unbiased Empirical Risk Minimization (uERM) for calibrator fitting

**Concept.** This is the paper's most distinctive contribution. The
observed labels are not a uniform sample of the (user, item) population
— in implicit-feedback recommenders, popular items are observed far
more often than tail items. Naively fitting a calibrator on this biased
sample inherits the bias. The fix is to reweight each labeled example
by `1 / P(observed | x)` (an inverse-propensity score) so that the
calibrator's training risk is an unbiased estimator of the population
risk. They report 7.40%–76.52% ECE reductions from switching naive →
unbiased fitting **for every parametric family they tested**.

1) **Aligned?** **Not aligned.** Both the training BCE loss (`src/train.py`)
and the runtime calibrator fit (`src/export_submission.py:_Calibrator.fit_from_labeled`)
treat each labeled example as equally informative. We do nothing about
the platform's adaptive-labeling policy that selects which K=5 items per
data category get revealed. If that policy biases toward, say, items the
platform expects to be informative (high entropy), our calibrator is fit
on an unrepresentative slice and will systematically miscalibrate the
silent items.
2) **What to align with.** Two pieces:
   (a) **Estimate per-item revelation propensities.** The platform's
   labeling logic lives in `submission/labeling.py` and the test-time
   harness's adaptive labeler. We would need to either (i) read out the
   platform's selection probabilities directly (likely not exposed), or
   (ii) fit a small propensity model from the empirical distribution of
   revealed-vs-unrevealed items across rounds — features could be raw
   item-pool features (pool features + judge p_yes + nn-feature
   agreement). Output: `pi_i ∈ (0, 1]`.
   (b) **Use the IPS-weighted BCE** in `_fit_intercept_only` /
   `_fit_temp_intercept` and in any new families:
   `loss = mean( (1/pi_i) * BCE(y_i, sigmoid(z_i / T + b)) )`.
   Clip propensities at `pi_min ≈ 0.01` to bound variance (this is the
   standard self-normalized IPS trick).
3) **Difficulty.** Medium to medium-high.
   - The fitter change itself is trivial (~10 lines per family — replace
     `mean()` with `(w * ...).mean() / w.mean()`).
   - Estimating `pi_i` is the hard part because the platform reveals
     so few labels per round (75) that the propensity model has to be
     simple or strongly regularized, and we have no ground-truth
     propensity oracle. A reasonable starting point: assume the platform
     samples items uniformly at random within a data category (which
     matches the harness's default), in which case `pi_i ≡ 1` and uERM
     reduces to plain ERM — i.e., we are already optimal. If the
     platform's policy is more adaptive than that, we need to infer
     `pi_i` from the labeled-vs-unlabeled imbalance per category over
     multiple rounds.
   - In practice this is best done as a no-op feature flag wired into
     the calibrator that *defaults to uniform weights* and only turns on
     when an empirically-fit propensity model is shipped.
4) **Size gain.** This is the lever with the highest *headroom* but
also the highest *uncertainty*.
   - If the platform reveals labels uniformly within a category: gain
     is **exactly zero** — uERM degenerates to ERM.
   - If the platform reveals labels by an informativeness-driven policy
     (entropy / disagreement / etc.), the paper's numbers suggest
     0.01–0.05 absolute ECE reductions and **could plausibly translate
     to 0.003–0.015 NLL** on our 0.444 baseline.
   - Realistically: I would not invest here until we have empirically
     verified that the platform's labeling is non-uniform across items
     within a data category. Step 1 is a 50-line analysis on
     historical revealed-label distributions, not a code change.

## 5. Unbiased test/validation sets

**Concept.** A related point in the paper: to *measure* calibration
honestly, you need an unbiased test set. They build one by holding out
a uniform random subsample of (user, item) pairs and using that for
calibration evaluation only.

1) **Aligned?** Partially aligned. Our item cold-start split holds out
item_keys disjoint from train (`src/data.py:make_item_cold_start_split`),
which is the *competition-relevant* notion of unbiasedness for our
setting. We also flag the random-row split as leaky (good). We do
**not** maintain a held-out subset specifically reserved for *calibrator*
fitting and evaluation that is uniform over items — calibrator stats
are reported on the full validation split.
2) **What to align with.** Reserve a small (~5%) uniform-over-items
slice of the validation set as a calibration-only fold. Compute ECE /
Brier on it separately. This is purely a measurement / reporting
change; no model change.
3) **Difficulty.** Low. Add a `calibration_fold` flag to
`SplitSpec` and surface it in cell 14 / 14b of the notebook.
~30 lines of code + a column in the results table.
4) **Size gain.** Zero on the leaderboard score directly. But it
strictly improves our ability to *measure* whether items 1, 3, and 4
are helping. Worth doing as a precursor to any calibrator change.

## 6. Joint training of model + calibrator

**Concept.** Not the headline of the paper but a recurring theme in
the calibration literature: rather than two-stage (fit model → fit
calibrator on a small holdout), you can jointly train the model with a
parametric calibrator as a final layer, sometimes with a
calibration-specific regularizer (e.g., focal loss, label-smoothing,
mixup-on-targets).

1) **Aligned?** Not aligned. Our model is BCE-trained without any
calibration-aware regularizer (no focal loss, no label smoothing). The
runtime calibrator is fit purely on the K=5 revealed labels, never
jointly with the model.
2) **What to align with.** Two cheap options that don't restructure
training:
   (a) Add label smoothing (`epsilon=0.01–0.05`) to BCE targets in
   `src/train.py`. Acts as a Bayesian shrinkage prior on the head's
   confidence and is known to reduce overconfidence at the cost of a
   small NLL hit on already-well-calibrated models.
   (b) Add temperature as a learnable scalar at the end of the model
   (not just at calibration time) trained with the same BCE. This is
   essentially "always-on" temperature scaling.
3) **Difficulty.** Trivial. Each is a 1-line change.
4) **Size gain.** Almost certainly neutral or slightly negative.
Label smoothing typically hurts NLL by 0.001–0.005 in BCE-trained
classifiers that are already calibrated, while reducing AUC by similar
amounts. Joint temperature is redundant with the runtime calibrator.
**Not recommended.**

---

## Summary table

| # | Idea | Aligned? | Effort | Expected NLL win |
|---|------|----------|--------|-----------------:|
| 1 | Calibrator richer than temperature scaling | Partially (1- and 2-param Platt already) | Low | 0.001–0.005 |
| 2 | Treat as ranker rather than classifier | No, and we shouldn't | Low–Med | 0–0.005 (mostly AUC, not NLL) |
| 3 | Parametric families (beta/gamma) | Partially (Platt only) | Low | 0.001–0.005 |
| 4 | uERM / IPS-weighted calibrator fitting | No | Med–High (propensity model is the bottleneck) | 0 if labeling is uniform, else 0.003–0.015 |
| 5 | Held-out unbiased calibration fold | Partially | Low | 0 on leaderboard, important infrastructure |
| 6 | Joint model + calibrator training | No | Trivial | Negative or neutral |

## Recommended ordering if we act on this report

1. **#5 first** (measurement infrastructure): one afternoon, no risk.
2. **#1 + #3** together (beta calibration tier in `src/calibration.py`
   and `_Calibrator`): one afternoon, low risk, modest upside.
3. **#4 only if we first verify** that the platform's revealed-label
   sampling is meaningfully non-uniform. If it is uniform (the default
   harness behavior), there is no win here at all and the
   IPS-weighting infrastructure is wasted work.

The report's framing is useful, but its quantitative gains (5%–25%
relative ECE, 7%–76% from uERM) are reported against *uncalibrated
BPR-style ranking scores* on *biased implicit-feedback test sets*.
Neither premise fully holds here, so we should expect roughly an order
of magnitude smaller absolute gains than the paper headlines.
