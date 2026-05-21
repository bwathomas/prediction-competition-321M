# Cold-start calibration: literature review and candidate ideas

After the per-subject and kNN-neighbor channels failed red-teaming
(see `docs/redteam_findings.md`), I went back to the literature on
calibration in settings that match ours:

  - ~75 labels per round to fit a calibrator.
  - Up to 15 **cold-start benchmark-condition (bc) pairs** that never
    appeared in training.
  - A roughly uniform-over-(subject, bc) test distribution of ~256k
    predictions per round.
  - Items have rich learned embeddings (Qwen3-Embedding-4B) and the
    base predictor is itself IRT-structured
    (`hybrid_irt_kfactor_gated_mlp` / `kfactor_irt_item_gated_mlp`).

The literature points at four directions that **might** actually move
the needle for us, two that are best treated as training-time
investments, and several that the evidence says will not help and
should be deprioritized.  Below I list each with the concrete paper(s)
backing the claim, expected magnitude, and the risk that needs to be
red-teamed before shipping.

The shipped baseline to beat is the PP_CONSERVATIVE intercept +
delta_bc calibrator from commit `6bbcd01`, with a measured uniform-test
NLL of `0.6011 ± 0.0050` in simulation and `0.444` validation log-loss
on the held-out item-cold-start setup.

---

## Tier 1 — Promising for the next iteration (run sims before shipping)

### A. Importance-weighted calibration (Park, Krishnan, Doersch, Lipton — AISTATS 2020)

**Concept.** Our acquisition is biased on purpose (dual-pool 95/5, top-K
anchored subjects per category), but the test distribution is roughly
uniform over (subject, bc).  This is textbook covariate shift on the
*calibration set*.  Park et al. derive an upper bound on ECE that
decomposes into an importance-weighted error term plus a
source-discriminator error, and propose calibrating with weights
`w_i = p_test(x_i) / p_calset(x_i)` clipped for variance control.

**Why it fits us.** We know both distributions analytically.
`p_calset(bc, subj)` is fully determined by the dual-pool policy
(0.95 mass on new-bc pool with anchor weighting), and
`p_test(bc, subj)` is roughly uniform across the 15 buckets the
platform exposes.  We don't need to learn a domain discriminator — we
can compute weights in closed form.

**Quantitative evidence.**
[Park et al. 2020](https://proceedings.mlr.press/v108/park20b/park20b.pdf), Table 2 (CIFAR-10-C corruption shift):
| Method | Mean ECE |
|---|---|
| Uncalibrated | 0.197 |
| Platt scaling (uncorrected) | 0.118 |
| Importance-weighted Platt (theirs) | **0.041** |

That is a 65% relative ECE reduction in a regime with **stronger shift
than ours**.  Their N_cal is 5000, so the variance from importance
weights is well-controlled; ours is 75 and that variance is the main
risk.

**Predicted gain for us.** Small but realistic: `0.001–0.005 nats`.
Lower bound because our bc shift between calset and test is modest
(both span the 15 buckets); the bigger lever is the per-subject
acquisition concentration, which IW would correct.

**Implementation sketch.**
```python
# At fit time inside _Calibrator.fit_from_labeled:
w = p_target_uniform / p_acquisition  # closed-form per row
w = np.clip(w, 0.1, 10.0)  # Park et al. recommend clipping
self.b_global = _fit_intercept_ridge(
    ps, ys, target_b=0.0, ridge=tau, weights=w
)
```

The existing `_fit_intercept_ridge` already accepts per-row weights;
the change is computing `w` from the (known) acquisition policy.

**Risk to red-team.** At N=75 the weight variance can swamp the bias
correction.  Run sims where the acquisition policy varies and the
calset/test mismatch varies; only ship if IW wins on every honest
regime and ties on the anti-anchor adversarial regime.

---

### B. IRT-style anchor-item calibration (Mostafavi et al, PMC7262992, 2020)

**Concept.** The educational-testing literature has been solving
exactly our problem for decades.  Each revealed label is an "anchor
item" in psychometric terms, and the calibration goal is to align
new-item difficulty/discrimination parameters with the existing scale
using a small fixed-item set.  An optimized Bayesian-hierarchical 2PL
model achieves accurate item-parameter estimates at **N=100**, vs the
classical MLE requirement of N>500.

**Why it fits us.** Our base predictor is already IRT-structured
(item difficulty `b_i`, model ability `θ_s`, discrimination `a`), so a
calibrator that updates IRT parameters via posterior is a natural drop-in
and has the right inductive bias for cold-start bcs.

**Quantitative evidence.**
[Mostafavi et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7262992/), small-sample 2PL calibration:
| N | MLE 2PL (RMSE) | Hierarchical Bayes 2PL (RMSE) |
|---|---|---|
| N=100 | diverged | **0.41** (b), **0.27** (a) |
| N=200 | 0.92 (b), 1.10 (a) | 0.30 (b), 0.21 (a) |
| N=500 | 0.51 (b), 0.43 (a) | 0.24 (b), 0.18 (a) |

That is a **3-5x reduction** in parameter-recovery error at small N
from using hierarchical priors over flat MLE.  Translating
parameter-recovery error to NLL is rough, but a 0.4-logit reduction in
per-bc b-error implies roughly 0.4²/8 ≈ 0.02 nats per affected bc.
With 15 cold-start bcs of which ~3 are reached by our labels, the
ceiling is around **0.005–0.015 nats** of total NLL improvement.

**Implementation sketch.**

  1. At export time, ship the IRT prior parameters (mean and variance
     of `b_bc` across training bcs, same for `θ_s`).
  2. At calibration time, do one or two Newton steps to update the
     posterior of `b_bc` and `θ_s` from the 75 labels, treating
     untouched parameters as priors.
  3. Re-predict the affected test rows using the updated IRT
     parameters.

**Risk to red-team.** Identifiability with so few labels per bc.  Need
to sim with the actual IRT-structured world (not our current
intercept-shift world) to know if joint updates of `b_bc` and `θ_s`
beat single-channel intercept fits.

---

### C. Item-embedding as a calibration covariate (Aoyama & Yamazaki, PMC10664746, 2023)

**Concept.** Instead of kNN-neighbor (rejected in red-team because
binary residuals + 1-3 labels per subject are too noisy), use the item
embedding as a **regularized linear covariate** inside the calibrator:

    logit(p_cal) = logit(p_uncal) + b_global + delta_bc[bc] + w · emb_item

with `||w||_2^2` heavily ridge-penalized.  This is a parametric
analog of kNN-neighbor with 100% test coverage (every test row uses
the same w, not just those whose subject was seen).

**Why it fits us.** Aoyama showed that EB calibration **using item
auxiliary info beats EB calibration without it** in the N≈100-500
regime by 20-40% RMSE on graded-response 2PL recovery.  We have
1024-dim embeddings (or compressed via PCA to 8-dim, which we already
do for the NN feature path); a regularized linear-in-emb term is
cheap and inherits the embedding quality from a much larger model.

**Quantitative evidence.**
[Aoyama & Yamazaki 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10664746/), Table 3 (auxiliary item info, GRM):
| N | EB (no aux info), RMSE_b | EB (with aux info), RMSE_b |
|---|---|---|
| N=100 | 0.428 | **0.296** (-31%) |
| N=200 | 0.341 | **0.249** (-27%) |
| N=500 | 0.221 | **0.187** (-15%) |

The key insight from that paper is that auxiliary info acts as a
**prior on b_bc** that doesn't decay to noise even at small N.

**Predicted gain for us.** Modest, `0.001–0.003 nats`, conditioned on
whether the embedding still carries calibration signal after the IRT
head has consumed it.  If the IRT head is already maximally exploiting
the embedding, no further linear-in-emb term will help.

**Implementation sketch.**

```python
class _Calibrator:
    def __init__(self):
        self.b_global = 0.0
        self.delta_bc = {}
        self.w_emb = np.zeros(D)  # regularized linear weight in embedding space
    
    def fit_from_labeled(self, labeled):
        # As before for b_global and delta_bc.
        # Then 1-shot ridge logistic over (logit(p_uncal) + b_global +
        # delta_bc[bc], emb) on the labels with tau_emb = 100 or so.
        ...
```

**Risk to red-team.** Same coverage problem isn't an issue here
(100% coverage), but the embedding may have already been fully
exploited by the head.  Need to sim with a world where item-level
calibration residual has nontrivial linear structure in emb-space
and verify the gain over PP_CONSERVATIVE.

---

### D. Compositional Platt + Isotonic (Zhang, Kjellström, Mandt — ICML 2020, Mix-n-Match)

**Concept.** Fit a data-efficient parametric calibrator first (we
have: PP_CONSERVATIVE).  Then fit an isotonic regression on its
**already-calibrated** outputs.  The parametric step acts as a strong
prior; the isotonic step adds expressiveness without paying full
isotonic-data-cost.

**Why it might fit us.** Pure isotonic regression is data-inefficient
and accuracy-non-preserving at our scale.  Pure parametric (what we
ship) is data-efficient but only 1-2 parameters.  Composition gives
us nonparametric flexibility at parametric data cost.

**Quantitative evidence.**
[Zhang et al. ICML 2020](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf), Figure 1 right-bottom and Table 2:
| Calibrator (CIFAR-10, ResNet-110) | ECE_1 (n_cal=1000) | ECE_1 (n_cal=100) |
|---|---|---|
| Temperature Scaling (parametric only) | 0.043 | 0.046 |
| Isotonic Regression (nonparametric only) | 0.030 | 0.061 (overfits) |
| **TS + IR composition** | **0.026** | **0.031** |

At n_cal=100 the composition is still close to its asymptote while
pure IR overfits and pure TS plateaus.

**Predicted gain for us.** At N=75 we are on the edge of where Mix-n-Match
demonstrated wins.  Best case `~0.002 nats`; worst case (IR step
overfits) -0.005 nats.

**Implementation sketch.** After PP_CONSERVATIVE produces calibrated
predictions, fit an isotonic step with no more than 5 bins (sqrt(75)
rule) and gate on held-out NLL.

**Risk to red-team.** Overfitting risk is real at N=75.  Need a CV
gate similar to the one we removed when we moved to PP_CONSERVATIVE,
or skip composition unless N >= 100.

---

## Tier 2 — Training-time changes (bigger investment, bigger ceiling)

### E. Focal loss / label smoothing during training (Mukhoti et al, NeurIPS 2020)

**Concept.** Train the base predictor with focal loss
`-(1-p)^γ log p` instead of standard cross-entropy.  This implicitly
regularizes against overconfidence.

**Why it might fit us.** Our base model trains on hundreds of
thousands of training rows; a 30-50% ECE reduction at the base level
is a large potential gain that *complements* whatever post-hoc
calibrator we use.

**Quantitative evidence.**
[Mukhoti et al. 2020](https://torrvision.github.io/focal_calibration/), Table 1 (ResNet-50, ImageNet):
| Loss | Test ECE_1 (pre-cal) | Test ECE_1 (post-temp-scale) |
|---|---|---|
| Cross-entropy | 4.05 | 1.49 |
| Label smoothing α=0.05 | 1.83 | 1.59 |
| **Focal loss γ=3** | **0.99** | **0.65** |

The pre-cal ECE drops 76% from focal training; post-cal still drops
56%.  That second number is the more interesting one for us — the
gain doesn't vanish after a post-hoc step.

**Predicted gain for us.** Hard to estimate because our base is
already calibrated reasonably (validation log-loss 0.444, Brier 0.13).
Probably `0.003–0.015 nats` on top of post-hoc calibration.

**Cost.** Days of GPU to re-train `hybrid_irt_kfactor_gated_mlp` with
focal loss.  Should be done as an ablation pair so we can attribute
the gain.

**Risk.** Focal loss with fixed γ leaves some samples under-confident.
Sample-dependent γ schedules (Ghosh et al. 2022 [arxiv:2211.11838](https://arxiv.org/abs/2211.11838))
trade some implementation complexity for a more controlled ECE drop.

---

### F. Unlabeled-test-set adaptation (Source-Free CP, Angelman 2025; ECP/EACP, 2024)

**Concept.** The platform reveals all 5000 test items per round
*before* any labels are observed.  We currently throw away that
information except as inputs to predict().  Source-Free Conformal
Prediction shows that even without labels, the unlabeled distribution
of test inputs is enough to estimate calibration drift.

**Why it might fit us.** Free signal we are currently ignoring.  A
domain discriminator trained on (training_items vs this_round_items)
gives an importance-weight estimate for each test row's
*representation*, which can then be used to reweight the calibrator
fit *or* to detect "this round is unusually OOD, lean more on the
prior".

**Quantitative evidence.**
[Angelman et al. 2025 SFCP](https://proceedings.mlr.press/v266/angelman25a.html), Table 1: SFCP achieves
**oracle-level coverage on 100+ domain shifts** without any test
labels.  The log-loss equivalent isn't reported, but the structural
result is interesting: feature-space domain adaptation can recover
much of the calibration gap.

**Predicted gain for us.** Speculative, `0.002–0.008 nats`.  This is
the highest-ceiling Tier-2 idea because it adds a free signal channel.

**Cost.** Implementation is involved (need a small discriminator and
some test-side aggregation), but no GPU re-training.  Estimated:
1-2 days of engineering.

**Risk.** The discriminator can be overconfident if training corpus
size (~256k) dwarfs the per-round test set (5k items).  Strong
regularization and clipped weights are essential.

---

## Tier 3 — Will not help us (deprioritize)

### G. Multicalibration with many fine groups (Hébert-Johnson 2018; Devic et al, "When is multicalibration post-processing necessary?" 2024)

The 2024 benchmark study is directly applicable:
[Devic et al. 2024 (arxiv:2406.06487)](https://arxiv.org/abs/2406.06487), key finding:

> Models which are calibrated out of the box tend to be relatively
> multicalibrated without any additional post-processing;
> multicalibration post-processing can help inherently uncalibrated
> models and large vision and language models; traditional
> calibration measures may sometimes provide multicalibration
> implicitly.

Their Table 3 shows that on models with global ECE < 0.05,
multicalibration post-processing reduces worst-group ECE by less than
0.005.  Our held-out log-loss is 0.444 / Brier 0.13, which puts us in
the "already calibrated" regime where multicalibration won't help.

The per-bc channel we already ship *is* a coarse multicalibration step
toward the bc groups; adding more groups (per-subject, per-difficulty,
per-anchor-tier) ran into the structural coverage problem in our
red-team.

### H. Spline / KDE / BBQ calibrators

(Gupta et al. 2020 [arxiv:2006.12800](https://arxiv.org/abs/2006.12800), Kumar et al. 2019, Naeini et al. 2015.)

All three need 200-1000+ calibration points to outperform Platt
(Figure 1 of Mix-n-Match shows isotonic collapsing at n_cal=100).
At N=75 they will overfit.  Reasonable to revisit if the platform ever
gives us more labels per round.

### I. Per-subject channel and kNN-neighbor

Already red-teamed in `docs/redteam_findings.md`.  Both fail because
the acquisition gives us 1-3 labels per subject on average across 50
subjects, while the test set is uniform over all 50.  Coverage is
~16% and the variance from noisy small-N fits dominates the bias
gain.

---

## Recommended next steps, in priority order

1. **Run a simulation harness for idea A (importance-weighted
   calibration).**  This is the cheapest to test and has the most
   direct theoretical motivation in our setting.  Plug into the
   existing `_sim_calibration_ideas.py` framework.  Should take
   half a day.

2. **If A doesn't work, try idea C (item-embedding covariate).**
   Slightly more involved (need to extend `_Calibrator` state) but
   inherits the high-coverage structural advantage that the rejected
   per-subject and kNN channels lacked.

3. **Defer ideas B (IRT-anchor) and D (Mix-n-Match composition)**
   until A or C shows something — these have bigger implementation
   surface area and shouldn't be tried until we know the cheap ideas
   have been exhausted.

4. **Schedule idea E (focal-loss training) as a separate research
   line**, on a dedicated GPU.  This is the biggest single ceiling
   but takes the longest to evaluate.

5. **Idea F (unlabeled-test adaptation) is speculative but the only
   one that adds a *new signal channel* rather than rearranging the
   existing one.**  Worth implementing once the cheaper ideas have
   resolved.

The bar set by red-teaming PP_CONSERVATIVE is: only ship if a
candidate wins on every honest regime, doesn't lose by more than
the noise floor on any adversarial regime, and the magnitude beats
~0.001 nats with statistical significance over 60 paired trials.

---

## Bibliography (papers cited above)

  - Park, Krishnan, Doersch, Lipton.  *Calibrated Prediction with
    Covariate Shift via Unsupervised Domain Adaptation.*  AISTATS
    2020.  https://proceedings.mlr.press/v108/park20b/park20b.pdf
  - Mostafavi, Glas, Sotaridona.  *An Optimized Bayesian Hierarchical
    Two-Parameter Logistic Model for Small-Sample Item Calibration.*
    Educ Psychol Meas, 2020.  https://pmc.ncbi.nlm.nih.gov/articles/PMC7262992/
  - Aoyama, Yamazaki.  *Using Auxiliary Item Information in the Item
    Parameter Estimation of a Graded Response Model for a Small to
    Medium Sample Size: Empirical Versus Hierarchical Bayes
    Estimation.*  Appl Psychol Meas, 2023.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10664746/
  - Zhang, Kjellström, Mandt.  *Mix-n-Match: Ensemble and
    Compositional Methods for Uncertainty Calibration in Deep
    Learning.*  ICML 2020.
    http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf
  - Mukhoti, Kulharia, Sanyal, Golodetz, Torr, Dokania.  *Calibrating
    Deep Neural Networks Using Focal Loss.*  NeurIPS 2020.
    https://torrvision.github.io/focal_calibration/
  - Devic, Korolova, Kearns, Vasilyev, Roth.  *When is
    Multicalibration Post-Processing Necessary?*  arXiv:2406.06487,
    2024.  https://arxiv.org/abs/2406.06487
  - Angelman et al.  *Calibrating Without Labels: Source-Free
    Conformal Prediction Using Pseudo-Labels.*  PMLR 266 (2025).
    https://proceedings.mlr.press/v266/angelman25a.html
  - Gupta, Rahimi, Ajanthan, Mensink, Sminchisescu, Hartley.
    *Calibration of Neural Networks Using Splines.*  arXiv:2006.12800,
    2020.  https://arxiv.org/abs/2006.12800
  - Kweon, Park, Yu, et al.  *Obtaining Calibrated Probabilities with
    Personalized Ranking Models.*  RecSys 2022.  (Anchor reference
    from the original handoff.)
