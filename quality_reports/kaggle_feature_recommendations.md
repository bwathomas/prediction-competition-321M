# Kaggle feature-engineering canon → cold-start passrate prediction

**Date:** 2026-06-04
**Task grounding:** predict P(subject passes a benchmark item), **item cold-start** (test
items unseen, subjects seen). Raw material: per-item / per-subject TEXT EMBEDDINGS (frozen
encoder), the SUBJECT×ITEM training passrate matrix, k-means clusters on item embeddings,
CSV metadata. Models: 2-layer linear-stacked ensemble; agent chooses feature ablations.

**Hard constraints honored by every recommendation below.** Content- or
data(label/embedding)-derived only. NO feature from another model's learned parameters or
outputs (no LLM-judge, no solver-proxy, no fitted IRT θ/u, no trained-member predictions,
no PCA-of-embeddings tail). Every feature must be computable for an item with **zero
training responses** — it may use the item's text/embedding and the *subject's training
history over OTHER items*, but never the test item's own labels. The pivotal cold-start
trick is that all "item difficulty" signal must be carried over from *labeled neighbor
items* (kNN-on-embeddings), never from the item itself — exactly the existing `nn__*`
design. Our job is to widen and sharpen that lever, not invent a new one.

Dropout-class tags follow `aide/feature_catalog.py`:
`neutral_item` (item CONTENT, never masked), `subject_proxy` (encodes which subject /
its ability / its neighbor passrates), `benchmark_proxy` (encodes which benchmark/condition).
A pair feature keyed on the subject's history is `subject_proxy`; an item-only embedding
statistic is `neutral_item`.

---

## 1. The Kaggle feature-engineering canon for tabular + embedding data (2023–2025)

Synthesis of current strong-competitor practice, with sources.

1. **Groupby aggregations are the single highest-ROI family.** The NVIDIA Kaggle
   Grandmasters playbook names `groupby(COL1)[COL2].agg(STAT)` the most powerful technique,
   with `STAT ∈ {mean, std, count, min, max, nunique, skew}`, plus **quantile** features
   (they used percentiles `[5,10,40,45,55,60,90,95]` → 8 columns per groupby) and
   **histogram-bin counts** per group. cuDF lets winners generate/test **10,000+** such
   features and keep the survivors. [NVIDIA playbook], [NVIDIA cuDF first-place].

2. **Target / mean encoding with smoothing** is "one of the secret sauces." The smoothing
   recipe blends the in-category mean with the global mean,
   `enc = (n·ȳ_cat + m·ȳ_global)/(n+m)`, so rare categories shrink to the prior and unseen
   categories get the prior outright. Higher `m` = stronger shrinkage. **Out-of-fold /
   K-fold cross-fitting** (encode each fold from the *other* folds) is mandatory to avoid
   leakage — sklearn's `TargetEncoder` does this by construction.
   [Train in Data], [Kaggle target-encoding course], [H2O target-encoding], [K-Fold TE].

3. **Count / frequency encoding** — replace a category by its train frequency; cheap, robust
   for high-cardinality keys, and orthogonal to the target-encoded value (it carries
   *support*, not *mean*). Often paired with the mean encoding as the "n" companion column.
   [NVIDIA cuDF], [4 ways to encode high-cardinality].

4. **Interaction features** — concatenate/cross categorical columns (e.g. all 28 pairs of 8
   categoricals) then target- or count-encode the cross; **ratio features** of engineered
   columns (`count/nunique`, `std/count`) are explicitly called out. [NVIDIA cuDF].

5. **Embedding-distance & kNN-target features** — winners build a kNN index on
   embeddings and pull **neighbor-target aggregates** (mean/weighted-mean/std of the K
   nearest neighbors' labels) as features; this is the standard way to inject "similar-row
   behavior" into a GBDT/linear model. Cosine-distance to references, distance to the K-th
   neighbor (OOD signal), and similarity-weighted means are all common.
   [NVIDIA cuDF kNN/GraphSAGE], [Analytics Vidhya FE].

6. **Cluster features** — k-means/GMM on embeddings, then cluster-id (one-hot or target-
   encoded), distance-to-centroid, **soft/responsibility membership**, and per-cluster
   target aggregates. In the embedding-OOD literature, soft-cluster responsibilities and
   their entropy, and distance to the 2nd-nearest centroid (margin), are standard
   typicality/OOD signals. [Cluster-game OOD], [Soft-clustering NNK], [DEC summary].

7. **Dimensionality treatments** — raw embeddings are fed directly to NN heads, but for
   tabular/GBDT members teams compress via PCA/UMAP/autoencoder OR replace the full vector
   with *derived scalars* (distances, densities, cluster stats). NOTE: our constraints
   forbid the PCA-of-embeddings tail; the sanctioned compression here is **derived geometric
   scalars** (distances, local density, LID), which are not "another model's parameters."

8. **Validation discipline** — every label-derived feature is computed **out-of-fold** and,
   when the test regime is "unseen entity," the CV split must mimic it (here: **group-by-item
   / group-by-benchmark folds** so neighbor-target features are estimated on held-out items,
   matching cold-start). This is the difference between a feature that helps CV-and-LB vs.
   one that only helps CV. [Train in Data], [K-Fold TE], [NVIDIA playbook].

**Mapping to our setting.** We already do (2),(3),(5),(6) partially and (4) lightly. The
biggest *unexploited* canon items for a **cold-start log-loss** task are: (a) **multi-K and
multi-radius** neighbor profiles (we use a single K=16), (b) **groupby-quantile / shape
aggregations** of neighbor labels (we only have mean/std/entropy), (c) **soft-cluster +
margin + local-density** geometry, and (d) **calibration-residual** style features (predict
from neighbors, measure how off the subject usually is). Everything below is a *derivative*
of machinery we already own (the FAISS index, the passrate CSR tables, the k-means
centroids), so incremental engineering cost is low.

---

## 2. NN-derived derivatives to add

All are functions of `(neighbor_passrates, masks, similarities)` for the query item plus the
**subject's** training history — cold-start-safe (the query item contributes only its
embedding, never its own labels). Existing `nn__*` cells (23) are NOT repeated. Format:
**name** — definition — *axis* — `dropout_class`.

**Multi-K profiles (cheap; biggest single win).** Today every cell uses one K=16. Recompute
the core aggregates at several K and feed the *profile* + its *slope*:

- **`nn__passrate_mean_K{4,8,32,64}`** — neighbor passrate mean at multiple K. — pair —
  `subject_proxy`.
- **`nn__passrate_K_slope`** — (mean@K_large − mean@K_small): does difficulty estimate drift
  as the neighborhood widens? A locally-hard item embedded in an easy region. — pair —
  `subject_proxy`.
- **`nn__coverage_K_slope`** — how fast labeled-coverage grows with K; proxies how isolated
  the item is in the subject's answered set. — pair — `subject_proxy`.

**Distance-weighted shape / variance (groupby-quantile canon on neighbor labels).** We have
mean/std/entropy but no *shape*:

- **`nn__passrate_weighted_var`** — similarity-weighted variance of neighbor labels
  `Σwᵢ(yᵢ−ȳ_w)²/Σwᵢ`; sharper local-difficulty uncertainty than the unweighted `passrate_std`.
  — pair — `subject_proxy`.
- **`nn__passrate_q25 / q50 / q75 / iqr`** — quantiles of the K neighbor labels (the
  groupby-quantile canon). Median is robust to one mislabeled neighbor; IQR is a second
  uncertainty channel. — pair — `subject_proxy`.
- **`nn__frac_neighbors_pass`** — fraction of labeled neighbors with mean-label > 0.5 (a
  hard-vote companion to the soft mean; trees split on it cleanly). — pair — `subject_proxy`.

**Multi-radius agreement entropy (local difficulty texture).** `neighbor_label_entropy`
exists at one radius; add radius-resolved versions:

- **`nn__label_entropy_innerK / outerK`** — Bernoulli entropy of the passrate among the
  closest ⌊K/2⌋ vs the farthest ⌊K/2⌋ neighbors; a gap means difficulty is locally smooth
  vs. jumpy. — pair — `subject_proxy`.
- **`nn__agreement_radius_decay`** — |mean_innerK − mean_outerK|; how fast the difficulty
  estimate decays with distance. — pair — `subject_proxy`.

**Local difficulty RANK / calibration (the strongest novel lever).** These ask "given how
this subject usually does, is this neighborhood unusually hard *for them*?":

- **`nn__local_difficulty_rank`** — neighbor passrate mean **minus the subject's global
  mean** (`subject_mean`): item-region difficulty *relative to* the subject's baseline. This
  is the single most predictive transform for a per-(subject,item) log-loss task and is
  currently only available implicitly (model must subtract two features). — pair —
  `subject_proxy`.
- **`nn__calibration_residual`** — over the subject's *labeled* neighbors, mean of
  `(actual_label − global_item_passrate)` i.e. how much this subject beats/trails the crowd
  on items like this; carried to the unseen item via its neighborhood. Pure cold-start: uses
  the subject's history on neighbor items only. — pair — `subject_proxy`.
- **`nn__subjfamily_minus_global_gap`** — `passrate_family_conditional − passrate_mean`
  (already-computed cells): is the subject's *family* unusually good/bad in this region vs
  the global neighborhood? A free interaction of two existing cells. — pair — `subject_proxy`
  (+`benchmark_proxy` only if family is benchmark-correlated; tag `subject_proxy`).

**Embedding-geometry / OOD reliability gates (neutral_item — never masked, pure item geometry).**
These tell the model *how much to trust* the NN cells; they use only the index geometry, no
labels, so they are `neutral_item` and survive subject dropout:

- **`nn__local_density`** — mean similarity of the item to its K nearest *training* items
  (denser region ⇒ NN estimate more reliable). — item — `neutral_item`.
- **`nn__lid_estimate`** — local intrinsic dimensionality (MLE / Hill estimator on the K
  neighbor distances, `LID = −[ (1/K)Σ ln(rᵢ/r_K) ]⁻¹`); high LID ⇒ the neighborhood is
  spread across many directions ⇒ a worse 1-NN proxy. A documented OOD/typicality signal.
  — item — `neutral_item`. [LID refs]
- **`nn__dist_gap_1_to_K`** — `top1_similarity − distance_to_kth_neighbor` reframed as the
  *absolute* radius of the neighborhood (item isolation). — item — `neutral_item`.
- **`nn__reciprocal_neighbor_frac`** — fraction of the item's K neighbors that also list it
  among *their* K neighbors (mutual-kNN). Low reciprocity ⇒ the item is a hub-or-outlier and
  its neighbor labels are less trustworthy; a classic hubness diagnostic. — item —
  `neutral_item`.

---

## 3. Cluster-derived derivatives to add

We currently have: cluster one-hot, `cluster_id`, `cd__centroid_dist_{i}` (top-m centroid
distances), `m2_cluster_mean` (cluster difficulty target-enc), and the NN cell
`cluster_passrate_subject_query`. Derivatives:

**Soft / margin geometry (all `neutral_item` — pure item-embedding geometry):**

- **`clu__soft_responsibility_{top3}`** — softmax over (−distance) to the nearest few
  centroids (Student-t or RBF kernel); a continuous membership vector that beats a hard
  one-hot for boundary items. — item — `neutral_item`. [Soft-clustering NNK, DEC]
- **`clu__margin_1to2`** — distance to nearest centroid − distance to 2nd-nearest
  (assignment confidence). — item — `neutral_item`.
- **`clu__responsibility_entropy`** — entropy of the soft-responsibility vector
  (how "between clusters" the item is). — item — `neutral_item`.
- **`clu__typicality`** — item distance-to-its-centroid **z-scored within its cluster**
  (silhouette-lite: small ⇒ prototypical, large ⇒ atypical member). — item — `neutral_item`.
- **`clu__size_log1p`** — log size of the item's assigned cluster (rarity / support of the
  region). — item — `neutral_item`.

**Multi-resolution cluster ids (`neutral_item`):**

- **`clu_id_K{small,large}`** — re-fit k-means at a coarse and a fine K (or one
  agglomerative cut); coarse ids generalize, fine ids localize. Feed both id columns. — item
  — `neutral_item`.

**Within-cluster subject behavior (target-encoded; `subject_proxy`):**

- **`clu__subject_minus_cluster_gap`** — `cluster_passrate_subject_query − m2_cluster_mean`:
  subject's affinity for this *region* relative to the region's global difficulty. Direct
  cold-start difficulty-for-this-subject signal; a free interaction of two existing
  features. — pair — `subject_proxy`.
- **`clu__subject_cluster_affinity_profile`** — for the query item's cluster, the subject's
  shrunk passrate **and** observation count in that cluster (the `n` companion). The count
  gates how much to trust the affinity. — pair — `subject_proxy`.
- **`clu__cluster_difficulty_std`** — within-cluster **std** of item passrates (groupby-std
  canon): is this a homogeneous-difficulty region or a mixed one? Complements the
  cluster-mean. — item — `neutral_item` (label-derived but item-region, not subject).

**Cluster-conditional, soft (`subject_proxy`):**

- **`clu__soft_weighted_subject_passrate`** — subject's cluster passrates **blended by the
  soft-responsibility weights** instead of hard assignment; smoother than the hard
  `cluster_passrate_subject_query` for boundary items. — pair — `subject_proxy`.

---

## 4. Other high-value additions (canon items we under-use)

**Count / frequency encodings (the missing "n" companions; cheap):**

- **`cnt__subject_obs_count_log1p`** already exists as `subject_obs_count`; add
  **`cnt__neighbor_subject_support`** = number of the query item's K neighbors the subject
  has actually answered (raw support behind every `nn__*` mean). Gates NN trust. — pair —
  `subject_proxy`.
- **`cnt__cluster_obs_count_log1p`** — subject's #observations inside the query cluster
  (companion to `clu__subject_cluster_affinity`). — pair — `subject_proxy`.

**Metadata groupby aggregations (CSV; the playbook's #1 family, currently absent over
metadata):**

- **`grp__org_passrate_mean/std`**, **`grp__family_passrate_mean/std`**,
  **`grp__macro_family_passrate_mean/std`** — shrunk groupby-aggregates of the subject's
  passrate by `organization / family / macro_family` (mean **and** std — std is missing
  today, only means exist via `m2_subj_*`). — subject — `subject_proxy`.
- **`grp__topic_passrate_mean`**, **`grp__age_bin_passrate_mean`** — benchmark-side groupby
  target-encodings over `topic` and binned `age` (with smoothing + OOF). Benchmark-conditional,
  so dropout-masked. — benchmark — `benchmark_proxy`.
- **`grp__has_conditions × topic`** interaction target-encoding (cross of two CSV
  categoricals → mean). — benchmark — `benchmark_proxy`.

**Interaction / ratio features (canon #4; free, from existing columns):**

- **`int__subjectmean_x_clusterdiff`** — `subject_mean × m2_cluster_mean` (or their gap):
  lets the *linear* stacker express "strong subject on an easy region" without a hidden
  layer. — pair — `subject_proxy`.
- **`ratio__coverage_over_lid`** — `nn__coverage / (1+nn__lid_estimate)`: a single
  trust-scalar combining label support and geometric reliability. — pair — `subject_proxy`.
- **`int__params_x_release`** — `log_params × release_date` (CSV-only subject interaction;
  capability tends to scale with both). — subject — `subject_proxy`.

**Smoothing-scheme upgrades to existing target encodings (no new columns, better columns):**

- Add a **second smoothing strength** for `m2_*` mean-encodings (e.g. a high-`m` and a
  low-`m` variant) and let the ensemble pick — different members benefit from different
  shrinkage; this is a known diversity source. — same axes/classes as the parents.
- Ensure all label-derived features (`nn__*`, `m2_*`, `grp__*`, `clu__*subject*`) are
  computed with **group-by-item OOF folds** so their CV behavior matches cold-start test;
  this is a correctness fix more than a new feature, but it is the difference between these
  features generalizing or not.

---

## 5. Ranked shortlist — top 8 by value-for-effort (cold-start log-loss)

Ranked by expected log-loss gain ÷ engineering cost. **(D)** flags the ones most likely to
add **ensemble diversity** (errors uncorrelated with the existing mean/weighted-mean NN
cells), which the 2-layer stack converts directly into gain.

1. **`nn__local_difficulty_rank`** (neighbor-mean − subject_mean), pair / `subject_proxy`.
   The canonical per-(subject,item) signal; one subtraction; almost certainly the largest
   single win.
2. **`nn__calibration_residual`** — subject's beat-the-crowd margin carried via neighbors,
   pair / `subject_proxy`. **(D)** — orthogonal to absolute-passrate cells; pure relative
   skill, a different error mode.
3. **Multi-K passrate profile + `nn__passrate_K_slope`**, pair / `subject_proxy`. **(D)** —
   cheap (reuse the FAISS query), exposes neighborhood-scale sensitivity the single-K cells
   cannot.
4. **`clu__subject_minus_cluster_gap`** (+ its `n` count), pair / `subject_proxy`. Free
   interaction of two existing features; region-difficulty-for-this-subject; complements the
   NN view with the cluster view.
5. **`nn__lid_estimate` + `nn__local_density`**, item / `neutral_item`. **(D)** — reliability
   gates; never masked, so they keep working under subject dropout and let the stack
   down-weight bad NN estimates (heteroscedastic-aware). Distinct geometric signal.
6. **Metadata groupby std aggregates** (`grp__family_passrate_std`, `grp__org_passrate_std`,
   …), subject / `subject_proxy`. The playbook's #1 family; we have means but no dispersion;
   low cost via groupby. **(D)** for the std channel.
7. **Soft-cluster geometry** (`clu__soft_responsibility_top3`, `clu__margin_1to2`,
   `clu__responsibility_entropy`), item / `neutral_item`. **(D)** — smooth boundary-item
   signal orthogonal to hard cluster one-hot; survives dropout.
8. **`nn__passrate_q50/iqr` + `nn__passrate_weighted_var`**, pair / `subject_proxy`. Robust
   shape of the neighbor-label distribution; cheap; guards against single-neighbor noise.

**Diversity note.** Items 2, 3, 5, 6(std), 7 are the strongest diversifiers: each introduces
an error mode (relative-skill, scale-sensitivity, geometric-reliability, dispersion,
boundary-smoothness) that the existing mean/weighted-mean/top1 NN cells do not capture, so a
linear stacker should give them non-trivial weight even where they are individually weaker
than the absolute-passrate cells. **Caveat on every label-derived item (1–4, 6, 8):** they
MUST be computed with **group-by-item out-of-fold folds**, or their CV gain will not transfer
to cold-start test items.

---

## Sources

- [NVIDIA — The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [NVIDIA — Grandmaster Pro Tip: Winning First Place with Feature Engineering Using cuDF pandas](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- [Train in Data — Target Encoder: a powerful categorical encoding method (smoothing)](https://www.blog.trainindata.com/target-encoder-a-powerful-categorical-encoding-method/)
- [Kaggle Courses — Target Encoding (smoothing, leakage)](https://fralfaro.github.io/kaggle-courses/kaggle/07.%20Feature%20Engineering/tutorial/06.%20target-encoding/)
- [H2O — Target Encoding (blending / smoothing parameters)](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/target-encoding.html)
- [Medium/Accredian — K-Fold Target Encoding for High-Cardinality Features](https://medium.com/accredian/k-fold-target-encoding-for-high-cardinality-features-b7f82f6efb77)
- [Towards Data Science — 4 ways to encode categorical features with high cardinality](https://towardsdatascience.com/4-ways-to-encode-categorical-features-with-high-cardinality-1bc6d8fd7b13/)
- [Analytics Vidhya — Feature Engineering for Kaggle Competition (kNN/embedding features)](https://medium.com/analytics-vidhya/feature-engineering-for-kaggle-competition-5616196bf274)
- [arXiv 2203.08549 — Is it all a cluster game? OOD detection based on clustering in the embedding space](https://arxiv.org/pdf/2203.08549)
- [arXiv 2407.13141 — OOD Detection through Soft Clustering with Non-Negative Kernel Regression](https://arxiv.org/pdf/2407.13141)
- [Local Intrinsic Dimensionality III: Density and Similarity (SpringerLink)](https://link.springer.com/chapter/10.1007/978-3-030-60936-8_19)
- [arXiv 2411.16145 — Local Intrinsic Dimensionality for embeddings](https://arxiv.org/html/2411.16145v1)
