"""Member 2 v5: attribute + neighborhood tree (no M1 leakage).

Design rationale (see chat-thread literature review): v3/v4 failed
Gate 3d because ``logit_p_m1`` as the first feature caused the GBDT
to memorize Member 1's prediction (corr ~ 0.96 with M1) and the
cumulative boost saturated a fraction of predictions to p=1.0. v5
removes ALL M1-derived inputs and replaces them with item-attribute,
historical-count, target-encoded, and neighborhood signals -- the
literature-standard cold-start GBDT recipe (Facebook 2014; LightGBM
PR #3234; Wang et al. 2023; AliBoost KDD 2025).

The output is structurally orthogonal to Member 1 by construction
(M1's prediction is never an input), so the stacker can extract
incremental value from v5's tree-shaped decision boundaries even
when v5's standalone NLL is modest.

Schema (49 numeric columns, no native categoricals so the compiled
``gbdt_member`` numpy walker stays in its supported ``<=`` regime;
manual OOF target encoding via the builder replaces LightGBM's native
categorical handling):

  Bucket B  -- item content (19 cols, cold-item-stable):
    0  pool_token_len
    1  pool_char_len
    2  pool_has_latex
    3  pool_has_code
    4  pool_n_questions
    5  pool_n_numbers
    6  pool_is_multiple_choice
    7  pool_n_choices
    8  pool_lang_en
    9  centroid_dist_0
   10  centroid_dist_1
   11  centroid_dist_2
   12  centroid_dist_3
   13  centroid_dist_4
   14  centroid_dist_5
   15  centroid_dist_6
   16  centroid_dist_7
   17  benchmark_age_value
   18  benchmark_age_mask

  Bucket C  -- historical counts (9 cols, no target leakage):
   19  subject_obs_count_log1p
   20  cluster_obs_count_log1p
   21  bc_obs_count_log1p
   22  macro_family_obs_count_log1p
   23  organization_obs_count_log1p
   24  family_obs_count_log1p
   25  subj_x_cluster_obs_count_log1p
   26  subj_x_bc_obs_count_log1p
   27  bc_redacted_mask

  Bucket TE  -- Bayesian-shrunk OOF target encodings (8 cols):
   28  subject_passrate_meanenc
   29  cluster_passrate_meanenc
   30  bc_passrate_meanenc
   31  macro_family_passrate_meanenc
   32  organization_passrate_meanenc
   33  family_passrate_meanenc
   34  subj_x_cluster_passrate_meanenc
   35  subj_x_bc_passrate_meanenc

  Bucket U  -- unknown-id inductive bias flags (6 cols):
   36  is_unknown_subject
   37  is_unknown_cluster
   38  is_unknown_bc
   39  is_unknown_macro_family
   40  is_unknown_organization
   41  is_unknown_family

  Bucket D  -- neighborhood aggregates (7 cols, from nn_features.py):
   42  nn_passrate_weighted_mean
   43  nn_coverage
   44  nn_mean_similarity
   45  nn_effective_neighbor_count
   46  nn_passrate_subject_conditional
   47  nn_passrate_benchmark_conditional
   48  nn_distance_to_kth_neighbor

Total: 49 features.

Trained as a DIRECT binary GBDT (no init_score, no residual framing,
output = sigmoid(tree_raw)). See ``fit_gbdt_member(init_pred_train=None)``.
The bounded-leaf property of a binary objective, combined with the
weaker per-feature signal (no logit_p_m1 amplifier), prevents the
cumulative-logit saturation that broke v4.

This module follows the project convention of using `warnings.warn`
for soft / numerical-quality issues and reserving exceptions for
hard programming errors (shape mismatches, NaN/Inf inputs). Callers
should listen for the `Member2V5Warning` category to detect feature
quality issues without halting the pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


_EPS_PROB = 1e-7
_DEFAULT_SMOOTHING = 30.0


class Member2V5Warning(UserWarning):
    """Soft data-quality issues in the v5 feature pipeline.

    Callers (notebook gates, tests) can filter on this category to
    convert warnings back to errors during development without
    catching unrelated warnings from numpy / lightgbm.
    """


# ---------------------------------------------------------------------------
# Locked column layout. Do NOT reorder -- LightGBM trees index features
# by integer column position. Any change here requires bumping the cache
# prefix in the notebook (``gbdt_oof_v5_attr_nn_v1`` and
# ``member2_v5_attr_nn_global_v1``).
# ---------------------------------------------------------------------------
MEMBER2_V5_FEATURE_NAMES: Tuple[str, ...] = (
    # Bucket B: item content (19)
    "pool_token_len",
    "pool_char_len",
    "pool_has_latex",
    "pool_has_code",
    "pool_n_questions",
    "pool_n_numbers",
    "pool_is_multiple_choice",
    "pool_n_choices",
    "pool_lang_en",
    "centroid_dist_0",
    "centroid_dist_1",
    "centroid_dist_2",
    "centroid_dist_3",
    "centroid_dist_4",
    "centroid_dist_5",
    "centroid_dist_6",
    "centroid_dist_7",
    "benchmark_age_value",
    "benchmark_age_mask",
    # Bucket C: historical counts (9)
    "subject_obs_count_log1p",
    "cluster_obs_count_log1p",
    "bc_obs_count_log1p",
    "macro_family_obs_count_log1p",
    "organization_obs_count_log1p",
    "family_obs_count_log1p",
    "subj_x_cluster_obs_count_log1p",
    "subj_x_bc_obs_count_log1p",
    "bc_redacted_mask",
    # Bucket TE: target encodings (8)
    "subject_passrate_meanenc",
    "cluster_passrate_meanenc",
    "bc_passrate_meanenc",
    "macro_family_passrate_meanenc",
    "organization_passrate_meanenc",
    "family_passrate_meanenc",
    "subj_x_cluster_passrate_meanenc",
    "subj_x_bc_passrate_meanenc",
    # Bucket U: unknown flags (6)
    "is_unknown_subject",
    "is_unknown_cluster",
    "is_unknown_bc",
    "is_unknown_macro_family",
    "is_unknown_organization",
    "is_unknown_family",
    # Bucket D: NN aggregates (7)
    "nn_passrate_weighted_mean",
    "nn_coverage",
    "nn_mean_similarity",
    "nn_effective_neighbor_count",
    "nn_passrate_subject_conditional",
    "nn_passrate_benchmark_conditional",
    "nn_distance_to_kth_neighbor",
)
M2V5_FEATURE_DIM = len(MEMBER2_V5_FEATURE_NAMES)

# Bucket-boundary column indices for callers that want to slice by group.
M2V5_BUCKET_B_END = 19   # item content
M2V5_BUCKET_C_END = 28   # + counts (9)
M2V5_BUCKET_TE_END = 36  # + target encodings (8)
M2V5_BUCKET_U_END = 42   # + unknown flags (6)
M2V5_BUCKET_D_END = 49   # + NN aggregates (7)

if M2V5_FEATURE_DIM != 49:
    # The module-level constant is what gets pickled into cache keys;
    # warn loudly if it ever drifts from the documented contract.
    warnings.warn(
        f"MEMBER2_V5_FEATURE_NAMES has {M2V5_FEATURE_DIM} entries, "
        "expected 49. Either the schema was edited without updating "
        "this constant, or the constant is stale. Bump cache prefix "
        "before training.",
        Member2V5Warning,
        stacklevel=2,
    )


# Index of the 7 NN aggregate columns inside the upstream 23-col NN matrix.
# Order matches the trailing 7 names of MEMBER2_V5_FEATURE_NAMES.
M2V5_NN_SOURCE_INDICES: Tuple[int, ...] = (
    1,   # passrate_weighted_mean
    3,   # coverage
    6,   # mean_similarity
    8,   # effective_neighbor_count
    15,  # passrate_subject_conditional
    19,  # passrate_benchmark_conditional
    14,  # distance_to_kth_neighbor
)


# ---------------------------------------------------------------------------
# Feature-builder state. Same conceptual shape as v3's builder but with
# the addition of per-id target-encoded subject passrate and the two 2-D
# cell mean-encoded passrates (subj x bc, subj x cluster). No M1 inputs.
# ---------------------------------------------------------------------------
@dataclass
class Member2V5FeatureBuilder:
    """Per-id lookup tables needed to materialize v5 features at runtime.

    All ``*_passrate`` arrays are Bayesian-shrunk:

        passrate_id = (sum_y_in_id + smoothing * global_mean)
                      / (n_in_id + smoothing)

    ``*_log1p_n`` arrays store ``log1p(observation_count)`` so the
    tree can split on log-scale uncertainty proxies. Subject -> trait
    id maps (``subject_to_*_id``) let v5 derive macro_family /
    organization / family ids per row from the row's subject_id, so
    callers do not need to compute the trait ids themselves.

    The builder also stores ``pool_mean`` / ``pool_std`` so the
    pool-feature columns can be z-scored at apply time using the
    SAME statistics fit at training time (prevents val/test
    distribution leakage through standardisation).
    """

    # Per-id log1p obs counts (1-D arrays, length = vocab size).
    subj_log1p_n: np.ndarray
    cluster_log1p_n: np.ndarray
    bc_log1p_n: np.ndarray
    macro_log1p_n: np.ndarray
    org_log1p_n: np.ndarray
    fam_log1p_n: np.ndarray

    # Per-id Bayesian-shrunk pass rates (1-D arrays, length = vocab size).
    subj_passrate: np.ndarray
    cluster_passrate: np.ndarray
    bc_passrate: np.ndarray
    macro_passrate: np.ndarray
    org_passrate: np.ndarray
    fam_passrate: np.ndarray

    # 2-D cell counts and mean-encoded pass rates. Shape:
    #   subj_x_bc_*       -> (n_subjects, n_bcs)
    #   subj_x_cluster_*  -> (n_subjects, n_clusters)
    subj_x_bc_log1p_n: np.ndarray
    subj_x_bc_passrate: np.ndarray
    subj_x_cluster_log1p_n: np.ndarray
    subj_x_cluster_passrate: np.ndarray

    # Subject -> trait id map (shape [n_subjects]).
    subject_to_macro_family_id: np.ndarray
    subject_to_organization_id: np.ndarray
    subject_to_family_id: np.ndarray

    n_subjects: int
    n_clusters: int
    n_bcs: int
    n_macro_families: int
    n_organizations: int
    n_families: int
    global_mean: float
    smoothing: float

    n_train_rows_fit: int = 0
    fit_method: str = "bayes_shrunk_attr_nn_v1"

    def __post_init__(self) -> None:
        # Shape audits. These guard against silent feature-pipeline drift;
        # we hard-fail (ValueError) because mismatched shapes produce
        # garbage matrix output that downstream gates cannot detect.
        for name, arr, length in [
            ("subj_log1p_n", self.subj_log1p_n, self.n_subjects),
            ("cluster_log1p_n", self.cluster_log1p_n, self.n_clusters),
            ("bc_log1p_n", self.bc_log1p_n, self.n_bcs),
            ("macro_log1p_n", self.macro_log1p_n, self.n_macro_families),
            ("org_log1p_n", self.org_log1p_n, self.n_organizations),
            ("fam_log1p_n", self.fam_log1p_n, self.n_families),
            ("subj_passrate", self.subj_passrate, self.n_subjects),
            ("cluster_passrate", self.cluster_passrate, self.n_clusters),
            ("bc_passrate", self.bc_passrate, self.n_bcs),
            ("macro_passrate", self.macro_passrate, self.n_macro_families),
            ("org_passrate", self.org_passrate, self.n_organizations),
            ("fam_passrate", self.fam_passrate, self.n_families),
            (
                "subject_to_macro_family_id",
                self.subject_to_macro_family_id,
                self.n_subjects,
            ),
            (
                "subject_to_organization_id",
                self.subject_to_organization_id,
                self.n_subjects,
            ),
            (
                "subject_to_family_id",
                self.subject_to_family_id,
                self.n_subjects,
            ),
        ]:
            if arr.shape != (int(length),):
                raise ValueError(
                    f"v5 builder: {name} shape {arr.shape} != ({int(length)},)"
                )
        for name, arr, expected in [
            (
                "subj_x_bc_log1p_n",
                self.subj_x_bc_log1p_n,
                (self.n_subjects, self.n_bcs),
            ),
            (
                "subj_x_bc_passrate",
                self.subj_x_bc_passrate,
                (self.n_subjects, self.n_bcs),
            ),
            (
                "subj_x_cluster_log1p_n",
                self.subj_x_cluster_log1p_n,
                (self.n_subjects, self.n_clusters),
            ),
            (
                "subj_x_cluster_passrate",
                self.subj_x_cluster_passrate,
                (self.n_subjects, self.n_clusters),
            ),
        ]:
            if arr.shape != expected:
                raise ValueError(
                    f"v5 builder: {name} shape {arr.shape} != {expected}"
                )

    # I/O -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Numpy-arrays-only dict suitable for ``np.savez``."""
        return {
            "subj_log1p_n": self.subj_log1p_n.astype(np.float32, copy=False),
            "cluster_log1p_n": self.cluster_log1p_n.astype(np.float32, copy=False),
            "bc_log1p_n": self.bc_log1p_n.astype(np.float32, copy=False),
            "macro_log1p_n": self.macro_log1p_n.astype(np.float32, copy=False),
            "org_log1p_n": self.org_log1p_n.astype(np.float32, copy=False),
            "fam_log1p_n": self.fam_log1p_n.astype(np.float32, copy=False),
            "subj_passrate": self.subj_passrate.astype(np.float32, copy=False),
            "cluster_passrate": self.cluster_passrate.astype(np.float32, copy=False),
            "bc_passrate": self.bc_passrate.astype(np.float32, copy=False),
            "macro_passrate": self.macro_passrate.astype(np.float32, copy=False),
            "org_passrate": self.org_passrate.astype(np.float32, copy=False),
            "fam_passrate": self.fam_passrate.astype(np.float32, copy=False),
            "subj_x_bc_log1p_n": self.subj_x_bc_log1p_n.astype(
                np.float32, copy=False
            ),
            "subj_x_bc_passrate": self.subj_x_bc_passrate.astype(
                np.float32, copy=False
            ),
            "subj_x_cluster_log1p_n": self.subj_x_cluster_log1p_n.astype(
                np.float32, copy=False
            ),
            "subj_x_cluster_passrate": self.subj_x_cluster_passrate.astype(
                np.float32, copy=False
            ),
            "subject_to_macro_family_id": self.subject_to_macro_family_id.astype(
                np.int32, copy=False
            ),
            "subject_to_organization_id": self.subject_to_organization_id.astype(
                np.int32, copy=False
            ),
            "subject_to_family_id": self.subject_to_family_id.astype(
                np.int32, copy=False
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        n_subjects: int,
        n_clusters: int,
        n_bcs: int,
        n_macro_families: int,
        n_organizations: int,
        n_families: int,
        global_mean: float,
        smoothing: float,
        n_train_rows_fit: int = 0,
        fit_method: str = "bayes_shrunk_attr_nn_v1",
    ) -> "Member2V5FeatureBuilder":
        return cls(
            subj_log1p_n=np.asarray(data["subj_log1p_n"], dtype=np.float32),
            cluster_log1p_n=np.asarray(data["cluster_log1p_n"], dtype=np.float32),
            bc_log1p_n=np.asarray(data["bc_log1p_n"], dtype=np.float32),
            macro_log1p_n=np.asarray(data["macro_log1p_n"], dtype=np.float32),
            org_log1p_n=np.asarray(data["org_log1p_n"], dtype=np.float32),
            fam_log1p_n=np.asarray(data["fam_log1p_n"], dtype=np.float32),
            subj_passrate=np.asarray(data["subj_passrate"], dtype=np.float32),
            cluster_passrate=np.asarray(data["cluster_passrate"], dtype=np.float32),
            bc_passrate=np.asarray(data["bc_passrate"], dtype=np.float32),
            macro_passrate=np.asarray(data["macro_passrate"], dtype=np.float32),
            org_passrate=np.asarray(data["org_passrate"], dtype=np.float32),
            fam_passrate=np.asarray(data["fam_passrate"], dtype=np.float32),
            subj_x_bc_log1p_n=np.asarray(
                data["subj_x_bc_log1p_n"], dtype=np.float32
            ),
            subj_x_bc_passrate=np.asarray(
                data["subj_x_bc_passrate"], dtype=np.float32
            ),
            subj_x_cluster_log1p_n=np.asarray(
                data["subj_x_cluster_log1p_n"], dtype=np.float32
            ),
            subj_x_cluster_passrate=np.asarray(
                data["subj_x_cluster_passrate"], dtype=np.float32
            ),
            subject_to_macro_family_id=np.asarray(
                data["subject_to_macro_family_id"], dtype=np.int32
            ),
            subject_to_organization_id=np.asarray(
                data["subject_to_organization_id"], dtype=np.int32
            ),
            subject_to_family_id=np.asarray(
                data["subject_to_family_id"], dtype=np.int32
            ),
            n_subjects=int(n_subjects),
            n_clusters=int(n_clusters),
            n_bcs=int(n_bcs),
            n_macro_families=int(n_macro_families),
            n_organizations=int(n_organizations),
            n_families=int(n_families),
            global_mean=float(global_mean),
            smoothing=float(smoothing),
            n_train_rows_fit=int(n_train_rows_fit),
            fit_method=str(fit_method),
        )


# ---------------------------------------------------------------------------
# Fit helpers.
# ---------------------------------------------------------------------------
def _shrunken_passrate(
    sum_y: np.ndarray,
    n: np.ndarray,
    global_mean: float,
    smoothing: float,
) -> np.ndarray:
    """Bayesian-shrunk per-id pass rate, fp32."""
    out = (sum_y + smoothing * global_mean) / (n + smoothing)
    return out.astype(np.float32, copy=False)


def fit_member2_v5_feature_builder(
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
    labels: np.ndarray,
    n_subjects: int,
    n_clusters: int,
    n_bcs: int,
    n_macro_families: int,
    n_organizations: int,
    n_families: int,
    subject_to_macro_family_id: np.ndarray,
    subject_to_organization_id: np.ndarray,
    subject_to_family_id: np.ndarray,
    smoothing: float = _DEFAULT_SMOOTHING,
) -> Member2V5FeatureBuilder:
    """Aggregate per-metadata-level obs counts and mean-encoded pass rates.

    Args:
      subject_ids, cluster_ids, bc_ids:
        ``[N]`` int64 arrays. ``-1`` entries mark "unknown" and are
        excluded from the aggregations (they still contribute to the
        global_mean via valid-subject restriction).
      labels:
        ``[N]`` float in ``{0, 1}`` (soft labels accepted; the
        aggregator just sums them).
      n_*:
        Vocabulary sizes for the categorical axes. Callers should
        pre-bound these by ``max(declared, observed_max + 1)`` over
        the union of fold-train and fold-OOF ids to avoid undersize
        errors.
      subject_to_*_id:
        ``[n_subjects]`` int32 arrays mapping each subject id to its
        macro_family / organization / family. ``-1`` rows mark
        "subject id out of range" -- they fall through to the global
        mean at apply time.
      smoothing:
        Bayesian prior strength. Larger = more shrinkage toward
        global mean for sparse cells. Default 30 matches v2/v3.
    """
    subj = np.asarray(subject_ids, dtype=np.int64)
    clus = np.asarray(cluster_ids, dtype=np.int64)
    bc = np.asarray(bc_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    if not (subj.shape == clus.shape == bc.shape == y.shape):
        raise ValueError(
            f"shape mismatch: subj={subj.shape}, cluster={clus.shape}, "
            f"bc={bc.shape}, y={y.shape}"
        )
    if subj.ndim != 1:
        raise ValueError(f"all input arrays must be 1-D, got subj.ndim={subj.ndim}")
    if float(smoothing) < 0.0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    if int(n_subjects) <= 0 or int(n_clusters) <= 0 or int(n_bcs) <= 0:
        raise ValueError(
            f"n_subjects, n_clusters, n_bcs must all be > 0; "
            f"got ({n_subjects}, {n_clusters}, {n_bcs})"
        )

    # Vocabulary-fit validation. Undersized vocabularies produce a
    # cryptic IndexError deep inside np.add.at; raise an actionable
    # ValueError with the canonical fix (use ``max(declared,
    # int(arr.max()) + 1)`` over fold-train + fold-OOF id arrays).
    for name, arr, n_expected in [
        ("subject_ids", subj, n_subjects),
        ("cluster_ids", clus, n_clusters),
        ("bc_ids", bc, n_bcs),
    ]:
        if arr.size > 0:
            max_obs = int(arr.max())
            if max_obs >= int(n_expected):
                raise ValueError(
                    f"observed max({name}) = {max_obs} but n_{name[:-4]}s = "
                    f"{int(n_expected)}. The caller's declared vocabulary is "
                    "undersized for the actual data. Likely fix: pass "
                    "``fold_cond_context.n_clusters`` (which auto-grows) "
                    "instead of the raw CFG['clustering']['k'], and bound "
                    "n_bcs / n_subjects with ``max(declared, "
                    "int(arr.max()) + 1)`` over any id array fed into "
                    "this fit."
                )

    valid_subj_mask = subj >= 0
    if int(valid_subj_mask.sum()) == 0:
        global_mean = 0.5
    else:
        global_mean = float(y[valid_subj_mask].mean())
    sm = float(smoothing)

    # ---- 1-D per-id aggregates --------------------------------------
    def _agg_1d(ids_arr, n_ids):
        mask = ids_arr >= 0
        if int(mask.sum()) == 0:
            return (
                np.zeros(int(n_ids), dtype=np.float32),
                np.full(int(n_ids), float(global_mean), dtype=np.float32),
            )
        ids_clean = ids_arr[mask]
        y_clean = y[mask]
        n_per = np.bincount(ids_clean, minlength=int(n_ids)).astype(np.float64)
        sum_per = np.bincount(
            ids_clean, weights=y_clean, minlength=int(n_ids)
        ).astype(np.float64)
        log1p_n = np.log1p(n_per).astype(np.float32)
        passrate = _shrunken_passrate(sum_per, n_per, global_mean, sm)
        return log1p_n, passrate

    subj_log1p_n, subj_passrate = _agg_1d(subj, n_subjects)
    cluster_log1p_n, cluster_passrate = _agg_1d(clus, n_clusters)
    bc_log1p_n, bc_passrate = _agg_1d(bc, n_bcs)

    # ---- Subject-trait derived axes (macro_family / org / family) ---
    s2macro = np.asarray(subject_to_macro_family_id, dtype=np.int64)
    s2org = np.asarray(subject_to_organization_id, dtype=np.int64)
    s2fam = np.asarray(subject_to_family_id, dtype=np.int64)
    if (
        s2macro.shape != (int(n_subjects),)
        or s2org.shape != (int(n_subjects),)
        or s2fam.shape != (int(n_subjects),)
    ):
        raise ValueError(
            "subject_to_{macro_family,organization,family}_id must each "
            f"have shape ({int(n_subjects)},); got "
            f"{s2macro.shape}, {s2org.shape}, {s2fam.shape}"
        )

    def _per_row_trait(map_arr):
        out = np.full(subj.shape, -1, dtype=np.int64)
        m = subj >= 0
        out[m] = map_arr[subj[m]]
        return out

    macro_per_row = _per_row_trait(s2macro)
    org_per_row = _per_row_trait(s2org)
    fam_per_row = _per_row_trait(s2fam)

    macro_log1p_n, macro_passrate = _agg_1d(macro_per_row, n_macro_families)
    org_log1p_n, org_passrate = _agg_1d(org_per_row, n_organizations)
    fam_log1p_n, fam_passrate = _agg_1d(fam_per_row, n_families)

    # ---- 2-D cell counts AND mean-encoded passrates -----------------
    subj_x_bc_n = np.zeros((int(n_subjects), int(n_bcs)), dtype=np.float64)
    subj_x_bc_sum_y = np.zeros((int(n_subjects), int(n_bcs)), dtype=np.float64)
    valid_sbc = (subj >= 0) & (bc >= 0)
    if int(valid_sbc.sum()) > 0:
        np.add.at(subj_x_bc_n, (subj[valid_sbc], bc[valid_sbc]), 1.0)
        np.add.at(
            subj_x_bc_sum_y, (subj[valid_sbc], bc[valid_sbc]), y[valid_sbc]
        )
    subj_x_bc_log1p_n = np.log1p(subj_x_bc_n).astype(np.float32)
    subj_x_bc_passrate = _shrunken_passrate(
        subj_x_bc_sum_y, subj_x_bc_n, global_mean, sm
    )

    subj_x_cluster_n = np.zeros((int(n_subjects), int(n_clusters)), dtype=np.float64)
    subj_x_cluster_sum_y = np.zeros(
        (int(n_subjects), int(n_clusters)), dtype=np.float64
    )
    valid_sc = (subj >= 0) & (clus >= 0)
    if int(valid_sc.sum()) > 0:
        np.add.at(subj_x_cluster_n, (subj[valid_sc], clus[valid_sc]), 1.0)
        np.add.at(
            subj_x_cluster_sum_y, (subj[valid_sc], clus[valid_sc]), y[valid_sc]
        )
    subj_x_cluster_log1p_n = np.log1p(subj_x_cluster_n).astype(np.float32)
    subj_x_cluster_passrate = _shrunken_passrate(
        subj_x_cluster_sum_y, subj_x_cluster_n, global_mean, sm
    )

    return Member2V5FeatureBuilder(
        subj_log1p_n=subj_log1p_n,
        cluster_log1p_n=cluster_log1p_n,
        bc_log1p_n=bc_log1p_n,
        macro_log1p_n=macro_log1p_n,
        org_log1p_n=org_log1p_n,
        fam_log1p_n=fam_log1p_n,
        subj_passrate=subj_passrate,
        cluster_passrate=cluster_passrate,
        bc_passrate=bc_passrate,
        macro_passrate=macro_passrate,
        org_passrate=org_passrate,
        fam_passrate=fam_passrate,
        subj_x_bc_log1p_n=subj_x_bc_log1p_n,
        subj_x_bc_passrate=subj_x_bc_passrate,
        subj_x_cluster_log1p_n=subj_x_cluster_log1p_n,
        subj_x_cluster_passrate=subj_x_cluster_passrate,
        subject_to_macro_family_id=s2macro.astype(np.int32, copy=False),
        subject_to_organization_id=s2org.astype(np.int32, copy=False),
        subject_to_family_id=s2fam.astype(np.int32, copy=False),
        n_subjects=int(n_subjects),
        n_clusters=int(n_clusters),
        n_bcs=int(n_bcs),
        n_macro_families=int(n_macro_families),
        n_organizations=int(n_organizations),
        n_families=int(n_families),
        global_mean=float(global_mean),
        smoothing=float(sm),
        n_train_rows_fit=int(subj.shape[0]),
    )


# ---------------------------------------------------------------------------
# Apply: build the [N, 49] feature matrix.
# ---------------------------------------------------------------------------
def _safe_lookup_1d(
    table: np.ndarray, ids: np.ndarray, default: float
) -> np.ndarray:
    """Per-row lookup with ``ids < 0`` -> ``default`` and bounds clipping."""
    ids_arr = np.asarray(ids, dtype=np.int64)
    if table.ndim != 1:
        raise ValueError(f"table must be 1-D, got shape {table.shape}")
    n_max = int(table.shape[0])
    in_range = (ids_arr >= 0) & (ids_arr < n_max)
    out = np.full(ids_arr.shape, float(default), dtype=np.float64)
    out[in_range] = table[ids_arr[in_range]].astype(np.float64, copy=False)
    return out


def _safe_lookup_2d(
    table: np.ndarray,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    default: float,
) -> np.ndarray:
    """Per-row (row_id, col_id) lookup with sentinel handling."""
    r = np.asarray(row_ids, dtype=np.int64)
    c = np.asarray(col_ids, dtype=np.int64)
    if table.ndim != 2:
        raise ValueError(f"table must be 2-D, got shape {table.shape}")
    if r.shape != c.shape:
        raise ValueError(f"row/col id shape mismatch: {r.shape} vs {c.shape}")
    rmax, cmax = int(table.shape[0]), int(table.shape[1])
    in_range = (r >= 0) & (r < rmax) & (c >= 0) & (c < cmax)
    out = np.full(r.shape, float(default), dtype=np.float64)
    if int(in_range.sum()) > 0:
        out[in_range] = table[r[in_range], c[in_range]].astype(
            np.float64, copy=False
        )
    return out


def _nn_aggregate_columns(nn_matrix: np.ndarray | None, N: int) -> np.ndarray:
    """Select the 7 v5 NN columns from a 23-col ``nn_features`` matrix.

    Missing or wrong-shape input -> all zeros + a soft warning so the
    pipeline keeps running with degraded NN signal rather than halting.
    """
    out = np.zeros((N, 7), dtype=np.float32)
    if nn_matrix is None:
        warnings.warn(
            "v5 build: nn_matrix is None -- NN aggregate columns will "
            "be zero-filled. The tree loses neighborhood signal but does "
            "not crash. Pass the [N, 23] nn_train_mat / nn_oof_mat slice "
            "to fix.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    nn = np.asarray(nn_matrix, dtype=np.float32)
    if nn.ndim != 2 or nn.shape[0] != int(N):
        warnings.warn(
            f"v5 build: nn_matrix shape {nn.shape} incompatible with N={N} "
            "(expected (N, >=20)). NN columns zero-filled.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    max_src = max(M2V5_NN_SOURCE_INDICES)
    if nn.shape[1] <= max_src:
        warnings.warn(
            f"v5 build: nn_matrix has {nn.shape[1]} columns; v5 needs at "
            f"least {max_src + 1}. NN columns zero-filled.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    for k, src in enumerate(M2V5_NN_SOURCE_INDICES):
        col = nn[:, src]
        # Replace non-finite with 0 silently (per-row degradation, not a
        # global failure -- nn_features can legitimately emit NaN for
        # subjects with no labeled neighbors).
        out[:, k] = np.where(np.isfinite(col), col, 0.0).astype(np.float32)
    return out


def _pool_columns(
    pool_matrix: np.ndarray | None, N: int, n_pool_cols: int
) -> np.ndarray:
    """Select 17 pool/centroid cols from the upstream pool matrix.

    The notebook's ``pool_features_z`` is a DataFrame indexed by
    item_key with 9 pool cols + 8 centroid_dist cols (total 17). The
    caller must reindex it to the row's item_key order and pass a
    ``[N, 17]`` numpy array. Missing input -> zero fill + warning.
    """
    out = np.zeros((N, int(n_pool_cols)), dtype=np.float32)
    if pool_matrix is None:
        warnings.warn(
            "v5 build: pool_matrix is None -- item-content columns "
            "(pool + centroid) zero-filled. Tree loses item-attribute "
            "signal but does not crash.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    pm = np.asarray(pool_matrix, dtype=np.float32)
    if pm.ndim != 2 or pm.shape[0] != int(N) or pm.shape[1] != int(n_pool_cols):
        warnings.warn(
            f"v5 build: pool_matrix shape {pm.shape} != ({N}, {n_pool_cols}). "
            "Item-content columns zero-filled.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    out[:] = np.where(np.isfinite(pm), pm, 0.0).astype(np.float32, copy=False)
    return out


def _benchmark_age_columns(
    benchmark_age: np.ndarray | None, N: int
) -> np.ndarray:
    """Return ``[N, 2]`` (value, mask) for the benchmark_age feature.

    NaN in the input -> value=0, mask=0 (mask convention matches
    member_features.py: 1.0 means observed, 0.0 means missing).
    """
    out = np.zeros((N, 2), dtype=np.float32)
    if benchmark_age is None:
        return out
    arr = np.asarray(benchmark_age, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(N):
        warnings.warn(
            f"v5 build: benchmark_age length {arr.shape[0]} != N={N}. "
            "Age columns zero-filled.",
            Member2V5Warning,
            stacklevel=2,
        )
        return out
    finite = np.isfinite(arr)
    out[finite, 0] = arr[finite].astype(np.float32, copy=False)
    out[finite, 1] = 1.0
    return out


def build_member2_v5_features(
    builder: Member2V5FeatureBuilder,
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
    bc_redacted_mask: np.ndarray | None = None,
    pool_features: np.ndarray | None = None,
    benchmark_age: np.ndarray | None = None,
    nn_features_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize the ``[N, 49]`` fp32 feature matrix.

    Column order is locked by :data:`MEMBER2_V5_FEATURE_NAMES`. The
    output is finite (NaN/Inf -> 0) so the GBDT trainer's parity
    check never trips on stray non-finite values.

    Args:
      builder: fit state from :func:`fit_member2_v5_feature_builder`.
      subject_ids, cluster_ids, bc_ids:
        ``[N]`` int arrays. -1 indicates unknown / out-of-vocab.
      bc_redacted_mask:
        ``[N]`` float in ``{0, 1}``. Optional; defaults to 0.
      pool_features:
        ``[N, 17]`` float32 -- 9 pool cols + 8 centroid_dist cols in
        the canonical order from item_features.POOL_FEATURE_NAMES +
        ``centroid_dist_{0..7}``. Caller must reindex
        ``pool_features_z`` by row item_key before passing.
      benchmark_age:
        ``[N]`` float64 -- benchmark age per row (NaN allowed for
        missing). The function emits value + mask as columns 17, 18.
      nn_features_matrix:
        ``[N, 23]`` float32 -- the upstream nn_features matrix for
        the rows being built. v5 selects 7 of the 23 columns; the
        rest are ignored.
    """
    subj = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    clus = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    bc = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    N = int(subj.shape[0])
    if not (clus.shape[0] == bc.shape[0] == N):
        raise ValueError(
            "subject_ids/cluster_ids/bc_ids must all have the same length; "
            f"got subj={subj.shape}, clus={clus.shape}, bc={bc.shape}"
        )

    if bc_redacted_mask is None:
        bc_redact = np.zeros(N, dtype=np.float32)
    else:
        bc_redact = np.asarray(bc_redacted_mask, dtype=np.float32).reshape(-1)
        if bc_redact.shape[0] != N:
            raise ValueError(
                f"bc_redacted_mask shape {bc_redact.shape} != ({N},)"
            )

    g = float(builder.global_mean)

    # ---- Bucket B: item content (9 pool + 8 centroid = 17) + age (2) ----
    pool_cols = _pool_columns(pool_features, N, 17)
    age_cols = _benchmark_age_columns(benchmark_age, N)

    # ---- Bucket C: historical counts (8) + redaction mask (1) ----------
    n_sub = int(builder.n_subjects)
    subj_safe = np.clip(subj, 0, max(n_sub - 1, 0))
    macro_ids = np.where(
        subj >= 0,
        builder.subject_to_macro_family_id[subj_safe].astype(np.int64, copy=False),
        -1,
    )
    org_ids = np.where(
        subj >= 0,
        builder.subject_to_organization_id[subj_safe].astype(np.int64, copy=False),
        -1,
    )
    fam_ids = np.where(
        subj >= 0,
        builder.subject_to_family_id[subj_safe].astype(np.int64, copy=False),
        -1,
    )

    subj_log1p_n = _safe_lookup_1d(builder.subj_log1p_n, subj, default=0.0)
    cluster_log1p_n = _safe_lookup_1d(
        builder.cluster_log1p_n, clus, default=0.0
    )
    bc_log1p_n = _safe_lookup_1d(builder.bc_log1p_n, bc, default=0.0)
    macro_log1p_n = _safe_lookup_1d(
        builder.macro_log1p_n, macro_ids, default=0.0
    )
    org_log1p_n = _safe_lookup_1d(builder.org_log1p_n, org_ids, default=0.0)
    fam_log1p_n = _safe_lookup_1d(builder.fam_log1p_n, fam_ids, default=0.0)
    sc_log1p_n = _safe_lookup_2d(
        builder.subj_x_cluster_log1p_n, subj, clus, default=0.0
    )
    sbc_log1p_n = _safe_lookup_2d(
        builder.subj_x_bc_log1p_n, subj, bc, default=0.0
    )

    # ---- Bucket TE: target encodings (8) -------------------------------
    subj_passrate = _safe_lookup_1d(builder.subj_passrate, subj, default=g)
    cluster_passrate = _safe_lookup_1d(
        builder.cluster_passrate, clus, default=g
    )
    bc_passrate = _safe_lookup_1d(builder.bc_passrate, bc, default=g)
    macro_passrate = _safe_lookup_1d(
        builder.macro_passrate, macro_ids, default=g
    )
    org_passrate = _safe_lookup_1d(builder.org_passrate, org_ids, default=g)
    fam_passrate = _safe_lookup_1d(builder.fam_passrate, fam_ids, default=g)
    sc_passrate = _safe_lookup_2d(
        builder.subj_x_cluster_passrate, subj, clus, default=g
    )
    sbc_passrate = _safe_lookup_2d(
        builder.subj_x_bc_passrate, subj, bc, default=g
    )

    # ---- Bucket U: unknown flags (6) -----------------------------------
    is_unk_subject = (subj < 0).astype(np.float32)
    is_unk_cluster = (clus < 0).astype(np.float32)
    is_unk_bc = (bc < 0).astype(np.float32)
    is_unk_macro = (macro_ids < 0).astype(np.float32)
    is_unk_org = (org_ids < 0).astype(np.float32)
    is_unk_fam = (fam_ids < 0).astype(np.float32)

    # ---- Bucket D: NN aggregates (7) -----------------------------------
    nn_cols = _nn_aggregate_columns(nn_features_matrix, N)

    out = np.empty((N, M2V5_FEATURE_DIM), dtype=np.float32)
    # B (0..18)
    out[:, 0:17] = pool_cols
    out[:, 17:19] = age_cols
    # C (19..27)
    out[:, 19] = subj_log1p_n.astype(np.float32, copy=False)
    out[:, 20] = cluster_log1p_n.astype(np.float32, copy=False)
    out[:, 21] = bc_log1p_n.astype(np.float32, copy=False)
    out[:, 22] = macro_log1p_n.astype(np.float32, copy=False)
    out[:, 23] = org_log1p_n.astype(np.float32, copy=False)
    out[:, 24] = fam_log1p_n.astype(np.float32, copy=False)
    out[:, 25] = sc_log1p_n.astype(np.float32, copy=False)
    out[:, 26] = sbc_log1p_n.astype(np.float32, copy=False)
    out[:, 27] = bc_redact
    # TE (28..35)
    out[:, 28] = subj_passrate.astype(np.float32, copy=False)
    out[:, 29] = cluster_passrate.astype(np.float32, copy=False)
    out[:, 30] = bc_passrate.astype(np.float32, copy=False)
    out[:, 31] = macro_passrate.astype(np.float32, copy=False)
    out[:, 32] = org_passrate.astype(np.float32, copy=False)
    out[:, 33] = fam_passrate.astype(np.float32, copy=False)
    out[:, 34] = sc_passrate.astype(np.float32, copy=False)
    out[:, 35] = sbc_passrate.astype(np.float32, copy=False)
    # U (36..41)
    out[:, 36] = is_unk_subject
    out[:, 37] = is_unk_cluster
    out[:, 38] = is_unk_bc
    out[:, 39] = is_unk_macro
    out[:, 40] = is_unk_org
    out[:, 41] = is_unk_fam
    # D (42..48)
    out[:, 42:49] = nn_cols

    # Defensive finite-fill. Should be a no-op given upstream guards
    # but cheap insurance against a NaN/Inf squeak-through producing
    # a silent tree-split divergence.
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Shipped state. The notebook caches this; the runtime template loads it.
# ---------------------------------------------------------------------------
@dataclass
class Member2V5State:
    """Everything the runtime needs to score Member 2 v5 on a batch.

    Composition recipe at apply time::

        X = build_member2_v5_features(state.builder, subject_ids=..., ...)
        p2 = gbdt_apply_batch(state.gbdt, X)  # direct binary, sigmoid baked in
    """

    gbdt: object  # GBDTMemberState; left untyped to avoid pulling
                  # gbdt_member at import time.
    builder: Member2V5FeatureBuilder
    feature_names: Tuple[str, ...] = field(
        default_factory=lambda: MEMBER2_V5_FEATURE_NAMES
    )
    version: str = "v5.0"

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != MEMBER2_V5_FEATURE_NAMES:
            warnings.warn(
                "Member2V5State feature_names mismatch with the module-level "
                "MEMBER2_V5_FEATURE_NAMES. Did the schema change without a "
                "version bump? Continuing with the stored names.",
                Member2V5Warning,
                stacklevel=2,
            )
