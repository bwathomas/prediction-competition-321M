"""Member 2 v3: M1 calibration tree.

A small GBDT whose job is to learn *when to trust Member 1's prediction
vs when to shrink it toward the subject mean*, based on per-metadata-
level observation counts and an explicit "unknown ___" inductive bias
on each id field.

Motivation (vs Member 2 v2):
    v2 was trained on ~23 mean-encoded marginals + interactions; on
    cold items its features carry NO item-specific signal so the trees
    can only learn (subject, cluster, bc) cell aggregates. Those
    aggregates are systematically wrong for individual cold items,
    which produced the famous v2 fold-0 OOF NLL=0.778 (worse than the
    constant-mean predictor at 0.628). v3 adds back item information
    via M1's prediction as an input feature, but keeps the tree small
    so it doesn't just relearn M1.

Feature schema (25 numeric columns, no native categoricals so the
``gbdt_member`` numpy walker stays in its supported "<=" regime):
    0. logit_p_m1                       -- M1's prediction, logit space
    1. logit_subject_mean               -- subject baseline, logit space
    2. logit_disagreement               -- f0 - f1
    3. abs_logit_disagreement           -- |f2|, an easy "M1 is unsure" rule
    4. p_m1_uncertainty                 -- p_m1 * (1 - p_m1), max at 0.5
    5. subject_obs_count_log1p          -- how well do we know this subject
    6. cluster_obs_count_log1p          -- how well do we know this cluster
    7. bc_obs_count_log1p               -- how well do we know this bc
    8. macro_family_obs_count_log1p     -- how well do we know this macro_family
    9. organization_obs_count_log1p     -- how well do we know this org
   10. family_obs_count_log1p           -- how well do we know this family
   11. subj_x_bc_obs_count_log1p        -- (subject, bc) cell count
   12. subj_x_cluster_obs_count_log1p   -- (subject, cluster) cell count
   13. cluster_passrate_meanenc         -- shrunken cluster pass rate
   14. bc_passrate_meanenc              -- shrunken bc pass rate
   15. macro_family_passrate_meanenc    -- shrunken macro pass rate
   16. organization_passrate_meanenc    -- shrunken org pass rate
   17. family_passrate_meanenc          -- shrunken family pass rate
   18. is_unknown_subject               -- explicit "subject id < 0" flag
   19. is_unknown_cluster               -- explicit "cluster id < 0" flag
   20. is_unknown_bc                    -- explicit "bc id < 0" flag  <-- user-requested inductive bias
   21. is_unknown_macro_family          -- explicit "macro_family id < 0"
   22. is_unknown_organization          -- explicit "org id < 0"
   23. is_unknown_family                -- explicit "family id < 0"
   24. bc_redacted_mask                 -- benchmark-condition redaction flag

Trained as a residual booster with ``subject_mean`` as ``init_pred``,
identical machinery to v2 (``fit_gbdt_member`` / ``compose_residual_*``).
Cold items where the tree has nothing useful to add converge to a near-
zero residual and the prediction falls back to ``subject_mean``; this
is the structural fix for v2's "blow up below baseline" failure mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


_EPS_PROB = 1e-7
_DEFAULT_SMOOTHING = 30.0


# ---------------------------------------------------------------------------
# Locked column layout (do NOT reorder -- it would invalidate every saved
# state silently because the GBDT trees index features by integer column
# position, not by name).
# ---------------------------------------------------------------------------
MEMBER2_V3_FEATURE_NAMES: Tuple[str, ...] = (
    "logit_p_m1",
    "logit_subject_mean",
    "logit_disagreement",
    "abs_logit_disagreement",
    "p_m1_uncertainty",
    "subject_obs_count_log1p",
    "cluster_obs_count_log1p",
    "bc_obs_count_log1p",
    "macro_family_obs_count_log1p",
    "organization_obs_count_log1p",
    "family_obs_count_log1p",
    "subj_x_bc_obs_count_log1p",
    "subj_x_cluster_obs_count_log1p",
    "cluster_passrate_meanenc",
    "bc_passrate_meanenc",
    "macro_family_passrate_meanenc",
    "organization_passrate_meanenc",
    "family_passrate_meanenc",
    "is_unknown_subject",
    "is_unknown_cluster",
    "is_unknown_bc",
    "is_unknown_macro_family",
    "is_unknown_organization",
    "is_unknown_family",
    "bc_redacted_mask",
)
M2V3_FEATURE_DIM = len(MEMBER2_V3_FEATURE_NAMES)
assert M2V3_FEATURE_DIM == 25, (
    f"MEMBER2_V3_FEATURE_NAMES has {M2V3_FEATURE_DIM} entries; the v3 "
    "spec is locked at 25. Bump the cache prefix in the notebook before "
    "changing this constant."
)


# ---------------------------------------------------------------------------
# Feature builder state (the "what does my training set look like?" half
# of v3; the GBDT booster is the other half and lives in GBDTMemberState).
# ---------------------------------------------------------------------------
@dataclass
class Member2V3FeatureBuilder:
    """Per-id lookup tables needed to materialize v3 features at runtime.

    All ``*_passrate`` arrays are Bayesian-shrunk:
        passrate_id = (sum_y_in_id + smoothing * global_mean)
                      / (n_in_id + smoothing)

    All ``*_log1p_n`` arrays store ``log1p(observation_count)`` so the
    tree can split on log-scale uncertainty proxies.

    Subject -> trait id maps (``subject_to_*_id``) are needed because
    ``macro_family`` / ``organization`` / ``family`` are subject-level
    attributes; a row's macro_family is looked up via its subject id.
    """

    # Per-id log1p(observation count) -- shape [n_ids_for_that_axis].
    subj_log1p_n: np.ndarray
    cluster_log1p_n: np.ndarray
    bc_log1p_n: np.ndarray
    macro_log1p_n: np.ndarray
    org_log1p_n: np.ndarray
    fam_log1p_n: np.ndarray

    # Per-id shrunken pass rate (continuous, in (0, 1)). We do NOT
    # ship subj_passrate as a feature because subject_mean is already
    # the residual anchor; including it would be perfectly collinear
    # with the init_score and wastes a tree split.
    cluster_passrate: np.ndarray
    bc_passrate: np.ndarray
    macro_passrate: np.ndarray
    org_passrate: np.ndarray
    fam_passrate: np.ndarray

    # 2-D cell counts. Stored dense float32 because the resolutions are
    # modest: 907 * 224 = 0.8 MB for subj_x_bc; 907 * 64 = 0.2 MB for
    # subj_x_cluster. Worth the constant memory for the tree's interaction
    # detection power.
    subj_x_bc_log1p_n: np.ndarray
    subj_x_cluster_log1p_n: np.ndarray

    # Subject -> trait map (shape [n_subjects]).
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
    fit_method: str = "bayes_shrunk_logit_residual"

    def __post_init__(self) -> None:
        # Shape audits.
        for name, arr, length in [
            ("subj_log1p_n", self.subj_log1p_n, self.n_subjects),
            ("cluster_log1p_n", self.cluster_log1p_n, self.n_clusters),
            ("bc_log1p_n", self.bc_log1p_n, self.n_bcs),
            ("macro_log1p_n", self.macro_log1p_n, self.n_macro_families),
            ("org_log1p_n", self.org_log1p_n, self.n_organizations),
            ("fam_log1p_n", self.fam_log1p_n, self.n_families),
            ("cluster_passrate", self.cluster_passrate, self.n_clusters),
            ("bc_passrate", self.bc_passrate, self.n_bcs),
            ("macro_passrate", self.macro_passrate, self.n_macro_families),
            ("org_passrate", self.org_passrate, self.n_organizations),
            ("fam_passrate", self.fam_passrate, self.n_families),
            ("subject_to_macro_family_id", self.subject_to_macro_family_id, self.n_subjects),
            ("subject_to_organization_id", self.subject_to_organization_id, self.n_subjects),
            ("subject_to_family_id", self.subject_to_family_id, self.n_subjects),
        ]:
            if arr.shape != (int(length),):
                raise ValueError(
                    f"v3 builder: {name} shape {arr.shape} != ({int(length)},)"
                )
        if self.subj_x_bc_log1p_n.shape != (self.n_subjects, self.n_bcs):
            raise ValueError(
                f"v3 builder: subj_x_bc_log1p_n shape "
                f"{self.subj_x_bc_log1p_n.shape} != "
                f"({self.n_subjects}, {self.n_bcs})"
            )
        if self.subj_x_cluster_log1p_n.shape != (self.n_subjects, self.n_clusters):
            raise ValueError(
                f"v3 builder: subj_x_cluster_log1p_n shape "
                f"{self.subj_x_cluster_log1p_n.shape} != "
                f"({self.n_subjects}, {self.n_clusters})"
            )

    # I/O ----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a numpy-arrays-only dict suitable for ``np.savez``."""
        return {
            "subj_log1p_n": self.subj_log1p_n.astype(np.float32, copy=False),
            "cluster_log1p_n": self.cluster_log1p_n.astype(np.float32, copy=False),
            "bc_log1p_n": self.bc_log1p_n.astype(np.float32, copy=False),
            "macro_log1p_n": self.macro_log1p_n.astype(np.float32, copy=False),
            "org_log1p_n": self.org_log1p_n.astype(np.float32, copy=False),
            "fam_log1p_n": self.fam_log1p_n.astype(np.float32, copy=False),
            "cluster_passrate": self.cluster_passrate.astype(np.float32, copy=False),
            "bc_passrate": self.bc_passrate.astype(np.float32, copy=False),
            "macro_passrate": self.macro_passrate.astype(np.float32, copy=False),
            "org_passrate": self.org_passrate.astype(np.float32, copy=False),
            "fam_passrate": self.fam_passrate.astype(np.float32, copy=False),
            "subj_x_bc_log1p_n": self.subj_x_bc_log1p_n.astype(np.float32, copy=False),
            "subj_x_cluster_log1p_n": self.subj_x_cluster_log1p_n.astype(np.float32, copy=False),
            "subject_to_macro_family_id": self.subject_to_macro_family_id.astype(np.int32, copy=False),
            "subject_to_organization_id": self.subject_to_organization_id.astype(np.int32, copy=False),
            "subject_to_family_id": self.subject_to_family_id.astype(np.int32, copy=False),
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
        fit_method: str = "bayes_shrunk_logit_residual",
    ) -> "Member2V3FeatureBuilder":
        return cls(
            subj_log1p_n=np.asarray(data["subj_log1p_n"], dtype=np.float32),
            cluster_log1p_n=np.asarray(data["cluster_log1p_n"], dtype=np.float32),
            bc_log1p_n=np.asarray(data["bc_log1p_n"], dtype=np.float32),
            macro_log1p_n=np.asarray(data["macro_log1p_n"], dtype=np.float32),
            org_log1p_n=np.asarray(data["org_log1p_n"], dtype=np.float32),
            fam_log1p_n=np.asarray(data["fam_log1p_n"], dtype=np.float32),
            cluster_passrate=np.asarray(data["cluster_passrate"], dtype=np.float32),
            bc_passrate=np.asarray(data["bc_passrate"], dtype=np.float32),
            macro_passrate=np.asarray(data["macro_passrate"], dtype=np.float32),
            org_passrate=np.asarray(data["org_passrate"], dtype=np.float32),
            fam_passrate=np.asarray(data["fam_passrate"], dtype=np.float32),
            subj_x_bc_log1p_n=np.asarray(data["subj_x_bc_log1p_n"], dtype=np.float32),
            subj_x_cluster_log1p_n=np.asarray(data["subj_x_cluster_log1p_n"], dtype=np.float32),
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


def fit_member2_v3_feature_builder(
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
) -> Member2V3FeatureBuilder:
    """Aggregate per-metadata-level observation counts and pass rates.

    Args:
      subject_ids, cluster_ids, bc_ids:
        ``[N]`` int64 arrays. ``-1`` entries mark "unknown" and are
        excluded from the aggregations (they still contribute to the
        global_mean estimate via subject==-1 dropout).
      labels:
        ``[N]`` float in ``{0, 1}`` (soft labels accepted; the
        aggregator just sums them).
      n_*:
        Vocabulary sizes for the categorical axes.
      subject_to_*_id:
        ``[n_subjects]`` int32 arrays mapping each subject id to its
        macro_family / organization / family. ``-1`` rows mark
        "subject id out of range" -- they fall through to the global
        mean at apply time.
      smoothing:
        Bayesian prior strength. Larger = more shrinkage toward
        global mean for sparse cells. Default 30 matches v2.
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

    # Vocab-fit validation. Without this, undersized n_clusters / n_bcs
    # / n_subjects throws a cryptic ``IndexError: index X is out of
    # bounds for axis Y with size Z`` deep inside ``np.add.at`` on the
    # 2-D cell-count step -- you have to trace back from the
    # subj_x_cluster line to figure out which axis blew up. The
    # canonical fix in the caller is to pass
    # ``fold_cond_context.n_clusters`` (which already auto-grows from
    # CFG['clustering']['k'] when the clustering produces more clusters
    # than declared, as the conditional-passrate context does); for
    # subjects/bcs use ``indexer.n_subjects`` / ``indexer.n_bc`` plus
    # the same max-over-observed-ids guard.
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

    subj_log1p_n, _ = _agg_1d(subj, n_subjects)
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

    # Per-row trait id (use -1 sentinel for unknown subject -> unknown trait).
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

    # ---- 2-D cell counts --------------------------------------------
    # np.add.at is the right tool here (np.bincount is 1-D only).
    subj_x_bc_n = np.zeros((int(n_subjects), int(n_bcs)), dtype=np.float64)
    valid_sbc = (subj >= 0) & (bc >= 0)
    if int(valid_sbc.sum()) > 0:
        np.add.at(subj_x_bc_n, (subj[valid_sbc], bc[valid_sbc]), 1.0)
    subj_x_bc_log1p_n = np.log1p(subj_x_bc_n).astype(np.float32)

    subj_x_cluster_n = np.zeros((int(n_subjects), int(n_clusters)), dtype=np.float64)
    valid_sc = (subj >= 0) & (clus >= 0)
    if int(valid_sc.sum()) > 0:
        np.add.at(subj_x_cluster_n, (subj[valid_sc], clus[valid_sc]), 1.0)
    subj_x_cluster_log1p_n = np.log1p(subj_x_cluster_n).astype(np.float32)

    return Member2V3FeatureBuilder(
        subj_log1p_n=subj_log1p_n,
        cluster_log1p_n=cluster_log1p_n,
        bc_log1p_n=bc_log1p_n,
        macro_log1p_n=macro_log1p_n,
        org_log1p_n=org_log1p_n,
        fam_log1p_n=fam_log1p_n,
        cluster_passrate=cluster_passrate,
        bc_passrate=bc_passrate,
        macro_passrate=macro_passrate,
        org_passrate=org_passrate,
        fam_passrate=fam_passrate,
        subj_x_bc_log1p_n=subj_x_bc_log1p_n,
        subj_x_cluster_log1p_n=subj_x_cluster_log1p_n,
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
# Apply: build the [N, 25] feature matrix.
# ---------------------------------------------------------------------------
def _logit_clip(p: np.ndarray, eps: float = _EPS_PROB) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


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


def build_member2_v3_features(
    builder: Member2V3FeatureBuilder,
    *,
    p_m1: np.ndarray,
    subject_mean: np.ndarray,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
    bc_redacted_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize the ``[N, 25]`` fp32 feature matrix.

    Column order is locked by :data:`MEMBER2_V3_FEATURE_NAMES`. The
    output is finite (NaN/Inf -> 0) so the GBDT trainer's parity check
    never trips on stray non-finite values.
    """
    p_m1_arr = np.asarray(p_m1, dtype=np.float64).reshape(-1)
    sm_arr = np.asarray(subject_mean, dtype=np.float64).reshape(-1)
    subj = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    clus = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    bc = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    N = int(p_m1_arr.shape[0])
    if not (
        sm_arr.shape[0] == subj.shape[0] == clus.shape[0] == bc.shape[0] == N
    ):
        raise ValueError(
            "all input arrays must have the same length N, got "
            f"p_m1={p_m1_arr.shape}, sm={sm_arr.shape}, subj={subj.shape}, "
            f"clus={clus.shape}, bc={bc.shape}"
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

    # M1 / subject-anchor block (cols 0-4).
    logit_p_m1 = _logit_clip(p_m1_arr)
    logit_sm = _logit_clip(sm_arr)
    disagreement = logit_p_m1 - logit_sm
    abs_disagreement = np.abs(disagreement)
    p_m1_clip = np.clip(p_m1_arr, _EPS_PROB, 1.0 - _EPS_PROB)
    p_m1_uncertainty = p_m1_clip * (1.0 - p_m1_clip)

    # 1-D obs-count lookups (cols 5-10).
    subj_log1p_n = _safe_lookup_1d(builder.subj_log1p_n, subj, default=0.0)
    cluster_log1p_n = _safe_lookup_1d(builder.cluster_log1p_n, clus, default=0.0)
    bc_log1p_n = _safe_lookup_1d(builder.bc_log1p_n, bc, default=0.0)

    # Derive per-row macro/org/family ids via subject map; sentinel -1
    # propagates so the obs-count / passrate lookups produce the same
    # fallback as for an unknown subject.
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

    macro_log1p_n = _safe_lookup_1d(builder.macro_log1p_n, macro_ids, default=0.0)
    org_log1p_n = _safe_lookup_1d(builder.org_log1p_n, org_ids, default=0.0)
    fam_log1p_n = _safe_lookup_1d(builder.fam_log1p_n, fam_ids, default=0.0)

    # 2-D cell counts (cols 11-12).
    sbc_log1p_n = _safe_lookup_2d(
        builder.subj_x_bc_log1p_n, subj, bc, default=0.0
    )
    sc_log1p_n = _safe_lookup_2d(
        builder.subj_x_cluster_log1p_n, subj, clus, default=0.0
    )

    # Passrate look-ups (cols 13-17). Unknown ids -> global_mean.
    cluster_passrate = _safe_lookup_1d(builder.cluster_passrate, clus, default=g)
    bc_passrate = _safe_lookup_1d(builder.bc_passrate, bc, default=g)
    macro_passrate = _safe_lookup_1d(builder.macro_passrate, macro_ids, default=g)
    org_passrate = _safe_lookup_1d(builder.org_passrate, org_ids, default=g)
    fam_passrate = _safe_lookup_1d(builder.fam_passrate, fam_ids, default=g)

    # Explicit unknown-id flags (cols 18-23). These are the "inductive bias"
    # the model uses to learn 'if the metadata is missing, ignore the
    # corresponding aggregate and lean on subject_mean'.
    is_unk_subject = (subj < 0).astype(np.float32)
    is_unk_cluster = (clus < 0).astype(np.float32)
    is_unk_bc = (bc < 0).astype(np.float32)
    is_unk_macro = (macro_ids < 0).astype(np.float32)
    is_unk_org = (org_ids < 0).astype(np.float32)
    is_unk_fam = (fam_ids < 0).astype(np.float32)

    out = np.empty((N, M2V3_FEATURE_DIM), dtype=np.float32)
    out[:, 0] = logit_p_m1.astype(np.float32, copy=False)
    out[:, 1] = logit_sm.astype(np.float32, copy=False)
    out[:, 2] = disagreement.astype(np.float32, copy=False)
    out[:, 3] = abs_disagreement.astype(np.float32, copy=False)
    out[:, 4] = p_m1_uncertainty.astype(np.float32, copy=False)
    out[:, 5] = subj_log1p_n.astype(np.float32, copy=False)
    out[:, 6] = cluster_log1p_n.astype(np.float32, copy=False)
    out[:, 7] = bc_log1p_n.astype(np.float32, copy=False)
    out[:, 8] = macro_log1p_n.astype(np.float32, copy=False)
    out[:, 9] = org_log1p_n.astype(np.float32, copy=False)
    out[:, 10] = fam_log1p_n.astype(np.float32, copy=False)
    out[:, 11] = sbc_log1p_n.astype(np.float32, copy=False)
    out[:, 12] = sc_log1p_n.astype(np.float32, copy=False)
    out[:, 13] = cluster_passrate.astype(np.float32, copy=False)
    out[:, 14] = bc_passrate.astype(np.float32, copy=False)
    out[:, 15] = macro_passrate.astype(np.float32, copy=False)
    out[:, 16] = org_passrate.astype(np.float32, copy=False)
    out[:, 17] = fam_passrate.astype(np.float32, copy=False)
    out[:, 18] = is_unk_subject
    out[:, 19] = is_unk_cluster
    out[:, 20] = is_unk_bc
    out[:, 21] = is_unk_macro
    out[:, 22] = is_unk_org
    out[:, 23] = is_unk_fam
    out[:, 24] = bc_redact

    # Defensive finite-fill. Should be a no-op given the bounded inputs,
    # but cheap insurance against an upstream NaN producing a silent
    # tree-split divergence.
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Shipped state. The notebook caches this; the runtime template loads it.
# ---------------------------------------------------------------------------
@dataclass
class Member2V3State:
    """Everything the runtime needs to score Member 2 v3 on one row.

    Composition recipe at apply time:
        x = build_member2_v3_features(state.builder, p_m1=p_m1_row,
                                      subject_mean=sm_row, subject_ids=...)
        p2 = gbdt_compose_residual_batch(state.gbdt, x, sm_row)
    """
    gbdt: object  # GBDTMemberState; left untyped here so this module
                  # doesn't pull gbdt_member at import time (the runtime
                  # template is pure-numpy + dataclass).
    builder: Member2V3FeatureBuilder
    feature_names: Tuple[str, ...] = field(
        default_factory=lambda: MEMBER2_V3_FEATURE_NAMES
    )
    version: str = "v3.0"

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != MEMBER2_V3_FEATURE_NAMES:
            raise ValueError(
                "Member2V3State feature_names mismatch with the module-level "
                "MEMBER2_V3_FEATURE_NAMES. Did the schema change without a "
                "version bump?"
            )
