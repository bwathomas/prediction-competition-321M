"""Drop-in 23-dim NN-feature runtime for package-free submissions.

Purpose
-------
The currently-shipped nemotron (`nemotron_trc_v5`) and LGAI (`LGAI_fixed`)
submodels compute an **8-dim** NN feature vector via an inlined copy of
``src/nn_features.py::_aggregate_nn_features`` (with ``NN_FEATURE_DIM == 8``)
and feed it to their trained residual heads.

Our CURRENT pipeline (``src/nn_features.py``) uses ``NN_FEATURE_DIM == 23``:
cells 0..7 are the SAME legacy 8, cells 8..22 are self-derived + trait-
conditional / context passrate features.

This module lets a submodel **emit the full 23-dim vector** (so new ensemble
members that expect 23 inputs are fed correctly) while remaining **back-compat**
with the legacy trained head (which is fed ``vec[:8]``). It does this by:

  1. Re-implementing the 23-dim aggregator (verbatim numerics from
     ``src/nn_features.py``) with NO scipy / pandas / sklearn dependency at the
     hot path. Loading the conditional context uses ``scipy.sparse`` ONLY if
     present; if scipy is unavailable at runtime the conditional cells
     (15..22) fall back to ``fallback_value`` and you still get a valid 23-dim
     vector whose first 15 cells (0..14) are exact.
  2. A ``ConditionalContextRuntime`` loader that reads the on-disk bag written
     by ``src/nn_features.py::ConditionalPassrateContext.save`` (the layout
     under ``DR/artifacts/nn_features/``) and resolves per-query
     ``cond_inputs`` exactly as ``_resolve_conditional_inputs`` does.

Runtime import policy (HARD): only ``numpy`` + stdlib are imported at module
scope. ``scipy.sparse`` is imported lazily and guarded; a missing scipy
degrades cells 15..22 to fallback rather than raising.

WIRING (see scripts/ship/NN23_PATCH_NOTES.md for exact line edits):
    full23 = compute_nn_features_23(cache, item_emb, subject_id,
                                    cond_ctx=COND_CTX,
                                    query_benchmark_key=bc_key,
                                    query_cluster_id=cluster_id,
                                    k=NN_RUNTIME_K, ...)
    legacy8 = full23[:8]          # -> existing trained head (back-compat)
    new_member_feats = full23     # -> any new member that wants 23-dim

The two arrays are SLICES of ONE computation, so cells 0..7 of `full23`
are bit-identical to the legacy 8-dim vector the head was trained on.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Mapping

import numpy as np

LOG = logging.getLogger("nn23_runtime")

# Locked schema -- must equal src/nn_features.py::NN_FEATURE_DIM.
NN_FEATURE_DIM_23: int = 23
LEGACY_NN_FEATURE_DIM: int = 8
MISSING_TRAIT_ID: int = 0  # mirrors src/nn_features.py


# ===========================================================================
# Pure aggregation -- 23-dim. Numerics copied VERBATIM from
# src/nn_features.py::_aggregate_nn_features (+ _aggregate_trait_conditional).
# If you change one, change the other. The validation harness asserts the
# schema is byte-identical for cells 0..14 and that cells 15..22 redact to
# `fallback_value` when `cond_inputs` is None.
# ===========================================================================


def _aggregate_trait_conditional(
    passrates: np.ndarray,        # [B, K]
    masks: np.ndarray,            # [B, K]
    redact_row: np.ndarray,       # [B]
    fallback_value: float,
) -> np.ndarray:
    pr_safe = np.where(masks > 0, passrates, 0.0).astype(np.float32)
    n_labeled = masks.sum(axis=1)
    has_any = n_labeled > 0
    pr_sum = pr_safe.sum(axis=1)
    out = np.where(
        has_any,
        pr_sum / np.maximum(n_labeled, 1.0),
        fallback_value,
    ).astype(np.float32)
    out = np.where(redact_row > 0, np.float32(fallback_value), out).astype(np.float32)
    return out


def _aggregate_nn_features_23(
    neighbor_passrates: np.ndarray,    # [B, K] mean labels (NaN where missing)
    neighbor_masks: np.ndarray,        # [B, K] 1 where labeled, 0 otherwise
    similarities: np.ndarray,          # [B, K]
    *,
    fallback_value: float,
    top1_missing_sentinel: float,
    cond_inputs: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pure NN aggregation. Returns [B, 23] float32.

    Cells 0..7  : legacy aggregators (bit-identical to the 8-dim runtime).
    Cells 8..14 : self-derived additions (no extra tables needed).
    Cells 15..22: conditional/context features; each falls back to
                  ``fallback_value`` when its ``cond_inputs`` key is absent.
    """
    passrates = np.asarray(neighbor_passrates, dtype=np.float32)
    masks = np.asarray(neighbor_masks, dtype=np.float32)
    sims = np.asarray(similarities, dtype=np.float32)

    if passrates.ndim == 1:
        passrates = passrates[None, :]
        masks = masks[None, :]
        sims = sims[None, :]

    B, K = passrates.shape

    pr_safe = np.where(masks > 0, passrates, 0.0).astype(np.float32)

    n_labeled = masks.sum(axis=1)
    has_any = n_labeled > 0

    mean_sim = sims.mean(axis=1).astype(np.float32)

    pr_sum = pr_safe.sum(axis=1)
    pr_mean = np.where(has_any, pr_sum / np.maximum(n_labeled, 1.0), fallback_value)

    sim_safe = np.where(masks > 0, sims, 0.0).astype(np.float32)
    weights = np.clip(sim_safe, 0.0, None)
    weight_sum = weights.sum(axis=1)
    weighted = np.where(
        (weight_sum > 1e-9) & has_any,
        (weights * pr_safe).sum(axis=1) / np.maximum(weight_sum, 1e-9),
        np.where(has_any, pr_mean, fallback_value),
    ).astype(np.float32)

    diff = (pr_safe - pr_mean[:, None]) * masks
    sq = (diff * diff).sum(axis=1)
    var = np.where(has_any, sq / np.maximum(n_labeled, 1.0), 0.0)
    pr_std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    pr_std = np.where(has_any, pr_std, fallback_value).astype(np.float32)

    coverage = (n_labeled / float(max(1, K))).astype(np.float32)

    top1_mask = masks[:, 0]
    top1_label = np.where(top1_mask > 0, passrates[:, 0], top1_missing_sentinel)
    top1_label = top1_label.astype(np.float32)
    top1_sim = sims[:, 0].astype(np.float32)

    n_labeled_log = np.log1p(n_labeled).astype(np.float32)

    # ----- Self-derived additions (cells 8..14) -----
    weight_sq_sum = (weights * weights).sum(axis=1)
    eff_count = np.where(
        weight_sq_sum > 1e-9,
        (weight_sum * weight_sum) / np.maximum(weight_sq_sum, 1e-9),
        0.0,
    ).astype(np.float32)

    top1_minus_topk = (top1_sim - mean_sim).astype(np.float32)

    bootstrap_se = np.where(
        n_labeled > 0,
        pr_std / np.sqrt(np.maximum(n_labeled, 1.0)),
        fallback_value,
    ).astype(np.float32)

    p_clip = np.clip(pr_mean, 1e-7, 1.0 - 1e-7)
    entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))
    entropy = np.where(has_any, entropy, 0.0).astype(np.float32)

    top1_label_match = np.where(
        top1_mask > 0,
        (passrates[:, 0] > 0.5).astype(np.float32),
        fallback_value,
    ).astype(np.float32)

    sim_min = sims.min(axis=1).astype(np.float32)
    sim_median = np.median(sims, axis=1).astype(np.float32)
    span = top1_sim - sim_min
    sim_skew = np.where(
        np.abs(span) > 1e-9,
        (top1_sim - sim_median) / np.where(np.abs(span) > 1e-9, span, 1.0),
        0.0,
    ).astype(np.float32)

    distance_to_kth = sims[:, -1].astype(np.float32)

    # ----- Tier 2/3: conditional + context features (cells 15..22) -----
    cond_inputs = dict(cond_inputs or {})
    fb = np.float32(fallback_value)

    def _cond_pair(pref: str) -> np.ndarray:
        pr = cond_inputs.get(f"{pref}_passrates")
        mk = cond_inputs.get(f"{pref}_masks")
        rd = cond_inputs.get(f"{pref}_redact")
        if pr is None or mk is None:
            return np.full(B, fb, dtype=np.float32)
        rd_arr = (
            np.asarray(rd, dtype=np.float32).reshape(-1)
            if rd is not None
            else np.zeros(B, dtype=np.float32)
        )
        return _aggregate_trait_conditional(
            np.asarray(pr, dtype=np.float32),
            np.asarray(mk, dtype=np.float32),
            rd_arr,
            float(fallback_value),
        )

    passrate_subject_cond = _cond_pair("subject")            # 15
    passrate_family_cond = _cond_pair("family")              # 16
    passrate_macro_family_cond = _cond_pair("macro_family")  # 17
    passrate_organization_cond = _cond_pair("organization")  # 18
    passrate_bench_cond = _cond_pair("bench_match")          # 19

    # 20. neighbor_freshness_diff (pre-aggregated scalar by caller).
    fresh_val = cond_inputs.get("neighbor_freshness_diff")
    fresh_redact = cond_inputs.get("freshness_redact")
    if fresh_val is None:
        freshness_diff = np.full(B, fb, dtype=np.float32)
    else:
        freshness_diff = np.asarray(fresh_val, dtype=np.float32).reshape(-1)
        if fresh_redact is not None:
            redact_arr = np.asarray(fresh_redact, dtype=np.float32).reshape(-1)
            freshness_diff = np.where(redact_arr > 0, fb, freshness_diff).astype(np.float32)

    # 21. n_distinct_subjects_in_neighborhood (per-item stat, no redaction).
    distinct_per_neighbor = cond_inputs.get("distinct_subj_per_neighbor")
    if distinct_per_neighbor is None:
        n_distinct_subj = np.full(B, fb, dtype=np.float32)
    else:
        ds = np.asarray(distinct_per_neighbor, dtype=np.float32)
        n_distinct_subj = np.log1p(ds.mean(axis=1)).astype(np.float32)

    # 22. cluster_passrate_subject_query (per-row scalar).
    cps_val = cond_inputs.get("cluster_passrate_subject_query")
    cps_redact = cond_inputs.get("cluster_redact")
    if cps_val is None:
        cluster_passrate_subj = np.full(B, fb, dtype=np.float32)
    else:
        cluster_passrate_subj = np.asarray(cps_val, dtype=np.float32).reshape(-1)
        if cps_redact is not None:
            redact_arr = np.asarray(cps_redact, dtype=np.float32).reshape(-1)
            cluster_passrate_subj = np.where(
                redact_arr > 0, fb, cluster_passrate_subj
            ).astype(np.float32)

    out = np.stack(
        [
            pr_mean.astype(np.float32),
            weighted,
            pr_std,
            coverage,
            top1_label,
            top1_sim,
            mean_sim,
            n_labeled_log,
            eff_count,
            top1_minus_topk,
            bootstrap_se,
            entropy,
            top1_label_match,
            sim_skew,
            distance_to_kth,
            passrate_subject_cond,
            passrate_family_cond,
            passrate_macro_family_cond,
            passrate_organization_cond,
            passrate_bench_cond,
            freshness_diff,
            n_distinct_subj,
            cluster_passrate_subj,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=fallback_value, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(out, dtype=np.float32)


# ===========================================================================
# Conditional-passrate-context runtime loader.
#
# Reads the bag written by src/nn_features.py::ConditionalPassrateContext.save
# (the DR/artifacts/nn_features/ layout) and resolves per-query cond_inputs
# exactly as src/nn_features.py::_resolve_conditional_inputs does, restricted
# to the SINGLE-ROW (B=1) inference path the submodels use.
#
# File layout it expects under `ctx_dir`:
#   conditional_meta.json
#   subject_passrate.npz / subject_passrate_mask.npz
#   family_passrate.npz  / family_passrate_mask.npz
#   macro_family_passrate.npz / macro_family_passrate_mask.npz
#   organization_passrate.npz / organization_passrate_mask.npz
#   subject_to_family_id.npy / subject_to_macro_family_id.npy /
#     subject_to_organization_id.npy
#   item_benchmark_id.npy / item_benchmark_age.npy /
#     item_distinct_subj_count.npy / item_global_passrate.npy /
#     item_global_passrate_mask.npy / item_cluster_id.npy
#   cluster_subject_passrate.npz / cluster_subject_passrate_mask.npz
#
# Optional sidecar (ships next to the bag; written by the exporter):
#   benchmark_to_id.json    -- maps raw benchmark string -> item_benchmark_id
#                              space, so the runtime can resolve a query's
#                              benchmark id from its bc_key. If absent, cell
#                              19 (benchmark-conditional) redacts.
# ===========================================================================


class ConditionalContextRuntime:
    """Lazy, scipy-optional loader for the conditional NN context.

    Construct with ``ConditionalContextRuntime.maybe_load(ctx_dir)`` -- it
    returns ``None`` (so the aggregator falls back) if the directory or its
    meta is missing, or if scipy is unavailable. Never raises on a partial
    bundle; any missing component degrades the corresponding cell to
    fallback.
    """

    def __init__(self) -> None:
        self.ok: bool = False
        self.meta: dict = {}
        self.n_subjects = 0
        self.n_items = 0
        self.n_clusters = 0
        # sparse CSR (scipy) matrices
        self.subject_pr = None
        self.subject_mk = None
        self.family_pr = None
        self.family_mk = None
        self.macro_pr = None
        self.macro_mk = None
        self.org_pr = None
        self.org_mk = None
        self.cluster_pr = None
        self.cluster_mk = None
        # dense per-subject / per-item arrays
        self.subject_to_family_id = None
        self.subject_to_macro_family_id = None
        self.subject_to_organization_id = None
        self.item_benchmark_id = None
        self.item_benchmark_age = None
        self.item_distinct_subj_count = None
        self.item_global_passrate = None
        self.item_global_passrate_mask = None
        self.item_cluster_id = None
        # benchmark string -> id sidecar (optional)
        self.benchmark_to_id: dict[str, int] = {}

    @classmethod
    def maybe_load(cls, ctx_dir: str | Path) -> "ConditionalContextRuntime | None":
        ctx_dir = Path(ctx_dir)
        meta_path = ctx_dir / "conditional_meta.json"
        if not meta_path.exists():
            LOG.info("nn23: no conditional context at %s; cells 15..22 -> fallback", ctx_dir)
            return None
        try:
            from scipy import sparse  # type: ignore
        except Exception as exc:  # noqa: BLE001
            LOG.warning("nn23: scipy unavailable (%s); cells 15..22 -> fallback", exc)
            return None
        self = cls()
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            schema_dim = int(self.meta.get("feature_dim", NN_FEATURE_DIM_23))
            if schema_dim != NN_FEATURE_DIM_23:
                LOG.warning(
                    "nn23: conditional context feature_dim=%d != %d; refusing to "
                    "load (cells 15..22 -> fallback)",
                    schema_dim,
                    NN_FEATURE_DIM_23,
                )
                return None
            self.n_subjects = int(self.meta.get("n_subjects", 0))
            self.n_items = int(self.meta.get("n_items", 0))
            self.n_clusters = int(self.meta.get("n_clusters", 0))

            def _npz(name: str):
                p = ctx_dir / name
                return sparse.load_npz(p).tocsr() if p.exists() else None

            def _npy(name: str):
                p = ctx_dir / name
                return np.load(p) if p.exists() else None

            self.subject_pr = _npz("subject_passrate.npz")
            self.subject_mk = _npz("subject_passrate_mask.npz")
            self.family_pr = _npz("family_passrate.npz")
            self.family_mk = _npz("family_passrate_mask.npz")
            self.macro_pr = _npz("macro_family_passrate.npz")
            self.macro_mk = _npz("macro_family_passrate_mask.npz")
            self.org_pr = _npz("organization_passrate.npz")
            self.org_mk = _npz("organization_passrate_mask.npz")
            self.cluster_pr = _npz("cluster_subject_passrate.npz")
            self.cluster_mk = _npz("cluster_subject_passrate_mask.npz")

            self.subject_to_family_id = _npy("subject_to_family_id.npy")
            self.subject_to_macro_family_id = _npy("subject_to_macro_family_id.npy")
            self.subject_to_organization_id = _npy("subject_to_organization_id.npy")
            self.item_benchmark_id = _npy("item_benchmark_id.npy")
            self.item_benchmark_age = _npy("item_benchmark_age.npy")
            self.item_distinct_subj_count = _npy("item_distinct_subj_count.npy")
            self.item_global_passrate = _npy("item_global_passrate.npy")
            self.item_global_passrate_mask = _npy("item_global_passrate_mask.npy")
            self.item_cluster_id = _npy("item_cluster_id.npy")

            bmap_path = ctx_dir / "benchmark_to_id.json"
            if bmap_path.exists():
                try:
                    raw = json.loads(bmap_path.read_text(encoding="utf-8")) or {}
                    self.benchmark_to_id = {str(k): int(v) for k, v in raw.items()}
                except Exception:
                    self.benchmark_to_id = {}
        except Exception as exc:  # noqa: BLE001
            LOG.warning("nn23: failed loading conditional context (%s); fallback", exc)
            return None
        self.ok = True
        return self

    # ----- single-row CSR lookup helpers (B=1 path) -----

    @staticmethod
    def _row_lookup(csr, row_id: int, cols: np.ndarray) -> np.ndarray:
        """Return values of `csr[row_id, cols]` (0 where absent). cols: [K]."""
        out = np.zeros(cols.shape[0], dtype=np.float32)
        if csr is None or row_id < 0 or row_id >= csr.shape[0]:
            return out
        start, end = csr.indptr[row_id], csr.indptr[row_id + 1]
        rc = csr.indices[start:end]
        rv = csr.data[start:end]
        if rc.size == 0:
            return out
        order = np.argsort(rc)
        sc = rc[order]
        sv = rv[order]
        pos = np.searchsorted(sc, cols)
        pc = np.clip(pos, 0, sc.size - 1)
        hit = (pos < sc.size) & (sc[pc] == cols)
        out[:] = np.where(hit, sv[pc], 0.0)
        return out

    @staticmethod
    def _row_mask(csr, row_id: int, cols: np.ndarray) -> np.ndarray:
        out = np.zeros(cols.shape[0], dtype=np.float32)
        if csr is None or row_id < 0 or row_id >= csr.shape[0]:
            return out
        start, end = csr.indptr[row_id], csr.indptr[row_id + 1]
        rc = csr.indices[start:end]
        if rc.size == 0:
            return out
        order = np.argsort(rc)
        sc = rc[order]
        pos = np.searchsorted(sc, cols)
        pc = np.clip(pos, 0, sc.size - 1)
        hit = (pos < sc.size) & (sc[pc] == cols)
        return hit.astype(np.float32)

    def _trait_id(self, mapping: np.ndarray | None, subject_id: int) -> int:
        if mapping is None or subject_id < 0 or subject_id >= mapping.shape[0]:
            return MISSING_TRAIT_ID
        return int(mapping[subject_id])

    def resolve_single(
        self,
        *,
        subject_id: int,
        neighbor_idx: np.ndarray,            # [K] int neighbor rows (training items)
        query_benchmark_id: int = -1,        # -1 = unknown / redacted
        query_benchmark_age: float = float("nan"),
        query_cluster_id: int = -1,          # -1 = unknown / redacted
        subject_meta_redacted: bool = False,
        fallback_value: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Build the B=1 ``cond_inputs`` dict for `_aggregate_nn_features_23`.

        Mirrors src/nn_features.py::_resolve_conditional_inputs but specialized
        to a single query row. Any unavailable table yields a redacting key
        (or omits it), so the aggregator falls back per-cell.
        """
        if not self.ok:
            return {}
        n_subj = int(self.n_subjects)
        if n_subj <= 0:
            return {}
        nidx = np.asarray(neighbor_idx, dtype=np.int64).reshape(-1)  # [K]
        K = nidx.shape[0]
        n_items = int(self.n_items)
        clip_nidx = np.clip(nidx, 0, max(n_items - 1, 0))

        sid_oor = (subject_id < 0) or (subject_id >= n_subj)
        subj_redact_full = bool(subject_meta_redacted) or sid_oor

        fam_id = MISSING_TRAIT_ID if sid_oor else self._trait_id(self.subject_to_family_id, subject_id)
        macro_id = MISSING_TRAIT_ID if sid_oor else self._trait_id(self.subject_to_macro_family_id, subject_id)
        org_id = MISSING_TRAIT_ID if sid_oor else self._trait_id(self.subject_to_organization_id, subject_id)

        fam_redact = subject_meta_redacted or (fam_id == MISSING_TRAIT_ID)
        macro_redact = subject_meta_redacted or (macro_id == MISSING_TRAIT_ID)
        org_redact = subject_meta_redacted or (org_id == MISSING_TRAIT_ID)

        out: dict[str, np.ndarray] = {}

        # 15. subject-conditional
        out["subject_passrates"] = self._row_lookup(self.subject_pr, subject_id, clip_nidx)[None, :]
        out["subject_masks"] = self._row_mask(self.subject_mk, subject_id, clip_nidx)[None, :]
        out["subject_redact"] = np.array([1.0 if subj_redact_full else 0.0], dtype=np.float32)

        # 16-18. family / macro_family / organization
        out["family_passrates"] = self._row_lookup(self.family_pr, fam_id, clip_nidx)[None, :]
        out["family_masks"] = self._row_mask(self.family_mk, fam_id, clip_nidx)[None, :]
        out["family_redact"] = np.array([1.0 if fam_redact else 0.0], dtype=np.float32)

        out["macro_family_passrates"] = self._row_lookup(self.macro_pr, macro_id, clip_nidx)[None, :]
        out["macro_family_masks"] = self._row_mask(self.macro_mk, macro_id, clip_nidx)[None, :]
        out["macro_family_redact"] = np.array([1.0 if macro_redact else 0.0], dtype=np.float32)

        out["organization_passrates"] = self._row_lookup(self.org_pr, org_id, clip_nidx)[None, :]
        out["organization_masks"] = self._row_mask(self.org_mk, org_id, clip_nidx)[None, :]
        out["organization_redact"] = np.array([1.0 if org_redact else 0.0], dtype=np.float32)

        # 19. benchmark-match (per-neighbor global passrate, masked to matching bench)
        if self.item_benchmark_id is not None and self.item_global_passrate is not None:
            nbench = self.item_benchmark_id[clip_nidx]
            bench_q = int(query_benchmark_id)
            bench_match = (nbench == bench_q) & (bench_q >= 0)
            bench_pr = self.item_global_passrate[clip_nidx]
            bench_mk_global = (
                self.item_global_passrate_mask[clip_nidx]
                if self.item_global_passrate_mask is not None
                else np.ones(K, dtype=np.float32)
            )
            out["bench_match_passrates"] = np.where(bench_match, bench_pr, 0.0).astype(np.float32)[None, :]
            out["bench_match_masks"] = (bench_match.astype(np.float32) * bench_mk_global).astype(np.float32)[None, :]
            out["bench_match_redact"] = np.array([1.0 if bench_q < 0 else 0.0], dtype=np.float32)

        # 20. freshness diff = query_age - mean(neighbor age over known ages)
        if self.item_benchmark_age is not None:
            nages = self.item_benchmark_age[clip_nidx]
            nage_mask = (~np.isnan(nages)).astype(np.float32)
            nage_safe = np.where(np.isnan(nages), 0.0, nages).astype(np.float32)
            n_known = float(nage_mask.sum())
            mean_neighbor_age = (nage_safe * nage_mask).sum() / max(n_known, 1.0) if n_known > 0 else 0.0
            q_age = float(query_benchmark_age)
            fresh_redact = (not math.isfinite(q_age)) or (n_known == 0)
            fresh_diff = 0.0 if fresh_redact else (q_age - mean_neighbor_age)
            out["neighbor_freshness_diff"] = np.array([fresh_diff], dtype=np.float32)
            out["freshness_redact"] = np.array([1.0 if fresh_redact else 0.0], dtype=np.float32)

        # 21. distinct subjects per neighbor (per-item, no redaction)
        if self.item_distinct_subj_count is not None:
            out["distinct_subj_per_neighbor"] = self.item_distinct_subj_count[clip_nidx].astype(np.float32)[None, :]

        # 22. cluster-passrate-subject-query
        q_cluster = int(query_cluster_id)
        cluster_redact = (q_cluster < 0) or sid_oor
        cps = 0.0
        if (not cluster_redact) and self.cluster_pr is not None and q_cluster < int(self.n_clusters):
            cps = float(self._row_lookup(self.cluster_pr, q_cluster, np.array([subject_id], dtype=np.int64))[0])
        out["cluster_passrate_subject_query"] = np.array([cps], dtype=np.float32)
        out["cluster_redact"] = np.array([1.0 if cluster_redact else 0.0], dtype=np.float32)

        return out

    def benchmark_id_for(self, benchmark: str) -> int:
        """Resolve a raw benchmark string to its item_benchmark_id. -1 if unknown."""
        return int(self.benchmark_to_id.get(str(benchmark), -1))


# ===========================================================================
# Drop-in 23-dim feature computation for the submodel's _TrainingItemCache.
#
# This is the function the submodel's `_get_nn_features` should call instead
# of `TRAINING_CACHE.compute_nn_features(...)`. It reuses the cache's existing
# `nearest()` + sparse subject-passrate lookup for cells 0..14, then overlays
# the conditional context for cells 15..22.
# ===========================================================================


def compute_nn_features_23(
    cache,                                   # the submodel's _TrainingItemCache
    item_emb: np.ndarray,
    subject_id: int,
    *,
    cond_ctx: "ConditionalContextRuntime | None" = None,
    query_benchmark_id: int = -1,
    query_benchmark_age: float = float("nan"),
    query_cluster_id: int = -1,
    subject_meta_redacted: bool = False,
    k: int = 16,
    fallback_value: float = 0.0,
    top1_missing_sentinel: float = -1.0,
) -> np.ndarray:
    """Return the full 23-dim NN feature vector (cells 0..7 == legacy 8).

    Uses the SAME neighbor query + subject-passrate lookup the legacy 8-dim
    path uses, so ``compute_nn_features_23(...)[:8]`` is bit-identical to the
    legacy ``cache.compute_nn_features(...)`` output. Cells 8..22 are added on
    top. When ``cond_ctx is None`` (or partial), cells 15..22 fall back.

    Returns ``np.zeros(23)`` under the same guard conditions the legacy path
    returns ``np.zeros(8)`` (no passrate cache, unseen subject, empty neighbor
    set) -- so the back-compat ``[:8]`` slice is still all-zeros, matching the
    head's "unseen subject -> zero NN vector" training contract.
    """
    dim = NN_FEATURE_DIM_23
    if (
        getattr(cache, "nn_passrate", None) is None
        or getattr(cache, "nn_passrate_mask", None) is None
        or subject_id is None
        or subject_id < 0
    ):
        return np.zeros(dim, dtype=np.float32)
    kk = int(k)
    if kk < 1:
        return np.zeros(dim, dtype=np.float32)
    idx, sims = cache.nearest(item_emb, k=kk)
    if idx.size == 0:
        return np.zeros(dim, dtype=np.float32)
    n_rows = cache.nn_passrate.shape[0]
    if subject_id >= n_rows:
        return np.zeros(dim, dtype=np.float32)

    row_pr = cache.nn_passrate.getrow(int(subject_id))
    row_mk = cache.nn_passrate_mask.getrow(int(subject_id))
    n_items = int(cache.nn_passrate.shape[1])
    passrates = np.zeros(kk, dtype=np.float32)
    masks = np.zeros(kk, dtype=np.float32)
    valid_idx = np.clip(idx, 0, n_items - 1)
    if row_pr.nnz > 0:
        cols = row_pr.indices
        vals = row_pr.data
        order = np.argsort(cols)
        sorted_cols = cols[order]
        sorted_vals = vals[order]
        pos = np.searchsorted(sorted_cols, valid_idx)
        pos_clipped = np.clip(pos, 0, sorted_cols.size - 1)
        hit = (pos < sorted_cols.size) & (sorted_cols[pos_clipped] == valid_idx)
        passrates = np.where(hit, sorted_vals[pos_clipped], 0.0).astype(np.float32)
    if row_mk.nnz > 0:
        mcols = row_mk.indices
        order = np.argsort(mcols)
        sorted_m = mcols[order]
        pos = np.searchsorted(sorted_m, valid_idx)
        pos_clipped = np.clip(pos, 0, sorted_m.size - 1)
        hit = (pos < sorted_m.size) & (sorted_m[pos_clipped] == valid_idx)
        masks = hit.astype(np.float32)

    cond_inputs = None
    if cond_ctx is not None and getattr(cond_ctx, "ok", False):
        try:
            cond_inputs = cond_ctx.resolve_single(
                subject_id=int(subject_id),
                neighbor_idx=valid_idx,
                query_benchmark_id=int(query_benchmark_id),
                query_benchmark_age=float(query_benchmark_age),
                query_cluster_id=int(query_cluster_id),
                subject_meta_redacted=bool(subject_meta_redacted),
                fallback_value=float(fallback_value),
            )
        except Exception:  # noqa: BLE001
            LOG.exception("nn23: conditional resolve failed; cells 15..22 -> fallback")
            cond_inputs = None

    feats = _aggregate_nn_features_23(
        passrates,
        masks,
        sims.astype(np.float32),
        fallback_value=fallback_value,
        top1_missing_sentinel=top1_missing_sentinel,
        cond_inputs=cond_inputs,
    )
    return feats.reshape(-1)[:dim].astype(np.float32, copy=False)


# ===========================================================================
# Self-test: cells 0..14 of the 23-dim output must equal the legacy 8-dim
# output (for cells 0..7) and be finite; cells 15..22 must redact to
# fallback when cond_inputs is None.
# ===========================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, K = 5, 16
    pr = rng.random((B, K)).astype(np.float32)
    mk = (rng.random((B, K)) > 0.3).astype(np.float32)
    sims = np.sort(rng.random((B, K)).astype(np.float32), axis=1)[:, ::-1].copy()

    full = _aggregate_nn_features_23(
        pr, mk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0, cond_inputs=None
    )
    assert full.shape == (B, 23), full.shape
    assert np.all(np.isfinite(full)), "non-finite output"
    # cells 15..22 must all be fallback (0.0) with no cond_inputs
    assert np.allclose(full[:, 15:], 0.0), "conditional cells should redact to 0"

    # legacy 8-dim reproduction check: re-derive cells 0..7 with the inline
    # legacy recipe and compare.
    pr_safe = np.where(mk > 0, pr, 0.0)
    nlab = mk.sum(1)
    has = nlab > 0
    msim = sims.mean(1)
    prm = np.where(has, pr_safe.sum(1) / np.maximum(nlab, 1.0), 0.0)
    assert np.allclose(full[:, 0], prm.astype(np.float32)), "cell0 drift"
    assert np.allclose(full[:, 6], msim.astype(np.float32)), "cell6 drift"
    assert np.allclose(full[:, 7], np.log1p(nlab).astype(np.float32)), "cell7 drift"
    print("nn23_runtime self-test OK: shape", full.shape,
          "| cells15..22 redact to fallback | legacy cells 0/6/7 match")
