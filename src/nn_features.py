"""Nearest-neighbor (NN) features for the residual MLP.

This module builds and queries a *full-fidelity* nearest-neighbor index over
training items at training time, and produces an 8-scalar feature vector per
``(subject, item)`` query summarizing the subject's performance on the
top-K nearest training items.

Feature schema (locked, fed into the residual MLP):

    nn_feats[0] = passrate_mean             # mean label of subject on top-K neighbors
    nn_feats[1] = passrate_weighted_mean    # similarity-weighted mean
    nn_feats[2] = passrate_std              # uncertainty signal
    nn_feats[3] = coverage                  # fraction of top-K neighbors with a
                                            # label for this subject
    nn_feats[4] = top1_label                # nearest neighbor's label
                                            # (or top1_missing_sentinel if missing)
    nn_feats[5] = top1_similarity           # similarity of the nearest neighbor
    nn_feats[6] = mean_similarity           # average similarity across top-K
    nn_feats[7] = n_labeled_neighbors_log1p # raw count, log1p-scaled

The pure aggregation helper ``_aggregate_nn_features`` is bit-identical
to the runtime implementation shipped inside ``submission/model.py`` -- the
runtime literally inlines the same function so train and test never drift.
The two-cache design (full-fidelity training index vs. compressed runtime
index) is intentional: training learns on the highest-quality signal, runtime
ships an aggressively PCA'd + int8 approximation, and the residual MLP
absorbs the compression noise via the ``coverage`` / ``n_labeled_neighbors``
features.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("nn_features")

NN_FEATURE_DIM: int = 23
NN_FEATURE_NAMES: tuple[str, ...] = (
    # Tier 0: legacy 8 features (locked order; cells [0..7]).
    "passrate_mean",
    "passrate_weighted_mean",
    "passrate_std",
    "coverage",
    "top1_label",
    "top1_similarity",
    "mean_similarity",
    "n_labeled_neighbors_log1p",
    # Tier 1+2: self-derived additions (cells [8..14]). All seven are
    # functions of (passrates, masks, sims) only -- no new training-
    # time tables required.
    "effective_neighbor_count",          # (Σw)^2 / Σ(w^2), Herfindahl-style
    "top1_minus_topk_similarity",        # top1_sim - mean_sim (peakedness)
    "bootstrap_se_passrate",             # passrate_std / sqrt(n_labeled)
    "neighbor_label_entropy",            # H(p_mean) with Bernoulli formula
    "top1_label_match",                  # 1.0 if top1 label > 0.5, else 0.0
    "sim_distribution_skew",             # (top1 - median) / (top1 - min)
    "distance_to_kth_neighbor",          # min similarity (the K-th NN), in [-1, 1] for cosine
    # Tier 2/3: conditional + context features (cells [15..22]). Each
    # of these requires the conditional training-time context (a bag
    # of per-trait passrate matrices + per-item context arrays).
    # When the runtime cannot build / load that context for a row, the
    # corresponding cell falls back to ``fallback_value`` (redaction).
    "passrate_subject_conditional",      # 15: per-(query subject, K NN) passrate.
                                         #     Aliased to passrate_mean today; kept
                                         #     as a separate column so the redaction
                                         #     semantics can diverge in the future.
    "passrate_family_conditional",       # 16: passrate of "subjects in same family
                                         #     as query subject" on each NN, aggregated.
                                         #     Redacted when query subject's family is
                                         #     the MISSING token (id 0).
    "passrate_macro_family_conditional", # 17: same idea for macro_family.
    "passrate_organization_conditional", # 18: same idea for organization.
    "passrate_benchmark_conditional",    # 19: per-NN global passrate, restricted to
                                         #     neighbors whose benchmark_id matches
                                         #     the query benchmark. Redacted when
                                         #     query benchmark is unknown / hidden.
    "neighbor_freshness_diff",           # 20: query_benchmark_age - mean(NN benchmark_age).
                                         #     Positive => query is "fresher" than its
                                         #     neighbors. Redacted when query age missing.
    "n_distinct_subjects_in_neighborhood",# 21: log1p(mean over K neighbors of the
                                         #     distinct-subject count for that train
                                         #     item). No redaction (per-item stat).
    "cluster_passrate_subject_query",    # 22: query subject's overall passrate within
                                         #     the cluster the query item belongs to.
                                         #     Redacted when query cluster_id is unknown.
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NNFeaturesConfig:
    """Hyperparameters for nearest-neighbor feature computation.

    ``feature_dim`` is locked to the schema above; it is exposed for
    introspection but not user-tunable.
    """

    enabled: bool = True
    k: int = 16
    similarity: str = "cosine"            # cosine | ip | l2
    feature_dim: int = NN_FEATURE_DIM
    fallback_value: float = 0.0
    top1_missing_sentinel: float = -1.0
    exclude_self_in_training: bool = True
    cache_dir: str = "artifacts/nn_features"
    # Promote the FAISS IndexFlat to GPU when one is visible. The
    # on-disk artifact is always the CPU index (faiss.write_index does
    # not accept a GPU handle), so this only affects query-time speed.
    # Ignored on machines with no GPU / no faiss-gpu build.
    prefer_gpu: bool = True
    # When True, drop ``self.embeddings`` after FAISS is successfully
    # built. The FAISS IndexFlat keeps its own internal copy of the
    # embeddings, so the duplicate is pure overhead -- saves ~N*D*4
    # bytes (1.6 GB per 100k items at D=4096). The brute-force fallback
    # in ``nearest()`` is unavailable while embeddings are dropped, so
    # a FAISS-search failure becomes a hard error instead of silently
    # falling back. Documented as opt-in for that reason.
    free_embeddings_after_faiss: bool = False

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "NNFeaturesConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            k=int(d.get("k", 16)),
            similarity=str(d.get("similarity", "cosine")),
            feature_dim=int(d.get("feature_dim", NN_FEATURE_DIM)),
            fallback_value=float(d.get("fallback_value", 0.0)),
            top1_missing_sentinel=float(d.get("top1_missing_sentinel", -1.0)),
            exclude_self_in_training=bool(
                d.get("exclude_self_in_training", True)
            ),
            cache_dir=str(d.get("cache_dir", "artifacts/nn_features")),
            prefer_gpu=bool(d.get("prefer_gpu", True)),
            free_embeddings_after_faiss=bool(
                d.get("free_embeddings_after_faiss", False)
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numerics: shared aggregation helper (copied verbatim into submission/model.py)
# ---------------------------------------------------------------------------


def _aggregate_trait_conditional(
    passrates: np.ndarray,        # [B, K] per-(trait_id, neighbor) mean labels (0 where missing)
    masks: np.ndarray,            # [B, K] 1 where observation exists for the (trait_id, neighbor) pair
    redact_row: np.ndarray,       # [B] 1 if THIS row should be redacted (trait unknown)
    fallback_value: float,
) -> np.ndarray:
    """Aggregate a per-(B, K) trait-conditional passrate to a per-row scalar.

    Same mean-over-labeled-neighbors recipe as ``passrate_mean``. Rows
    where ``redact_row`` is 1 are forced to ``fallback_value``
    independent of how many neighbors were observed; this implements
    the metadata-redaction contract for the conditional passrate
    features (e.g. when the query subject's family is the
    ``__MISSING__`` token, the family-conditional cell goes to fallback
    even if the per-(family, neighbor) table happens to have entries).
    """
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


def _aggregate_nn_features(
    neighbor_passrates: np.ndarray,    # [B, K] mean labels (NaN where missing)
    neighbor_masks: np.ndarray,        # [B, K] 1 where labeled, 0 otherwise
    similarities: np.ndarray,          # [B, K]
    *,
    fallback_value: float,
    top1_missing_sentinel: float,
    cond_inputs: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pure NN aggregation. Returns [B, NN_FEATURE_DIM] float32.

    ``neighbor_passrates`` may contain NaN where no observation exists -- the
    mask must independently report which entries are valid. Callers are
    responsible for keeping the two arrays in sync.

    Output columns are locked to :data:`NN_FEATURE_NAMES`:
      * cells 0..7 are the legacy aggregators
      * cells 8..14 are the self-derived additions (effective neighbor
        count, top1-topk gap, bootstrap SE, label entropy, top1 label
        match, similarity-distribution skew, distance to k-th neighbor)
        -- all derivable from the ``(passrates, masks, sims)`` triple
      * cells 15..22 are the **conditional / context** features: per-trait
        passrate aggregations (subject / family / macro_family /
        organization), benchmark-match passrate, neighbor freshness diff,
        distinct subjects in the neighborhood, and the query subject's
        cluster-conditional passrate. Computing them requires extra
        per-row inputs that the caller pre-resolves through the
        :class:`ConditionalPassrateContext`; pass them through
        ``cond_inputs``. When ``cond_inputs`` is ``None`` (or any
        individual key is absent), the corresponding cell falls back to
        ``fallback_value`` -- the runtime uses this redaction path when
        the conditional cache is unavailable or when subject / benchmark
        metadata is hidden for that row.

    This function is shipped *both* in this training module and inlined
    inside the runtime ``submission/model.py`` (rendered by
    ``export_submission.py``). If you change one, change the other.
    """
    passrates = np.asarray(neighbor_passrates, dtype=np.float32)
    masks = np.asarray(neighbor_masks, dtype=np.float32)
    sims = np.asarray(similarities, dtype=np.float32)

    if passrates.ndim == 1:
        passrates = passrates[None, :]
        masks = masks[None, :]
        sims = sims[None, :]

    B, K = passrates.shape

    # Replace NaNs in passrate with 0 so they don't poison the weighted sums;
    # the mask is the source of truth for "this entry is real".
    pr_safe = np.where(masks > 0, passrates, 0.0).astype(np.float32)

    n_labeled = masks.sum(axis=1)                                # [B]
    has_any = n_labeled > 0

    mean_sim = sims.mean(axis=1).astype(np.float32)              # [B]

    # mean
    pr_sum = pr_safe.sum(axis=1)
    pr_mean = np.where(has_any, pr_sum / np.maximum(n_labeled, 1.0), fallback_value)

    # weighted mean (by similarity, only over labeled entries)
    sim_safe = np.where(masks > 0, sims, 0.0).astype(np.float32)
    # Shift sims into a non-negative weight space. For cosine / IP sims this
    # rescales the [-1, 1] (or unbounded ip) range to [0, +inf); for l2-like
    # distances this would need negation upstream. Documented as "similarity".
    weights = np.clip(sim_safe, 0.0, None)
    weight_sum = weights.sum(axis=1)
    weighted = np.where(
        (weight_sum > 1e-9) & has_any,
        (weights * pr_safe).sum(axis=1) / np.maximum(weight_sum, 1e-9),
        np.where(has_any, pr_mean, fallback_value),
    ).astype(np.float32)

    # std (over labeled entries only; biased / population variance for stability)
    diff = (pr_safe - pr_mean[:, None]) * masks
    sq = (diff * diff).sum(axis=1)
    var = np.where(has_any, sq / np.maximum(n_labeled, 1.0), 0.0)
    pr_std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    pr_std = np.where(has_any, pr_std, fallback_value).astype(np.float32)

    coverage = (n_labeled / float(max(1, K))).astype(np.float32)

    # top-1: nearest neighbor is column 0 (callers MUST present neighbors in
    # descending similarity order).
    top1_mask = masks[:, 0]
    top1_label = np.where(top1_mask > 0, passrates[:, 0], top1_missing_sentinel)
    top1_label = top1_label.astype(np.float32)
    top1_sim = sims[:, 0].astype(np.float32)

    n_labeled_log = np.log1p(n_labeled).astype(np.float32)

    # ----- Self-derived additions (cells 8..14) -----
    # 8. effective_neighbor_count: Herfindahl-style "effective sample size"
    #    of the similarity weights. (Σw)² / Σ(w²). Distinguishes a single
    #    dominant neighbor from a flat distribution of similar weights.
    weight_sq_sum = (weights * weights).sum(axis=1)
    eff_count = np.where(
        weight_sq_sum > 1e-9,
        (weight_sum * weight_sum) / np.maximum(weight_sq_sum, 1e-9),
        0.0,
    ).astype(np.float32)

    # 9. top1_minus_topk_similarity: top1_sim - mean_sim. Positive +
    #    large => the nearest neighbor stands out; near zero => flat
    #    neighborhood. Cheap peakedness signal the model otherwise has
    #    to construct by subtracting features.
    top1_minus_topk = (top1_sim - mean_sim).astype(np.float32)

    # 10. bootstrap_se_passrate: passrate_std / sqrt(n_labeled). The
    #     standard error of the mean estimate; lets downstream heads
    #     learn an inverse-variance shrinkage between the NN-derived
    #     prediction and the population prior.
    bootstrap_se = np.where(
        n_labeled > 0,
        pr_std / np.sqrt(np.maximum(n_labeled, 1.0)),
        fallback_value,
    ).astype(np.float32)

    # 11. neighbor_label_entropy: Bernoulli entropy at p=pr_mean. Captures
    #     "neighbors all agree" vs "50/50 ambiguous" beyond what
    #     passrate_std captures (entropy != variance for the same p).
    p_clip = np.clip(pr_mean, 1e-7, 1.0 - 1e-7)
    entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))
    entropy = np.where(has_any, entropy, 0.0).astype(np.float32)

    # 12. top1_label_match: 1.0 if the nearest labeled neighbor's mean
    #     label is > 0.5, else 0.0. Tree models (Member 2 GBDT in the
    #     stacker) sometimes pick this up faster than the continuous
    #     top1_label because the tree can split cleanly on the boolean.
    top1_label_match = np.where(
        top1_mask > 0,
        (passrates[:, 0] > 0.5).astype(np.float32),
        fallback_value,
    ).astype(np.float32)

    # 13. sim_distribution_skew: (top1 - median) / (top1 - min + eps).
    #     A single scalar that captures another shape parameter of the
    #     similarity distribution that mean / top1 / mean_sim miss.
    sim_min = sims.min(axis=1).astype(np.float32)
    sim_median = np.median(sims, axis=1).astype(np.float32)
    span = top1_sim - sim_min
    sim_skew = np.where(
        np.abs(span) > 1e-9,
        (top1_sim - sim_median) / np.where(np.abs(span) > 1e-9, span, 1.0),
        0.0,
    ).astype(np.float32)

    # 14. distance_to_kth_neighbor: similarity of the K-th (least-similar)
    #     retrieved neighbor. A real OOD detector: when even the nearest
    #     items are far away, every other NN-derived feature is unreliable
    #     and the model can learn to down-weight them. For cosine sim this
    #     lives in [-1, 1]; for IP sim, unbounded. We pass it through
    #     unchanged because downstream heads operate on z-scored inputs.
    distance_to_kth = sims[:, -1].astype(np.float32)

    # ----- Tier 2/3: conditional + context features (cells 15..22) -----
    # Each cell is initialized to the redaction fallback. When
    # ``cond_inputs`` carries the matching arrays, we overwrite. Any
    # missing key keeps the cell at fallback (per-cell graceful
    # degradation -- a half-shipped runtime cache still produces a
    # valid feature vector instead of an exception).
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

    # 15. passrate_subject_conditional. Same recipe as cells 0 / 16-18
    #     but keyed on subject_id rather than a subject trait.
    passrate_subject_cond = _cond_pair("subject")

    # 16-18. family / macro_family / organization conditional passrates.
    passrate_family_cond = _cond_pair("family")
    passrate_macro_family_cond = _cond_pair("macro_family")
    passrate_organization_cond = _cond_pair("organization")

    # 19. passrate_benchmark_conditional: per-row aggregate of the
    #     neighbors' GLOBAL passrates restricted to neighbors that share
    #     the query benchmark. The caller supplies pre-masked per-(B, K)
    #     arrays; we reuse the same trait-conditional aggregator.
    passrate_bench_cond = _cond_pair("bench_match")

    # 20. neighbor_freshness_diff: per-row scalar (already aggregated
    #     by the caller as ``query_age - mean(neighbor_age)`` over
    #     neighbors with known ages). The caller passes the redaction
    #     mask via ``freshness_redact``.
    fresh_val = cond_inputs.get("neighbor_freshness_diff")
    fresh_redact = cond_inputs.get("freshness_redact")
    if fresh_val is None:
        freshness_diff = np.full(B, fb, dtype=np.float32)
    else:
        freshness_diff = np.asarray(fresh_val, dtype=np.float32).reshape(-1)
        if fresh_redact is not None:
            redact_arr = np.asarray(fresh_redact, dtype=np.float32).reshape(-1)
            freshness_diff = np.where(redact_arr > 0, fb, freshness_diff).astype(np.float32)

    # 21. n_distinct_subjects_in_neighborhood: log1p of the mean per-
    #     neighbor distinct-subject count. Per-item stat, no redaction.
    distinct_per_neighbor = cond_inputs.get("distinct_subj_per_neighbor")
    if distinct_per_neighbor is None:
        n_distinct_subj = np.full(B, fb, dtype=np.float32)
    else:
        ds = np.asarray(distinct_per_neighbor, dtype=np.float32)
        # Average across the K neighbors; note we DO NOT mask by
        # ``masks`` here -- distinct-subject counts are a property of
        # the train item itself, independent of whether the query
        # subject has answered it.
        n_distinct_subj = np.log1p(ds.mean(axis=1)).astype(np.float32)

    # 22. cluster_passrate_subject_query: per-row scalar (subject's
    #     overall passrate within the query item's cluster). The
    #     caller pre-resolves the lookup; we apply redaction.
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


# ---------------------------------------------------------------------------
# TrainingNNIndex: full-fidelity index used at training / eval time
# ---------------------------------------------------------------------------


class TrainingNNIndex:
    """Full-fidelity NN index over training items.

    Holds ``item_keys`` (list[str]), an fp32 embedding matrix [N, D]
    (optionally L2-normalized when ``similarity == "cosine"``), and an
    optional FAISS index. Persists to ``out_dir`` with deterministic file
    names; the on-disk layout is idempotent (re-using a fresh
    ``build_from_lookup`` call returns the cached files if the item_keys
    match).

    The index never persists self-exclusion state. The caller passes
    ``query_item_keys`` to ``nearest()`` so each query can skip its own row
    when needed.
    """

    EMBEDDINGS_FILE = "training_index_embeddings.npy"
    KEYS_FILE = "training_index_keys.json"
    META_FILE = "training_index_meta.json"
    FAISS_FILE = "training_index.faiss"

    def __init__(self, cfg: NNFeaturesConfig):
        self.cfg = cfg
        self.item_keys: list[str] = []
        self.key_to_row: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None  # [N, D] fp32 (normed for cosine)
        self._faiss_index = None
        self._faiss_attempted = False
        self._faiss_error: str | None = None

    # ------------------------------------------------------------------ build

    @classmethod
    def build_from_lookup(
        cls,
        item_emb_lookup: Mapping[str, np.ndarray],
        out_dir: Path,
        cfg: NNFeaturesConfig,
        *,
        item_keys: list[str] | None = None,
    ) -> "TrainingNNIndex":
        """Stack the embeddings, normalize if cosine, build FAISS, persist.

        Idempotent: if ``out_dir`` already contains a matching index (same
        item_keys, same similarity), the cached files are reused.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if item_keys is None:
            item_keys = sorted(item_emb_lookup.keys())
        else:
            item_keys = [str(k) for k in item_keys]

        # Try to load an existing index whose keys + similarity match.
        existing = cls._maybe_load_existing(out_dir, cfg, item_keys)
        if existing is not None:
            LOG.info(
                "TrainingNNIndex: reusing on-disk index at %s (N=%d D=%d)",
                out_dir,
                existing.embeddings.shape[0],
                existing.embeddings.shape[1],
            )
            return existing

        first = np.asarray(item_emb_lookup[item_keys[0]], dtype=np.float32)
        D = int(first.shape[-1])
        N = len(item_keys)
        emb = np.empty((N, D), dtype=np.float32)
        for i, k in enumerate(item_keys):
            v = np.asarray(item_emb_lookup[k], dtype=np.float32)
            if v.shape != (D,):
                raise ValueError(
                    f"embedding for item_key={k!r} has shape {v.shape}; expected {(D,)}"
                )
            emb[i] = v
        if cfg.similarity == "cosine":
            # Chunked in-place L2-normalize. Row-wise norm + division is
            # bit-identical regardless of how we slice rows, but doing it
            # one chunk at a time avoids two transient [N, D] copies that
            # the previous one-shot form generated:
            #
            #   1. ``np.linalg.norm`` materializes ``emb * emb`` internally
            #      (size [N, D]).
            #   2. ``emb = (emb / norms).astype(np.float32)`` allocates a
            #      fresh [N, D] result before overwriting the binding.
            #
            # The chunked form bounds the transient peak to
            # ``norm_chunk * D * 4`` bytes (~67 MB at chunk=4096, D=4096),
            # which is what keeps Qwen3-Embedding-8B fitting on a 12 GB
            # Colab.
            norm_chunk = 4096
            for _s in range(0, N, norm_chunk):
                _e = min(_s + norm_chunk, N)
                _row = emb[_s:_e]
                _n = np.linalg.norm(_row, axis=1, keepdims=True)
                _n[_n < 1e-12] = 1.0
                _row /= _n.astype(np.float32, copy=False)

        np.save(out_dir / cls.EMBEDDINGS_FILE, emb)
        (out_dir / cls.KEYS_FILE).write_text(
            json.dumps(item_keys), encoding="utf-8"
        )
        (out_dir / cls.META_FILE).write_text(
            json.dumps(
                {
                    "n_items": N,
                    "dim": D,
                    "similarity": cfg.similarity,
                    "feature_dim": cfg.feature_dim,
                    "k": cfg.k,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self = cls(cfg)
        self.item_keys = item_keys
        self.key_to_row = {k: i for i, k in enumerate(item_keys)}
        self.embeddings = emb
        self._try_build_faiss(out_dir)
        # FAISS IndexFlat keeps its own internal copy of the [N, D]
        # vectors, so the duplicate ``self.embeddings`` is overhead.
        # The streaming + chunked query paths only need ``key_to_row``
        # plus the FAISS index, so dropping ``self.embeddings`` here is
        # safe whenever FAISS is available.
        if (
            bool(getattr(cfg, "free_embeddings_after_faiss", False))
            and self._faiss_index is not None
        ):
            LOG.info(
                "TrainingNNIndex: dropping self.embeddings after FAISS "
                "build (saves ~%.2f GB)",
                float(N) * float(D) * 4.0 / (1024.0**3),
            )
            self.embeddings = None
        return self

    @classmethod
    def load(cls, in_dir: Path, cfg: NNFeaturesConfig) -> "TrainingNNIndex":
        in_dir = Path(in_dir)
        item_keys = json.loads((in_dir / cls.KEYS_FILE).read_text(encoding="utf-8"))
        emb = np.load(in_dir / cls.EMBEDDINGS_FILE).astype(np.float32, copy=False)
        self = cls(cfg)
        self.item_keys = list(map(str, item_keys))
        self.key_to_row = {k: i for i, k in enumerate(self.item_keys)}
        self.embeddings = emb
        self._maybe_load_faiss(in_dir)
        return self

    @classmethod
    def _maybe_load_existing(
        cls,
        out_dir: Path,
        cfg: NNFeaturesConfig,
        item_keys: list[str],
    ) -> "TrainingNNIndex | None":
        keys_path = out_dir / cls.KEYS_FILE
        emb_path = out_dir / cls.EMBEDDINGS_FILE
        meta_path = out_dir / cls.META_FILE
        if not (keys_path.exists() and emb_path.exists() and meta_path.exists()):
            return None
        try:
            existing_keys = json.loads(keys_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if list(map(str, existing_keys)) != list(map(str, item_keys)):
            return None
        if str(meta.get("similarity", "")) != cfg.similarity:
            return None
        return cls.load(out_dir, cfg)

    @classmethod
    def try_load_existing(
        cls,
        out_dir: Path,
        cfg: NNFeaturesConfig,
        item_keys: Sequence[str],
    ) -> "TrainingNNIndex | None":
        """Public cache-only loader. Returns ``None`` if no matching index.

        Use this when the caller already has cached NN feature matrices on
        disk and only needs the index for downstream sanity checks. Unlike
        :meth:`build_from_lookup`, this never falls through to a rebuild --
        it simply returns ``None`` if anything about the cached files does
        not match (keys, similarity, missing files).
        """
        return cls._maybe_load_existing(
            Path(out_dir), cfg, [str(k) for k in item_keys]
        )

    # --------------------------------------------------------------- backend

    def _maybe_to_gpu(self, faiss_module, cpu_index):
        """Move ``cpu_index`` to GPU when ``cfg.prefer_gpu`` is on and a GPU exists.

        Returns the GPU-resident index on success, the original CPU
        index on any failure. We persist a CPU copy to disk regardless
        (faiss.write_index doesn't accept a GPU index) so reloads are
        symmetric.
        """
        if not bool(getattr(self.cfg, "prefer_gpu", True)):
            return cpu_index
        try:
            n_gpus = int(faiss_module.get_num_gpus())
        except Exception:  # noqa: BLE001
            return cpu_index
        if n_gpus <= 0:
            return cpu_index
        try:
            res = faiss_module.StandardGpuResources()
            return faiss_module.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as exc:  # noqa: BLE001
            LOG.info(
                "FAISS GPU promotion failed (%s); keeping CPU index", exc
            )
            return cpu_index

    def _try_build_faiss(self, out_dir: Path) -> None:
        if self.embeddings is None:
            return
        try:
            import faiss  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = (
                f"faiss not installed ({type(exc).__name__}: {exc})"
            )
            self._faiss_attempted = True
            return
        try:
            N, D = self.embeddings.shape
            if self.cfg.similarity in ("cosine", "ip"):
                index = faiss.IndexFlatIP(D)
            else:
                index = faiss.IndexFlatL2(D)
            index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
            # Persist the CPU copy first so the on-disk format stays
            # symmetric whether or not a GPU is attached.
            faiss.write_index(index, str(out_dir / self.FAISS_FILE))
            # Then optionally promote to GPU for query-time speed.
            self._faiss_index = self._maybe_to_gpu(faiss, index)
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = f"faiss build failed: {exc}"
        finally:
            self._faiss_attempted = True

    def _maybe_load_faiss(self, in_dir: Path) -> None:
        path = in_dir / self.FAISS_FILE
        if not path.exists():
            return
        try:
            import faiss  # type: ignore

            cpu_index = faiss.read_index(str(path))
            self._faiss_index = self._maybe_to_gpu(faiss, cpu_index)
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = f"faiss read failed: {exc}"

    # --------------------------------------------------------------- queries

    def _normalize_queries(self, queries: np.ndarray) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.cfg.similarity == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            q = (q / norms).astype(np.float32)
        return np.ascontiguousarray(q, dtype=np.float32)

    def nearest(
        self,
        query_embeds: np.ndarray,
        k: int | None = None,
        *,
        exclude_self: bool = True,
        query_keys: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_indices [B, k], similarities [B, k]).

        When ``exclude_self`` is True and ``query_keys`` matches a training
        item, the first match equal to that key is dropped. Implementation
        uses k+1 candidates and removes the self-match (when present), which
        is robust to both exact-self matches and PCA / quantization-induced
        ordering quirks.
        """
        if self.embeddings is None:
            raise RuntimeError("TrainingNNIndex.nearest called before build")
        k_eff = int(k or self.cfg.k)
        kk = max(1, k_eff)
        # Query k+1 so we can drop the self-match cleanly when present.
        kq = kk + 1
        kq = min(kq, self.embeddings.shape[0])
        q = self._normalize_queries(query_embeds)

        if self._faiss_index is not None:
            try:
                sims, idx = self._faiss_index.search(q, kq)
            except Exception as exc:  # noqa: BLE001
                if self.embeddings is None:
                    raise RuntimeError(
                        "FAISS search failed and brute-force fallback is "
                        "unavailable because self.embeddings was freed "
                        "(free_embeddings_after_faiss=True). Original "
                        f"error: {exc}"
                    ) from exc
                LOG.warning(
                    "FAISS search failed (%s); brute-force fallback", exc
                )
                sims, idx = self._brute_force(q, kq)
        else:
            if self.embeddings is None:
                raise RuntimeError(
                    "TrainingNNIndex.nearest: FAISS index unavailable and "
                    "self.embeddings was freed; cannot search."
                )
            sims, idx = self._brute_force(q, kq)

        if self.cfg.similarity == "l2":
            # IndexFlatL2 returns squared distances; flip sign so callers
            # can still treat the second array as "higher = closer".
            sims = -sims

        if exclude_self and query_keys is not None:
            self_rows = np.array(
                [self.key_to_row.get(str(k), -1) for k in query_keys],
                dtype=np.int64,
            )
            idx, sims = self._strip_self(idx, sims, self_rows, kk)
        else:
            idx = idx[:, :kk]
            sims = sims[:, :kk]
        return idx.astype(np.int64, copy=False), sims.astype(np.float32, copy=False)

    def _brute_force(
        self, q: np.ndarray, kq: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Higher = better for cosine / IP; for L2 we return *squared distances*
        # to match the FAISS convention.
        embs = self.embeddings
        if self.cfg.similarity == "l2":
            d2 = (
                (q * q).sum(axis=1, keepdims=True)
                + (embs * embs).sum(axis=1)[None, :]
                - 2.0 * q @ embs.T
            )
            d2 = np.clip(d2, 0.0, None).astype(np.float32)
            order = np.argpartition(d2, min(kq, d2.shape[1] - 1), axis=1)[:, :kq]
            row_idx = np.arange(d2.shape[0])[:, None]
            best = order[
                row_idx, np.argsort(d2[row_idx, order], axis=1, kind="stable")
            ]
            sims = d2[row_idx, best].astype(np.float32)
            return sims, best.astype(np.int64)
        sims_all = (q @ embs.T).astype(np.float32)
        order = np.argpartition(-sims_all, min(kq, sims_all.shape[1] - 1), axis=1)[
            :, :kq
        ]
        row_idx = np.arange(sims_all.shape[0])[:, None]
        best = order[
            row_idx, np.argsort(-sims_all[row_idx, order], axis=1, kind="stable")
        ]
        sims = sims_all[row_idx, best].astype(np.float32)
        return sims, best.astype(np.int64)

    @staticmethod
    def _strip_self(
        idx: np.ndarray,
        sims: np.ndarray,
        self_rows: np.ndarray,
        kk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_idx = np.empty((idx.shape[0], kk), dtype=np.int64)
        out_sims = np.empty((idx.shape[0], kk), dtype=np.float32)
        for i in range(idx.shape[0]):
            row = idx[i]
            srow = sims[i]
            self_id = int(self_rows[i])
            if self_id < 0:
                out_idx[i] = row[:kk]
                out_sims[i] = srow[:kk]
                continue
            mask = row != self_id
            kept_idx = row[mask][:kk]
            kept_sim = srow[mask][:kk]
            if kept_idx.shape[0] < kk:
                # Edge case: the candidate set was too small even after the
                # +1 buffer. Pad with the most-similar non-self again.
                pad = kk - kept_idx.shape[0]
                kept_idx = np.concatenate([kept_idx, kept_idx[:1].repeat(pad)])
                kept_sim = np.concatenate([kept_sim, kept_sim[:1].repeat(pad)])
            out_idx[i] = kept_idx
            out_sims[i] = kept_sim
        return out_idx, out_sims


# ---------------------------------------------------------------------------
# Sparse pass-rate tables: per-(subject, item) mean label + observation mask
# ---------------------------------------------------------------------------


def build_passrate_table(
    train_df: pd.DataFrame,
    item_index_map: Mapping[str, int],
    subject_index_map: Mapping[str, int],
):
    """Build the sparse pass-rate and observation-mask matrices.

    Returns ``(passrate_csr, mask_csr)`` of shape ``[n_subjects, n_items]``.
    Entries store the mean label of ``(subject, item)`` pairs in training; the
    mask is 1 where there is at least one observation.

    Subject ids that lie outside ``subject_index_map``'s value range are
    skipped (they correspond to UNK at index 0; the caller is responsible
    for setting up UNK semantics).
    """
    from scipy import sparse  # type: ignore

    required = {"subject_key", "item_key", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(
            f"train_df missing required cols: {sorted(required - set(train_df.columns))}"
        )

    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    df = df[df["item_key"].isin(item_index_map)]
    df = df[df["subject_key"].isin(subject_index_map)]
    if df.empty:
        n_subjects = max(int(max(subject_index_map.values()) + 1), 1)
        n_items = max(int(max(item_index_map.values()) + 1), 1)
        empty = sparse.csr_matrix((n_subjects, n_items), dtype=np.float32)
        return empty, empty.copy()

    grouped = (
        df.groupby(["subject_key", "item_key"], sort=False)["label"]
        .mean()
        .reset_index()
    )
    rows = grouped["subject_key"].map(subject_index_map).to_numpy(dtype=np.int64)
    cols = grouped["item_key"].map(item_index_map).to_numpy(dtype=np.int64)
    vals = grouped["label"].astype(np.float32).to_numpy()

    n_subjects = int(max(int(max(subject_index_map.values())) + 1, 1))
    n_items = int(max(int(max(item_index_map.values())) + 1, 1))
    passrate = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_subjects, n_items), dtype=np.float32
    )
    mask = sparse.csr_matrix(
        (np.ones_like(vals, dtype=np.float32), (rows, cols)),
        shape=(n_subjects, n_items),
        dtype=np.float32,
    )
    return passrate, mask


# ---------------------------------------------------------------------------
# Conditional passrate context: per-trait passrate tables + per-item context
# arrays + per-subject trait-id maps. Used by the new (cells [15..22])
# conditional NN features. The same shape works at training time and at
# inference: the runtime ships these tables alongside the existing
# subject_passrate.npz so that both train and test see bit-identical
# feature vectors for the conditional cells.
# ---------------------------------------------------------------------------


# Trait id 0 is reserved for the MISSING / unknown sentinel. The metadata
# preprocessor already uses this convention (subject_meta_cat_ids[s, j]
# == 0 means "this subject has no value for the j-th categorical
# field"); we adopt it here so the runtime can map a subject straight
# to the same id space without re-processing.
MISSING_TRAIT_ID = 0


@dataclass
class ConditionalPassrateContext:
    """Training-time tables that drive the conditional NN features.

    Each subject-trait-conditional passrate is stored as a pair of CSR
    matrices keyed by ``[trait_id, item_row]``: one with the mean label
    value, one with the observation mask. ``trait_id == MISSING_TRAIT_ID``
    is the sentinel for "subject has no known value for this trait" and
    is excluded from the table so that lookups for redacted rows fall
    through cleanly to the fallback path in :func:`_aggregate_nn_features`.

    The per-item arrays carry context that is independent of the query
    subject: each train item's benchmark id, age (used for freshness),
    distinct-subject count (used for diversity), and cluster id (used
    for the cluster-conditional cell). ``item_global_passrate`` /
    ``item_global_passrate_mask`` are the per-item mean labels across
    all training subjects -- the runtime feeds them as the ``passrate``
    field for cell 19 (benchmark-conditional).

    Per-subject trait-id maps (``subject_to_*_id``) let the aggregator
    resolve a query subject to its trait id with a single fancy-index;
    the runtime ships them as ``int32`` arrays of length ``n_subjects``.

    Cluster-by-subject passrate is also a CSR matrix
    (``[n_clusters, n_subjects]``) holding the subject's mean label
    over training items in each cluster.

    All shapes are nailed down via :meth:`assert_shapes` at construction
    so a partially-built context fails loudly.
    """

    # --- Subject-trait-conditional passrate (4 traits). Trait id 0 is MISSING.
    subject_passrate_csr: object               # CSR [n_subjects + 1, n_items]
    subject_passrate_mask_csr: object
    family_passrate_csr: object                # CSR [n_families,  n_items]
    family_passrate_mask_csr: object
    macro_family_passrate_csr: object          # CSR [n_macro_families, n_items]
    macro_family_passrate_mask_csr: object
    organization_passrate_csr: object          # CSR [n_organizations, n_items]
    organization_passrate_mask_csr: object

    # --- Per-subject trait-id maps. -1 marks "subject id out of range".
    # Index 0 of each map is the UNK subject id (mirrors Indexer's UNK).
    subject_to_family_id: np.ndarray           # int32 [n_subjects]
    subject_to_macro_family_id: np.ndarray
    subject_to_organization_id: np.ndarray

    # --- Per-train-item context arrays.
    item_benchmark_id: np.ndarray              # int32 [n_items]; -1 = unknown
    item_benchmark_age: np.ndarray             # float32 [n_items]; NaN = unknown
    item_distinct_subj_count: np.ndarray       # int32 [n_items]
    item_global_passrate: np.ndarray           # float32 [n_items]
    item_global_passrate_mask: np.ndarray      # float32 [n_items] (1 if any obs)
    item_cluster_id: np.ndarray                # int32 [n_items]; -1 if unclustered

    # --- Per-(cluster, subject) cluster passrate.
    cluster_subject_passrate_csr: object       # CSR [n_clusters, n_subjects]
    cluster_subject_passrate_mask_csr: object

    n_subjects: int = 0
    n_items: int = 0
    n_families: int = 0
    n_macro_families: int = 0
    n_organizations: int = 0
    n_clusters: int = 0

    def assert_shapes(self) -> None:
        """Sanity-check shapes match the declared cardinalities."""
        assert int(self.subject_passrate_csr.shape[0]) >= int(self.n_subjects), (
            f"subject_passrate rows {self.subject_passrate_csr.shape[0]} "
            f"< n_subjects {self.n_subjects}"
        )
        assert int(self.family_passrate_csr.shape[0]) == int(self.n_families)
        assert int(self.macro_family_passrate_csr.shape[0]) == int(self.n_macro_families)
        assert int(self.organization_passrate_csr.shape[0]) == int(self.n_organizations)
        assert int(self.cluster_subject_passrate_csr.shape[0]) == int(self.n_clusters)
        assert self.subject_to_family_id.shape == (self.n_subjects,)
        assert self.subject_to_macro_family_id.shape == (self.n_subjects,)
        assert self.subject_to_organization_id.shape == (self.n_subjects,)
        for arr_name in (
            "item_benchmark_id",
            "item_benchmark_age",
            "item_distinct_subj_count",
            "item_global_passrate",
            "item_global_passrate_mask",
            "item_cluster_id",
        ):
            arr = getattr(self, arr_name)
            assert arr.shape == (self.n_items,), (
                f"{arr_name} shape {arr.shape} != ({self.n_items},)"
            )

    # ---------------------------------------------------------------- save / load

    def save(self, out_dir: Path) -> Path:
        """Persist to ``out_dir`` as a bag of .npy / .npz files plus meta.json.

        Layout (every file is plain numpy / scipy.sparse.save_npz, so the
        runtime can load with no torch / pandas dependency):

          subject_passrate.npz / subject_passrate_mask.npz
          family_passrate.npz / family_passrate_mask.npz
          macro_family_passrate.npz / macro_family_passrate_mask.npz
          organization_passrate.npz / organization_passrate_mask.npz
          subject_to_family_id.npy
          subject_to_macro_family_id.npy
          subject_to_organization_id.npy
          item_benchmark_id.npy
          item_benchmark_age.npy
          item_distinct_subj_count.npy
          item_global_passrate.npy
          item_global_passrate_mask.npy
          item_cluster_id.npy
          cluster_subject_passrate.npz / cluster_subject_passrate_mask.npz
          conditional_meta.json

        ``conditional_meta.json`` records the format version + cardinalities
        + the schema-dim tag so the runtime can refuse to load a stale
        bundle that was built against a different feature schema.
        """
        from scipy import sparse  # local import: no global scipy dep

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(out_dir / "subject_passrate.npz", self.subject_passrate_csr)
        sparse.save_npz(out_dir / "subject_passrate_mask.npz", self.subject_passrate_mask_csr)
        sparse.save_npz(out_dir / "family_passrate.npz", self.family_passrate_csr)
        sparse.save_npz(out_dir / "family_passrate_mask.npz", self.family_passrate_mask_csr)
        sparse.save_npz(out_dir / "macro_family_passrate.npz", self.macro_family_passrate_csr)
        sparse.save_npz(out_dir / "macro_family_passrate_mask.npz", self.macro_family_passrate_mask_csr)
        sparse.save_npz(out_dir / "organization_passrate.npz", self.organization_passrate_csr)
        sparse.save_npz(out_dir / "organization_passrate_mask.npz", self.organization_passrate_mask_csr)
        np.save(out_dir / "subject_to_family_id.npy", self.subject_to_family_id)
        np.save(out_dir / "subject_to_macro_family_id.npy", self.subject_to_macro_family_id)
        np.save(out_dir / "subject_to_organization_id.npy", self.subject_to_organization_id)
        np.save(out_dir / "item_benchmark_id.npy", self.item_benchmark_id)
        np.save(out_dir / "item_benchmark_age.npy", self.item_benchmark_age)
        np.save(out_dir / "item_distinct_subj_count.npy", self.item_distinct_subj_count)
        np.save(out_dir / "item_global_passrate.npy", self.item_global_passrate)
        np.save(out_dir / "item_global_passrate_mask.npy", self.item_global_passrate_mask)
        np.save(out_dir / "item_cluster_id.npy", self.item_cluster_id)
        sparse.save_npz(out_dir / "cluster_subject_passrate.npz", self.cluster_subject_passrate_csr)
        sparse.save_npz(out_dir / "cluster_subject_passrate_mask.npz", self.cluster_subject_passrate_mask_csr)
        meta = {
            "format_version": 1,
            "n_subjects": int(self.n_subjects),
            "n_items": int(self.n_items),
            "n_families": int(self.n_families),
            "n_macro_families": int(self.n_macro_families),
            "n_organizations": int(self.n_organizations),
            "n_clusters": int(self.n_clusters),
            "feature_dim": int(NN_FEATURE_DIM),
        }
        (out_dir / "conditional_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out_dir

    @classmethod
    def load(cls, in_dir: Path) -> "ConditionalPassrateContext":
        """Mirror of :meth:`save` -- reads the bag back into memory."""
        from scipy import sparse

        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "conditional_meta.json").read_text(encoding="utf-8"))
        return cls(
            subject_passrate_csr=sparse.load_npz(in_dir / "subject_passrate.npz"),
            subject_passrate_mask_csr=sparse.load_npz(in_dir / "subject_passrate_mask.npz"),
            family_passrate_csr=sparse.load_npz(in_dir / "family_passrate.npz"),
            family_passrate_mask_csr=sparse.load_npz(in_dir / "family_passrate_mask.npz"),
            macro_family_passrate_csr=sparse.load_npz(in_dir / "macro_family_passrate.npz"),
            macro_family_passrate_mask_csr=sparse.load_npz(in_dir / "macro_family_passrate_mask.npz"),
            organization_passrate_csr=sparse.load_npz(in_dir / "organization_passrate.npz"),
            organization_passrate_mask_csr=sparse.load_npz(in_dir / "organization_passrate_mask.npz"),
            subject_to_family_id=np.load(in_dir / "subject_to_family_id.npy"),
            subject_to_macro_family_id=np.load(in_dir / "subject_to_macro_family_id.npy"),
            subject_to_organization_id=np.load(in_dir / "subject_to_organization_id.npy"),
            item_benchmark_id=np.load(in_dir / "item_benchmark_id.npy"),
            item_benchmark_age=np.load(in_dir / "item_benchmark_age.npy"),
            item_distinct_subj_count=np.load(in_dir / "item_distinct_subj_count.npy"),
            item_global_passrate=np.load(in_dir / "item_global_passrate.npy"),
            item_global_passrate_mask=np.load(in_dir / "item_global_passrate_mask.npy"),
            item_cluster_id=np.load(in_dir / "item_cluster_id.npy"),
            cluster_subject_passrate_csr=sparse.load_npz(in_dir / "cluster_subject_passrate.npz"),
            cluster_subject_passrate_mask_csr=sparse.load_npz(in_dir / "cluster_subject_passrate_mask.npz"),
            n_subjects=int(meta.get("n_subjects", 0)),
            n_items=int(meta.get("n_items", 0)),
            n_families=int(meta.get("n_families", 0)),
            n_macro_families=int(meta.get("n_macro_families", 0)),
            n_organizations=int(meta.get("n_organizations", 0)),
            n_clusters=int(meta.get("n_clusters", 0)),
        )


def build_conditional_passrate_context(
    *,
    train_df: pd.DataFrame,
    item_index_map: Mapping[str, int],
    subject_index_map: Mapping[str, int],
    subject_to_family_id: np.ndarray,
    subject_to_macro_family_id: np.ndarray,
    subject_to_organization_id: np.ndarray,
    item_benchmark_id: np.ndarray,
    item_benchmark_age: np.ndarray,
    item_cluster_id: np.ndarray,
    n_families: int,
    n_macro_families: int,
    n_organizations: int,
    n_clusters: int,
) -> ConditionalPassrateContext:
    """Build a :class:`ConditionalPassrateContext` from the training rows.

    The 4 trait-conditional passrate matrices share the existing
    :func:`build_passrate_table` recipe (sparse mean label per
    ``(trait_id, item_row)`` pair, plus its observation mask). The per-
    item context arrays come straight from the caller-supplied inputs;
    we only compute the global passrate / distinct-subject count from
    ``train_df`` itself.

    Args:
        train_df: must have ``subject_key``, ``item_key``, ``label``
            columns. Rows with item / subject keys outside the index
            maps are filtered.
        item_index_map / subject_index_map: identical to the maps used
            by :func:`build_passrate_table`.
        subject_to_*_id: per-subject 1-D int arrays, length == max
            subject id + 1. Index 0 is the UNK row (must map to
            ``MISSING_TRAIT_ID == 0``); index ``s`` for ``s >= 1``
            yields that subject's trait id (0 if missing).
        item_benchmark_id: per-item int array, length == max item row + 1.
            -1 marks unknown / unclustered.
        item_benchmark_age: per-item float array (NaN = unknown).
        item_cluster_id: per-item int array (-1 = unclustered).
        n_*: cardinalities for each trait / cluster.

    Returns:
        Fully-populated :class:`ConditionalPassrateContext`. The caller
        is expected to call :meth:`assert_shapes` if they want a sanity
        check, then :meth:`save` to ship into the runtime cache.
    """
    from scipy import sparse  # local import: no global scipy dep

    required = {"subject_key", "item_key", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(
            f"train_df missing required cols: {sorted(required - set(train_df.columns))}"
        )

    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    df = df[df["item_key"].isin(item_index_map)]
    df = df[df["subject_key"].isin(subject_index_map)]

    n_subjects = max(int(max(subject_index_map.values()) + 1), 1)
    n_items = max(int(max(item_index_map.values()) + 1), 1)

    # Resolve subject ids + per-subject trait ids upfront so the rest is
    # vectorized.
    subj_ids = df["subject_key"].map(subject_index_map).to_numpy(dtype=np.int64)
    item_ids = df["item_key"].map(item_index_map).to_numpy(dtype=np.int64)
    labels = df["label"].astype(np.float32).to_numpy()

    fam_ids = subject_to_family_id[subj_ids]
    macro_ids = subject_to_macro_family_id[subj_ids]
    org_ids = subject_to_organization_id[subj_ids]

    # Auto-grow trait cardinalities so an off-by-one in the caller's
    # vocabulary count never blows up the COO assembly. The shipped
    # context records the *effective* cardinality so the runtime sees a
    # consistent shape.
    def _effective_card(ids: np.ndarray, declared: int, *, name: str) -> int:
        actual = int(ids.max()) + 1 if ids.size else 0
        eff = max(int(declared), actual)
        if actual > int(declared):
            LOG.warning(
                "build_conditional_passrate_context: %s cardinality "
                "auto-grown from %d -> %d (max id observed: %d)",
                name,
                int(declared),
                eff,
                actual - 1,
            )
        return max(eff, 1)

    n_families_eff = _effective_card(fam_ids, n_families, name="n_families")
    n_macro_families_eff = _effective_card(
        macro_ids, n_macro_families, name="n_macro_families"
    )
    n_organizations_eff = _effective_card(
        org_ids, n_organizations, name="n_organizations"
    )

    def _build_csr(
        rows: np.ndarray,
        n_rows: int,
        *,
        skip_missing_row: bool,
    ) -> tuple[object, object]:
        """Mean label per (rows, item_ids) pair, plus its observation mask.

        ``skip_missing_row`` controls whether row id ``MISSING_TRAIT_ID``
        (== 0) is excluded. We exclude it for trait tables (family /
        macro_family / organization) because trait id 0 is the
        ``__MISSING__`` sentinel: any query that resolves to that id
        falls through to the redaction fallback and never reads from the
        CSR. We do NOT exclude row 0 for the subject table -- subject
        id 0 is a real subject in the indexer.
        """
        ix = pd.DataFrame({"row": rows, "col": item_ids, "y": labels})
        if skip_missing_row:
            ix = ix[ix["row"] != MISSING_TRAIT_ID]
        if ix.empty:
            empty = sparse.csr_matrix((n_rows, n_items), dtype=np.float32)
            return empty, empty.copy()
        agg = ix.groupby(["row", "col"], sort=False)["y"].mean().reset_index()
        rs = agg["row"].to_numpy(dtype=np.int64)
        cs = agg["col"].to_numpy(dtype=np.int64)
        vs = agg["y"].astype(np.float32).to_numpy()
        pr = sparse.csr_matrix(
            (vs, (rs, cs)), shape=(n_rows, n_items), dtype=np.float32
        )
        mk = sparse.csr_matrix(
            (np.ones_like(vs, dtype=np.float32), (rs, cs)),
            shape=(n_rows, n_items),
            dtype=np.float32,
        )
        return pr, mk

    subj_pr, subj_mk = _build_csr(subj_ids, max(n_subjects, 1), skip_missing_row=False)
    fam_pr, fam_mk = _build_csr(fam_ids, n_families_eff, skip_missing_row=True)
    macro_pr, macro_mk = _build_csr(macro_ids, n_macro_families_eff, skip_missing_row=True)
    org_pr, org_mk = _build_csr(org_ids, n_organizations_eff, skip_missing_row=True)

    # Per-item global passrate + observation mask. Mean label across
    # every training (subject, item) row regardless of subject trait.
    if df.empty:
        item_pr_full = np.zeros(n_items, dtype=np.float32)
        item_pr_mask = np.zeros(n_items, dtype=np.float32)
        item_distinct = np.zeros(n_items, dtype=np.int32)
    else:
        per_item_mean = (
            df.groupby("item_key", sort=False)["label"].mean()
        )
        per_item_distinct = (
            df.groupby("item_key", sort=False)["subject_key"].nunique()
        )
        item_pr_full = np.zeros(n_items, dtype=np.float32)
        item_pr_mask = np.zeros(n_items, dtype=np.float32)
        item_distinct = np.zeros(n_items, dtype=np.int32)
        for k, v in per_item_mean.items():
            ridx = item_index_map.get(str(k))
            if ridx is None:
                continue
            item_pr_full[int(ridx)] = float(v)
            item_pr_mask[int(ridx)] = 1.0
        for k, v in per_item_distinct.items():
            ridx = item_index_map.get(str(k))
            if ridx is None:
                continue
            item_distinct[int(ridx)] = int(v)

    # Cluster-by-subject passrate. Aggregate (cluster_id, subject_id)
    # mean labels across the train rows. We auto-grow ``n_clusters`` if
    # the actual cluster IDs ever exceed the declared cardinality (which
    # can happen when ``CFG["clustering"]["k"]`` and the upstream FAISS
    # k-means K disagree).
    n_clusters_eff = max(int(n_clusters), 1)
    if df.empty or n_clusters <= 0:
        cluster_pr = sparse.csr_matrix(
            (n_clusters_eff, n_subjects), dtype=np.float32
        )
        cluster_mk = cluster_pr.copy()
    else:
        cl_for_row = item_cluster_id[item_ids]
        keep = cl_for_row >= 0
        if not keep.any():
            cluster_pr = sparse.csr_matrix(
                (n_clusters_eff, n_subjects), dtype=np.float32
            )
            cluster_mk = cluster_pr.copy()
        else:
            cl_kept = cl_for_row[keep]
            actual_cl = int(cl_kept.max()) + 1
            if actual_cl > n_clusters_eff:
                LOG.warning(
                    "build_conditional_passrate_context: n_clusters auto-grown "
                    "from %d -> %d (max cluster id observed: %d)",
                    n_clusters_eff,
                    actual_cl,
                    actual_cl - 1,
                )
                n_clusters_eff = actual_cl
            cdf = pd.DataFrame(
                {
                    "row": cl_kept,
                    "col": subj_ids[keep],
                    "y": labels[keep],
                }
            )
            agg = cdf.groupby(["row", "col"], sort=False)["y"].mean().reset_index()
            rs = agg["row"].to_numpy(dtype=np.int64)
            cs = agg["col"].to_numpy(dtype=np.int64)
            vs = agg["y"].astype(np.float32).to_numpy()
            cluster_pr = sparse.csr_matrix(
                (vs, (rs, cs)),
                shape=(n_clusters_eff, n_subjects),
                dtype=np.float32,
            )
            cluster_mk = sparse.csr_matrix(
                (np.ones_like(vs, dtype=np.float32), (rs, cs)),
                shape=(n_clusters_eff, n_subjects),
                dtype=np.float32,
            )

    # Pad/normalize per-subject trait id arrays to length == n_subjects.
    def _pad(arr: np.ndarray, n: int) -> np.ndarray:
        a = np.asarray(arr, dtype=np.int32).reshape(-1)
        if a.shape[0] >= n:
            return a[:n].astype(np.int32, copy=False)
        return np.concatenate([a, np.zeros(n - a.shape[0], dtype=np.int32)]).astype(np.int32)

    return ConditionalPassrateContext(
        subject_passrate_csr=subj_pr,
        subject_passrate_mask_csr=subj_mk,
        family_passrate_csr=fam_pr,
        family_passrate_mask_csr=fam_mk,
        macro_family_passrate_csr=macro_pr,
        macro_family_passrate_mask_csr=macro_mk,
        organization_passrate_csr=org_pr,
        organization_passrate_mask_csr=org_mk,
        subject_to_family_id=_pad(subject_to_family_id, n_subjects),
        subject_to_macro_family_id=_pad(subject_to_macro_family_id, n_subjects),
        subject_to_organization_id=_pad(subject_to_organization_id, n_subjects),
        item_benchmark_id=np.asarray(item_benchmark_id, dtype=np.int32).reshape(-1)[:n_items],
        item_benchmark_age=np.asarray(item_benchmark_age, dtype=np.float32).reshape(-1)[:n_items],
        item_distinct_subj_count=item_distinct,
        item_global_passrate=item_pr_full,
        item_global_passrate_mask=item_pr_mask,
        item_cluster_id=np.asarray(item_cluster_id, dtype=np.int32).reshape(-1)[:n_items],
        cluster_subject_passrate_csr=cluster_pr,
        cluster_subject_passrate_mask_csr=cluster_mk,
        n_subjects=int(n_subjects),
        n_items=int(n_items),
        n_families=int(n_families_eff),
        n_macro_families=int(n_macro_families_eff),
        n_organizations=int(n_organizations_eff),
        n_clusters=int(n_clusters_eff),
    )


# ---------------------------------------------------------------------------
# Compute NN features for a batch of (subject, item) queries
# ---------------------------------------------------------------------------


def _lookup_csr_pairs(
    row_ids: np.ndarray,             # [B]
    col_ids: np.ndarray,             # [B, K]
    pr_csr,
    mk_csr,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic per-(row, col) lookup against two CSR matrices.

    Same algorithm as :func:`_lookup_neighbor_passrates` but parametrized
    on the *row* axis (not just subject id) so we can reuse it for the
    family / macro_family / organization tables. Returns the value
    matrix and the observation mask, both shape ``[B, K]``.
    """
    B, K = col_ids.shape
    out_pr = np.zeros((B, K), dtype=np.float32)
    out_mask = np.zeros((B, K), dtype=np.float32)
    n_rows = pr_csr.shape[0]
    pr_indptr = pr_csr.indptr
    pr_indices = pr_csr.indices
    pr_data = pr_csr.data
    mk_indptr = mk_csr.indptr
    mk_indices = mk_csr.indices
    for s in np.unique(row_ids):
        if s < 0 or s >= n_rows:
            continue
        rows_for = np.where(row_ids == s)[0]
        cols_for = col_ids[rows_for]
        start = pr_indptr[s]
        end = pr_indptr[s + 1]
        row_cols = pr_indices[start:end]
        row_vals = pr_data[start:end]
        if row_cols.size:
            order = np.argsort(row_cols)
            sorted_cols = row_cols[order]
            sorted_vals = row_vals[order]
            pos = np.searchsorted(sorted_cols, cols_for)
            pos_clipped = np.clip(pos, 0, sorted_cols.size - 1)
            hit = (pos < sorted_cols.size) & (sorted_cols[pos_clipped] == cols_for)
            out_pr[rows_for] = np.where(hit, sorted_vals[pos_clipped], 0.0)
        mstart = mk_indptr[s]
        mend = mk_indptr[s + 1]
        m_cols = mk_indices[mstart:mend]
        if m_cols.size:
            order = np.argsort(m_cols)
            sorted_m = m_cols[order]
            pos = np.searchsorted(sorted_m, cols_for)
            pos_clipped = np.clip(pos, 0, sorted_m.size - 1)
            hit = (pos < sorted_m.size) & (sorted_m[pos_clipped] == cols_for)
            out_mask[rows_for] = hit.astype(np.float32)
    return out_pr, out_mask


def _resolve_conditional_inputs(
    context: ConditionalPassrateContext | None,
    subject_ids: np.ndarray,            # [B]
    neighbor_idx: np.ndarray,           # [B, K]
    *,
    query_benchmark_ids: np.ndarray | None,    # [B] int32, -1 = unknown / redacted
    query_benchmark_age: np.ndarray | None,    # [B] float32, NaN = unknown / redacted
    query_cluster_ids: np.ndarray | None,      # [B] int32, -1 = unknown / redacted
    subject_meta_redacted: np.ndarray | None,  # [B] bool/int, 1 = redact ALL subject-trait cells
) -> dict[str, np.ndarray]:
    """Turn a :class:`ConditionalPassrateContext` + per-query inputs into
    the dict that :func:`_aggregate_nn_features` expects under
    ``cond_inputs``.

    No-context path: returns an empty dict (every conditional cell will
    fall back via the aggregator's own redaction logic).

    The *_redacted flags collapse multiple sources of "we cannot use
    this trait for this row" into a single per-cell mask:

      - ``subject_meta_redacted=True`` zeroes out cells 15-18 (subject /
        family / macro_family / organization conditional passrates).
      - ``query_benchmark_ids[b] < 0`` zeroes cell 19 (benchmark match).
      - ``np.isnan(query_benchmark_age[b])`` zeroes cell 20 (freshness).
      - ``query_cluster_ids[b] < 0`` zeroes cell 22 (cluster passrate).

    Cell 21 (n_distinct_subjects) never redacts -- it is a per-train-item
    statistic that does not depend on the query subject's metadata.
    """
    if context is None:
        return {}
    B, K = neighbor_idx.shape
    sids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    nidx = np.asarray(neighbor_idx, dtype=np.int64)

    subj_redact = (
        np.asarray(subject_meta_redacted, dtype=np.int32).reshape(-1).astype(np.float32)
        if subject_meta_redacted is not None
        else np.zeros(B, dtype=np.float32)
    )
    n_subj = int(context.n_subjects)
    if n_subj <= 0:
        # Degenerate context; everything below this point would have
        # to special-case empty arrays. Returning {} forces every
        # conditional cell to fall back, which is what we want.
        return {}
    safe_sids = np.clip(sids, 0, n_subj - 1)
    fam_ids = np.where(
        (sids >= 0) & (sids < n_subj),
        np.take(context.subject_to_family_id, safe_sids),
        MISSING_TRAIT_ID,
    ).astype(np.int64)
    macro_ids = np.where(
        (sids >= 0) & (sids < n_subj),
        np.take(context.subject_to_macro_family_id, safe_sids),
        MISSING_TRAIT_ID,
    ).astype(np.int64)
    org_ids = np.where(
        (sids >= 0) & (sids < n_subj),
        np.take(context.subject_to_organization_id, safe_sids),
        MISSING_TRAIT_ID,
    ).astype(np.int64)

    # Per-row redact masks for each subject trait: redact when the
    # subject id is OOR, OR when the trait id is the MISSING sentinel,
    # OR when the caller passed subject_meta_redacted.
    fam_redact = (subj_redact > 0) | (fam_ids == MISSING_TRAIT_ID)
    macro_redact = (subj_redact > 0) | (macro_ids == MISSING_TRAIT_ID)
    org_redact = (subj_redact > 0) | (org_ids == MISSING_TRAIT_ID)
    subj_redact_full = (subj_redact > 0) | (sids < 0) | (sids >= n_subj)

    # Subject-conditional uses the existing subject_passrate matrix in
    # the context (functionally identical to the legacy passrate_csr
    # we already pass through the aggregator's first 8 features).
    subj_pr_kk, subj_mk_kk = _lookup_csr_pairs(
        sids,
        nidx,
        context.subject_passrate_csr,
        context.subject_passrate_mask_csr,
    )
    fam_pr_kk, fam_mk_kk = _lookup_csr_pairs(
        fam_ids,
        nidx,
        context.family_passrate_csr,
        context.family_passrate_mask_csr,
    )
    macro_pr_kk, macro_mk_kk = _lookup_csr_pairs(
        macro_ids,
        nidx,
        context.macro_family_passrate_csr,
        context.macro_family_passrate_mask_csr,
    )
    org_pr_kk, org_mk_kk = _lookup_csr_pairs(
        org_ids,
        nidx,
        context.organization_passrate_csr,
        context.organization_passrate_mask_csr,
    )

    # Benchmark match: per-(B, K) global neighbor passrate, masked off
    # whenever the neighbor's benchmark != the query benchmark.
    n_items = int(context.n_items)
    nbench = context.item_benchmark_id[
        np.clip(nidx, 0, max(n_items - 1, 0))
    ]  # [B, K] int32
    if query_benchmark_ids is None:
        bench_q = np.full(B, -1, dtype=np.int32)
    else:
        bench_q = np.asarray(query_benchmark_ids, dtype=np.int32).reshape(-1)
    bench_q = bench_q[:, None]  # [B, 1]
    bench_match = (nbench == bench_q) & (bench_q >= 0)
    bench_pr = context.item_global_passrate[
        np.clip(nidx, 0, max(n_items - 1, 0))
    ]  # [B, K]
    bench_mk_global = context.item_global_passrate_mask[
        np.clip(nidx, 0, max(n_items - 1, 0))
    ]  # [B, K]
    bench_match_pr = np.where(bench_match, bench_pr, 0.0).astype(np.float32)
    bench_match_mk = (bench_match.astype(np.float32) * bench_mk_global).astype(np.float32)
    bench_redact = bench_q.reshape(-1) < 0  # [B]

    # Freshness: query age - mean(neighbor age) over neighbors with
    # known ages.
    nages = context.item_benchmark_age[np.clip(nidx, 0, max(n_items - 1, 0))]
    nage_mask = (~np.isnan(nages)).astype(np.float32)
    nage_safe = np.where(np.isnan(nages), 0.0, nages).astype(np.float32)
    n_known = nage_mask.sum(axis=1)
    mean_neighbor_age = np.where(
        n_known > 0,
        (nage_safe * nage_mask).sum(axis=1) / np.maximum(n_known, 1.0),
        0.0,
    ).astype(np.float32)
    if query_benchmark_age is None:
        q_age = np.full(B, np.nan, dtype=np.float32)
    else:
        q_age = np.asarray(query_benchmark_age, dtype=np.float32).reshape(-1)
    fresh_redact = np.isnan(q_age) | (n_known == 0)
    fresh_diff = np.where(
        fresh_redact,
        0.0,
        q_age - mean_neighbor_age,
    ).astype(np.float32)

    # Distinct subjects per neighbor: per-(B, K) lookup via the per-
    # train-item count array. No redaction.
    distinct_per_neighbor = context.item_distinct_subj_count[
        np.clip(nidx, 0, max(n_items - 1, 0))
    ].astype(np.float32)

    # Cluster-passrate-subject: per-row scalar lookup.
    if query_cluster_ids is None:
        q_cluster = np.full(B, -1, dtype=np.int64)
    else:
        q_cluster = np.asarray(query_cluster_ids, dtype=np.int64).reshape(-1)
    cluster_redact = (q_cluster < 0) | (sids < 0) | (sids >= n_subj)
    # Vectorized lookup: for each unique cluster id, fetch the row
    # from cluster_subject_passrate_csr and gather subject_ids.
    cps = np.zeros(B, dtype=np.float32)
    cps_mk = np.zeros(B, dtype=np.float32)
    n_clusters = int(context.n_clusters)
    cps_csr = context.cluster_subject_passrate_csr
    cps_mask_csr = context.cluster_subject_passrate_mask_csr
    cps_indptr = cps_csr.indptr
    cps_indices = cps_csr.indices
    cps_data = cps_csr.data
    cps_mk_indptr = cps_mask_csr.indptr
    cps_mk_indices = cps_mask_csr.indices
    for c in np.unique(q_cluster):
        if c < 0 or c >= n_clusters:
            continue
        rows_for = np.where(q_cluster == c)[0]
        sub_for = sids[rows_for]
        s0 = cps_indptr[c]
        s1 = cps_indptr[c + 1]
        row_cols = cps_indices[s0:s1]
        row_vals = cps_data[s0:s1]
        if row_cols.size:
            order = np.argsort(row_cols)
            sc = row_cols[order]
            sv = row_vals[order]
            pos = np.searchsorted(sc, sub_for)
            pc = np.clip(pos, 0, sc.size - 1)
            hit = (pos < sc.size) & (sc[pc] == sub_for)
            cps[rows_for] = np.where(hit, sv[pc], 0.0)
        ms0 = cps_mk_indptr[c]
        ms1 = cps_mk_indptr[c + 1]
        m_cols = cps_mk_indices[ms0:ms1]
        if m_cols.size:
            order = np.argsort(m_cols)
            sm = m_cols[order]
            pos = np.searchsorted(sm, sub_for)
            pc = np.clip(pos, 0, sm.size - 1)
            hit = (pos < sm.size) & (sm[pc] == sub_for)
            cps_mk[rows_for] = hit.astype(np.float32)

    # If the (cluster, subject) pair has no observations even though
    # both ids are valid, we still hand the value through; the
    # downstream model can always learn to discount it via the
    # ``coverage`` cell already in nn_feats. Redaction only kicks in
    # for "we don't know cluster/subject id at all".
    return {
        "subject_passrates": subj_pr_kk,
        "subject_masks": subj_mk_kk,
        "subject_redact": subj_redact_full.astype(np.float32),
        "family_passrates": fam_pr_kk,
        "family_masks": fam_mk_kk,
        "family_redact": fam_redact.astype(np.float32),
        "macro_family_passrates": macro_pr_kk,
        "macro_family_masks": macro_mk_kk,
        "macro_family_redact": macro_redact.astype(np.float32),
        "organization_passrates": org_pr_kk,
        "organization_masks": org_mk_kk,
        "organization_redact": org_redact.astype(np.float32),
        "bench_match_passrates": bench_match_pr,
        "bench_match_masks": bench_match_mk,
        "bench_match_redact": bench_redact.astype(np.float32),
        "neighbor_freshness_diff": fresh_diff,
        "freshness_redact": fresh_redact.astype(np.float32),
        "distinct_subj_per_neighbor": distinct_per_neighbor,
        "cluster_passrate_subject_query": cps,
        "cluster_redact": cluster_redact.astype(np.float32),
    }


def _lookup_neighbor_passrates(
    subject_ids: np.ndarray,         # [B]
    neighbor_indices: np.ndarray,    # [B, K]
    passrate_csr,
    passrate_mask_csr,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse fancy-indexing of (subject, item) -> (passrate, mask).

    Implemented row-by-row on the sparse matrices to keep the working set
    in CSR form (~B small slices); avoids materializing the full
    [n_subjects, n_items] dense matrix.
    """
    B, K = neighbor_indices.shape
    out_pr = np.zeros((B, K), dtype=np.float32)
    out_mask = np.zeros((B, K), dtype=np.float32)

    n_rows = passrate_csr.shape[0]
    pr_indptr = passrate_csr.indptr
    pr_indices = passrate_csr.indices
    pr_data = passrate_csr.data
    mk_indptr = passrate_mask_csr.indptr
    mk_indices = passrate_mask_csr.indices

    # Group by subject id so we touch each CSR row at most once per group.
    for s in np.unique(subject_ids):
        if s < 0 or s >= n_rows:
            continue
        rows_for_subject = np.where(subject_ids == s)[0]
        # Items asked about for this subject (flatten over the group).
        cols_for_subject = neighbor_indices[rows_for_subject]    # [b_s, K]
        # Dense view of the CSR row for this subject; cheap on a single row
        # because csr_matrix exposes the row's indices / data directly.
        start = pr_indptr[s]
        end = pr_indptr[s + 1]
        row_cols = pr_indices[start:end]
        row_vals = pr_data[start:end]
        if row_cols.size:
            # Build a sorted lookup once per subject.
            order = np.argsort(row_cols)
            sorted_cols = row_cols[order]
            sorted_vals = row_vals[order]
            pos = np.searchsorted(sorted_cols, cols_for_subject)
            pos_clipped = np.clip(pos, 0, sorted_cols.size - 1)
            hit = (pos < sorted_cols.size) & (sorted_cols[pos_clipped] == cols_for_subject)
            out_pr[rows_for_subject] = np.where(hit, sorted_vals[pos_clipped], 0.0)
        # mask
        mstart = mk_indptr[s]
        mend = mk_indptr[s + 1]
        m_cols = mk_indices[mstart:mend]
        if m_cols.size:
            order = np.argsort(m_cols)
            sorted_m = m_cols[order]
            pos = np.searchsorted(sorted_m, cols_for_subject)
            pos_clipped = np.clip(pos, 0, sorted_m.size - 1)
            hit = (pos < sorted_m.size) & (sorted_m[pos_clipped] == cols_for_subject)
            out_mask[rows_for_subject] = hit.astype(np.float32)
    return out_pr, out_mask


def compute_nn_features(
    query_embeds: np.ndarray,
    query_item_keys: list[str] | None,
    subject_ids: np.ndarray,
    nn_index: TrainingNNIndex,
    passrate_csr,
    passrate_mask_csr,
    cfg: NNFeaturesConfig,
    *,
    exclude_self: bool | None = None,
    conditional_context: ConditionalPassrateContext | None = None,
    query_benchmark_ids: np.ndarray | None = None,
    query_benchmark_age: np.ndarray | None = None,
    query_cluster_ids: np.ndarray | None = None,
    subject_meta_redacted: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``[B, NN_FEATURE_DIM]`` float32 NN feature matrix.

    Steps:
    1. Get top-k neighbors via ``nn_index.nearest``.
    2. Look up per-(subject, neighbor) pass-rate + mask from the sparse
       matrices (vectorized via fancy indexing).
    3. (Optional) Resolve the conditional inputs for cells [15..22]
       from ``conditional_context``.
    4. Aggregate into the locked-23-scalar feature vector via
       ``_aggregate_nn_features``.

    Conditional inputs ``conditional_context`` + ``query_*`` are optional;
    when ``None``, cells 15..22 fall back to ``cfg.fallback_value``.
    """
    if exclude_self is None:
        exclude_self = bool(cfg.exclude_self_in_training)
    k = int(cfg.k)
    neighbor_idx, sims = nn_index.nearest(
        query_embeds,
        k=k,
        exclude_self=exclude_self,
        query_keys=query_item_keys,
    )
    subject_ids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    if subject_ids.shape[0] != neighbor_idx.shape[0]:
        raise ValueError(
            f"subject_ids length {subject_ids.shape[0]} != queries {neighbor_idx.shape[0]}"
        )
    passrates, masks = _lookup_neighbor_passrates(
        subject_ids, neighbor_idx, passrate_csr, passrate_mask_csr
    )
    cond_inputs = _resolve_conditional_inputs(
        conditional_context,
        subject_ids,
        neighbor_idx,
        query_benchmark_ids=query_benchmark_ids,
        query_benchmark_age=query_benchmark_age,
        query_cluster_ids=query_cluster_ids,
        subject_meta_redacted=subject_meta_redacted,
    )
    return _aggregate_nn_features(
        passrates,
        masks,
        sims,
        fallback_value=float(cfg.fallback_value),
        top1_missing_sentinel=float(cfg.top1_missing_sentinel),
        cond_inputs=cond_inputs,
    )


def compute_nn_features_streaming(
    *,
    query_item_keys: list[str],
    item_emb_lookup: Mapping[str, np.ndarray],
    subject_ids: np.ndarray,
    nn_index: TrainingNNIndex,
    passrate_csr,
    passrate_mask_csr,
    cfg: NNFeaturesConfig,
    exclude_self: bool | None = None,
    query_chunk_size: int = 4096,
    conditional_context: ConditionalPassrateContext | None = None,
    query_benchmark_ids: np.ndarray | None = None,
    query_benchmark_age: np.ndarray | None = None,
    query_cluster_ids: np.ndarray | None = None,
    subject_meta_redacted: np.ndarray | None = None,
) -> np.ndarray:
    """Memory-bounded equivalent of :func:`compute_nn_features`.

    Produces a bit-identical ``[B, NN_FEATURE_DIM]`` matrix to
    ``compute_nn_features`` but never materializes the full ``[B, D]``
    query embedding matrix. Instead:

    1. Dedupes by ``query_item_keys`` (neighbor structure is a function
       of the query item embedding alone, not the subject id).
    2. Stacks unique embeddings *one chunk at a time* from
       ``item_emb_lookup`` and feeds the chunk to ``nn_index.nearest``.
       Peak working set: ``query_chunk_size * D * 4`` bytes plus the
       per-unique cached neighbor arrays (``Nu * k * 12`` bytes total).
    3. Expands neighbor indices / similarities back to per-row before
       the per-(subject, neighbor) passrate lookup.

    This is what enables Qwen3-Embedding-8B (D=4096) NN feature
    computation on a 12 GB Colab without OOM: the original
    ``compute_nn_features`` path peaks at ~3x the
    ``[B, D]`` matrix (one for the user-supplied stack, one for the
    cosine-normalized copy, one transient ``np.ascontiguousarray``
    copy), which crosses ~14 GB for B=300k, D=4096.

    Args:
        query_item_keys: per-row item keys (length B); duplicates expected.
        item_emb_lookup: maps item_key -> 1D float array of length D. Only
            the keys that actually appear in ``query_item_keys`` are
            looked up.
        subject_ids: per-row subject indices (length B).
        nn_index: pre-built training nearest-neighbor index.
        passrate_csr / passrate_mask_csr: sparse [n_subjects, n_items]
            matrices over the training index.
        cfg: same ``NNFeaturesConfig`` used to build ``nn_index``.
        exclude_self: forwarded to ``nn_index.nearest``. ``None`` means
            ``cfg.exclude_self_in_training``.
        query_chunk_size: number of unique queries per ``nn_index.nearest``
            call. Lowered values reduce peak RAM at modest runtime cost.

    Returns:
        ``np.ndarray`` of shape ``[B, NN_FEATURE_DIM]`` (float32).
    """
    if exclude_self is None:
        exclude_self = bool(cfg.exclude_self_in_training)
    k = int(cfg.k)
    chunk = max(1, int(query_chunk_size))

    keys_arr = np.asarray([str(x) for x in query_item_keys])
    sids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    if sids.shape[0] != keys_arr.shape[0]:
        raise ValueError(
            f"query_item_keys length {keys_arr.shape[0]} != subject_ids length {sids.shape[0]}"
        )

    unique_keys, inverse = np.unique(keys_arr, return_inverse=True)
    Nu = int(unique_keys.shape[0])

    unique_idx = np.empty((Nu, k), dtype=np.int64)
    unique_sims = np.empty((Nu, k), dtype=np.float32)

    for s in range(0, Nu, chunk):
        e = min(s + chunk, Nu)
        chunk_keys = unique_keys[s:e].tolist()
        first = item_emb_lookup[chunk_keys[0]]
        D = int(np.asarray(first, dtype=np.float32).shape[-1])
        chunk_emb = np.empty((e - s, D), dtype=np.float32)
        for j, key in enumerate(chunk_keys):
            v = np.asarray(item_emb_lookup[key], dtype=np.float32)
            if v.shape != (D,):
                raise ValueError(
                    f"embedding for item_key={key!r} has shape {v.shape}; expected {(D,)}"
                )
            chunk_emb[j] = v
        idx_chunk, sims_chunk = nn_index.nearest(
            chunk_emb,
            k=k,
            exclude_self=exclude_self,
            query_keys=chunk_keys,
        )
        unique_idx[s:e] = idx_chunk
        unique_sims[s:e] = sims_chunk
        # Drop the per-chunk stacked / normalized copies eagerly so we
        # don't carry a 4-5 GB peak across iterations.
        del chunk_emb, idx_chunk, sims_chunk

    neighbor_idx = unique_idx[inverse]      # [B, K]
    sims = unique_sims[inverse]             # [B, K]

    passrates, masks = _lookup_neighbor_passrates(
        sids, neighbor_idx, passrate_csr, passrate_mask_csr
    )
    cond_inputs = _resolve_conditional_inputs(
        conditional_context,
        sids,
        neighbor_idx,
        query_benchmark_ids=query_benchmark_ids,
        query_benchmark_age=query_benchmark_age,
        query_cluster_ids=query_cluster_ids,
        subject_meta_redacted=subject_meta_redacted,
    )
    return _aggregate_nn_features(
        passrates,
        masks,
        sims,
        fallback_value=float(cfg.fallback_value),
        top1_missing_sentinel=float(cfg.top1_missing_sentinel),
        cond_inputs=cond_inputs,
    )


__all__ = [
    "NN_FEATURE_DIM",
    "NN_FEATURE_NAMES",
    "NNFeaturesConfig",
    "TrainingNNIndex",
    "ConditionalPassrateContext",
    "MISSING_TRAIT_ID",
    "_aggregate_nn_features",
    "_aggregate_trait_conditional",
    "_lookup_csr_pairs",
    "_resolve_conditional_inputs",
    "build_passrate_table",
    "build_conditional_passrate_context",
    "compute_nn_features",
    "compute_nn_features_streaming",
]
