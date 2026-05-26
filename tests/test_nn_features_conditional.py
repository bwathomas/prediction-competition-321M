"""Unit tests for the conditional NN feature pipeline.

Covers:

  1. ``_aggregate_nn_features`` schema integrity for cells 15..22 when
     ``cond_inputs`` is None (every cell falls back to ``fallback_value``)
     and when each cell is provided in isolation (no cross-talk).
  2. ``_aggregate_trait_conditional`` matches the per-row
     mean-over-labeled-neighbors formula with explicit redaction.
  3. ``ConditionalPassrateContext`` builder produces shape-correct sparse
     matrices, the right per-item arrays, and a faithful round-trip
     through ``save`` / ``load``.
  4. ``_resolve_conditional_inputs`` combines a context with per-row
     query inputs and drives the aggregator to bit-identical outputs
     (training-time / runtime parity property).

CPU-only, FAISS-free, < 100 ms per test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nn_features import (
    NN_FEATURE_DIM,
    NN_FEATURE_NAMES,
    MISSING_TRAIT_ID,
    ConditionalPassrateContext,
    _aggregate_nn_features,
    _aggregate_trait_conditional,
    _resolve_conditional_inputs,
    build_conditional_passrate_context,
)


# ---------------------------------------------------------------------------
# _aggregate_trait_conditional: redaction + mean recipe
# ---------------------------------------------------------------------------


def test_trait_conditional_matches_mean_over_labeled_neighbors() -> None:
    pr = np.array(
        [
            [0.8, 0.0, 0.6, 0.0],   # 2 obs -> mean (0.8 + 0.6) / 2 = 0.7
            [0.0, 0.0, 0.0, 0.0],   # 0 obs -> fallback
            [1.0, 0.0, 0.0, 0.0],   # 1 obs -> 1.0
        ],
        dtype=np.float32,
    )
    mk = np.array(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    redact = np.array([0, 0, 0], dtype=np.float32)
    out = _aggregate_trait_conditional(pr, mk, redact, fallback_value=0.123)
    assert out[0] == pytest.approx(0.7, abs=1e-6)
    assert out[1] == pytest.approx(0.123, abs=1e-6)
    assert out[2] == pytest.approx(1.0, abs=1e-6)


def test_trait_conditional_redaction_overrides_observations() -> None:
    pr = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    mk = np.array([[1, 1, 1]], dtype=np.float32)
    redact = np.array([1], dtype=np.float32)
    out = _aggregate_trait_conditional(pr, mk, redact, fallback_value=0.42)
    assert out[0] == pytest.approx(0.42, abs=1e-6)


# ---------------------------------------------------------------------------
# _aggregate_nn_features: cells 15..22 default to fallback when cond_inputs is None
# ---------------------------------------------------------------------------


def test_aggregator_falls_back_for_cells_15_22_when_no_context() -> None:
    """No cond_inputs => 8 conditional cells == fallback_value, every row."""
    B, K = 3, 5
    rng = np.random.default_rng(0)
    pr = rng.uniform(0, 1, (B, K)).astype(np.float32)
    mk = (rng.uniform(0, 1, (B, K)) > 0.3).astype(np.float32)
    sims = rng.uniform(0, 1, (B, K)).astype(np.float32)
    fb = 0.123
    out = _aggregate_nn_features(
        pr, mk, sims, fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=None,
    )
    assert out.shape == (B, NN_FEATURE_DIM)
    cond_cols = out[:, 15:23]
    assert np.allclose(cond_cols, fb)


def test_aggregator_partial_cond_inputs_only_overrides_named_cells() -> None:
    """If we provide only freshness, only cell 20 changes; cells 15..19, 21,
    22 stay at fallback. Verifies no cross-cell leakage in the dispatch."""
    B, K = 1, 3
    pr = np.zeros((B, K), dtype=np.float32)
    mk = np.zeros((B, K), dtype=np.float32)
    sims = np.zeros((B, K), dtype=np.float32)
    fb = 0.2
    cond_inputs = {
        "neighbor_freshness_diff": np.array([1.5], dtype=np.float32),
        "freshness_redact": np.array([0], dtype=np.float32),
    }
    out = _aggregate_nn_features(
        pr, mk, sims,
        fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )[0]
    # Cell 20 -> 1.5; everything else in 15..22 stays at fallback.
    assert out[NN_FEATURE_NAMES.index("neighbor_freshness_diff")] == pytest.approx(1.5, abs=1e-6)
    for n in (
        "passrate_subject_conditional",
        "passrate_family_conditional",
        "passrate_macro_family_conditional",
        "passrate_organization_conditional",
        "passrate_benchmark_conditional",
        "n_distinct_subjects_in_neighborhood",
        "cluster_passrate_subject_query",
    ):
        assert out[NN_FEATURE_NAMES.index(n)] == pytest.approx(fb, abs=1e-6)


def test_aggregator_distinct_subjects_uses_log1p_of_mean() -> None:
    """Cell 21: log1p(mean(distinct_per_neighbor)). No redaction."""
    B, K = 1, 4
    pr = np.zeros((B, K), dtype=np.float32)
    mk = np.zeros((B, K), dtype=np.float32)
    sims = np.zeros((B, K), dtype=np.float32)
    cond_inputs = {
        "distinct_subj_per_neighbor": np.array([[2, 4, 4, 6]], dtype=np.float32),
    }
    out = _aggregate_nn_features(
        pr, mk, sims,
        fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )[0]
    expected = float(np.log1p((2 + 4 + 4 + 6) / 4.0))  # log1p(4.0)
    assert out[NN_FEATURE_NAMES.index("n_distinct_subjects_in_neighborhood")] == pytest.approx(expected, abs=1e-6)


def test_aggregator_freshness_redact_forces_fallback() -> None:
    B, K = 2, 3
    pr = np.zeros((B, K), dtype=np.float32)
    mk = np.zeros((B, K), dtype=np.float32)
    sims = np.zeros((B, K), dtype=np.float32)
    cond_inputs = {
        "neighbor_freshness_diff": np.array([2.0, 5.0], dtype=np.float32),
        "freshness_redact": np.array([0, 1], dtype=np.float32),
    }
    fb = 0.0
    out = _aggregate_nn_features(
        pr, mk, sims,
        fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    assert out[0, NN_FEATURE_NAMES.index("neighbor_freshness_diff")] == pytest.approx(2.0, abs=1e-6)
    assert out[1, NN_FEATURE_NAMES.index("neighbor_freshness_diff")] == pytest.approx(fb, abs=1e-6)


def test_aggregator_cluster_redact_forces_fallback() -> None:
    B = 2
    K = 2
    pr = np.zeros((B, K), dtype=np.float32)
    mk = np.zeros((B, K), dtype=np.float32)
    sims = np.zeros((B, K), dtype=np.float32)
    cond_inputs = {
        "cluster_passrate_subject_query": np.array([0.7, 0.4], dtype=np.float32),
        "cluster_redact": np.array([0, 1], dtype=np.float32),
    }
    out = _aggregate_nn_features(
        pr, mk, sims,
        fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    assert out[0, NN_FEATURE_NAMES.index("cluster_passrate_subject_query")] == pytest.approx(0.7, abs=1e-6)
    assert out[1, NN_FEATURE_NAMES.index("cluster_passrate_subject_query")] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Conditional context builder + resolver: full pipeline
# ---------------------------------------------------------------------------


def _make_tiny_context() -> tuple[ConditionalPassrateContext, dict]:
    """Tiny synthetic dataset with deterministic answers for every feature.

    Layout:
      4 train items (item_keys 'i0', 'i1', 'i2', 'i3')
      3 subjects (subject_keys 's0', 's1', 's2')
        s0: family_id=1, macro_family_id=1, organization_id=1
        s1: family_id=1, macro_family_id=1, organization_id=2
        s2: family_id=2, macro_family_id=2, organization_id=2
      Train rows:
        (s0, i0, 1.0)  (s0, i1, 0.0)  (s0, i2, 1.0)
        (s1, i0, 0.5)  (s1, i1, 1.0)  (s1, i3, 0.0)
        (s2, i0, 0.0)  (s2, i2, 0.0)
      Items:
        i0: benchmark=10, age=2.0, cluster=0
        i1: benchmark=10, age=3.0, cluster=0
        i2: benchmark=20, age=4.0, cluster=1
        i3: benchmark=20, age=5.0, cluster=1
    """
    df = pd.DataFrame(
        {
            "subject_key": ["s0", "s0", "s0", "s1", "s1", "s1", "s2", "s2"],
            "item_key":    ["i0", "i1", "i2", "i0", "i1", "i3", "i0", "i2"],
            "label":       [1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        }
    )
    item_index_map = {"i0": 0, "i1": 1, "i2": 2, "i3": 3}
    subject_index_map = {"s0": 0, "s1": 1, "s2": 2}
    # Subject -> trait id arrays; index 0 .. 2.
    s2fam = np.array([1, 1, 2], dtype=np.int32)
    s2macro = np.array([1, 1, 2], dtype=np.int32)
    s2org = np.array([1, 2, 2], dtype=np.int32)
    item_benchmark_id = np.array([10, 10, 20, 20], dtype=np.int32)
    item_benchmark_age = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    item_cluster_id = np.array([0, 0, 1, 1], dtype=np.int32)
    ctx = build_conditional_passrate_context(
        train_df=df,
        item_index_map=item_index_map,
        subject_index_map=subject_index_map,
        subject_to_family_id=s2fam,
        subject_to_macro_family_id=s2macro,
        subject_to_organization_id=s2org,
        item_benchmark_id=item_benchmark_id,
        item_benchmark_age=item_benchmark_age,
        item_cluster_id=item_cluster_id,
        n_families=3,         # ids 0..2; trait id 0 reserved for MISSING
        n_macro_families=3,
        n_organizations=3,
        n_clusters=2,
    )
    return ctx, {
        "item_index_map": item_index_map,
        "subject_index_map": subject_index_map,
    }


def test_context_builder_shape_invariants() -> None:
    ctx, _ = _make_tiny_context()
    ctx.assert_shapes()
    assert ctx.n_subjects == 3
    assert ctx.n_items == 4
    assert ctx.n_clusters == 2
    # Subject 0 has answered 3 items -> 3 entries in subject_passrate row 0.
    row0 = ctx.subject_passrate_csr.getrow(0)
    assert row0.nnz == 3
    assert ctx.item_distinct_subj_count.tolist() == [3, 2, 2, 1]


def test_context_builder_global_passrate_and_mask() -> None:
    """item_global_passrate must equal the per-item mean across observed
    subjects, with mask=1 for any item that has at least one row."""
    ctx, _ = _make_tiny_context()
    # i0: (1.0 + 0.5 + 0.0) / 3 = 0.5
    # i1: (0.0 + 1.0) / 2 = 0.5
    # i2: (1.0 + 0.0) / 2 = 0.5
    # i3: 0.0
    expected_pr = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    expected_mk = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(ctx.item_global_passrate, expected_pr, atol=1e-6)
    assert np.allclose(ctx.item_global_passrate_mask, expected_mk, atol=1e-6)


def test_context_builder_family_passrate_aggregates_across_subjects_in_family() -> None:
    """family=1 = {s0, s1}. Family-1 row of family_passrate_csr is the mean
    label across (s0, s1) per item."""
    ctx, _ = _make_tiny_context()
    fam_row = ctx.family_passrate_csr.getrow(1).toarray().reshape(-1)
    fam_mk = ctx.family_passrate_mask_csr.getrow(1).toarray().reshape(-1)
    # i0: (s0=1.0, s1=0.5) -> 0.75 ; mask = 1
    # i1: (s0=0.0, s1=1.0) -> 0.5 ; mask = 1
    # i2: (s0=1.0)         -> 1.0 ; mask = 1
    # i3: (s1=0.0)         -> 0.0 ; mask = 1
    assert np.allclose(fam_row, [0.75, 0.5, 1.0, 0.0], atol=1e-6)
    assert np.allclose(fam_mk,  [1.0, 1.0, 1.0, 1.0], atol=1e-6)


def test_context_builder_skips_missing_trait_rows() -> None:
    """Trait id 0 (MISSING) must be empty -- queries with that id always
    fall back to redaction even if the data accidentally produced rows."""
    ctx, _ = _make_tiny_context()
    fam_row0 = ctx.family_passrate_csr.getrow(0)
    macro_row0 = ctx.macro_family_passrate_csr.getrow(0)
    org_row0 = ctx.organization_passrate_csr.getrow(0)
    assert fam_row0.nnz == 0
    assert macro_row0.nnz == 0
    assert org_row0.nnz == 0


def test_context_builder_cluster_subject_passrate_matches_hand_computed() -> None:
    """cluster 0 = {i0, i1}; cluster 1 = {i2, i3}.
    cluster_subject_passrate[c, s] = mean(label) over rows where item is in
    cluster c and subject == s.
    """
    ctx, _ = _make_tiny_context()
    cps = ctx.cluster_subject_passrate_csr.toarray()
    # cluster 0:
    #   s0: i0=1.0, i1=0.0 -> 0.5
    #   s1: i0=0.5, i1=1.0 -> 0.75
    #   s2: i0=0.0         -> 0.0
    # cluster 1:
    #   s0: i2=1.0         -> 1.0
    #   s1: i3=0.0         -> 0.0
    #   s2: i2=0.0         -> 0.0
    expected = np.array(
        [[0.5, 0.75, 0.0],
         [1.0, 0.0,  0.0]],
        dtype=np.float32,
    )
    assert np.allclose(cps, expected, atol=1e-6)


def test_resolve_conditional_inputs_drives_full_aggregator() -> None:
    """End-to-end: resolve inputs from a context, then feed
    ``_aggregate_nn_features``. Cells 15..22 should reflect the
    hand-computed values, AND the legacy cells 0..14 should be
    untouched.
    """
    ctx, _ = _make_tiny_context()
    # Two queries:
    #   row 0: subject_id=0 (s0), benchmark_q=10, cluster_q=0, age=2.5
    #   row 1: subject_id=1 (s1), benchmark_q=20, cluster_q=1, age=4.5
    subject_ids = np.array([0, 1], dtype=np.int64)
    # Neighbors (precomputed): row 0 -> [i0, i1, i3], row 1 -> [i2, i0, i3]
    neighbor_idx = np.array([[0, 1, 3], [2, 0, 3]], dtype=np.int64)
    sims = np.array([[1.0, 0.7, 0.4], [0.9, 0.6, 0.3]], dtype=np.float32)
    pr_kk, mk_kk = np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
    cond_inputs = _resolve_conditional_inputs(
        ctx,
        subject_ids,
        neighbor_idx,
        query_benchmark_ids=np.array([10, 20], dtype=np.int32),
        query_benchmark_age=np.array([2.5, 4.5], dtype=np.float32),
        query_cluster_ids=np.array([0, 1], dtype=np.int32),
        subject_meta_redacted=np.array([0, 0], dtype=np.int32),
    )
    out = _aggregate_nn_features(
        pr_kk, mk_kk, sims,
        fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )

    # ---- Cell 15: passrate_subject_conditional ----
    # Row 0 (s0): per-(s0, neighbor) labels = [1.0, 0.0, NaN] -> mean(0.5)
    # Row 1 (s1): per-(s1, neighbor) labels = [NaN, 0.5, 0.0] -> mean(0.25)
    sc = NN_FEATURE_NAMES.index("passrate_subject_conditional")
    assert out[0, sc] == pytest.approx(0.5, abs=1e-6)
    assert out[1, sc] == pytest.approx(0.25, abs=1e-6)

    # ---- Cell 16: passrate_family_conditional ----
    # family=1 row of family_passrate is [0.75, 0.5, 1.0, 0.0], all observed.
    # Row 0 (fam=1) over neighbors [i0, i1, i3] -> (0.75 + 0.5 + 0.0)/3 = 0.4166...
    # Row 1 (fam=1) over neighbors [i2, i0, i3] -> (1.0 + 0.75 + 0.0)/3 = 0.5833...
    fc = NN_FEATURE_NAMES.index("passrate_family_conditional")
    assert out[0, fc] == pytest.approx((0.75 + 0.5 + 0.0) / 3, abs=1e-5)
    assert out[1, fc] == pytest.approx((1.0 + 0.75 + 0.0) / 3, abs=1e-5)

    # ---- Cell 19: passrate_benchmark_conditional ----
    # Row 0 (bench_q=10): neighbors at i0(b=10), i1(b=10), i3(b=20).
    #   Matched: i0, i1 with global passrates 0.5, 0.5 -> mean = 0.5
    # Row 1 (bench_q=20): neighbors at i2(b=20), i0(b=10), i3(b=20).
    #   Matched: i2, i3 with global passrates 0.5, 0.0 -> mean = 0.25
    bc = NN_FEATURE_NAMES.index("passrate_benchmark_conditional")
    assert out[0, bc] == pytest.approx(0.5, abs=1e-5)
    assert out[1, bc] == pytest.approx(0.25, abs=1e-5)

    # ---- Cell 20: neighbor_freshness_diff ----
    # Row 0: q_age=2.5; neighbor ages = [2.0, 3.0, 5.0] -> mean=3.333..
    #   diff = 2.5 - 3.333.. = -0.833..
    # Row 1: q_age=4.5; neighbor ages = [4.0, 2.0, 5.0] -> mean=3.666..
    #   diff = 4.5 - 3.666.. = 0.833..
    fd = NN_FEATURE_NAMES.index("neighbor_freshness_diff")
    assert out[0, fd] == pytest.approx(2.5 - (2.0 + 3.0 + 5.0) / 3, abs=1e-5)
    assert out[1, fd] == pytest.approx(4.5 - (4.0 + 2.0 + 5.0) / 3, abs=1e-5)

    # ---- Cell 21: n_distinct_subjects_in_neighborhood ----
    # item_distinct_subj_count = [3, 2, 2, 1]
    # Row 0: neighbors [i0, i1, i3] -> mean(3,2,1)=2.0 -> log1p(2.0)
    # Row 1: neighbors [i2, i0, i3] -> mean(2,3,1)=2.0 -> log1p(2.0)
    nd = NN_FEATURE_NAMES.index("n_distinct_subjects_in_neighborhood")
    assert out[0, nd] == pytest.approx(float(np.log1p(2.0)), abs=1e-5)
    assert out[1, nd] == pytest.approx(float(np.log1p(2.0)), abs=1e-5)

    # ---- Cell 22: cluster_passrate_subject_query ----
    # cps[c=0, s=0] = 0.5; cps[c=1, s=1] = 0.0
    cp = NN_FEATURE_NAMES.index("cluster_passrate_subject_query")
    assert out[0, cp] == pytest.approx(0.5, abs=1e-6)
    assert out[1, cp] == pytest.approx(0.0, abs=1e-6)


def test_resolve_redacts_when_query_metadata_is_unknown() -> None:
    """Pass benchmark_id=-1, age=NaN, cluster=-1, subject_meta_redacted=1
    for a single row: cells 15..20, 22 must drop to fallback while cell 21
    (per-item distinct count) stays informative."""
    ctx, _ = _make_tiny_context()
    subject_ids = np.array([0], dtype=np.int64)
    neighbor_idx = np.array([[0, 1, 3]], dtype=np.int64)
    sims = np.array([[1.0, 0.7, 0.4]], dtype=np.float32)
    pr_kk = np.zeros((1, 3), dtype=np.float32)
    mk_kk = np.zeros((1, 3), dtype=np.float32)
    cond_inputs = _resolve_conditional_inputs(
        ctx,
        subject_ids,
        neighbor_idx,
        query_benchmark_ids=np.array([-1], dtype=np.int32),
        query_benchmark_age=np.array([np.nan], dtype=np.float32),
        query_cluster_ids=np.array([-1], dtype=np.int32),
        subject_meta_redacted=np.array([1], dtype=np.int32),
    )
    fb = 0.123
    out = _aggregate_nn_features(
        pr_kk, mk_kk, sims,
        fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )[0]
    for nm in (
        "passrate_subject_conditional",
        "passrate_family_conditional",
        "passrate_macro_family_conditional",
        "passrate_organization_conditional",
        "passrate_benchmark_conditional",
        "neighbor_freshness_diff",
        "cluster_passrate_subject_query",
    ):
        assert out[NN_FEATURE_NAMES.index(nm)] == pytest.approx(fb, abs=1e-6), (
            f"{nm} should be redacted to fallback"
        )
    # Cell 21 is per-item only; should still reflect log1p(mean([3, 2, 1]))
    n_distinct = out[NN_FEATURE_NAMES.index("n_distinct_subjects_in_neighborhood")]
    assert n_distinct == pytest.approx(float(np.log1p(2.0)), abs=1e-5)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_context_save_load_roundtrip(tmp_path: Path) -> None:
    ctx, _ = _make_tiny_context()
    out_dir = tmp_path / "cond"
    ctx.save(out_dir)
    ctx2 = ConditionalPassrateContext.load(out_dir)
    ctx2.assert_shapes()
    assert (
        ctx2.subject_passrate_csr.toarray().tolist()
        == ctx.subject_passrate_csr.toarray().tolist()
    )
    assert (
        ctx2.family_passrate_csr.toarray().tolist()
        == ctx.family_passrate_csr.toarray().tolist()
    )
    assert (
        ctx2.cluster_subject_passrate_csr.toarray().tolist()
        == ctx.cluster_subject_passrate_csr.toarray().tolist()
    )
    assert ctx2.subject_to_family_id.tolist() == ctx.subject_to_family_id.tolist()
    assert ctx2.item_distinct_subj_count.tolist() == ctx.item_distinct_subj_count.tolist()
    assert ctx2.item_global_passrate.tolist() == ctx.item_global_passrate.tolist()
    assert ctx2.item_cluster_id.tolist() == ctx.item_cluster_id.tolist()
    assert ctx2.n_subjects == ctx.n_subjects
    assert ctx2.n_clusters == ctx.n_clusters


# ---------------------------------------------------------------------------
# Defensive: empty context returns empty cond_inputs without crashing
# ---------------------------------------------------------------------------


def test_resolve_with_none_context_returns_empty_dict() -> None:
    out = _resolve_conditional_inputs(
        None,
        np.array([0], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
        query_benchmark_ids=None,
        query_benchmark_age=None,
        query_cluster_ids=None,
        subject_meta_redacted=None,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Defensive auto-grow when callers under-declare cardinalities
#
# Real-world failure mode: the notebook reads ``CFG["clustering"]["k"]``
# but the upstream FAISS pipeline used a different K, so the cluster
# IDs in ``item_cluster_id`` exceed the declared ``n_clusters``. The
# same risk applies to trait IDs when a subject's encoded family /
# macro_family / organization id slips above ``vocab.n_tokens``.
# ``build_conditional_passrate_context`` must not raise -- it auto-grows
# the matrix dimension and records the effective cardinality.
# ---------------------------------------------------------------------------


def _build_ctx_with_overrides(
    *,
    s2fam: np.ndarray | None = None,
    s2macro: np.ndarray | None = None,
    s2org: np.ndarray | None = None,
    item_cluster_id: np.ndarray | None = None,
    n_families: int = 3,
    n_macro_families: int = 3,
    n_organizations: int = 3,
    n_clusters: int = 2,
):
    df = pd.DataFrame(
        {
            "subject_key": ["s0", "s0", "s1", "s1", "s2"],
            "item_key":    ["i0", "i1", "i0", "i2", "i3"],
            "label":       [1.0, 0.0, 0.5, 1.0, 0.0],
        }
    )
    return build_conditional_passrate_context(
        train_df=df,
        item_index_map={"i0": 0, "i1": 1, "i2": 2, "i3": 3},
        subject_index_map={"s0": 0, "s1": 1, "s2": 2},
        subject_to_family_id=s2fam if s2fam is not None else np.array([1, 1, 2], dtype=np.int32),
        subject_to_macro_family_id=s2macro if s2macro is not None else np.array([1, 1, 2], dtype=np.int32),
        subject_to_organization_id=s2org if s2org is not None else np.array([1, 2, 2], dtype=np.int32),
        item_benchmark_id=np.array([10, 10, 20, 20], dtype=np.int32),
        item_benchmark_age=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        item_cluster_id=item_cluster_id if item_cluster_id is not None else np.array([0, 0, 1, 1], dtype=np.int32),
        n_families=n_families,
        n_macro_families=n_macro_families,
        n_organizations=n_organizations,
        n_clusters=n_clusters,
    )


def test_context_builder_auto_grows_n_clusters_when_undersized() -> None:
    """Reproduces the Codabench Colab failure: cluster ids reach 7 but
    caller declared n_clusters=4. Builder must auto-grow, not raise."""
    item_cluster_id = np.array([0, 3, 7, 5], dtype=np.int32)
    ctx = _build_ctx_with_overrides(
        item_cluster_id=item_cluster_id, n_clusters=4
    )
    ctx.assert_shapes()
    assert ctx.cluster_subject_passrate_csr.shape[0] == 8, (
        "expected matrix to grow to max_id+1 = 8, got "
        f"{ctx.cluster_subject_passrate_csr.shape[0]}"
    )
    assert ctx.n_clusters == 8


def test_context_builder_auto_grows_n_families_when_undersized() -> None:
    """Same defensive path for trait cardinalities -- if a subject's
    family id is above the declared vocab size, the matrix grows
    rather than COO-asserting."""
    s2fam = np.array([1, 5, 9], dtype=np.int32)  # max id 9, declared 3
    ctx = _build_ctx_with_overrides(s2fam=s2fam, n_families=3)
    ctx.assert_shapes()
    assert ctx.family_passrate_csr.shape[0] == 10
    assert ctx.n_families == 10


def test_context_builder_keeps_declared_when_oversized() -> None:
    """If the declared cardinality is already large enough, we honor it
    so the CSR row count stays predictable for downstream consumers."""
    ctx = _build_ctx_with_overrides(n_families=50, n_clusters=64)
    assert ctx.family_passrate_csr.shape[0] == 50
    assert ctx.cluster_subject_passrate_csr.shape[0] == 64
    assert ctx.n_families == 50
    assert ctx.n_clusters == 64


def test_context_builder_handles_all_unassigned_clusters() -> None:
    """If every train item has cluster_id == -1 we still get an empty
    matrix sized to the declared (or default) n_clusters."""
    item_cluster_id = np.full(4, -1, dtype=np.int32)
    ctx = _build_ctx_with_overrides(
        item_cluster_id=item_cluster_id, n_clusters=4
    )
    assert ctx.cluster_subject_passrate_csr.shape == (4, 3)
    assert ctx.cluster_subject_passrate_csr.nnz == 0
