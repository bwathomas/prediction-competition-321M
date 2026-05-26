"""Regression tests for the honest NN-feature-redaction diagnostics.

The user observed an 83.2% "fallback" rate on cell 20
(``neighbor_freshness_diff``) for the training split, while every other
conditional cell sat in the 0-7% range. The notebook computed that figure
with ``np.mean(np.abs(nn_train_mat[:, i] - cfg.fallback_value) <= 1e-9)``,
which is a *naive proxy* that conflates two independent signals:

  1. Genuine redaction (the cell's redact mask was 1).
  2. Legitimate zero output -- e.g.
     ``neighbor_freshness_diff = q_age - mean(neighbor_age)`` is *exactly*
     0 whenever the query and its neighbors share a (z-scored) benchmark
     date. With clustered ages this is common.

These tests pin the new
:func:`src.nn_features.conditional_redaction_diagnostics` helper as the
*correct* answer (counts directly off the redaction MASK from
``_resolve_conditional_inputs``) and pin the
``compute_nn_features_streaming(return_diagnostics=True)`` plumbing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.nn_features import (
    NN_FEATURE_DIM,
    NN_FEATURE_NAMES,
    NNFeaturesConfig,
    TrainingNNIndex,
    _aggregate_nn_features,
    _resolve_conditional_inputs,
    build_conditional_passrate_context,
    build_passrate_table,
    compute_nn_features_streaming,
    conditional_redaction_diagnostics,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_minimal_context(
    *,
    n_items: int = 8,
    n_subj: int = 5,
    item_age_pattern: str = "all_zero",
):
    """Build a tiny ConditionalPassrateContext + a passrate CSR.

    ``item_age_pattern``:
      * ``"all_zero"``: every train item has ``benchmark_age == 0`` (full
        coverage, zero variance).
      * ``"all_nan"``: every train item has NaN benchmark_age (zero
        coverage; redaction must fire on the neighbor side).
      * ``"linear"``: train items have ages 0, 1, ..., n_items-1 (full
        coverage, varied).
    """
    item_keys = [f"item_{i}" for i in range(n_items)]
    item_index_map = {k: i for i, k in enumerate(item_keys)}
    subj_keys = ["<unk>"] + [f"sub_{i}" for i in range(n_subj - 1)]
    subject_index_map = {k: i for i, k in enumerate(subj_keys)}

    rng = np.random.default_rng(42)
    rows = []
    for s_idx in range(1, n_subj):
        for it_idx in range(n_items):
            rows.append(
                {
                    "subject_key": subj_keys[s_idx],
                    "item_key": item_keys[it_idx],
                    "label": float(rng.uniform() < 0.5),
                }
            )
    df = pd.DataFrame(rows)

    item_benchmark_id = np.arange(n_items, dtype=np.int32)
    if item_age_pattern == "all_zero":
        item_benchmark_age = np.zeros(n_items, dtype=np.float32)
    elif item_age_pattern == "all_nan":
        item_benchmark_age = np.full(n_items, np.nan, dtype=np.float32)
    elif item_age_pattern == "linear":
        item_benchmark_age = np.arange(n_items, dtype=np.float32)
    else:
        raise ValueError(item_age_pattern)
    item_cluster_id = np.full(n_items, -1, dtype=np.int32)

    s2fam = np.zeros(n_subj, dtype=np.int32)
    s2macro = np.zeros(n_subj, dtype=np.int32)
    s2org = np.zeros(n_subj, dtype=np.int32)

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
        n_families=1,
        n_macro_families=1,
        n_organizations=1,
        n_clusters=1,
    )
    pr_csr, pr_mk_csr = build_passrate_table(
        train_df=df,
        item_index_map=item_index_map,
        subject_index_map=subject_index_map,
    )
    return ctx, df, item_keys, subj_keys, item_index_map, subject_index_map, pr_csr, pr_mk_csr


# ---------------------------------------------------------------------------
# Pure resolver-level tests (no streaming)
# ---------------------------------------------------------------------------


def test_legitimate_zero_freshness_is_not_counted_as_redaction():
    """All ages identical, no missingness -> fresh_diff is *exactly* 0.

    The honest diagnostic reports 0 redactions (correct).
    The naive ``|x - fb| <= 1e-9`` heuristic reports 100% (FALSE positive).
    """
    ctx, *_ = _build_minimal_context(item_age_pattern="all_zero")
    B, K = 64, 4
    rng = np.random.default_rng(0)
    sids = rng.integers(1, ctx.n_subjects, size=B).astype(np.int32)
    nidx = rng.integers(0, ctx.n_items, size=(B, K)).astype(np.int32)
    sims = np.full((B, K), 0.5, dtype=np.float32)

    cond_inputs = _resolve_conditional_inputs(
        context=ctx,
        subject_ids=sids,
        neighbor_idx=nidx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=np.zeros(B, dtype=np.float32),
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
    )
    feats = _aggregate_nn_features(
        np.zeros((B, K), dtype=np.float32),
        np.zeros((B, K), dtype=np.float32),
        sims,
        fallback_value=0.0,
        top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    cell_idx = NN_FEATURE_NAMES.index("neighbor_freshness_diff")
    naive_frac = float(np.mean(np.abs(feats[:, cell_idx] - 0.0) <= 1e-9))
    assert naive_frac == pytest.approx(1.0), (
        "Sanity check: with identical ages the cell value is always 0, so the "
        "naive |x-fb|<=eps diagnostic must report 100% (this is exactly the "
        "false positive we are fixing)."
    )

    diag = conditional_redaction_diagnostics(cond_inputs, n_rows=B)
    assert diag["per_cell"]["neighbor_freshness_diff"] == 0, (
        "Honest diagnostic must read the redaction MASK, not the output value. "
        "No row had isnan(q_age) or n_known==0, so 0 redactions."
    )


def test_diagnostic_counts_match_per_cell_redact_masks():
    """Pin: per_cell counts equal sum of each '*_redact' mask."""
    ctx, *_ = _build_minimal_context(item_age_pattern="linear", n_items=10)
    B, K = 50, 3
    rng = np.random.default_rng(1)
    sids = rng.integers(1, ctx.n_subjects, size=B).astype(np.int32)
    nidx = rng.integers(0, ctx.n_items, size=(B, K)).astype(np.int32)

    q_age = rng.normal(size=B).astype(np.float32)
    q_age[: B // 3] = np.nan
    q_bench = np.zeros(B, dtype=np.int32)
    q_bench[B // 4 :] = -1
    q_cluster = np.full(B, -1, dtype=np.int64)
    sub_redacted = np.zeros(B, dtype=np.int32)
    sub_redacted[: B // 5] = 1

    cond_inputs = _resolve_conditional_inputs(
        context=ctx,
        subject_ids=sids,
        neighbor_idx=nidx,
        query_benchmark_ids=q_bench,
        query_benchmark_age=q_age,
        query_cluster_ids=q_cluster,
        subject_meta_redacted=sub_redacted,
    )
    diag = conditional_redaction_diagnostics(cond_inputs, n_rows=B)
    pc = diag["per_cell"]
    pairs = [
        ("passrate_subject_conditional", "subject_redact"),
        ("passrate_family_conditional", "family_redact"),
        ("passrate_macro_family_conditional", "macro_family_redact"),
        ("passrate_organization_conditional", "organization_redact"),
        ("passrate_benchmark_conditional", "bench_match_redact"),
        ("neighbor_freshness_diff", "freshness_redact"),
        ("cluster_passrate_subject_query", "cluster_redact"),
    ]
    for cell_name, mask_key in pairs:
        expected = int((cond_inputs[mask_key] > 0).sum())
        assert pc[cell_name] == expected, (
            f"per_cell['{cell_name}']={pc[cell_name]} disagrees with "
            f"sum({mask_key})={expected}"
        )


def test_diagnostic_handles_no_context():
    """``None`` cond_inputs -> every redactable cell is fully redacted."""
    diag = conditional_redaction_diagnostics(None, n_rows=100)
    assert diag["n_rows"] == 100
    pc = diag["per_cell"]
    for cell in (
        "passrate_subject_conditional",
        "passrate_family_conditional",
        "passrate_macro_family_conditional",
        "passrate_organization_conditional",
        "passrate_benchmark_conditional",
        "neighbor_freshness_diff",
        "cluster_passrate_subject_query",
    ):
        assert pc[cell] == 100


def test_query_side_missing_drives_freshness_redaction():
    """isnan(q_age) for K_q rows -> exactly K_q freshness redactions."""
    ctx, *_ = _build_minimal_context(item_age_pattern="all_zero")
    B, K = 100, 4
    rng = np.random.default_rng(2)
    sids = rng.integers(1, ctx.n_subjects, size=B).astype(np.int32)
    nidx = rng.integers(0, ctx.n_items, size=(B, K)).astype(np.int32)

    q_age = rng.normal(size=B).astype(np.float32)
    miss = rng.uniform(size=B) < 0.3
    q_age[miss] = np.nan

    cond_inputs = _resolve_conditional_inputs(
        context=ctx,
        subject_ids=sids,
        neighbor_idx=nidx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=q_age,
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
    )
    diag = conditional_redaction_diagnostics(cond_inputs, n_rows=B)
    assert diag["per_cell"]["neighbor_freshness_diff"] == int(miss.sum())


def test_neighbor_side_missing_drives_freshness_redaction():
    """All train items have NaN age -> 100% freshness redaction even with
    fully-known query ages.
    """
    ctx, *_ = _build_minimal_context(item_age_pattern="all_nan", n_items=8)
    B, K = 64, 4
    rng = np.random.default_rng(3)
    sids = rng.integers(1, ctx.n_subjects, size=B).astype(np.int32)
    nidx = rng.integers(0, ctx.n_items, size=(B, K)).astype(np.int32)

    cond_inputs = _resolve_conditional_inputs(
        context=ctx,
        subject_ids=sids,
        neighbor_idx=nidx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=np.zeros(B, dtype=np.float32),
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
    )
    diag = conditional_redaction_diagnostics(cond_inputs, n_rows=B)
    assert diag["per_cell"]["neighbor_freshness_diff"] == B


# ---------------------------------------------------------------------------
# End-to-end: compute_nn_features_streaming(return_diagnostics=True)
# ---------------------------------------------------------------------------


def _build_streaming_fixture(*, n_items=8, n_subj=5, dim=4, item_age_pattern="linear"):
    ctx, df, item_keys, subj_keys, item_index_map, subject_index_map, pr_csr, pr_mk_csr = (
        _build_minimal_context(
            n_items=n_items, n_subj=n_subj, item_age_pattern=item_age_pattern
        )
    )

    rng = np.random.default_rng(0)
    item_emb_lookup = {
        k: rng.normal(size=dim).astype(np.float32) for k in item_keys
    }
    cfg = NNFeaturesConfig(
        k=3,
        similarity="cosine",
        feature_dim=NN_FEATURE_DIM,
        fallback_value=0.0,
        exclude_self_in_training=False,
    )

    # Build the index by writing to a tmp dir.
    import tempfile

    tmp = tempfile.mkdtemp(prefix="nn_diag_test_")
    nn_index = TrainingNNIndex.build_from_lookup(
        item_emb_lookup=item_emb_lookup,
        out_dir=tmp,
        cfg=cfg,
        item_keys=item_keys,
    )
    return cfg, ctx, item_emb_lookup, nn_index, pr_csr, pr_mk_csr, item_keys, subject_index_map


def test_streaming_return_diagnostics_returns_tuple_with_correct_shape():
    cfg, ctx, item_emb_lookup, nn_index, pr_csr, pr_mk_csr, item_keys, subj_idx = (
        _build_streaming_fixture()
    )
    rng = np.random.default_rng(7)
    B = 30
    qkeys = list(rng.choice(item_keys, size=B))
    sids = rng.integers(1, max(2, len(subj_idx)), size=B).astype(np.int32)
    q_bench = rng.integers(0, ctx.n_items, size=B).astype(np.int32)
    q_age = rng.normal(size=B).astype(np.float32)
    q_age[: B // 4] = np.nan
    q_cluster = np.full(B, -1, dtype=np.int64)

    feats = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=nn_index,
        passrate_csr=pr_csr,
        passrate_mask_csr=pr_mk_csr,
        cfg=cfg,
        exclude_self=False,
        conditional_context=ctx,
        query_benchmark_ids=q_bench,
        query_benchmark_age=q_age,
        query_cluster_ids=q_cluster,
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
        return_diagnostics=False,
    )
    assert feats.shape == (B, NN_FEATURE_DIM)

    feats2, diag = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=nn_index,
        passrate_csr=pr_csr,
        passrate_mask_csr=pr_mk_csr,
        cfg=cfg,
        exclude_self=False,
        conditional_context=ctx,
        query_benchmark_ids=q_bench,
        query_benchmark_age=q_age,
        query_cluster_ids=q_cluster,
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
        return_diagnostics=True,
    )
    assert feats2.shape == (B, NN_FEATURE_DIM)
    np.testing.assert_array_equal(feats, feats2)

    assert isinstance(diag, dict)
    assert diag["n_rows"] == B
    assert "per_cell" in diag
    pc = diag["per_cell"]
    for cell in (
        "passrate_subject_conditional",
        "neighbor_freshness_diff",
        "passrate_benchmark_conditional",
        "cluster_passrate_subject_query",
    ):
        assert cell in pc and 0 <= pc[cell] <= B

    assert "freshness" in diag
    fr = diag["freshness"]
    assert fr["n_query_age_known"] == int(np.isfinite(q_age).sum())
    assert fr["n_query_total"] == B
    assert fr["n_train_items_total"] == ctx.n_items
    assert fr["n_train_items_with_known_age"] == int(
        np.isfinite(ctx.item_benchmark_age).sum()
    )


def test_streaming_diagnostic_freshness_count_pinpoints_nan_q_age():
    """When only the query side has NaN ages, the honest count must equal
    isnan(q_age).sum().
    """
    cfg, ctx, item_emb_lookup, nn_index, pr_csr, pr_mk_csr, item_keys, subj_idx = (
        _build_streaming_fixture(item_age_pattern="all_zero")
    )
    rng = np.random.default_rng(11)
    B = 50
    qkeys = list(rng.choice(item_keys, size=B))
    sids = rng.integers(1, max(2, len(subj_idx)), size=B).astype(np.int32)
    q_age = rng.normal(size=B).astype(np.float32)
    q_age[B // 2 :] = np.nan

    _, diag = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=nn_index,
        passrate_csr=pr_csr,
        passrate_mask_csr=pr_mk_csr,
        cfg=cfg,
        exclude_self=False,
        conditional_context=ctx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=q_age,
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
        return_diagnostics=True,
    )
    assert diag["per_cell"]["neighbor_freshness_diff"] == int(
        np.isnan(q_age).sum()
    )
    assert diag["freshness"]["n_query_age_known"] == int(
        np.isfinite(q_age).sum()
    )
    # neighbor side is fully populated (item_age_pattern='all_zero')
    assert diag["freshness"]["n_train_items_with_known_age"] == ctx.n_items


def test_streaming_diagnostic_freshness_count_pinpoints_nan_neighbors():
    """When neighbors have NaN ages but queries are fully known, the honest
    count must show 100% redaction with the neighbor-side localizer
    'frac_rows_with_zero_known_neighbors == 1'.
    """
    cfg, ctx, item_emb_lookup, nn_index, pr_csr, pr_mk_csr, item_keys, subj_idx = (
        _build_streaming_fixture(item_age_pattern="all_nan")
    )
    rng = np.random.default_rng(13)
    B = 50
    qkeys = list(rng.choice(item_keys, size=B))
    sids = rng.integers(1, max(2, len(subj_idx)), size=B).astype(np.int32)
    q_age = rng.normal(size=B).astype(np.float32)

    _, diag = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=nn_index,
        passrate_csr=pr_csr,
        passrate_mask_csr=pr_mk_csr,
        cfg=cfg,
        exclude_self=False,
        conditional_context=ctx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=q_age,
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
        return_diagnostics=True,
    )
    assert diag["per_cell"]["neighbor_freshness_diff"] == B
    assert diag["freshness"]["n_query_age_known"] == B
    assert diag["freshness"]["frac_rows_with_zero_known_neighbors"] == 1.0
    assert diag["freshness"]["n_train_items_with_known_age"] == 0


def test_streaming_default_behavior_unchanged_without_flag():
    """Backward compat: omitting return_diagnostics returns the raw matrix."""
    cfg, ctx, item_emb_lookup, nn_index, pr_csr, pr_mk_csr, item_keys, subj_idx = (
        _build_streaming_fixture()
    )
    rng = np.random.default_rng(17)
    B = 20
    qkeys = list(rng.choice(item_keys, size=B))
    sids = rng.integers(1, max(2, len(subj_idx)), size=B).astype(np.int32)

    out = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=nn_index,
        passrate_csr=pr_csr,
        passrate_mask_csr=pr_mk_csr,
        cfg=cfg,
        exclude_self=False,
        conditional_context=ctx,
        query_benchmark_ids=np.zeros(B, dtype=np.int32),
        query_benchmark_age=np.zeros(B, dtype=np.float32),
        query_cluster_ids=np.full(B, -1, dtype=np.int64),
        subject_meta_redacted=np.zeros(B, dtype=np.int32),
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (B, NN_FEATURE_DIM)
