"""Red-team parity tests for the conditional NN-feature pipeline.

The training-time aggregator and the runtime aggregator must produce
*bit-identical* outputs for the same inputs -- the runtime literally
inlines the function via the ``_RUNTIME_MODEL_PY`` raw string in
``src/export_submission.py``. These tests:

  1. Extract the runtime helpers (``_aggregate_nn_features``,
     ``_aggregate_trait_conditional``, ``_lookup_csr_pairs``,
     ``_resolve_conditional_inputs_runtime``) from the raw runtime
     template by ``exec``-ing the necessary fragment into a namespace.
  2. Compare them against the canonical implementations in
     ``src.nn_features`` for a battery of synthetic inputs covering
     legacy cells, conditional cells, and redaction edge cases.
  3. Round-trip a ``ConditionalPassrateContext`` through ``save`` /
     ``load`` and re-verify parity.

If a test fails, it usually means the two copies of
``_aggregate_nn_features`` have drifted; check the comments at the top
of each function and update both in lockstep.
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import src.export_submission as exp
import src.nn_features as nnf


# ---------------------------------------------------------------------------
# Extract the runtime helpers from the embedded template.
# ---------------------------------------------------------------------------


def _extract_runtime_helpers() -> types.ModuleType:
    """Find the runtime aggregator + resolver inside ``_RUNTIME_MODEL_PY``
    and exec them into a fresh module namespace.

    We do NOT exec the entire template (it pulls in torch + the entire
    runtime submission machinery). Instead we slice out just the
    helpers we care about: ``_aggregate_trait_conditional``,
    ``_aggregate_nn_features``, ``_lookup_csr_pairs``,
    ``_resolve_conditional_inputs_runtime`` and the constant
    ``_MISSING_TRAIT_ID``. Locating the slice by string markers keeps
    the test resilient to small reorderings in the template.
    """
    src = exp._RUNTIME_MODEL_PY
    start_marker = "def _aggregate_trait_conditional"
    end_marker = "# Inlined model classes (mirror src/models.py)"
    s = src.index(start_marker)
    e = src.index(end_marker, s)
    fragment = src[s:e]

    mod = types.ModuleType("_runtime_helpers_under_test")
    mod.__dict__["np"] = np
    mod.__dict__["LOG"] = type("L", (), {"warning": staticmethod(lambda *a, **k: None)})()
    exec(compile(textwrap.dedent(fragment), "<runtime-helpers-test>", "exec"), mod.__dict__)
    return mod


_RT = _extract_runtime_helpers()


# ---------------------------------------------------------------------------
# Aggregator parity: training-time output == runtime output, byte for byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 7])
def test_aggregator_parity_legacy_cells_only(seed: int) -> None:
    """Without ``cond_inputs`` both copies must agree on the 23-column
    output (cells 0..14 carry signal; cells 15..22 are fallback)."""
    rng = np.random.default_rng(seed)
    B, K = 5, 7
    pr = rng.uniform(0, 1, (B, K)).astype(np.float32)
    mk = (rng.uniform(0, 1, (B, K)) > 0.3).astype(np.float32)
    sims = rng.uniform(-1, 1, (B, K)).astype(np.float32)
    train_out = nnf._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.123, top1_missing_sentinel=-1.0
    )
    rt_out = _RT._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.123, top1_missing_sentinel=-1.0
    )
    assert train_out.shape == rt_out.shape == (B, nnf.NN_FEATURE_DIM)
    np.testing.assert_array_equal(train_out, rt_out)


def test_aggregator_parity_with_full_cond_inputs() -> None:
    """When conditional inputs are provided in full, both copies must
    still agree byte for byte."""
    B, K = 4, 6
    rng = np.random.default_rng(11)
    pr = rng.uniform(0, 1, (B, K)).astype(np.float32)
    mk = (rng.uniform(0, 1, (B, K)) > 0.3).astype(np.float32)
    sims = rng.uniform(0, 1, (B, K)).astype(np.float32)
    cond_inputs = {
        "subject_passrates": rng.uniform(0, 1, (B, K)).astype(np.float32),
        "subject_masks": (rng.uniform(0, 1, (B, K)) > 0.5).astype(np.float32),
        "subject_redact": np.array([0, 1, 0, 0], dtype=np.float32),
        "family_passrates": rng.uniform(0, 1, (B, K)).astype(np.float32),
        "family_masks": (rng.uniform(0, 1, (B, K)) > 0.5).astype(np.float32),
        "family_redact": np.array([0, 0, 1, 0], dtype=np.float32),
        "macro_family_passrates": rng.uniform(0, 1, (B, K)).astype(np.float32),
        "macro_family_masks": (rng.uniform(0, 1, (B, K)) > 0.5).astype(np.float32),
        "macro_family_redact": np.zeros(B, dtype=np.float32),
        "organization_passrates": rng.uniform(0, 1, (B, K)).astype(np.float32),
        "organization_masks": (rng.uniform(0, 1, (B, K)) > 0.5).astype(np.float32),
        "organization_redact": np.zeros(B, dtype=np.float32),
        "bench_match_passrates": rng.uniform(0, 1, (B, K)).astype(np.float32),
        "bench_match_masks": (rng.uniform(0, 1, (B, K)) > 0.5).astype(np.float32),
        "bench_match_redact": np.array([0, 0, 0, 1], dtype=np.float32),
        "neighbor_freshness_diff": rng.uniform(-2, 2, B).astype(np.float32),
        "freshness_redact": np.array([0, 1, 0, 0], dtype=np.float32),
        "distinct_subj_per_neighbor": rng.uniform(1, 10, (B, K)).astype(np.float32),
        "cluster_passrate_subject_query": rng.uniform(0, 1, B).astype(np.float32),
        "cluster_redact": np.array([0, 0, 1, 0], dtype=np.float32),
    }
    train_out = nnf._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    rt_out = _RT._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    np.testing.assert_array_equal(train_out, rt_out)


def test_aggregator_parity_with_partial_cond_inputs() -> None:
    """Partial dict (e.g. only freshness): both copies must apply the
    same per-cell fallback."""
    B, K = 3, 4
    pr = np.zeros((B, K), dtype=np.float32)
    mk = np.zeros((B, K), dtype=np.float32)
    sims = np.zeros((B, K), dtype=np.float32)
    cond_inputs = {
        "neighbor_freshness_diff": np.array([1.0, -1.0, 0.5], dtype=np.float32),
        "distinct_subj_per_neighbor": np.array([[2, 4, 4, 6]] * B, dtype=np.float32),
    }
    train_out = nnf._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    rt_out = _RT._aggregate_nn_features(
        pr, mk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=cond_inputs,
    )
    np.testing.assert_array_equal(train_out, rt_out)


# ---------------------------------------------------------------------------
# Trait-conditional helper parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 13])
def test_trait_conditional_helper_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pr = rng.uniform(0, 1, (5, 6)).astype(np.float32)
    mk = (rng.uniform(0, 1, (5, 6)) > 0.4).astype(np.float32)
    redact = (rng.uniform(0, 1, 5) > 0.7).astype(np.float32)
    a = nnf._aggregate_trait_conditional(pr, mk, redact, fallback_value=0.5)
    b = _RT._aggregate_trait_conditional(pr, mk, redact, fallback_value=0.5)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# CSR lookup parity
# ---------------------------------------------------------------------------


def test_csr_lookup_parity() -> None:
    """Random small CSR + (row, col) pairs: training and runtime versions
    must return identical (passrates, masks)."""
    rng = np.random.default_rng(99)
    rows, cols = 4, 8
    dense_pr = rng.uniform(0, 1, (rows, cols)).astype(np.float32)
    dense_mk = (rng.uniform(0, 1, (rows, cols)) > 0.5).astype(np.float32)
    pr_csr = sp.csr_matrix(dense_pr * dense_mk)
    mk_csr = sp.csr_matrix(dense_mk)
    row_ids = rng.integers(0, rows, size=6)
    col_ids = rng.integers(0, cols, size=(6, 3))
    a_pr, a_mk = nnf._lookup_csr_pairs(row_ids, col_ids, pr_csr, mk_csr)
    b_pr, b_mk = _RT._lookup_csr_pairs(row_ids, col_ids, pr_csr, mk_csr)
    np.testing.assert_array_equal(a_pr, b_pr)
    np.testing.assert_array_equal(a_mk, b_mk)


# ---------------------------------------------------------------------------
# End-to-end: build a tiny context, run training-time + runtime resolvers
# in parallel, verify byte-identical aggregator output.
# ---------------------------------------------------------------------------


def _tiny_context_for_parity() -> tuple:
    """Hand-rolled ConditionalPassrateContext (3 subjects, 4 items, 2 clusters).

    Mirrors the synthetic data in ``test_nn_features_conditional.py`` so
    we exercise a non-degenerate bag of CSR matrices.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "subject_key": ["s0", "s0", "s1", "s1", "s2"],
            "item_key":    ["i0", "i1", "i0", "i2", "i3"],
            "label":       [1.0, 0.0, 0.5, 1.0, 0.0],
        }
    )
    item_index_map = {"i0": 0, "i1": 1, "i2": 2, "i3": 3}
    subject_index_map = {"s0": 0, "s1": 1, "s2": 2}
    s2fam = np.array([1, 1, 2], dtype=np.int32)
    s2macro = np.array([1, 1, 2], dtype=np.int32)
    s2org = np.array([1, 2, 2], dtype=np.int32)
    item_benchmark_id = np.array([10, 10, 20, 20], dtype=np.int32)
    item_benchmark_age = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    item_cluster_id = np.array([0, 0, 1, 1], dtype=np.int32)
    ctx = nnf.build_conditional_passrate_context(
        train_df=df,
        item_index_map=item_index_map,
        subject_index_map=subject_index_map,
        subject_to_family_id=s2fam,
        subject_to_macro_family_id=s2macro,
        subject_to_organization_id=s2org,
        item_benchmark_id=item_benchmark_id,
        item_benchmark_age=item_benchmark_age,
        item_cluster_id=item_cluster_id,
        n_families=3,
        n_macro_families=3,
        n_organizations=3,
        n_clusters=2,
    )
    return ctx, df


def test_end_to_end_train_vs_runtime_parity_per_row() -> None:
    """For each row in a small batch, drive the runtime resolver
    (single-row API) and the training resolver (batched API) and verify
    the resulting feature vectors are bit-identical."""
    ctx, _ = _tiny_context_for_parity()
    subject_ids = np.array([0, 1, 2], dtype=np.int64)
    neighbor_idx = np.array(
        [
            [0, 1, 3],
            [2, 0, 3],
            [3, 2, 1],
        ],
        dtype=np.int64,
    )
    sims = np.array(
        [
            [1.0, 0.7, 0.4],
            [0.9, 0.6, 0.3],
            [0.8, 0.5, 0.2],
        ],
        dtype=np.float32,
    )
    pr_kk = np.zeros((3, 3), dtype=np.float32)
    mk_kk = np.zeros((3, 3), dtype=np.float32)
    bench_ids = np.array([10, 20, 20], dtype=np.int32)
    bench_age = np.array([2.5, 4.5, 4.5], dtype=np.float32)
    cluster_ids = np.array([0, 1, 1], dtype=np.int32)

    train_cond = nnf._resolve_conditional_inputs(
        ctx,
        subject_ids,
        neighbor_idx,
        query_benchmark_ids=bench_ids,
        query_benchmark_age=bench_age,
        query_cluster_ids=cluster_ids,
        subject_meta_redacted=np.zeros(3, dtype=np.int32),
    )
    train_out = nnf._aggregate_nn_features(
        pr_kk, mk_kk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=train_cond,
    )

    for b in range(3):
        rt_cond = _RT._resolve_conditional_inputs_runtime(
            subject_id=int(subject_ids[b]),
            neighbor_idx=neighbor_idx[b],
            query_benchmark_id=int(bench_ids[b]),
            query_benchmark_age=float(bench_age[b]),
            query_cluster_id=int(cluster_ids[b]),
            subject_meta_redacted=False,
            subject_passrate_csr=ctx.subject_passrate_csr,
            subject_passrate_mask_csr=ctx.subject_passrate_mask_csr,
            family_passrate_csr=ctx.family_passrate_csr,
            family_passrate_mask_csr=ctx.family_passrate_mask_csr,
            macro_family_passrate_csr=ctx.macro_family_passrate_csr,
            macro_family_passrate_mask_csr=ctx.macro_family_passrate_mask_csr,
            organization_passrate_csr=ctx.organization_passrate_csr,
            organization_passrate_mask_csr=ctx.organization_passrate_mask_csr,
            subject_to_family_id=ctx.subject_to_family_id,
            subject_to_macro_family_id=ctx.subject_to_macro_family_id,
            subject_to_organization_id=ctx.subject_to_organization_id,
            item_benchmark_id=ctx.item_benchmark_id,
            item_benchmark_age=ctx.item_benchmark_age,
            item_distinct_subj_count=ctx.item_distinct_subj_count,
            item_global_passrate=ctx.item_global_passrate,
            item_global_passrate_mask=ctx.item_global_passrate_mask,
            cluster_subject_passrate_csr=ctx.cluster_subject_passrate_csr,
            cluster_subject_passrate_mask_csr=ctx.cluster_subject_passrate_mask_csr,
        )
        rt_out_b = _RT._aggregate_nn_features(
            pr_kk[b : b + 1],
            mk_kk[b : b + 1],
            sims[b : b + 1],
            fallback_value=0.0,
            top1_missing_sentinel=-1.0,
            cond_inputs=rt_cond,
        )[0]
        np.testing.assert_allclose(
            rt_out_b,
            train_out[b],
            atol=1e-6,
            rtol=0.0,
            err_msg=f"row {b}: train and runtime disagree",
        )


def test_save_load_roundtrip_preserves_aggregator_output(tmp_path: Path) -> None:
    """Round-trip a context through save/load and verify the aggregator
    output is unchanged. Catches accidental dtype / shape coercions."""
    ctx, _ = _tiny_context_for_parity()
    out_dir = tmp_path / "cond"
    ctx.save(out_dir)
    ctx2 = nnf.ConditionalPassrateContext.load(out_dir)
    subject_ids = np.array([0, 1, 2], dtype=np.int64)
    neighbor_idx = np.array(
        [
            [0, 1, 3],
            [2, 0, 3],
            [3, 2, 1],
        ],
        dtype=np.int64,
    )
    sims = np.full((3, 3), 0.5, dtype=np.float32)
    pr_kk = np.zeros((3, 3), dtype=np.float32)
    mk_kk = np.zeros((3, 3), dtype=np.float32)
    bench_ids = np.array([10, 20, 20], dtype=np.int32)
    bench_age = np.array([2.5, 4.5, 4.5], dtype=np.float32)
    cluster_ids = np.array([0, 1, 1], dtype=np.int32)

    a = nnf._aggregate_nn_features(
        pr_kk, mk_kk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=nnf._resolve_conditional_inputs(
            ctx, subject_ids, neighbor_idx,
            query_benchmark_ids=bench_ids,
            query_benchmark_age=bench_age,
            query_cluster_ids=cluster_ids,
            subject_meta_redacted=None,
        ),
    )
    b = nnf._aggregate_nn_features(
        pr_kk, mk_kk, sims, fallback_value=0.0, top1_missing_sentinel=-1.0,
        cond_inputs=nnf._resolve_conditional_inputs(
            ctx2, subject_ids, neighbor_idx,
            query_benchmark_ids=bench_ids,
            query_benchmark_age=bench_age,
            query_cluster_ids=cluster_ids,
            subject_meta_redacted=None,
        ),
    )
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Defensive: if the resolver receives empty / None contexts, the
# runtime path must still emit a valid 23-d vector (cells 15..22 fallback).
# ---------------------------------------------------------------------------


def test_runtime_with_no_context_returns_legacy_features_only() -> None:
    """Replicates the early bundle case: cache files missing -> runtime
    aggregator output must equal the training output for the legacy
    cells, and cells [15..22] must equal fallback_value."""
    B, K = 2, 3
    rng = np.random.default_rng(0)
    pr = rng.uniform(0, 1, (B, K)).astype(np.float32)
    mk = (rng.uniform(0, 1, (B, K)) > 0.4).astype(np.float32)
    sims = rng.uniform(0, 1, (B, K)).astype(np.float32)
    fb = 0.123
    rt_out = _RT._aggregate_nn_features(
        pr, mk, sims, fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=None,
    )
    train_out = nnf._aggregate_nn_features(
        pr, mk, sims, fallback_value=fb, top1_missing_sentinel=-1.0,
        cond_inputs=None,
    )
    np.testing.assert_array_equal(rt_out, train_out)
    assert np.allclose(rt_out[:, 15:23], fb)
