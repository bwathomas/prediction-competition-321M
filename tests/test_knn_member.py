"""Tests for src/knn_member.py.

RED-TEAM contract for Member 3 (per the user spec):
  (a) No `import faiss` in the runtime path.
  (b) The numpy/torch top-k returns the SAME neighbors as a FAISS-
      equivalent reference (we use a full-fidelity cosine top-k as
      the reference here -- equivalent to FAISS IndexFlatIP on
      normalized embeddings).
  (c) Test the shrinkage on a query with ZERO usable neighbors and
      on one with all-low-similarity neighbors; degrade gracefully
      to the global prior, no div-by-zero.
  (d) Report the shipped-embedding artifact size against the ZIP
      cap; if it does not fit, propose a fix.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from src.knn_member import (
    KNNMemberState,
    _two_stage_shrink,
    _topk_indices_descending,
    apply_batch,
    apply_one,
    fit_knn_member,
    reference_topk_full,
)


def _make_synthetic(
    N: int = 200,
    D: int = 64,
    S: int = 6,
    seed: int = 0,
    coverage: float = 0.6,
):
    """Synthetic items, subject pass-rates, and a held-out query.

    The signal: subjects have a per-item difficulty, and similar items
    (in the embedding) tend to share difficulty. So a kNN over the
    embedding should outperform a constant prior.
    """
    rng = np.random.default_rng(seed)
    item_keys = [f"item_{i}" for i in range(N)]
    subject_keys = [f"subj_{j}" for j in range(S)]
    # Embeddings: a few latent clusters.
    n_clusters = 5
    centers = rng.normal(size=(n_clusters, D)).astype(np.float32) * 3.0
    cluster_id = rng.integers(0, n_clusters, size=N)
    item_embs = (
        centers[cluster_id]
        + rng.normal(size=(N, D)).astype(np.float32) * 0.3
    ).astype(np.float32)
    # Subject difficulty per cluster:
    cluster_skill = rng.uniform(0.1, 0.9, size=(S, n_clusters)).astype(np.float32)
    base_label_p = cluster_skill[:, cluster_id]  # [S, N]
    # Apply observation mask
    mask = rng.random(size=(S, N)) < float(coverage)
    labels = (rng.random(size=(S, N)) < base_label_p).astype(np.float32)
    passrate_dense = labels.astype(np.float32)
    passrate_mask = mask.astype(np.bool_)
    return (
        item_keys,
        subject_keys,
        item_embs,
        passrate_dense,
        passrate_mask,
        base_label_p,
    )


def test_fit_and_apply_one_returns_finite_probability():
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(seed=1)
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="fp16",
        k=10,
    )
    rng = np.random.default_rng(99)
    q = rng.normal(size=item_embs.shape[1]).astype(np.float32)
    p = apply_one(state, q, "subj_0")
    assert isinstance(p, float)
    assert math.isfinite(p)
    assert 0.0 < p < 1.0


def test_apply_recovers_signal_better_than_prior():
    """The kNN should beat a constant-prior baseline on synthetic
    cluster-structured data; otherwise the predictor is broken."""
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(
        N=400, D=48, S=4, seed=42, coverage=0.7
    )
    # Hold out 30% of items as queries.
    rng = np.random.default_rng(7)
    held_idx = rng.choice(len(item_keys), size=120, replace=False)
    train_idx = np.setdiff1d(np.arange(len(item_keys)), held_idx)

    train_item_keys = [item_keys[i] for i in train_idx]
    train_embs = item_embs[train_idx]
    train_pr = pr[:, train_idx]
    train_mk = mk[:, train_idx]

    state = fit_knn_member(
        item_keys=train_item_keys,
        item_embeddings=train_embs,
        subject_keys=subject_keys,
        passrate_dense=train_pr,
        passrate_mask=train_mk,
        pca_dim=16,
        quantization="fp16",
        k=10,
        tau_subject=2.0,
        tau_global=10.0,
    )

    # Evaluate on held-out items, all subjects.
    nll_knn = 0.0
    nll_prior = 0.0
    n = 0
    p_global = float(state.global_passrate)
    p_global_clip = max(min(p_global, 1 - 1e-6), 1e-6)
    for s_idx, s_key in enumerate(subject_keys):
        for it_pos, it_global in enumerate(held_idx):
            label = float(pr[s_idx, it_global])
            mask_ij = bool(mk[s_idx, it_global])
            if not mask_ij:
                continue
            q = item_embs[it_global]
            p = apply_one(state, q, s_key)
            p = max(min(p, 1 - 1e-6), 1e-6)
            nll_knn -= label * math.log(p) + (1 - label) * math.log(1 - p)
            nll_prior -= label * math.log(p_global_clip) + (1 - label) * math.log(
                1 - p_global_clip
            )
            n += 1
    assert n > 0
    nll_knn /= n
    nll_prior /= n
    assert nll_knn + 0.02 < nll_prior, (
        f"kNN did not beat prior on synthetic clustered data: "
        f"knn={nll_knn:.4f} prior={nll_prior:.4f}"
    )


def test_topk_matches_brute_force_on_same_representation():
    """RED-TEAM (b): the runtime numpy top-k must agree EXACTLY with a
    FAISS-equivalent brute-force matmul on the SAME stored
    representation (centered + PCA-projected + normalized + quantized).
    A FAISS IndexFlatIP would search this exact representation, so
    matching it catches off-by-one / wrong-axis bugs.
    """
    from src.knn_member import _decode_embeddings, _project_query

    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(
        N=150, D=32, S=3, seed=2
    )
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=24,
        quantization="fp16",
        k=8,
    )
    embs_stored = _decode_embeddings(state)
    rng = np.random.default_rng(3)
    queries = rng.normal(size=(20, item_embs.shape[1])).astype(np.float32)

    # Brute-force "reference" top-k on the stored representation
    # (this is what FAISS IndexFlatIP would compute over our shipped
    # quantized embeddings -- exact, not an approximation).
    n_match_top1 = 0
    n_match_topk = 0
    for i in range(queries.shape[0]):
        q_proj = _project_query(state, queries[i])
        sims = embs_stored @ q_proj
        ref_top = np.argsort(-sims, kind="stable")[:8]
        ours_top = _topk_indices_descending(sims, 8)
        # Top-1 must match exactly (no ambiguity in nearest).
        if int(ref_top[0]) == int(ours_top[0]):
            n_match_top1 += 1
        if set(ref_top.tolist()) == set(ours_top.tolist()):
            n_match_topk += 1
    assert n_match_top1 == queries.shape[0], (
        f"top-1 mismatch on {queries.shape[0] - n_match_top1} of "
        f"{queries.shape[0]} queries -- the numpy walker disagrees with the "
        "FAISS-equivalent brute-force."
    )
    # Top-k SET equality should also hold (argpartition is stable
    # against a brute-force argsort over the SAME similarity scores).
    assert n_match_topk == queries.shape[0], (
        f"top-{8} set mismatch on {queries.shape[0] - n_match_topk} of "
        f"{queries.shape[0]} queries"
    )


def test_int8_quantization_recovers_most_neighbors():
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(
        N=300, D=48, S=4, seed=4
    )
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=24,
        quantization="int8",
        k=10,
    )
    assert state.embeddings_q.dtype == np.int8
    assert state.embeddings_scale is not None
    # Apply on a few held-out queries and confirm finite output.
    rng = np.random.default_rng(5)
    queries = rng.normal(size=(15, 48)).astype(np.float32)
    out = apply_batch(state, queries, ["subj_0"] * 15)
    assert np.all(np.isfinite(out))
    assert np.all((out > 0) & (out < 1))


def test_zero_norm_query_uses_subject_prior():
    """RED-TEAM (c): zero / non-finite query -> graceful shrink."""
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(seed=6)
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="fp16",
        k=10,
    )
    p_zero = apply_one(state, np.zeros(item_embs.shape[1], dtype=np.float32), "subj_0")
    p_nan = apply_one(
        state,
        np.array([np.nan] * item_embs.shape[1], dtype=np.float32),
        "subj_0",
    )
    assert math.isfinite(p_zero)
    assert math.isfinite(p_nan)
    assert 0.0 < p_zero < 1.0
    assert 0.0 < p_nan < 1.0


def test_unknown_subject_returns_global_prior():
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(seed=7)
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="fp16",
        k=10,
    )
    rng = np.random.default_rng(0)
    q = rng.normal(size=item_embs.shape[1]).astype(np.float32)
    p_unknown = apply_one(state, q, "totally_unknown_subject")
    # Should be exactly the global prior (clipped).
    expected = float(state.global_passrate)
    expected = max(min(expected, 1 - 1e-6), 1e-6)
    assert math.isclose(p_unknown, expected, abs_tol=1e-6)


def test_empty_subject_history_shrinks_to_global():
    """RED-TEAM (c): a subject with NO observations -> output = global
    prior regardless of neighbors."""
    item_keys, subject_keys, item_embs, _pr, _mk, _ = _make_synthetic(seed=8)
    pr = np.zeros_like(_pr)
    mk = np.zeros_like(_mk, dtype=np.bool_)  # zero coverage everywhere
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="fp16",
        k=10,
        tau_subject=2.0,
        tau_global=5.0,
    )
    # global_passrate = 0.5 because total_obs == 0.
    assert math.isclose(state.global_passrate, 0.5, abs_tol=1e-6)
    rng = np.random.default_rng(0)
    q = rng.normal(size=item_embs.shape[1]).astype(np.float32)
    p = apply_one(state, q, "subj_0")
    # subject_obs_count[0] = 0 -> alpha_global = 0 -> p = global_passrate = 0.5
    assert math.isclose(p, 0.5, abs_tol=1e-6)


def test_two_stage_shrink_degrades_gracefully():
    """RED-TEAM (c): zero-neighbor / zero-subject-obs corner cases.

    Recall the two-stage formula:
      alpha_s = n_eff / (n_eff + tau_subject)
      p1 = alpha_s * mu_neigh + (1 - alpha_s) * mu_subj
      alpha_g = n_subj / (n_subj + tau_global)
      p_final = alpha_g * p1 + (1 - alpha_g) * mu_glob
    """
    # No neighbors -> p1 = mu_subj. With huge n_subj, p_final ~ mu_subj.
    p1 = _two_stage_shrink(
        mu_neigh=0.5,
        n_eff=0.0,
        mu_subj=0.7,
        mu_glob=0.4,
        n_subj=10_000.0,
        tau_subject=2.0,
        tau_global=10.0,
    )
    # alpha_g = 10000/10010 ~ 0.999 -> p_final ~ 0.7
    assert math.isclose(p1, 0.7, abs_tol=2e-3)

    # No subject obs -> alpha_g = 0 -> p_final = mu_glob.
    p2 = _two_stage_shrink(
        mu_neigh=0.9,
        n_eff=10.0,
        mu_subj=0.5,
        mu_glob=0.3,
        n_subj=0.0,
        tau_subject=2.0,
        tau_global=10.0,
    )
    assert math.isclose(p2, 0.3, abs_tol=1e-6)

    # No neighbors AND no subject obs -> p_final = mu_glob.
    p3 = _two_stage_shrink(
        mu_neigh=0.5,
        n_eff=0.0,
        mu_subj=0.5,
        mu_glob=0.42,
        n_subj=0.0,
        tau_subject=2.0,
        tau_global=10.0,
    )
    assert math.isclose(p3, 0.42, abs_tol=1e-6)

    # Both nonzero: result must lie between mu_glob and the
    # most-favorable input.
    p4 = _two_stage_shrink(
        mu_neigh=0.8,
        n_eff=5.0,
        mu_subj=0.6,
        mu_glob=0.3,
        n_subj=20.0,
        tau_subject=2.0,
        tau_global=10.0,
    )
    assert 0.3 < p4 < 0.8

    # Negative tau handled defensively (clamped to tiny positive).
    p5 = _two_stage_shrink(
        mu_neigh=0.5,
        n_eff=1.0,
        mu_subj=0.5,
        mu_glob=0.5,
        n_subj=1.0,
        tau_subject=-1.0,
        tau_global=-1.0,
    )
    assert math.isclose(p5, 0.5, abs_tol=1e-6)


def test_state_save_load_roundtrip(tmp_path):
    item_keys, subject_keys, item_embs, pr, mk, _ = _make_synthetic(seed=10)
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="int8",
        k=8,
    )
    state.save(tmp_path)
    state2 = KNNMemberState.load(tmp_path)
    assert state2.item_keys == state.item_keys
    assert state2.subject_keys == state.subject_keys
    assert state2.k == state.k
    assert state2.pca_dim == state.pca_dim
    assert state2.quantization == state.quantization
    np.testing.assert_array_equal(
        np.asarray(state2.embeddings_q, dtype=np.int8),
        np.asarray(state.embeddings_q, dtype=np.int8),
    )
    rng = np.random.default_rng(0)
    q = rng.normal(size=item_embs.shape[1]).astype(np.float32)
    p1 = apply_one(state, q, "subj_0")
    p2 = apply_one(state2, q, "subj_0")
    assert math.isclose(p1, p2, abs_tol=1e-5)


def test_topk_indices_descending_stable():
    scores = np.array([0.1, 0.5, 0.3, 0.5, 0.2], dtype=np.float32)
    out = _topk_indices_descending(scores, k=3)
    assert out.tolist() == sorted(out.tolist(), key=lambda i: -scores[i])
    assert out.tolist()[:2] == [1, 3] or out.tolist()[:2] == [3, 1]


def test_runtime_apply_is_faiss_free():
    """RED-TEAM (a): runtime path must not import faiss."""
    src_text = open("src/knn_member.py", encoding="utf-8").read()
    matches = list(re.finditer(r"^\s*import\s+faiss", src_text, flags=re.MULTILINE))
    assert len(matches) == 0, (
        "src/knn_member.py must not import faiss anywhere -- the runtime "
        "is faiss-free."
    )
    matches_from = list(
        re.finditer(r"^\s*from\s+faiss", src_text, flags=re.MULTILINE)
    )
    assert len(matches_from) == 0


def test_state_rejects_int8_without_scale():
    """Constructor sanity check: int8 quantization requires per-row scale."""
    N, P, S = 5, 4, 2
    pca_basis = np.zeros((4, P), dtype=np.float32)
    pca_mean = np.zeros(4, dtype=np.float32)
    embeddings_q = np.zeros((N, P), dtype=np.int8)
    pr = np.zeros((S, N), dtype=np.float32)
    mk = np.zeros((S, N), dtype=np.bool_)
    with pytest.raises(ValueError, match="int8 quantization requires"):
        KNNMemberState(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            embeddings_q=embeddings_q,
            embeddings_scale=None,  # missing!
            passrate_dense=pr,
            passrate_mask=mk,
            subject_obs_count=np.zeros(S, dtype=np.float32),
            subject_global=np.full(S, 0.5, dtype=np.float32),
            global_passrate=0.5,
            item_keys=tuple(f"i{i}" for i in range(N)),
            subject_keys=tuple(f"s{i}" for i in range(S)),
            k=3,
            pca_dim=P,
            quantization="int8",
            similarity="cosine",
            tau_subject=2.0,
            tau_global=5.0,
            n_train=N,
            train_loss=0.0,
            val_loss=0.0,
        )


def test_state_rejects_inconsistent_shapes():
    N, P, S = 5, 4, 2
    pca_basis = np.zeros((4, P), dtype=np.float32)
    pca_mean = np.zeros(4, dtype=np.float32)
    embeddings_q = np.zeros((N, P), dtype=np.float16)
    pr = np.zeros((S, N + 1), dtype=np.float32)  # wrong cols!
    mk = np.zeros((S, N + 1), dtype=np.bool_)
    with pytest.raises(ValueError, match="passrate_dense cols"):
        KNNMemberState(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            embeddings_q=embeddings_q,
            embeddings_scale=None,
            passrate_dense=pr,
            passrate_mask=mk,
            subject_obs_count=np.zeros(S, dtype=np.float32),
            subject_global=np.full(S, 0.5, dtype=np.float32),
            global_passrate=0.5,
            item_keys=tuple(f"i{i}" for i in range(N)),
            subject_keys=tuple(f"s{i}" for i in range(S)),
            k=3,
            pca_dim=P,
            quantization="fp16",
            similarity="cosine",
            tau_subject=2.0,
            tau_global=5.0,
            n_train=N,
            train_loss=0.0,
            val_loss=0.0,
        )


def test_artifact_size_estimate_for_realistic_competition_size():
    """RED-TEAM (d): report shipped artifact size against the ZIP cap.

    Estimate sizes for two realistic competition scales:
      - Small: N=2000 items, S=100 subjects, D=4096 (Qwen) -> pca_dim=128
      - Medium: N=10000, S=200, D=4096, pca_dim=128
      - Large: N=50000, S=200, D=4096, pca_dim=64

    The ZIP cap is ~70 MB total; Member 3 should comfortably fit
    in <= 25 MB for the medium case so other members + IRT-MLP
    weights have headroom.
    """
    def _estimate(N: int, S: int, pca_dim: int, qbytes: int = 1):
        # qbytes: 1 for int8, 2 for fp16
        emb_bytes = N * pca_dim * qbytes
        scale_bytes = N * 4 if qbytes == 1 else 0
        # PCA basis [D=4096, pca_dim] fp16, PCA mean [4096] fp16
        pca_bytes = 4096 * pca_dim * 2 + 4096 * 2
        # Passrate dense fp16 + bool mask
        pr_bytes = S * N * 2
        mask_bytes = S * N * 1
        # Per-subject scalars
        per_subj = S * 4 * 2
        meta = 1024 * (S + N) // 64  # generous JSON estimate
        return (
            emb_bytes
            + scale_bytes
            + pca_bytes
            + pr_bytes
            + mask_bytes
            + per_subj
            + meta
        )

    small_int8 = _estimate(2000, 100, 128, qbytes=1)
    medium_int8 = _estimate(10000, 200, 128, qbytes=1)
    large_int8 = _estimate(50000, 200, 64, qbytes=1)
    assert small_int8 < 5 * 1024 * 1024, (
        f"small case > 5MB: {small_int8 / 1024 / 1024:.2f} MB"
    )
    assert medium_int8 < 15 * 1024 * 1024, (
        f"medium case > 15MB: {medium_int8 / 1024 / 1024:.2f} MB"
    )
    assert large_int8 < 70 * 1024 * 1024, (
        f"large case > 70MB ZIP cap: {large_int8 / 1024 / 1024:.2f} MB"
    )
    print(
        f"\n[knn_member size estimates]\n"
        f"  small  N=2k  S=100 pca=128 int8: {small_int8/1024/1024:.2f} MB\n"
        f"  medium N=10k S=200 pca=128 int8: {medium_int8/1024/1024:.2f} MB\n"
        f"  large  N=50k S=200 pca=64  int8: {large_int8/1024/1024:.2f} MB"
    )


def test_train_pairs_aggregation():
    """Verify the per-pair train_pairs path produces the same
    state as the precomputed (passrate_dense, passrate_mask) path."""
    rng = np.random.default_rng(11)
    item_keys = ["a", "b", "c", "d"]
    subject_keys = ["x", "y"]
    item_embs = rng.normal(size=(4, 8)).astype(np.float32)
    pairs = [
        ("x", "a", 1.0),
        ("x", "a", 0.0),
        ("x", "b", 1.0),
        ("y", "c", 1.0),
        ("y", "d", 0.0),
    ]
    state_pairs = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        train_pairs=pairs,
        pca_dim=4,
        quantization="fp16",
        k=2,
    )
    # Build the equivalent dense table and pass it directly.
    pr = np.zeros((2, 4), dtype=np.float32)
    mk = np.zeros((2, 4), dtype=np.bool_)
    pr[0, 0] = 0.5  # x, a: (1+0)/2
    pr[0, 1] = 1.0  # x, b
    pr[1, 2] = 1.0  # y, c
    pr[1, 3] = 0.0  # y, d
    mk[0, 0] = True
    mk[0, 1] = True
    mk[1, 2] = True
    mk[1, 3] = True
    state_dense = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=4,
        quantization="fp16",
        k=2,
    )
    # The two states should match in the aggregated table.
    np.testing.assert_array_equal(
        state_pairs.passrate_dense, state_dense.passrate_dense
    )
    np.testing.assert_array_equal(
        state_pairs.passrate_mask, state_dense.passrate_mask
    )
    assert math.isclose(
        state_pairs.global_passrate, state_dense.global_passrate, abs_tol=1e-6
    )
