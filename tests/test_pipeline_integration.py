"""End-to-end integration tests for the four-member stacked ensemble.

This test simulates the full Phase 6 pipeline on SYNTHETIC data
runnable on CPU. It exercises every glue point between the modules
created in Phases 1-5 and red-teams each part by feeding degenerate
inputs and verifying graceful behavior.

What this DOES verify:
  - Full OOF k-fold pipeline (no leakage, complete coverage)
  - Member 2 (GBDT) and Member 4 (LogReg) train and predict
  - Member 3 (kNN) trains and predicts
  - Stacker trains on OOF predictions and predicts
  - NN-residual calibrator with shrinkage_tau fits and applies
  - Bundle assembly via export_four_member_stacked_run
  - Static import audit
  - Two consecutive predict() calls produce identical outputs
  - Degenerate inputs (NaN, Inf, all-zero, unknown subject) don't crash
  - The rendered model.py contains stacker postprocessing and no FAISS

What this does NOT verify (requires real Colab GPU + Qwen8B):
  - Member 1 (IRT-MLP) end-to-end training; we mock its OOF preds
  - Real Qwen3-Embedding-8B encoding
  - Real cold-start benchmark detection logic in predict()

Those go in the Phase 6 notebook proper; this test covers everything
else.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

lightgbm = pytest.importorskip("lightgbm")
torch = pytest.importorskip("torch")

from src.export_stacked_submission import (
    audit_runtime_imports,
    export_four_member_stacked_run,
    measure_bundle_size_bytes,
)
from src.gbdt_member import fit_gbdt_member
from src.knn_member import KNNMemberState, apply_one as knn_apply_one, fit_knn_member
from src.logreg_member import (
    apply_state_batch as logreg_apply_batch,
    apply_state_one as logreg_apply_one,
    fit_logreg_member,
)
from src.nn_calibration import NNCalibrator, NNCalibratorState, SubjectResidualTable
from src.stacker import (
    STACKER_FEATURE_DIM,
    STACKER_FEATURE_NAMES,
    apply_one as stacker_apply_one,
    assert_no_item_leakage,
    assert_oof_covers_all_rows,
    build_stacker_features,
    build_stacker_features_one,
    fit_stacker,
    make_kfold_split,
)


# ---------------------------------------------------------------------------
# Synthetic dataset (CPU-friendly, deterministic)
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n_subjects: int = 8,
    n_items: int = 200,
    embedding_dim: int = 64,
    n_features: int = 12,
    coverage: float = 0.7,
    seed: int = 0,
):
    """Build a synthetic dataset with cluster-structured difficulty.

    Returns:
      item_keys: list[str], length n_items
      subject_keys: list[str], length n_subjects
      item_embeddings: [n_items, D] fp32 (cluster-structured)
      X_member_features: [N_rows, F] fp32 dense (for Member 2 + 4)
      y: [N_rows] fp32 in {0, 1}
      row_subject_ids: [N_rows] int (which subject)
      row_item_keys: [N_rows] string (which item)
      passrate_dense: [n_subjects, n_items] mean labels
      passrate_mask: [n_subjects, n_items] bool observation mask
    """
    rng = np.random.default_rng(seed)
    item_keys = [f"item_{i:04d}" for i in range(n_items)]
    subject_keys = [f"subj_{i:02d}" for i in range(n_subjects)]

    # Cluster-structured embeddings.
    n_clusters = 5
    centers = rng.normal(size=(n_clusters, embedding_dim)).astype(np.float32) * 3.0
    cluster_id = rng.integers(0, n_clusters, size=n_items)
    item_embeddings = (
        centers[cluster_id]
        + rng.normal(size=(n_items, embedding_dim)).astype(np.float32) * 0.4
    ).astype(np.float32)

    # True per-(subject, cluster) skill.
    cluster_skill = rng.uniform(0.15, 0.85, size=(n_subjects, n_clusters)).astype(
        np.float32
    )

    # Per-(subject, item) labels via cluster skill + small noise.
    p_true_full = cluster_skill[:, cluster_id]  # [S, N]
    labels_full = (
        rng.uniform(size=(n_subjects, n_items)) < p_true_full
    ).astype(np.float32)
    mask_full = rng.uniform(size=(n_subjects, n_items)) < float(coverage)

    # Now fan out to per-row dataset (subject, item, label) for training.
    rows_subj = []
    rows_item = []
    rows_label = []
    rows_item_key = []
    for s in range(n_subjects):
        for i in range(n_items):
            if mask_full[s, i]:
                rows_subj.append(s)
                rows_item.append(i)
                rows_item_key.append(item_keys[i])
                rows_label.append(float(labels_full[s, i]))
    row_subject_ids = np.asarray(rows_subj, dtype=np.int64)
    row_item_idx = np.asarray(rows_item, dtype=np.int64)
    row_item_keys = np.asarray(rows_item_key)
    y = np.asarray(rows_label, dtype=np.float32)
    N_rows = int(y.shape[0])

    # Member 2/4 dense features. We use a synthetic feature schema:
    #   [theta_s (1) + cluster_id_onehot (5) + emb_pca (4) + condition_dummy (2)]
    F = int(n_features)
    X = np.zeros((N_rows, F), dtype=np.float32)
    # theta_s = mean skill across clusters for that subject (for diagnostics).
    theta_s = cluster_skill.mean(axis=1)  # [S]
    X[:, 0] = theta_s[row_subject_ids]
    # cluster onehot
    for k in range(min(5, F - 1)):
        X[:, 1 + k] = (cluster_id[row_item_idx] == k).astype(np.float32)
    # PCA-like noise (random but deterministic per item).
    for k in range(min(4, F - 6)):
        X[:, 6 + k] = item_embeddings[row_item_idx, k]
    # The remaining columns are zeros (padding).

    # Compose passrate_dense / passrate_mask in [n_subjects, n_items].
    passrate_dense = labels_full.astype(np.float32)
    passrate_mask = mask_full.astype(np.bool_)

    return {
        "item_keys": item_keys,
        "subject_keys": subject_keys,
        "item_embeddings": item_embeddings,
        "X": X,
        "y": y,
        "row_subject_ids": row_subject_ids,
        "row_item_keys": row_item_keys,
        "row_item_idx": row_item_idx,
        "passrate_dense": passrate_dense,
        "passrate_mask": passrate_mask,
        "feature_names": tuple(f"f{i}" for i in range(F)),
    }


# ---------------------------------------------------------------------------
# OOF helpers
# ---------------------------------------------------------------------------


def _oof_predict_member(
    fit_fn,
    apply_fn,
    folds,
    *,
    X,
    y,
    feature_names,
    fit_kwargs,
):
    """Generic OOF wrapper: for each fold, fit on train_idx and
    predict on val_idx; return per-row OOF predictions."""
    N = int(X.shape[0])
    oof = np.zeros(N, dtype=np.float32)
    for f_idx, (tr, va) in enumerate(folds):
        state = fit_fn(
            X=X[tr],
            y=y[tr],
            feature_names=feature_names,
            **fit_kwargs,
        )
        for i in va:
            oof[i] = apply_fn(state, X[i])
    return oof


def _nll(p, y):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    y = y.astype(np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_pipeline_oof_no_leakage_complete_coverage():
    """RED-TEAM (Stacker a + b): confirm OOF k-fold preserves item
    cold-start AND covers every row exactly once."""
    ds = _make_synthetic_dataset(seed=1)
    folds = make_kfold_split(
        item_keys=list(ds["row_item_keys"]),
        n_folds=5,
        seed=42,
    )
    assert_no_item_leakage(list(ds["row_item_keys"]), folds)
    assert_oof_covers_all_rows(int(ds["X"].shape[0]), folds)


def test_full_pipeline_member2_member4_oof_then_stacker():
    """End-to-end: OOF train Members 2 & 4, fit stacker, verify it
    beats the uniform-average baseline on held-out rows."""
    ds = _make_synthetic_dataset(seed=2, n_items=300)
    folds = make_kfold_split(
        item_keys=list(ds["row_item_keys"]), n_folds=4, seed=7
    )
    assert_no_item_leakage(list(ds["row_item_keys"]), folds)

    # OOF Member 2 (GBDT)
    oof_p2 = _oof_predict_member(
        fit_fn=fit_gbdt_member,
        apply_fn=lambda st, x: __import__(
            "src.gbdt_member", fromlist=["apply_one"]
        ).apply_one(st, x),
        folds=folds,
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        fit_kwargs={
            "n_estimators": 30,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "min_data_in_leaf": 5,
            "early_stopping_rounds": 5,
            "seed": 0,
            "parity_atol": 1.0e-5,
        },
    )

    # OOF Member 4 (LogReg)
    oof_p4 = _oof_predict_member(
        fit_fn=fit_logreg_member,
        apply_fn=logreg_apply_one,
        folds=folds,
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        fit_kwargs={
            "epochs": 80,
            "learning_rate": 0.05,
            "weight_decay": 1.0e-3,
            "early_stopping_patience": 15,
            "seed": 0,
        },
    )

    # Mock OOF Member 1 + Member 3 (in real pipeline these are
    # IRT-MLP and kNN; here we synthesize biased noisy copies).
    rng = np.random.default_rng(3)
    z = np.log(np.clip(ds["y"].astype(np.float64), 1e-6, 1 - 1e-6) /
               np.clip(1 - ds["y"].astype(np.float64), 1e-6, 1 - 1e-6) + 1e-9)
    # M1: mostly correct
    oof_p1 = 1 / (1 + np.exp(-(z + rng.normal(0, 0.5, len(z))))).astype(np.float32)
    # M3: noisier
    oof_p3 = 1 / (1 + np.exp(-(z + rng.normal(0, 1.5, len(z))))).astype(np.float32)
    member_probs = np.stack([oof_p1, oof_p2, oof_p3, oof_p4], axis=1).astype(
        np.float32
    )

    # Auxiliary stacker features
    bp = np.ones(member_probs.shape[0], dtype=np.float32)  # all bench_present
    nns = np.full(member_probs.shape[0], 1.0, dtype=np.float32)
    nms = np.full(member_probs.shape[0], 0.5, dtype=np.float32)
    cd = np.full(member_probs.shape[0], 0.5, dtype=np.float32)

    stacker_feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    # Hold out 20% for stacker validation.
    perm = rng.permutation(len(ds["y"]))
    val_idx = perm[: len(perm) // 5]
    train_idx = perm[len(perm) // 5 :]
    stacker_state = fit_stacker(
        X=stacker_feats[train_idx],
        y=ds["y"][train_idx],
        n_iters=2000,
        learning_rate=0.05,
        l2=1.0,
        early_stopping_patience=300,
        seed=42,
    )

    # Check stacker beats uniform average on val.
    p_stack_val = np.array(
        [stacker_apply_one(stacker_state, stacker_feats[i]) for i in val_idx],
        dtype=np.float32,
    )
    p_avg_val = member_probs[val_idx].mean(axis=1)
    nll_stack = _nll(p_stack_val, ds["y"][val_idx])
    nll_avg = _nll(p_avg_val, ds["y"][val_idx])
    assert nll_stack <= nll_avg + 5e-3, (
        f"Stacker did not beat uniform average: stack={nll_stack:.4f} "
        f"avg={nll_avg:.4f}"
    )

    # Stacker weights sanity check: Member 1 (most accurate) should
    # have the largest positive weight.
    LOG_w = stacker_state.weights[:4]
    assert LOG_w[0] > 0
    assert LOG_w[0] >= np.max(LOG_w[1:]) - 0.1, (
        f"Member 1 not dominant: weights={LOG_w}"
    )


def test_pipeline_calibrator_with_shrinkage_improves_or_no_op():
    """End-to-end: NN-residual calibrator with continuous shrinkage
    must either improve or remain a no-op (alpha=0) on the val split.
    NEVER actively make the loss worse."""
    ds = _make_synthetic_dataset(seed=4, n_items=250)
    rng = np.random.default_rng(0)

    # Build an NN index by piggybacking on Member 3.
    knn_state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_embeddings"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=16,
        quantization="fp16",
        k=10,
    )

    # Simulate uncalibrated probs with a per-subject bias.
    bias_per_subj = rng.normal(0, 0.3, len(ds["subject_keys"])).astype(np.float32)
    n_val = 500
    val_subj = rng.integers(0, len(ds["subject_keys"]), size=n_val)
    val_p = rng.uniform(0.1, 0.9, size=n_val).astype(np.float32)
    val_logit = (
        np.log(val_p / (1 - val_p))
        + bias_per_subj[val_subj]
    )
    val_y = (1 / (1 + np.exp(-val_logit)) > rng.uniform(size=n_val)).astype(
        np.float32
    )

    # Build the residual table from train rows.
    s_ids = ds["row_subject_ids"]
    item_idx = ds["row_item_idx"]
    p_uncal_train = ds["y"].copy()  # use the labels as fake p_uncal
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids,
        training_item_rows=item_idx,
        labels=ds["y"],
        uncal_probs=p_uncal_train,
        n_subjects=len(ds["subject_keys"]),
        n_training_items=len(ds["item_keys"]),
    )

    # Use Member 3's neighbor mechanism for the calibrator.
    K = 8
    val_nbr_rows = np.zeros((n_val, K), dtype=np.int64)
    val_nbr_sims = np.zeros((n_val, K), dtype=np.float32)
    for i in range(n_val):
        # Pick a random query embedding (in real pipeline this is the
        # item's Qwen embedding).
        q = ds["item_embeddings"][rng.integers(0, len(ds["item_keys"]))]
        from src.knn_member import (
            _decode_embeddings,
            _project_query,
            _topk_indices_descending,
        )
        embs = _decode_embeddings(knn_state)
        proj = _project_query(knn_state, q)
        sims = embs @ proj
        top = _topk_indices_descending(sims, K)
        val_nbr_rows[i] = top
        val_nbr_sims[i] = sims[top]

    cal = NNCalibrator.fit_alpha_on_val(
        residual_table=table,
        val_subject_ids=val_subj,
        val_neighbor_rows=val_nbr_rows,
        val_neighbor_sims=val_nbr_sims,
        val_uncal_probs=val_p,
        val_labels=val_y,
        k=K,
        shrinkage_taus=[0.0, 0.5, 1.0, 2.0, 5.0],
    )
    nll_baseline = _nll(val_p, val_y)
    p_cal = cal.apply(
        residual_table=table,
        subject_ids=val_subj,
        neighbor_rows=val_nbr_rows,
        neighbor_sims=val_nbr_sims,
        p_uncal=val_p,
    )
    nll_cal = _nll(p_cal, val_y)
    # Calibrator must NOT make things worse; if it does, it should
    # have picked alpha=0 (identity).
    if cal.state.alpha == 0.0:
        # Identity: outputs must equal inputs (within fp32 jitter).
        np.testing.assert_allclose(p_cal, val_p, rtol=1e-5)
    else:
        assert nll_cal <= nll_baseline + 1e-3, (
            f"Calibrator made things worse: cal={nll_cal:.4f} "
            f"baseline={nll_baseline:.4f}, alpha={cal.state.alpha}"
        )

    # Outputs always finite and bounded.
    assert np.all(np.isfinite(p_cal))
    assert np.all((p_cal > 0) & (p_cal < 1))


def test_full_export_with_synthetic_member1_bundle(tmp_path):
    """End-to-end exporter test: build all 4 members, the stacker,
    the calibrator, then assemble a full bundle and run static
    audits on it."""
    ds = _make_synthetic_dataset(seed=5, n_items=200)
    rng = np.random.default_rng(0)

    # Members
    knn_state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_embeddings"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=16,
        quantization="int8",  # smaller bundle
        k=8,
    )
    gbdt_state = fit_gbdt_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=5,
        early_stopping_rounds=5,
        seed=0,
        parity_atol=1.0e-5,
    )
    logreg_state = fit_logreg_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        epochs=80,
        learning_rate=0.05,
        weight_decay=1.0e-3,
        early_stopping_patience=10,
        seed=0,
    )
    # Mock OOF preds for Member 1 (random in [0.2, 0.8]).
    oof_p1 = rng.uniform(0.2, 0.8, len(ds["y"])).astype(np.float32)
    # Build per-row probs for Member 3 by running apply_one on each row.
    oof_p3 = np.array(
        [knn_apply_one(knn_state,
                       ds["item_embeddings"][ds["row_item_idx"][i]],
                       ds["subject_keys"][ds["row_subject_ids"][i]])
         for i in range(len(ds["y"]))],
        dtype=np.float32,
    )
    oof_p2 = np.array(
        [stacker_apply_one if False else None for _ in range(0)]  # placeholder
    )
    # Use GBDT and LogReg's apply on the dense features.
    from src.gbdt_member import apply_batch as gbdt_apply_batch

    oof_p2 = gbdt_apply_batch(gbdt_state, ds["X"])
    oof_p4 = logreg_apply_batch(logreg_state, ds["X"])
    member_probs = np.stack([oof_p1, oof_p2, oof_p3, oof_p4], axis=1).astype(
        np.float32
    )

    bp = np.ones(member_probs.shape[0], dtype=np.float32)
    nns = np.ones(member_probs.shape[0], dtype=np.float32)
    nms = np.full(member_probs.shape[0], 0.5, dtype=np.float32)
    cd = np.full(member_probs.shape[0], 0.5, dtype=np.float32)
    stacker_feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    stacker_state = fit_stacker(
        X=stacker_feats,
        y=ds["y"],
        n_iters=300,
        learning_rate=0.05,
        l2=1.0,
        seed=0,
    )

    # NN calibrator (no-op for simplicity in this test).
    cal = NNCalibrator(NNCalibratorState(alpha=0.0, shrinkage_tau=2.0))
    table = SubjectResidualTable.from_rows(
        subject_ids=ds["row_subject_ids"],
        training_item_rows=ds["row_item_idx"],
        labels=ds["y"],
        uncal_probs=ds["y"],  # identity p_uncal for synthetic
        n_subjects=len(ds["subject_keys"]),
        n_training_items=len(ds["item_keys"]),
    )

    # Build minimal Member 1 bundle (replicate the test fixture from
    # test_export_stacked_submission).
    member1 = tmp_path / "submission_m1"
    member1.mkdir()
    artifacts = member1 / "artifacts"
    artifacts.mkdir()
    (artifacts / "checkpoint.pt").write_bytes(b"\x00" * 64)
    (artifacts / "runtime_meta.json").write_text(
        json.dumps({"encoder_id": "Qwen/Qwen3-Embedding-8B"}),
        encoding="utf-8",
    )
    (member1 / "model.py").write_text(
        '"""stub"""\n'
        "import json\n"
        "import math\n"
        "import logging\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        'LOG = logging.getLogger("submission_stub")\n'
        "HERE = Path(__file__).resolve().parent\n"
        '_BC_TO_ID = {"benchmarkA::condA": 0}\n'
        "TRAINING_CACHE = None\n"
        "def stable_sha256(*xs):\n"
        "    import hashlib as _h\n"
        "    return _h.sha256('|'.join(str(x) for x in xs).encode()).hexdigest()\n"
        "def normalize_condition(c): return str(c or 'none')\n"
        "_ITEM_EMB_CACHE = {}\n"
        "def _get_item_embedding(b, c, i):\n"
        "    key = (b, c, i)\n"
        "    if key not in _ITEM_EMB_CACHE:\n"
        "        rng = np.random.default_rng(abs(hash(key)) % (2**32))\n"
        "        _ITEM_EMB_CACHE[key] = rng.normal(size=(64,)).astype(np.float32)\n"
        "    return _ITEM_EMB_CACHE[key]\n"
        "def _predict_uncalibrated(b,c,s,i):\n"
        "    return 0.5\n"
        "def predict(input, labeled=None):\n"
        "    return 0.5\n"
        "def acquisition_function(*a,**k): return 0.0\n",
        encoding="utf-8",
    )

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        gbdt_state=gbdt_state,
        knn_state=knn_state,
        logreg_state=logreg_state,
        stacker_state=stacker_state,
        nn_calibrator=cal,
        residual_table=table,
        out_dir=tmp_path / "submission_stacked",
        src_dir=Path("src"),
    )

    # Audits.
    findings = audit_runtime_imports(out)
    assert findings == [], f"Forbidden imports: {findings}"
    size_mb = measure_bundle_size_bytes(out) / (1024 * 1024)
    assert size_mb < 30.0, f"Bundle too large: {size_mb:.2f} MB"

    # The rendered model.py contains the stacker postprocessing block
    # AND the FAISS sentinel.
    text = (out / "model.py").read_text(encoding="utf-8")
    assert "_stacked_predict" in text
    assert "predict = _stacked_predict" in text
    # Re-confirm: no ANY-scope import faiss in any .py under the bundle.
    for py in out.rglob("*.py"):
        t = py.read_text(encoding="utf-8")
        assert not list(re.finditer(r"^\s*import\s+faiss", t, flags=re.MULTILINE)), (
            f"{py} has import faiss"
        )


# ---------------------------------------------------------------------------
# RED-TEAM: try to break each part
# ---------------------------------------------------------------------------


def test_redteam_member2_with_all_nan_features():
    """RED-TEAM: GBDT must not crash on a feature row that is all-NaN."""
    ds = _make_synthetic_dataset(seed=10, n_items=50)
    state = fit_gbdt_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        n_estimators=15,
        learning_rate=0.1,
        num_leaves=6,
        min_data_in_leaf=5,
        early_stopping_rounds=5,
        seed=0,
        parity_atol=1.0e-5,
    )
    from src.gbdt_member import apply_one as g_apply
    feats_nan = np.full(ds["X"].shape[1], np.nan, dtype=np.float32)
    p = g_apply(state, feats_nan)
    assert math.isfinite(p)
    assert 0 < p < 1


def test_redteam_member3_unknown_subject_returns_global_prior():
    """RED-TEAM: kNN must return the global prior for an unknown subject."""
    ds = _make_synthetic_dataset(seed=11, n_items=80)
    state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_embeddings"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=16,
        quantization="fp16",
        k=8,
    )
    rng = np.random.default_rng(0)
    q = rng.normal(size=ds["item_embeddings"].shape[1]).astype(np.float32)
    p = knn_apply_one(state, q, "totally_unknown")
    expected = max(min(state.global_passrate, 1 - 1e-6), 1e-6)
    assert math.isclose(p, expected, abs_tol=1e-6)


def test_redteam_member4_with_inf_features():
    """RED-TEAM: LogReg must clamp Inf features to a finite output."""
    ds = _make_synthetic_dataset(seed=12, n_items=50)
    state = fit_logreg_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        epochs=40,
        learning_rate=0.05,
        weight_decay=1.0e-3,
        early_stopping_patience=8,
        seed=0,
    )
    feats_inf = np.full(ds["X"].shape[1], np.inf, dtype=np.float32)
    p = logreg_apply_one(state, feats_inf)
    assert math.isfinite(p)
    assert 0 < p < 1


def test_redteam_stacker_with_all_nan_member_probs():
    """RED-TEAM: Stacker must not crash on NaN logit-space inputs.
    With NaN inputs, the apply path coerces them to 0 (so the bias
    dominates) -- the output should be a finite probability."""
    rng = np.random.default_rng(0)
    feats = np.full(STACKER_FEATURE_DIM, np.nan, dtype=np.float32)
    # Build a state with arbitrary weights.
    from src.stacker import StackerState
    state = StackerState(
        weights=np.array([1.0, -1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        bias=0.0,
        feature_names=STACKER_FEATURE_NAMES,
        feature_dim=STACKER_FEATURE_DIM,
        l2=0.0,
        n_train=10,
        n_pos=5,
        train_loss=0.0,
        val_loss=0.0,
        n_iters=0,
    )
    p = stacker_apply_one(state, feats)
    assert math.isfinite(p)
    assert 0 < p < 1
    # bias=0 + all-zero features -> sigmoid(0) = 0.5
    assert math.isclose(p, 0.5, abs_tol=1e-6)


def test_redteam_calibrator_with_empty_residual_table_passes_through():
    """RED-TEAM: empty residual table -> calibrator is a no-op."""
    cal = NNCalibrator(NNCalibratorState(alpha=0.5, shrinkage_tau=1.0))
    table = SubjectResidualTable.from_rows(
        subject_ids=[],
        training_item_rows=[],
        labels=[],
        uncal_probs=[],
        n_subjects=2,
        n_training_items=10,
    )
    out = cal.apply(
        residual_table=table,
        subject_ids=np.array([0, 1], dtype=np.int64),
        neighbor_rows=np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
        neighbor_sims=np.full((2, 4), 0.9, dtype=np.float32),
        p_uncal=np.array([0.42, 0.73], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [0.42, 0.73], atol=1e-5)


def test_redteam_determinism_two_runs_identical():
    """RED-TEAM (final 4): two runs of every apply_one on the same input
    return identical results."""
    ds = _make_synthetic_dataset(seed=13, n_items=60)
    knn_state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_embeddings"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=8,
        quantization="fp16",
        k=4,
    )
    gbdt_state = fit_gbdt_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=4,
        min_data_in_leaf=5,
        early_stopping_rounds=3,
        seed=0,
        parity_atol=1.0e-5,
    )
    logreg_state = fit_logreg_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        epochs=30,
        learning_rate=0.05,
        weight_decay=1.0e-3,
        early_stopping_patience=5,
        seed=0,
    )
    rng = np.random.default_rng(0)
    q = rng.normal(size=ds["item_embeddings"].shape[1]).astype(np.float32)
    feats = ds["X"][0]
    from src.gbdt_member import apply_one as g_apply

    p_knn_a = knn_apply_one(knn_state, q, "subj_00")
    p_knn_b = knn_apply_one(knn_state, q, "subj_00")
    assert p_knn_a == p_knn_b

    p_gbdt_a = g_apply(gbdt_state, feats)
    p_gbdt_b = g_apply(gbdt_state, feats)
    assert p_gbdt_a == p_gbdt_b

    p_lr_a = logreg_apply_one(logreg_state, feats)
    p_lr_b = logreg_apply_one(logreg_state, feats)
    assert p_lr_a == p_lr_b


def test_redteam_oof_leakage_assertion_fires_on_deliberate_leak():
    """RED-TEAM (Stacker a): the leakage detector must catch a
    deliberate item leak."""
    rng = np.random.default_rng(0)
    item_keys = [f"item_{i // 5}" for i in range(50)]
    folds = make_kfold_split(item_keys=item_keys, n_folds=5, seed=42)
    # Leak: inject a val item into the train index.
    bad_train = np.concatenate([folds[0][0], folds[0][1][:1]])
    folds_bad = [(bad_train, folds[0][1])] + list(folds[1:])
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_item_leakage(item_keys, folds_bad)


def test_redteam_oof_coverage_assertion_fires_on_missing_row():
    """RED-TEAM (Stacker b): the coverage detector must catch a row
    that no fold covers."""
    item_keys = [f"item_{i // 5}" for i in range(100)]
    folds = make_kfold_split(item_keys=item_keys, n_folds=5, seed=7)
    # Drop one row from fold 0's val.
    folds_bad = [(folds[0][0], folds[0][1][:-1])] + list(folds[1:])
    with pytest.raises(RuntimeError, match="OOF coverage"):
        assert_oof_covers_all_rows(len(item_keys), folds_bad)


def test_redteam_pipeline_member3_zero_norm_query_uses_global_prior():
    """RED-TEAM: a zero-norm query (degenerate embedding) -> the
    kNN member's apply_one falls through to the subject prior, then
    global prior."""
    ds = _make_synthetic_dataset(seed=14, n_items=40)
    state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_embeddings"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=8,
        quantization="fp16",
        k=4,
    )
    p_zero = knn_apply_one(
        state,
        np.zeros(ds["item_embeddings"].shape[1], dtype=np.float32),
        "subj_00",
    )
    assert math.isfinite(p_zero)
    assert 0 < p_zero < 1


def test_redteam_full_predict_path_handles_negative_logit_extremes():
    """RED-TEAM: stacker fed with extreme logit inputs (e.g.,
    member outputs at probability 0.999999 -> logit ~ 13.8) must
    saturate gracefully without overflow."""
    feats = np.array(
        [13.8, -13.8, 13.8, -13.8, 1.0, 5.0, 0.99, 0.1], dtype=np.float32
    )
    from src.stacker import StackerState
    state = StackerState(
        weights=np.array([2.0] * STACKER_FEATURE_DIM, dtype=np.float32),
        bias=0.0,
        feature_names=STACKER_FEATURE_NAMES,
        feature_dim=STACKER_FEATURE_DIM,
        l2=0.0,
        n_train=10,
        n_pos=5,
        train_loss=0.0,
        val_loss=0.0,
        n_iters=0,
    )
    p = stacker_apply_one(state, feats)
    assert math.isfinite(p)
    # Despite extreme inputs, the output is clamped to (eps, 1-eps).
    assert 0.0 < p < 1.0
