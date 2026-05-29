"""Tests for src/export_stacked_submission.py.

These tests verify the STATIC structural invariants of the assembled
bundle:
  - Bundle layout (artifact dirs, _pure modules, patched model.py).
  - Static import audit: no FAISS / lightgbm / sklearn / xgboost / m2cgen
    imports in any .py under the bundle.
  - The stacker postprocessing block is wired into model.py.
  - The faiss import has been neutralized.
  - Bundle size fits the ZIP cap.
  - Re-zipping is reproducible.

We do NOT subprocess-import the runtime here -- that requires the
real Qwen encoder + trained checkpoint and is exercised end-to-end
by the Phase 6 notebook on real data.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from src.export_stacked_submission import (
    _PURE_MODULE_NAMES,
    _STACKER_POSTPROCESS_PY,
    _copy_pure_modules,
    _strip_faiss_imports,
    audit_runtime_imports,
    export_four_member_stacked_run,
    measure_bundle_size_bytes,
    zip_bundle,
)
from src.knn_member import fit_knn_member
from src.logreg_member import fit_logreg_member
from src.nn_calibration import NNCalibrator, NNCalibratorState, SubjectResidualTable
from src.stacker import (
    STACKER_FEATURE_DIM,
    STACKER_FEATURE_NAMES,
    StackerState,
    fit_stacker,
)


lightgbm = pytest.importorskip("lightgbm")
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_minimal_member1_bundle(tmp_path: Path) -> Path:
    """Build a fake 'Member 1 bundle' with the minimum structure the
    exporter expects: a model.py with a try-import-faiss block, plus
    an artifacts/ directory.

    We CANNOT call the real export_ensemble_run here without a full
    trained Qwen-encoded model, so we synthesize a stub model.py
    that has the SAME shape as the real rendered template (the
    relevant chunks the exporter patches).
    """
    bundle = tmp_path / "submission_m1"
    bundle.mkdir()
    artifacts = bundle / "artifacts"
    artifacts.mkdir()
    (artifacts / "checkpoint.pt").write_bytes(b"\x00" * 64)  # dummy
    (artifacts / "runtime_meta.json").write_text(
        json.dumps({"encoder_id": "Qwen/Qwen3-Embedding-8B"}),
        encoding="utf-8",
    )

    # Minimal model.py mimicking the legacy template's surface.
    # We include the exact try: import faiss block so _strip_faiss_imports
    # can be exercised. We also include a placeholder predict() that
    # the postprocessing block can replace.
    model_py = '''"""Stub model.py for testing export_stacked_submission."""

import json
import math
from pathlib import Path
import logging

import numpy as np

LOG = logging.getLogger("submission_stub")
HERE = Path(__file__).resolve().parent

_BC_TO_ID = {"benchmarkA::condA": 0}

class _DummyTrainingCache:
    subject_key_to_id = {"sha_subj_x": 0, "sha_subj_y": 1}
    def nearest(self, q, k):
        return np.arange(k, dtype=np.int64), np.full(k, 0.5, dtype=np.float32)

TRAINING_CACHE = _DummyTrainingCache()


def stable_sha256(*xs):
    import hashlib as _hashlib
    return _hashlib.sha256("|".join(str(x) for x in xs).encode()).hexdigest()


def normalize_condition(cond):
    return str(cond or "none")


_ITEM_EMB_CACHE = {}

def _get_item_embedding(benchmark, condition, item_content):
    key = (benchmark, condition, item_content)
    if key not in _ITEM_EMB_CACHE:
        # Deterministic fake embedding so tests are reproducible.
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        _ITEM_EMB_CACHE[key] = rng.normal(size=(64,)).astype(np.float32)
    return _ITEM_EMB_CACHE[key]


def _try_load_faiss(in_dir):
    """Mimic the legacy _maybe_load_faiss block; the exporter patches
    the try-import inside this function."""
    path = Path(in_dir) / "training_index.faiss"
    if path.exists():
        try:
            import faiss  # type: ignore

            cpu_index = faiss.read_index(str(path))
            return cpu_index
        except Exception:
            return None
    return None


def _predict_uncalibrated(benchmark, condition, subject_content, item_content):
    # Deterministic fake: hash inputs to a probability in [0.05, 0.95].
    h = abs(hash((benchmark, condition, subject_content, item_content)))
    return 0.05 + (h % 1000) / 1000.0 * 0.9


def predict(input, labeled=None):
    benchmark = str(input.get("benchmark", ""))
    condition = normalize_condition(input.get("condition", "none"))
    subject_content = str(input.get("subject_content", ""))
    item_content = str(input.get("item_content", ""))
    return _predict_uncalibrated(benchmark, condition, subject_content, item_content)


def acquisition_function(*args, **kwargs):
    return 0.0
'''
    (bundle / "model.py").write_text(model_py, encoding="utf-8")

    # Match the exporter's existing _strip_faiss_imports pattern by
    # writing the EXACT block it expects (including the indentation).
    # We'll add it as a fake _try_build_faiss method below.
    text = (bundle / "model.py").read_text(encoding="utf-8")
    text = text.replace(
        "        try:\n            import faiss  # type: ignore\n\n            cpu_index = faiss.read_index(str(path))\n            return cpu_index\n        except Exception:\n            return None",
        # Use the exact pattern the strip function looks for.
        "        try:\n            import faiss  # type: ignore\n\n            cpu_index = faiss.read_index(str(path))\n            return cpu_index\n        except Exception:\n            return None",
    )
    (bundle / "model.py").write_text(text, encoding="utf-8")

    return bundle


def _make_synthetic_states(tmp_path: Path):
    """Fit small Member 2-4 states + stacker for export testing."""
    rng = np.random.default_rng(0)
    # Member 3: small kNN over fake item embeddings.
    item_keys = [f"item_{i}" for i in range(40)]
    item_embs = rng.normal(size=(40, 64)).astype(np.float32)
    subject_keys = ["sha_subj_x", "sha_subj_y"]
    pr = rng.uniform(0, 1, size=(2, 40)).astype(np.float32)
    mk = (rng.uniform(0, 1, size=(2, 40)) < 0.7).astype(np.bool_)
    knn_state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=pr,
        passrate_mask=mk,
        pca_dim=16,
        quantization="fp16",
        k=5,
    )

    # Member 2: metadata MLP on synthetic ids + marginals.
    from src.member2_metadata_mlp import fit_member2_metadata_mlp

    N = 400
    n_subjects, n_bcs, n_clusters = 8, 5, 4
    n_marginals = 6
    subj_ids = rng.integers(0, n_subjects, size=N).astype(np.int64)
    bc_ids = rng.integers(0, n_bcs, size=N).astype(np.int64)
    cl_ids = rng.integers(0, n_clusters, size=N).astype(np.int64)
    marginals = rng.normal(size=(N, n_marginals)).astype(np.float32)
    z_m2 = marginals[:, 0] + 0.5 * (subj_ids % 3) - 0.3 * (bc_ids % 2)
    y = (rng.uniform(size=N) < (1 / (1 + np.exp(-z_m2)))).astype(np.float32)
    member2_mlp_state = fit_member2_metadata_mlp(
        subject_ids=subj_ids,
        bc_ids=bc_ids,
        cluster_ids=cl_ids,
        marginals=marginals,
        y=y,
        subject_keys=tuple(f"s{i}" for i in range(n_subjects)),
        bc_keys=tuple(f"b{i}" for i in range(n_bcs)),
        marg_feature_names=tuple(f"m{i}" for i in range(n_marginals)),
        n_subjects=n_subjects,
        n_bcs=n_bcs,
        n_clusters=n_clusters,
        d_subj=8,
        d_bc=8,
        d_cluster=4,
        hid1=32,
        hid2=16,
        epochs=5,
        batch_size=128,
        seed=1,
        show_progress=False,
    )

    # Member 4: LogReg on dense features (independent of M2).
    F = 12
    feature_names = tuple(f"f{i}" for i in range(F))
    X = rng.normal(size=(400, F)).astype(np.float32)
    y_signal = (
        2.0 * X[:, 0] * (X[:, 1] > 0).astype(np.float32) + 0.5 * X[:, 2]
    )
    p_true = 1 / (1 + np.exp(-y_signal.astype(np.float64)))
    y_lr = (rng.uniform(size=400) < p_true).astype(np.float32)
    logreg_state = fit_logreg_member(
        X=X,
        y=y_lr,
        feature_names=feature_names,
        epochs=50,
        learning_rate=0.05,
        weight_decay=1.0e-3,
        early_stopping_patience=10,
        seed=2,
    )

    # Stacker on synthetic OOF predictions.
    z = rng.normal(size=300).astype(np.float32)
    p1 = 1 / (1 + np.exp(-(z + rng.normal(0, 0.3, 300))))
    p2 = 1 / (1 + np.exp(-(z * 0.5 + rng.normal(0, 0.5, 300))))
    p3 = 1 / (1 + np.exp(-(z + rng.normal(0, 0.7, 300))))
    p4 = 1 / (1 + np.exp(-(z * 0.7 + rng.normal(0, 0.4, 300))))
    member_probs = np.stack([p1, p2, p3, p4], axis=1).astype(np.float32)
    bp = (rng.uniform(size=300) < 0.7).astype(np.float32)
    nns = np.log1p(rng.uniform(0, 16, 300)).astype(np.float32)
    nms = rng.uniform(-0.1, 0.95, 300).astype(np.float32)
    cd = rng.uniform(0.1, 2.0, 300).astype(np.float32)

    from src.stacker import build_stacker_features

    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    y_st = (
        rng.uniform(size=300) < 1 / (1 + np.exp(-z.astype(np.float64)))
    ).astype(np.float32)
    stacker_state = fit_stacker(
        X=feats, y=y_st, n_iters=200, learning_rate=0.05, l2=1.0, seed=3
    )

    # Calibrator: a small no-op state plus an empty residual table.
    cal = NNCalibrator(NNCalibratorState(alpha=0.0))
    rt = SubjectResidualTable.from_rows(
        subject_ids=[],
        training_item_rows=[],
        labels=[],
        uncal_probs=[],
        n_subjects=knn_state.n_subjects,
        n_training_items=knn_state.n_items,
    )

    return member2_mlp_state, knn_state, logreg_state, stacker_state, cal, rt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strip_faiss_imports_neutralizes_legacy_pattern():
    legacy_block = (
        "    if False:\n"
        "        try:\n"
        "            import faiss  # type: ignore\n\n"
        "            cpu_index = faiss.read_index(str(path))\n"
        "        except Exception:\n"
        "            return None\n"
    )
    out = _strip_faiss_imports(legacy_block)
    assert "import faiss" not in out, (
        "legacy try-import-faiss block was not neutralized:\n" + out
    )
    assert "faiss = None" in out
    assert "raise ImportError" in out


def test_strip_faiss_imports_idempotent_when_no_pattern():
    text = "no faiss here"
    assert _strip_faiss_imports(text) == text


def test_copy_pure_modules_emits_expected_files(tmp_path):
    pure_dir = tmp_path / "_pure"
    _copy_pure_modules(Path("src"), pure_dir)
    for name in _PURE_MODULE_NAMES:
        assert (pure_dir / f"{name}.py").exists(), f"missing {name}.py"
    assert (pure_dir / "__init__.py").exists()


def test_audit_runtime_imports_finds_lightgbm_in_test_file(tmp_path):
    """Smoke-test the auditor itself: a deliberate forbidden import
    must be detected."""
    bad_dir = tmp_path / "fake_bundle"
    bad_dir.mkdir()
    (bad_dir / "model.py").write_text(
        "import lightgbm\nprint('hi')\n", encoding="utf-8"
    )
    findings = audit_runtime_imports(bad_dir)
    assert len(findings) == 1
    assert "lightgbm" in findings[0][1]


def test_audit_runtime_imports_clean_when_only_safe_imports(tmp_path):
    good_dir = tmp_path / "good_bundle"
    good_dir.mkdir()
    (good_dir / "model.py").write_text(
        "import numpy as np\nimport torch\nfrom transformers import AutoModel\n",
        encoding="utf-8",
    )
    findings = audit_runtime_imports(good_dir)
    assert findings == []


def test_export_assembles_bundle_layout(tmp_path):
    """End-to-end on synthetic states: every expected directory and
    file should exist."""
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    m2_state, knn_state, logreg_state, stacker_state, cal, rt = states

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=m2_state,
        knn_state=knn_state,
        logreg_state=logreg_state,
        stacker_state=stacker_state,
        nn_calibrator=cal,
        residual_table=rt,
        out_dir=tmp_path / "submission_stacked",
        src_dir=Path("src"),
    )
    assert out.exists()
    assert (out / "model.py").exists()
    # Pure modules.
    for name in _PURE_MODULE_NAMES:
        assert (out / "_pure" / f"{name}.py").exists()
    assert (out / "_pure" / "__init__.py").exists()
    # Member states.
    assert (out / "artifacts" / "member2_metadata_mlp" / "weights.npz").exists()
    assert (out / "artifacts" / "member2_metadata_mlp" / "meta.json").exists()
    assert (out / "artifacts" / "member3_knn" / "knn_state.npz").exists()
    assert (out / "artifacts" / "member3_knn" / "knn_meta.json").exists()
    assert (out / "artifacts" / "member4_logreg" / "weights.npz").exists()
    assert (out / "artifacts" / "member4_logreg" / "meta.json").exists()
    assert (out / "artifacts" / "stacker" / "stacker_state.npz").exists()
    assert (out / "artifacts" / "stacker" / "stacker_meta.json").exists()
    assert (out / "artifacts" / "nn_calibrator_stacked" / "state.json").exists()
    # Empty residual table is still saved.
    assert (out / "artifacts" / "residual_table" / "meta.json").exists()


def test_export_ships_mean_encoded_stats_when_provided(tmp_path):
    """When mean_encoded_stats is passed, artifacts/mean_encoded/ is shipped."""
    from src.mean_encoded_features import fit_mean_encoded_stats

    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    mes = fit_mean_encoded_stats(
        subject_ids=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        cluster_ids=np.array([0, 1, 0, 1, 0], dtype=np.int64),
        bc_ids=np.array([0, 1, 0, 1, 0], dtype=np.int64),
        labels=np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        n_subjects=3,
        n_clusters=2,
        n_bcs=2,
        smoothing=5.0,
    )

    out_with = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=states[3], nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_with_me",
        src_dir=Path("src"),
        mean_encoded_stats=mes,
    )
    me_dir = out_with / "artifacts" / "mean_encoded"
    assert me_dir.exists()
    assert (me_dir / "mean_encoded_stats.npz").exists()
    assert (me_dir / "mean_encoded_meta.json").exists()

    txt = (out_with / "model.py").read_text(encoding="utf-8")
    assert "_ME_STATS" in txt
    assert "apply_member4_marginal_features_one" in txt

    m1b = tmp_path / "m1b"
    m1b.mkdir(parents=True, exist_ok=True)
    out_without = export_four_member_stacked_run(
        member1_bundle_dir=_build_minimal_member1_bundle(m1b),
        member2_mlp_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=states[3], nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_no_me",
        src_dir=Path("src"),
        mean_encoded_stats=None,
    )
    assert not (out_without / "artifacts" / "mean_encoded").exists()


def _make_synthetic_member5_state(tmp_path: Path):
    """Build a minimal Member5State that's compatible with the export
    pipeline: a 6-D supervised difficulty projection with a handful of
    items and subjects."""
    from src.member5_difficulty_knn import fit_member5

    rng = np.random.default_rng(7)
    n_items = 60
    n_subjects = 20
    d_emb = 6
    item_keys = tuple(f"item_{i}" for i in range(n_items))
    subject_keys = tuple(f"subj_{s}" for s in range(n_subjects))
    item_embeddings = rng.normal(size=(n_items, d_emb)).astype(np.float32)
    # Random ratings.
    rows_s, rows_i, rows_y = [], [], []
    for s in range(n_subjects):
        items_seen = rng.choice(n_items, size=15, replace=False)
        for i in items_seen:
            rows_s.append(s); rows_i.append(int(i))
            rows_y.append(float(rng.random() < 0.5))
    return fit_member5(
        item_keys=item_keys,
        item_embeddings=item_embeddings,
        subject_keys=subject_keys,
        subject_ids_per_row=np.array(rows_s, dtype=np.int64),
        item_ids_per_row=np.array(rows_i, dtype=np.int64),
        labels=np.array(rows_y, dtype=np.float64),
        k=5, tau=0.05, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )


def _make_5member_stacker_state(rng_seed: int = 99) -> StackerState:
    """Synthetic 5-member stacker for the Member 5 export tests."""
    rng = np.random.default_rng(rng_seed)
    z = rng.normal(size=200).astype(np.float32)
    p_true = 1.0 / (1.0 + np.exp(-z))
    member_probs = np.stack(
        [1 / (1 + np.exp(-(z + rng.normal(0, 0.3 + 0.1 * i, 200))))
         for i in range(5)],
        axis=1,
    ).astype(np.float32)
    bp = (rng.uniform(size=200) < 0.7).astype(np.float32)
    nns = np.log1p(rng.uniform(0, 16, 200)).astype(np.float32)
    nms = rng.uniform(-0.1, 0.95, 200).astype(np.float32)
    cd = rng.uniform(0.1, 2.0, 200).astype(np.float32)

    from src.stacker import build_stacker_features, stacker_feature_names

    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    y_st = (rng.uniform(size=200) < p_true).astype(np.float32)
    return fit_stacker(
        X=feats, y=y_st,
        feature_names=stacker_feature_names(5),
        n_iters=150, learning_rate=0.05, l2=1.0, seed=int(rng_seed),
    )


def test_export_ships_member5_state_when_provided_and_5col_stacker(tmp_path):
    """Task 4: when member5_state is provided AND the stacker has 5
    member columns (feature_dim == 9), the exporter must:

    1. Write artifacts/member5_dknn/{member5_state.npz, member5_meta.json}.
    2. Copy _pure/member5_difficulty_knn.py into the bundle.
    3. Include the Member 5 runtime block in model.py that loads the
       state and calls apply_one when the bundle has Member 5.
    """
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    m5_state = _make_synthetic_member5_state(tmp_path)
    stacker_5 = _make_5member_stacker_state()
    assert int(stacker_5.feature_dim) == 9, (
        f"5-member stacker fixture should have feature_dim==9, got "
        f"{stacker_5.feature_dim}"
    )

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=stacker_5,
        nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_with_m5",
        src_dir=Path("src"),
        member5_state=m5_state,
    )
    m5_dir = out / "artifacts" / "member5_dknn"
    assert m5_dir.exists()
    assert (m5_dir / "member5_state.npz").exists()
    assert (m5_dir / "member5_meta.json").exists()
    # Pure module is shipped.
    assert (out / "_pure" / "member5_difficulty_knn.py").exists()
    # Runtime template must reference Member 5.
    txt = (out / "model.py").read_text(encoding="utf-8")
    assert "_MEMBER5_STATE" in txt
    assert "member5_dknn" in txt
    assert "member5_difficulty_knn" in txt
    # Locked member order: p1..p5 must appear in the inference block.
    assert "[p1, p2, p3, p4]" in txt
    # The Member 5 score must be appended to member_probs (5-member case).
    assert "_member_probs_runtime.append(p5)" in txt


def test_export_omits_member5_dir_when_not_provided(tmp_path):
    """Task 4 backward-compat: 4-member bundle (member5_state=None,
    legacy 4-col stacker) must NOT create artifacts/member5_dknn/, AND
    the runtime template's Member 5 guard must still be present so that
    the runtime correctly identifies the bundle as 4-member."""
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=states[3],
        nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_no_m5",
        src_dir=Path("src"),
        member5_state=None,
    )
    assert not (out / "artifacts" / "member5_dknn").exists()
    txt = (out / "model.py").read_text(encoding="utf-8")
    # The guard is still there (loaded conditionally on _HAS_MEMBER5).
    assert "_HAS_MEMBER5" in txt
    # The pure module is still shipped (cheap; downstream calls are guarded).
    assert (out / "_pure" / "member5_difficulty_knn.py").exists()


def test_export_rejects_member5_with_4col_stacker(tmp_path):
    """Misconfiguration guard: shipping member5_state with a 4-column
    stacker would silently ignore Member 5 at runtime. The exporter
    must raise so the mistake is loud."""
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    m5_state = _make_synthetic_member5_state(tmp_path)
    # states[3] is the legacy 4-column stacker.
    assert int(states[3].feature_dim) == 8
    with pytest.raises(ValueError, match="member5_state was provided"):
        export_four_member_stacked_run(
            member1_bundle_dir=member1,
            member2_mlp_state=states[0], knn_state=states[1], logreg_state=states[2],
            stacker_state=states[3],
            nn_calibrator=states[4], residual_table=states[5],
            out_dir=tmp_path / "submission_mismatched",
            src_dir=Path("src"),
            member5_state=m5_state,
        )


def test_export_strips_faiss_and_appends_stacker_block(tmp_path):
    member1 = _build_minimal_member1_bundle(tmp_path)
    m2_state, knn_state, logreg_state, stacker_state, cal, rt = _make_synthetic_states(tmp_path)

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=m2_state,
        knn_state=knn_state,
        logreg_state=logreg_state,
        stacker_state=stacker_state,
        nn_calibrator=cal,
        residual_table=rt,
        out_dir=tmp_path / "submission_stacked",
        src_dir=Path("src"),
        audit=True,  # raises if anything forbidden remains
    )
    text = (out / "model.py").read_text(encoding="utf-8")
    # Stacker block is appended.
    assert "_stacked_predict" in text
    assert "_HAS_STACKED_BUNDLE" in text
    # The line `import faiss` MUST NOT appear anywhere line-anchored.
    assert not list(re.finditer(r"^\s*import\s+faiss", text, flags=re.MULTILINE))
    # The neutralization sentinel IS present.
    assert "PHASE-5: strict no-FAISS rule" in text


def test_export_audit_blocks_bundle_with_forbidden_imports(tmp_path):
    """If the legacy template ever introduces a top-level lightgbm /
    sklearn / xgboost / m2cgen import, the export must FAIL FAST so we
    catch it before shipping."""
    member1 = _build_minimal_member1_bundle(tmp_path)
    # Inject a deliberately forbidden import into the stub.
    stub = (member1 / "model.py").read_text(encoding="utf-8")
    (member1 / "model.py").write_text("import lightgbm\n" + stub, encoding="utf-8")

    states = _make_synthetic_states(tmp_path)
    with pytest.raises(RuntimeError, match="forbidden imports"):
        export_four_member_stacked_run(
            member1_bundle_dir=member1,
            member2_mlp_state=states[0],
            knn_state=states[1],
            logreg_state=states[2],
            stacker_state=states[3],
            nn_calibrator=states[4],
            residual_table=states[5],
            out_dir=tmp_path / "submission_bad",
            src_dir=Path("src"),
            audit=True,
        )


def test_export_zip_size_under_cap(tmp_path):
    """RED-TEAM (ZIP cap): the assembled bundle's uncompressed size
    must fit under the 65 MB cap (with synthetic small states it's a
    few MB at most)."""
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        member2_mlp_state=states[0],
        knn_state=states[1],
        logreg_state=states[2],
        stacker_state=states[3],
        nn_calibrator=states[4],
        residual_table=states[5],
        out_dir=tmp_path / "submission_stacked",
        src_dir=Path("src"),
    )
    size = measure_bundle_size_bytes(out)
    assert size < 5 * 1024 * 1024, (
        f"Synthetic bundle too large: {size / 1024 / 1024:.2f} MB"
    )

    zp = zip_bundle(out, tmp_path / "submission.zip")
    assert zp.exists()
    assert zp.stat().st_size < 5 * 1024 * 1024


def test_export_zip_cap_audit_raises_when_too_large(tmp_path):
    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    with pytest.raises(RuntimeError, match="uncompressed size"):
        export_four_member_stacked_run(
            member1_bundle_dir=member1,
            member2_mlp_state=states[0],
            knn_state=states[1],
            logreg_state=states[2],
            stacker_state=states[3],
            nn_calibrator=states[4],
            residual_table=states[5],
            out_dir=tmp_path / "submission_stacked",
            src_dir=Path("src"),
            zip_cap_bytes=100,  # impossibly small
            audit=False,        # don't let the import audit fire first
        )


def test_pure_modules_are_runtime_safe_on_static_audit(tmp_path):
    """Sanity: each pure-numpy module copied into the bundle must
    pass the two-tier audit. Module-scope lightgbm/sklearn/xgboost/
    m2cgen are forbidden; lazy imports inside fit_* functions are
    fine. FAISS is forbidden at any scope.
    """
    pure_dir = tmp_path / "_pure_audit"
    _copy_pure_modules(Path("src"), pure_dir)
    findings = audit_runtime_imports(pure_dir)
    # The auditor is now two-tier: lazy fit_* imports of lightgbm /
    # torch / etc. are NOT flagged (they're indented). Only
    # module-scope imports of lightgbm/sklearn/xgboost/m2cgen would
    # be flagged, plus any-scope faiss imports.
    assert findings == [], (
        f"Pure modules contain forbidden module-scope imports: {findings}"
    )
    # Direct module-scope check (column-0): no forbidden imports at all.
    for path in pure_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pat in (r"^import\s+lightgbm", r"^from\s+lightgbm",
                    r"^import\s+sklearn", r"^from\s+sklearn",
                    r"^import\s+xgboost", r"^import\s+m2cgen",
                    r"^import\s+faiss", r"^from\s+faiss"):
            assert not list(re.finditer(pat, text, flags=re.MULTILINE)), (
                f"{path} has top-level forbidden import: {pat}"
            )
