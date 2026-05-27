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
from src.gbdt_member import fit_gbdt_member
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

    # Member 2: GBDT on synthetic features.
    F = 12
    feature_names = tuple(f"f{i}" for i in range(F))
    X = rng.normal(size=(400, F)).astype(np.float32)
    y_signal = (
        2.0 * X[:, 0] * (X[:, 1] > 0).astype(np.float32) + 0.5 * X[:, 2]
    )
    p_true = 1 / (1 + np.exp(-y_signal.astype(np.float64)))
    y = (rng.uniform(size=400) < p_true).astype(np.float32)
    gbdt_state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=feature_names,
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        seed=1,
        parity_atol=1.0e-5,
    )

    # Member 4: LogReg on the same features.
    logreg_state = fit_logreg_member(
        X=X,
        y=y,
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

    return gbdt_state, knn_state, logreg_state, stacker_state, cal, rt


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
    gbdt_state, knn_state, logreg_state, stacker_state, cal, rt = states

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        gbdt_state=gbdt_state,
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
    assert (out / "artifacts" / "member2_gbdt" / "trees.npz").exists()
    assert (out / "artifacts" / "member2_gbdt" / "meta.json").exists()
    assert (out / "artifacts" / "member3_knn" / "knn_state.npz").exists()
    assert (out / "artifacts" / "member3_knn" / "knn_meta.json").exists()
    assert (out / "artifacts" / "member4_logreg" / "weights.npz").exists()
    assert (out / "artifacts" / "member4_logreg" / "meta.json").exists()
    assert (out / "artifacts" / "stacker" / "stacker_state.npz").exists()
    assert (out / "artifacts" / "stacker" / "stacker_meta.json").exists()
    assert (out / "artifacts" / "nn_calibrator_stacked" / "state.json").exists()
    # Empty residual table is still saved.
    assert (out / "artifacts" / "residual_table" / "meta.json").exists()


def test_export_ships_subject_mean_table_when_provided(tmp_path):
    """Task 3: when subject_mean_table is passed, the exporter must
    materialize artifacts/subject_mean/{subject_mean.npy, meta.json}
    AND the runtime template must reference the table-aware anchor
    selection block.

    When subject_mean_table=None, the artifacts dir must NOT exist and
    the template still works (legacy: Member 1 anchor)."""
    from src.subject_mean import fit_subject_mean_table

    member1 = _build_minimal_member1_bundle(tmp_path)
    states = _make_synthetic_states(tmp_path)
    sm_table = fit_subject_mean_table(
        subject_ids=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        labels=np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        n_subjects=3,
        smoothing=5.0,
    )

    out_with = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        gbdt_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=states[3], nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_with_sm",
        src_dir=Path("src"),
        subject_mean_table=sm_table,
    )
    sm_dir = out_with / "artifacts" / "subject_mean"
    assert sm_dir.exists()
    assert (sm_dir / "subject_mean.npy").exists()
    assert (sm_dir / "subject_obs_count.npy").exists()
    assert (sm_dir / "meta.json").exists()
    meta = json.loads((sm_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_subjects"] == 3
    assert math.isclose(meta["smoothing"], 5.0, abs_tol=1e-9)
    assert 0.0 <= meta["global_mean"] <= 1.0
    # Re-load the saved .npy and confirm values round-trip.
    saved = np.load(sm_dir / "subject_mean.npy")
    np.testing.assert_allclose(saved, sm_table.subject_mean, rtol=1e-10)

    # Runtime template must include the table-aware anchor branch.
    txt = (out_with / "model.py").read_text(encoding="utf-8")
    assert "_SUBJECT_MEAN_TABLE" in txt
    assert "_SUBJECT_MEAN_GLOBAL" in txt
    # The Task-3 anchor selection logic must be present.
    assert "subject_mean[subject_id]" in txt or "_SUBJECT_MEAN_TABLE[_s_id]" in txt

    # And: without the table, the artifacts dir must NOT exist and the
    # template should still load (the legacy Member 1 anchor branch
    # kicks in via _SUBJECT_MEAN_TABLE is None).
    second_root = tmp_path / "second_run"
    second_root.mkdir()
    member1_b = _build_minimal_member1_bundle(second_root)
    out_without = export_four_member_stacked_run(
        member1_bundle_dir=member1_b,
        gbdt_state=states[0], knn_state=states[1], logreg_state=states[2],
        stacker_state=states[3], nn_calibrator=states[4], residual_table=states[5],
        out_dir=tmp_path / "submission_no_sm",
        src_dir=Path("src"),
        subject_mean_table=None,
    )
    assert not (out_without / "artifacts" / "subject_mean").exists()
    # Template still loads the table-aware block (with fallback to None).
    txt2 = (out_without / "model.py").read_text(encoding="utf-8")
    assert "_SUBJECT_MEAN_TABLE" in txt2
    # Fallback comment must be present so future readers know it's intentional.
    assert "legacy / pre-Task-3" in txt2 or "fall back" in txt2


def test_export_strips_faiss_and_appends_stacker_block(tmp_path):
    member1 = _build_minimal_member1_bundle(tmp_path)
    gbdt_state, knn_state, logreg_state, stacker_state, cal, rt = _make_synthetic_states(tmp_path)

    out = export_four_member_stacked_run(
        member1_bundle_dir=member1,
        gbdt_state=gbdt_state,
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
            gbdt_state=states[0],
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
        gbdt_state=states[0],
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
            gbdt_state=states[0],
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
