"""Phase 5 of the four-member stacked-ensemble upgrade.

Bundle a four-member stacked ensemble (Member 1 IRT-MLP + Member 2
GBDT + Member 3 FAISS-free kNN + Member 4 LogReg, fused by an OOF
ridge stacker and post-calibrated by a single shrinkage NN-residual
calibrator) into a Codabench submission directory.

Design
------
Rather than rewrite the legacy ``_RUNTIME_MODEL_PY`` template (which
inlines ~3000 lines of model classes, encoder loading, Indexer, NN
features, and per-batch caching), we **wrap** the existing
``export_ensemble_run`` output:

  1. Call ``export_ensemble_run`` to produce a standard
     IRT-MLP-ensemble submission bundle. This handles all the heavy
     pieces (encoder, Indexer, model classes, pool/cluster/NN features,
     per-batch caching) with code paths that are already tested.

  2. Add new artifact directories beside the ensemble checkpoint:
     ``artifacts/member2_gbdt/``, ``artifacts/member3_knn/``,
     ``artifacts/member4_logreg/``, ``artifacts/stacker/``,
     ``artifacts/nn_calibrator_stacked/``,
     ``artifacts/residual_table/``.

  3. Copy the pure-numpy runtime modules into
     ``submission/_pure/``:
       - ``gbdt_member.py``        (apply_one for Member 2)
       - ``knn_member.py``         (apply_one for Member 3)
       - ``logreg_member.py``      (apply_one for Member 4)
       - ``stacker.py``            (apply_one for the stacker)
       - ``nn_calibration.py``     (apply for the residual calibrator)
       - ``member_features.py``    (build_member_features_one)

  4. **Patch the rendered ``model.py``**:
     a. Remove the ``try: import faiss`` lines so the strict no-FAISS
        rule is satisfied (the ``_TrainingItemCache`` already has a
        brute-force fallback).
     b. Append a four-member-stacking block that:
        - Loads the Member 2-4 / stacker / calibrator / residual_table
          states at module scope.
        - Defines pure-numpy fusion helpers.
        - Wraps the existing ``predict`` with a new one that combines
          Member 1's logit (the legacy path's intermediate result)
          with the other three members' predictions, runs them through
          the stacker, and applies the post-stacker NN-residual
          calibrator once.

  5. Re-zip.

This minimizes risk by reusing all the well-tested legacy template
code for Member 1 / encoder / Indexer / NN features.

The user-spec RED-TEAM (final, whole pipeline) blocks 1-6 will be
exercised by the Phase 6 notebook on real trained weights; this
module's tests verify only the static structural invariants
(bundle layout, no forbidden imports in the rendered model.py,
ZIP-cap audit).
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping

LOG = logging.getLogger("export_stacked")

# Hard ZIP-size cap (per CODABENCH_SUBMISSION_GUIDE / RUNTIME_ENV.md).
# We target <=65 MB to leave headroom for compression variation.
_DEFAULT_ZIP_CAP_BYTES: int = int(65 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Stacker postprocessing block (appended to model.py at the end)
# ---------------------------------------------------------------------------
#
# This is a string of Python code that the exporter appends to the
# legacy-rendered ``model.py``. It loads Member 2-4 + stacker +
# calibrator state at module scope and replaces the global ``predict``
# binding with a new one that calls all four members and applies the
# stacker + calibrator.
#
# Critical contract: the new ``predict()`` MUST re-use the exact
# ``benchmark`` / ``condition`` / ``subject_content`` / ``item_content``
# extraction logic that the legacy ``predict()`` uses, so the cache
# keys / fingerprints stay coherent. We achieve this by reusing the
# legacy template's helpers (``_predict_uncalibrated``,
# ``_get_item_embedding``, ``stable_sha256``, ``normalize_condition``)
# directly -- they're already at module scope.

_STACKER_POSTPROCESS_PY = r'''
# ---------------------------------------------------------------------------
# Four-member stacked-ensemble postprocessing (Phase 5 of the upgrade).
# Loaded only when artifacts/stacker/ exists; otherwise this block is
# a no-op and the legacy single-Member-1 predict() stays in effect.
# ---------------------------------------------------------------------------

import sys as _sys

_STACKED_ROOT = HERE / "artifacts"
_PURE_DIR = HERE / "_pure"
_HAS_STACKED_BUNDLE = (
    (_STACKED_ROOT / "stacker").exists()
    and (_STACKED_ROOT / "member2_gbdt").exists()
    and (_STACKED_ROOT / "member3_knn").exists()
    and (_STACKED_ROOT / "member4_logreg").exists()
    and _PURE_DIR.exists()
)

if _HAS_STACKED_BUNDLE:
    if str(_PURE_DIR) not in _sys.path:
        _sys.path.insert(0, str(_PURE_DIR))

    # The pure-numpy modules. None of them imports torch / lightgbm /
    # faiss / sklearn at module scope; the offline-only fit functions
    # have those imports inside the function body and never fire at
    # runtime.
    import gbdt_member as _gbdt_mod
    import knn_member as _knn_mod
    import logreg_member as _logreg_mod
    import stacker as _stacker_mod
    import nn_calibration as _nn_calib_mod
    import member_features as _mfeat_mod

    LOG.info("[stacked] loading 4-member ensemble state")
    _GBDT_STATE = _gbdt_mod.GBDTMemberState.load(_STACKED_ROOT / "member2_gbdt")
    _KNN_STATE = _knn_mod.KNNMemberState.load(_STACKED_ROOT / "member3_knn")
    _LOGREG_STATE = _logreg_mod.LogRegMemberState.load(_STACKED_ROOT / "member4_logreg")
    _STACKER_STATE = _stacker_mod.StackerState.load(_STACKED_ROOT / "stacker")

    # Task 4: optional Member 5 (difficulty-projected kNN). When the
    # bundle ships an artifacts/member5_dknn/ directory AND the stacker
    # was fit with 5 member columns (feature_dim == 9), Member 5 is
    # scored and fed into the stacker. When absent OR when the stacker
    # was fit with only 4 columns (legacy/pre-Task-4 bundles), the
    # runtime stays on the 4-member path.
    _MEMBER5_DIR = _STACKED_ROOT / "member5_dknn"
    _HAS_MEMBER5 = (
        _MEMBER5_DIR.exists()
        and int(_STACKER_STATE.feature_dim) >= 9
    )
    if _HAS_MEMBER5:
        import member5_difficulty_knn as _m5_mod
        _MEMBER5_STATE = _m5_mod.Member5State.load(_MEMBER5_DIR)
        LOG.info(
            "[stacked] loaded Member 5 (difficulty-kNN): "
            "n_items=%d  n_subjects=%d  k=%d  tau=%.3f",
            int(_MEMBER5_STATE.n_items), int(_MEMBER5_STATE.n_subjects),
            int(_MEMBER5_STATE.k), float(_MEMBER5_STATE.tau),
        )
    else:
        _m5_mod = None
        _MEMBER5_STATE = None

    # Calibrator: state dict + optional residual table.
    _NN_CAL_STACKED_DIR = _STACKED_ROOT / "nn_calibrator_stacked"
    _NN_CAL_STATE_PATH = _NN_CAL_STACKED_DIR / "state.json"
    if _NN_CAL_STATE_PATH.exists():
        _NN_CAL_STATE_RAW = json.loads(_NN_CAL_STATE_PATH.read_text(encoding="utf-8"))
        _NN_CAL_STACKED = _nn_calib_mod.NNCalibrator.from_dict(_NN_CAL_STATE_RAW)
    else:
        _NN_CAL_STACKED = _nn_calib_mod.NNCalibrator(
            _nn_calib_mod.NNCalibratorState(alpha=0.0)
        )
    _RESIDUAL_TABLE_DIR = _STACKED_ROOT / "residual_table"
    if _RESIDUAL_TABLE_DIR.exists():
        _RESIDUAL_TABLE = _nn_calib_mod.SubjectResidualTable.load(_RESIDUAL_TABLE_DIR)
    else:
        # Empty residual table -> calibrator is a no-op.
        _RESIDUAL_TABLE = _nn_calib_mod.SubjectResidualTable.from_rows(
            subject_ids=[], training_item_rows=[], labels=[], uncal_probs=[],
            n_subjects=int(_KNN_STATE.n_subjects),
            n_training_items=int(_KNN_STATE.n_items),
        )

    # Task 3: optional subject_mean table for Member 2's residual anchor.
    # When present, Member 2 composes against `subject_mean[subject_id]`
    # instead of Member 1's prediction (the legacy anchor). When absent,
    # the runtime falls back to Member 1 -- safe for pre-Task-3 bundles.
    _SUBJECT_MEAN_DIR = _STACKED_ROOT / "subject_mean"
    if _SUBJECT_MEAN_DIR.exists():
        _SUBJECT_MEAN_TABLE = np.load(_SUBJECT_MEAN_DIR / "subject_mean.npy").astype(np.float64)
        _SUBJECT_MEAN_META = json.loads(
            (_SUBJECT_MEAN_DIR / "meta.json").read_text(encoding="utf-8")
        )
        _SUBJECT_MEAN_GLOBAL = float(_SUBJECT_MEAN_META.get("global_mean", 0.5))
        LOG.info(
            "[stacked] loaded subject_mean table (n=%d, global_mean=%.4f) "
            "for Member 2 residual anchor (Task 3).",
            int(_SUBJECT_MEAN_TABLE.shape[0]), _SUBJECT_MEAN_GLOBAL,
        )
    else:
        _SUBJECT_MEAN_TABLE = None
        _SUBJECT_MEAN_GLOBAL = 0.5

    # Reuse Member 3's FAISS-free top-k for the calibrator's
    # neighbor lookup. The neighbor IDs returned are indices into
    # Member 3's item_keys, which MUST be the same ordering as the
    # residual table's training-item rows (the offline pipeline
    # enforces this).
    def _faiss_free_topk_for_calib(item_emb_full, k):
        q = _knn_mod._project_query(_KNN_STATE, item_emb_full)
        embs = _knn_mod._decode_embeddings(_KNN_STATE)
        sims = embs @ q
        top_idx = _knn_mod._topk_indices_descending(sims, int(k))
        return top_idx, sims[top_idx]

    # Save reference to the legacy uncalibrated forward.
    _legacy_predict_uncalibrated = _predict_uncalibrated

    def _stacked_predict(input, labeled=None):
        try:
            benchmark = str(input.get("benchmark", "") or "")
            condition = normalize_condition(input.get("condition", "none"))
            subject_content = str(input.get("subject_content", "") or "")
            item_content = str(input.get("item_content", "") or "")
            subject_key = stable_sha256(subject_content)

            # Member 1: the existing IRT-MLP / coverage-blend ensemble.
            # _legacy_predict_uncalibrated does the heavy work
            # (encoder forward, NN feature build, IRT-MLP forward, blend).
            p1 = float(_legacy_predict_uncalibrated(
                benchmark, condition, subject_content, item_content
            ))

            # Get the cached item embedding (already computed by the
            # legacy path; this is just a dict lookup).
            item_emb = _get_item_embedding(benchmark, condition, item_content)

            # Member 3: pure-numpy kNN-similarity.
            p3 = float(_knn_mod.apply_one(_KNN_STATE, item_emb, subject_key))

            # Members 2 & 4 share a feature vector (member_features schema).
            # Build a runtime feature vector from the same inputs the
            # legacy path uses. We accept the offline schema's contract
            # (theta_s + u_s + subject metadata + pool features +
            # centroid distances + cluster id one-hot + NN features +
            # condition one-hot). The notebook builds this; the runtime
            # path mirrors that builder exactly.
            #
            # NOTE: The Phase 6 notebook passes the precomputed runtime
            # feature builder closure here via _RUNTIME_FEATURE_BUILDER;
            # if the builder is missing, fall back to a zeros vector
            # which makes Members 2 & 4 emit their bias predictions.
            try:
                feats_m24 = _RUNTIME_FEATURE_BUILDER(
                    benchmark=benchmark,
                    condition=condition,
                    subject_key=subject_key,
                    item_emb=item_emb,
                )
            except NameError:
                feats_m24 = np.zeros(_GBDT_STATE.feature_dim, dtype=np.float32)

            # ---- Member 2 (GBDT) ----
            # Residual-mode states (output_mode='residual_logit') were
            # trained to predict logit(y) - logit(p_member1); the
            # runtime composer adds p1's logit back to recover a
            # probability. Legacy binary-mode states return the
            # probability directly via apply_one. The output_mode
            # attribute defaults to 'probability' for states saved
            # before the residual mode was introduced, so this branch
            # is safe for old artifacts too.
            _gbdt_mode = getattr(_GBDT_STATE, "output_mode", "probability")
            if int(_GBDT_STATE.feature_dim) == int(feats_m24.shape[0]):
                feats_for_gbdt = feats_m24
            else:
                # Feature-dim mismatch -- the runtime feature builder isn't
                # wired up. In residual mode with zero features and bias
                # near 0 the tree contribution is small, so composing
                # with p1 still yields a sensible probability (~p1).
                feats_for_gbdt = np.zeros(
                    int(_GBDT_STATE.feature_dim), dtype=np.float32
                )
            if _gbdt_mode == "residual_logit":
                # ``compose_residual_one`` is only present in
                # post-2026-05-26 builds. ``getattr`` keeps this defensive
                # if a frozen older copy of gbdt_member.py is in the
                # bundle for some reason.
                _compose = getattr(_gbdt_mod, "compose_residual_one", None)
                if _compose is None:
                    p2 = float(p1)
                else:
                    # Task 3: prefer subject_mean[subject_id] as the
                    # residual anchor (when the subject_mean table was
                    # shipped). Pre-Task-3 bundles ship no subject_mean
                    # table, in which case fall back to Member 1's
                    # prediction as the anchor (legacy behavior).
                    if _SUBJECT_MEAN_TABLE is not None:
                        _s_id = int(_SUBJECT_TO_ID.get(subject_key, 0))
                        if 0 <= _s_id < int(_SUBJECT_MEAN_TABLE.shape[0]):
                            _gbdt_anchor = float(_SUBJECT_MEAN_TABLE[_s_id])
                        else:
                            _gbdt_anchor = _SUBJECT_MEAN_GLOBAL
                    else:
                        _gbdt_anchor = float(p1)
                    p2 = float(_compose(_GBDT_STATE, feats_for_gbdt, _gbdt_anchor))
            else:
                p2 = float(_gbdt_mod.apply_one(_GBDT_STATE, feats_for_gbdt))

            # ---- Member 4 (LogReg) ----
            if int(_LOGREG_STATE.feature_dim) == int(feats_m24.shape[0]):
                p4 = float(_logreg_mod.apply_state_one(_LOGREG_STATE, feats_m24))
            else:
                # Member 4 may have been trained on a different feature
                # schema (e.g. the post-Task-3 14-dim mean-encoded
                # marginals); fall back to its bias prediction.
                p4 = float(_logreg_mod.apply_state_one(
                    _LOGREG_STATE,
                    np.zeros(int(_LOGREG_STATE.feature_dim), dtype=np.float32),
                ))

            # Auxiliary stacker features.
            try:
                bench_present = float(1.0 if benchmark in _BC_TO_ID or any(
                    benchmark in str(k) for k in _BC_TO_ID
                ) else 0.0)
            except Exception:
                bench_present = 0.0
            # Mean similarity over the Member-3 top-k (cheap).
            _topk_idx, _topk_sims = _faiss_free_topk_for_calib(
                item_emb, k=int(_KNN_STATE.k)
            )
            mean_sim = float(np.mean(_topk_sims)) if _topk_sims.size > 0 else 0.0
            n_support = float(np.log1p(int(_topk_sims.size)))
            centroid_dist = 0.5  # Phase 6 may upgrade this from the centroid module.

            # Task 4: score Member 5 (difficulty-kNN) when present and
            # pass it as the 5th member; otherwise stay on the legacy
            # 4-member path. We MUST keep the [p1..pN] order locked --
            # the stacker weights are aligned to this exact order.
            _member_probs_runtime = [p1, p2, p3, p4]
            if _HAS_MEMBER5 and _MEMBER5_STATE is not None and _m5_mod is not None:
                try:
                    p5 = float(_m5_mod.apply_one(
                        _MEMBER5_STATE, item_emb, subject_key,
                    ))
                except Exception as _m5_exc:
                    LOG.warning("[stacked] Member 5 apply failed (%s); "
                                "falling back to its global mean.", _m5_exc)
                    p5 = float(min(max(float(_MEMBER5_STATE.global_mean),
                                       1e-6), 1.0 - 1e-6))
                _member_probs_runtime.append(p5)
            stacker_feats = _stacker_mod.build_stacker_features_one(
                member_probs=_member_probs_runtime,
                bench_present=bench_present,
                nn_neighbor_support=n_support,
                nn_mean_similarity=mean_sim,
                centroid_distance=centroid_dist,
            )
            p_stacked = float(_stacker_mod.apply_one(_STACKER_STATE, stacker_feats))

            # Single-shot post-stacker NN-residual calibration.
            try:
                _subj_id_for_cal = int(_TRAINING_CACHE_subject_lookup_safe(subject_key))
            except Exception:
                _subj_id_for_cal = -1
            if _subj_id_for_cal >= 0:
                p_final = float(_NN_CAL_STACKED.apply_one(
                    residual_table=_RESIDUAL_TABLE,
                    subject_id=_subj_id_for_cal,
                    neighbor_rows=_topk_idx,
                    neighbor_sims=_topk_sims,
                    p_uncal=p_stacked,
                ))
            else:
                p_final = p_stacked
            # Defensive clamp.
            p_final = float(min(max(p_final, 1e-6), 1.0 - 1e-6))
            if not math.isfinite(p_final):
                return float(p_stacked)
            return p_final
        except Exception as exc:
            LOG.exception("[stacked] predict() failed; falling back to Member 1 only: %s", exc)
            try:
                return float(_legacy_predict(input, labeled))
            except Exception:
                return 0.5

    # Helper for resolving subject_id -> residual table row. Tries the
    # legacy training cache first (it owns subject_key_to_id); falls
    # back to Member 3's subject_index().
    def _TRAINING_CACHE_subject_lookup_safe(subject_key):
        if (
            "TRAINING_CACHE" in globals()
            and TRAINING_CACHE is not None
            and getattr(TRAINING_CACHE, "subject_key_to_id", None)
        ):
            return int(TRAINING_CACHE.subject_key_to_id.get(str(subject_key), -1))
        return int(_KNN_STATE.subject_index(str(subject_key)))

    # Save the legacy predict() so we can fall back on errors.
    _legacy_predict = predict
    predict = _stacked_predict
    LOG.info("[stacked] four-member stacked predict() installed")
'''


# ---------------------------------------------------------------------------
# Faiss-strip patch
# ---------------------------------------------------------------------------
#
# The strict no-FAISS rule says ``import faiss`` must not appear in
# any runtime file, even inside try/except. The legacy template has
# exactly one such block; we replace it with a sentinel that produces
# the same effect (faiss is unavailable -> brute-force fallback fires).

_FAISS_TRY_PATTERN_OLD = (
    "try:\n"
    "            import faiss  # type: ignore\n"
)
_FAISS_TRY_PATTERN_NEW = (
    "try:\n"
    "            faiss = None  # PHASE-5: strict no-FAISS rule (see RUNTIME_ENV.md)\n"
    "            raise ImportError('faiss disabled by phase-5 stacked exporter')\n"
)


def _strip_faiss_imports(model_py_text: str) -> str:
    """Remove the lone ``import faiss`` from the rendered model.py.

    The legacy template wraps the only faiss import in a try/except,
    so we just neutralize it. The brute-force fallback in
    _TrainingItemCache.nearest() takes over.
    """
    if _FAISS_TRY_PATTERN_OLD not in model_py_text:
        # Already stripped or template changed; return unchanged.
        return model_py_text
    return model_py_text.replace(
        _FAISS_TRY_PATTERN_OLD, _FAISS_TRY_PATTERN_NEW, 1
    )


# ---------------------------------------------------------------------------
# Pure-numpy modules to copy into submission/_pure/
# ---------------------------------------------------------------------------

_PURE_MODULE_NAMES: tuple[str, ...] = (
    "gbdt_member",
    "knn_member",
    "logreg_member",
    "stacker",
    "nn_calibration",
    "member_features",
    # Task 4: Member 5 (difficulty-projected kNN). Pure-numpy at runtime;
    # shipped alongside knn_member.py. The model.py block above only
    # imports it when artifacts/member5_dknn/ exists AND the stacker
    # was fit with 5 member columns.
    "member5_difficulty_knn",
)


def _copy_pure_modules(src_dir: Path, dst_dir: Path) -> None:
    """Copy the runtime-safe pure-numpy modules into the bundle.

    Only the modules listed in ``_PURE_MODULE_NAMES`` are copied. We
    also write an empty ``__init__.py`` so the directory is a package.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "__init__.py").write_text("", encoding="utf-8")
    for name in _PURE_MODULE_NAMES:
        src_path = src_dir / f"{name}.py"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Pure module source missing: {src_path}. The exporter "
                "expects all runtime-safe modules to live under src/."
            )
        shutil.copyfile(src_path, dst_dir / f"{name}.py")


# ---------------------------------------------------------------------------
# Static audit
# ---------------------------------------------------------------------------


# Two tiers of forbidden imports:
# - MODULE_SCOPE: not allowed at column 0 (no leading whitespace).
#   Lazy imports inside fit_* function bodies are tolerated because
#   those functions are never called at runtime.
# - ANY_SCOPE: not allowed anywhere, including inside function bodies.
#   FAISS belongs here because the user spec says "no import faiss
#   ANYWHERE in the runtime path" and the legacy template's
#   try-import is neutralized by _strip_faiss_imports.
_FORBIDDEN_RUNTIME_IMPORT_MODULE_SCOPE: tuple[str, ...] = (
    r"^import\s+lightgbm",
    r"^from\s+lightgbm",
    r"^import\s+sklearn",
    r"^from\s+sklearn",
    r"^import\s+xgboost",
    r"^from\s+xgboost",
    r"^import\s+m2cgen",
    r"^from\s+m2cgen",
)
_FORBIDDEN_RUNTIME_IMPORT_ANY_SCOPE: tuple[str, ...] = (
    r"^\s*import\s+faiss",
    r"^\s*from\s+faiss",
)


def audit_runtime_imports(submission_dir: Path) -> list[tuple[Path, str, int]]:
    """Scan every .py under ``submission_dir`` for forbidden imports.

    Returns a list of ``(file, pattern, lineno)`` triples. An empty
    list means the audit passed.

    Lazy imports inside fit_* function bodies are tolerated for
    lightgbm / sklearn / xgboost / m2cgen because those functions
    never execute at runtime; FAISS is forbidden at any scope.
    """
    import re

    findings: list[tuple[Path, str, int]] = []
    all_patterns = (
        list(_FORBIDDEN_RUNTIME_IMPORT_MODULE_SCOPE)
        + list(_FORBIDDEN_RUNTIME_IMPORT_ANY_SCOPE)
    )
    for py_path in submission_dir.rglob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for pattern in all_patterns:
            for m in re.finditer(pattern, text, flags=re.MULTILINE):
                lineno = text[: m.start()].count("\n") + 1
                findings.append((py_path, pattern, lineno))
    return findings


def measure_bundle_size_bytes(submission_dir: Path) -> int:
    """Sum of all file sizes under ``submission_dir`` (UNCOMPRESSED).

    The actual ZIP will be smaller (compression). We use the
    uncompressed size as a conservative estimate.
    """
    total = 0
    for f in submission_dir.rglob("*"):
        if f.is_file():
            total += int(f.stat().st_size)
    return int(total)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def export_four_member_stacked_run(
    *,
    member1_bundle_dir: Path | str,
    gbdt_state: Any,
    knn_state: Any,
    logreg_state: Any,
    stacker_state: Any,
    nn_calibrator: Any | None = None,
    residual_table: Any | None = None,
    out_dir: Path | str,
    src_dir: Path | str = "src",
    runtime_feature_builder_py: str | None = None,
    subject_mean_table: Any | None = None,
    member5_state: Any | None = None,
    zip_cap_bytes: int = _DEFAULT_ZIP_CAP_BYTES,
    audit: bool = True,
) -> Path:
    """Wrap a Member-1 ensemble bundle into a four-member stacked submission.

    Parameters
    ----------
    member1_bundle_dir : Path
        An existing ``submission/`` directory produced by
        :func:`src.export_submission.export_ensemble_run`. The
        contents are copied into ``out_dir`` and then patched in
        place. The original bundle is NOT modified.
    gbdt_state : GBDTMemberState
        Member 2 fitted state.
    knn_state : KNNMemberState
        Member 3 fitted state.
    logreg_state : LogRegMemberState
        Member 4 fitted state.
    stacker_state : StackerState
        Fitted stacker.
    nn_calibrator : NNCalibrator | None
        Post-stacker NN-residual calibrator. If ``None``, the
        bundle ships an alpha=0 (no-op) calibrator.
    residual_table : SubjectResidualTable | None
        Residual table for the calibrator. If ``None``, the
        calibrator is a no-op regardless of ``nn_calibrator``.
    out_dir : Path
        Where to write the new bundle.
    src_dir : Path
        Path to the source ``src/`` directory containing the
        pure-numpy modules to ship.
    runtime_feature_builder_py : str | None
        Optional Python source defining ``_RUNTIME_FEATURE_BUILDER``.
        See module docstring; if omitted, Members 2 & 4 fall back to
        their bias predictions and the stacker downweights them
        accordingly. Phase 6's notebook supplies this.
    subject_mean_table : SubjectMeanTable | None
        Task 3 (Member 2 v2): the per-subject mean pass-rate table used
        as the GBDT residual anchor. When provided, shipped to
        ``artifacts/subject_mean/`` and the runtime uses
        ``subject_mean[subject_id]`` as the Member-2 anchor instead of
        Member 1's prediction. When ``None``, the runtime falls back
        to Member 1 as the anchor (legacy / pre-Task-3 behavior).
    member5_state : Member5State | None
        Task 4: Member 5 (difficulty-projected kNN). When provided AND
        the stacker was fit with 5 member columns (feature_dim == 9),
        the state is shipped to ``artifacts/member5_dknn/`` and the
        runtime scores it alongside Members 1-4. When ``None`` (or
        when the stacker has only 4 member columns), the runtime
        stays on the 4-member path; this preserves backward
        compatibility with pre-Task-4 bundles.
    zip_cap_bytes : int
        Hard upper bound on the bundle's uncompressed total size.
    audit : bool
        If True, run the static import audit and raise on findings.

    Returns
    -------
    Path
        The output bundle directory.
    """
    member1_bundle_dir = Path(member1_bundle_dir)
    out_dir = Path(out_dir)
    src_dir = Path(src_dir)

    if not member1_bundle_dir.exists():
        raise FileNotFoundError(
            f"member1_bundle_dir does not exist: {member1_bundle_dir}"
        )
    if not (member1_bundle_dir / "model.py").exists():
        raise FileNotFoundError(
            f"Member 1 bundle missing model.py at {member1_bundle_dir}"
        )

    # Step 1: copy the Member 1 bundle.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(member1_bundle_dir, out_dir)

    # Step 2: write member states.
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    gbdt_state.save(artifacts_dir / "member2_gbdt")
    knn_state.save(artifacts_dir / "member3_knn")
    logreg_state.save(artifacts_dir / "member4_logreg")
    stacker_state.save(artifacts_dir / "stacker")

    if nn_calibrator is not None:
        cal_dir = artifacts_dir / "nn_calibrator_stacked"
        cal_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        (cal_dir / "state.json").write_text(
            _json.dumps(nn_calibrator.to_dict(), indent=2),
            encoding="utf-8",
        )
    if residual_table is not None:
        residual_table.save(artifacts_dir / "residual_table")

    # Task 4: ship Member 5 (difficulty-projected kNN) when provided.
    # We guard against the misconfiguration of shipping member5_state
    # without a 5-column stacker, since the runtime would silently
    # ignore it. Raise an explicit error to make the misconfig loud.
    if member5_state is not None:
        if int(stacker_state.feature_dim) < 9:
            raise ValueError(
                "member5_state was provided but stacker_state has "
                f"feature_dim={int(stacker_state.feature_dim)} (<9), meaning "
                "the stacker was fit on only 4 member columns and would "
                "silently ignore Member 5 at runtime. Re-train the stacker "
                "with [N, 5] member_probs OR drop member5_state from the "
                "export call."
            )
        member5_state.save(artifacts_dir / "member5_dknn")

    # Task 3: ship the subject_mean table so the runtime can use it as
    # the Member-2 residual anchor (instead of Member 1's prediction).
    if subject_mean_table is not None:
        sm_dir = artifacts_dir / "subject_mean"
        sm_dir.mkdir(parents=True, exist_ok=True)
        import json as _json_sm
        import numpy as _np_sm
        _np_sm.save(sm_dir / "subject_mean.npy",
                    _np_sm.asarray(subject_mean_table.subject_mean, dtype=_np_sm.float64))
        _np_sm.save(sm_dir / "subject_obs_count.npy",
                    _np_sm.asarray(subject_mean_table.subject_obs_count, dtype=_np_sm.float64))
        (sm_dir / "meta.json").write_text(
            _json_sm.dumps(
                {
                    "global_mean": float(subject_mean_table.global_mean),
                    "smoothing": float(subject_mean_table.smoothing),
                    "n_subjects": int(subject_mean_table.subject_mean.shape[0]),
                    "schema_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Step 3: copy pure-numpy modules.
    _copy_pure_modules(src_dir, out_dir / "_pure")

    # Step 4: patch model.py (strip faiss + append stacker postprocessing).
    model_py = out_dir / "model.py"
    text = model_py.read_text(encoding="utf-8")
    text = _strip_faiss_imports(text)
    if runtime_feature_builder_py:
        # Inject the user-provided feature builder ABOVE the stacker
        # postprocessing block so _RUNTIME_FEATURE_BUILDER is in scope.
        text += "\n\n" + runtime_feature_builder_py.rstrip() + "\n"
    text += _STACKER_POSTPROCESS_PY
    model_py.write_text(text, encoding="utf-8")

    # Step 5: audit + size check.
    if audit:
        findings = audit_runtime_imports(out_dir)
        if findings:
            details = "\n".join(
                f"  {f}: {pat} (line {ln})" for (f, pat, ln) in findings
            )
            raise RuntimeError(
                f"Runtime import audit FAILED -- forbidden imports:\n{details}"
            )

    total_bytes = measure_bundle_size_bytes(out_dir)
    LOG.info(
        "[stacked] bundle assembled at %s, uncompressed size=%.2f MB",
        out_dir,
        total_bytes / (1024.0 * 1024.0),
    )
    if total_bytes > int(zip_cap_bytes):
        raise RuntimeError(
            f"Bundle uncompressed size {total_bytes / (1024**2):.2f} MB > "
            f"cap {zip_cap_bytes / (1024**2):.2f} MB. Reduce member sizes "
            "(PCA dim, fp16 -> int8, drop training cache, etc.) or raise "
            "zip_cap_bytes if you've measured the actual ZIP."
        )

    return out_dir


def zip_bundle(
    submission_dir: Path | str,
    zip_path: Path | str,
) -> Path:
    """Standard ``zip -r submission.zip submission/`` equivalent."""
    submission_dir = Path(submission_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in submission_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(submission_dir.parent))
    return zip_path


__all__ = [
    "export_four_member_stacked_run",
    "audit_runtime_imports",
    "measure_bundle_size_bytes",
    "zip_bundle",
]
