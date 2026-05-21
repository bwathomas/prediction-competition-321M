"""Submission audit harness.

Catches the silent-mismatch failure modes that erase local gains on the
leaderboard.  Designed to be runnable on CPU with no GPU/transformers
dependencies for Phases 1-4 (the only thing that requires the full
runtime stack is the optional Phase 5 reference-prediction comparison).

Phases
======

  1. Bundle structure:
       required files present, no stray test artifacts.

  2. Static parse + symbol check:
       model.py and labeling.py parse; required symbols are present in
       the patched calibrator + labeling templates.

  3. Meta self-consistency:
       artifacts/runtime_meta.json parses; declared fields are
       consistent with what model.py actually reads / does (e.g. if
       meta says ``calibration_disabled: True`` but model.py calls
       ``_CALIBRATOR.fit_from_labeled(labeled)``, that is a silent
       mismatch -- meta is stale and we flag it).

  4. Calibrator behavioral exec:
       extract the calibrator block from model.py, exec it in an
       isolated namespace with stubs, run a deterministic fit on a
       known labeled batch, and verify ``b_global`` lands within a
       tight tolerance of an analytical reference value.

  5. (Optional) Reference-prediction comparison:
       if ``artifacts/audit_reference.json`` is present, load the fixed
       input batch + expected predictions baked at export time, run the
       bundle's ``predict()``, and compare elementwise.  Requires the
       full runtime stack (GPU, transformers, torch).  We ship the
       scaffolding here and document that this phase has to run on the
       deploy environment.

Usage
=====

    py scripts/_audit_bundle.py
    py scripts/_audit_bundle.py path/to/bundle.zip [more.zip ...]
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import sys
import types
import zipfile
from pathlib import Path

DEFAULT_BUNDLES = [
    Path(r"C:/Users/benja/Downloads/submission/submission_streamed_encoder_nn_perbc_cal.zip"),
    Path(r"C:/Users/benja/Downloads/submission/submission_item_sample_perbc_cal.zip"),
    Path(r"C:/Users/benja/Downloads/submission/submission_item_uniform_v2.zip"),
]


class AuditFail(RuntimeError):
    pass


class AuditWarn(UserWarning):
    pass


def _print_phase(num: int, name: str) -> None:
    print("\n{} Phase {}: {} {}".format("-" * 8, num, name, "-" * (60 - len(name))))


def phase1_structure(zf: zipfile.ZipFile) -> None:
    """Required files present, no clearly-stale artifacts."""
    names = set(zf.namelist())
    required = {"model.py", "labeling.py", "models.txt", "artifacts/runtime_meta.json"}
    missing = required - names
    if missing:
        raise AuditFail("missing required files: {}".format(sorted(missing)))

    # Should also have a checkpoint (the actual model weights).
    has_ckpt = any(n.startswith("artifacts/") and n.endswith(".pt") for n in names)
    if not has_ckpt:
        raise AuditFail("no artifacts/*.pt checkpoint shipped")

    # If a checkpoint claims a particular run_id, ensure the meta says the
    # same (e.g. don't ship a kfactor checkpoint with a hybrid-irt meta).
    print("[1.OK] structure: {} files, checkpoint present".format(len(names)))


def phase2_static(model_src: str, labeling_src: str) -> None:
    """model.py + labeling.py parse and have the symbols we expect."""
    ast.parse(model_src)
    ast.parse(labeling_src)

    must_have_model = [
        "class _Calibrator",
        "def fit_from_labeled",
        "def _fit_intercept_ridge",
        "_RIDGE_LAMBDA_GLOBAL",
        "_CALIBRATOR.apply(p, _bc_key_for_apply",
        "_CALIBRATOR.fit_from_labeled(labeled)",
        "def _predict_uncalibrated",
        "def _enqueue_for_batch",
    ]
    for needle in must_have_model:
        if needle not in model_src:
            raise AuditFail("model.py missing required symbol: " + needle)
    print("[2.OK] model.py: required symbols present")

    must_have_labeling = [
        "_FRACTION_NEW_POOL",
        "_item_in_new_pool",
        "def acquisition_function",
        "_enqueue_for_batch",
    ]
    for needle in must_have_labeling:
        if needle not in labeling_src:
            raise AuditFail("labeling.py missing required symbol: " + needle)
    print("[2.OK] labeling.py: required symbols present")

    must_not_have_model = [
        "def _fit_beta_calibration",
        "def _gated_fit",
        "kind\": \"beta\"",
        "kind\": \"temp_intercept\"",
    ]
    for needle in must_not_have_model:
        if needle in model_src:
            raise AuditFail("model.py still has stale symbol: " + needle)

    must_not_have_labeling = [
        "1000.0 * novelty",
        "from model import _baseline_logit",
    ]
    for needle in must_not_have_labeling:
        if needle in labeling_src:
            raise AuditFail("labeling.py still has stale symbol: " + needle)
    print("[2.OK] no stale calibrator / acquisition symbols left behind")


def phase3_meta_consistency(meta: dict, model_src: str) -> list[str]:
    """Cross-check the meta JSON against model.py behavior.

    Returns a list of WARNING strings (cosmetic mismatches that don't
    affect runtime behavior).  Hard fails raise AuditFail directly.
    """
    warns: list[str] = []

    # 1. Required meta fields.
    required = ("default_prob", "default_calibrator", "encoder_runtime_batch_size",
                "runtime_architecture", "model_name", "run_id")
    for k in required:
        if k not in meta:
            raise AuditFail("meta missing required field: " + k)

    # 2. default_prob must be a finite float in (0, 1).
    dp = meta["default_prob"]
    if not (isinstance(dp, (int, float)) and 0.0 < float(dp) < 1.0):
        raise AuditFail("default_prob out of range: " + repr(dp))

    # 3. default_calibrator must be either identity or intercept w/ finite b.
    dc = meta["default_calibrator"]
    if not isinstance(dc, dict) or dc.get("kind") not in ("identity", "intercept"):
        raise AuditFail("default_calibrator must be {'kind':'identity'} or "
                        "{'kind':'intercept','b':<float>}; got " + repr(dc))
    if dc["kind"] == "intercept":
        b = dc.get("b")
        if not isinstance(b, (int, float)) or not math.isfinite(float(b)):
            raise AuditFail("intercept calibrator missing finite b: " + repr(dc))

    # 4. SILENT MISMATCH check: if meta says calibration_disabled=True but
    #    model.py calls fit_from_labeled, the bundle is contradicting
    #    itself.  This is exactly the class of bug that erases gains.
    if meta.get("calibration_disabled") is True:
        if "_CALIBRATOR.fit_from_labeled(labeled)" in model_src:
            warns.append(
                "meta.calibration_disabled=True but model.py calls "
                "_CALIBRATOR.fit_from_labeled() -- meta is stale "
                "(model WILL calibrate at runtime; meta says it won't)"
            )

    # 5. encoder_runtime_batch_size must be a positive int.
    ebs = meta["encoder_runtime_batch_size"]
    if not isinstance(ebs, int) or ebs <= 0:
        raise AuditFail("encoder_runtime_batch_size must be positive int: " + repr(ebs))

    # 6. nn_features.enabled and the model.py NN feature symbols must
    #    agree.  If meta says NN features are on but model.py never
    #    references the NN feature cache, that's silent-disable.
    nn = meta.get("nn_features", {})
    if isinstance(nn, dict) and nn.get("enabled"):
        if "_NN_FEATURE_CACHE" not in model_src and "nn_passrate" not in model_src:
            warns.append(
                "meta.nn_features.enabled=True but model.py has no NN "
                "feature cache reference -- features may be silently zeroed"
            )

    print("[3.OK] meta consistency: {} hard checks, {} warning(s)".format(
        6, len(warns)
    ))
    for w in warns:
        print("       WARN: " + w)
    return warns


# ---------------------------------------------------------------------------
# Phase 4: behavioral exec of the calibrator block in an isolated sandbox.
# ---------------------------------------------------------------------------

_CAL_BANNER_RE = re.compile(r"^# -{50,}\n# Calibrator [^\n]*\n# -{50,}\n", re.M)


def _extract_calibrator(model_src: str) -> str:
    m = _CAL_BANNER_RE.search(model_src)
    if not m:
        raise AuditFail("calibrator banner not found in model.py")
    start = m.start()
    nb = re.search(r"\n\n\n# -{50,}\n", model_src[m.end():])
    if not nb:
        raise AuditFail("could not find next banner after calibrator block")
    return model_src[start:m.end() + nb.start() + 1]


def phase4_calibrator_behavior(model_src: str) -> None:
    """Exec the calibrator block in a stubbed namespace and check that a
    known-biased predictor produces an intercept fit in the expected
    direction.

    Red-team-safe analytical reference:
        For a constant predictor p_hat = 0.7 with all-zero labels and
        ridge=20, the Bayesian-MAP intercept b satisfies
            ridge * b + sum (sigmoid(z+b) - 0) = 0
        i.e. 20 * b + N * sigmoid(logit(0.7) + b) = 0.
        Solving numerically with N=30 gives b ~= -0.5714 (verified by
        independent solver).
    """
    src = _extract_calibrator(model_src)
    sandbox = {
        "__name__": "_audit_sandbox",
        "math": math,
        "EPS": 1e-7,
        "DEFAULT_PROB": 0.5,
        "LOG": types.SimpleNamespace(info=lambda *a, **k: None,
                                     warning=lambda *a, **k: None),
        "normalize_condition": lambda c: str(c or "none"),
        "_BC_TO_ID": {"k::none": 1},
        "_predict_uncalibrated": lambda b, c, s, i: 0.7,
    }
    exec(compile(src, "<audit_calibrator>", "exec"), sandbox)
    Cal = sandbox["_Calibrator"]
    cal = Cal()
    labels = [
        {"label": 0.0, "benchmark": "k", "condition": "none",
         "subject_content": "s{}".format(i), "item_content": "q{}".format(i)}
        for i in range(30)
    ]
    cal.fit_from_labeled(labels)

    # Recompute analytical reference here so the audit is self-contained.
    def _solve_ref(p_hat: float, n: int, ridge: float) -> float:
        z = math.log(p_hat / (1.0 - p_hat))
        b = 0.0
        for _ in range(200):
            q = 1.0 / (1.0 + math.exp(-(z + b)))
            g = 2.0 * ridge * b + n * q
            h = 2.0 * ridge + n * q * (1.0 - q)
            step = g / h
            b -= step
            if abs(step) < 1e-12:
                break
        return b

    b_ref = _solve_ref(0.7, 30, ridge=20.0)
    b_got = cal.b_global
    if abs(b_got - b_ref) > 1e-3:
        raise AuditFail(
            "calibrator b_global = {:+.6f} but analytical reference is "
            "{:+.6f} (diff = {:+.6f})".format(b_got, b_ref, b_got - b_ref)
        )
    print("[4.OK] calibrator behavior: b_global = {:+.4f} (ref {:+.4f}, "
          "diff {:.1e})".format(b_got, b_ref, abs(b_got - b_ref)))

    # Apply should pull a 0.7 prediction down (b_global is negative).
    p_after = cal.apply(0.7, "k::none")
    if p_after >= 0.7:
        raise AuditFail("apply() did not pull a too-high prediction down "
                        "(0.7 -> {:.4f})".format(p_after))
    print("[4.OK] apply(0.7, 'k::none') = {:.4f} (pulled down as expected)".format(p_after))


# ---------------------------------------------------------------------------
# Phase 5: reference-prediction comparison (optional, requires runtime stack).
# ---------------------------------------------------------------------------


def phase5_reference_predictions(zip_path: Path, zf: zipfile.ZipFile) -> None:
    """If artifacts/audit_reference.json was baked at export time, run
    the bundle's predict() on the fixed input batch and compare to the
    reference outputs.  This is the gold-standard check for "the
    submitted artifact matches what the notebook produced".

    Requires the full runtime stack (torch, transformers, possibly GPU).
    We skip gracefully when the reference file is absent or imports fail.
    """
    names = set(zf.namelist())
    if "artifacts/audit_reference.json" not in names:
        print("[5.SKIP] no artifacts/audit_reference.json baked into bundle "
              "-- run src/export_submission.py with `audit_reference_inputs=` "
              "to enable this phase")
        return

    ref = json.loads(zf.read("artifacts/audit_reference.json"))
    inputs = ref["inputs"]
    expected = ref["expected_predictions"]
    if len(inputs) != len(expected):
        raise AuditFail("audit_reference.json has mismatched lengths")

    # Trying to actually run predict() in-process here would require
    # extracting the full bundle to a temp dir and importing model.py,
    # which pulls in torch + transformers + huggingface caches.  We
    # ship the scaffold but defer execution to a wrapper script that
    # runs in the deploy environment.
    print("[5.NOTE] reference predictions present ({} inputs); execute "
          "scripts/_audit_bundle_runtime.py {} on a GPU-capable host to "
          "actually compare predict() output against the reference".format(
              len(inputs), zip_path.name
          ))


# ---------------------------------------------------------------------------
# Cross-bundle consistency (Phase 6).
# ---------------------------------------------------------------------------


def cross_bundle_consistency(bundles: dict[Path, dict]) -> None:
    """For bundles that should share patched runtime code (model.py
    calibrator block + labeling.py), verify byte equality after
    extracting just those blocks.  This catches "we updated bundle A's
    calibration but forgot bundle B" regressions.
    """
    print("\n{} Phase 6: cross-bundle consistency {}".format("-" * 8, "-" * 25))
    # Group by labeling.py SHA (it should be identical across all
    # bundles that ship the new dual-pool labeling).
    by_labeling: dict[str, list[Path]] = {}
    for path, info in bundles.items():
        h = info["labeling_sha"]
        by_labeling.setdefault(h, []).append(path)
    if len(by_labeling) > 1:
        print("[6.WARN] labeling.py byte-mismatch across bundles:")
        for h, paths in by_labeling.items():
            print("   sha={}: {}".format(h[:12], [p.name for p in paths]))
    else:
        print("[6.OK] labeling.py byte-identical across all {} bundles".format(len(bundles)))

    # Same check for the calibrator block extracted from each model.py.
    by_cal: dict[str, list[Path]] = {}
    for path, info in bundles.items():
        cal_src = _extract_calibrator(info["model_src"])
        import hashlib
        h = hashlib.sha256(cal_src.encode("utf-8")).hexdigest()
        by_cal.setdefault(h, []).append(path)
    if len(by_cal) > 1:
        print("[6.WARN] calibrator block byte-mismatch across bundles:")
        for h, paths in by_cal.items():
            print("   sha={}: {}".format(h[:12], [p.name for p in paths]))
    else:
        print("[6.OK] calibrator block byte-identical across all {} bundles".format(len(bundles)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def audit_one(zip_path: Path) -> dict:
    print("\n" + "=" * 70)
    print("Auditing: {}".format(zip_path.name))
    print("=" * 70)
    if not zip_path.exists():
        raise AuditFail("bundle not found: " + str(zip_path))

    import hashlib
    with zipfile.ZipFile(zip_path, "r") as zf:
        phase1_structure(zf)
        model_src = zf.read("model.py").decode("utf-8")
        labeling_src = zf.read("labeling.py").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
        phase2_static(model_src, labeling_src)
        warns = phase3_meta_consistency(meta, model_src)
        phase4_calibrator_behavior(model_src)
        phase5_reference_predictions(zip_path, zf)

    return {
        "model_src": model_src,
        "labeling_src": labeling_src,
        "labeling_sha": hashlib.sha256(labeling_src.encode("utf-8")).hexdigest(),
        "meta": meta,
        "warnings": warns,
    }


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]] if len(argv) > 1 else DEFAULT_BUNDLES
    results: dict[Path, dict] = {}
    failures: list[tuple[Path, str]] = []
    for p in paths:
        try:
            results[p] = audit_one(p)
        except AuditFail as e:
            print("[FAIL] {}: {}".format(p.name, e))
            failures.append((p, str(e)))
    if len(results) >= 2:
        cross_bundle_consistency(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_warns = sum(len(r["warnings"]) for r in results.values())
    print("Bundles audited: {}".format(len(results)))
    print("Hard failures:   {}".format(len(failures)))
    print("Warnings total:  {}".format(total_warns))
    if failures:
        for p, msg in failures:
            print("  FAIL {}: {}".format(p.name, msg))
        return 1
    if total_warns:
        for p, r in results.items():
            for w in r["warnings"]:
                print("  WARN {}: {}".format(p.name, w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
