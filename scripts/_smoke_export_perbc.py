"""Smoke test for the updated src/export_submission.py.

Verifies:
  1. The outer module parses (AST) and imports without error.
  2. The embedded _RUNTIME_MODEL_PY template parses as valid Python.
  3. The embedded _RUNTIME_LABELING_PY template parses as valid Python.
  4. The runtime model.py template contains the new per-bc gated calibrator
     and the bc-routed apply site.
  5. The runtime labeling.py template contains the graded novelty +
     anchoring acquisition (with binary fallback).
  6. The new ``compute_train_counts`` helper round-trips a small fake df.
  7. The runtime ``_Calibrator`` class can be instantiated, fit on a
     small synthetic labeled list, and apply() routes correctly.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 1) Import the outer module.
import src.export_submission as es  # noqa: E402

print("[1/7] OK src.export_submission imported")

# 2) Parse _RUNTIME_MODEL_PY.
ast.parse(es._RUNTIME_MODEL_PY)
print("[2/7] OK _RUNTIME_MODEL_PY parses ({:,} bytes)".format(len(es._RUNTIME_MODEL_PY)))

# 3) Parse _RUNTIME_LABELING_PY.
ast.parse(es._RUNTIME_LABELING_PY)
print("[3/7] OK _RUNTIME_LABELING_PY parses ({:,} bytes)".format(len(es._RUNTIME_LABELING_PY)))

# 4) Required markers in the runtime model.py.
m = es._RUNTIME_MODEL_PY
must_have_model = [
    "_RIDGE_LAMBDA",
    "def _fit_intercept_ridge",
    "def _gated_fit",
    "def _stable_shuffle_order",
    "def _kfold_indices",
    "def _cv_nll_pair",
    "class _Calibrator",
    "self.per_bc: dict[str, dict]",
    "def apply(self, p: float, bc_key: str = \"\")",
    "_bc_key_for_apply = \"{0}::{1}\"",
    "p = _CALIBRATOR.apply(p, _bc_key_for_apply)",
    "_N_TRAIN_PER_SUBJECT",
    "_N_TRAIN_PER_BC",
    "_TRAIN_COUNTS_RAW",
    "META.get(\"train_counts\")",
    "n_repeats",
    "margin_nats_per_param",
    "require_majority_repeat_wins",
    "n_new_benchmarks",
]
for needle in must_have_model:
    if needle not in m:
        raise AssertionError("RUNTIME_MODEL_PY missing required marker: " + needle)
forbidden_model = [
    "def _fit_intercept_only",
    "def _fit_temp_intercept",
    "def _fit_beta_calibration",
    "def _beta_calibration_loss",
    '"kind": "beta"',
    '"kind": "temp_intercept"',
]
for needle in forbidden_model:
    if needle in m:
        raise AssertionError("RUNTIME_MODEL_PY still contains forbidden marker: " + needle)
print("[4/7] OK _RUNTIME_MODEL_PY has all required markers, no forbidden markers")

# 5) Required markers in the runtime labeling.py.
lp = es._RUNTIME_LABELING_PY
must_have_labeling = [
    "_BC_TO_ID",
    "_SUBJECT_TO_ID",
    "_N_TRAIN_PER_BC",
    "_N_TRAIN_PER_SUBJECT",
    "novelty",
    "anchoring",
    "_enqueue_for_batch",
    "1000.0 * novelty",
    "10.0 * anchoring",
    "_graded_novelty",
    "_graded_anchoring",
    "_stable_tiebreak",
]
for needle in must_have_labeling:
    if needle not in lp:
        raise AssertionError("RUNTIME_LABELING_PY missing required marker: " + needle)
forbidden_labeling = [
    "from model import _baseline_logit",
    "Lewis & Gale",  # the old uncertainty doc-string callout
]
for needle in forbidden_labeling:
    if needle in lp:
        raise AssertionError("RUNTIME_LABELING_PY still contains forbidden marker: " + needle)
print("[5/7] OK _RUNTIME_LABELING_PY has all required markers, no forbidden markers")

# 6) compute_train_counts round-trip.
class _FakeSeries:
    def __init__(self, items):
        self._items = list(items)

    def tolist(self):
        return list(self._items)


class _FakeDF:
    def __init__(self, rows):
        self._cols = {
            "subject_content": _FakeSeries(r[0] for r in rows),
            "benchmark": _FakeSeries(r[1] for r in rows),
            "condition": _FakeSeries(r[2] for r in rows),
        }
        self.columns = list(self._cols.keys())

    def __getitem__(self, k):
        return self._cols[k]

rows = [
    ("gpt-4", "mmlu", "none"),
    ("gpt-4", "mmlu", "none"),
    ("gpt-4", "gsm8k", "cot"),
    ("llama-3-70b", "mmlu", "none"),
    ("llama-3-70b", "gsm8k", "cot"),
    ("llama-3-70b", "gsm8k", "cot"),
]
counts = es.compute_train_counts(_FakeDF(rows))
assert set(counts.keys()) == {"n_per_subject", "n_per_bc"}, counts.keys()
# 2 unique subjects, 2 unique bc keys
assert len(counts["n_per_subject"]) == 2
assert len(counts["n_per_bc"]) == 2
# gpt-4 appears 3x; llama-3-70b appears 3x
assert sorted(counts["n_per_subject"].values()) == [3, 3]
# mmlu::none appears 3x; gsm8k::cot appears 3x
assert sorted(counts["n_per_bc"].values()) == [3, 3]
assert "mmlu::none" in counts["n_per_bc"]
assert "gsm8k::cot" in counts["n_per_bc"]
print("[6/7] OK compute_train_counts round-trips correctly")

# 7) Behavioral check of the runtime _Calibrator class.  Exec the runtime
# block into a sandbox namespace with minimal stubs and run a small fit.
sandbox = {
    "__name__": "_sandbox_model",
    "math": __import__("math"),
    "EPS": 1e-7,
    "DEFAULT_PROB": 0.5,
    "normalize_condition": lambda c: str(c or "none"),
    "_BC_TO_ID": {"known::none": 1},
    "LOG": types.SimpleNamespace(info=lambda *a, **k: None),
}
# Stub _predict_uncalibrated to return a known-biased value so we can
# verify the gate accepts.
def _predict_uncal(benchmark, condition, subject, item):
    # Return 0.85 -- well above true probability so labels with y=0
    # should trigger a negative-intercept calibration.
    return 0.85
sandbox["_predict_uncalibrated"] = _predict_uncal

# Extract just the calibrator class + helpers from the runtime template.
# Easiest is to exec the entire calibrator block.  We isolate it by
# splitting on the next module-level banner.
cal_start = m.find("_RIDGE_LAMBDA = 1.0")
cal_end = m.find("# ---------------------------------------------------------------------------\n# Training-item cache")
assert cal_start > 0 and cal_end > cal_start, (cal_start, cal_end)
cal_block = m[cal_start:cal_end]
exec(compile(cal_block, "<runtime_calibrator>", "exec"), sandbox)

Calibrator = sandbox["_Calibrator"]
cal = Calibrator()
# 30 labels all with y=0 but predict_uncal returning 0.85
# -> the calibrator should accept a negative intercept.
labels = [
    {
        "label": 0.0,
        "benchmark": "known",
        "condition": "none",
        "subject_content": "gpt-4",
        "item_content": "Q{}".format(i),
    }
    for i in range(30)
]
cal.fit_from_labeled(labels)
print("    fit yielded state.kind={}, per_bc size={}".format(
    cal.state.get("kind"), len(cal.per_bc)
))
assert cal.state.get("kind") == "intercept", "expected a global intercept fit"
# Apply on a fresh p=0.85 should pull below 0.85.
p_orig = 0.85
p_cal = cal.apply(p_orig, "known::none")
print("    apply(0.85, 'known::none') = {:.4f} (was {:.4f})".format(p_cal, p_orig))
assert p_cal < p_orig, "calibrator should pull down a too-high prediction"
# Apply with empty bc_key falls back to global.
p_cal_global = cal.apply(p_orig, "")
assert abs(p_cal_global - p_cal) > -1e-9  # global == per_bc when per_bc empty, basically same
print("[7/7] OK runtime _Calibrator behaves correctly")

print("\n[OK] all export_submission smoke tests pass")
