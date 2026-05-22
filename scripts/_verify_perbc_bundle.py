"""Structural verification of one or more perbc_cal bundles."""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIPS = [
    Path(r"C:/Users/benja/Downloads/submission/submission_streamed_encoder_nn_perbc_cal.zip"),
    Path(r"C:/Users/benja/Downloads/submission/submission_item_sample_perbc_cal.zip"),
    Path(r"C:/Users/benja/Downloads/submission/submission_item_uniform_v2.zip"),
    Path(r"C:/Users/benja/Downloads/submission/submission_new_model_perbc_cal.zip"),
]
ZIPS = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_ZIPS


def verify(ZIP: Path) -> None:
    print("=" * 70)
    print("Verifying:", ZIP.name)
    print("=" * 70)
    with zipfile.ZipFile(ZIP, "r") as zf:
        m = zf.read("model.py").decode("utf-8")
        labeling_py = zf.read("labeling.py").decode("utf-8")
        meta = zf.read("artifacts/runtime_meta.json").decode("utf-8")
    _verify_one(m, labeling_py, meta)


def _verify_one(m: str, labeling_py: str, meta: str) -> None:

    ast.parse(m)
    ast.parse(labeling_py)
    print("model.py:    OK syntax,", len(m), "bytes")
    print("labeling.py: OK syntax,", len(labeling_py), "bytes")

    i = m.find("def predict(input: dict")
    end = m.find("\n\n\n", i)
    print("\n--- predict() body in shipped bundle ---")
    print(m[i:end])
    print("---\n")

    bc_off = m.find("_BC_TO_ID:")
    sub_off = m.find("_SUBJECT_TO_ID:")
    pred_off = m.find("def predict(input: dict")
    print(
        "_BC_TO_ID offset:", bc_off,
        "  _SUBJECT_TO_ID offset:", sub_off,
        "  predict() offset:", pred_off,
    )
    assert bc_off < pred_off, "_BC_TO_ID must be defined before predict()"
    assert sub_off < pred_off, "_SUBJECT_TO_ID must be defined before predict()"
    assert "_CALIBRATOR.apply(p, _bc_key_for_apply)" in m
    assert "_bc_key_for_apply = \"{0}::{1}\"" in m
    assert "def _fit_beta_calibration" not in m
    assert "_RIDGE_LAMBDA_GLOBAL" in m, "must use new PP_CONSERVATIVE ridge"
    assert "_RIDGE_LAMBDA_BC" in m, "must use new PP_CONSERVATIVE ridge"
    assert "_RIDGE_LAMBDA_TYPE" in m, "must declare TYPE-conditional ridge"
    assert "self.per_bc" in m
    assert "self.per_bc: dict[str, float]" in m, "per_bc values must now be floats, not state dicts"
    assert "self.b_global" in m
    assert "self.delta_type" in m, "calibrator must expose delta_type field"
    assert "is_new_list" in m, "calibrator fit must compute is_new_list"
    assert "target_b" in m
    # Gated-fit machinery should be GONE now.
    assert "def _gated_fit" not in m, "gated_fit replaced by partial pooling"
    assert "margin_nats_per_param" not in m
    assert "require_majority_repeat_wins" not in m
    # The fit block must be active (not the disabled "_ = labeled" form).
    assert "_CALIBRATOR.fit_from_labeled(labeled)" in m
    assert "_ = labeled  # touch" not in m
    print("[OK] model.py structural assertions hold")

    assert "_BC_TO_ID" in labeling_py
    assert "_SUBJECT_TO_ID" in labeling_py
    assert "_N_TRAIN_PER_BC" in labeling_py
    assert "_N_TRAIN_PER_SUBJECT" in labeling_py
    assert "_FRACTION_NEW_POOL" in labeling_py
    assert "_item_in_new_pool" in labeling_py
    assert "is_new_bc" in labeling_py
    assert "in_new_pool" in labeling_py
    assert "eligible" in labeling_py
    assert "anchoring" in labeling_py
    assert "_enqueue_for_batch" in labeling_py
    assert "10.0 * anchoring" in labeling_py
    assert "1000.0 * novelty" not in labeling_py, (
        "labeling.py must use dual-pool stratification, not multiplied novelty"
    )
    assert "from model import _baseline_logit" not in labeling_py
    print("[OK] labeling.py structural assertions hold")
    print()


if __name__ == "__main__":
    for z in ZIPS:
        verify(z)
