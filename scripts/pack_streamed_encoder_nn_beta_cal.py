"""Pack a streamed-encoder + NN bundle with tiered beta calibration + smart labeling.

Starts from ``submission_streamed_encoder_nn.zip`` and patches two things
from ``submission_k_factor.zip``:

  1. ``model.py`` -- swap in the tiered ``_Calibrator`` block (beta /
     Platt / intercept) AND graft the ``_baseline_logit`` fast-path so
     ``labeling.py`` can return a real uncertainty score.
  2. ``labeling.py`` -- replace the enqueue-only constant-0.0 variant with
     the streamed-flush + uncertainty version (enqueue + ``-|p-0.5|``).

Everything else (HF offline env vars, streamed-flush plumbing, NN cache,
checkpoint, runtime_meta.json) is copied verbatim from the base bundle.

The tiered calibrator selects automatically by the number ``N`` of
revealed labels passed in ``predict(input, labeled)``:

    N >= 60  -> 3-parameter beta calibration (Kull/Filho/Flach 2017)
                Newton + 3x3 Hessian, ridge, backtracking line search.
    N >= 30  -> 2-parameter temperature + intercept (Platt).
    N >=  5  -> 1-parameter intercept-only logit shift.
    else     -> identity (the previous baseline).

On the competition setting (15 categories x K=5 = 75 revealed labels)
the operative branch is beta calibration -- a strictly more expressive
family than temp_intercept, which can correct both the slope and the
asymmetric tail behavior the trained head exhibits when N is large
enough to fit it stably.

Source bundle:
  - submission_streamed_encoder_nn.zip  (donor of every file except
    model.py's calibrator block)

Donor bundle:
  - submission_k_factor.zip             (source of the tiered calibrator,
    _baseline_logit fast-path, and uncertainty labeling.py)
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

SUB_DIR = Path(r"C:/Users/benja/Downloads/submission")
BASE_ZIP = SUB_DIR / "submission_streamed_encoder_nn.zip"
DONOR_ZIP = SUB_DIR / "submission_k_factor.zip"
OUT_ZIP = SUB_DIR / "submission_streamed_encoder_nn_beta_cal.zip"


# Markers that bracket the exact region we replace inside model.py.  The
# OLD region starts at the ``Calibrator (intercept-only ...`` banner and
# ends just before the next top-level banner (``Training-item cache
# (nearest-neighbor lookup over training items).``).  The donor region
# starts at the same calibrator banner and ends just before the next
# top-level banner in the k-factor bundle, which is ``# Tokenized text
# precompute helpers``.  We do not pattern-match on those next-banner
# strings, we just slice from the END of the OLD calibrator block to the
# beginning of the next ``# ---`` banner in each file -- this is robust
# against minor wording differences in the surrounding sections.
CAL_BANNER_RE = re.compile(
    r"^# -{50,}\n# Calibrator [^\n]*\n# -{50,}\n",
    flags=re.MULTILINE,
)


def _extract_calibrator_block(model_py: str, *, label: str) -> tuple[int, int]:
    """Return (start, end) char offsets covering the calibrator section.

    ``start`` is the first char of the ``# -----`` banner; ``end`` is the
    first char of the *next* ``# -----`` banner.  The caller can slice
    [start:end) to get exactly the calibrator block (banner + helpers).
    """
    m = CAL_BANNER_RE.search(model_py)
    if not m:
        raise RuntimeError(
            f"{label}: could not find the calibrator banner in model.py"
        )
    start = m.start()
    # The next top-level banner is a line of >=50 dashes, optionally
    # preceded by a newline.  We look from the END of the calibrator
    # banner forward.
    next_banner = re.search(
        r"\n\n\n# -{50,}\n",
        model_py[m.end():],
    )
    if not next_banner:
        raise RuntimeError(
            f"{label}: could not find the next banner after the calibrator block"
        )
    end = m.end() + next_banner.start() + 1  # keep the trailing newlines
    return start, end


def _patch_calibrator(base_model_py: str, donor_model_py: str) -> str:
    b_start, b_end = _extract_calibrator_block(base_model_py, label="base")
    d_start, d_end = _extract_calibrator_block(donor_model_py, label="donor")
    base_block = base_model_py[b_start:b_end]
    donor_block = donor_model_py[d_start:d_end]
    # Sanity-checks: the OLD block must NOT advertise a beta branch (we
    # are upgrading from it), and the NEW block MUST advertise one.
    if "beta calibration" in base_block.lower():
        raise RuntimeError(
            "base bundle already advertises beta calibration -- nothing to do"
        )
    if "beta calibration" not in donor_block.lower():
        raise RuntimeError(
            "donor calibrator block does not mention beta calibration -- "
            "the donor bundle may not be the k-factor variant we expect"
        )
    patched = base_model_py[:b_start] + donor_block + base_model_py[b_end:]
    # Sanity: the patched file should have all four fitters and the
    # beta-calibration loss helper.
    for needle in (
        "def _fit_intercept_only",
        "def _fit_temp_intercept",
        "def _fit_beta_calibration",
        "def _beta_calibration_loss",
        '"kind": "beta"',
    ):
        if needle not in patched:
            raise RuntimeError(f"patched model.py is missing {needle!r}")
    return patched


BASELINE_START = "# --- Fast-path baseline logits for acquisition_function"
BASELINE_END_MARKER = '\n\nLOG.info(\n    "Submission ready:'


def _extract_baseline_block(model_py: str, *, label: str) -> str:
    start = model_py.find(BASELINE_START)
    if start < 0:
        raise RuntimeError(f"{label}: could not find baseline-logit banner")
    end = model_py.find(BASELINE_END_MARKER, start)
    if end < 0:
        raise RuntimeError(f"{label}: could not find end marker after baseline block")
    block = model_py[start:end]
    if "def _baseline_logit" not in block:
        raise RuntimeError(f"{label}: baseline block missing _baseline_logit")
    return block


def _patch_baseline_logits(model_py: str, donor_model_py: str) -> str:
    if "def _baseline_logit" in model_py:
        print("[INFO] model.py already has _baseline_logit; skipping graft")
        return model_py
    block = _extract_baseline_block(donor_model_py, label="donor")
    anchor = "_LAST_LABELED_FINGERPRINT: tuple | None = None\n\n\nLOG.info("
    if anchor not in model_py:
        raise RuntimeError(
            "base model.py: could not find insertion anchor for baseline logits "
            "(expected _LAST_LABELED_FINGERPRINT followed by LOG.info Submission ready)"
        )
    patched = model_py.replace(
        anchor,
        "_LAST_LABELED_FINGERPRINT: tuple | None = None\n\n\n" + block + "\n\nLOG.info(",
        1,
    )
    for needle in ("_BASELINE_MU", "_SUBJECT_BASELINE", "_BC_BASELINE", "def _baseline_logit"):
        if needle not in patched:
            raise RuntimeError(f"patched model.py is missing {needle!r}")
    return patched


def _labeling_from_donor(donor_labeling_py: str) -> str:
    """Return k-factor labeling.py with a one-line context tweak for encoder+NN."""
    if "_baseline_logit" not in donor_labeling_py:
        raise RuntimeError("donor labeling.py does not import _baseline_logit")
    if "_enqueue_for_batch" not in donor_labeling_py:
        raise RuntimeError("donor labeling.py does not call _enqueue_for_batch")
    # Keep donor logic verbatim; only adjust the module docstring opener so
    # the shipped bundle is self-describing.
    return donor_labeling_py.replace(
        "(streamed-flush + uncertainty).",
        "(streamed-encoder + NN + uncertainty).",
        1,
    )


def main() -> int:
    if not BASE_ZIP.exists():
        print(f"ERROR: base bundle not found: {BASE_ZIP}", file=sys.stderr)
        return 1
    if not DONOR_ZIP.exists():
        print(f"ERROR: donor bundle not found: {DONOR_ZIP}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(BASE_ZIP, "r") as base_zf:
        base_names = base_zf.namelist()
        with base_zf.open("model.py", "r") as fh:
            base_model_py = fh.read().decode("utf-8")
    with zipfile.ZipFile(DONOR_ZIP, "r") as donor_zf:
        with donor_zf.open("model.py", "r") as fh:
            donor_model_py = fh.read().decode("utf-8")
        with donor_zf.open("labeling.py", "r") as fh:
            donor_labeling_py = fh.read().decode("utf-8")

    patched = _patch_calibrator(base_model_py, donor_model_py)
    patched = _patch_baseline_logits(patched, donor_model_py)
    labeling_py = _labeling_from_donor(donor_labeling_py)

    # Show a one-line diff summary so the user can confirm the size delta.
    delta = len(patched) - len(base_model_py)
    print(
        f"[INFO] model.py: base={len(base_model_py):,} bytes, "
        f"patched={len(patched):,} bytes, delta={delta:+,} bytes"
    )
    print(f"[INFO] labeling.py: {len(labeling_py):,} bytes (uncertainty + enqueue)")

    # Compile-check the patched source so we never ship a syntax error.
    import ast

    ast.parse(patched)
    ast.parse(labeling_py)
    print("[INFO] patched model.py and labeling.py parse as valid Python")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(BASE_ZIP, "r") as src_zf, zipfile.ZipFile(
        OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as dst_zf:
        for name in base_names:
            info = src_zf.getinfo(name)
            if name == "model.py":
                # Preserve the directory entries' timestamps but write
                # fresh content.
                new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                new_info.compress_type = zipfile.ZIP_DEFLATED
                new_info.external_attr = info.external_attr
                dst_zf.writestr(new_info, patched.encode("utf-8"))
            elif name == "labeling.py":
                new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                new_info.compress_type = zipfile.ZIP_DEFLATED
                new_info.external_attr = info.external_attr
                dst_zf.writestr(new_info, labeling_py.encode("utf-8"))
            else:
                with src_zf.open(name, "r") as fh:
                    payload = fh.read()
                # Copy directory entries (size 0) as ZipInfo so empty
                # dirs are preserved.
                if info.is_dir():
                    dst_zf.writestr(info, b"")
                else:
                    dst_zf.writestr(info, payload)

    final_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"[OK] wrote {OUT_ZIP} ({final_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
