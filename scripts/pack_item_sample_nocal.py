"""Repack ``submission_item_sample.zip`` with the runtime calibrator disabled.

The base bundle's ``predict()`` re-fits a beta / Platt / intercept-only
calibrator on every distinct ``labeled`` fingerprint and then composes
``_CALIBRATOR.apply(...)`` on top of the head's uncalibrated probability.
For ablation diagnostics, we want the head's *raw* output -- "run as
though nothing has been revealed".  Two minimal edits to ``model.py``
accomplish that, leaving the rest of the runtime (streamed flush, head
weights, baseline-logit acquisition, offline env vars) untouched:

  1. In ``predict()``, replace the ``if labeled: ...`` block (which fits
     the calibrator and clears the prob cache when the fingerprint
     changes) with a no-op.  The cache is preserved across calls; the
     calibrator stays at its module-init default (``identity``).

  2. Replace the ``p = _CALIBRATOR.apply(p)`` line with a comment so we
     also drop the (no-op-when-identity but still pointless) call.

Bonus: tag ``runtime_meta.json`` with ``"calibration_disabled": true`` and
rename the architecture tag to ``streamed_flush_v1+nocal``.

Run from the repo root:

    py scripts/pack_item_sample_nocal.py

Reads:  C:/Users/benja/Downloads/submission/submission_item_sample.zip
Writes: C:/Users/benja/Downloads/submission/submission_item_sample_nocal.zip
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


SRC = Path(r"C:\Users\benja\Downloads\submission\submission_item_sample.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_item_sample_nocal.zip")


OLD_FIT_BLOCK = """        if labeled:
            fp = _labeled_fingerprint(labeled)
            if fp != _LAST_LABELED_FINGERPRINT:
                _LAST_LABELED_FINGERPRINT = fp
                _PROB_CACHE.clear()
                _CALIBRATOR = _Calibrator(META.get("default_calibrator"))
                _CALIBRATOR.fit_from_labeled(labeled)
"""

NEW_FIT_BLOCK = """        # Calibration intentionally disabled in this variant: ignore
        # ``labeled`` entirely so the runtime returns the head's raw
        # uncalibrated probability as if no labels had been revealed.
        # (The module-init ``_CALIBRATOR`` is left at its META default,
        # ``{"kind": "identity"}``.)
        _ = labeled  # touch so the linter doesn't flag the unused arg
"""

OLD_APPLY_LINE = "        p = _CALIBRATOR.apply(p)\n"
NEW_APPLY_LINE = "        # p = _CALIBRATOR.apply(p)  # disabled: uncalibrated variant\n"


def patch_model_py(raw: bytes) -> bytes:
    text = raw.decode("utf-8")

    if OLD_FIT_BLOCK not in text:
        raise SystemExit(
            "pack_item_sample_nocal: could not find the calibrator-fit block in "
            "model.py; the base bundle has drifted from the expected format."
        )
    if text.count(OLD_FIT_BLOCK) != 1:
        raise SystemExit(
            "pack_item_sample_nocal: OLD_FIT_BLOCK is non-unique in model.py "
            f"(found {text.count(OLD_FIT_BLOCK)} occurrences)."
        )
    if OLD_APPLY_LINE not in text:
        raise SystemExit(
            "pack_item_sample_nocal: could not find the _CALIBRATOR.apply(p) line."
        )
    if text.count(OLD_APPLY_LINE) != 1:
        raise SystemExit(
            "pack_item_sample_nocal: OLD_APPLY_LINE is non-unique in model.py "
            f"(found {text.count(OLD_APPLY_LINE)} occurrences)."
        )

    text = text.replace(OLD_FIT_BLOCK, NEW_FIT_BLOCK, 1)
    text = text.replace(OLD_APPLY_LINE, NEW_APPLY_LINE, 1)
    return text.encode("utf-8")


def patch_labeling_py(raw: bytes) -> bytes:
    """Drop docstring lines that reference post-hoc calibration."""
    text = raw.decode("utf-8")
    text = text.replace(
        "     where the head's a priori uncertainty is highest -- exactly where\n"
        "     a small Platt / beta calibrator fitted on those labels has the\n"
        "     largest leverage on the post-calibration log-loss.\n",
        "     where the head's a priori uncertainty is highest.\n",
    )
    text = text.replace(
        "(streamed-flush + uncertainty).",
        "(streamed-flush + uncertainty, no post-hoc calibration).",
        1,
    )
    return text.encode("utf-8")


def patch_meta_json(raw: bytes) -> bytes:
    meta = json.loads(raw)
    meta["runtime_architecture"] = str(
        meta.get("runtime_architecture", "streamed_flush_v1")
    ) + "+nocal"
    meta["calibration_disabled"] = True
    meta.setdefault("default_calibrator", {})["kind"] = "identity"
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"pack_item_sample_nocal: source bundle not found: {SRC}")
    if DST.exists():
        DST.unlink()

    with zipfile.ZipFile(SRC, "r") as src_zf, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst_zf:
        for info in src_zf.infolist():
            data = src_zf.read(info.filename)
            if info.filename.endswith("model.py"):
                data = patch_model_py(data)
                print(
                    f"  patched {info.filename} "
                    f"({len(data)} bytes, calibrator-fit + apply disabled)"
                )
            elif info.filename.endswith("labeling.py"):
                data = patch_labeling_py(data)
                print(f"  patched {info.filename} (docstring: no calibration)")
            elif info.filename.endswith("runtime_meta.json"):
                data = patch_meta_json(data)
                print(f"  patched {info.filename} (tagged +nocal)")
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.external_attr = info.external_attr
            dst_zf.writestr(new_info, data)

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
