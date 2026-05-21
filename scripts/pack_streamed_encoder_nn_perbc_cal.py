"""Pack a streamed-encoder + NN bundle with per-benchmark calibration + novelty labeling.

This is the next iteration on top of ``submission_streamed_encoder_nn_beta_cal``.
The calibrator + labeling sources are factored into
``scripts/_perbc_cal_sources.py``; this script just sets the source and
destination paths and applies the transformation.

Source bundle:
  - submission_streamed_encoder_nn_beta_cal.zip
Output bundle:
  - submission_streamed_encoder_nn_perbc_cal.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _perbc_cal_sources as src  # noqa: E402

SUB_DIR = Path(r"C:/Users/benja/Downloads/submission")
BASE_ZIP = SUB_DIR / "submission_streamed_encoder_nn_beta_cal.zip"
OUT_ZIP = SUB_DIR / "submission_streamed_encoder_nn_perbc_cal.zip"


def main() -> int:
    if not BASE_ZIP.exists():
        print("ERROR: base bundle not found: " + str(BASE_ZIP), file=sys.stderr)
        return 1

    with zipfile.ZipFile(BASE_ZIP, "r") as base_zf:
        base_names = base_zf.namelist()
        base_model_py = base_zf.read("model.py").decode("utf-8")

    src.required_model_py_prereqs(base_model_py)
    patched = src.replace_calibrator_block(base_model_py)
    patched = src.replace_apply_site(patched)
    src.sanity_check_model_py(patched)
    src.sanity_check_labeling_py(src.NEW_LABELING_PY)

    delta = len(patched) - len(base_model_py)
    print("[INFO] model.py: base={:,} bytes, patched={:,} bytes, delta={:+,} bytes".format(
        len(base_model_py), len(patched), delta
    ))

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(BASE_ZIP, "r") as src_zf, zipfile.ZipFile(
        OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as dst_zf:
        for name in base_names:
            info = src_zf.getinfo(name)
            if name == "model.py":
                ni = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                ni.compress_type = zipfile.ZIP_DEFLATED
                ni.external_attr = info.external_attr
                dst_zf.writestr(ni, patched.encode("utf-8"))
            elif name == "labeling.py":
                ni = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                ni.compress_type = zipfile.ZIP_DEFLATED
                ni.external_attr = info.external_attr
                dst_zf.writestr(ni, src.NEW_LABELING_PY.encode("utf-8"))
            elif name == "artifacts/runtime_meta.json":
                ni = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                ni.compress_type = zipfile.ZIP_DEFLATED
                ni.external_attr = info.external_attr
                dst_zf.writestr(ni, src.clear_calibration_disabled_in_meta(src_zf.read(name)))
            else:
                if info.is_dir():
                    dst_zf.writestr(info, b"")
                else:
                    dst_zf.writestr(info, src_zf.read(name))

    print("[OK] wrote {} ({:.2f} MB)".format(
        OUT_ZIP, OUT_ZIP.stat().st_size / (1024 * 1024)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
