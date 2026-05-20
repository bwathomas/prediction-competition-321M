"""Pack the bisection bundles into ZIPs ready for Codabench upload.

Produces three tiny ZIPs in C:\\Users\\benja\\Downloads\\submission\\:

  * submission_bisection_01_no_torch.zip
  * submission_bisection_02_torch_cuda.zip
  * submission_bisection_03_state_share.zip

Each bundle contains only model.py + labeling.py (POSIX-style paths,
ZIP_DEFLATED, no __pycache__). No checkpoints, no NN cache, no models.txt
-- these are pure diagnostic bundles. See docs/batching_bisection.md
for what each tests and how to read its progress file.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BISECTION_DIR = ROOT / "bisection"
DST_DIR = Path(r"C:\Users\benja\Downloads\submission")

BUNDLES = [
    ("01_no_torch", "submission_bisection_01_no_torch.zip"),
    ("02_torch_cuda", "submission_bisection_02_torch_cuda.zip"),
    ("03_state_share", "submission_bisection_03_state_share.zip"),
]


def pack_one(src_dir: Path, dst_zip: Path) -> int:
    if dst_zip.exists():
        dst_zip.unlink()
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(
        dst_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_dir():
                continue
            if "__pycache__" in p.parts:
                continue
            # POSIX-style arcname relative to src_dir. The platform unzips
            # on Linux and is sensitive to backslashes in paths.
            arcname = p.relative_to(src_dir).as_posix()
            zf.write(p, arcname=arcname)
            n += 1
    return n


def main() -> int:
    if not BISECTION_DIR.exists():
        print(f"ERROR: bisection dir missing: {BISECTION_DIR}", file=sys.stderr)
        return 1

    for sub, zip_name in BUNDLES:
        src = BISECTION_DIR / sub
        dst = DST_DIR / zip_name
        if not src.exists():
            print(f"  ! skipping (source missing): {src}")
            continue
        n = pack_one(src, dst)
        size_kb = dst.stat().st_size / 1024.0
        print(f"  [OK] {zip_name:48s}  {n} files,  {size_kb:>7.2f} KB")

        # Quick verification: peek inside the produced zip.
        with zipfile.ZipFile(dst, "r") as zf:
            names = sorted(i.filename for i in zf.infolist())
            assert "model.py" in names, f"{zip_name} missing model.py"
            assert "labeling.py" in names, f"{zip_name} missing labeling.py"
            for nm in names:
                assert "\\" not in nm, f"{zip_name} has backslash path: {nm}"
            print(f"       -> {names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
