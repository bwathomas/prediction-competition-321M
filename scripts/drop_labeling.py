"""Strip ``labeling.py`` out of an existing submission zip.

Re-packs ``submission_judge_slim.zip`` into
``submission_judge_no_labeling.zip``, excluding ``labeling.py`` and any
``__pycache__`` entry. Everything else (model.py, models.txt, artifacts/,
cache/) is copied through verbatim with the original POSIX archive paths.

When ``labeling.py`` is absent the platform falls back to a random
top-K-per-category labeling pass, exactly as documented in
``starting_kit/README.md``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission_judge_slim.zip")
DST = Path(r"C:\Users\benja\Downloads\submission_judge_no_labeling.zip")

DROP_NAMES = {"labeling.py"}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source zip missing: {SRC}")
    if DST.exists():
        DST.unlink()

    kept: list[tuple[str, int]] = []
    dropped: list[str] = []

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            name = info.filename
            if name in DROP_NAMES or name.endswith("/labeling.py"):
                dropped.append(name)
                continue
            if "__pycache__" in name.split("/"):
                dropped.append(name)
                continue
            data = zin.read(name)
            zout.writestr(info, data)
            kept.append((name, len(data)))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"Wrote {DST}  ({size_mb:.2f} MB)")
    if dropped:
        print("Dropped entries:")
        for n in dropped:
            print(f"  - {n}")
    else:
        print("No entries dropped (source had no labeling.py?).")
    print("Kept entries:")
    for n, sz in kept:
        print(f"  {n:50s} {sz:>14,d} B")

    with zipfile.ZipFile(DST, "r") as zf:
        names = {i.filename for i in zf.infolist()}
        assert "labeling.py" not in names, "labeling.py still present"
        assert "model.py" in names, "model.py missing"
        assert any(n.startswith("artifacts/") for n in names), "artifacts/ missing"
    print("\nVerification: labeling.py absent; model.py + artifacts/ present.")


if __name__ == "__main__":
    main()
