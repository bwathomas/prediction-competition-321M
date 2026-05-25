"""Tiny .py -> .ipynb converter for the percent-format notebooks.

Reads a script with ``# %%`` (code) / ``# %% [markdown]`` (md) markers
and writes the equivalent .ipynb. Used to produce
``notebooks/qwen8b_minimalist.ipynb`` from
``notebooks/qwen8b_minimalist.py``. We avoid jupytext as a hard
dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def py_to_ipynb(in_path: Path, out_path: Path) -> None:
    text = in_path.read_text(encoding="utf-8")
    cells: list[dict] = []
    cur_kind: str = "code"
    cur_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur_lines
        if not cur_lines:
            return
        # Strip leading blank lines so the cell starts on real code.
        while cur_lines and cur_lines[0].strip() == "":
            cur_lines.pop(0)
        # Drop trailing blank lines too.
        while cur_lines and cur_lines[-1].strip() == "":
            cur_lines.pop()
        if not cur_lines:
            cur_lines = []
            return
        if cur_kind == "markdown":
            # Strip the leading "# " from each line so the markdown
            # rendering doesn't show "# " everywhere.
            md_lines = []
            for ln in cur_lines:
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln == "#":
                    md_lines.append("")
                else:
                    md_lines.append(ln)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [s + "\n" for s in md_lines[:-1]] + [md_lines[-1]],
            })
        else:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [s + "\n" for s in cur_lines[:-1]] + [cur_lines[-1]],
            })
        cur_lines = []

    for raw in text.splitlines():
        if raw.startswith("# %% [markdown]"):
            _flush()
            cur_kind = "markdown"
            continue
        if raw.startswith("# %%"):
            _flush()
            cur_kind = "code"
            continue
        cur_lines.append(raw)
    _flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: py_to_ipynb.py <input.py> <output.ipynb>\n")
        sys.exit(2)
    in_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    py_to_ipynb(in_path, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
