"""Convert a `# %%`-style paired Python file into a Jupyter .ipynb.

This is a tiny tool with zero third-party dependencies. We keep it in the
repo so the notebook is reproducible from source: edit the .py in Cursor,
run this script, get the .ipynb back.

Usage:

    py scripts/build_ipynb_from_py.py notebooks/a100_ablation_notebook.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CELL_HEADER = re.compile(r"^\s*#\s*%%(.*)$")


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Yield (cell_type, source_lines) tuples.

    `# %% [markdown]` or `# %%[markdown]` -> markdown cell (lines stripped of `#` prefix)
    `# %%` -> code cell (verbatim)
    """
    cells: list[tuple[str, list[str]]] = []
    current_type = "code"
    current: list[str] = []
    started = False
    for line in text.splitlines():
        m = CELL_HEADER.match(line)
        if m:
            if started:
                cells.append((current_type, current))
            current_type = (
                "markdown" if "markdown" in (m.group(1) or "").lower() else "code"
            )
            current = []
            started = True
            continue
        current.append(line)
    if started:
        cells.append((current_type, current))
    elif current:
        cells.append(("code", current))
    return cells


def build_notebook(cells: list[tuple[str, list[str]]]) -> dict:
    nb_cells = []
    for cell_type, lines in cells:
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        if not lines:
            continue
        if cell_type == "markdown":
            md_lines = []
            for ln in lines:
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln.startswith("#"):
                    md_lines.append(ln[1:].lstrip())
                else:
                    md_lines.append(ln)
            source = "\n".join(md_lines)
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": source.splitlines(keepends=True),
                }
            )
        else:
            source = "\n".join(lines)
            nb_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": source.splitlines(keepends=True),
                }
            )
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source .py file with `# %%` cell markers")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .ipynb (default: same path with .ipynb extension)",
    )
    args = parser.parse_args(argv)
    src = Path(args.source)
    out = Path(args.out) if args.out else src.with_suffix(".ipynb")
    text = src.read_text(encoding="utf-8")
    cells = split_cells(text)
    nb = build_notebook(cells)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code cells, "
          f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')} markdown cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
