"""Count response rows across the public training data and export a 10-row sample.

Run from the `starting_kit/` directory:
    py testing/export_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUT_CSV = Path(__file__).resolve().parent / "sample_10_rows.csv"

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def is_response_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".parquet"):
        return False
    if name in REGISTRY_FILES:
        return False
    if name.endswith("_traces.parquet"):
        return False
    return True


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    for key, label in (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ):
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def main() -> None:
    response_files = sorted(p for p in DATA_DIR.iterdir() if is_response_file(p))

    per_file_counts: dict[str, int] = {}
    total_rows = 0
    for path in response_files:
        n = pd.read_parquet(path, columns=["item_id"]).shape[0]
        per_file_counts[path.name] = n
        total_rows += n

    print("Per-benchmark response row counts:")
    for name, n in per_file_counts.items():
        print(f"  {name:<32} {n:>10,}")
    print(f"\nTOTAL response rows across {len(response_files)} benchmarks: {total_rows:,}")

    subjects = pd.read_parquet(DATA_DIR / "subjects.parquet")
    items = pd.read_parquet(DATA_DIR / "items.parquet")
    benchmarks = pd.read_parquet(DATA_DIR / "benchmarks.parquet")
    subjects_by_id = {row["subject_id"]: row.to_dict() for _, row in subjects.iterrows()}
    items_by_id = {row["item_id"]: row.to_dict() for _, row in items.iterrows()}
    benchmarks_by_id = {row["benchmark_id"]: row.to_dict() for _, row in benchmarks.iterrows()}

    sample_src = next(p for p in response_files if p.name == "mathvista_mini.parquet")
    sample = pd.read_parquet(sample_src).head(10)

    rendered_rows = []
    for _, row in sample.iterrows():
        subject = subjects_by_id.get(row["subject_id"], {})
        item = items_by_id.get(row["item_id"], {})
        benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
        rendered_rows.append(
            {
                "benchmark": benchmark.get("benchmark_id") or row["benchmark_id"],
                "condition": row["test_condition"] or "none",
                "subject_content": render_subject_content(subject, row["subject_id"]),
                "item_content": item.get("content"),
                "label": row["response"],
                "subject_id": row["subject_id"],
                "item_id": row["item_id"],
                "trial": row["trial"],
                "correct_answer": row["correct_answer"],
            }
        )

    pd.DataFrame(rendered_rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote 10-row sample (from {sample_src.name}) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
