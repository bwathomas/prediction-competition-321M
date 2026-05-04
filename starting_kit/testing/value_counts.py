"""Compute unique-value counts for `benchmark`, `condition`, and `subject_content`
across every response row in Data/ and write them to a single CSV.

Run from the `starting_kit/` directory:
    py testing/value_counts.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUT_CSV = Path(__file__).resolve().parent / "value_counts.csv"

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def is_response_file(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )


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

    benchmark_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    subject_id_counts: Counter[str] = Counter()

    for path in response_files:
        df = pd.read_parquet(
            path, columns=["benchmark_id", "test_condition", "subject_id"]
        )
        condition = df["test_condition"].fillna("none").replace("", "none")
        benchmark_counts.update(df["benchmark_id"].astype(str).tolist())
        condition_counts.update(condition.astype(str).tolist())
        subject_id_counts.update(df["subject_id"].astype(str).tolist())

    benchmarks_df = pd.read_parquet(DATA_DIR / "benchmarks.parquet")
    canonical_benchmark = {
        row["benchmark_id"]: row["benchmark_id"] for _, row in benchmarks_df.iterrows()
    }
    benchmark_counts_canonical: Counter[str] = Counter()
    for raw, n in benchmark_counts.items():
        benchmark_counts_canonical[canonical_benchmark.get(raw, raw)] += n

    subjects_df = pd.read_parquet(DATA_DIR / "subjects.parquet")
    subject_by_id = {row["subject_id"]: row.to_dict() for _, row in subjects_df.iterrows()}
    subject_content_counts: Counter[str] = Counter()
    for sid, n in subject_id_counts.items():
        rendered = render_subject_content(subject_by_id.get(sid, {}), sid)
        subject_content_counts[rendered] += n

    rows = []
    for value, n in sorted(benchmark_counts_canonical.items(), key=lambda kv: -kv[1]):
        rows.append({"field": "benchmark", "value": value, "count": n})
    for value, n in sorted(condition_counts.items(), key=lambda kv: -kv[1]):
        rows.append({"field": "condition", "value": value, "count": n})
    for value, n in sorted(subject_content_counts.items(), key=lambda kv: -kv[1]):
        rows.append({"field": "subject_content", "value": value, "count": n})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    total = sum(benchmark_counts_canonical.values())
    print(f"Total response rows scanned: {total:,}")
    print(f"Unique benchmark values:        {len(benchmark_counts_canonical):,}")
    print(f"Unique condition values:        {len(condition_counts):,}")
    print(f"Unique subject_content values:  {len(subject_content_counts):,}")
    print(f"\nWrote {len(out):,} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
