"""Inspect the structure of `subject_content` strings.

For each field that `render_subject_content()` may emit, report:
  - how many of the 909 observed subjects actually have that field
  - how many response rows that translates to (weighted)
  - how many unique values that field takes (across observed subjects)

Also dumps two CSVs:
  testing/subject_content_field_summary.csv  - one row per field
  testing/subject_content_field_values.csv   - one row per (field, value, ...)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
HERE = Path(__file__).resolve().parent

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

RENDERED_FIELDS: list[tuple[str, str]] = [
    ("display_name", "Name"),
    ("provider", "Organization"),
    ("params", "Parameters"),
    ("release_date", "Released"),
    ("family", "Family"),
]


def is_response_file(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )


def main() -> None:
    response_files = sorted(p for p in DATA_DIR.iterdir() if is_response_file(p))

    subject_id_counts: Counter[str] = Counter()
    for path in response_files:
        sids = pd.read_parquet(path, columns=["subject_id"])["subject_id"].astype(str)
        subject_id_counts.update(sids.tolist())

    subjects_df = pd.read_parquet(DATA_DIR / "subjects.parquet")
    print(f"subjects.parquet columns ({len(subjects_df.columns)}): "
          f"{list(subjects_df.columns)}")
    print(f"subjects.parquet rows: {len(subjects_df):,}")
    print(f"distinct subject_ids appearing in responses: {len(subject_id_counts):,}\n")

    observed_ids = set(subject_id_counts)
    obs = subjects_df[subjects_df["subject_id"].isin(observed_ids)].copy()
    print(f"Of {len(observed_ids):,} observed subject_ids, "
          f"{len(obs):,} are present in subjects.parquet "
          f"(missing: {len(observed_ids) - len(obs):,})\n")

    summary_rows = []
    value_rows = []
    for src_col, label in RENDERED_FIELDS:
        if src_col not in obs.columns:
            present_subjects = 0
            present_rows = 0
            unique_values = 0
        else:
            mask = obs[src_col].notna() & (obs[src_col].astype(str) != "")
            present_subj_ids = set(obs.loc[mask, "subject_id"].astype(str))
            present_subjects = len(present_subj_ids)
            present_rows = sum(subject_id_counts[sid] for sid in present_subj_ids)

            value_to_subjects: dict[str, int] = {}
            value_to_rows: dict[str, int] = {}
            for _, r in obs.loc[mask, ["subject_id", src_col]].iterrows():
                v = str(r[src_col])
                value_to_subjects[v] = value_to_subjects.get(v, 0) + 1
                value_to_rows[v] = value_to_rows.get(v, 0) + subject_id_counts[str(r["subject_id"])]
            unique_values = len(value_to_subjects)

            for v, n_subj in sorted(value_to_subjects.items(), key=lambda kv: -kv[1]):
                value_rows.append(
                    {
                        "field": label,
                        "source_column": src_col,
                        "value": v,
                        "n_subjects_with_value": n_subj,
                        "n_response_rows_with_value": value_to_rows[v],
                    }
                )

        summary_rows.append(
            {
                "field": label,
                "source_column": src_col,
                "n_observed_subjects_with_field": present_subjects,
                "pct_observed_subjects_with_field": round(
                    100 * present_subjects / max(1, len(obs)), 2
                ),
                "n_response_rows_with_field": present_rows,
                "pct_response_rows_with_field": round(
                    100 * present_rows / max(1, sum(subject_id_counts.values())), 2
                ),
                "n_unique_values": unique_values,
            }
        )

    summary = pd.DataFrame(summary_rows)
    print("Per-field summary:")
    print(summary.to_string(index=False))

    summary.to_csv(HERE / "subject_content_field_summary.csv", index=False)
    pd.DataFrame(value_rows).to_csv(HERE / "subject_content_field_values.csv", index=False)
    print(f"\nWrote {HERE / 'subject_content_field_summary.csv'}")
    print(f"Wrote {HERE / 'subject_content_field_values.csv'} ({len(value_rows):,} rows)")


if __name__ == "__main__":
    main()
