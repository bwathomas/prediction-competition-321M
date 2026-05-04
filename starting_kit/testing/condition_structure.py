"""Analyze the structure of `condition` strings.

Hypothesis from earlier inspection: conditions look like
    none
    key1=value1
    key1=value1|key2=value2|...

This script verifies that, enumerates the keys ("fields"), and counts unique
values per key. It also flags any conditions that do NOT match the pattern.

Outputs:
  testing/condition_field_summary.csv  - one row per field
  testing/condition_field_values.csv   - one row per (field, value)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
HERE = Path(__file__).resolve().parent

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def is_response_file(p: Path) -> bool:
    n = p.name
    return (
        n.endswith(".parquet")
        and n not in REGISTRY_FILES
        and not n.endswith("_traces.parquet")
    )


def normalize(c: object) -> str:
    if c is None:
        return "none"
    s = str(c)
    if s == "" or s.lower() == "nan":
        return "none"
    return s


def main() -> None:
    response_files = sorted(p for p in DATA_DIR.iterdir() if is_response_file(p))

    cond_counts: Counter[str] = Counter()
    cond_to_benchmarks: defaultdict[str, set[str]] = defaultdict(set)
    for path in response_files:
        df = pd.read_parquet(path, columns=["test_condition", "benchmark_id"])
        df["test_condition"] = df["test_condition"].fillna("none").replace("", "none")
        cond_counts.update(df["test_condition"].astype(str).tolist())
        for c, b in zip(df["test_condition"].astype(str), df["benchmark_id"].astype(str)):
            cond_to_benchmarks[c].add(b)

    total_rows = sum(cond_counts.values())
    n_unique = len(cond_counts)

    formulaic = 0
    nonformulaic: list[str] = []
    field_value_subjects: defaultdict[str, Counter[str]] = defaultdict(Counter)
    field_value_rows: defaultdict[str, Counter[str]] = defaultdict(Counter)
    field_cooccurrence: defaultdict[frozenset[str], int] = defaultdict(int)
    none_rows = cond_counts.get("none", 0)

    for cond, n_rows in cond_counts.items():
        if cond == "none":
            field_cooccurrence[frozenset()] += 1
            continue
        parts = cond.split("|")
        ok = all("=" in p for p in parts)
        if not ok:
            nonformulaic.append(cond)
            continue
        formulaic += 1
        keys_in_this_cond = []
        for part in parts:
            k, _, v = part.partition("=")
            field_value_subjects[k][v] += 1
            field_value_rows[k][v] += n_rows
            keys_in_this_cond.append(k)
        field_cooccurrence[frozenset(keys_in_this_cond)] += 1

    print(f"Total response rows:          {total_rows:,}")
    print(f"Unique condition strings:     {n_unique:,}")
    print(f"  - 'none':                   1  ({none_rows:,} rows = "
          f"{100*none_rows/total_rows:.1f}%)")
    print(f"  - formulaic key=value[|...]: {formulaic:,}")
    print(f"  - non-formulaic:             {len(nonformulaic):,}")
    if nonformulaic:
        for c in nonformulaic[:20]:
            print(f"      ! {c!r}")

    print()
    print(f"Number of distinct keys (fields): {len(field_value_subjects)}")
    print()
    summary_rows = []
    print(f"{'field':<10} {'#unique values':>14} {'#cond strings':>14} "
          f"{'#response rows':>16}")
    print("-" * 60)
    for k in sorted(field_value_subjects, key=lambda x: -sum(field_value_rows[x].values())):
        n_vals = len(field_value_subjects[k])
        n_cond = sum(field_value_subjects[k].values())
        n_rows = sum(field_value_rows[k].values())
        print(f"{k:<10} {n_vals:>14,} {n_cond:>14,} {n_rows:>16,}")
        summary_rows.append(
            {
                "field": k,
                "n_unique_values": n_vals,
                "n_condition_strings_using_field": n_cond,
                "n_response_rows_using_field": n_rows,
            }
        )

    print()
    print("Field co-occurrence patterns (which keys appear together):")
    for keyset, n in sorted(field_cooccurrence.items(), key=lambda kv: -kv[1]):
        label = "|".join(sorted(keyset)) if keyset else "(no keys -> 'none')"
        print(f"  {n:>4} condition strings use keys: {label}")

    pd.DataFrame(summary_rows).to_csv(HERE / "condition_field_summary.csv", index=False)

    value_rows = []
    for k, vals in field_value_subjects.items():
        for v, n_cond in sorted(vals.items(), key=lambda kv: -kv[1]):
            value_rows.append(
                {
                    "field": k,
                    "value": v,
                    "n_condition_strings_with_value": n_cond,
                    "n_response_rows_with_value": field_value_rows[k][v],
                }
            )
    pd.DataFrame(value_rows).to_csv(HERE / "condition_field_values.csv", index=False)
    print(f"\nWrote {HERE / 'condition_field_summary.csv'}")
    print(f"Wrote {HERE / 'condition_field_values.csv'} ({len(value_rows):,} rows)")


if __name__ == "__main__":
    main()
