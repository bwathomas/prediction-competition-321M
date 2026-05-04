"""Build a CSV with one row per unique model name observed anywhere in the data.

Source of names (unioned, deduplicated, case-sensitive):
  - subjects.parquet["display_name"]   (the canonical name used in subject_content)
  - subjects.parquet["raw_labels_seen"] (list of alternate label strings observed
    during ingestion; included to honor "every unique model name anywhere in the
    data")

Output columns:
  name, organization, parameters, release_date, family

The latter four are intentionally left blank for a downstream agent to populate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUT_CSV = Path(__file__).resolve().parent / "model_info.csv"


def main() -> None:
    subjects = pd.read_parquet(DATA_DIR / "subjects.parquet")

    display_names = {
        str(v).strip()
        for v in subjects["display_name"].dropna().tolist()
        if str(v).strip()
    }

    raw_label_names: set[str] = set()
    for entry in subjects["raw_labels_seen"].dropna().tolist():
        try:
            iterator = list(entry)
        except TypeError:
            iterator = [entry]
        for x in iterator:
            s = str(x).strip()
            if s:
                raw_label_names.add(s)

    union_names = sorted(display_names | raw_label_names, key=str.lower)

    df = pd.DataFrame(
        {
            "name": union_names,
            "organization": "",
            "parameters": "",
            "release_date": "",
            "family": "",
        }
    )
    df.to_csv(OUT_CSV, index=False)

    extras = raw_label_names - display_names
    print(f"display_name unique:                    {len(display_names):,}")
    print(f"raw_labels_seen unique:                 {len(raw_label_names):,}")
    print(f"In raw_labels_seen but not display_name: {len(extras):,}")
    print(f"Union (rows written):                   {len(union_names):,}")
    print(f"\nWrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
