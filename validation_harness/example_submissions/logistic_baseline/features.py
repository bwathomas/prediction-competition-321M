"""Shared feature extraction for the logistic baseline.

Used by both the offline trainer (`train_logistic.py`) and the runtime
`model.py` so that the exact same transformation is applied at fit time and
at inference time.

Design choices
--------------
* No cross-term interactions. Each column produced here is fed into the
  pipeline independently; downstream sklearn does one-hot / scale / impute,
  but never builds (benchmark x condition) products.
* Missing values are EXPECTED. Many rows have an unknown organization, an
  unknown release date, or a model name that isn't in `model_info.csv`. We
  represent unknowns as:
    - NaN for numeric columns (parameters, release_date, age) so the
      pipeline's imputer fills them with the training mean and a
      "<col>_is_missing" indicator captures the flag. We pre-compute the
      indicator column here so the missingness signal survives even when the
      imputer collapses the value to the mean.
    - the literal string "unknown" for categorical columns (organization,
      family, macro_family, topic, name_prefix, condition_kind).
* Condition strings come in many shapes ("none", "skill=coarse_perception",
  "metric=utility|attack=important_instructions"). We do NOT one-hot the raw
  string -- there are 215+ unique values, and the spec warns extra values
  may appear at test time. Instead we use a small bag-of-keys representation
  (one binary indicator per known top-level key, e.g. "cond_has_skill") plus
  a "cond_is_none" flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# Keys we accept inside a condition string (everything else collapses to
# "cond_has_other"). Derived from condition_field_summary.csv.
KNOWN_CONDITION_KEYS: tuple[str, ...] = (
    "skill",
    "aspect",
    "subset",
    "source",
    "prompt",
    "metric",
    "attack",
    "judge",
    "mode",
    "criterion",
    "aggregate",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "params",
    "release_date",
    "benchmark_age",
)

# All categorical columns the trainer one-hots. Keep this list in sync with
# `_to_feature_row` below.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "benchmark",
    "topic",
    "organization",
    "family",
    "macro_family",
    "params_bucket",
    "release_year_bucket",
)

# Binary indicator columns (already 0/1) -- treated as numeric, no scaling.
INDICATOR_FEATURES: tuple[str, ...] = (
    "params_is_missing",
    "release_is_missing",
    "model_is_known",
    "benchmark_age_is_missing",
    "benchmark_has_conditions",
    "cond_is_none",
    *(f"cond_has_{k}" for k in KNOWN_CONDITION_KEYS),
    "cond_has_other",
)


def extract_name_from_subject_content(subject_content: str) -> str:
    """Pull the model display name out of the four-field `subject_content`.

    The runtime contract says the string starts with a "Name:" line; extra
    metadata lines (Organization, Parameters, ...) may follow.
    """
    if not isinstance(subject_content, str):
        return ""
    for line in subject_content.splitlines():
        s = line.strip()
        if s.lower().startswith("name:"):
            return s.split(":", 1)[1].strip()
    return ""


def _safe_float(value: object) -> float:
    """Parse a value into float, returning NaN for blanks / 'unknown' / errors."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() in {"unknown", "nan", "none", "null", ""}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _params_bucket(params: float) -> str:
    """Coarse bucket for model size in billions of parameters."""
    if not np.isfinite(params):
        return "unknown"
    if params < 3:
        return "tiny_<3B"
    if params < 8:
        return "small_3-8B"
    if params < 20:
        return "medium_8-20B"
    if params < 70:
        return "large_20-70B"
    if params < 200:
        return "xlarge_70-200B"
    return "huge_200B+"


def _release_year_bucket(release_date: float) -> str:
    """Coarse bucket for the release_date column.

    `release_date` in `model_info.csv` is encoded as an integer "days ago"
    relative to some reference date (older models have larger values). We
    don't need exact calendar dates -- buckets group "newer" vs "older".
    """
    if not np.isfinite(release_date):
        return "unknown"
    if release_date < 200:
        return "very_recent"
    if release_date < 400:
        return "recent"
    if release_date < 700:
        return "mid"
    if release_date < 1000:
        return "older"
    return "ancient"


def parse_condition(raw: object) -> dict[str, int]:
    """Bag-of-keys representation of the `condition` field.

    Returns a dict whose values are 0/1 indicators:
        cond_is_none             1 iff condition normalizes to "none"
        cond_has_<key>           1 iff <key>=... appears in the condition
        cond_has_other           1 iff some non-recognized key=... appears
    """
    out = {"cond_is_none": 0, "cond_has_other": 0}
    for k in KNOWN_CONDITION_KEYS:
        out[f"cond_has_{k}"] = 0

    s = str(raw).strip() if raw is not None else ""
    if not s or s.lower() in {"none", "nan", "null", ""}:
        out["cond_is_none"] = 1
        return out

    for piece in s.split("|"):
        if "=" not in piece:
            out["cond_has_other"] = 1
            continue
        key, _ = piece.split("=", 1)
        key = key.strip()
        if key in KNOWN_CONDITION_KEYS:
            out[f"cond_has_{key}"] = 1
        else:
            out["cond_has_other"] = 1
    return out


class FeatureBuilder:
    """Stateless-at-inference feature builder backed by lookup tables.

    The lookup tables (model_info.csv, benchmark_info.csv) are tiny; we keep
    them as in-memory dicts so per-row inference is just a hash lookup.
    """

    def __init__(
        self,
        model_info: Mapping[str, Mapping[str, object]],
        benchmark_info: Mapping[str, Mapping[str, object]],
    ) -> None:
        self.model_info = dict(model_info)
        self.benchmark_info = dict(benchmark_info)

    @classmethod
    def from_csvs(cls, model_info_csv: Path, benchmark_info_csv: Path) -> "FeatureBuilder":
        m = pd.read_csv(model_info_csv).fillna("unknown")
        b = pd.read_csv(benchmark_info_csv).fillna("unknown")
        m_lookup = {
            str(row["name"]): {
                "organization": str(row.get("organization", "unknown")),
                "params": _safe_float(row.get("parameters")),
                "release_date": _safe_float(row.get("release_date")),
                "family": str(row.get("family", "unknown")),
                "macro_family": str(row.get("macro-family", "unknown")),
            }
            for _, row in m.iterrows()
        }
        b_lookup = {
            str(row["benchmark"]): {
                "topic": str(row.get("topic", "unknown")),
                "age": _safe_float(row.get("age")),
                "has_conditions": int(_safe_float(row.get("has_conditions")) > 0)
                if pd.notna(_safe_float(row.get("has_conditions")))
                else 0,
            }
            for _, row in b.iterrows()
        }
        return cls(m_lookup, b_lookup)

    def to_feature_row(
        self, *, benchmark: str, condition: str, subject_content: str, item_content: str
    ) -> dict[str, object]:
        """Convert the four runtime input fields into a flat feature dict."""
        del item_content  # not used by this simple model

        name = extract_name_from_subject_content(subject_content)
        m = self.model_info.get(name, {})
        model_known = name in self.model_info

        params = _safe_float(m.get("params"))
        release = _safe_float(m.get("release_date"))

        b = self.benchmark_info.get(benchmark, {})
        bench_age = _safe_float(b.get("age"))

        cond_feats = parse_condition(condition)

        row: dict[str, object] = {
            # categorical
            "benchmark": str(benchmark) if benchmark else "unknown",
            "topic": str(b.get("topic", "unknown")) if b else "unknown",
            "organization": str(m.get("organization", "unknown")) if m else "unknown",
            "family": str(m.get("family", "unknown")) if m else "unknown",
            "macro_family": str(m.get("macro_family", "unknown")) if m else "unknown",
            "params_bucket": _params_bucket(params),
            "release_year_bucket": _release_year_bucket(release),
            # numeric (NaN-tolerant; downstream imputer fills)
            "params": params,
            "release_date": release,
            "benchmark_age": bench_age,
            # indicators (always 0/1)
            "params_is_missing": int(not np.isfinite(params)),
            "release_is_missing": int(not np.isfinite(release)),
            "model_is_known": int(model_known),
            "benchmark_age_is_missing": int(not np.isfinite(bench_age)),
            "benchmark_has_conditions": int(b.get("has_conditions", 0)) if b else 0,
        }
        row.update(cond_feats)
        return row

    def transform_dataframe(
        self, df: pd.DataFrame, *, show_progress: bool = False
    ) -> pd.DataFrame:
        """Vectorized feature build (pandas merge + str ops, no Python loops).

        ~50-200x faster than `to_dict(orient="records")` + per-row dict
        construction for the 100k-5M row tables we deal with.

        `show_progress` is accepted for API symmetry but the work is one
        vectorized pass; we just print stage timestamps.
        """
        import time

        t0 = time.perf_counter()

        bench = df["benchmark"].astype(str)
        cond_raw = df["condition"].astype(str) if "condition" in df.columns else pd.Series(
            ["none"] * len(df), index=df.index
        )
        subj = df["subject_content"].astype(str)

        names = subj.str.extract(r"(?im)^\s*Name:\s*(.*?)\s*$", expand=False).fillna("")
        known_name_set = set(self.model_info.keys())
        model_known_arr = names.isin(known_name_set).astype(int).values

        m_df = pd.DataFrame.from_dict(self.model_info, orient="index").rename_axis("name")
        for col in ("organization", "params", "release_date", "family", "macro_family"):
            if col not in m_df.columns:
                m_df[col] = np.nan
        m_df = m_df.reset_index()

        b_df = pd.DataFrame.from_dict(self.benchmark_info, orient="index").rename_axis("benchmark")
        for col in ("topic", "age", "has_conditions"):
            if col not in b_df.columns:
                b_df[col] = np.nan
        b_df = b_df.reset_index()

        joined = pd.DataFrame(
            {
                "name": names.values,
                "benchmark_in": bench.values,
                "condition_in": cond_raw.values,
            }
        )
        joined = joined.merge(m_df, on="name", how="left")
        joined = joined.merge(
            b_df.rename(columns={"benchmark": "benchmark_in"}),
            on="benchmark_in",
            how="left",
        )

        params = pd.to_numeric(joined["params"], errors="coerce").astype(float)
        release = pd.to_numeric(joined["release_date"], errors="coerce").astype(float)
        bench_age = pd.to_numeric(joined["age"], errors="coerce").astype(float)

        if show_progress:
            print(f"  [features] joined lookup tables in {time.perf_counter()-t0:.2f}s")

        cond_clean = joined["condition_in"].fillna("none").astype(str)
        unique_conditions = pd.unique(cond_clean.values)
        cond_lookup_rows = [parse_condition(c) for c in unique_conditions]
        cond_lookup = pd.DataFrame(cond_lookup_rows, index=unique_conditions)
        cond_indicator_cols = list(cond_lookup.columns)
        cond_lookup = cond_lookup.reindex(cond_clean.values).reset_index(drop=True)
        cond_indicators = {col: cond_lookup[col].astype(int).values for col in cond_indicator_cols}

        if show_progress:
            print(f"  [features] parsed {len(joined):,} conditions in {time.perf_counter()-t0:.2f}s")

        params_bucket = np.where(
            ~np.isfinite(params), "unknown",
            np.where(params < 3, "tiny_<3B",
            np.where(params < 8, "small_3-8B",
            np.where(params < 20, "medium_8-20B",
            np.where(params < 70, "large_20-70B",
            np.where(params < 200, "xlarge_70-200B", "huge_200B+")))))
        )
        release_bucket = np.where(
            ~np.isfinite(release), "unknown",
            np.where(release < 200, "very_recent",
            np.where(release < 400, "recent",
            np.where(release < 700, "mid",
            np.where(release < 1000, "older", "ancient"))))
        )

        out = pd.DataFrame(index=joined.index)
        out["benchmark"] = joined["benchmark_in"].fillna("unknown").astype(str)
        out["topic"] = joined["topic"].fillna("unknown").astype(str)
        out["organization"] = joined["organization"].fillna("unknown").astype(str)
        out["family"] = joined["family"].fillna("unknown").astype(str)
        out["macro_family"] = joined["macro_family"].fillna("unknown").astype(str)
        out["params_bucket"] = params_bucket
        out["release_year_bucket"] = release_bucket

        out["params"] = params.values
        out["release_date"] = release.values
        out["benchmark_age"] = bench_age.values

        has_cond_raw = pd.to_numeric(joined["has_conditions"], errors="coerce").fillna(0).astype(int)

        out["params_is_missing"] = (~np.isfinite(params)).astype(int).values
        out["release_is_missing"] = (~np.isfinite(release)).astype(int).values
        out["model_is_known"] = model_known_arr
        out["benchmark_age_is_missing"] = (~np.isfinite(bench_age)).astype(int).values
        out["benchmark_has_conditions"] = (has_cond_raw > 0).astype(int).values

        for k, arr in cond_indicators.items():
            out[k] = arr

        if show_progress:
            print(f"  [features] assembled feature frame in {time.perf_counter()-t0:.2f}s "
                  f"(rows={len(out):,}, cols={out.shape[1]})")
        return out


def all_feature_columns() -> list[str]:
    """Return the canonical column order produced by `to_feature_row`."""
    return [
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
        *INDICATOR_FEATURES,
    ]


def feature_groups() -> dict[str, tuple[str, ...]]:
    """Map group label -> tuple of source column names. Used for reporting
    aggregate feature-group importance after training."""
    return {
        "benchmark":         ("benchmark",),
        "topic":             ("topic",),
        "benchmark_age":     ("benchmark_age", "benchmark_age_is_missing"),
        "benchmark_has_cond":("benchmark_has_conditions",),
        "condition":         ("cond_is_none", "cond_has_other",
                              *(f"cond_has_{k}" for k in KNOWN_CONDITION_KEYS)),
        "organization":      ("organization",),
        "family":            ("family",),
        "macro_family":      ("macro_family",),
        "params":            ("params", "params_bucket", "params_is_missing"),
        "release":           ("release_date", "release_year_bucket", "release_is_missing"),
        "model_is_known":    ("model_is_known",),
    }


def coerce_numeric_columns(
    feature_df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES
) -> pd.DataFrame:
    """Force the numeric columns to float (NaN-safe)."""
    out = feature_df.copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
