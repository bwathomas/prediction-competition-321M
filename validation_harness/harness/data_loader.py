"""Load the public training data into a single joined DataFrame.

The HuggingFace dump that ships in `starting_kit/Data/` is split into:
- response tables (one per benchmark, e.g. mmlupro.parquet)
- registry tables: subjects.parquet, items.parquet, benchmarks.parquet
- trace tables (*_traces.parquet) -- intentionally NOT loaded, different schema

This module joins everything into the four-field shape that model.predict()
expects, plus a few bookkeeping columns the harness needs (subject_id,
item_id, label, item_variant_id later).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import normalize_condition

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def list_response_files(data_dir: Path) -> list[Path]:
    """Return sorted list of response parquets (excluding registry & traces)."""
    out = []
    for p in sorted(data_dir.iterdir()):
        n = p.name
        if not n.endswith(".parquet"):
            continue
        if n in REGISTRY_FILES:
            continue
        if n.endswith("_traces.parquet"):
            continue
        out.append(p)
    return out


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    """Reproduce starting_kit/README.md's render_subject_content.

    Always emits a "Name:" line. Optional fields are appended only when truthy.
    The hosted runtime may add more lines at test time -- treat as plain text.
    """
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


def load_responses(
    data_dir: Path,
    *,
    benchmarks: Iterable[str] | None = None,
    max_rows_per_benchmark: int | None = None,
    drop_nan_labels: bool = True,
) -> pd.DataFrame:
    """Load and join all response tables into one DataFrame.

    Parameters
    ----------
    data_dir : Path to the folder containing the parquet dump.
    benchmarks : optional whitelist of benchmark file basenames (without .parquet)
        e.g. {"mmlupro", "ai2d_test"}. None loads everything.
    max_rows_per_benchmark : if set, sample at most this many rows per benchmark
        (useful for fast local iteration).
    drop_nan_labels : drop rows whose response field is null.

    Returns columns:
        subject_id, item_id, benchmark, condition, subject_content,
        item_content, label, trial, correct_answer
    """
    data_dir = Path(data_dir)
    response_files = list_response_files(data_dir)
    if benchmarks is not None:
        wanted = set(benchmarks)
        response_files = [p for p in response_files if p.stem in wanted]

    subjects = pd.read_parquet(data_dir / "subjects.parquet")
    items = pd.read_parquet(data_dir / "items.parquet")
    benchmarks_df = pd.read_parquet(data_dir / "benchmarks.parquet")

    subject_by_id = {row["subject_id"]: row.to_dict() for _, row in subjects.iterrows()}
    item_content_by_id = dict(zip(items["item_id"].astype(str), items["content"].astype(str)))
    benchmark_canonical = dict(
        zip(benchmarks_df["benchmark_id"].astype(str), benchmarks_df["benchmark_id"].astype(str))
    )

    parts: list[pd.DataFrame] = []
    for path in response_files:
        df = pd.read_parquet(path)
        if max_rows_per_benchmark is not None and len(df) > max_rows_per_benchmark:
            df = df.sample(n=max_rows_per_benchmark, random_state=0).reset_index(drop=True)

        df["benchmark"] = df["benchmark_id"].astype(str).map(
            lambda b: benchmark_canonical.get(b, b)
        )
        df["condition"] = df["test_condition"].map(normalize_condition)
        df["subject_id"] = df["subject_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)

        df["item_content"] = df["item_id"].map(item_content_by_id).fillna("")
        df["subject_content"] = df["subject_id"].map(
            lambda sid: render_subject_content(subject_by_id.get(sid, {}), sid)
        )
        df["label"] = pd.to_numeric(df["response"], errors="coerce")

        keep = [
            "subject_id",
            "item_id",
            "benchmark",
            "condition",
            "subject_content",
            "item_content",
            "label",
            "trial",
            "correct_answer",
        ]
        keep = [c for c in keep if c in df.columns]
        parts.append(df[keep])

    out = pd.concat(parts, axis=0, ignore_index=True)
    if drop_nan_labels:
        out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out


def _stable_bucket(key: str, n_buckets: int, seed: int) -> int:
    """Deterministic 0..n_buckets-1 bucket via sha1(seed:key)."""
    import hashlib
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8", errors="replace")).digest()
    n = int.from_bytes(h[:8], "big", signed=False)
    return n % n_buckets


def add_data_category(
    df: pd.DataFrame,
    *,
    mode: str = "random",
    n_categories: int = 15,
    seed: int = 0,
    variant_col: str = "item_variant_id",
) -> pd.DataFrame:
    """Attach a `data_category` column used ONLY for sampling and label budgets.

    The README says hidden sampling is "stratified across data categories" and
    that adaptive labeling reveals K per data category. The official category
    mapping is NOT published, so this harness defaults to `mode="random"` with
    15 buckets, computed as a stable hash of `item_variant_id`. Same variant
    always lands in the same bucket (so per-variant sampling stays consistent),
    and the assignment is independent of `benchmark` -- the harness purposely
    does NOT bake the benchmark identity into the category, to avoid letting
    that signal leak through the stratification step.

    Modes
    -----
    "random"   : sha1-bucket the `item_variant_id` into n_categories buckets.
                 Requires `variant_col` to be present; call
                 `add_item_variant_id(df)` first.
    "benchmark": legacy mode, one category per benchmark name. Available for
                 debugging/comparison but NOT the default.

    `seed` only affects "random" mode. Same seed => same assignment.
    """
    df = df.copy()
    if mode == "random":
        if variant_col not in df.columns:
            raise KeyError(
                f"add_data_category(mode='random') requires column {variant_col!r}; "
                "call add_item_variant_id(df) first."
            )
        if n_categories < 1:
            raise ValueError("n_categories must be >= 1")
        width = max(2, len(str(n_categories - 1)))
        df["data_category"] = (
            df[variant_col]
            .astype(str)
            .map(lambda v: f"cat_{_stable_bucket(v, n_categories, seed):0{width}d}")
        )
    elif mode == "benchmark":
        df["data_category"] = df["benchmark"].astype(str)
    else:
        raise ValueError(f"Unknown data_category mode: {mode!r}")
    return df
