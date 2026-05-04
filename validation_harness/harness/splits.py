"""Item cold-start split.

The official competition is item cold-start (NOT subject cold-start), so:
  - validation subjects MUST also appear in training
  - validation item-variants MUST NOT appear in training
  - same upstream item under different normalized conditions counts as
    different item-variants

We construct an `item_variant_id`. Per benchmark we detect whether the
official item_id is already condition-specific (each item_id appears under at
most one normalized condition); if so we use the official id. Otherwise we
combine it with a stable hash of (normalized condition, item_content) so a
condition flip produces a fresh variant.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import normalize_condition


def _stable_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def add_item_variant_id(
    df: pd.DataFrame,
    *,
    benchmark_col: str = "benchmark",
    item_id_col: str = "item_id",
    condition_col: str = "condition",
    item_content_col: str = "item_content",
    out_col: str = "item_variant_id",
) -> pd.DataFrame:
    """Attach `item_variant_id` to df. Per-benchmark policy:

    - if every item_id in this benchmark appears under exactly one normalized
      condition, the variant id is "<benchmark>::<item_id>"
    - otherwise the variant id is "<benchmark>::<sha1(condition|item_content)>"

    Returns a copy of df with the new column.
    """
    df = df.copy()
    df[condition_col] = df[condition_col].map(normalize_condition)

    cond_per_item = (
        df.groupby([benchmark_col, item_id_col])[condition_col].nunique().reset_index()
    )
    max_cond_by_bench = cond_per_item.groupby(benchmark_col)[condition_col].max()
    condition_specific_benchmarks = set(max_cond_by_bench[max_cond_by_bench <= 1].index)

    is_cs = df[benchmark_col].isin(condition_specific_benchmarks)

    cs_ids = (
        df[benchmark_col].astype(str) + "::" + df[item_id_col].astype(str)
    )
    nc_ids = (
        df[benchmark_col].astype(str)
        + "::"
        + (
            df[condition_col].astype(str)
            + "|"
            + df[item_content_col].astype(str)
        ).map(lambda s: _stable_hash(s))
    )

    df[out_col] = np.where(is_cs, cs_ids, nc_ids)
    return df


@dataclass
class SplitReport:
    """Bookkeeping returned alongside the split DataFrames."""

    n_train_rows: int
    n_val_rows: int
    n_val_unseen_subject_rows: int
    n_train_variants: int
    n_val_variants: int
    n_train_subjects: int
    n_val_subjects: int
    n_overlap_variants: int  # MUST be 0
    held_out_benchmarks: tuple[str, ...]


def make_item_cold_start_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    holdout_benchmarks: Iterable[str] | None = None,
    variant_col: str = "item_variant_id",
    subject_col: str = "subject_id",
    benchmark_col: str = "benchmark",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split by item-variant, item cold-start.

    Parameters
    ----------
    df : DataFrame produced by data_loader.load_responses (must contain
        item_variant_id; if missing, call add_item_variant_id first).
    val_fraction : fraction of item-variants to put in validation.
    seed : RNG seed.
    holdout_benchmarks : optional set of benchmark names to fully hold out
        ("stricter held-out benchmark" stress mode). Variants from these
        benchmarks go entirely to validation; the remaining benchmarks are
        split by `val_fraction` as usual. Subjects in held-out benchmarks
        still must appear in training somewhere (any benchmark).

    Returns
    -------
    train_df, val_df, val_unseen_subject_df, report

    `val_df` is the official-like validation set: every row's subject is
    guaranteed to be present in training. `val_unseen_subject_df` is the
    non-official stress-test bucket of validation rows whose subject never
    appears in training; report this separately, do NOT mix it into the
    main score.
    """
    if variant_col not in df.columns:
        raise KeyError(
            f"{variant_col!r} not in df. Call add_item_variant_id() first."
        )

    rng = np.random.default_rng(seed)
    holdout_benchmarks = tuple(holdout_benchmarks or ())

    all_variants = df[[variant_col, benchmark_col]].drop_duplicates()
    if holdout_benchmarks:
        held_mask = all_variants[benchmark_col].isin(holdout_benchmarks)
        held_variants = set(all_variants.loc[held_mask, variant_col])
        normal_pool = all_variants.loc[~held_mask, variant_col].to_numpy()
    else:
        held_variants = set()
        normal_pool = all_variants[variant_col].to_numpy()

    n_val_from_normal = int(round(val_fraction * len(normal_pool)))
    n_val_from_normal = max(0, min(len(normal_pool), n_val_from_normal))
    perm = rng.permutation(len(normal_pool))
    val_normal = set(normal_pool[perm[:n_val_from_normal]].tolist())
    val_variants = held_variants | val_normal
    train_variants = set(all_variants[variant_col]) - val_variants

    train_df = df[df[variant_col].isin(train_variants)].copy()
    raw_val_df = df[df[variant_col].isin(val_variants)].copy()

    train_subjects = set(train_df[subject_col].astype(str))
    val_subject_seen = raw_val_df[subject_col].astype(str).isin(train_subjects)
    val_df = raw_val_df[val_subject_seen].copy()
    val_unseen_subject_df = raw_val_df[~val_subject_seen].copy()

    overlap = train_variants & val_variants  # must be empty by construction
    report = SplitReport(
        n_train_rows=len(train_df),
        n_val_rows=len(val_df),
        n_val_unseen_subject_rows=len(val_unseen_subject_df),
        n_train_variants=len(train_variants),
        n_val_variants=len(val_variants),
        n_train_subjects=len(train_subjects),
        n_val_subjects=val_df[subject_col].nunique(),
        n_overlap_variants=len(overlap),
        held_out_benchmarks=holdout_benchmarks,
    )
    return train_df, val_df, val_unseen_subject_df, report
