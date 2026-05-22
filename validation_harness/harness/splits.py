"""Item cold-start split.

The official competition is item cold-start (NOT subject cold-start), so:
  - validation subjects MUST also appear in training
  - validation items MUST NOT appear in training
  - "same item" is judged by upstream content, NOT by the (condition,
    content) pair: if the same item_content shows up under two different
    conditions, the encoder produces near-identical embeddings for both,
    and a model that has seen one of them at training time has effectively
    seen the other -- so they MUST land on the same side of the split.

We construct two columns:

  * ``item_variant_id`` -- "same item under different conditions = different
    variants". Used by the platform for ``data_category`` bucketing and by
    the rounds harness for adaptive-labeling. Built per-benchmark: if every
    item_id in the benchmark appears under at most one normalized condition,
    the variant id is "<benchmark>::<item_id>"; otherwise it is
    "<benchmark>::<sha1(condition|item_content)>".

  * ``item_split_key`` -- "same item content = same key, regardless of
    condition". Always "<benchmark>::<sha1(item_content)>". This is the
    key the splitter uses for the train/val partition so that no
    item_content ever appears on both sides. Without this, multi-condition
    benchmarks (ultrafeedback, agentdojo, livecodebench, afrimedqa, ...)
    leaked ~88% of val items back into train, masking overfitting from
    val_NLL-based epoch selection.
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


def add_item_split_key(
    df: pd.DataFrame,
    *,
    benchmark_col: str = "benchmark",
    item_content_col: str = "item_content",
    out_col: str = "item_split_key",
) -> pd.DataFrame:
    """Attach ``item_split_key`` = "<benchmark>::<sha1(item_content)>".

    Unlike ``item_variant_id`` this collapses conditions: two rows with the
    same ``item_content`` always get the same key, regardless of
    ``condition``. Use this column as the train/val split key so the model
    cannot have "seen" a val item at training time under a different
    condition.

    Returns a copy of df with the new column.
    """
    df = df.copy()
    df[out_col] = (
        df[benchmark_col].astype(str)
        + "::"
        + df[item_content_col].astype(str).map(_stable_hash)
    )
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
    val_fraction: float = 0.05,
    seed: int = 0,
    holdout_benchmarks: Iterable[str] | None = None,
    variant_col: str = "item_split_key",
    subject_col: str = "subject_id",
    benchmark_col: str = "benchmark",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split by item, item cold-start.

    Parameters
    ----------
    df : DataFrame produced by data_loader.load_responses. Must contain the
        column named by ``variant_col``. The default is ``item_split_key``
        (content-only; call ``add_item_split_key`` first); pass
        ``variant_col="item_variant_id"`` to opt back into the legacy
        condition-aware split, but be aware that this leaks multi-condition
        benchmarks across train/val.
    val_fraction : fraction of split keys to put in validation.
    seed : RNG seed.
    holdout_benchmarks : optional set of benchmark names to fully hold out
        ("stricter held-out benchmark" stress mode). Items from these
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
        if variant_col == "item_split_key" and "item_content" in df.columns:
            df = add_item_split_key(df)
        elif variant_col == "item_variant_id" and {
            "benchmark",
            "item_id",
            "condition",
            "item_content",
        }.issubset(df.columns):
            df = add_item_variant_id(df)
        else:
            raise KeyError(
                f"{variant_col!r} not in df. "
                "Call add_item_split_key() (recommended) or "
                "add_item_variant_id() (legacy) first."
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
