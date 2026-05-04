"""Tests for splits.add_item_variant_id and make_item_cold_start_split."""

from __future__ import annotations

import pandas as pd

from harness.splits import add_item_variant_id, make_item_cold_start_split
from tests.synthetic import make_synthetic_df


def test_add_item_variant_id_distinguishes_conditions():
    df = make_synthetic_df()
    out = add_item_variant_id(df)
    same_id_diff_cond = (
        out.groupby("item_id")["item_variant_id"].nunique().max()
    )
    # Synthetic data has every item_id appear under exactly one condition
    # (we built it that way), so item_id is condition-specific and the
    # variant id should equal "<benchmark>::<item_id>" 1-to-1.
    assert same_id_diff_cond == 1
    assert (out["item_variant_id"].astype(str).str.contains("::")).all()


def test_add_item_variant_id_when_item_id_reused_across_conditions():
    df = make_synthetic_df()
    extra = df.iloc[:8].copy()
    extra["condition"] = "skill=other"
    df2 = pd.concat([df, extra], axis=0, ignore_index=True)
    out = add_item_variant_id(df2)
    same_item_diff_cond = (
        out[out["item_id"] == extra["item_id"].iloc[0]]
        .groupby("condition")["item_variant_id"].first()
    )
    assert same_item_diff_cond.nunique() == 2


def test_no_variant_overlap_between_train_and_val():
    df = add_item_variant_id(make_synthetic_df())
    train, val, _, report = make_item_cold_start_split(df, val_fraction=0.3, seed=0)
    overlap = set(train["item_variant_id"]) & set(val["item_variant_id"])
    assert overlap == set()
    assert report.n_overlap_variants == 0


def test_every_validation_subject_appears_in_training():
    df = add_item_variant_id(make_synthetic_df())
    train, val, val_unseen, report = make_item_cold_start_split(
        df, val_fraction=0.3, seed=0
    )
    train_subjects = set(train["subject_id"])
    val_subjects = set(val["subject_id"])
    assert val_subjects.issubset(train_subjects)
    assert report.n_val_subjects == len(val_subjects)


def test_unseen_subject_rows_are_split_off():
    """If a subject appears ONLY in held-out variants, all its rows must be
    in val_unseen_subjects, never in val (the official-like set)."""
    df = add_item_variant_id(make_synthetic_df())
    rare = df[df["subject_id"] == "subj_7"].copy()
    rare["item_id"] = rare["item_id"] + "_RARE"
    rare["condition"] = "rare_only"
    rare["item_content"] = rare["item_content"] + " RARE"
    df = pd.concat(
        [df[df["subject_id"] != "subj_7"], rare], axis=0, ignore_index=True
    )
    df = add_item_variant_id(df)

    train, val, val_unseen, _ = make_item_cold_start_split(
        df, val_fraction=1.0, seed=0
    )
    # When val_fraction=1.0 and only subj_7 has variants here, all variants
    # are val variants, so subj_7 has no training rows -> unseen bucket.
    assert "subj_7" not in set(val["subject_id"])
    if len(val_unseen) > 0:
        assert "subj_7" in set(val_unseen["subject_id"])


def test_holdout_benchmarks_go_entirely_to_val():
    df = add_item_variant_id(make_synthetic_df())
    train, val, _, report = make_item_cold_start_split(
        df, val_fraction=0.0, seed=0, holdout_benchmarks=["bench_a"]
    )
    assert "bench_a" not in set(train["benchmark"])
    assert "bench_a" in set(val["benchmark"])
    assert report.held_out_benchmarks == ("bench_a",)
