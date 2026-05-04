"""Tests for add_data_category (random buckets vs benchmark)."""

from __future__ import annotations

import pandas as pd
import pytest

from harness.data_loader import add_data_category
from harness.splits import add_item_variant_id
from tests.synthetic import make_synthetic_df


def _df():
    return add_item_variant_id(make_synthetic_df().drop(columns=["data_category"]))


def test_random_mode_is_default():
    df = _df()
    out = add_data_category(df)
    assert out["data_category"].str.startswith("cat_").all()


def test_random_mode_produces_n_categories():
    df = _df()
    out = add_data_category(df, mode="random", n_categories=15, seed=0)
    cats = set(out["data_category"])
    assert len(cats) <= 15
    assert all(c.startswith("cat_") for c in cats)


def test_random_mode_assigns_per_variant_consistently():
    """Every row of the same item_variant_id MUST share the same category."""
    df = _df()
    out = add_data_category(df, mode="random", n_categories=15, seed=0)
    per_variant = out.groupby("item_variant_id")["data_category"].nunique()
    assert per_variant.max() == 1


def test_random_mode_is_deterministic_across_calls():
    df = _df()
    a = add_data_category(df, mode="random", n_categories=15, seed=0)
    b = add_data_category(df, mode="random", n_categories=15, seed=0)
    assert (a["data_category"].values == b["data_category"].values).all()


def test_random_mode_seed_changes_assignment():
    df = _df()
    a = add_data_category(df, mode="random", n_categories=15, seed=0)
    b = add_data_category(df, mode="random", n_categories=15, seed=1)
    assert (a["data_category"].values != b["data_category"].values).any()


def test_random_mode_independent_of_benchmark():
    """A given category should generally span multiple benchmarks (since the
    bucket is hashed from item_variant_id, not from benchmark)."""
    df = _df()
    out = add_data_category(df, mode="random", n_categories=15, seed=0)
    benches_per_cat = out.groupby("data_category")["benchmark"].nunique()
    assert benches_per_cat.max() >= 2


def test_random_mode_requires_item_variant_id():
    df = make_synthetic_df().drop(columns=["data_category"])
    with pytest.raises(KeyError):
        add_data_category(df, mode="random")


def test_benchmark_mode_legacy_still_works():
    df = _df()
    out = add_data_category(df, mode="benchmark")
    assert set(out["data_category"]) == set(out["benchmark"])


def test_unknown_mode_raises():
    df = _df()
    with pytest.raises(ValueError):
        add_data_category(df, mode="not_a_mode")
