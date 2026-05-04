"""Tests for stratified-by-data_category variant sampling."""

from __future__ import annotations

from harness.splits import add_item_variant_id
from harness.sampling import stratified_sample_variants
from tests.synthetic import make_synthetic_df


def test_returns_unique_variant_ids():
    df = add_item_variant_id(make_synthetic_df())
    out = stratified_sample_variants(df, n_samples=20, seed=0)
    assert len(out) == len(set(out))


def test_categories_balanced_when_sufficient_supply():
    df = add_item_variant_id(make_synthetic_df())
    out = stratified_sample_variants(df, n_samples=20, seed=0)
    cat_for = dict(zip(df["item_variant_id"], df["data_category"]))
    counts = {}
    for v in out:
        counts[cat_for[v]] = counts.get(cat_for[v], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_caps_at_supply_when_n_too_large():
    df = add_item_variant_id(make_synthetic_df())
    total_variants = df["item_variant_id"].nunique()
    out = stratified_sample_variants(df, n_samples=10_000, seed=0)
    assert len(out) == total_variants


def test_seed_is_reproducible():
    df = add_item_variant_id(make_synthetic_df())
    a = stratified_sample_variants(df, n_samples=15, seed=42)
    b = stratified_sample_variants(df, n_samples=15, seed=42)
    assert a == b


def test_remainder_is_random_but_seeded():
    df = add_item_variant_id(make_synthetic_df())
    a = stratified_sample_variants(df, n_samples=18, seed=1)
    b = stratified_sample_variants(df, n_samples=18, seed=2)
    assert a != b
