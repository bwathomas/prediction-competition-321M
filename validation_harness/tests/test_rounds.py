"""Tests for run_official_like_round.

Verifies all 8 invariants from the spec:
  3. predict receives exactly the four allowed string fields
  4. acquisition_function receives exactly the four allowed string fields
  5. labeled dicts contain exactly the four fields plus label
  6. data_category is never passed into predict or acquisition_function
  7. adaptive label budget is at most K times the number of categories present
  8. invalid acquisition output triggers random fallback for the whole round
(invariants 1-2 about train/val variant overlap and seen subjects are in
test_splits.py)
"""

from __future__ import annotations

import math

from harness.rounds import run_official_like_round
from harness.splits import add_item_variant_id, make_item_cold_start_split
from harness.utils import INPUT_FIELDS
from tests.synthetic import (
    make_spy_labeling,
    make_spy_model,
    make_synthetic_df,
)

ALLOWED_INPUT_KEYS = tuple(sorted(INPUT_FIELDS))
ALLOWED_LABELED_KEYS = tuple(sorted(INPUT_FIELDS + ("label",)))


def _split():
    df = add_item_variant_id(make_synthetic_df())
    train, val, _, _ = make_item_cold_start_split(df, val_fraction=0.5, seed=0)
    return train, val


def test_predict_receives_exactly_four_string_fields():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert model._spy_state["calls"], "predict was never called"
    for c in model._spy_state["calls"]:
        assert c["input_keys"] == ALLOWED_INPUT_KEYS
        assert all(t == "str" for t in c["input_types"].values())


def test_acquisition_receives_exactly_four_string_fields():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert labeling._spy_state["calls"], "acquisition_function was never called"
    for c in labeling._spy_state["calls"]:
        assert c["input_keys"] == ALLOWED_INPUT_KEYS
        assert all(t == "str" for t in c["input_types"].values())


def test_labeled_dicts_contain_exactly_four_fields_plus_label():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert result.labeled, "labeled list should be non-empty"
    for d in result.labeled:
        assert tuple(sorted(d.keys())) == ALLOWED_LABELED_KEYS
        assert isinstance(d["label"], float)


def test_data_category_never_passed_to_predict_or_acquisition():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    forbidden = {"data_category", "subject_id", "item_id", "item_variant_id", "label"}
    for c in model._spy_state["calls"]:
        assert forbidden.isdisjoint(c["input"].keys())
    for c in labeling._spy_state["calls"]:
        assert forbidden.isdisjoint(c["input"].keys())


def test_label_budget_at_most_K_times_categories_present():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    K = 3
    result = run_official_like_round(train, val, model, labeling, N=40, K=K, seed=0)
    assert result.n_labeled <= K * result.n_categories
    cats_in_labeled = {d["benchmark"] for d in result.labeled}
    assert cats_in_labeled.issubset(set(result.candidates["benchmark"]))


def test_label_budget_caps_at_per_category_supply():
    """If a category has only 2 candidates and K=5, only 2 labels revealed."""
    import pandas as pd
    train, val = _split()
    val_small = pd.concat(
        [
            val[val["benchmark"] == "bench_a"].head(2),
            val[val["benchmark"] != "bench_a"],
        ],
        axis=0,
        ignore_index=True,
    )
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(
        train, val_small, model, labeling, N=10_000, K=5, seed=0
    )
    a_labeled = sum(1 for d in result.labeled if d["benchmark"] == "bench_a")
    assert a_labeled <= 2


def test_acquisition_exception_triggers_random_fallback():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling(raise_on=lambda i: i == 3)
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert result.used_random_acquisition is True
    assert result.fallback_reason and "raised" in result.fallback_reason
    assert result.n_labeled > 0


def test_acquisition_nonfinite_triggers_random_fallback():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling(return_nonfinite_on=lambda i: i == 5)
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert result.used_random_acquisition is True
    assert result.fallback_reason and "non-finite" in result.fallback_reason


def test_missing_labeling_module_uses_random_K_per_category():
    train, val = _split()
    model = make_spy_model()
    result = run_official_like_round(train, val, model, None, N=40, K=2, seed=0)
    assert result.used_random_acquisition is True
    assert result.fallback_reason and "no labeling_module" in result.fallback_reason
    assert 0 < result.n_labeled <= 2 * result.n_categories


def test_same_labeled_list_passed_to_every_predict_call():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    expected_len = result.n_labeled
    assert all(c["labeled_len"] == expected_len for c in model._spy_state["calls"])
    if expected_len > 0:
        expected_keys = tuple(sorted(INPUT_FIELDS + ("label",)))
        assert all(c["labeled_keys"] == expected_keys for c in model._spy_state["calls"])


def test_predict_called_for_every_candidate():
    train, val = _split()
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    assert len(model._spy_state["calls"]) == result.n_candidates


def test_round_is_reproducible_under_same_seed():
    train, val = _split()
    m1, l1 = make_spy_model(), make_spy_labeling()
    m2, l2 = make_spy_model(), make_spy_labeling()
    r1 = run_official_like_round(train, val, m1, l1, N=20, K=2, seed=42)
    r2 = run_official_like_round(train, val, m2, l2, N=20, K=2, seed=42)
    assert r1.n_candidates == r2.n_candidates
    assert r1.n_labeled == r2.n_labeled
    assert (
        sorted(r1.candidates["item_variant_id"].tolist())
        == sorted(r2.candidates["item_variant_id"].tolist())
    )
