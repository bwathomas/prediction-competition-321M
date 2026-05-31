"""Unit tests for the CoT / item-type interaction helpers in item_features."""

from __future__ import annotations

import numpy as np

from src.item_features import (
    COT_INTERACTION_BASE,
    ITEM_TYPE_NAMES,
    build_cot_interactions,
    compute_pool_features,
    cot_interaction_names,
    is_cot_from_condition,
    item_type_onehot,
)


def test_is_cot_from_condition_detects_variants():
    assert is_cot_from_condition("chain-of-thought") == 1.0
    assert is_cot_from_condition("CoT") == 1.0
    assert is_cot_from_condition("step by step") == 1.0
    assert is_cot_from_condition("zero-shot") == 0.0
    assert is_cot_from_condition("none") == 0.0
    assert is_cot_from_condition("") == 0.0
    assert is_cot_from_condition(None) == 0.0  # type: ignore[arg-type]


def test_item_type_onehot_is_mutually_exclusive():
    for cond_text in [
        "def foo():\n    return 1",          # code
        "What is 2+2?\nA) 3\nB) 4\nC) 5",    # mcq
        r"Compute $\int_0^1 x\,dx$",          # math (latex)
        "Describe the causes of the war.",   # prose
    ]:
        pool = compute_pool_features(cond_text)
        t = item_type_onehot(pool)
        assert set(t.keys()) == set(ITEM_TYPE_NAMES)
        assert sum(t.values()) == 1.0  # exactly one type set


def test_item_type_priority_code_over_mcq():
    # Has both code and MC markers -> code wins by priority.
    pool = compute_pool_features("```\nx=1\n```\nA) yes\nB) no")
    t = item_type_onehot(pool)
    assert t["type_code"] == 1.0
    assert t["type_mcq"] == 0.0


def test_build_cot_interactions_zeroes_non_cot_rows():
    base_names = ["has_code", "token_len", "type_mcq"]
    base = np.array(
        [[1.0, 2.0, 0.0],
         [1.0, -1.0, 1.0],
         [0.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    is_cot = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    inter, names = build_cot_interactions(
        base, base_names, is_cot, bases=base_names,
    )
    assert names == ["cot_x_has_code", "cot_x_token_len", "cot_x_type_mcq"]
    assert inter.shape == (3, 3)
    # Row 0 is CoT -> equals base; row 1 is non-CoT -> all zero; row 2 CoT.
    np.testing.assert_allclose(inter[0], base[0])
    np.testing.assert_allclose(inter[1], np.zeros(3))
    np.testing.assert_allclose(inter[2], base[2])


def test_build_cot_interactions_skips_missing_bases():
    base_names = ["has_code"]
    base = np.array([[1.0], [0.0]], dtype=np.float32)
    is_cot = np.array([1.0, 1.0], dtype=np.float32)
    inter, names = build_cot_interactions(
        base, base_names, is_cot, bases=("has_code", "nonexistent"),
    )
    assert names == ["cot_x_has_code"]
    assert inter.shape == (2, 1)


def test_cot_interaction_names_default():
    names = cot_interaction_names()
    assert names == [f"cot_x_{b}" for b in COT_INTERACTION_BASE]
    assert len(names) == len(COT_INTERACTION_BASE)
