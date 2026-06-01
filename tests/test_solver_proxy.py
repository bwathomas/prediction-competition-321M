"""Pure-helper tests for src.solver_proxy (no GPU / no model load)."""

from __future__ import annotations

import math

from src.solver_proxy import (
    PROXY_FEATURE_NAMES,
    SolverProxyConfig,
    build_proxy_row_vector,
    config_hash,
    cross_model_disagreement,
    extended_sample_features,
    modal_answer,
    normalize_answer,
    vote_statistics,
)
from src.solver_proxy import (  # private but stable; tested directly
    _answer_type,
    _count_steps,
    _EXTENDED_FEATURE_NAMES,
)


def test_normalize_boxed_wins():
    assert normalize_answer("blah \\boxed{42} done") == "42"
    assert normalize_answer("...\\boxed{4.0}") == "4"


def test_normalize_answer_tail():
    assert normalize_answer("reasoning...\nAnswer: 17") == "17"
    assert normalize_answer("The answer is C.") == "c"


def test_normalize_mc_last_line():
    assert normalize_answer("long reasoning\n\nB") == "b"


def test_normalize_refusal():
    assert normalize_answer("") == "<none>"
    assert normalize_answer("   ") == "<none>"


def test_vote_statistics_unanimous():
    vs = vote_statistics(["42", "42", "42", "42", "42"])
    assert vs["self_consistency"] == 1.0
    assert vs["answer_entropy"] == 0.0
    assert vs["fsd"] == 1.0
    assert vs["refusal_rate"] == 0.0


def test_vote_statistics_split():
    vs = vote_statistics(["a", "a", "a", "b", "b"])
    assert abs(vs["self_consistency"] - 0.6) < 1e-9
    assert abs(vs["fsd"] - 0.2) < 1e-9
    assert vs["answer_entropy"] > 0.0


def test_vote_statistics_all_distinct_high_entropy():
    vs = vote_statistics(["a", "b", "c", "d", "e"])
    assert vs["self_consistency"] == 0.2
    assert abs(vs["answer_entropy"] - 1.0) < 1e-9  # max entropy, normalized


def test_vote_statistics_refusals_separated():
    vs = vote_statistics(["7", "7", "<none>", "<none>"])
    assert abs(vs["refusal_rate"] - 0.5) < 1e-9
    # consistency computed on the valid answers only -> unanimous
    assert vs["self_consistency"] == 1.0


def test_modal_answer_ignores_refusals():
    assert modal_answer(["<none>", "9", "9", "<none>"]) == "9"
    assert modal_answer(["<none>", "<none>"]) == "<none>"


def test_cross_model_disagreement():
    modal = {
        "m1": {"i1": "a", "i2": "a", "i3": "x"},
        "m2": {"i1": "a", "i2": "b", "i3": "y"},
        "m3": {"i1": "a", "i2": "b", "i3": "z"},
    }
    dis = cross_model_disagreement(modal)
    assert dis["i1"] == 0.0                       # all agree
    assert abs(dis["i2"] - (1.0 - 2.0 / 3.0)) < 1e-9
    assert abs(dis["i3"] - (1.0 - 1.0 / 3.0)) < 1e-9  # all disagree


def test_cross_model_disagreement_needs_two():
    modal = {"m1": {"i1": "a"}, "m2": {"i1": "<none>"}}
    dis = cross_model_disagreement(modal)
    assert math.isnan(dis["i1"])  # only one non-refusal vote


def test_build_proxy_row_vector():
    v = build_proxy_row_vector(["a", "b", "a", "c"], {"a": 1.0, "b": 2.0}, fill=-1.0)
    assert list(v) == [1.0, 2.0, 1.0, -1.0]


def test_config_hash_changes_with_params():
    a = SolverProxyConfig()
    b = SolverProxyConfig(n_samples=a.n_samples + 1)
    assert config_hash(a) != config_hash(b)
    assert config_hash(a) == config_hash(SolverProxyConfig())


# ---------------------------------------------------------------------------
# Extended length/step/format features (cheap, no GPU)
# ---------------------------------------------------------------------------


def test_count_steps_numbered_wins():
    s = "1. think about it\n2) try the formula\n3. answer\nAnswer: 42"
    assert _count_steps(s) == 3


def test_count_steps_paragraphs_fallback():
    s = "first paragraph here\n\nsecond paragraph here\n\nthird"
    assert _count_steps(s) == 3


def test_count_steps_empty_zero():
    assert _count_steps("") == 0
    assert _count_steps("   ") == 0


def test_answer_type_buckets():
    assert _answer_type("42") == "number"
    assert _answer_type("3.14") == "number"
    assert _answer_type("b") == "mc"
    assert _answer_type("hello world") == "text"
    assert _answer_type("<none>") == "none"
    assert _answer_type("") == "none"


def test_extended_features_returns_full_schema():
    out = extended_sample_features([], [])
    assert set(out.keys()) == set(_EXTENDED_FEATURE_NAMES)
    assert all(v == 0.0 for v in out.values())
    # And every extended key shows up in the public PROXY_FEATURE_NAMES tuple.
    for k in _EXTENDED_FEATURE_NAMES:
        assert k in PROXY_FEATURE_NAMES


def test_extended_features_basic_stats():
    samples = [
        "Step 1: try x\nStep 2: try y\nAnswer: 42",
        "Step 1: alternative\nAnswer: 42",
        "thinking...\n\n\\boxed{42}",
    ]
    parsed = ["42", "42", "42"]
    out = extended_sample_features(samples, parsed)
    assert out["answer_chars_mean"] == 2.0
    assert out["answer_chars_std"] == 0.0
    assert out["boxed_rate"] == 1.0 / 3.0
    assert out["trace_steps_mean"] > 0.0
    assert out["answer_format_consistency"] == 1.0   # all numbers


def test_extended_features_mixed_types_inconsistency():
    samples = ["a long answer here.", "B", "42"]
    parsed = ["a long answer here", "b", "42"]
    out = extended_sample_features(samples, parsed)
    # Three different answer TYPES (text, mc, number) -> low format consistency.
    assert out["answer_format_consistency"] < 0.5
    # Variability across samples -> non-zero std.
    assert out["answer_chars_std"] > 0.0


def test_extended_features_handles_refusals():
    samples = ["", "  ", "Answer: 7"]
    parsed = ["<none>", "<none>", "7"]
    out = extended_sample_features(samples, parsed)
    # Two refusals counted as zero-length answers; the one valid parsed
    # answer is "7" (1 char). Mean over [0, 0, 1] = 1/3.
    assert abs(out["answer_chars_mean"] - 1.0 / 3.0) < 1e-12
    # Only one valid answer -> format consistency is 1.0 by definition.
    assert out["answer_format_consistency"] == 1.0


def test_extended_features_length_mismatch_raises():
    try:
        extended_sample_features(["a", "b"], ["a"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_proxy_feature_names_includes_all_core_and_extended():
    # The combined tuple must be a superset of both blocks.
    for k in ("self_consistency", "p_true", "mean_trace_len"):
        assert k in PROXY_FEATURE_NAMES
    for k in _EXTENDED_FEATURE_NAMES:
        assert k in PROXY_FEATURE_NAMES
