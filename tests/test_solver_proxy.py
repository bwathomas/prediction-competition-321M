"""Pure-helper tests for src.solver_proxy (no GPU / no model load)."""

from __future__ import annotations

import math

from src.solver_proxy import (
    SolverProxyConfig,
    build_proxy_row_vector,
    config_hash,
    cross_model_disagreement,
    modal_answer,
    normalize_answer,
    vote_statistics,
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
