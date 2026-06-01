"""Pure-helper tests for src.solver_proxy (no GPU / no model load)."""

from __future__ import annotations

import math

from pathlib import Path

import pandas as pd

from src.solver_proxy import (
    PROXY_CACHE_COLUMNS,
    PROXY_FEATURE_NAMES,
    SolverProxyConfig,
    backfill_derived_features,
    build_proxy_row_vector,
    config_hash,
    cross_model_disagreement,
    extended_sample_features,
    extract_features_from_samples,
    modal_answer,
    normalize_answer,
    vote_statistics,
)
from src.solver_proxy import (  # private but stable; tested directly
    _answer_type,
    _backfill_dataframe,
    _cache_path,
    _count_steps,
    _deserialize_raw_samples,
    _EXTENDED_FEATURE_NAMES,
    _serialize_raw_samples,
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


# ---------------------------------------------------------------------------
# raw_samples persistence + backfill round-trip
# ---------------------------------------------------------------------------


def test_proxy_cache_columns_include_raw_samples():
    assert "raw_samples" in PROXY_CACHE_COLUMNS


def test_serialize_roundtrip_list_str():
    payload = ["sample one", "sample two", ""]
    s = _serialize_raw_samples(payload)
    assert isinstance(s, str)
    back = _deserialize_raw_samples(s)
    assert back == payload


def test_serialize_handles_none():
    assert _serialize_raw_samples(None) is None
    assert _deserialize_raw_samples(None) is None
    assert _deserialize_raw_samples("") is None


def test_deserialize_passthrough_list():
    # On the in-memory path (pre-parquet) we may already have a list.
    assert _deserialize_raw_samples(["a", "b"]) == ["a", "b"]


def test_extract_features_from_samples_includes_every_proxy_name():
    samples = [
        "1. think\n2. answer\n\\boxed{42}",
        "Step 1: try x\nStep 2: try y\nAnswer: 42",
        "Answer: 42",
    ]
    feats = extract_features_from_samples(samples, p_true_value=0.7)
    # Must include EVERY scalar in PROXY_FEATURE_NAMES (stable superset).
    for k in PROXY_FEATURE_NAMES:
        assert k in feats, f"missing feature: {k}"
    assert feats["p_true"] == 0.7              # caller-supplied p_true preserved
    assert feats["self_consistency"] == 1.0    # all answers parse to "42"
    assert feats["boxed_rate"] > 0.0
    assert feats["mean_trace_len"] > 0.0       # whitespace fallback nonzero


def test_extract_features_zero_samples_returns_full_schema():
    feats = extract_features_from_samples([])
    for k in PROXY_FEATURE_NAMES:
        assert k in feats
    # P(True) defaults to 0.5 when not supplied; sample-derived stats are 0.
    assert feats["p_true"] == 0.5
    assert feats["self_consistency"] == 0.0
    assert feats["mean_trace_len"] == 0.0


def test_backfill_dataframe_rederives_missing_columns():
    # Simulate a legacy cache row (no raw_samples) + a new row (with).
    legacy = {
        "item_key": "leg1",
        "self_consistency": 0.6,
        "answer_entropy": 0.4,
        "fsd": 0.2,
        "n_distinct": 0.5,
        "mean_trace_len": 100.0,
        "refusal_rate": 0.0,
        "p_true": 0.55,
        "modal_answer": "42",
        "raw_samples": None,
        "model_id": "Qwen/Qwen3-8B",
        "config_hash": "fakehash",
    }
    new = dict(legacy)
    new["item_key"] = "new1"
    new["raw_samples"] = _serialize_raw_samples([
        "Answer: 42", "Answer: 42", "\\boxed{42}"
    ])
    df = pd.DataFrame([legacy, new])
    new_df, n = _backfill_dataframe(df)
    assert n == 1   # only the row with raw_samples was backfilled
    # Extended columns added to BOTH rows (NaN on legacy, derived on new).
    for col in _EXTENDED_FEATURE_NAMES:
        assert col in new_df.columns
    legacy_row = new_df[new_df["item_key"] == "leg1"].iloc[0]
    new_row = new_df[new_df["item_key"] == "new1"].iloc[0]
    # Legacy row: extended columns are NaN.
    import math
    assert math.isnan(float(legacy_row["boxed_rate"]))
    # New row: extended columns populated; boxed_rate > 0 (one of three).
    assert float(new_row["boxed_rate"]) > 0.0
    # P(True) preserved from cached row (not overwritten to default 0.5).
    assert abs(float(new_row["p_true"]) - 0.55) < 1e-9
    # Legacy aggregate columns untouched.
    assert abs(float(legacy_row["self_consistency"]) - 0.6) < 1e-9


def test_backfill_skips_empty_or_missing_raw_samples():
    df = pd.DataFrame([
        {"item_key": "a", "raw_samples": None, "self_consistency": 0.1,
         "modal_answer": "x", "model_id": "m", "config_hash": "h"},
        {"item_key": "b", "raw_samples": _serialize_raw_samples([]),
         "self_consistency": 0.2, "modal_answer": "y",
         "model_id": "m", "config_hash": "h"},
    ])
    _, n = _backfill_dataframe(df)
    assert n == 0


def test_backfill_on_dataframe_without_raw_samples_column_is_noop():
    df = pd.DataFrame([
        {"item_key": "a", "self_consistency": 0.1, "modal_answer": "x"},
    ])
    out, n = _backfill_dataframe(df)
    assert n == 0
    # Original row unchanged.
    assert out.iloc[0]["self_consistency"] == 0.1


def test_backfill_derived_features_round_trip_on_disk(tmp_path: Path):
    cfg = SolverProxyConfig(
        model_id="test/model-id",
        cache_dir=str(tmp_path),
        n_samples=3,
        compute_p_true=False,
    )
    # Hand-craft a parquet at the cfg's slug path.
    path = _cache_path(cfg)
    legacy_row = {
        "item_key": "leg1",
        "self_consistency": 0.5, "answer_entropy": 0.5, "fsd": 0.0,
        "n_distinct": 1.0, "mean_trace_len": 0.0, "refusal_rate": 0.0,
        "p_true": 0.5, "modal_answer": "<none>", "raw_samples": None,
        "model_id": cfg.model_id, "config_hash": config_hash(cfg),
    }
    new_row = dict(legacy_row)
    new_row["item_key"] = "new1"
    new_row["raw_samples"] = _serialize_raw_samples([
        "1. think\n2. solve\nAnswer: 42",
        "Answer: 42",
        "Step 1: yes\nStep 2: no\n\\boxed{42}",
    ])
    pd.DataFrame([legacy_row, new_row]).to_parquet(path, index=False)
    out_path, n = backfill_derived_features(cfg)
    assert out_path == path
    assert n == 1
    reread = pd.read_parquet(path)
    new_back = reread[reread["item_key"] == "new1"].iloc[0]
    # Extended columns populated for the row that had raw_samples.
    assert float(new_back["boxed_rate"]) > 0.0
    assert float(new_back["trace_steps_mean"]) > 0.0
    # Round-trip preserved the raw_samples payload.
    samples_back = _deserialize_raw_samples(new_back["raw_samples"])
    assert isinstance(samples_back, list) and len(samples_back) == 3
