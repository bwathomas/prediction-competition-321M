"""Regression coverage for the notebook's metadata-coverage assertion.

Cell 7 of ``notebooks/qwen8b_four_member_stacked.py`` defines a local
``_assert_metadata_coverage`` helper that fails LOUD when a declared
schema field made it through ``MetadataPreprocessor`` with zero data
populated (e.g. CSV column missing, ``age`` rename failed, join-key
mismatch). We test the same logic here against the canonical
``MetadataPreprocessor`` + ``build_metadata_id_tables`` pipeline so a
silent regression in those modules surfaces immediately.

CPU-only, < 100 ms, no FAISS / no torch tensors > a few KB.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import pytest
import torch

from src.metadata_features import (
    MetadataPreprocessor,
    MetadataSchema,
    build_metadata_id_tables,
)


def _coverage_for(id_tables, schema, *, side: str, field: str) -> float:
    """Replicate the notebook's coverage formula for one field.

    Mask convention: ``mask < 0.5`` means PRESENT (set by
    ``NumericScaler.transform`` whenever the source value was finite).
    """
    if side == "subject_cat":
        cat = id_tables.subject_cat_ids.cpu().numpy()[1:]
        j = list(schema.subject_categorical).index(field)
        return float((cat[:, j] != 0).sum()) / max(cat.shape[0], 1)
    if side == "subject_num":
        num = id_tables.subject_num.cpu().numpy()[1:]
        j = list(schema.subject_numeric).index(field)
        return float((num[:, 2 * j + 1] < 0.5).sum()) / max(num.shape[0], 1)
    if side == "bench_cat":
        cat = id_tables.bc_cat_ids.cpu().numpy()[1:]
        j = list(schema.benchmark_categorical).index(field)
        return float((cat[:, j] != 0).sum()) / max(cat.shape[0], 1)
    if side == "bench_num":
        num = id_tables.bc_num.cpu().numpy()[1:]
        j = list(schema.benchmark_numeric).index(field)
        return float((num[:, 2 * j + 1] < 0.5).sum()) / max(num.shape[0], 1)
    raise ValueError(side)


def _build_tables(
    *,
    model_info: pd.DataFrame,
    benchmark_info: pd.DataFrame,
    schema: MetadataSchema,
    subject_to_id: Mapping[str, int],
    bc_to_id: Mapping[str, int],
    subject_content_by_key: Mapping[str, str],
):
    mp = MetadataPreprocessor.fit(model_info, benchmark_info, schema=schema)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=subject_to_id,
        bc_to_id=bc_to_id,
        subject_content_by_key=subject_content_by_key,
    )
    return mp, tables


def _good_inputs(schema: MetadataSchema | None = None):
    """Construct a fully-populated synthetic dataset so coverage is 100%."""
    schema = schema or MetadataSchema()
    model_info = pd.DataFrame(
        {
            "name": ["alpha", "beta", "gamma"],
            "organization": ["OrgA", "OrgA", "OrgB"],
            "family": ["Llama", "Llama", "Mistral"],
            "macro_family": ["DenseLM", "DenseLM", "DenseLM"],
            "parameters": [7e9, 70e9, 12e9],
            "release_date": [2024.1, 2024.5, 2024.9],
        }
    )
    benchmark_info = pd.DataFrame(
        {
            "benchmark": ["bench_x", "bench_y", "bench_z"],
            "topic": ["math", "code", "math"],
            "age": [12.0, 18.0, 5.0],
        }
    )
    subject_to_id = {"<unk>": 0, "key_alpha": 1, "key_beta": 2, "key_gamma": 3}
    bc_to_id = {"<unk>": 0, "bench_x::cond1": 1, "bench_y::cond1": 2, "bench_z::cond2": 3}
    subject_content_by_key = {
        "key_alpha": "Name: alpha\nOrg: OrgA",
        "key_beta": "Name: beta\nOrg: OrgA",
        "key_gamma": "Name: gamma\nOrg: OrgB",
    }
    return {
        "model_info": model_info,
        "benchmark_info": benchmark_info,
        "schema": schema,
        "subject_to_id": subject_to_id,
        "bc_to_id": bc_to_id,
        "subject_content_by_key": subject_content_by_key,
    }


def test_full_population_yields_100_percent_for_every_field() -> None:
    """Sanity baseline: a complete dataset should pass the coverage
    diagnostic with every field at 100%. This pins the mask-convention
    interpretation -- if NumericScaler's mask semantics ever invert,
    this test fails immediately."""
    args = _good_inputs()
    schema = args["schema"]
    _, tables = _build_tables(**args)
    for field in schema.subject_categorical:
        assert _coverage_for(tables, schema, side="subject_cat", field=field) > 0.5, field
    for field in schema.subject_numeric:
        assert _coverage_for(tables, schema, side="subject_num", field=field) > 0.99, field
    for field in schema.benchmark_categorical:
        assert _coverage_for(tables, schema, side="bench_cat", field=field) > 0.5, field
    for field in schema.benchmark_numeric:
        assert _coverage_for(tables, schema, side="bench_num", field=field) > 0.99, field


def test_benchmark_age_column_missing_yields_zero_coverage() -> None:
    """Reproduces the user's failure: benchmark_info has no ``age`` /
    ``benchmark_age`` column, so every bc gets mask=1 (missing)."""
    args = _good_inputs()
    args["benchmark_info"] = args["benchmark_info"].drop(columns=["age"])
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="bench_num", field="benchmark_age")
    assert cov == 0.0, f"expected 0% benchmark_age coverage, got {cov}"


def test_age_renamed_to_benchmark_age_is_picked_up() -> None:
    """Confirm the ``age`` -> ``benchmark_age`` rename inside
    ``_normalize_benchmark_info`` actually fires."""
    args = _good_inputs()
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="bench_num", field="benchmark_age")
    assert cov > 0.99


def test_age_in_canonical_name_also_works() -> None:
    """If the source CSV already uses ``benchmark_age`` directly, the
    rename is a no-op and coverage is still 100%."""
    args = _good_inputs()
    args["benchmark_info"] = args["benchmark_info"].rename(
        columns={"age": "benchmark_age"}
    )
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="bench_num", field="benchmark_age")
    assert cov > 0.99


def test_age_with_non_numeric_strings_yields_zero_coverage() -> None:
    """If every value is a non-numeric string, ``pd.to_numeric(errors='coerce')``
    NaNs them all out and coverage is 0%."""
    args = _good_inputs()
    args["benchmark_info"]["age"] = ["Unknown", "Unknown", "Unknown"]
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="bench_num", field="benchmark_age")
    assert cov == 0.0


def test_subject_release_date_missing_yields_zero_coverage() -> None:
    """Same defensive check on the subject side."""
    args = _good_inputs()
    args["model_info"] = args["model_info"].drop(columns=["release_date"])
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="subject_num", field="release_date")
    assert cov == 0.0


def test_subject_log_params_derives_from_parameters() -> None:
    """``log_params`` is computed from the ``parameters`` source column
    -- if ``parameters`` is missing, coverage is 0%."""
    args = _good_inputs()
    args["model_info"] = args["model_info"].drop(columns=["parameters"])
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="subject_num", field="log_params")
    assert cov == 0.0


def test_subject_categorical_join_failure_yields_zero_coverage() -> None:
    """When subject_content's display name does not match anything in
    model_info, every subject ends up with category id 0 (MISSING)."""
    args = _good_inputs()
    args["subject_content_by_key"] = {
        k: "Name: not_a_real_name\nOrg: ???"
        for k in args["subject_content_by_key"]
    }
    _, tables = _build_tables(**args)
    for field in args["schema"].subject_categorical:
        cov = _coverage_for(tables, args["schema"], side="subject_cat", field=field)
        assert cov == 0.0, f"{field}: expected 0%, got {cov}"


def test_partial_coverage_above_threshold_passes() -> None:
    """A field with at least one value populated should not trigger
    the ``min_coverage`` failure (which the notebook sets to 1%)."""
    args = _good_inputs()
    args["benchmark_info"].loc[1:, "age"] = np.nan  # only row 0 has data
    _, tables = _build_tables(**args)
    cov = _coverage_for(tables, args["schema"], side="bench_num", field="benchmark_age")
    assert cov == pytest.approx(1.0 / 3, abs=1e-6)
