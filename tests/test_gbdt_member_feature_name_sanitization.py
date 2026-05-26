"""Regression tests for the LightGBM feature-name sanitizer in
``src.gbdt_member``.

LightGBM rejects feature names containing any of ``"\\,:[]{}`` with
``LightGBMError: Do not support special JSON characters in feature
name``. Our schema is mostly machine-emitted (``pool__token_len``,
``cluster__017``, ...) but raw ``condition`` strings come from the
training data and can contain commas (for example a tag like
``1-shot, no_cot``), colons, brackets, or quotes. We sanitize before
passing names to LightGBM so a malformed raw value does not abort
training that has already burned ~10 minutes of feature
construction.

These tests are unit-level (no real LightGBM required) and pin both
the character set we replace and the disambiguation behavior when
two distinct names collide after sanitization.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


GBDT = importlib.import_module("src.gbdt_member")


def test_sanitizer_replaces_each_forbidden_char():
    """All eight forbidden chars get mapped to ``_``."""
    sample = ['a"b', "a\\b", "a,b", "a:b", "a[b", "a]b", "a{b", "a}b"]
    out = GBDT._sanitize_for_lightgbm(sample)
    for original, sanitized in zip(sample, out, strict=True):
        # Every illegal char must have been replaced.
        assert all(ch not in sanitized for ch in '"\\,:[]{}'), (
            f"sanitizer left an illegal char in {sanitized!r} "
            f"(from {original!r})"
        )
    # Each sanitized name should be of the form ``a_b`` for these
    # single-illegal-char inputs.
    assert out == ["a_b"] + [f"a_b__dup{i}" for i in range(1, 8)]


def test_sanitizer_preserves_clean_names_unchanged():
    """A name with no illegal chars is returned verbatim."""
    sample = [
        "theta_s",
        "u_s_0",
        "subj_cat__family__001",
        "pool__token_len",
        "cluster__017",
        "nn__passrate_subject_conditional",
        "cond__default",
    ]
    assert GBDT._sanitize_for_lightgbm(sample) == sample


def test_sanitizer_disambiguates_post_replace_collisions():
    """Two distinct raw names that collapse to the same sanitized form
    must remain distinct (LightGBM also rejects duplicate names)."""
    raw = ["cond__1-shot,no_cot", "cond__1-shot:no_cot"]
    out = GBDT._sanitize_for_lightgbm(raw)
    assert len(set(out)) == len(out), (
        f"sanitizer produced duplicate names: {out}"
    )
    assert all(ch not in name for name in out for ch in '"\\,:[]{}')


def test_sanitizer_handles_empty_input():
    assert GBDT._sanitize_for_lightgbm([]) == []


def test_sanitizer_handles_empty_string_after_replace():
    """A raw name composed entirely of illegal chars maps to the
    placeholder ``feat`` rather than ``''`` (which LightGBM rejects)."""
    out = GBDT._sanitize_for_lightgbm([",,,", "[]"])
    assert all(name and not name.isspace() for name in out)
    # Two distinct raw names must not collide on the same placeholder.
    assert len(set(out)) == 2


def test_sanitizer_coerces_non_string_inputs():
    """Defensive: numbers / None should not crash the sanitizer."""
    out = GBDT._sanitize_for_lightgbm([1, 2.5])
    assert out == ["1", "2.5"]


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_fit_gbdt_member_succeeds_with_dirty_feature_names():
    """End-to-end check: a column called ``cond__1-shot, no_cot`` no
    longer crashes ``fit_gbdt_member``."""
    rng = np.random.default_rng(0)
    N, F = 256, 6
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    dirty_names = (
        "theta_s",
        "u_s_0",
        "cond__1-shot, no_cot",     # comma
        "cond__benchmark:[redacted]",  # colon + brackets
        'cond__"quoted"',            # quotes
        "cond__a\\b",                # backslash
    )
    state = GBDT.fit_gbdt_member(
        X=X,
        y=y,
        feature_names=dirty_names,
        n_estimators=4,
        num_leaves=4,
        min_data_in_leaf=4,
        early_stopping_rounds=4,
        seed=0,
        val_fraction=0.25,
    )
    # Saved state keeps the ORIGINAL names (downstream introspection
    # / packaging expects bit-identical schema names).
    assert state.feature_names == dirty_names
    # And the trees actually predict.
    preds = GBDT.apply_batch(state, X[:8])
    assert preds.shape == (8,)
    assert np.all(np.isfinite(preds))
