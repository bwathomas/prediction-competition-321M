"""Tests for the Submission wrapper (reload between rounds)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.submission import Submission

EXAMPLES = Path(__file__).resolve().parent.parent / "example_submissions"


def test_loads_constant_baseline():
    s = Submission(EXAMPLES / "constant_baseline")
    assert s.model is not None
    assert s.labeling is None  # no labeling.py in this example
    assert s.model.predict({"benchmark": "x", "condition": "none",
                             "subject_content": "Name: a", "item_content": "q"}) == 0.5


def test_loads_random_baseline_with_labeling():
    s = Submission(EXAMPLES / "random_baseline")
    assert s.model is not None
    assert s.labeling is not None
    assert hasattr(s.labeling, "acquisition_function")


def test_reset_clears_module_level_state():
    s = Submission(EXAMPLES / "random_baseline")
    inp = {
        "benchmark": "x",
        "condition": "none",
        "subject_content": "Name: a",
        "item_content": "q",
    }
    s.labeling.acquisition_function(inp)
    seen_before = set(s.labeling._SEEN_SUBJECTS)
    assert "Name: a" in seen_before
    s.reset()
    assert "Name: a" not in set(s.labeling._SEEN_SUBJECTS)


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        Submission(EXAMPLES)  # not a submission dir


def test_require_labeling_flag():
    with pytest.raises(FileNotFoundError):
        Submission(EXAMPLES / "constant_baseline", require_labeling=True)
