"""Shared utilities for the validation harness.

The four input fields are the *only* thing model.predict and
labeling.acquisition_function are allowed to receive. We centralize the
constant and the row->input dict construction here so the rest of the
harness can't accidentally leak data_category, subject_id, item_id, label,
etc. into participant code.
"""

from __future__ import annotations

from typing import Mapping

# The exact four-string contract from README.md (do not reorder).
INPUT_FIELDS: tuple[str, str, str, str] = (
    "benchmark",
    "condition",
    "subject_content",
    "item_content",
)


def normalize_condition(value: object) -> str:
    """Normalize a raw test_condition value to the runtime contract.

    Per README.md: missing / null / blank conditions become the literal "none".
    """
    if value is None:
        return "none"
    s = str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "none"
    return s


def row_to_input(row: Mapping[str, object]) -> dict[str, str]:
    """Build the four-field input dict the runtime would pass to predict().

    Always returns a fresh dict with exactly the four allowed keys, all str.
    Conditions are normalized again defensively so callers don't have to.
    """
    return {
        "benchmark": str(row["benchmark"]),
        "condition": normalize_condition(row.get("condition")),
        "subject_content": str(row["subject_content"]),
        "item_content": str(row["item_content"]),
    }
