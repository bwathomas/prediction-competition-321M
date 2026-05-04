"""Toy acquisition: prefer items whose item_content is short and
subjects whose name we have not yet asked about. Single-pass, uses
module-level state to remember subjects already scored.
"""

from __future__ import annotations

_SEEN_SUBJECTS: set[str] = set()


def acquisition_function(input: dict) -> float:
    name = input["subject_content"].split("\n", 1)[0]  # "Name: ..."
    novelty = 0.0 if name in _SEEN_SUBJECTS else 1.0
    _SEEN_SUBJECTS.add(name)
    short_bonus = 1.0 / (1.0 + len(input["item_content"]))
    return novelty + short_bonus
