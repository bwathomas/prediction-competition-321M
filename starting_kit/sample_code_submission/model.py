"""Sample code submission for the Predictive AI Evaluation Challenge.

Your submission must define a single function:

    predict(input: dict, labeled: list[dict] | None = None) -> float

The ingestion program calls predict() once per hidden (model_id, item_id) pair. Module-level code
runs once when the container starts. Load weights, tokenizers, prompt
templates here. Heavy training must be done OFFLINE (e.g. publish a model
to HuggingFace and load it at module init).

`input` keys
------------
    benchmark        Benchmark identifier (e.g. "mmlupro", "ai2d_test").
    condition        Test condition (e.g. "zero-shot"). Literal "none" when
                     no condition applies.
    subject_content  Text description of the AI subject being evaluated,
                     beginning with a "Name:" line.
    item_content     The question / prompt / task text the subject is asked.

`labeled` (optional)
--------------------
    A list of dicts shaped like `input` plus a `label` field (0 or 1).
    These are revealed via adaptive labeling (see labeling.py). May be None
    or empty.

Return value
------------
    A single float in [0, 1], the predicted probability that the subject
    answers the item correctly.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# Replace this with model loading, tokenizer setup, prompt templates, etc.
# ---------------------------------------------------------------------------


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return the predicted probability that the subject answers correctly."""
    return 0.5
