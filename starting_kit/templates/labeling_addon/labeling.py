"""Template: optional adaptive labeling add-on.

Copy this file into your submission as `labeling.py` to score hidden
(model_id, item_id) pairs by labeling priority. The ingestion program calls
acquisition_function() once per pair BEFORE calling predict(). Higher = more
desired. The active platform reveals K=5 labels per data category. If this
function raises, times out, or returns a non-finite value, the platform falls
back to random label selection for the round.
"""

from __future__ import annotations


def acquisition_function(input: dict) -> float:
    """Return a labeling-priority score for one pair. Higher = more desired.

    Parameters
    ----------
    input : dict
        Same shape as the `input` passed to predict():
        keys benchmark, condition, subject_content, item_content.
    """
    return 0.0
