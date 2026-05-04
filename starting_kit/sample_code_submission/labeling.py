"""Sample adaptive labeling strategy for the Predictive AI Evaluation Challenge.

This file is OPTIONAL. If included in your submission, the ingestion program
calls acquisition_function() once per hidden (model_id, item_id) pair, BEFORE
calling predict(). Higher scores indicate pairs you want labeled more. The
active platform reveals K=5 labels per data category, using the top valid scores
within each category, and passes those labeled inputs to predict() as the
`labeled` argument.

If you don't include this file, the platform reveals a default per-category
random sample of labels. If acquisition_function() raises, times out, or
returns a non-finite value for any candidate, the platform uses that same
random-selection fallback for the round.
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
