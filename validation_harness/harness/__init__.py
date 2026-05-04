"""Local validation harness for the Predictive AI Evaluation Challenge.

Mirrors the official platform's protocol as closely as possible:
- item cold-start (validation item-variants do NOT appear in training)
- validation subjects MUST appear in training
- model.predict receives ONLY {benchmark, condition, subject_content, item_content}
- adaptive labeling reveals top-K candidates per data category
- score on negative log-likelihood (higher = better) and AUC-ROC

See README.md for usage.
"""

from .utils import INPUT_FIELDS, normalize_condition, row_to_input
from .splits import (
    add_item_variant_id,
    make_item_cold_start_split,
)
from .sampling import stratified_sample_variants
from .rounds import run_official_like_round
from .scoring import mean_log_likelihood, auc_roc, score_round
from .submission import Submission

__all__ = [
    "INPUT_FIELDS",
    "normalize_condition",
    "row_to_input",
    "add_item_variant_id",
    "make_item_cold_start_split",
    "stratified_sample_variants",
    "run_official_like_round",
    "mean_log_likelihood",
    "auc_roc",
    "score_round",
    "Submission",
]
