"""Runtime entry point for the logistic baseline.

Loads the pretrained sklearn Pipeline, extracts its weights into a
`FastScorer` (so per-row inference is ~10us instead of ~30ms), and
exposes the four-field predict() the harness / Codabench platform
expects.

Memoization: the only signal we use from `subject_content` is the model
name, and `item_content` is unused -- so identical (benchmark, condition,
name) triples must yield identical probabilities. We cache them in a
dict to avoid recomputing the logit even when it's already cheap.

Returns DEFAULT_PROB if anything goes wrong (so we never crash the
round) and clips the final probability to [eps, 1-eps] so log-likelihood
is always finite.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import joblib

from fast_scorer import FastScorer
from features import FeatureBuilder, extract_name_from_subject_content

EPS = 1e-3
DEFAULT_PROB = 0.5

_BUNDLE = joblib.load(HERE / "pipeline.joblib")
_PIPELINE = _BUNDLE["pipeline"]

_FEATURE_BUILDER = FeatureBuilder.from_csvs(
    HERE / "model_info.csv",
    HERE / "benchmark_info.csv",
)
_SCORER = FastScorer.from_pipeline(_PIPELINE)

_CACHE: dict[tuple[str, str, str], float] = {}


def predict(input, labeled=None):
    """Return the predicted probability that the subject answers correctly."""
    del labeled
    try:
        benchmark = str(input.get("benchmark", "") or "")
        condition = str(input.get("condition", "none") or "none")
        subject_content = str(input.get("subject_content", "") or "")
        name = extract_name_from_subject_content(subject_content)

        key = (benchmark, condition, name)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        feats = _FEATURE_BUILDER.to_feature_row(
            benchmark=benchmark,
            condition=condition,
            subject_content=subject_content,
            item_content="",
        )
        p = _SCORER.score(feats)
        if not math.isfinite(p):
            p = DEFAULT_PROB
        p = float(min(max(p, EPS), 1.0 - EPS))
        _CACHE[key] = p
        return p
    except Exception:
        return DEFAULT_PROB
