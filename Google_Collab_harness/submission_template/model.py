"""Runtime model.py for the metadata-only latent-factor submission.

This file is dropped into the submission folder by `_build_submission_folder`
in `run_latent_factor_colab.py`, alongside:
    best_model.pt          PyTorch checkpoint
    preprocessor.pkl       fitted MetadataPreprocessor (with model_info /
                           benchmark_info baked in)
    model_info.csv         (audit copy)
    benchmark_info.csv     (audit copy)
    latent_factor_pytorch.py  the model class lives here

The harness loads this module ONCE per round (via Submission.reset()) and
then calls predict(input, labeled) per candidate. We cache by
(benchmark, condition, model_name) since the model is intentionally
item-content-independent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

from latent_factor_pytorch import LatentFactorInference, load_artifacts

EPS = 1e-3
DEFAULT_PROB = 0.5

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL, _PREPROCESSOR, _CONFIG = load_artifacts(HERE, device=_DEVICE)
_INFERENCE = LatentFactorInference(_MODEL, _PREPROCESSOR, device=_DEVICE)


def predict(input, labeled=None):
    """Return the predicted probability that the subject answers correctly."""
    del labeled
    try:
        benchmark = str(input.get("benchmark", "") or "")
        condition = str(input.get("condition", "none") or "none")
        subject_content = str(input.get("subject_content", "") or "")
        p = _INFERENCE.predict_one(benchmark, condition, subject_content)
    except Exception:
        return DEFAULT_PROB
    if not math.isfinite(p):
        return DEFAULT_PROB
    return float(min(max(p, EPS), 1.0 - EPS))
