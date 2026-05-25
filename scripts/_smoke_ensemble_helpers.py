"""Quick offline smoke test for src/ensemble_helpers.py."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.ensemble_helpers import (
    _project_simplex,
    _sigmoid,
    brier_vec,
    compute_diversity_metrics,
    fit_optimal_weights,
    log_loss_vec,
)

rng = np.random.default_rng(0)
N, M = 1000, 4
y = (rng.random(N) < 0.6).astype(float)
P = {}
for i in range(M):
    sign = 1 if i < 2 else -1
    noise = 0.2 * rng.standard_normal(N)
    p = 0.5 + sign * 0.25 * (y - 0.5) + noise
    P[f"m{i}"] = np.clip(p, 0.01, 0.99)

rep = compute_diversity_metrics(P, y)
print("diag pearson:", np.diag(rep.pearson.values))
print("diag mad:", np.diag(rep.mean_abs_diff.values))
print("per_model_loss:")
print(rep.per_model_loss.round(5))
print()
print("--- fits ---")
for m in (
    "uniform_prob",
    "uniform_logit",
    "simplex_prob",
    "simplex_logit",
    "unconstrained_logit",
):
    fit = fit_optimal_weights(P, y, method=m)
    print(
        f"{m:25s} loss={fit.log_loss:.5f}  brier={fit.brier:.5f}  "
        f"weights={np.round(fit.weights, 3).tolist()}  notes={fit.notes!r}"
    )

# simplex projection sanity
v = np.array([0.7, 0.3, -0.4, 0.5])
proj = _project_simplex(v)
print("simplex projection:", proj, "sum:", proj.sum())
assert abs(proj.sum() - 1.0) < 1e-9
assert (proj >= -1e-9).all()

# sigmoid stability for big z
z = np.array([-1e3, -10, 0, 10, 1e3], dtype=np.float64)
print("sigmoid(big):", _sigmoid(z))

# log_loss + brier on partial-finite vectors
preds = np.array([0.1, 0.2, np.nan, 0.7, 0.9])
labels = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
print("log_loss_vec:", log_loss_vec(preds, labels))
print("brier_vec:", brier_vec(preds, labels))
print("OK")
