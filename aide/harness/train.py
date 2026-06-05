"""Two-phase (trial -> full) training with the diversity-aware promotion gate (spec §5.1).

A linear stacker gains from uncorrelated members as much as from individually-strong
ones, so a trial candidate is promoted to a full run if it is EITHER competitive (NLL
within X of the best comparable architecture) OR diversifying (its OOF residuals weakly
correlated with the existing pool — admitted even when its standalone NLL is worse).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def diversity_score(residuals, pool_residuals) -> float:
    """1 - mean SIGNED correlation of this candidate's residuals with each pool member.

    Signed (not absolute) so anti-correlated errors score as MORE diverse — they cancel
    in the stack (cf. negative-correlation learning). Empty pool -> 1.0 (maximally
    diverse: nothing to be correlated with).
    """
    pool = list(pool_residuals)
    if not pool:
        return 1.0
    corrs = [_safe_corr(residuals, pr) for pr in pool]
    return float(1.0 - np.mean(corrs))


def promotion_gate(cand_nll: float, group_best_nll: float, X: float,
                   diversity: float, D: float) -> bool:
    """Promote if competitive (within X of best comparable) OR diversifying (>= D)."""
    competitive = cand_nll <= group_best_nll + X
    diversifying = diversity >= D
    return bool(competitive or diversifying)


@dataclass
class PromotionResult:
    promoted: bool
    trial_nll: float
    full_nll: object  # float when promoted, else None
    diversity: float


def run_two_phase(eval_fn, trial_ds, full_ds, *, group_best_nll: float, X: float,
                  pool_resids, D: float) -> PromotionResult:
    """eval_fn(ds) -> (nll, residuals). Runs trial; only on a passing gate runs full."""
    trial_nll, trial_resid = eval_fn(trial_ds)
    diversity = diversity_score(trial_resid, pool_resids)
    if not promotion_gate(trial_nll, group_best_nll, X, diversity, D):
        return PromotionResult(promoted=False, trial_nll=trial_nll, full_nll=None, diversity=diversity)
    full_nll, _ = eval_fn(full_ds)
    return PromotionResult(promoted=True, trial_nll=trial_nll, full_nll=full_nll, diversity=diversity)
