"""Cheap go/no-go harness for a candidate *item-side* proxy feature.

Given (a) any ``item_key -> scalar`` proxy ``z`` (e.g. solver self-consistency
entropy) and (b) the training labels + an item-grouped fold schedule, this
module answers the only two questions that matter before spending GPU-hours:

1. **Does the proxy carry real signal** that survives the model you already
   have? We never compare ``corr(z, label)`` -- that is the diagnostic that
   (correctly) killed the direct LLM judge. Instead we fit an honest
   *item-cold* baseline from the NN pass-rate features (the dominant item-side
   signal the production stack already extracts), then measure whether a tiny
   one-feature logit correction ``logit(p) = logit(p_base) + a + b*z`` reduces
   **held-out NLL**. ``ΔNLL < 0`` on item-disjoint eval rows == real,
   non-redundant signal.

2. **Is the gain where we expect it** -- i.e. concentrated on the cold,
   neighborless items where the production pipeline is weakest? We slice the
   ΔNLL by NN-support quartile and put an *item-clustered* bootstrap CI on
   every number so a real -0.005 nat win is distinguishable from noise.

Everything here is pure NumPy (no torch / sklearn): a compact IRLS logistic
solver does both the baseline fit and the one-feature correction, so the
harness runs in seconds on millions of rows and is trivially unit-tested.

The honest-split contract: the proxy ``z`` is item-level, so the fit/eval
partition is over **items**, not rows, and the NN baseline is produced
out-of-fold. Both guarantee the reported ΔNLL is never in-sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

LOG = logging.getLogger("proxy_eval")

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Compact IRLS logistic regression (NumPy, L2, optional fixed offset)
# ---------------------------------------------------------------------------


def fit_logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    offset: np.ndarray | None = None,
    n_iter: int = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """Newton/IRLS fit of ``P(y=1) = sigmoid(offset + b0 + X @ w)``.

    Returns the coefficient vector ``[b0, w_0, ..., w_{k-1}]`` (bias first).
    The bias is never regularized; the L2 penalty applies to ``w`` only. An
    optional per-row ``offset`` (e.g. ``logit(p_base)``) is held fixed -- this
    is how the one-feature *correction* on top of a frozen baseline is fit.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, k = X.shape
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float64), X], axis=1)
    off = (
        np.zeros(n, dtype=np.float64)
        if offset is None
        else np.asarray(offset, dtype=np.float64).reshape(-1)
    )
    w = np.zeros(k + 1, dtype=np.float64)
    reg = np.eye(k + 1, dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0  # never penalize the bias
    for _ in range(int(n_iter)):
        eta = off + Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        W = np.clip(p * (1.0 - p), 1e-9, None)
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * W[:, None]).T @ Xb + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w


def _predict_logits(X: np.ndarray, w: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float64), X], axis=1)
    eta = Xb @ w
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64).reshape(-1)
    return eta


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def bce(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy (natural log / nats)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Honest item-cold NN baseline ("current signal" stand-in)
# ---------------------------------------------------------------------------


def fit_nn_baseline_oof(
    nn_mat: np.ndarray,
    labels: np.ndarray,
    train_row_idx_per_fold: Sequence[np.ndarray],
    oof_row_idx_per_fold: Sequence[np.ndarray],
    *,
    l2: float = 5.0,
    standardize: bool = True,
) -> np.ndarray:
    """Out-of-fold logistic over the NN pass-rate feature block.

    This is the cheap, leakage-free stand-in for "the signal we already have":
    the production stack's strongest item-side member is exactly an NN
    pass-rate model, so a logistic on ``nn_mat`` is a fair lower bound on the
    current ensemble for the purpose of *orthogonality* testing. Each row's
    baseline prediction comes from a fold that never trained on that row's
    item, so ``p_base`` is honest item-cold.

    Returns ``p_base`` aligned with ``nn_mat`` rows (every row covered exactly
    once by the union of the folds' OOF index sets).
    """
    nn_mat = np.asarray(nn_mat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    n = nn_mat.shape[0]
    p_base = np.full(n, np.nan, dtype=np.float64)
    for tr_idx, oof_idx in zip(train_row_idx_per_fold, oof_row_idx_per_fold):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        oof_idx = np.asarray(oof_idx, dtype=np.int64)
        Xtr = nn_mat[tr_idx]
        Xoof = nn_mat[oof_idx]
        if standardize:
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xtr = (Xtr - mu) / sd
            Xoof = (Xoof - mu) / sd
        w = fit_logistic_irls(Xtr, labels[tr_idx], l2=l2)
        p_base[oof_idx] = _sigmoid(_predict_logits(Xoof, w))
    # Any uncovered rows (shouldn't happen with a partition) fall back to the
    # global base rate so downstream code never sees a NaN.
    missing = ~np.isfinite(p_base)
    if missing.any():
        p_base[missing] = float(np.clip(labels.mean(), _EPS, 1.0 - _EPS))
    return p_base


# ---------------------------------------------------------------------------
# Incremental-NLL test: does z reduce held-out NLL on top of p_base?
# ---------------------------------------------------------------------------


@dataclass
class IncrementalNLLResult:
    """Outcome of one incremental-NLL slice."""

    name: str
    n_rows: int
    n_items: int
    nll_base: float
    nll_with_z: float
    delta_nll: float            # nll_with_z - nll_base ; negative == z helps
    delta_nll_ci: tuple[float, float]  # item-clustered bootstrap CI
    coef_z: float               # fitted correction slope on standardized z
    item_partial_corr: float    # corr(z_item, mean residual_item)

    @property
    def helps(self) -> bool:
        """``True`` when the bootstrap CI for ΔNLL lies fully below 0."""
        return self.delta_nll_ci[1] < 0.0

    def format_line(self) -> str:
        lo, hi = self.delta_nll_ci
        flag = "  <-- SIGNAL" if self.helps else ""
        return (
            f"  [{self.name:<18}] n_rows={self.n_rows:>9,} n_items={self.n_items:>6,}  "
            f"NLL {self.nll_base:.5f} -> {self.nll_with_z:.5f}  "
            f"ΔNLL={self.delta_nll:+.5f} (95% CI [{lo:+.5f},{hi:+.5f}])  "
            f"b_z={self.coef_z:+.3f}  r_item={self.item_partial_corr:+.3f}{flag}"
        )


def _fit_eval_item_split(
    item_key_per_row: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into fit/eval halves by *item* (item-disjoint)."""
    uniq = np.unique(item_key_per_row)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.shape[0])
    half = uniq.shape[0] // 2
    fit_items = set(uniq[perm[:half]].tolist())
    is_fit = np.fromiter(
        (k in fit_items for k in item_key_per_row),
        count=item_key_per_row.shape[0],
        dtype=bool,
    )
    return np.where(is_fit)[0], np.where(~is_fit)[0]


def _delta_nll_on_eval(
    p_base_eval: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    a: float,
    b: float,
) -> tuple[float, float]:
    """(nll_base, nll_with_z) on eval rows given fitted correction (a, b)."""
    eta_base = _logit(p_base_eval)
    p_new = _sigmoid(eta_base + a + b * z_eval)
    return bce(y_eval, p_base_eval), bce(y_eval, p_new)


def _item_clustered_bootstrap_delta(
    p_base_eval: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    item_eval: np.ndarray,
    a: float,
    b: float,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Item-clustered bootstrap CI for ΔNLL = nll_with_z - nll_base.

    Resamples whole *items* (rows of the same item move together) so the CI
    respects the true independent unit. Returns ``(lo, hi)`` percentile CI.
    """
    if n_boot <= 0:
        return (np.nan, np.nan)
    eta_base = _logit(p_base_eval)
    p_new = _sigmoid(eta_base + a + b * z_eval)
    y = np.asarray(y_eval, dtype=np.float64)
    pb = np.clip(p_base_eval, _EPS, 1.0 - _EPS)
    pn = np.clip(p_new, _EPS, 1.0 - _EPS)
    # Per-row contributions to NLL; ΔNLL = mean(loss_new - loss_base).
    loss_base = -(y * np.log(pb) + (1.0 - y) * np.log(1.0 - pb))
    loss_new = -(y * np.log(pn) + (1.0 - y) * np.log(1.0 - pn))
    d_row = loss_new - loss_base

    uniq, inv = np.unique(item_eval, return_inverse=True)
    n_items = uniq.shape[0]
    # Pre-aggregate per item: sum of d_row and count, so each bootstrap draw
    # is an O(n_items) gather instead of O(n_rows).
    d_sum = np.bincount(inv, weights=d_row, minlength=n_items)
    cnt = np.bincount(inv, minlength=n_items).astype(np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        pick = rng.integers(0, n_items, size=n_items)
        stats[bi] = d_sum[pick].sum() / max(cnt[pick].sum(), 1.0)
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return (lo, hi)


def incremental_nll_test(
    *,
    p_base: np.ndarray,
    z_per_row: np.ndarray,
    labels: np.ndarray,
    item_key_per_row: np.ndarray,
    name: str = "all",
    n_boot: int = 500,
    seed: int = 0,
) -> IncrementalNLLResult:
    """Fit ``logit(p)=logit(p_base)+a+b*z`` on half the items, score the other.

    ``z`` is standardized (using fit-half stats) so ``b`` is interpretable and
    the IRLS stays well-conditioned. Returns held-out ΔNLL with an
    item-clustered bootstrap CI and the item-level partial correlation between
    ``z`` and the baseline residual.
    """
    p_base = np.asarray(p_base, dtype=np.float64).reshape(-1)
    z = np.asarray(z_per_row, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    items = np.asarray(item_key_per_row).reshape(-1)
    n = p_base.shape[0]
    if not (z.shape[0] == y.shape[0] == items.shape[0] == n):
        raise ValueError("incremental_nll_test: length mismatch among inputs")

    fit_idx, eval_idx = _fit_eval_item_split(items, seed=seed)
    if fit_idx.size == 0 or eval_idx.size == 0:
        raise ValueError("incremental_nll_test: empty fit/eval split (need >=2 items)")

    # Standardize z on the fit half only.
    mu = float(z[fit_idx].mean())
    sd = float(z[fit_idx].std())
    sd = sd if sd > 1e-8 else 1.0
    z_std = (z - mu) / sd

    # Fit the 1-feature correction with logit(p_base) as a fixed offset.
    w = fit_logistic_irls(
        z_std[fit_idx].reshape(-1, 1),
        y[fit_idx],
        l2=1e-3,
        offset=_logit(p_base[fit_idx]),
    )
    a, b = float(w[0]), float(w[1])

    nll_base, nll_with_z = _delta_nll_on_eval(
        p_base[eval_idx], z_std[eval_idx], y[eval_idx], a, b
    )
    ci = _item_clustered_bootstrap_delta(
        p_base[eval_idx], z_std[eval_idx], y[eval_idx], items[eval_idx],
        a, b, n_boot=n_boot, seed=seed + 1,
    )

    # Item-level partial corr: aggregate residual + z to one row per item on
    # the eval half (so it mirrors the held-out ΔNLL), then Pearson.
    resid = y[eval_idx] - p_base[eval_idx]
    uniq, inv = np.unique(items[eval_idx], return_inverse=True)
    r_item = np.bincount(inv, weights=resid, minlength=uniq.shape[0]) / np.maximum(
        np.bincount(inv, minlength=uniq.shape[0]), 1
    )
    z_item = np.bincount(inv, weights=z_std[eval_idx], minlength=uniq.shape[0]) / np.maximum(
        np.bincount(inv, minlength=uniq.shape[0]), 1
    )
    if r_item.std() < 1e-9 or z_item.std() < 1e-9:
        pcorr = 0.0
    else:
        pcorr = float(np.corrcoef(z_item, r_item)[0, 1])

    return IncrementalNLLResult(
        name=name,
        n_rows=int(eval_idx.size),
        n_items=int(uniq.shape[0]),
        nll_base=nll_base,
        nll_with_z=nll_with_z,
        delta_nll=nll_with_z - nll_base,
        delta_nll_ci=ci,
        coef_z=b,
        item_partial_corr=pcorr,
    )


# ---------------------------------------------------------------------------
# NN-support slicing
# ---------------------------------------------------------------------------


def support_quartile_masks(
    support_per_row: np.ndarray, *, n_buckets: int = 4
) -> list[tuple[str, np.ndarray]]:
    """Bucket rows by an NN-support scalar (low support == cold items).

    Returns ``[(label, mask), ...]`` with quartile edges computed from the
    distribution. Bucket Q1 is the lowest-support (coldest) slice -- the one
    where an orthogonal solvability proxy is expected to pay off most.
    """
    s = np.asarray(support_per_row, dtype=np.float64).reshape(-1)
    qs = np.quantile(s, np.linspace(0.0, 1.0, n_buckets + 1))
    # De-duplicate edges so heavily-tied supports don't crash digitize.
    qs = np.unique(qs)
    if qs.shape[0] < 2:
        return [("all", np.ones_like(s, dtype=bool))]
    edges = qs.copy()
    edges[-1] = np.inf
    bucket = np.digitize(s, edges[1:-1], right=False)
    out: list[tuple[str, np.ndarray]] = []
    nb = qs.shape[0] - 1
    for b in range(nb):
        out.append((f"Q{b + 1}_support", bucket == b))
    return out


def run_proxy_probe(
    *,
    p_base: np.ndarray,
    z_per_row: np.ndarray,
    labels: np.ndarray,
    item_key_per_row: np.ndarray,
    support_per_row: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
    n_support_buckets: int = 4,
) -> dict[str, IncrementalNLLResult]:
    """Full probe: overall ΔNLL + per-NN-support-quartile ΔNLL.

    Returns a dict ``{slice_name: IncrementalNLLResult}``. ``"all"`` is always
    present; per-quartile slices are added when ``support_per_row`` is given.
    """
    results: dict[str, IncrementalNLLResult] = {}
    results["all"] = incremental_nll_test(
        p_base=p_base, z_per_row=z_per_row, labels=labels,
        item_key_per_row=item_key_per_row, name="all", n_boot=n_boot, seed=seed,
    )
    if support_per_row is not None:
        for label, mask in support_quartile_masks(
            support_per_row, n_buckets=n_support_buckets
        ):
            if mask.sum() < 50 or np.unique(np.asarray(item_key_per_row)[mask]).size < 4:
                continue
            results[label] = incremental_nll_test(
                p_base=np.asarray(p_base)[mask],
                z_per_row=np.asarray(z_per_row)[mask],
                labels=np.asarray(labels)[mask],
                item_key_per_row=np.asarray(item_key_per_row)[mask],
                name=label, n_boot=n_boot, seed=seed,
            )
    return results


def format_probe_report(
    results: Mapping[str, IncrementalNLLResult], *, title: str = "proxy probe"
) -> str:
    """Pretty multi-line report for the notebook to print."""
    lines = [f"[{title}] incremental-NLL vs honest item-cold NN baseline:"]
    for key in ["all"] + sorted(k for k in results if k != "all"):
        if key in results:
            lines.append(results[key].format_line())
    any_signal = any(r.helps for r in results.values())
    verdict = (
        "VERDICT: proxy shows held-out signal in >=1 slice -- worth pursuing."
        if any_signal
        else "VERDICT: no slice clears the bootstrap CI -- likely redundant; do "
        "NOT spend GPU-hours scaling this up."
    )
    lines.append(verdict)
    return "\n".join(lines)


__all__ = [
    "IncrementalNLLResult",
    "bce",
    "fit_logistic_irls",
    "fit_nn_baseline_oof",
    "format_probe_report",
    "incremental_nll_test",
    "run_proxy_probe",
    "support_quartile_masks",
]
