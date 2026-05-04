"""Run one official-like evaluation round.

Pipeline (mirrors the platform):
  1. stratified-sample N item-variants from val_df by data_category
  2. expand to all (subject, variant) rows -- the candidate pool
  3. for each candidate, build the strict 4-key input dict
  4. acquisition_function (if present) is called once per candidate, in a
     single pass with no access to siblings except via its own module-level
     state -- exactly like the platform
  5. per data_category, reveal labels for the top-K candidates by score
     (random tie-break); fall back to random K if no labeling module, the
     module is missing acquisition_function, an acquisition call raises, or
     any score is non-finite
  6. build the labeled list (4 input fields + "label") and pass the SAME
     list to every model.predict call in the round
  7. attach predictions to a returned candidates DataFrame for scoring

Important guarantees enforced:
  - data_category is NEVER passed to predict / acquisition_function
  - subject_id, item_id, label are NEVER passed to acquisition_function
  - labeled dicts have exactly the 4 input fields plus "label"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .sampling import stratified_sample_variants
from .utils import INPUT_FIELDS, row_to_input


@dataclass
class RoundResult:
    candidates: pd.DataFrame  # adds _pred, _is_labeled, _acq_score, _acq_tiebreak
    labeled: list[dict]  # the list passed to predict()
    n_candidates: int
    n_labeled: int
    n_categories: int
    used_random_acquisition: bool
    fallback_reason: str | None
    seed: int
    K: int
    N: int


def _is_finite_number(x: Any) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _safe_acquisition(
    labeling_module: Any,
    inputs: list[dict[str, str]],
) -> tuple[list[float] | None, str | None]:
    """Call acquisition_function on every candidate.

    Returns (scores, None) on success, or (None, reason) if any call raises
    or returns a non-finite value.
    """
    fn = getattr(labeling_module, "acquisition_function", None)
    if fn is None:
        return None, "labeling_module has no acquisition_function"
    scores: list[float] = []
    for inp in inputs:
        try:
            s = fn(inp)
        except Exception as e:  # noqa: BLE001 - mirror platform's broad fallback
            return None, f"acquisition_function raised: {type(e).__name__}: {e}"
        if not _is_finite_number(s):
            return None, f"acquisition_function returned non-finite value: {s!r}"
        scores.append(float(s))
    return scores, None


def run_official_like_round(
    train_df: pd.DataFrame,  # accepted for symmetry / future caching, not consumed
    val_df: pd.DataFrame,
    model_module: Any,
    labeling_module: Any | None = None,
    *,
    N: int = 5000,
    K: int = 5,
    seed: int = 0,
    category_col: str = "data_category",
    variant_col: str = "item_variant_id",
) -> RoundResult:
    """Run one round end-to-end and return everything needed for scoring."""
    if not hasattr(model_module, "predict"):
        raise AttributeError("model_module must define predict(input, labeled)")

    rng = np.random.default_rng(seed)

    selected_variants = set(
        stratified_sample_variants(
            val_df,
            N,
            category_col=category_col,
            variant_col=variant_col,
            seed=seed,
        )
    )
    cand = val_df[val_df[variant_col].astype(str).isin(selected_variants)].copy()
    cand = cand.reset_index(drop=True)

    if len(cand) == 0:
        return RoundResult(
            candidates=cand,
            labeled=[],
            n_candidates=0,
            n_labeled=0,
            n_categories=0,
            used_random_acquisition=labeling_module is None,
            fallback_reason="no candidates",
            seed=seed,
            K=K,
            N=N,
        )

    cand_inputs: list[dict[str, str]] = [
        row_to_input(cand.iloc[i]) for i in range(len(cand))
    ]

    scores: list[float] | None = None
    fallback_reason: str | None = None
    used_random = False
    if labeling_module is None:
        used_random = True
        fallback_reason = "no labeling_module supplied"
    else:
        scores, fallback_reason = _safe_acquisition(labeling_module, cand_inputs)
        if scores is None:
            used_random = True
    if used_random:
        scores = rng.random(len(cand_inputs)).tolist()

    cand["_acq_score"] = scores
    cand["_acq_tiebreak"] = rng.random(len(cand))

    labeled_idx: list[int] = []
    cats_present = sorted(cand[category_col].astype(str).unique())
    for cat in cats_present:
        sub = cand[cand[category_col].astype(str) == cat]
        sub_sorted = sub.sort_values(
            by=["_acq_score", "_acq_tiebreak"], ascending=[False, True]
        )
        labeled_idx.extend(sub_sorted.head(K).index.tolist())
    cand["_is_labeled"] = cand.index.isin(labeled_idx)

    labeled: list[dict] = []
    for idx in labeled_idx:
        row = cand.loc[idx]
        d = row_to_input(row)
        d["label"] = float(row["label"])
        labeled.append(d)

    preds: list[float] = []
    for i, inp in enumerate(cand_inputs):
        out = model_module.predict(inp, labeled)
        try:
            p = float(out)
        except (TypeError, ValueError):
            p = float("nan")
        preds.append(p)
    cand["_pred"] = preds

    return RoundResult(
        candidates=cand,
        labeled=labeled,
        n_candidates=len(cand),
        n_labeled=len(labeled),
        n_categories=len(cats_present),
        used_random_acquisition=used_random,
        fallback_reason=fallback_reason if used_random else None,
        seed=seed,
        K=K,
        N=N,
    )
