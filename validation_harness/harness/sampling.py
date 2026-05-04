"""Stratified-by-data_category item-variant sampler.

Per the README, the platform samples N=5000 hidden items per submission,
"stratified across data categories". We treat that literally:
- every category present in the validation pool is used
- each category gets approximately N / num_categories item-variants
- if a category has fewer variants than its allocation, the slack is
  redistributed iteratively to other categories
- the leftover (when N does not divide evenly) is allocated by sampling
  categories without replacement using the seed

Sampling is over item-variants, NOT raw rows, because at runtime an item
variant is the indivisible unit each subject is asked about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _allocate_capped(n_total: int, capacities: dict[str, int], rng: np.random.Generator) -> dict[str, int]:
    """Allocate n_total slots across categories, never exceeding capacity.

    Equal share where possible; remainder distributed by random tie-breaking;
    slack from saturated categories iteratively redistributed.
    """
    cats = list(capacities.keys())
    allocation = {c: 0 for c in cats}
    remaining = min(n_total, sum(capacities.values()))
    pending = [c for c in cats if capacities[c] > 0]

    while remaining > 0 and pending:
        base = remaining // len(pending)
        if base == 0:
            order = list(pending)
            rng.shuffle(order)
            for c in order[:remaining]:
                if allocation[c] < capacities[c]:
                    allocation[c] += 1
                    remaining -= 1
            break
        progressed = False
        next_pending = []
        for c in pending:
            cap_left = capacities[c] - allocation[c]
            take = min(base, cap_left)
            if take > 0:
                allocation[c] += take
                remaining -= take
                progressed = True
            if allocation[c] < capacities[c]:
                next_pending.append(c)
        pending = next_pending
        if not progressed:
            break
    return allocation


def stratified_sample_variants(
    val_df: pd.DataFrame,
    n_samples: int,
    *,
    category_col: str = "data_category",
    variant_col: str = "item_variant_id",
    seed: int = 0,
) -> list[str]:
    """Return up to n_samples unique item-variant ids, stratified by category.

    The returned list is the set of variants whose induced rows form the
    candidate pool for the round.
    """
    rng = np.random.default_rng(seed)

    cat_to_variants: dict[str, list[str]] = {}
    cat_order = sorted(val_df[category_col].astype(str).unique())
    for cat in cat_order:
        v = val_df.loc[val_df[category_col].astype(str) == cat, variant_col].astype(str).unique()
        cat_to_variants[cat] = sorted(v.tolist())  # deterministic order pre-shuffle

    capacities = {c: len(v) for c, v in cat_to_variants.items()}
    allocation = _allocate_capped(n_samples, capacities, rng)

    selected: list[str] = []
    for cat in cat_order:
        n = allocation[cat]
        if n <= 0:
            continue
        variants = cat_to_variants[cat]
        if n >= len(variants):
            chosen = list(variants)
        else:
            idx = rng.choice(len(variants), size=n, replace=False)
            chosen = [variants[i] for i in idx]
        selected.extend(chosen)
    return selected
