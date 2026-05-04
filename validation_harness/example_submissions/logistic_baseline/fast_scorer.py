"""Linear model scorer that bypasses the sklearn Pipeline at inference time.

The fitted Pipeline is great for training and batched scoring, but per-row
inference through `ColumnTransformer.transform(1-row DataFrame)` costs
~30ms because of all the dict/array allocations.

This scorer extracts the fitted parameters once and computes the same
logit by hand. ~10us per call -- 1000x faster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class _NumericSpec:
    impute_mean: float
    scaler_mean: float
    scaler_scale: float
    coef: float


@dataclass
class _IndicatorSpec:
    coef: float


@dataclass
class _CategoricalSpec:
    """One sklearn OneHotEncoder column.

    `cat_to_coef` maps each known category string to its LR coefficient.
    `infrequent_coef` is used for any category collapsed by min_frequency,
    which is also where unseen test-time categories land thanks to
    handle_unknown='infrequent_if_exist'. None means "no infrequent sink"
    (in which case truly unseen categories contribute 0).
    """
    cat_to_coef: dict[str, float]
    infrequent_coef: float | None


class FastScorer:
    def __init__(
        self,
        numeric: dict[str, _NumericSpec],
        categorical: dict[str, _CategoricalSpec],
        indicator: dict[str, _IndicatorSpec],
        intercept: float,
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.indicator = indicator
        self.intercept = float(intercept)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> "FastScorer":
        pre: ColumnTransformer = pipeline.named_steps["pre"]
        clf: LogisticRegression = pipeline.named_steps["clf"]
        coef = clf.coef_.ravel()

        feature_names = list(pre.get_feature_names_out())
        coef_map = dict(zip(feature_names, coef))

        numeric: dict[str, _NumericSpec] = {}
        categorical: dict[str, _CategoricalSpec] = {}
        indicator: dict[str, _IndicatorSpec] = {}

        for name, transformer, columns in pre.transformers_:
            if name == "num":
                inner: Pipeline = transformer
                imputer = inner.named_steps["impute"]
                scaler = inner.named_steps["scale"]
                for i, col in enumerate(columns):
                    numeric[col] = _NumericSpec(
                        impute_mean=float(imputer.statistics_[i]),
                        scaler_mean=float(scaler.mean_[i]),
                        scaler_scale=float(scaler.scale_[i] or 1.0),
                        coef=float(coef_map.get(f"num__{col}", 0.0)),
                    )
            elif name == "cat":
                inner = transformer
                ohe = inner.named_steps["onehot"]
                for col_idx, col in enumerate(columns):
                    cats = list(ohe.categories_[col_idx])
                    infreq = None
                    if hasattr(ohe, "infrequent_categories_") and ohe.infrequent_categories_ is not None:
                        infreq = ohe.infrequent_categories_[col_idx]
                    cat_to_coef: dict[str, float] = {}
                    infrequent_coef: float | None = None
                    infreq_set = set(map(str, infreq)) if infreq is not None else set()
                    for cat in cats:
                        s = str(cat)
                        if s in infreq_set:
                            continue
                        feat = f"cat__{col}_{s}"
                        cat_to_coef[s] = float(coef_map.get(feat, 0.0))
                    if infreq_set:
                        infrequent_coef = float(coef_map.get(f"cat__{col}_infrequent_sklearn", 0.0))
                        for s in infreq_set:
                            cat_to_coef.setdefault(s, infrequent_coef)
                    categorical[col] = _CategoricalSpec(
                        cat_to_coef=cat_to_coef,
                        infrequent_coef=infrequent_coef,
                    )
            elif name == "ind":
                for col in columns:
                    indicator[col] = _IndicatorSpec(
                        coef=float(coef_map.get(f"ind__{col}", 0.0))
                    )
        return cls(numeric, categorical, indicator, float(clf.intercept_[0]))

    def score(self, features: Mapping[str, object]) -> float:
        """Compute sigmoid(w . phi(features) + b)."""
        z = self.intercept

        for col, spec in self.numeric.items():
            v = features.get(col)
            if v is None:
                v = spec.impute_mean
            else:
                try:
                    v = float(v)
                    if not math.isfinite(v):
                        v = spec.impute_mean
                except (TypeError, ValueError):
                    v = spec.impute_mean
            z += spec.coef * ((v - spec.scaler_mean) / spec.scaler_scale)

        for col, cat_spec in self.categorical.items():
            cat_value = features.get(col, "unknown")
            s = str(cat_value) if cat_value is not None else "unknown"
            coef = cat_spec.cat_to_coef.get(s)
            if coef is None and cat_spec.infrequent_coef is not None:
                coef = cat_spec.infrequent_coef
            if coef is not None:
                z += coef

        for col, ind_spec in self.indicator.items():
            v = features.get(col, 0)
            try:
                z += ind_spec.coef * float(v)
            except (TypeError, ValueError):
                pass

        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)
