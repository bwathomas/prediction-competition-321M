"""Evaluation harness: the only component that owns folds, recursion, dropout, and the
leakage probes. A dropped-in model (``fit(X, y)`` / ``predict(X) -> prob``) therefore
cannot see held-out items or undropped identity proxies — hygiene is enforced here, not
trusted to the model.

- ``oof_predict`` / ``evaluate``: layer-1, single 3-fold item-uniform OOF.
- ``build_oof_meta`` + ``recursive_evaluate``: layer-2, nested (recursive) OOF so the
  stacker never trains on a member's in-sample predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hygiene.dropout import apply_proxy_dropout
from ..hygiene.probes import assert_item_disjoint
from ..hygiene.splits import outer_folds, inner_folds
from .metrics import auc_roc, log_loss


@dataclass
class Dataset:
    X: np.ndarray
    feature_columns: list
    y: np.ndarray
    item_keys: np.ndarray
    subjects: np.ndarray
    benchmarks: np.ndarray


@dataclass
class DropoutConfig:
    subject_rate: float = 0.0
    benchmark_rate: float = 0.0


@dataclass
class EvalResult:
    nll: float
    auc: object
    n: int
    oof: np.ndarray


def _apply_dropout(X, feature_columns, subjects, benchmarks, cfg, rng):
    if cfg is None or (cfg.subject_rate == 0.0 and cfg.benchmark_rate == 0.0):
        return X
    Xm, _ = apply_proxy_dropout(
        X, feature_columns, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=cfg.subject_rate, benchmark_rate=cfg.benchmark_rate)
    return Xm


def oof_predict(model_factory, ds: Dataset, manifest, *, dropout=None, rng=None) -> np.ndarray:
    """One OOF prediction per row from a model that never trained on that row's item."""
    if rng is None:
        rng = np.random.default_rng(0)
    item_arr = np.asarray(ds.item_keys).astype(str)
    oof = np.full(len(ds.y), np.nan, dtype=np.float64)
    for fold in outer_folds(manifest):
        assert_item_disjoint(fold.train_item_keys, fold.oof_item_keys)
        tr = np.isin(item_arr, np.asarray(fold.train_item_keys, dtype=str))
        te = np.isin(item_arr, np.asarray(fold.oof_item_keys, dtype=str))
        Xtr = _apply_dropout(ds.X[tr], ds.feature_columns, ds.subjects[tr], ds.benchmarks[tr], dropout, rng)
        Xte = _apply_dropout(ds.X[te], ds.feature_columns, ds.subjects[te], ds.benchmarks[te], dropout, rng)
        model = model_factory()
        model.fit(Xtr, ds.y[tr])
        oof[te] = np.asarray(model.predict(Xte), dtype=np.float64)
    if np.isnan(oof).any():
        raise AssertionError("every row must receive exactly one OOF prediction")
    return oof


def evaluate(model_factory, ds: Dataset, manifest, *, dropout=None, seed=0) -> EvalResult:
    rng = np.random.default_rng(seed)
    oof = oof_predict(model_factory, ds, manifest, dropout=dropout, rng=rng)
    return EvalResult(nll=log_loss(ds.y, oof), auc=auc_roc(ds.y, oof), n=len(ds.y), oof=oof)


def build_oof_meta(member_factories, X, y, item_keys, *, n_folds, seed, outer_index,
                   feature_columns=None, subjects=None, benchmarks=None,
                   dropout=None, rng=None) -> np.ndarray:
    """Nested-OOF member predictions over a set of train rows (the stacker's training
    meta-features). Each cell is produced by a member that did NOT train on that row's
    item — the recursion leakage guard.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    item_arr = np.asarray(item_keys).astype(str)
    meta = np.full((len(y), len(member_factories)), np.nan, dtype=np.float64)
    for ifold in inner_folds(item_keys, n_folds, seed, outer_index):
        assert_item_disjoint(ifold.train_item_keys, ifold.oof_item_keys)
        itr = np.isin(item_arr, np.asarray(ifold.train_item_keys, dtype=str))
        ite = np.isin(item_arr, np.asarray(ifold.oof_item_keys, dtype=str))
        for mi, mf in enumerate(member_factories):
            Xtr = X[itr]
            Xte = X[ite]
            if (dropout is not None and feature_columns is not None
                    and subjects is not None and benchmarks is not None):
                Xtr = _apply_dropout(Xtr, feature_columns, subjects[itr], benchmarks[itr], dropout, rng)
                Xte = _apply_dropout(Xte, feature_columns, subjects[ite], benchmarks[ite], dropout, rng)
            m = mf()
            m.fit(Xtr, y[itr])
            meta[ite, mi] = np.asarray(m.predict(Xte), dtype=np.float64)
    if np.isnan(meta).any():
        raise AssertionError("every train row must receive an OOF meta-prediction from every member")
    return meta


def recursive_evaluate(member_factories, stacker_factory, ds: Dataset, manifest, *,
                       dropout=None, seed=0) -> EvalResult:
    """Two-layer nested-OOF evaluation. Per outer fold: build OOF meta-features over the
    fold's train rows, fit the stacker on them, fit members on all train rows, predict
    the outer OOF rows, and stack. Concatenated final predictions -> NLL.
    """
    rng = np.random.default_rng(seed)
    item_arr = np.asarray(ds.item_keys).astype(str)
    final = np.full(len(ds.y), np.nan, dtype=np.float64)
    for fold in outer_folds(manifest):
        tr = np.isin(item_arr, np.asarray(fold.train_item_keys, dtype=str))
        te = np.isin(item_arr, np.asarray(fold.oof_item_keys, dtype=str))
        Xtr, ytr = ds.X[tr], ds.y[tr]
        meta_tr = build_oof_meta(
            member_factories, Xtr, ytr, item_arr[tr],
            n_folds=manifest.n_folds, seed=manifest.seed, outer_index=fold.index,
            feature_columns=ds.feature_columns, subjects=ds.subjects[tr],
            benchmarks=ds.benchmarks[tr], dropout=dropout, rng=rng)
        stacker = stacker_factory()
        stacker.fit(meta_tr, ytr)
        Xte = ds.X[te]
        cols = []
        for mf in member_factories:
            Xf = _apply_dropout(Xtr, ds.feature_columns, ds.subjects[tr], ds.benchmarks[tr], dropout, rng) \
                if dropout is not None else Xtr
            Xp = _apply_dropout(Xte, ds.feature_columns, ds.subjects[te], ds.benchmarks[te], dropout, rng) \
                if dropout is not None else Xte
            m = mf()
            m.fit(Xf, ytr)
            cols.append(np.asarray(m.predict(Xp), dtype=np.float64))
        meta_te = np.column_stack(cols)
        final[te] = np.asarray(stacker.predict(meta_te), dtype=np.float64)
    if np.isnan(final).any():
        raise AssertionError("every row must receive exactly one final prediction")
    return EvalResult(nll=log_loss(ds.y, final), auc=auc_roc(ds.y, final), n=len(ds.y), oof=final)
