"""Evaluation harness: the only component that owns folds, recursion, dropout, and the
leakage probes. A dropped-in model (``fit(X, y)`` / ``predict(X) -> prob``) therefore
cannot see held-out items or undropped identity proxies — hygiene is enforced here, not
trusted to the model.

- ``oof_predict`` / ``evaluate``: layer-1, single 3-fold item-uniform OOF.
- ``build_oof_meta`` + ``recursive_evaluate``: layer-2, nested (recursive) OOF so the
  stacker never trains on a member's in-sample predictions.

Dropout discipline (the subtle part): within a fold the dropped subject/benchmark SET is
chosen ONCE from a fold-deterministic rng and applied to BOTH the train and the held-out
matrices. Choosing per-call from an advancing rng would drop different entities in train
vs test — train/serve skew and a scoring-time identity leak. Every masked matrix is then
checked by ``assert_no_proxy_leak`` on exactly its dropped rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hygiene.dropout import choose_dropped, mask_dropped
from ..hygiene.probes import (
    assert_item_disjoint, assert_no_proxy_leak, assert_columns_covered)
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


def _enabled(cfg) -> bool:
    return cfg is not None and (cfg.subject_rate > 0.0 or cfg.benchmark_rate > 0.0)


def _dropout_fold(Xtr, Xte, cols, subj_tr, bench_tr, subj_te, bench_te, cfg, seed_seq):
    """Choose ONE dropped set for the fold (deterministic in seed_seq) and apply it to
    both matrices, probing each. Returns (Xtr_masked, Xte_masked)."""
    if not _enabled(cfg):
        return np.asarray(Xtr, dtype=np.float32), np.asarray(Xte, dtype=np.float32)
    rng = np.random.default_rng(list(seed_seq))
    su = np.concatenate([np.asarray(subj_tr).astype(str), np.asarray(subj_te).astype(str)])
    bu = np.concatenate([np.asarray(bench_tr).astype(str), np.asarray(bench_te).astype(str)])
    dropped_subj, dropped_bench = choose_dropped(
        subjects=su, benchmarks=bu,
        subject_rate=cfg.subject_rate, benchmark_rate=cfg.benchmark_rate, rng=rng)
    out = []
    for X, subj, bench in [(Xtr, subj_tr, bench_tr), (Xte, subj_te, bench_te)]:
        Xm, info = mask_dropped(X, cols, subjects=subj, benchmarks=bench,
                                dropped_subjects=dropped_subj, dropped_benchmarks=dropped_bench)
        assert_no_proxy_leak(Xm, cols, ["subject"], rows=info["drop_rows"]["subject"])
        assert_no_proxy_leak(Xm, cols, ["benchmark"], rows=info["drop_rows"]["benchmark"])
        out.append(Xm)
    return out[0], out[1]


def _maybe_check_columns(ds, neutral_prefixes):
    if neutral_prefixes is not None:
        assert_columns_covered(ds.feature_columns, neutral_prefixes=neutral_prefixes)


def oof_predict(model_factory, ds: Dataset, manifest, *, dropout=None, seed=0,
                neutral_prefixes=None) -> np.ndarray:
    """One OOF prediction per row from a model that never trained on that row's item."""
    _maybe_check_columns(ds, neutral_prefixes)
    item_arr = np.asarray(ds.item_keys).astype(str)
    oof = np.full(len(ds.y), np.nan, dtype=np.float64)
    for fold in outer_folds(manifest):
        assert_item_disjoint(fold.train_item_keys, fold.oof_item_keys)
        tr = np.isin(item_arr, np.asarray(fold.train_item_keys, dtype=str))
        te = np.isin(item_arr, np.asarray(fold.oof_item_keys, dtype=str))
        Xtr, Xte = _dropout_fold(
            ds.X[tr], ds.X[te], ds.feature_columns,
            ds.subjects[tr], ds.benchmarks[tr], ds.subjects[te], ds.benchmarks[te],
            dropout, seed_seq=[seed, fold.index])
        model = model_factory()
        model.fit(Xtr, ds.y[tr])
        oof[te] = np.asarray(model.predict(Xte), dtype=np.float64)
    if np.isnan(oof).any():
        raise AssertionError("every row must receive exactly one OOF prediction")
    return oof


def evaluate(model_factory, ds: Dataset, manifest, *, dropout=None, seed=0,
             neutral_prefixes=None) -> EvalResult:
    oof = oof_predict(model_factory, ds, manifest, dropout=dropout, seed=seed,
                      neutral_prefixes=neutral_prefixes)
    return EvalResult(nll=log_loss(ds.y, oof), auc=auc_roc(ds.y, oof), n=len(ds.y), oof=oof)


def build_oof_meta(member_factories, X, y, item_keys, *, n_folds, seed, outer_index,
                   feature_columns=None, subjects=None, benchmarks=None,
                   dropout=None) -> np.ndarray:
    """Nested-OOF member predictions over a set of train rows (the stacker's training
    meta-features). Each cell is produced by a member that did NOT train on that row's
    item — the recursion leakage guard."""
    item_arr = np.asarray(item_keys).astype(str)
    meta = np.full((len(y), len(member_factories)), np.nan, dtype=np.float64)
    for ifold in inner_folds(item_keys, n_folds, seed, outer_index):
        assert_item_disjoint(ifold.train_item_keys, ifold.oof_item_keys)
        itr = np.isin(item_arr, np.asarray(ifold.train_item_keys, dtype=str))
        ite = np.isin(item_arr, np.asarray(ifold.oof_item_keys, dtype=str))
        if (_enabled(dropout) and feature_columns is not None
                and subjects is not None and benchmarks is not None):
            Xtr, Xte = _dropout_fold(
                X[itr], X[ite], feature_columns,
                subjects[itr], benchmarks[itr], subjects[ite], benchmarks[ite],
                dropout, seed_seq=[seed, outer_index, ifold.index])
        else:
            Xtr, Xte = X[itr], X[ite]
        for mi, mf in enumerate(member_factories):
            m = mf()
            m.fit(Xtr, y[itr])
            meta[ite, mi] = np.asarray(m.predict(Xte), dtype=np.float64)
    if np.isnan(meta).any():
        raise AssertionError("every train row must receive an OOF meta-prediction from every member")
    return meta


def recursive_evaluate(member_factories, stacker_factory, ds: Dataset, manifest, *,
                       dropout=None, seed=0, neutral_prefixes=None) -> EvalResult:
    """Two-layer nested-OOF evaluation. Per outer fold: build OOF meta-features over the
    fold's train rows, fit the stacker on them, fit members on all train rows, predict
    the outer OOF rows, and stack. Concatenated final predictions -> NLL."""
    _maybe_check_columns(ds, neutral_prefixes)
    item_arr = np.asarray(ds.item_keys).astype(str)
    final = np.full(len(ds.y), np.nan, dtype=np.float64)
    for fold in outer_folds(manifest):
        assert_item_disjoint(fold.train_item_keys, fold.oof_item_keys)
        tr = np.isin(item_arr, np.asarray(fold.train_item_keys, dtype=str))
        te = np.isin(item_arr, np.asarray(fold.oof_item_keys, dtype=str))
        Xtr, ytr = ds.X[tr], ds.y[tr]
        meta_tr = build_oof_meta(
            member_factories, Xtr, ytr, item_arr[tr],
            n_folds=manifest.n_folds, seed=manifest.seed, outer_index=fold.index,
            feature_columns=ds.feature_columns, subjects=ds.subjects[tr],
            benchmarks=ds.benchmarks[tr], dropout=dropout)
        stacker = stacker_factory()
        stacker.fit(meta_tr, ytr)
        # members trained on ALL train rows predict the outer OOF rows; one dropped set
        # for both the member-fit and the member-predict matrices (same C1 discipline)
        Xtr_m, Xte_m = _dropout_fold(
            Xtr, ds.X[te], ds.feature_columns,
            ds.subjects[tr], ds.benchmarks[tr], ds.subjects[te], ds.benchmarks[te],
            dropout, seed_seq=[seed, fold.index, -1])
        cols = []
        for mf in member_factories:
            m = mf()
            m.fit(Xtr_m, ytr)
            cols.append(np.asarray(m.predict(Xte_m), dtype=np.float64))
        final[te] = np.asarray(stacker.predict(np.column_stack(cols)), dtype=np.float64)
    if np.isnan(final).any():
        raise AssertionError("every row must receive exactly one final prediction")
    return EvalResult(nll=log_loss(ds.y, final), auc=auc_roc(ds.y, final), n=len(ds.y), oof=final)
