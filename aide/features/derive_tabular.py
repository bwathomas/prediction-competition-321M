"""Tabular OOF target-encoding codec — ``grp_subj__*``/``grp_bench__*`` (metadata
groupby passrates), ``m2_subj*`` (shrunk subject passrates, two smoothings), and
``int__``/``ratio__`` interactions.

Per Plan 4 §D the heavy columnar groupby (polars/cuDF on Colab over the long ~millions-row
table) is the accelerator; the *definition* — leave-own-fold-out shrunk target encoding —
is implemented and tested here in numpy so OOF correctness is locked locally. The encoding
of a row in fold ``f`` aggregates only labels from rows NOT in fold ``f`` (and ignores
unobserved ``nan`` labels), so a row's own fold can never leak into its own feature.

``m2_subj*`` is subject_proxy; ``grp_subj__*`` subject_proxy; ``grp_bench__*``
benchmark_proxy; ``int__``/``ratio__`` are subject_proxy (built over subject channels).
"""
from __future__ import annotations

import numpy as np

from aide.harness.funnel import FeatureBlock


def target_encode_oof(keys, y, fold_ids, *, m: float, global_mean: float | None = None,
                      stat: str = "mean") -> np.ndarray:
    """Leave-own-fold-out shrunk target encoding (``stat="mean"``) or OOF group std.

    For each row ``i`` with key ``k`` in fold ``f``, aggregate ``y`` over all rows ``j``
    with ``key[j]==k``, ``fold[j]!=f`` and ``y[j]`` observed. ``mean`` returns
    ``(sum + m*global_mean) / (count + m)`` (so an unseen-in-other-folds key falls back to
    ``global_mean``); ``std`` returns the OOF group standard deviation (0 if <2 points).
    """
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    fold_ids = np.asarray(fold_ids)
    obs = np.isfinite(y)
    if global_mean is None:
        global_mean = float(y[obs].mean()) if obs.any() else 0.0
    # std fallback for a key unseen in other folds: the global std (a real prior), so an
    # unseen key is NOT conflated with a perfectly-consistent (zero-variance) one.
    global_std = float(y[obs].std()) if obs.sum() >= 2 else 0.0

    out = np.empty(len(keys), dtype=float)
    # Per (key, fold): observed sum/count/sumsq, so each row subtracts ITS fold's
    # contribution to get the other-folds aggregate in O(rows) rather than O(rows^2).
    from collections import defaultdict
    tot = defaultdict(lambda: [0.0, 0.0, 0.0])   # key -> [sum, count, sumsq] over all folds
    per = defaultdict(lambda: [0.0, 0.0, 0.0])   # (key, fold) -> same, this fold only
    for k, yy, f, o in zip(keys, y, fold_ids, obs):
        if not o:
            continue
        tot[k][0] += yy; tot[k][1] += 1; tot[k][2] += yy * yy
        pk = (k, f)
        per[pk][0] += yy; per[pk][1] += 1; per[pk][2] += yy * yy

    for i, (k, f) in enumerate(zip(keys, fold_ids)):
        s, c, sq = tot[k]
        ps, pc, psq = per[(k, f)]
        os_, oc, osq = s - ps, c - pc, sq - psq   # other-folds aggregate
        if stat == "std":
            if oc >= 2:
                var = max(osq / oc - (os_ / oc) ** 2, 0.0)
                out[i] = float(np.sqrt(var))
            else:
                out[i] = global_std   # unseen/singleton key → global prior, not "consistent"
        else:
            out[i] = (os_ + m * global_mean) / (oc + m) if (oc + m) > 0 else global_mean
    return out


def _block(columns, cols_data, row_ids) -> FeatureBlock:
    if columns:
        X = np.column_stack([np.asarray(c, dtype=np.float32) for c in cols_data])
    else:
        X = np.zeros((len(row_ids), 0), dtype=np.float32)
    return FeatureBlock(X=np.asarray(X, dtype=np.float32), columns=list(columns),
                        row_ids=np.asarray(row_ids).astype(str))


def derive_tabular(*, row_ids, fold_ids, y, subject_keys,
                   subject_meta: dict, benchmark_meta: dict,
                   parents: dict | None = None, smoothings=(2.0, 20.0)):
    """Build the OOF tabular feature blocks. ``subject_meta``/``benchmark_meta`` map a
    metadata field name to a per-row key array; ``parents`` supplies already-derived
    columns used by the cheap ``int__``/``ratio__`` arithmetic."""
    fold_ids = np.asarray(fold_ids)
    y = np.asarray(y, dtype=float)
    m_low, m_high = sorted(smoothings)

    # ---- metadata groupby encodings (mean + std), OOF -------------------------------
    grp_subj_cols, grp_subj_data = [], []
    for field, keys in subject_meta.items():
        grp_subj_cols.append(f"grp_subj__{field}_passrate_mean")
        grp_subj_data.append(target_encode_oof(keys, y, fold_ids, m=m_high))
        grp_subj_cols.append(f"grp_subj__{field}_passrate_std")
        grp_subj_data.append(target_encode_oof(keys, y, fold_ids, m=0.0, stat="std"))

    grp_bench_cols, grp_bench_data = [], []
    for field, keys in benchmark_meta.items():
        grp_bench_cols.append(f"grp_bench__{field}_passrate_mean")
        grp_bench_data.append(target_encode_oof(keys, y, fold_ids, m=m_high))

    # ---- subject mean-encoding at two smoothings (ensemble diversity) ---------------
    m2_cols = ["m2_subj_mean_mlow", "m2_subj_mean_mhigh"]
    m2_data = [target_encode_oof(subject_keys, y, fold_ids, m=m_low),
               target_encode_oof(subject_keys, y, fold_ids, m=m_high)]

    # ---- cheap interactions over already-derived parent columns ---------------------
    # Each interaction names its required parents; a PARTIAL match (one parent present, its
    # partner missing) is almost always a wiring typo, so raise instead of silently
    # emitting an empty interactions block that only surfaces as a Colab coverage shortfall.
    int_specs = {
        "int__subjectmean_x_clusterdiff": (
            ("subject_mean", "cluster_difficulty"),
            lambda p: np.asarray(p["subject_mean"], float) * np.asarray(p["cluster_difficulty"], float)),
        "ratio__coverage_over_lid": (
            ("nn__coverage_K8", "geo__lid_estimate"),
            lambda p: np.asarray(p["nn__coverage_K8"], float)
            / np.where(np.abs(np.asarray(p["geo__lid_estimate"], float)) < 1e-9, 1.0,
                       np.asarray(p["geo__lid_estimate"], float))),
    }
    int_cols, int_data = [], []
    if parents:
        for col, (needed, fn) in int_specs.items():
            present = [k for k in needed if k in parents]
            if len(present) == len(needed):
                int_cols.append(col)
                int_data.append(fn(parents))
            elif present:  # some-but-not-all → wiring error
                missing = [k for k in needed if k not in parents]
                raise ValueError(
                    f"interaction {col!r} has parents {present} but is missing {missing}; "
                    f"check the parent-column wiring (cluster difficulty is 'm2_cluster_mean')")

    out = {
        "groupby_subject_metadata": _block(grp_subj_cols, grp_subj_data, row_ids),
        "groupby_benchmark_metadata": _block(grp_bench_cols, grp_bench_data, row_ids),
        "mean_encoded_subject": _block(m2_cols, m2_data, row_ids),
        "interactions_subject": _block(int_cols, int_data, row_ids),
    }
    return out
