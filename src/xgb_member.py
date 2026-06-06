"""Package-free XGBoost member: trained with XGBoost offline, traversed
with pure numpy at runtime (no ``import xgboost`` in ``model.py``).

Mirror of :mod:`src.gbdt_member` (LightGBM) but for XGBoost's tree
semantics, which differ in three ways that are easy to get wrong:

1. **Strict ``<`` split.** XGBoost routes ``x < split_condition`` to the
   ``yes`` child and ``x >= split_condition`` (and exactly-equal) to the
   ``no`` child. LightGBM uses ``x <= threshold`` -> left. We bake the
   strict-less comparison into this module's walker.

2. **Explicit ``missing`` child.** Each internal node stores ``yes``,
   ``no`` and ``missing`` node ids. ``missing`` is whichever of ``yes`` /
   ``no`` NaN inputs follow. We compile ``default_left = (missing == yes)``
   so the shared NaN-routes-to-``default_left`` walker reproduces it.

3. **Non-sequential node ids.** The JSON dump's ``nodeid`` values are not
   guaranteed to be a dense ``0..n-1`` DFS order, so we build an explicit
   ``nodeid -> compiled index`` map before wiring children.

Bias / base_score is recovered the same way as the LightGBM member:
compare ``booster.predict(X, output_margin=True)`` (sum of leaf margins +
base_margin) to the numpy sum-of-leaves on a set of anchor rows and store
the constant difference as ``bias``. Runtime output is
``sigmoid(sum_leaves + bias)``.

Parity discipline: :func:`fit_xgb_member` verifies the numpy walker
reproduces the XGBoost raw margin to ``< parity_atol`` and the probability
to ``< parity_atol`` on a held-out batch BEFORE returning, and raises
otherwise.

Runtime contract (identical signatures to gbdt_member):
``apply_one(state, feats) -> float``  (clamped to (eps, 1-eps))
``apply_batch(state, feats_matrix) -> np.ndarray[N] float32``
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("xgb_member")

_EPS = 1.0e-6
_LEAF_FEATURE_SENTINEL: int = -1


# ---------------------------------------------------------------------------
# State (the shipped artifact) -- same flat-array layout as gbdt_member
# ---------------------------------------------------------------------------


@dataclass
class XGBMemberState:
    """Fitted-and-shipped state of an XGBoost member.

    Layout is identical to :class:`src.gbdt_member.GBDTMemberState`:
    concatenated per-tree arrays plus ``tree_offsets``; child indices are
    GLOBAL (shifted by the tree offset at compile time). The only runtime
    difference from the LightGBM member is the split comparison: this
    walker uses strict ``<`` (XGBoost semantics) instead of ``<=``.
    """

    feature_concat: np.ndarray        # int32 [total_nodes], -1 = leaf
    threshold_concat: np.ndarray      # float64 [total_nodes], leaf margin when leaf
    left_concat: np.ndarray           # int32 [total_nodes] (the ``yes`` child)
    right_concat: np.ndarray          # int32 [total_nodes] (the ``no`` child)
    default_left_concat: np.ndarray   # bool [total_nodes], True if NaN -> yes
    tree_offsets: np.ndarray          # int32 [n_trees + 1]

    feature_dim: int
    feature_names: tuple[str, ...]
    bias: float                       # base_margin (logit(base_score))
    fit_method: str                   # "xgboost"
    n_train: int
    n_pos: int
    n_trees: int
    train_loss: float
    val_loss: float
    objective: str = "binary"
    output_mode: str = "probability"  # sigmoid(sum_leaves + bias)

    def __post_init__(self) -> None:
        n = int(self.feature_concat.shape[0])
        for arr_name in (
            "threshold_concat",
            "left_concat",
            "right_concat",
            "default_left_concat",
        ):
            arr = getattr(self, arr_name)
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"XGBMemberState: {arr_name} len {arr.shape[0]} != "
                    f"feature_concat len {n}"
                )
        if int(self.tree_offsets.shape[0]) != int(self.n_trees) + 1:
            raise ValueError(
                f"tree_offsets len {self.tree_offsets.shape[0]} != "
                f"n_trees+1 {int(self.n_trees) + 1}"
            )
        if int(self.tree_offsets[-1]) != n:
            raise ValueError(
                f"tree_offsets[-1] {int(self.tree_offsets[-1])} != total nodes {n}"
            )
        if int(len(self.feature_names)) != int(self.feature_dim):
            raise ValueError(
                f"feature_names len {len(self.feature_names)} != "
                f"feature_dim {self.feature_dim}"
            )
        if not math.isfinite(float(self.bias)):
            raise ValueError("XGBMemberState: bias is NaN/Inf")

    @property
    def total_nodes(self) -> int:
        return int(self.feature_concat.shape[0])

    # ---- I/O ----

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "trees.npz",
            feature_concat=self.feature_concat.astype(np.int32),
            threshold_concat=self.threshold_concat.astype(np.float64),
            left_concat=self.left_concat.astype(np.int32),
            right_concat=self.right_concat.astype(np.int32),
            default_left_concat=self.default_left_concat.astype(np.bool_),
            tree_offsets=self.tree_offsets.astype(np.int32),
        )
        meta = {
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "bias": float(self.bias),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "n_trees": int(self.n_trees),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "objective": str(self.objective),
            "output_mode": str(self.output_mode),
            "split_op": "lt",  # strict less-than (XGBoost)
            "format_version": 1,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "XGBMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "trees.npz") as npz:
            feature = npz["feature_concat"].astype(np.int32, copy=False)
            threshold = npz["threshold_concat"].astype(np.float64, copy=False)
            left = npz["left_concat"].astype(np.int32, copy=False)
            right = npz["right_concat"].astype(np.int32, copy=False)
            default_left = npz["default_left_concat"].astype(np.bool_, copy=False)
            tree_offsets = npz["tree_offsets"].astype(np.int32, copy=False)
        return cls(
            feature_concat=feature,
            threshold_concat=threshold,
            left_concat=left,
            right_concat=right,
            default_left_concat=default_left,
            tree_offsets=tree_offsets,
            feature_dim=int(meta["feature_dim"]),
            feature_names=tuple(meta["feature_names"]),
            bias=float(meta["bias"]),
            fit_method=str(meta.get("fit_method", "xgboost")),
            n_train=int(meta.get("n_train", 0)),
            n_pos=int(meta.get("n_pos", 0)),
            n_trees=int(meta.get("n_trees", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
            objective=str(meta.get("objective", "binary")),
            output_mode=str(meta.get("output_mode", "probability")),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference (strict ``<`` split)
# ---------------------------------------------------------------------------


def _sigmoid_stable_one(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _traverse_one_tree(state: XGBMemberState, tree_idx: int, features: np.ndarray) -> float:
    start = int(state.tree_offsets[tree_idx])
    end = int(state.tree_offsets[tree_idx + 1])
    feat = state.feature_concat
    thr = state.threshold_concat
    left = state.left_concat
    right = state.right_concat
    dleft = state.default_left_concat

    node = start
    for _ in range(end - start + 1):
        f = int(feat[node])
        if f == _LEAF_FEATURE_SENTINEL:
            return float(thr[node])
        v = features[f] if 0 <= f < int(features.shape[0]) else float("nan")
        if not np.isfinite(v):
            node = int(left[node]) if bool(dleft[node]) else int(right[node])
        elif float(v) < float(thr[node]):   # XGBoost: strict less -> yes (left)
            node = int(left[node])
        else:
            node = int(right[node])
    LOG.warning("XGB traversal bound hit on tree %d; returning 0.0", tree_idx)
    return 0.0


def apply_one(state: XGBMemberState, features: np.ndarray) -> float:
    if state.output_mode != "probability":
        raise RuntimeError(f"apply_one invalid for output_mode={state.output_mode!r}")
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim {state.feature_dim}"
        )
    raw = float(state.bias)
    for t in range(int(state.n_trees)):
        raw += _traverse_one_tree(state, t, features)
    if not math.isfinite(raw):
        return 0.5
    p = _sigmoid_stable_one(raw)
    return float(min(max(p, _EPS), 1.0 - _EPS))


def _walk_tree_batch(state: XGBMemberState, tree_idx: int, fm: np.ndarray) -> np.ndarray:
    B = int(fm.shape[0])
    start = int(state.tree_offsets[tree_idx])
    feat = state.feature_concat
    thr = state.threshold_concat
    left = state.left_concat
    right = state.right_concat
    dleft = state.default_left_concat

    node = np.full(B, start, dtype=np.int64)
    n_tree_nodes = int(state.tree_offsets[tree_idx + 1]) - start
    feat_dim = int(fm.shape[1])
    for _ in range(int(n_tree_nodes) + 1):
        f_idx = feat[node]
        is_leaf = f_idx == _LEAF_FEATURE_SENTINEL
        if bool(is_leaf.all()):
            break
        nl_rows = np.where(~is_leaf)[0]
        nl_node = node[nl_rows]
        fi = f_idx[nl_rows].astype(np.int64, copy=False)
        valid_f = (fi >= 0) & (fi < feat_dim)
        fi_safe = np.where(valid_f, fi, 0)
        fv = fm[nl_rows, fi_safe]
        if not bool(valid_f.all()):
            fv = np.where(valid_f, fv, np.nan)
        finite = np.isfinite(fv)
        th = thr[nl_node]
        go_left = ((fv < th) & finite) | (dleft[nl_node] & ~finite)  # strict <
        node[nl_rows] = np.where(go_left, left[nl_node], right[nl_node]).astype(
            np.int64, copy=False
        )
    return thr[node].astype(np.float64, copy=False)


def predict_raw(state: XGBMemberState, features_matrix: np.ndarray) -> np.ndarray:
    if features_matrix.ndim != 2:
        raise ValueError("features_matrix must be 2D")
    if int(features_matrix.shape[1]) != int(state.feature_dim):
        raise ValueError(
            f"features_matrix dim {features_matrix.shape[1]} != "
            f"state.feature_dim {state.feature_dim}"
        )
    N = int(features_matrix.shape[0])
    if N == 0:
        return np.empty(0, dtype=np.float64)
    fm = np.ascontiguousarray(features_matrix, dtype=np.float64)
    raw = np.full(N, float(state.bias), dtype=np.float64)
    for t in range(int(state.n_trees)):
        raw += _walk_tree_batch(state, t, fm)
    return raw


def apply_batch(state: XGBMemberState, features_matrix: np.ndarray) -> np.ndarray:
    if state.output_mode != "probability":
        raise RuntimeError(f"apply_batch invalid for output_mode={state.output_mode!r}")
    raw = predict_raw(state, features_matrix)
    if raw.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    out = np.empty_like(raw)
    pos = raw >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
    e = np.exp(raw[~pos])
    out[~pos] = e / (1.0 + e)
    return np.clip(out, _EPS, 1.0 - _EPS).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Tree compilation: XGBoost JSON dump -> flat arrays
# ---------------------------------------------------------------------------


def _parse_feature_index(split: Any) -> int:
    """``split`` is a feature name like ``"f3"`` (default DMatrix names) or
    a bare integer. Return the integer feature index."""
    if isinstance(split, (int, np.integer)):
        return int(split)
    s = str(split)
    if s.startswith("f") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    raise ValueError(
        f"Cannot parse XGBoost split feature {split!r}. Train the booster "
        "with default feature names (f0..fN) so splits are 'f<idx>'."
    )


def _compile_xgb_tree(nodes: Sequence[Mapping[str, Any]]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Compile one XGBoost tree (list of node dicts from the JSON dump) to
    flat local arrays. Children indices are LOCAL (within the tree)."""
    # Map raw nodeid -> dense compiled index, ordered by first appearance.
    by_id: dict[int, Mapping[str, Any]] = {}
    for nd in nodes:
        by_id[int(nd["nodeid"])] = nd
    # Deterministic order: sort by nodeid so compilation is reproducible.
    ordered_ids = sorted(by_id.keys())
    id_to_idx = {nid: i for i, nid in enumerate(ordered_ids)}
    n = len(ordered_ids)

    feature = np.full(n, _LEAF_FEATURE_SENTINEL, dtype=np.int32)
    threshold = np.zeros(n, dtype=np.float64)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    default_left = np.zeros(n, dtype=np.bool_)

    for nid in ordered_ids:
        idx = id_to_idx[nid]
        nd = by_id[nid]
        if "leaf" in nd:
            threshold[idx] = float(nd["leaf"])
            continue
        feature[idx] = _parse_feature_index(nd["split"])
        threshold[idx] = float(nd["split_condition"])
        yes_id = int(nd["yes"])
        no_id = int(nd["no"])
        miss_id = int(nd["missing"])
        left[idx] = id_to_idx[yes_id]    # ``yes`` == x < thr branch
        right[idx] = id_to_idx[no_id]
        default_left[idx] = (miss_id == yes_id)
    return feature, threshold, left, right, default_left


def _flatten_xgb_nodes(tree_struct: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``dump_format='json'`` tree is a nested dict; flatten it to a
    list of node dicts each carrying ``nodeid`` and (for internal nodes)
    ``split``/``split_condition``/``yes``/``no``/``missing``, or ``leaf``."""
    out: list[dict[str, Any]] = []

    def _walk(node: Mapping[str, Any]) -> None:
        if "leaf" in node:
            out.append({"nodeid": int(node["nodeid"]), "leaf": float(node["leaf"])})
            return
        out.append(
            {
                "nodeid": int(node["nodeid"]),
                "split": node["split"],
                "split_condition": float(node["split_condition"]),
                "yes": int(node["yes"]),
                "no": int(node["no"]),
                "missing": int(node["missing"]),
            }
        )
        for child in node.get("children", []):
            _walk(child)

    _walk(tree_struct)
    return out


def _concat_trees(per_tree: list[tuple[np.ndarray, ...]]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    feats, thrs, lefts, rights, dlefts = [], [], [], [], []
    offsets = [0]
    running = 0
    for feature, threshold, left, right, default_left in per_tree:
        n = int(feature.shape[0])
        feats.append(feature.astype(np.int32, copy=False))
        thrs.append(threshold.astype(np.float64, copy=False))
        lefts.append(np.where(left >= 0, left + running, -1).astype(np.int32))
        rights.append(np.where(right >= 0, right + running, -1).astype(np.int32))
        dlefts.append(default_left.astype(np.bool_, copy=False))
        running += n
        offsets.append(running)
    return (
        np.concatenate(feats),
        np.concatenate(thrs),
        np.concatenate(lefts),
        np.concatenate(rights),
        np.concatenate(dlefts),
        np.asarray(offsets, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Offline trainer + fail-fast parity
# ---------------------------------------------------------------------------


def compile_booster(
    booster: Any,
    *,
    feature_names: Sequence[str],
    feature_dim: int,
    anchor_X: np.ndarray,
    parity_atol: float = 1.0e-5,
    objective: str = "binary",
    n_train: int = 0,
    n_pos: int = 0,
) -> XGBMemberState:
    """Compile an already-trained XGBoost ``Booster`` to a numpy state and
    verify parity vs ``booster.predict`` on ``anchor_X``.

    Separated from :func:`fit_xgb_member` so a model trained elsewhere
    (e.g. a recovered AIDE solution) can be exported without retraining.
    """
    import xgboost as xgb  # offline only

    # With early stopping, booster.predict() defaults to best_iteration, but
    # get_dump() returns ALL boosted rounds. Compile and score the SAME tree
    # count (best_iteration+1 if set, else all) so the numpy walker matches
    # the package exactly. binary:logistic emits one tree per round.
    n_total = int(booster.num_boosted_rounds())
    best = getattr(booster, "best_iteration", None)
    try:
        best = int(best)
    except (TypeError, ValueError):
        best = None
    ntree = (best + 1) if (best is not None and 0 <= best < n_total) else n_total

    dump = booster.get_dump(dump_format="json", with_stats=False)[:ntree]
    per_tree: list[tuple[np.ndarray, ...]] = []
    for tree_json in dump:
        tree_struct = json.loads(tree_json)
        nodes = _flatten_xgb_nodes(tree_struct)
        per_tree.append(_compile_xgb_tree(nodes))
    feat, thr, l, r, dl, offsets = _concat_trees(per_tree)

    # Bias recovery: margin = sum_leaves + base_margin (constant).
    anchor = np.ascontiguousarray(anchor_X, dtype=np.float64)
    tmp = XGBMemberState(
        feature_concat=feat, threshold_concat=thr, left_concat=l, right_concat=r,
        default_left_concat=dl, tree_offsets=offsets,
        feature_dim=int(feature_dim), feature_names=tuple(str(s) for s in feature_names),
        bias=0.0, fit_method="xgboost", n_train=int(n_train), n_pos=int(n_pos),
        n_trees=len(per_tree), train_loss=0.0, val_loss=0.0,
        objective=objective, output_mode="probability",
    )
    sum_leaves = predict_raw(tmp, anchor)
    dmat = xgb.DMatrix(anchor, feature_names=[f"f{i}" for i in range(int(feature_dim))])
    margin = np.asarray(
        booster.predict(dmat, output_margin=True, iteration_range=(0, ntree)),
        dtype=np.float64,
    ).reshape(-1)
    delta = margin - sum_leaves
    bias = float(delta.mean())
    bias_std = float(delta.std())
    if bias_std > 1.0e-4:
        raise RuntimeError(
            f"XGB bias not constant across anchors: mean={bias} std={bias_std}. "
            "Check that the dump parsed all trees and feature indices correctly."
        )

    state = XGBMemberState(
        feature_concat=feat, threshold_concat=thr, left_concat=l, right_concat=r,
        default_left_concat=dl, tree_offsets=offsets,
        feature_dim=int(feature_dim), feature_names=tuple(str(s) for s in feature_names),
        bias=bias, fit_method="xgboost", n_train=int(n_train), n_pos=int(n_pos),
        n_trees=len(per_tree), train_loss=0.0, val_loss=0.0,
        objective=objective, output_mode="probability",
    )

    # ---- Parity (FAIL-FAST) ----
    raw_np = predict_raw(state, anchor)
    max_abs_raw = float(np.max(np.abs(raw_np - margin)))
    p_np = apply_batch(state, anchor)
    p_xgb = np.asarray(
        booster.predict(dmat, iteration_range=(0, ntree)), dtype=np.float64
    ).reshape(-1)
    max_abs_prob = float(np.max(np.abs(p_np - p_xgb)))
    if max_abs_raw > float(parity_atol):
        raise RuntimeError(
            f"XGB raw-margin parity failed: max abs error {max_abs_raw} > {parity_atol}."
        )
    if max_abs_prob > float(parity_atol):
        raise RuntimeError(
            f"XGB probability parity failed: max abs error {max_abs_prob} > {parity_atol}."
        )
    LOG.info(
        "XGB compile OK: n_trees=%d total_nodes=%d feature_dim=%d "
        "parity_raw=%.2e parity_prob=%.2e bias=%.4f",
        state.n_trees, state.total_nodes, state.feature_dim,
        max_abs_raw, max_abs_prob, state.bias,
    )
    return state


def fit_xgb_member(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    sample_weights: np.ndarray | None = None,
    val_fraction: float = 0.1,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 25,
    seed: int = 0,
    parity_atol: float = 1.0e-5,
    num_threads: int | None = None,
) -> XGBMemberState:
    """Train an XGBoost ``binary:logistic`` booster on (X, y in [0,1]) and
    compile to a parity-verified numpy state. Mirrors
    :func:`src.gbdt_member.fit_gbdt_member`."""
    import xgboost as xgb  # offline only

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.shape != (int(X.shape[0]),):
        raise ValueError(f"y shape {y.shape} != ({X.shape[0]},)")
    if int(len(feature_names)) != int(X.shape[1]):
        raise ValueError(
            f"feature_names len {len(feature_names)} != X cols {X.shape[1]}"
        )

    rng = np.random.default_rng(int(seed))
    N = int(X.shape[0])
    perm = rng.permutation(N)
    n_val = max(64, int(round(val_fraction * N)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    fdim = int(X.shape[1])
    fnames = [f"f{i}" for i in range(fdim)]
    Xtr = np.ascontiguousarray(X[train_idx], dtype=np.float32)
    Xva = np.ascontiguousarray(X[val_idx], dtype=np.float32)
    dtrain = xgb.DMatrix(
        Xtr, label=y[train_idx], feature_names=fnames,
        weight=None if sample_weights is None else sample_weights[train_idx],
    )
    dval = xgb.DMatrix(
        Xva, label=y[val_idx], feature_names=fnames,
        weight=None if sample_weights is None else sample_weights[val_idx],
    )
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": float(learning_rate),
        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "lambda": float(reg_lambda),
        "seed": int(seed),
        "verbosity": 0,
    }
    if num_threads is not None:
        params["nthread"] = int(num_threads)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(n_estimators),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )

    state = compile_booster(
        booster,
        feature_names=feature_names,
        feature_dim=fdim,
        anchor_X=X[rng.choice(N, size=min(256, N), replace=False)],
        parity_atol=parity_atol,
        objective="binary",
        n_train=N,
        n_pos=int(np.sum(y == 1.0)),
    )

    # Honest manual cross-entropy (soft labels) on val for reporting.
    eps = 1.0e-6
    p_val = np.clip(apply_batch(state, X[val_idx].astype(np.float64)), eps, 1.0 - eps)
    yv = y[val_idx].astype(np.float64)
    state.val_loss = float(-np.mean(yv * np.log(p_val) + (1.0 - yv) * np.log(1.0 - p_val)))
    p_tr = np.clip(apply_batch(state, X[train_idx].astype(np.float64)), eps, 1.0 - eps)
    yt = y[train_idx].astype(np.float64)
    state.train_loss = float(-np.mean(yt * np.log(p_tr) + (1.0 - yt) * np.log(1.0 - p_tr)))
    return state


__all__ = [
    "XGBMemberState",
    "apply_one",
    "apply_batch",
    "predict_raw",
    "compile_booster",
    "fit_xgb_member",
]
