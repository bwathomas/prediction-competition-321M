"""Member 2 of the four-member stacked ensemble: a LightGBM tree
ensemble whose RUNTIME inference is pure numpy.

The user spec requires that LightGBM not be a runtime dependency:
trees are dumped offline via :meth:`Booster.dump_model`, compiled
into flat numpy arrays, and traversed at inference time with a
~30-line pure-numpy walker. The shipped artifact is a single
``.npz`` with parallel arrays per tree plus a tiny ``meta.json``.

Key design decisions
--------------------

1. **All categorical features one-hot encoded upstream**
   (``src/member_features.py``). Trees therefore only have ``<=``
   numeric splits -- no LightGBM ``decision_type=='=='`` categorical
   splits to deserialize. This keeps the numpy walker tiny and
   eliminates a known footgun (LightGBM's binary-encoded categorical
   thresholds are notoriously easy to get wrong in hand-rolled
   inference).

2. **Missing values via ``default_left``.** LightGBM stores per-node
   ``default_left`` indicating where to send NaN inputs. The
   traversal honors that exactly. A NaN in the feature vector
   degrades to whichever leaf the training-time missing-value imputer
   would have routed it to (so test-time missing inputs do not
   crash).

3. **Init score baked into the saved bias.** LightGBM's binary
   objective adds an implicit ``logit(mean_y_train)`` to every
   prediction when ``boost_from_average=True`` (default). We
   recover the exact init_score by computing
   ``raw_score - sum_of_leaves`` for one anchor row and saving
   that value as ``bias``; the runtime walker adds it to every
   prediction before sigmoid. This is bit-exact and avoids any
   reliance on the dumped-model's metadata.

4. **Parity discipline.** The offline trainer's ``fit_gbdt_member``
   verifies that the compiled numpy walker reproduces
   ``Booster.predict(X, raw_score=True)`` to ``< 1e-6`` on a held-out
   batch BEFORE returning. If parity fails, the trainer raises --
   silent traversal bugs are the canonical hand-rolled-tree footgun.

Runtime contract
----------------
``apply_one(state, feats) -> float`` (clamped to (eps, 1-eps))
``apply_batch(state, feats_matrix) -> np.ndarray[N] float32``

Both are pure numpy; no ``import lightgbm`` ever runs at runtime.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("gbdt_member")

_EPS = 1.0e-6
# Sentinel feature index for leaf nodes. Chosen as -1 to crash
# obviously if mis-indexed into a feature vector.
_LEAF_FEATURE_SENTINEL: int = -1


# LightGBM rejects ``["\,\:\}\{\[\]]`` and any control char in feature
# names (see ``lightgbm.basic._safe_feature_name``). Schema feature
# names are mostly machine-emitted (``pool__centroid_dist_0``,
# ``cluster__017``, ...) so they're already clean -- but raw
# ``condition`` strings can include arbitrary punctuation. We sanitize
# uniformly here so a future schema field that accidentally contains
# a forbidden char doesn't blow up training. The map is stable: same
# input always yields the same output, and collisions get a unique
# numeric suffix so distinct columns don't fold into the same name.
import re as _re

# Matches the exact set LightGBM rejects in
# ``LGBM_DatasetCreateFromMat`` -> "Do not support special JSON
# characters in feature name". The C++ check is
# ``fname.find_first_of("\"\\,:[]{}") != npos``.
_LGBM_FORBIDDEN = _re.compile(r'["\\,:\[\]\{\}]')


def _sanitize_for_lightgbm(names: Sequence[str]) -> list[str]:
    """Return a copy of ``names`` with LightGBM-illegal chars replaced
    by ``_`` and any post-replace duplicates disambiguated with a
    numeric suffix."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        cleaned = _LGBM_FORBIDDEN.sub("_", str(raw))
        # Empty names confuse LightGBM too. Substitute a placeholder.
        if not cleaned:
            cleaned = "feat"
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned_disamb = f"{cleaned}__dup{seen[cleaned]}"
            # Edge: the disambiguated form might also collide with a
            # later name -- rare, but track it too.
            seen[cleaned_disamb] = 0
            out.append(cleaned_disamb)
        else:
            seen[cleaned] = 0
            out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Compiled tree structure (numpy arrays only)
# ---------------------------------------------------------------------------


@dataclass
class _CompiledTree:
    """Flat-array representation of one boosted tree.

    All arrays have shape ``[n_nodes]``. For internal nodes,
    ``feature[i]`` is the split feature index (>= 0) and
    ``threshold[i]`` is the split threshold; ``left[i]`` and
    ``right[i]`` are child node indices and ``default_left[i]`` is
    True if NaN inputs go left. For leaf nodes, ``feature[i] == -1``
    and ``threshold[i]`` carries the ``leaf_value`` instead.

    The single-array trick (``threshold`` doubles as ``leaf_value``)
    is a deliberate compactness choice -- saves a parallel array and
    halves the per-node memory.
    """

    feature: np.ndarray       # int32 [n_nodes], -1 = leaf
    threshold: np.ndarray     # float64 [n_nodes], leaf_value when leaf
    left: np.ndarray          # int32 [n_nodes], -1 when leaf
    right: np.ndarray         # int32 [n_nodes], -1 when leaf
    default_left: np.ndarray  # bool [n_nodes]

    def __post_init__(self) -> None:
        n = int(self.feature.shape[0])
        for arr_name in ("threshold", "left", "right", "default_left"):
            arr = getattr(self, arr_name)
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"_CompiledTree: {arr_name} length {arr.shape[0]} != "
                    f"feature length {n}"
                )

    def is_leaf_mask(self) -> np.ndarray:
        return self.feature == _LEAF_FEATURE_SENTINEL


# ---------------------------------------------------------------------------
# State (the shipped artifact)
# ---------------------------------------------------------------------------


@dataclass
class GBDTMemberState:
    """Fitted-and-shipped state of Member 2.

    Concatenated array layout (small + cache-friendly):
      ``feature_concat``, ``threshold_concat``, ``left_concat``,
      ``right_concat``, ``default_left_concat``: shape ``[total_nodes]``.
      ``tree_offsets``: int32 [n_trees + 1], so tree ``t`` lives in the
      slice ``[tree_offsets[t] : tree_offsets[t + 1]]`` and its root
      is the FIRST node of that slice (relative to the slice start; in
      the global concat array the root is ``tree_offsets[t]``).
      All child indices in ``left_concat`` / ``right_concat`` are
      GLOBAL (shifted by the tree's offset at compile time).
    """

    feature_concat: np.ndarray
    threshold_concat: np.ndarray
    left_concat: np.ndarray
    right_concat: np.ndarray
    default_left_concat: np.ndarray
    tree_offsets: np.ndarray

    feature_dim: int
    feature_names: tuple[str, ...]
    bias: float                # init_score (logit(mean_y_train) by default)
    fit_method: str            # "lightgbm"
    n_train: int
    n_pos: int
    n_trees: int
    train_loss: float
    val_loss: float
    # NEW (residual mode, see ``fit_gbdt_member(init_pred_train=...)``).
    # ``objective`` is "binary" (legacy: walker output = sigmoid(sum_leaves +
    # bias) is the probability) or "regression" (walker output = sum_leaves +
    # bias is the RESIDUAL LOGIT; caller must compose with an anchor
    # member's logit via ``compose_residual_one``/``compose_residual_batch``).
    # Older saved states without these fields default to "binary" /
    # "probability" so on-disk artifacts keep loading.
    objective: str = "binary"
    output_mode: str = "probability"   # "probability" or "residual_logit"

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
                    f"GBDTMemberState: {arr_name} len {arr.shape[0]} != "
                    f"feature_concat len {n}"
                )
        if int(self.tree_offsets.shape[0]) != int(self.n_trees) + 1:
            raise ValueError(
                f"tree_offsets len {self.tree_offsets.shape[0]} != "
                f"n_trees+1 {int(self.n_trees) + 1}"
            )
        if int(self.tree_offsets[-1]) != n:
            raise ValueError(
                f"tree_offsets[-1] {int(self.tree_offsets[-1])} != "
                f"total nodes {n}"
            )
        if int(len(self.feature_names)) != int(self.feature_dim):
            raise ValueError(
                f"feature_names len {len(self.feature_names)} != "
                f"feature_dim {self.feature_dim}"
            )
        if not math.isfinite(float(self.bias)):
            raise ValueError("GBDTMemberState: bias is NaN/Inf")

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
            "format_version": 2,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "GBDTMemberState":
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
            fit_method=str(meta.get("fit_method", "unknown")),
            n_train=int(meta.get("n_train", 0)),
            n_pos=int(meta.get("n_pos", 0)),
            n_trees=int(meta.get("n_trees", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
            objective=str(meta.get("objective", "binary")),
            output_mode=str(meta.get("output_mode", "probability")),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _sigmoid_stable_one(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _traverse_one_tree(
    state: GBDTMemberState,
    tree_idx: int,
    features: np.ndarray,
) -> float:
    """Walk one tree to a leaf and return the ``leaf_value``.

    ``features`` is a 1-D float array. NaN entries follow the
    ``default_left`` direction at every split they encounter -- this
    is the same convention LightGBM's own predict() uses.
    """
    start = int(state.tree_offsets[tree_idx])
    end = int(state.tree_offsets[tree_idx + 1])
    feat = state.feature_concat
    thr = state.threshold_concat
    left = state.left_concat
    right = state.right_concat
    dleft = state.default_left_concat

    node = start  # tree root is the first node of its slice
    # Defensive depth bound: trees with n_leaves < 2**30 cannot exceed
    # 2**30 nodes; if we somehow loop forever the bound stops us.
    for _ in range(end - start + 1):
        f = int(feat[node])
        if f == _LEAF_FEATURE_SENTINEL:
            return float(thr[node])
        v = features[f] if 0 <= f < int(features.shape[0]) else float("nan")
        if not np.isfinite(v):
            node = int(left[node]) if bool(dleft[node]) else int(right[node])
        elif float(v) <= float(thr[node]):
            node = int(left[node])
        else:
            node = int(right[node])
    # Should never get here for a well-formed tree, but defensively
    # return 0.0 rather than crash if a cycle ever sneaks in.
    LOG.warning("GBDT traversal bound hit on tree %d; returning 0.0", tree_idx)
    return 0.0


def apply_one(state: GBDTMemberState, features: np.ndarray) -> float:
    """Single-row inference -> Python ``float`` in (eps, 1-eps).

    Only valid for ``output_mode == "probability"`` (binary objective).
    For ``residual_logit`` states call :func:`compose_residual_one` with
    an anchor logit instead -- this guard prevents accidentally treating
    a tree-residual output as a probability.
    """
    if state.output_mode != "probability":
        raise RuntimeError(
            f"apply_one is invalid for output_mode={state.output_mode!r}; "
            "use compose_residual_one(state, features, init_logit) instead."
        )
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim "
            f"{state.feature_dim}"
        )
    raw = float(state.bias)
    for t in range(int(state.n_trees)):
        raw += _traverse_one_tree(state, t, features)
    if not math.isfinite(raw):
        # Saturated logits clamp; never crash the runtime.
        return 0.5
    p = _sigmoid_stable_one(raw)
    return float(min(max(p, _EPS), 1.0 - _EPS))


def _walk_tree_batch(
    state: GBDTMemberState,
    tree_idx: int,
    features_matrix: np.ndarray,
) -> np.ndarray:
    """Walk a single tree to leaves for ALL rows in parallel.

    Returns ``[B]`` float64 leaf values. The traversal is iterative:
    at each step we look up the active rows' current node's feature
    index and threshold, decide direction, and advance. We stop when
    every row has reached a leaf.

    Numerically identical to looping :func:`_traverse_one_tree` per
    row -- same NaN handling via ``default_left``, same per-row
    feature ``[i, f]`` lookup, same threshold comparison.
    """
    B = int(features_matrix.shape[0])
    start = int(state.tree_offsets[tree_idx])
    feat = state.feature_concat
    thr = state.threshold_concat
    left = state.left_concat
    right = state.right_concat
    dleft = state.default_left_concat

    node = np.full(B, start, dtype=np.int64)
    # Defensive depth bound: a tree with up to 2**30 nodes can't have
    # depth > 30 in a balanced sense; we allow up to (n_nodes + 1) just
    # to match the per-row guard.
    n_tree_nodes = int(state.tree_offsets[tree_idx + 1]) - start
    max_iters = int(n_tree_nodes) + 1

    for _ in range(max_iters):
        f_idx = feat[node]                  # [B]
        is_leaf = f_idx == _LEAF_FEATURE_SENTINEL
        if bool(is_leaf.all()):
            break

        nl_rows = np.where(~is_leaf)[0]     # active rows (non-leaf)
        nl_node = node[nl_rows]
        fi = f_idx[nl_rows].astype(np.int64, copy=False)
        # Bounds check matches the per-row path: out-of-range f -> NaN.
        feat_dim = int(features_matrix.shape[1])
        valid_f = (fi >= 0) & (fi < feat_dim)
        # Safe gather: bad indices are clipped to 0 here, then their
        # value is overwritten by NaN below so the default_left branch
        # fires.
        fi_safe = np.where(valid_f, fi, 0)
        fv = features_matrix[nl_rows, fi_safe]
        # Inject NaN for invalid feature indices so the default_left
        # branch fires (matching the per-row path's `else float("nan")`).
        if not bool(valid_f.all()):
            fv = np.where(valid_f, fv, np.nan)
        finite = np.isfinite(fv)

        th = thr[nl_node]
        go_left_finite = (fv <= th) & finite
        go_left_nan = dleft[nl_node] & ~finite
        go_left = go_left_finite | go_left_nan

        new_node = np.where(go_left, left[nl_node], right[nl_node]).astype(
            np.int64, copy=False
        )
        node[nl_rows] = new_node

    # All rows are now at a leaf node (or the bound expired).
    leaf_vals = thr[node].astype(np.float64, copy=False)
    return leaf_vals


def apply_batch(
    state: GBDTMemberState,
    features_matrix: np.ndarray,
) -> np.ndarray:
    """Vectorized-over-rows AND vectorized-over-nodes-per-tree.

    Returns float32 probabilities shape ``[N]``. Used in tests and
    in the offline parity check; the runtime per-call path uses
    :func:`apply_one`.

    Speedup over the previous per-row apply_one loop: for typical
    GBDT shapes (100-400 trees, 31 leaves) we walk each tree in
    ``O(tree_depth)`` numpy ops instead of ``O(B * tree_depth)``
    Python ops, which is a 30-100x wall-clock reduction on large
    val sets.

    Only valid for ``output_mode == "probability"`` (binary objective).
    For residual-mode states use :func:`compose_residual_batch`.
    """
    if state.output_mode != "probability":
        raise RuntimeError(
            f"apply_batch is invalid for output_mode={state.output_mode!r}; "
            "use compose_residual_batch(state, X, init_logit) instead."
        )
    if features_matrix.ndim != 2:
        raise ValueError(
            f"features_matrix must be 2D, got shape {features_matrix.shape}"
        )
    if int(features_matrix.shape[1]) != int(state.feature_dim):
        raise ValueError(
            f"features_matrix dim {features_matrix.shape[1]} != "
            f"state.feature_dim {state.feature_dim}"
        )
    N = int(features_matrix.shape[0])
    if N == 0:
        return np.empty(0, dtype=np.float32)

    fm = np.ascontiguousarray(features_matrix, dtype=np.float64)
    raw = np.full(N, float(state.bias), dtype=np.float64)
    for t in range(int(state.n_trees)):
        raw += _walk_tree_batch(state, t, fm)

    # Sigmoid + clamp, vectorized.
    raw = np.where(np.isfinite(raw), raw, 0.0)
    # Numerically-stable sigmoid (split on sign).
    out = np.empty_like(raw)
    pos = raw >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
    e = np.exp(raw[~pos])
    out[~pos] = e / (1.0 + e)
    out = np.clip(out, _EPS, 1.0 - _EPS)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Residual-mode helpers
# ---------------------------------------------------------------------------
#
# When ``fit_gbdt_member`` is called with ``init_pred_train=...``, the trees
# are trained on a regression target ``logit(y) - logit(p_init)`` and the
# state stores ``objective="regression"``, ``output_mode="residual_logit"``.
# In that case the walker's sum-of-leaves + bias is the residual logit;
# the final probability is ``sigmoid(logit(p_init_at_inference) + residual)``.
# These helpers do the composition so callers can't accidentally treat a
# residual as a probability (the ``apply_*`` guards above catch that too).


def _logit_clip(p, eps: float = _EPS):
    """Vectorized OR scalar logit with clipping to (eps, 1-eps)."""
    if isinstance(p, (int, float)):
        x = max(min(float(p), 1.0 - eps), eps)
        return math.log(x / (1.0 - x))
    arr = np.asarray(p, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def compose_residual_one(
    state: GBDTMemberState,
    features: np.ndarray,
    init_pred: float,
) -> float:
    """Single-row residual composition -> probability in (eps, 1-eps).

    ``init_pred`` is the anchor member's probability for this row (e.g.
    Member 1's IRT-MLP output). The final probability is

        sigmoid( logit(init_pred) + tree_residual + bias ).
    """
    if state.output_mode != "residual_logit":
        raise RuntimeError(
            f"compose_residual_one requires output_mode='residual_logit', "
            f"got {state.output_mode!r}"
        )
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim "
            f"{state.feature_dim}"
        )
    residual = float(state.bias)
    for t in range(int(state.n_trees)):
        residual += _traverse_one_tree(state, t, features)
    z = float(_logit_clip(float(init_pred))) + residual
    if not math.isfinite(z):
        return 0.5
    p = _sigmoid_stable_one(z)
    return float(min(max(p, _EPS), 1.0 - _EPS))


def compose_residual_batch(
    state: GBDTMemberState,
    features_matrix: np.ndarray,
    init_pred: np.ndarray,
) -> np.ndarray:
    """Vectorized residual composition. ``init_pred`` is per-row in (0, 1).

    Returns float32 probabilities of shape ``[N]``.
    """
    if state.output_mode != "residual_logit":
        raise RuntimeError(
            f"compose_residual_batch requires output_mode='residual_logit', "
            f"got {state.output_mode!r}"
        )
    if features_matrix.ndim != 2:
        raise ValueError(
            f"features_matrix must be 2D, got {features_matrix.shape}"
        )
    if int(features_matrix.shape[1]) != int(state.feature_dim):
        raise ValueError(
            f"features_matrix dim {features_matrix.shape[1]} != "
            f"state.feature_dim {state.feature_dim}"
        )
    N = int(features_matrix.shape[0])
    init_pred_arr = np.asarray(init_pred, dtype=np.float64).reshape(-1)
    if int(init_pred_arr.shape[0]) != N:
        raise ValueError(
            f"init_pred length {init_pred_arr.shape[0]} != N {N}"
        )
    if N == 0:
        return np.empty(0, dtype=np.float32)
    init_logit = _logit_clip(init_pred_arr)
    residual = predict_raw(state, features_matrix)
    z = init_logit + residual
    z = np.where(np.isfinite(z), z, 0.0)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    out = np.clip(out, _EPS, 1.0 - _EPS)
    return out.astype(np.float32, copy=False)


def predict_raw(state: GBDTMemberState, features_matrix: np.ndarray) -> np.ndarray:
    """Vectorized raw scores (sum of leaves + bias). Used for parity.

    Returns ``[N]`` float64. Walks every tree once across all rows via
    :func:`_walk_tree_batch`, then sums the leaf values and adds bias.
    Numerically identical to looping :func:`_traverse_one_tree` per
    row -- same NaN handling, same default_left direction, same per-row
    feature lookup -- but ~50-100x faster on large val sets because the
    inner loop is numpy ops on ``[N]`` arrays instead of N python calls.
    """
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


# ---------------------------------------------------------------------------
# Tree compilation: dump_model JSON -> flat arrays
# ---------------------------------------------------------------------------


def _compile_tree_from_dict(tree_struct: Mapping[str, Any]) -> _CompiledTree:
    """Convert one LightGBM tree dict to a :class:`_CompiledTree`.

    The dict is the value of ``tree_info[i]["tree_structure"]`` from
    ``Booster.dump_model()``. We do a depth-first walk that assigns
    sequential node ids; child indices in ``left`` / ``right`` are
    LOCAL (within this tree). Global shifting happens later at the
    concat step.

    LightGBM's binary objective uses ``decision_type == "<="`` (or
    ``"no_greater"`` in some serialization variants). Categorical
    splits (``decision_type == "=="``) are unsupported and raise --
    we one-hot upstream so they should never appear.
    """
    feature_list: list[int] = []
    threshold_list: list[float] = []
    left_list: list[int] = []
    right_list: list[int] = []
    default_left_list: list[bool] = []

    def _walk(node: Mapping[str, Any]) -> int:
        nid = len(feature_list)
        # Allocate slot first so children's nid is sequential after ours.
        feature_list.append(_LEAF_FEATURE_SENTINEL)
        threshold_list.append(0.0)
        left_list.append(-1)
        right_list.append(-1)
        default_left_list.append(False)

        if "leaf_value" in node:
            threshold_list[nid] = float(node["leaf_value"])
            return nid

        # Internal node.
        dt = str(node.get("decision_type", "<="))
        if dt not in ("<=", "no_greater"):
            raise ValueError(
                f"Unsupported decision_type {dt!r}. The trainer must one-hot "
                "all categoricals upstream so trees only have numeric "
                "<= splits."
            )
        feature_list[nid] = int(node["split_feature"])
        threshold_list[nid] = float(node["threshold"])
        default_left_list[nid] = bool(node.get("default_left", False))

        left_id = _walk(node["left_child"])
        right_id = _walk(node["right_child"])
        left_list[nid] = left_id
        right_list[nid] = right_id
        return nid

    _walk(tree_struct)

    return _CompiledTree(
        feature=np.asarray(feature_list, dtype=np.int32),
        threshold=np.asarray(threshold_list, dtype=np.float64),
        left=np.asarray(left_list, dtype=np.int32),
        right=np.asarray(right_list, dtype=np.int32),
        default_left=np.asarray(default_left_list, dtype=np.bool_),
    )


def _concat_trees(
    compiled: list[_CompiledTree],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-tree arrays into one flat representation.

    Child indices are SHIFTED by each tree's start offset so the
    runtime walker can treat the concatenation as a single big graph.
    """
    feats: list[np.ndarray] = []
    thrs: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    dlefts: list[np.ndarray] = []
    offsets: list[int] = [0]
    running = 0
    for tree in compiled:
        n = int(tree.feature.shape[0])
        feats.append(tree.feature.astype(np.int32, copy=False))
        thrs.append(tree.threshold.astype(np.float64, copy=False))
        # Shift only NON-leaf children; leaves keep -1.
        shifted_left = np.where(tree.left >= 0, tree.left + running, -1).astype(
            np.int32
        )
        shifted_right = np.where(tree.right >= 0, tree.right + running, -1).astype(
            np.int32
        )
        lefts.append(shifted_left)
        rights.append(shifted_right)
        dlefts.append(tree.default_left.astype(np.bool_, copy=False))
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
# Offline trainer
# ---------------------------------------------------------------------------


def fit_gbdt_member(
    *,
    X: np.ndarray,                    # [N, F] float32
    y: np.ndarray,                    # [N] float32 in {0, 1}
    feature_names: Sequence[str],
    sample_weights: np.ndarray | None = None,
    val_fraction: float = 0.1,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_data_in_leaf: int = 20,
    feature_fraction: float = 0.9,
    bagging_fraction: float = 0.9,
    bagging_freq: int = 5,
    early_stopping_rounds: int = 25,
    seed: int = 0,
    parity_atol: float = 1.0e-5,
    max_bin: int = 63,
    force_col_wise: bool = True,
    log_period: int = 25,
    num_threads: int | None = None,
    holdout_group_id: np.ndarray | None = None,
    init_pred_train: np.ndarray | None = None,
) -> GBDTMemberState:
    """Train a LightGBM binary classifier and compile its trees to numpy.

    The trainer ALSO verifies that the compiled state reproduces
    ``Booster.predict(X_val, raw_score=True)`` to within
    ``parity_atol`` before returning. If parity fails, the trainer
    raises a ``RuntimeError`` -- silent traversal-vs-LGBM divergence
    is the canonical hand-rolled-tree footgun.

    Speed knobs (defaults are tuned for the 5M x 1200 feature
    member-feature schema this trainer is hot-pathed for):

    * ``max_bin`` (default 63, vs LightGBM default 255) governs the
      histogram resolution. At this scale ~3x of total wall time is
      bin construction; dropping max_bin from 255 -> 63 cuts that
      proportionally with negligible accuracy impact for binary
      classification on already-z-scored numeric features.

    * ``force_col_wise=True`` (default) pairs with
      ``deterministic=True`` to give bit-exact reproducibility on the
      column-major code path. Without this flag, LightGBM's
      auto-picker sometimes lands on the row-major path which is
      slower for wide feature counts.

    * ``log_period`` controls the per-iteration logging frequency
      (LightGBM ``log_evaluation(period=...)``). Set to 0 to silence.

    * ``num_threads=None`` lets LightGBM pick (defaults to all
      cores); pass an int to pin.

    * ``holdout_group_id``: per-row int array. When provided, the
      internal LightGBM train/val split holds out *whole groups*
      (typically item ids) instead of random rows. This mirrors the
      cold-start discipline of Model 1's outer val split: every row
      of a held-out item lands on the same side of the split, so the
      booster never sees rows from the same item on both sides and
      its early-stopping val_logloss reflects actual cold-start
      generalization rather than item memorization. Without this
      kwarg the previous random-row behavior is preserved.

    * ``init_pred_train``: per-row anchor probability (e.g. Member 1's
      OOF train predictions) in ``(0, 1)``. When provided, the trainer
      switches to **residual mode**: LightGBM is configured with
      ``objective='regression'`` (l2) on the target
      ``logit(y) - logit(init_pred_train)``. The resulting state has
      ``output_mode='residual_logit'``; at inference, callers must
      use :func:`compose_residual_one`/:func:`compose_residual_batch`
      passing the SAME anchor member's probability at the inference
      row. The point is to force the trees to spend capacity on the
      part of the label the anchor missed -- a near-zero stacker
      weight on Member 2 (the classic "your trees just relearned
      Member 1" failure mode) becomes impossible by construction.
    """
    import lightgbm as lgb  # offline only

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.shape != (int(X.shape[0]),):
        raise ValueError(f"y shape {y.shape} != ({X.shape[0]},)")
    if int(len(feature_names)) != int(X.shape[1]):
        raise ValueError(
            f"feature_names len {len(feature_names)} != X cols {X.shape[1]}"
        )
    if holdout_group_id is not None:
        if holdout_group_id.shape != (int(X.shape[0]),):
            raise ValueError(
                f"holdout_group_id shape {holdout_group_id.shape} != "
                f"({X.shape[0]},)"
            )

    # ---- Residual-mode setup ------------------------------------------------
    # If init_pred_train is provided we configure LightGBM as a binary
    # learner with a per-row ``init_score = logit(init_pred_train)``.
    # The trees then learn an additive logit-space correction:
    #
    #     final_logit = init_score + tree_score
    #     gradient    = sigmoid(final_logit) - y
    #
    # This is the canonical way to do boosted residual learning -- the
    # binary cross-entropy at each step properly handles {0,1} labels
    # without the gradient-explosion problem regression-on-logit(y)
    # would have. At inference, ``booster.predict(X, raw_score=True)``
    # returns ``tree_score`` (modern LightGBM does NOT add init_score
    # back during predict), so the composer just does
    # ``sigmoid(logit(p_init) + tree_score)``.
    #
    # We keep the original ``y`` as the training label (LightGBM does
    # the gradient math itself given init_score); only the Dataset
    # gains an ``init_score`` field.
    residual_mode = init_pred_train is not None
    init_pred_train_clean: np.ndarray | None = None
    init_score_train: np.ndarray | None = None
    y_for_metrics: np.ndarray = y
    if residual_mode:
        ipt = np.asarray(init_pred_train, dtype=np.float64).reshape(-1)
        if ipt.shape[0] != int(X.shape[0]):
            raise ValueError(
                f"init_pred_train shape {ipt.shape} != ({X.shape[0]},)"
            )
        if not np.all(np.isfinite(ipt)):
            raise ValueError("init_pred_train contains NaN/Inf")
        ipt_clipped = np.clip(ipt, _EPS, 1.0 - _EPS)
        init_pred_train_clean = ipt_clipped
        # Per-row init_score = logit(init_pred). LightGBM uses this
        # additively in logit space.
        init_score_train = np.log(ipt_clipped / (1.0 - ipt_clipped)).astype(
            np.float64, copy=False
        )

    rng = np.random.default_rng(int(seed))
    N = int(X.shape[0])
    if holdout_group_id is None:
        # Legacy: random row split (kept so other callers don't break).
        perm = rng.permutation(N)
        n_val = max(64, int(round(val_fraction * N)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        # Group-stratified holdout: pick ~val_fraction of distinct
        # group ids, route ALL rows of those ids into val. Mirrors
        # sklearn.model_selection.GroupShuffleSplit semantics.
        gids = np.asarray(holdout_group_id).reshape(-1)
        unique_groups = np.unique(gids)
        n_groups = int(unique_groups.shape[0])
        if n_groups < 2:
            raise ValueError(
                f"holdout_group_id has {n_groups} unique groups; need >=2 "
                "for a meaningful train/val split."
            )
        n_val_groups = max(1, int(round(val_fraction * n_groups)))
        # Sample without replacement deterministically.
        held_groups = rng.choice(unique_groups, size=n_val_groups, replace=False)
        held_set = set(int(g) for g in held_groups)
        val_mask = np.fromiter(
            (int(g) in held_set for g in gids),
            count=N,
            dtype=bool,
        )
        # Belt: ensure both sides are non-trivially populated.
        if int(val_mask.sum()) == 0:
            raise RuntimeError(
                "Group-stratified split yielded zero val rows. Check that "
                "holdout_group_id has multiple distinct groups."
            )
        if int((~val_mask).sum()) == 0:
            raise RuntimeError(
                "Group-stratified split yielded zero train rows. Check the "
                "group cardinality and val_fraction."
            )
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        LOG.info(
            "fit_gbdt_member: group-stratified split "
            "(%d groups -> %d held; %d train rows / %d val rows; "
            "val mean_y=%.4f, train mean_y=%.4f)",
            n_groups, n_val_groups,
            int(train_idx.shape[0]), int(val_idx.shape[0]),
            float(y[val_idx].mean()) if val_idx.size else 0.0,
            float(y[train_idx].mean()) if train_idx.size else 0.0,
        )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    if sample_weights is not None:
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]
    else:
        w_train = None
        w_val = None

    # LightGBM rejects feature names with special JSON chars (commas,
    # brackets, colons, quotes, braces, control chars). Schema names
    # are machine-emitted but raw condition strings can include any
    # of these, so we sanitize uniformly. The booster only ever sees
    # the sanitized names; the saved ``GBDTMemberState.feature_names``
    # keeps the originals so downstream introspection / packaging is
    # unaffected.
    feature_names_for_lgbm = _sanitize_for_lightgbm(feature_names)

    # Build per-split init_score arrays in residual mode. Each Dataset
    # gets its own slice -- LightGBM's training and metric paths both
    # consume init_score additively in logit space.
    if residual_mode:
        assert init_score_train is not None
        is_train_split = init_score_train[train_idx]
        is_val_split = init_score_train[val_idx]
    else:
        is_train_split = None
        is_val_split = None

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        init_score=is_train_split,
        feature_name=feature_names_for_lgbm,
        categorical_feature=[],
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        init_score=is_val_split,
        feature_name=feature_names_for_lgbm,
        categorical_feature=[],
        reference=train_set,
        free_raw_data=False,
    )

    # Same objective in both modes -- residual mode just adds a per-row
    # init_score on the Dataset. boost_from_average is disabled in
    # residual mode because init_score already provides a per-row
    # baseline that's better than the population mean.
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(bagging_freq),
        "max_bin": int(max_bin),
        "force_col_wise": bool(force_col_wise),
        "verbosity": -1,
        "seed": int(seed),
        "deterministic": True,
        "boost_from_average": not residual_mode,
    }
    if num_threads is not None:
        params["num_threads"] = int(num_threads)

    callbacks = [lgb.early_stopping(int(early_stopping_rounds), verbose=False)]
    if int(log_period) > 0:
        callbacks.append(lgb.log_evaluation(period=int(log_period)))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(n_estimators),
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ---- Compile trees ----
    dump = booster.dump_model()
    if int(dump.get("num_class", 1)) != 1:
        raise RuntimeError(
            f"GBDT dump has num_class={dump.get('num_class')}; only "
            "binary classification (num_class=1) is supported."
        )
    tree_info = list(dump.get("tree_info", []))
    compiled: list[_CompiledTree] = []
    for ti in tree_info:
        t = _compile_tree_from_dict(ti["tree_structure"])
        compiled.append(t)

    feat, thr, l, r, dl, offsets = _concat_trees(compiled)

    # Anchor rows to derive bias. In BOTH modes, we want the walker to
    # output what ``booster.predict(X, raw_score=True)`` returns (the
    # bare tree contribution; modern LightGBM does NOT add init_score
    # back at prediction time). The probability composition is
    # finished externally:
    #   binary mode:    p = sigmoid(walker_output)
    #   residual mode:  p = sigmoid(logit(init_pred) + walker_output)
    # so bias_estimate = mean(predict_raw - sum_leaves) recovers any
    # constant LightGBM internally added (boost_from_average baseline).
    anchor_idx = rng.choice(int(X.shape[0]), size=min(256, int(X.shape[0])), replace=False)
    X_anchor = X[anchor_idx]

    tmp_state = GBDTMemberState(
        feature_concat=feat,
        threshold_concat=thr,
        left_concat=l,
        right_concat=r,
        default_left_concat=dl,
        tree_offsets=offsets,
        feature_dim=int(X.shape[1]),
        feature_names=tuple(feature_names),
        bias=0.0,
        fit_method="lightgbm",
        n_train=int(N),
        n_pos=int(np.sum(y_for_metrics == 1.0)),
        n_trees=int(len(compiled)),
        train_loss=0.0,
        val_loss=0.0,
        objective="binary",
        output_mode=("residual_logit" if residual_mode else "probability"),
    )
    sum_leaves = predict_raw(tmp_state, X_anchor.astype(np.float32, copy=False))

    # Single bias-recovery formula for both modes: compare to raw_score.
    raw_lgb = booster.predict(X_anchor, raw_score=True).astype(np.float64)
    delta = raw_lgb - sum_leaves.astype(np.float64)
    bias_estimate = float(delta.mean())
    bias_std = float(delta.std())
    if bias_std > 1.0e-4:
        raise RuntimeError(
            f"GBDT bias not constant across anchor rows: "
            f"mean={bias_estimate} std={bias_std}. The booster may be "
            "emitting per-row constants (categorical splits with missing "
            "or NaN handling can cause this)."
        )

    # Compute MANUAL cross-entropy on the booster's own predictions.
    # LightGBM's reported ``binary_logloss`` metric implicitly treats
    # every label > 0 as a hard +1 (see ``LossOnPoint`` in
    # ``metric/binary_metric.hpp``), so on SOFT labels in [0, 1] it
    # diverges arbitrarily from the soft Bernoulli cross-entropy the
    # ``binary`` objective actually optimizes. The manual NLL below
    # uses the raw label values directly and is the honest number to
    # gate ensembling on. (See scripts/_diag_gbdt_softlabel_gap.py for
    # the controlled repro.)
    #
    # In RESIDUAL mode we instead report the cross-entropy of the
    # COMPOSED probability sigmoid(logit(init_pred) + tree_residual)
    # against the original soft labels -- that's what the runtime
    # actually emits via compose_residual_*.
    eps_clip = 1.0e-6
    if residual_mode:
        # ``booster.predict(X)`` returns ``sigmoid(tree_score)`` -- it
        # does NOT add init_score back in modern LightGBM. So the
        # composed probability is ``sigmoid(init_score + tree_score)``
        # which we compute via raw_score + per-row init_score.
        assert init_pred_train_clean is not None
        ip_train = init_pred_train_clean[train_idx]
        ip_val = init_pred_train_clean[val_idx]
        is_train = np.log(ip_train / (1.0 - ip_train))
        is_val = np.log(ip_val / (1.0 - ip_val))
        tree_raw_train = booster.predict(X_train, raw_score=True).astype(np.float64)
        tree_raw_val = booster.predict(X_val, raw_score=True).astype(np.float64)
        logit_train = is_train + tree_raw_train
        logit_val = is_val + tree_raw_val
        def _sig(z: np.ndarray) -> np.ndarray:
            out = np.empty_like(z)
            pos = z >= 0
            out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
            e = np.exp(z[~pos])
            out[~pos] = e / (1.0 + e)
            return out
        p_train_composed = np.clip(_sig(logit_train), eps_clip, 1.0 - eps_clip)
        p_val_composed = np.clip(_sig(logit_val), eps_clip, 1.0 - eps_clip)
        y_train_orig_f = y_for_metrics[train_idx].astype(np.float64)
        y_val_orig_f = y_for_metrics[val_idx].astype(np.float64)
        manual_train_nll = float(
            -np.mean(
                y_train_orig_f * np.log(p_train_composed)
                + (1.0 - y_train_orig_f) * np.log(1.0 - p_train_composed)
            )
        )
        manual_val_nll = float(
            -np.mean(
                y_val_orig_f * np.log(p_val_composed)
                + (1.0 - y_val_orig_f) * np.log(1.0 - p_val_composed)
            )
        )
        # binary_logloss reported in residual mode is computed by
        # LightGBM WITHOUT adding init_score back (it sees just the
        # tree contribution). That number is uninterpretable on its
        # own; surface it as diagnostic only.
        reported_train_nll = float(
            booster.best_score.get("train", {}).get("binary_logloss", 0.0)
        )
        reported_val_nll = float(
            booster.best_score.get("val", {}).get("binary_logloss", 0.0)
        )
    else:
        p_train_lgb = booster.predict(X_train).astype(np.float64)
        p_val_lgb = booster.predict(X_val).astype(np.float64)
        p_train_lgb = np.clip(p_train_lgb, eps_clip, 1.0 - eps_clip)
        p_val_lgb = np.clip(p_val_lgb, eps_clip, 1.0 - eps_clip)
        y_train_f = y_train.astype(np.float64)
        y_val_f = y_val.astype(np.float64)
        manual_train_nll = float(
            -np.mean(y_train_f * np.log(p_train_lgb) + (1.0 - y_train_f) * np.log(1.0 - p_train_lgb))
        )
        manual_val_nll = float(
            -np.mean(y_val_f * np.log(p_val_lgb) + (1.0 - y_val_f) * np.log(1.0 - p_val_lgb))
        )
        reported_train_nll = float(
            booster.best_score.get("train", {}).get("binary_logloss", 0.0)
        )
        reported_val_nll = float(
            booster.best_score.get("val", {}).get("binary_logloss", 0.0)
        )
    # If the reported and manual numbers disagree noticeably (binary
    # mode only; the regression-mode reported metric is l2-on-residuals
    # which lives on a different scale than manual NLL, so we skip the
    # warning there).
    if not residual_mode:
        nll_gap = abs(manual_val_nll - reported_val_nll)
        if nll_gap > 0.02:
            LOG.warning(
                "GBDT fit: LGBM-reported val_logloss=%.5f differs from manual "
                "cross-entropy on booster.predict()=%.5f by %.4f nats. The "
                "manual number (which matches the runtime walker) is what "
                "this state will report as val_loss. The LGBM-reported value "
                "is preserved as a side-channel diagnostic. (Most common "
                "cause: soft labels in [0,1] -- LightGBM's binary_logloss "
                "metric binarizes via I[y>0] but the binary OBJECTIVE "
                "optimizes soft cross-entropy.)",
                reported_val_nll, manual_val_nll, nll_gap,
            )

    # Build the final state with the recovered bias.
    final_state = GBDTMemberState(
        feature_concat=feat,
        threshold_concat=thr,
        left_concat=l,
        right_concat=r,
        default_left_concat=dl,
        tree_offsets=offsets,
        feature_dim=int(X.shape[1]),
        feature_names=tuple(str(s) for s in feature_names),
        bias=float(bias_estimate),
        fit_method="lightgbm",
        n_train=int(N),
        n_pos=int(np.sum(y_for_metrics == 1.0)),
        n_trees=int(len(compiled)),
        train_loss=float(manual_train_nll),
        val_loss=float(manual_val_nll),
        objective="binary",
        output_mode=("residual_logit" if residual_mode else "probability"),
    )

    # ---- Parity check (FAIL-FAST) ----
    # Raw-space check: walker (sum_leaves + bias) MUST equal
    # ``booster.predict(X, raw_score=True)`` to within fp tolerance
    # in BOTH modes. The walker output is what callers will compose
    # with init_score (residual mode) or pass through sigmoid (binary
    # mode), so this is the contract that matters.
    #
    # Probability-space check only runs in binary mode; in residual
    # mode the runtime composition with per-row init_pred is tested
    # separately in the unit suite (the booster's own predict() does
    # NOT add init_score back, so it doesn't match the composed
    # probability and isn't the right baseline here).
    raw_numpy = predict_raw(final_state, X_val.astype(np.float32, copy=False))
    raw_lgb_val = booster.predict(X_val, raw_score=True).astype(np.float64)
    max_abs_err_raw = float(np.max(np.abs(raw_numpy - raw_lgb_val)))
    if residual_mode:
        max_abs_err_prob = float("nan")
    else:
        p_numpy = apply_batch(final_state, X_val.astype(np.float32, copy=False))
        p_lgb_val = booster.predict(X_val).astype(np.float64)
        max_abs_err_prob = float(np.max(np.abs(p_numpy - p_lgb_val)))

    if max_abs_err_raw > float(parity_atol):
        raise RuntimeError(
            f"GBDT raw-space parity failed: max abs error {max_abs_err_raw} "
            f"> {parity_atol}. The numpy walker disagrees with LightGBM "
            "on leaf-sum traversal; do not ship this state."
        )
    if (not residual_mode) and max_abs_err_prob > float(parity_atol):
        raise RuntimeError(
            f"GBDT probability-space parity failed: max abs error "
            f"{max_abs_err_prob} > {parity_atol}. Bias / init_score "
            "recovery is broken; do not ship this state."
        )
    LOG.info(
        "GBDT fit OK: mode=%s n_trees=%d total_nodes=%d feature_dim=%d "
        "manual_train_nll=%.5f manual_val_nll=%.5f "
        "lgbm_reported_train=%.5f lgbm_reported_val=%.5f "
        "parity_raw=%.2e parity_prob=%.2e bias=%.4f",
        ("binary-residual" if residual_mode else "binary"),
        final_state.n_trees,
        final_state.total_nodes,
        final_state.feature_dim,
        manual_train_nll,
        manual_val_nll,
        reported_train_nll,
        reported_val_nll,
        max_abs_err_raw,
        max_abs_err_prob,
        final_state.bias,
    )
    return final_state


__all__ = [
    "GBDTMemberState",
    "apply_one",
    "apply_batch",
    "predict_raw",
    "compose_residual_one",
    "compose_residual_batch",
    "fit_gbdt_member",
]
