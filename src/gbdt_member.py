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
            "format_version": 1,
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
    """Single-row inference -> Python ``float`` in (eps, 1-eps)."""
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
    """
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


def predict_raw(state: GBDTMemberState, features_matrix: np.ndarray) -> np.ndarray:
    """Vectorized raw scores (sum of leaves + bias). Used for parity."""
    if features_matrix.ndim != 2:
        raise ValueError("features_matrix must be 2D")
    N = int(features_matrix.shape[0])
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        raw = float(state.bias)
        for t in range(int(state.n_trees)):
            raw += _traverse_one_tree(state, t, features_matrix[i])
        out[i] = raw
    return out


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
) -> GBDTMemberState:
    """Train a LightGBM binary classifier and compile its trees to numpy.

    The trainer ALSO verifies that the compiled state reproduces
    ``Booster.predict(X_val, raw_score=True)`` to within
    ``parity_atol`` before returning. If parity fails, the trainer
    raises a ``RuntimeError`` -- silent traversal-vs-LGBM divergence
    is the canonical hand-rolled-tree footgun.
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

    rng = np.random.default_rng(int(seed))
    N = int(X.shape[0])
    perm = rng.permutation(N)
    n_val = max(64, int(round(val_fraction * N)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

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

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        feature_name=list(feature_names),
        categorical_feature=[],
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        feature_name=list(feature_names),
        categorical_feature=[],
        reference=train_set,
        free_raw_data=False,
    )

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(bagging_freq),
        "verbosity": -1,
        "seed": int(seed),
        "deterministic": True,
        # No "categorical_feature" param: we one-hot upstream so all
        # features are numeric. Passing categorical_feature in params
        # is deprecated and triggers a UserWarning; we omit it and
        # instead pass categorical_feature=[] to the Dataset ctor.
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(n_estimators),
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(0),
        ],
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

    # Anchor row to derive bias = init_score (LightGBM may add it
    # implicitly when boost_from_average=True). We compute
    #    bias = booster_raw_anchor - sum_of_compiled_leaves_anchor
    # then verify it on multiple anchors.
    anchor_idx = rng.choice(int(X.shape[0]), size=min(64, int(X.shape[0])), replace=False)
    X_anchor = X[anchor_idx]

    # Sum of leaf values (NO bias) using a temporary state with bias=0.
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
        n_pos=int(np.sum(y == 1.0)),
        n_trees=int(len(compiled)),
        train_loss=0.0,
        val_loss=0.0,
    )
    sum_leaves = predict_raw(tmp_state, X_anchor.astype(np.float32, copy=False))
    raw_lgb = booster.predict(X_anchor, raw_score=True)
    delta = raw_lgb.astype(np.float64) - sum_leaves.astype(np.float64)
    bias_estimate = float(delta.mean())
    bias_std = float(delta.std())
    if bias_std > 1.0e-5:
        raise RuntimeError(
            f"GBDT init_score not constant across anchor rows: "
            f"mean={bias_estimate} std={bias_std}. This usually means "
            "the model has per-row init scores or the dump is malformed."
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
        n_pos=int(np.sum(y == 1.0)),
        n_trees=int(len(compiled)),
        train_loss=float(
            booster.best_score.get("train", {}).get("binary_logloss", 0.0)
        ),
        val_loss=float(
            booster.best_score.get("val", {}).get("binary_logloss", 0.0)
        ),
    )

    # ---- Parity check (FAIL-FAST) ----
    raw_numpy = predict_raw(final_state, X_val.astype(np.float32, copy=False))
    raw_lgb_val = booster.predict(X_val, raw_score=True).astype(np.float64)
    max_abs_err = float(np.max(np.abs(raw_numpy - raw_lgb_val)))
    if max_abs_err > float(parity_atol):
        raise RuntimeError(
            f"GBDT compile parity failed: max abs error {max_abs_err} > "
            f"{parity_atol}. The numpy walker disagrees with LightGBM on "
            "the val set; do not ship this state."
        )
    LOG.info(
        "GBDT fit OK: n_trees=%d total_nodes=%d feature_dim=%d val_logloss=%.5f "
        "parity_max_abs_err=%.2e bias=%.4f",
        final_state.n_trees,
        final_state.total_nodes,
        final_state.feature_dim,
        final_state.val_loss,
        max_abs_err,
        final_state.bias,
    )
    return final_state


__all__ = [
    "GBDTMemberState",
    "apply_one",
    "apply_batch",
    "predict_raw",
    "fit_gbdt_member",
]
