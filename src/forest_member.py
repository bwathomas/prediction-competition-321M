"""Package-free random-forest / ExtraTrees member: trained with
scikit-learn offline, traversed with pure numpy at runtime (no
``import sklearn`` in ``model.py``).

Forest semantics differ from a boosted GBDT in two ways:

1. **Average, not sum.** A forest's prediction is the MEAN over trees of
   each tree's leaf output, not a sum. So this walker averages and there
   is no global ``bias`` term.

2. **Leaf output is already a probability** (classifier: the leaf's
   class-1 fraction) or a real value (regressor: the leaf mean target).
   There is no final sigmoid -- the averaged leaf outputs ARE the
   prediction. ``output_mode`` records which.

Like the other members, the split comparison is sklearn's ``x <=
threshold -> left``. Missing values: modern sklearn (>=1.4) stores
``tree_.missing_go_to_left`` per node; if present we honor it, otherwise
NaN routes left (a deterministic fallback; the offline parity check will
catch any disagreement against ``predict_proba``).

Runtime contract:
``apply_one(state, feats) -> float``  (clamped to (eps, 1-eps))
``apply_batch(state, feats_matrix) -> np.ndarray[N] float32``
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger("forest_member")

_EPS = 1.0e-6
_LEAF_FEATURE_SENTINEL: int = -1


@dataclass
class ForestMemberState:
    """Fitted-and-shipped state of a sklearn forest member.

    ``leaf_value`` carries, for leaf nodes, the per-tree leaf OUTPUT that
    gets averaged across trees (class-1 probability for a classifier, mean
    target for a regressor). The forest prediction is
    ``mean_t(leaf_value reached in tree t)`` -- no sigmoid, no bias.
    """

    feature_concat: np.ndarray        # int32 [total_nodes], -1 = leaf
    threshold_concat: np.ndarray      # float64 [total_nodes], leaf_value when leaf
    left_concat: np.ndarray           # int32 [total_nodes]
    right_concat: np.ndarray          # int32 [total_nodes]
    default_left_concat: np.ndarray   # bool [total_nodes], NaN -> left if True
    tree_offsets: np.ndarray          # int32 [n_trees + 1]

    feature_dim: int
    feature_names: tuple[str, ...]
    fit_method: str                   # "sklearn_extratrees" / "sklearn_randomforest"
    n_train: int
    n_pos: int
    n_trees: int
    train_loss: float
    val_loss: float
    objective: str = "forest"
    output_mode: str = "mean_proba"   # mean over trees; already a probability

    def __post_init__(self) -> None:
        n = int(self.feature_concat.shape[0])
        for arr_name in (
            "threshold_concat", "left_concat", "right_concat", "default_left_concat",
        ):
            arr = getattr(self, arr_name)
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"ForestMemberState: {arr_name} len {arr.shape[0]} != "
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
        if int(self.n_trees) < 1:
            raise ValueError("ForestMemberState: n_trees must be >= 1")

    @property
    def total_nodes(self) -> int:
        return int(self.feature_concat.shape[0])

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
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "n_trees": int(self.n_trees),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "objective": str(self.objective),
            "output_mode": str(self.output_mode),
            "format_version": 1,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "ForestMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "trees.npz") as npz:
            return cls(
                feature_concat=npz["feature_concat"].astype(np.int32, copy=False),
                threshold_concat=npz["threshold_concat"].astype(np.float64, copy=False),
                left_concat=npz["left_concat"].astype(np.int32, copy=False),
                right_concat=npz["right_concat"].astype(np.int32, copy=False),
                default_left_concat=npz["default_left_concat"].astype(np.bool_, copy=False),
                tree_offsets=npz["tree_offsets"].astype(np.int32, copy=False),
                feature_dim=int(meta["feature_dim"]),
                feature_names=tuple(meta["feature_names"]),
                fit_method=str(meta.get("fit_method", "sklearn_forest")),
                n_train=int(meta.get("n_train", 0)),
                n_pos=int(meta.get("n_pos", 0)),
                n_trees=int(meta.get("n_trees", 1)),
                train_loss=float(meta.get("train_loss", 0.0)),
                val_loss=float(meta.get("val_loss", 0.0)),
                objective=str(meta.get("objective", "forest")),
                output_mode=str(meta.get("output_mode", "mean_proba")),
            )


# ---------------------------------------------------------------------------
# Pure-numpy inference (mean over trees, no sigmoid)
# ---------------------------------------------------------------------------


def _traverse_one_tree(state: ForestMemberState, tree_idx: int, features: np.ndarray) -> float:
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
        elif float(v) <= float(thr[node]):    # sklearn: x <= threshold -> left
            node = int(left[node])
        else:
            node = int(right[node])
    LOG.warning("Forest traversal bound hit on tree %d; returning 0.0", tree_idx)
    return 0.0


def apply_one(state: ForestMemberState, features: np.ndarray) -> float:
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim {state.feature_dim}"
        )
    acc = 0.0
    for t in range(int(state.n_trees)):
        acc += _traverse_one_tree(state, t, features)
    p = acc / float(state.n_trees)
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, _EPS), 1.0 - _EPS))


def _walk_tree_batch(state: ForestMemberState, tree_idx: int, fm: np.ndarray) -> np.ndarray:
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
        go_left = ((fv <= th) & finite) | (dleft[nl_node] & ~finite)
        node[nl_rows] = np.where(go_left, left[nl_node], right[nl_node]).astype(
            np.int64, copy=False
        )
    return thr[node].astype(np.float64, copy=False)


def predict_mean(state: ForestMemberState, features_matrix: np.ndarray) -> np.ndarray:
    """Vectorized mean-over-trees leaf output (the raw forest prediction
    before clamping). Returns float64 ``[N]``."""
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
    acc = np.zeros(N, dtype=np.float64)
    for t in range(int(state.n_trees)):
        acc += _walk_tree_batch(state, t, fm)
    return acc / float(state.n_trees)


def apply_batch(state: ForestMemberState, features_matrix: np.ndarray) -> np.ndarray:
    p = predict_mean(state, features_matrix)
    if p.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, _EPS, 1.0 - _EPS).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Compilation: sklearn tree arrays -> flat arrays
# ---------------------------------------------------------------------------


def _compile_sklearn_tree(estimator: Any, is_classifier: bool) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Compile one fitted sklearn ``DecisionTree*`` to flat local arrays.

    ``estimator`` is a single tree (``forest.estimators_[i]``). sklearn's
    ``tree_`` already exposes dense ``0..n-1`` node arrays, so no id remap
    is needed. Leaf output:
      classifier -> value[node, 0, 1] / value[node, 0].sum()  (P(class=1))
      regressor  -> value[node, 0, 0]
    """
    tree = estimator.tree_
    n = int(tree.node_count)
    children_left = np.asarray(tree.children_left, dtype=np.int64)
    children_right = np.asarray(tree.children_right, dtype=np.int64)
    sk_feature = np.asarray(tree.feature, dtype=np.int64)   # -2 for leaves
    sk_threshold = np.asarray(tree.threshold, dtype=np.float64)
    value = np.asarray(tree.value)                          # [n,1,n_out]

    feature = np.full(n, _LEAF_FEATURE_SENTINEL, dtype=np.int32)
    threshold = np.zeros(n, dtype=np.float64)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    default_left = np.zeros(n, dtype=np.bool_)

    # Optional missing-value routing (sklearn >= 1.4).
    mgtl = getattr(tree, "missing_go_to_left", None)
    mgtl = None if mgtl is None else np.asarray(mgtl).astype(bool)

    is_leaf = children_left == -1   # sklearn TREE_LEAF == -1
    for nid in range(n):
        if is_leaf[nid]:
            if is_classifier:
                v = value[nid, 0]
                s = float(v.sum())
                # binary classifier: column index of class label 1.
                p1 = float(v[1] / s) if (s > 0 and v.shape[0] >= 2) else 0.0
                threshold[nid] = p1
            else:
                threshold[nid] = float(value[nid, 0, 0])
        else:
            feature[nid] = int(sk_feature[nid])
            threshold[nid] = float(sk_threshold[nid])
            left[nid] = int(children_left[nid])
            right[nid] = int(children_right[nid])
            if mgtl is not None:
                default_left[nid] = bool(mgtl[nid])
            else:
                default_left[nid] = True  # deterministic fallback
    return feature, threshold, left, right, default_left


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


def compile_forest(
    forest: Any,
    *,
    feature_names: Sequence[str],
    feature_dim: int,
    anchor_X: np.ndarray,
    parity_atol: float = 1.0e-5,
    n_train: int = 0,
    n_pos: int = 0,
) -> ForestMemberState:
    """Compile a fitted sklearn forest (RandomForest/ExtraTrees,
    classifier or regressor) to a parity-verified numpy state."""
    estimators = list(forest.estimators_)
    if not estimators:
        raise ValueError("forest has no estimators_")
    is_classifier = hasattr(forest, "classes_") and hasattr(forest, "predict_proba")
    fit_method = "sklearn_" + type(forest).__name__.lower()
    output_mode = "mean_proba" if is_classifier else "mean_value"

    per_tree = [_compile_sklearn_tree(est, is_classifier) for est in estimators]
    feat, thr, l, r, dl, offsets = _concat_trees(per_tree)

    state = ForestMemberState(
        feature_concat=feat, threshold_concat=thr, left_concat=l, right_concat=r,
        default_left_concat=dl, tree_offsets=offsets,
        feature_dim=int(feature_dim), feature_names=tuple(str(s) for s in feature_names),
        fit_method=fit_method, n_train=int(n_train), n_pos=int(n_pos),
        n_trees=len(estimators), train_loss=0.0, val_loss=0.0,
        objective="forest", output_mode=output_mode,
    )

    # ---- Parity (FAIL-FAST) vs sklearn ----
    anchor = np.ascontiguousarray(anchor_X, dtype=np.float64)
    p_np = predict_mean(state, anchor)
    if is_classifier:
        # column index of label==1
        classes = list(getattr(forest, "classes_", [0, 1]))
        col = classes.index(1) if 1 in classes else (len(classes) - 1)
        p_sk = np.asarray(forest.predict_proba(anchor), dtype=np.float64)[:, col]
    else:
        p_sk = np.asarray(forest.predict(anchor), dtype=np.float64).reshape(-1)
    max_abs = float(np.max(np.abs(p_np - p_sk)))
    if max_abs > float(parity_atol):
        raise RuntimeError(
            f"Forest parity failed: max abs error {max_abs} > {parity_atol} "
            f"(is_classifier={is_classifier})."
        )
    LOG.info(
        "Forest compile OK: %s n_trees=%d total_nodes=%d feature_dim=%d "
        "output_mode=%s parity=%.2e",
        fit_method, state.n_trees, state.total_nodes, state.feature_dim,
        output_mode, max_abs,
    )
    return state


def fit_forest_member(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    classifier: bool = True,
    n_estimators: int = 150,
    max_features: float | str = 0.3,
    min_samples_leaf: int = 20,
    max_depth: int | None = None,
    seed: int = 0,
    parity_atol: float = 1.0e-5,
    val_fraction: float = 0.1,
    num_threads: int | None = None,
) -> ForestMemberState:
    """Train an sklearn ExtraTrees member on (X, y) and compile to a
    parity-verified numpy state.

    ``classifier=True`` fits ``ExtraTreesClassifier`` on ``(y >= 0.5)``
    hard labels (leaf output = class-1 fraction). ``classifier=False``
    fits ``ExtraTreesRegressor`` on the soft labels (leaf output = mean
    target). Match whichever the AIDE roster used.
    """
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor  # offline only

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
    Xtr = np.ascontiguousarray(X[train_idx], dtype=np.float64)

    common = dict(
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
        max_depth=max_depth,
        random_state=int(seed),
        n_jobs=-1 if num_threads is None else int(num_threads),
    )
    if classifier:
        ytr = (y[train_idx] >= 0.5).astype(np.int64)
        forest = ExtraTreesClassifier(**common).fit(Xtr, ytr)
    else:
        forest = ExtraTreesRegressor(**common).fit(Xtr, y[train_idx].astype(np.float64))

    state = compile_forest(
        forest,
        feature_names=feature_names,
        feature_dim=int(X.shape[1]),
        anchor_X=X[rng.choice(N, size=min(256, N), replace=False)],
        parity_atol=parity_atol,
        n_train=N,
        n_pos=int(np.sum(y >= 0.5)),
    )

    eps = 1.0e-6
    p_val = np.clip(apply_batch(state, X[val_idx].astype(np.float64)), eps, 1.0 - eps)
    yv = y[val_idx].astype(np.float64)
    state.val_loss = float(-np.mean(yv * np.log(p_val) + (1.0 - yv) * np.log(1.0 - p_val)))
    return state


__all__ = [
    "ForestMemberState",
    "apply_one",
    "apply_batch",
    "predict_mean",
    "compile_forest",
    "fit_forest_member",
]
