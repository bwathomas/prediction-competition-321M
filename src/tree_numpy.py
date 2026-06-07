"""Pure-Python+numpy inference for a cuML/treelite RandomForest (soft-label regressor).

The Codabench runtime is numpy-only (no cuML/GPU), but our `etbig` member is a cuML
RandomForestRegressor. This module extracts the forest into flat numpy arrays (offline,
needs treelite) and runs traversal at inference with numpy ONLY.

Offline (build, needs treelite):
    arrs = extract_forest(cuml_model.as_treelite())   # or pass a treelite.Model
    save_forest(arrs, "etbig_forest.npz")

Runtime (numpy only):
    arrs = load_forest("etbig_forest.npz")
    p = forest_predict(X, arrs)        # X: [N, num_feature] float32 -> [N] float64 in [0,1]

Contract verified against cuML's own predict (see tests / verify_tree_numpy). The forest is
a regressor with average_tree_output=True + identity postprocessor, so the output is the mean
of per-tree leaf values. Internal nodes use comparison_op "<=": route LEFT when
``x[feature] <= threshold`` (treelite semantics); missing/NaN routes per ``default_left``.
"""
from __future__ import annotations

import json
import numpy as np


def extract_forest(tl_model) -> dict:
    """Flatten a treelite Model into concatenated per-node numpy arrays.

    Returns a dict of arrays (one entry per node, trees concatenated, with ``offsets``
    giving each tree's [start, end) slice). Assumes a numerical-split regression forest
    with identity postprocessor and mean tree aggregation (asserts these).
    """
    J = json.loads(tl_model.dump_as_json(pretty_print=False))
    assert J["task_type"] == "kRegressor", f"unexpected task_type {J['task_type']}"
    assert J.get("postprocessor", "identity") == "identity", J.get("postprocessor")
    assert bool(J.get("average_tree_output", True)), "expected average_tree_output=True"
    nf = int(J["num_feature"])
    base = float(np.asarray(J.get("base_scores", [0.0])).reshape(-1)[0])

    feat, thr, left, right, leaf, isleaf, dleft, offsets = [], [], [], [], [], [], [], [0]
    for t in J["trees"]:
        nodes = t["nodes"]; nn = len(nodes)
        F = np.zeros(nn, np.int32); T = np.zeros(nn, np.float32)
        L = np.full(nn, -1, np.int32); R = np.full(nn, -1, np.int32)
        LV = np.zeros(nn, np.float32); IL = np.zeros(nn, np.bool_); DL = np.zeros(nn, np.bool_)
        for nd in nodes:
            i = int(nd["node_id"])
            if nd.get("node_type") == "leaf_node" or "leaf_value" in nd:
                IL[i] = True; LV[i] = float(nd["leaf_value"])
            else:
                assert nd.get("comparison_op", "<=") == "<=", nd.get("comparison_op")
                assert not nd.get("categories_list"), "categorical splits unsupported"
                F[i] = int(nd["split_feature_id"]); T[i] = float(nd["threshold"])
                L[i] = int(nd["left_child"]); R[i] = int(nd["right_child"])
                DL[i] = bool(nd.get("default_left", True))
        feat.append(F); thr.append(T); left.append(L); right.append(R)
        leaf.append(LV); isleaf.append(IL); dleft.append(DL); offsets.append(offsets[-1] + nn)
    return {
        "feat": np.concatenate(feat), "thr": np.concatenate(thr),
        "left": np.concatenate(left), "right": np.concatenate(right),
        "leaf": np.concatenate(leaf), "isleaf": np.concatenate(isleaf),
        "dleft": np.concatenate(dleft), "offsets": np.asarray(offsets, np.int64),
        "num_feature": np.int64(nf), "num_tree": np.int64(len(J["trees"])),
        "base_score": np.float64(base),
    }


def save_forest(arrs: dict, path) -> None:
    np.savez_compressed(path, **arrs)


def load_forest(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def forest_predict(X: np.ndarray, arrs: dict, chunk: int = 200_000) -> np.ndarray:
    """Numpy-only forest prediction. X: [N, num_feature]; returns [N] float64.

    Per tree, traverse all rows in lock-step (vectorized over rows). Output = base_score +
    mean over trees of the reached leaf value (average_tree_output=True).
    """
    X = np.asarray(X, np.float32)
    feat, thr = arrs["feat"], arrs["thr"]
    left, right = arrs["left"], arrs["right"]
    leaf, isleaf, dleft = arrs["leaf"], arrs["isleaf"], arrs["dleft"]
    off = arrs["offsets"]; NT = int(arrs["num_tree"]); base = float(arrs["base_score"])
    N = X.shape[0]
    out = np.empty(N, np.float64)
    for c0 in range(0, N, chunk):
        Xc = X[c0:c0 + chunk]; n = Xc.shape[0]
        acc = np.zeros(n, np.float64)
        for ti in range(NT):
            s, e = int(off[ti]), int(off[ti + 1])
            f_, t_, l_, r_ = feat[s:e], thr[s:e], left[s:e], right[s:e]
            il_, dl_, lv_ = isleaf[s:e], dleft[s:e], leaf[s:e]
            node = np.zeros(n, np.int32)
            active = ~il_[node]
            while active.any():
                idx = np.nonzero(active)[0]
                nd = node[idx]
                xv = Xc[idx, f_[nd]]
                go_left = (xv <= t_[nd]) | (np.isnan(xv) & dl_[nd])
                node[idx] = np.where(go_left, l_[nd], r_[nd])
                active = ~il_[node]
            acc += lv_[node]
        out[c0:c0 + chunk] = base + acc / NT
    return out


__all__ = ["extract_forest", "save_forest", "load_forest", "forest_predict"]
