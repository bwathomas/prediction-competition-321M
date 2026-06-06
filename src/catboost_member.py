"""Package-free CatBoost member: trained with CatBoost offline, evaluated
with pure numpy at runtime (no ``import catboost`` in ``model.py``).

CatBoost uses **oblivious (symmetric) trees**: every node at depth ``k``
of a tree applies the SAME split ``(feature_k, border_k)``. A tree of
depth ``D`` is therefore fully described by ``D`` splits plus a flat table
of ``2**D`` leaf values. Evaluation computes a ``D``-bit leaf index from
the split outcomes and gathers the leaf.

Three CatBoost-specific gotchas this module handles:

1. **Float split is ``x > border`` -> bit 1** (not ``<=``).

2. **Leaf-index bit order is version-dependent.** Whether split ``k`` is
   the LSB or MSB of the leaf index differs across CatBoost serialization
   versions. We AUTO-DETECT the order at compile time by trying both and
   keeping whichever reproduces the model's ``RawFormulaVal`` on anchor
   rows, then store the chosen order in ``meta.json``.

3. **scale_and_bias.** CatBoost's raw score is ``scale * sum(leaf) +
   bias``. We recover ``scale`` and ``bias`` by least squares against
   ``predict(..., 'RawFormulaVal')`` on anchors and assert a tiny
   residual.

NaN routing follows each float feature's ``nan_mode`` (Min -> NaN is the
smallest value so ``x > border`` is False -> bit 0; Max -> bit 1).

Runtime output (Logloss / CrossEntropy objective) is
``sigmoid(scale * sum_leaves + bias)``.

Runtime contract:
``apply_one(state, feats) -> float``  (clamped to (eps, 1-eps))
``apply_batch(state, feats_matrix) -> np.ndarray[N] float32``
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger("catboost_member")

_EPS = 1.0e-6


@dataclass
class CatBoostMemberState:
    """Fitted-and-shipped state of a CatBoost member (oblivious trees).

    Ragged per-tree layout via offsets:
      ``split_feature`` / ``split_border`` / ``split_nan_max`` are
      concatenated over trees; tree ``t``'s splits are the slice
      ``[split_offsets[t] : split_offsets[t+1]]`` (its depth ``D_t``).
      ``leaf_values`` concatenated; tree ``t``'s ``2**D_t`` leaves are
      ``[leaf_offsets[t] : leaf_offsets[t+1]]``.
    """

    split_feature: np.ndarray     # int32 [total_splits]  (column index into x)
    split_border: np.ndarray      # float64 [total_splits]
    split_nan_max: np.ndarray     # bool [total_splits]  (NaN -> bit 1 if True)
    split_offsets: np.ndarray     # int32 [n_trees + 1]
    leaf_values: np.ndarray       # float64 [total_leaves]
    leaf_offsets: np.ndarray      # int32 [n_trees + 1]

    feature_dim: int
    feature_names: tuple[str, ...]
    scale: float
    bias: float
    bit_order: str                # "lsb" (split0 = LSB) or "msb"
    fit_method: str               # "catboost"
    n_train: int
    n_pos: int
    n_trees: int
    train_loss: float
    val_loss: float
    objective: str = "binary"
    output_mode: str = "probability"  # sigmoid(scale*sum_leaves + bias)

    def __post_init__(self) -> None:
        if int(self.split_offsets.shape[0]) != int(self.n_trees) + 1:
            raise ValueError(
                f"split_offsets len {self.split_offsets.shape[0]} != n_trees+1"
            )
        if int(self.leaf_offsets.shape[0]) != int(self.n_trees) + 1:
            raise ValueError(
                f"leaf_offsets len {self.leaf_offsets.shape[0]} != n_trees+1"
            )
        if int(self.split_offsets[-1]) != int(self.split_feature.shape[0]):
            raise ValueError("split_offsets[-1] != total_splits")
        if int(self.leaf_offsets[-1]) != int(self.leaf_values.shape[0]):
            raise ValueError("leaf_offsets[-1] != total_leaves")
        # Each tree: n_leaves == 2 ** n_splits.
        for t in range(int(self.n_trees)):
            d = int(self.split_offsets[t + 1] - self.split_offsets[t])
            nl = int(self.leaf_offsets[t + 1] - self.leaf_offsets[t])
            if nl != (1 << d):
                raise ValueError(
                    f"tree {t}: n_leaves {nl} != 2**depth {1 << d} (depth {d})"
                )
        if self.bit_order not in ("lsb", "msb"):
            raise ValueError(f"bit_order must be lsb/msb, got {self.bit_order!r}")
        if int(len(self.feature_names)) != int(self.feature_dim):
            raise ValueError("feature_names len != feature_dim")
        if not (math.isfinite(self.scale) and math.isfinite(self.bias)):
            raise ValueError("scale/bias is NaN/Inf")

    @property
    def total_splits(self) -> int:
        return int(self.split_feature.shape[0])

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "trees.npz",
            split_feature=self.split_feature.astype(np.int32),
            split_border=self.split_border.astype(np.float64),
            split_nan_max=self.split_nan_max.astype(np.bool_),
            split_offsets=self.split_offsets.astype(np.int32),
            leaf_values=self.leaf_values.astype(np.float64),
            leaf_offsets=self.leaf_offsets.astype(np.int32),
        )
        meta = {
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "scale": float(self.scale),
            "bias": float(self.bias),
            "bit_order": str(self.bit_order),
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
    def load(cls, in_dir: Path | str) -> "CatBoostMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "trees.npz") as npz:
            return cls(
                split_feature=npz["split_feature"].astype(np.int32, copy=False),
                split_border=npz["split_border"].astype(np.float64, copy=False),
                split_nan_max=npz["split_nan_max"].astype(np.bool_, copy=False),
                split_offsets=npz["split_offsets"].astype(np.int32, copy=False),
                leaf_values=npz["leaf_values"].astype(np.float64, copy=False),
                leaf_offsets=npz["leaf_offsets"].astype(np.int32, copy=False),
                feature_dim=int(meta["feature_dim"]),
                feature_names=tuple(meta["feature_names"]),
                scale=float(meta["scale"]),
                bias=float(meta["bias"]),
                bit_order=str(meta["bit_order"]),
                fit_method=str(meta.get("fit_method", "catboost")),
                n_train=int(meta.get("n_train", 0)),
                n_pos=int(meta.get("n_pos", 0)),
                n_trees=int(meta.get("n_trees", 0)),
                train_loss=float(meta.get("train_loss", 0.0)),
                val_loss=float(meta.get("val_loss", 0.0)),
                objective=str(meta.get("objective", "binary")),
                output_mode=str(meta.get("output_mode", "probability")),
            )


# ---------------------------------------------------------------------------
# Pure-numpy inference (oblivious trees)
# ---------------------------------------------------------------------------


def _leaf_index_batch(state: CatBoostMemberState, tree_idx: int, fm: np.ndarray) -> np.ndarray:
    """Compute the [B] leaf indices for one oblivious tree over all rows."""
    B = int(fm.shape[0])
    s0 = int(state.split_offsets[tree_idx])
    s1 = int(state.split_offsets[tree_idx + 1])
    D = s1 - s0
    feat_dim = int(fm.shape[1])
    idx = np.zeros(B, dtype=np.int64)
    for k in range(D):
        sk = s0 + k
        f = int(state.split_feature[sk])
        border = float(state.split_border[sk])
        nan_max = bool(state.split_nan_max[sk])
        if 0 <= f < feat_dim:
            x = fm[:, f]
        else:
            x = np.full(B, np.nan)
        finite = np.isfinite(x)
        bit = np.where(finite, x > border, nan_max).astype(np.int64)
        pos = k if state.bit_order == "lsb" else (D - 1 - k)
        idx += bit << pos
    return idx


def predict_raw(state: CatBoostMemberState, features_matrix: np.ndarray) -> np.ndarray:
    """Vectorized raw score: ``scale * sum_trees(leaf) + bias``. float64 [N]."""
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
        l0 = int(state.leaf_offsets[t])
        l1 = int(state.leaf_offsets[t + 1])
        leaves = state.leaf_values[l0:l1]
        idx = _leaf_index_batch(state, t, fm)
        acc += leaves[idx]
    return float(state.scale) * acc + float(state.bias)


def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(z), z, 0.0)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def apply_batch(state: CatBoostMemberState, features_matrix: np.ndarray) -> np.ndarray:
    raw = predict_raw(state, features_matrix)
    if raw.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    return np.clip(_sigmoid_vec(raw), _EPS, 1.0 - _EPS).astype(np.float32, copy=False)


def apply_one(state: CatBoostMemberState, features: np.ndarray) -> float:
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim {state.feature_dim}"
        )
    p = apply_batch(state, features.reshape(1, -1))
    return float(p[0])


# ---------------------------------------------------------------------------
# Compilation: CatBoost JSON model -> ragged numpy arrays
# ---------------------------------------------------------------------------


def _load_model_json(model: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cb.json"
        model.save_model(str(p), format="json")
        return json.loads(p.read_text(encoding="utf-8"))


def _float_index_map(mj: dict[str, Any]) -> dict[int, tuple[int, bool]]:
    """Map CatBoost ``float_feature_index`` -> (column index in x, nan_is_max).

    ``features_info.float_features`` entries carry ``feature_index`` (the
    float-feature ordinal used by splits) and ``flat_feature_index`` (the
    position in the original feature vector). ``nan_value_treatment`` is
    "AsIs"/"AsFalse"/"AsTrue" or ``nan_mode`` Min/Max depending on version.
    """
    fi = (mj.get("features_info") or {})
    floats = fi.get("float_features") or []
    out: dict[int, tuple[int, bool]] = {}
    for ent in floats:
        f_ord = int(ent.get("feature_index", ent.get("flat_feature_index", 0)))
        flat = int(ent.get("flat_feature_index", ent.get("feature_index", f_ord)))
        nan_treat = str(
            ent.get("nan_value_treatment", ent.get("nan_mode", "Min"))
        ).lower()
        nan_is_max = nan_treat in ("astrue", "max")
        out[f_ord] = (flat, nan_is_max)
    return out


def _extract_trees(mj: dict[str, Any], fmap: dict[int, tuple[int, bool]]) -> tuple[
    list[list[tuple[int, float, bool]]], list[np.ndarray]
]:
    """Return (per_tree_splits, per_tree_leaves).

    ``per_tree_splits[t]`` is a list of ``(column_index, border, nan_is_max)``
    in CatBoost split order. ``per_tree_leaves[t]`` is a ``2**D`` float array.
    """
    trees = mj.get("oblivious_trees")
    if trees is None:
        raise ValueError(
            "CatBoost JSON has no 'oblivious_trees'. Only symmetric (oblivious) "
            "trees are supported; train with grow_policy='SymmetricTree' (default)."
        )
    per_splits: list[list[tuple[int, float, bool]]] = []
    per_leaves: list[np.ndarray] = []
    for tree in trees:
        splits_raw = tree.get("splits", [])
        splits: list[tuple[int, float, bool]] = []
        for sp in splits_raw:
            stype = str(sp.get("split_type", "FloatFeature"))
            if stype != "FloatFeature":
                raise ValueError(
                    f"Unsupported CatBoost split_type {stype!r}. One-hot all "
                    "categoricals upstream so only FloatFeature splits appear."
                )
            f_ord = int(sp["float_feature_index"])
            border = float(sp["border"])
            col, nan_is_max = fmap.get(f_ord, (f_ord, False))
            splits.append((col, border, nan_is_max))
        leaves = np.asarray(tree.get("leaf_values", []), dtype=np.float64).reshape(-1)
        if leaves.shape[0] != (1 << len(splits)):
            raise ValueError(
                f"tree has {leaves.shape[0]} leaves != 2**{len(splits)} "
                f"= {1 << len(splits)}"
            )
        per_splits.append(splits)
        per_leaves.append(leaves)
    return per_splits, per_leaves


def _build_state_arrays(
    per_splits: list[list[tuple[int, float, bool]]],
    per_leaves: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sfeat, sbord, snan = [], [], []
    soff = [0]
    loff = [0]
    leaves_cat = []
    rs = 0
    rl = 0
    for splits, leaves in zip(per_splits, per_leaves):
        for col, border, nan_is_max in splits:
            sfeat.append(int(col))
            sbord.append(float(border))
            snan.append(bool(nan_is_max))
        rs += len(splits)
        soff.append(rs)
        leaves_cat.append(leaves)
        rl += int(leaves.shape[0])
        loff.append(rl)
    return (
        np.asarray(sfeat, dtype=np.int32),
        np.asarray(sbord, dtype=np.float64),
        np.asarray(snan, dtype=np.bool_),
        np.asarray(soff, dtype=np.int32),
        np.concatenate(leaves_cat) if leaves_cat else np.zeros(0, np.float64),
        np.asarray(loff, dtype=np.int32),
    )


def compile_catboost(
    model: Any,
    *,
    feature_names: Sequence[str],
    feature_dim: int,
    anchor_X: np.ndarray,
    parity_atol: float = 1.0e-5,
    n_train: int = 0,
    n_pos: int = 0,
) -> CatBoostMemberState:
    """Compile a fitted CatBoost classifier/regressor (oblivious trees) to a
    parity-verified numpy state. Auto-detects leaf-index bit order and
    recovers scale/bias from ``RawFormulaVal``."""
    mj = _load_model_json(model)
    fmap = _float_index_map(mj)
    per_splits, per_leaves = _extract_trees(mj, fmap)
    sfeat, sbord, snan, soff, leaves_cat, loff = _build_state_arrays(per_splits, per_leaves)
    n_trees = len(per_splits)

    anchor = np.ascontiguousarray(anchor_X, dtype=np.float64)
    raw_cb = np.asarray(
        model.predict(anchor, prediction_type="RawFormulaVal"), dtype=np.float64
    ).reshape(-1)

    def _sum_leaves(bit_order: str) -> np.ndarray:
        st = CatBoostMemberState(
            split_feature=sfeat, split_border=sbord, split_nan_max=snan,
            split_offsets=soff, leaf_values=leaves_cat, leaf_offsets=loff,
            feature_dim=int(feature_dim), feature_names=tuple(str(s) for s in feature_names),
            scale=1.0, bias=0.0, bit_order=bit_order, fit_method="catboost",
            n_train=int(n_train), n_pos=int(n_pos), n_trees=n_trees,
            train_loss=0.0, val_loss=0.0,
        )
        acc = np.zeros(anchor.shape[0], dtype=np.float64)
        for t in range(n_trees):
            l0, l1 = int(loff[t]), int(loff[t + 1])
            idx = _leaf_index_batch(st, t, anchor)
            acc += leaves_cat[l0:l1][idx]
        return acc

    # Auto-detect bit order + recover (scale, bias) by least squares:
    #   raw_cb ~= scale * sum_leaves + bias
    best = None
    for order in ("lsb", "msb"):
        s = _sum_leaves(order)
        A = np.column_stack([s, np.ones_like(s)])
        coef, *_ = np.linalg.lstsq(A, raw_cb, rcond=None)
        scale, bias = float(coef[0]), float(coef[1])
        resid = float(np.max(np.abs(scale * s + bias - raw_cb)))
        if best is None or resid < best[0]:
            best = (resid, order, scale, bias)
    assert best is not None
    resid, bit_order, scale, bias = best
    if resid > max(1.0e-4, float(parity_atol)):
        raise RuntimeError(
            f"CatBoost scale/bias recovery failed: residual {resid} for both bit "
            "orders. Tree extraction or float-feature index mapping is wrong."
        )

    state = CatBoostMemberState(
        split_feature=sfeat, split_border=sbord, split_nan_max=snan,
        split_offsets=soff, leaf_values=leaves_cat, leaf_offsets=loff,
        feature_dim=int(feature_dim), feature_names=tuple(str(s) for s in feature_names),
        scale=scale, bias=bias, bit_order=bit_order, fit_method="catboost",
        n_train=int(n_train), n_pos=int(n_pos), n_trees=n_trees,
        train_loss=0.0, val_loss=0.0, objective="binary", output_mode="probability",
    )

    # ---- Parity (FAIL-FAST): raw + probability ----
    raw_np = predict_raw(state, anchor)
    max_abs_raw = float(np.max(np.abs(raw_np - raw_cb)))
    if max_abs_raw > float(parity_atol):
        raise RuntimeError(
            f"CatBoost raw parity failed: max abs error {max_abs_raw} > {parity_atol} "
            f"(bit_order={bit_order})."
        )
    # Probability parity, if the model exposes predict_proba (classifier).
    if hasattr(model, "predict_proba"):
        p_np = apply_batch(state, anchor)
        proba = np.asarray(model.predict_proba(anchor), dtype=np.float64)
        p_cb = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
        max_abs_prob = float(np.max(np.abs(p_np - p_cb)))
        if max_abs_prob > max(1.0e-4, float(parity_atol)):
            raise RuntimeError(
                f"CatBoost probability parity failed: max abs error {max_abs_prob} "
                f"> {parity_atol}."
            )
    else:
        max_abs_prob = float("nan")

    LOG.info(
        "CatBoost compile OK: n_trees=%d total_splits=%d feature_dim=%d "
        "bit_order=%s scale=%.5f bias=%.5f parity_raw=%.2e parity_prob=%.2e",
        state.n_trees, state.total_splits, state.feature_dim, bit_order,
        scale, bias, max_abs_raw, max_abs_prob,
    )
    return state


def fit_catboost_member(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    classifier: bool = True,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    seed: int = 0,
    parity_atol: float = 1.0e-5,
    val_fraction: float = 0.1,
    num_threads: int | None = None,
) -> CatBoostMemberState:
    """Train a CatBoost member (Logloss/CrossEntropy, symmetric trees) on
    (X, y in [0,1]) and compile to a parity-verified numpy state.

    For soft labels CatBoost's ``CrossEntropy`` loss needs a *Classifier*
    (it accepts targets in [0,1]); ``classifier=True`` uses that. Mirrors
    the AIDE roster's ``cat`` member."""
    from catboost import CatBoostClassifier, CatBoostRegressor  # offline only

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
        learning_rate=float(learning_rate),
        depth=int(depth),
        l2_leaf_reg=float(l2_leaf_reg),
        random_seed=int(seed),
        grow_policy="SymmetricTree",
        verbose=False,
        allow_writing_files=False,
    )
    if num_threads is not None:
        common["thread_count"] = int(num_threads)
    if classifier:
        model = CatBoostClassifier(loss_function="CrossEntropy", **common)
        model.fit(Xtr, y[train_idx].astype(np.float64))
    else:
        model = CatBoostRegressor(loss_function="RMSE", **common)
        model.fit(Xtr, y[train_idx].astype(np.float64))

    state = compile_catboost(
        model,
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
    "CatBoostMemberState",
    "apply_one",
    "apply_batch",
    "predict_raw",
    "compile_catboost",
    "fit_catboost_member",
]
