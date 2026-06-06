"""FeatRoute member: a nested 8-group LightGBM ensemble blended by
logit-mean, whose RUNTIME inference is pure numpy.

``featroute`` is an ensemble of eight :mod:`src.gbdt_member` sub-models,
one per feature-group slice of the shared member-feature matrix. Each
sub-model is an ordinary binary GBDT trained on ``X[:, group_slices[g]]``;
the member's prediction is the **logit-mean** over the per-group
predictions:

    p = sigmoid( (1 / G) * sum_g  logit( gbdt_g(x_g) ) )

where ``logit(gbdt_g(x_g))`` is exactly the sub-model's raw score
(``gbdt_member.predict_raw`` = sum-of-leaves + bias) for a binary GBDT,
since ``apply_*`` finishes with a plain ``sigmoid`` of that raw score.
Working in raw/logit space (instead of clipping each group to a
probability and re-logit-ing) avoids the eps round-trip and is the
honest logit-mean.

The eight groups (locked order, matching the AIDE
``FoldFeatureStore.assemble`` group split):

    nn_label_derivatives, cluster_passrate, cluster_subject,
    counts_subject, centroid_distance, cluster_geometry, nn_geometry,
    item_cluster

Rationale for routing instead of one fat GBDT: each group is a
semantically coherent block; a per-group tree cannot leak signal across
blocks, and the logit-mean blend is a fixed (un-trained) combiner that
adds no extra parameters to overfit. It is also a cheap diversity source
for the layer-2 stacker.

Design decisions
----------------

1. **Reuse :mod:`src.gbdt_member` verbatim.** Every sub-model is a
   ``GBDTMemberState`` produced by ``fit_gbdt_member`` and traversed by
   the existing pure-numpy walker (``predict_raw``). No new tree code.

2. **Logit space = the sub-model raw score.** For a binary GBDT,
   ``apply_batch == sigmoid(predict_raw)``; hence
   ``logit(p_group) == predict_raw`` exactly. ``predict_raw`` is the
   group "logit" used in the mean -- no clipping, no information loss.

3. **Missing / empty groups are skipped.** A group whose slice selects
   zero columns (``slice`` with no width) or whose sub-model is ``None``
   contributes nothing to the mean, and the average is taken over the
   groups that *are* present. If NO group is present the member returns
   ``0.5`` (logit 0). This keeps the member robust to a family whose
   feature schema lacks one of the canonical blocks.

Runtime contract (pure numpy, no ``import lightgbm`` at runtime)
----------------------------------------------------------------
``apply_one(state, x) -> float``        clamped to (eps, 1-eps)
``apply_batch(state, X) -> np.ndarray`` float32 ``[N]``
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.gbdt_member import GBDTMemberState, predict_raw

LOG = logging.getLogger("featroute_member")

_EPS = 1.0e-6

# Locked group order. The adversary (the stacker) consumes the member's
# single scalar output, so this order is internal -- but pinning it makes
# save/load deterministic and keeps the logit-mean reproducible.
FEATROUTE_GROUP_NAMES: tuple[str, ...] = (
    "nn_label_derivatives",
    "cluster_passrate",
    "cluster_subject",
    "counts_subject",
    "centroid_distance",
    "cluster_geometry",
    "nn_geometry",
    "item_cluster",
)


# ---------------------------------------------------------------------------
# Slice (de)serialization helpers
# ---------------------------------------------------------------------------


def _slice_to_list(sl: slice) -> list[int | None]:
    """Serialize a ``slice`` to a JSON-friendly ``[start, stop, step]``."""
    return [sl.start, sl.stop, sl.step]


def _slice_from_list(triple: Sequence[int | None]) -> slice:
    start, stop, step = triple
    return slice(start, stop, step)


def _slice_width(sl: slice, n_cols: int) -> int:
    """Number of columns ``X[:, sl]`` selects from an ``n_cols``-wide X."""
    return len(range(*sl.indices(int(n_cols))))


# ---------------------------------------------------------------------------
# State (the shipped artifact)
# ---------------------------------------------------------------------------


@dataclass
class FeatRouteState:
    """Fitted-and-shipped state of the FeatRoute member.

    ``sub_states[i]`` is the GBDT for group ``group_names[i]`` over the
    columns ``X[:, group_slices[group_names[i]]]``. A ``None`` sub-state
    marks a group that was absent / empty at fit time; it is skipped in
    the logit-mean.

    ``feature_dim`` is the width of the FULL shared feature matrix the
    member expects at inference (so ``apply_*`` can validate input and
    slice it the same way training did).
    """

    sub_states: list[GBDTMemberState | None]
    group_slices: dict[str, slice]
    group_names: list[str]
    feature_dim: int
    fit_method: str = "featroute-gbdt-logit-mean"

    def __post_init__(self) -> None:
        if len(self.sub_states) != len(self.group_names):
            raise ValueError(
                f"sub_states len {len(self.sub_states)} != group_names len "
                f"{len(self.group_names)}"
            )
        for g in self.group_names:
            if g not in self.group_slices:
                raise ValueError(f"group {g!r} missing from group_slices")
        # Every present sub-model's feature_dim must equal its slice width
        # over the full matrix -- otherwise apply_* would feed it the wrong
        # column count.
        for name, sub in zip(self.group_names, self.sub_states):
            if sub is None:
                continue
            w = _slice_width(self.group_slices[name], self.feature_dim)
            if int(sub.feature_dim) != int(w):
                raise ValueError(
                    f"group {name!r}: sub-model feature_dim {sub.feature_dim} "
                    f"!= slice width {w} (feature_dim={self.feature_dim})"
                )

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def n_present_groups(self) -> int:
        return sum(1 for s in self.sub_states if s is not None)

    # ---- I/O ----

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        present: list[bool] = []
        for i, (name, sub) in enumerate(zip(self.group_names, self.sub_states)):
            if sub is None:
                present.append(False)
                continue
            present.append(True)
            # Each sub-model lands in its own deterministic subdir; the
            # GBDTMemberState owns its trees.npz + meta.json there.
            sub.save(out / f"group_{i:02d}_{name}")
        meta = {
            "group_names": list(self.group_names),
            "group_slices": {
                g: _slice_to_list(self.group_slices[g]) for g in self.group_names
            },
            "present": present,
            "feature_dim": int(self.feature_dim),
            "fit_method": str(self.fit_method),
            "format_version": 1,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "FeatRouteState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        group_names = list(meta["group_names"])
        group_slices = {
            g: _slice_from_list(meta["group_slices"][g]) for g in group_names
        }
        present = list(meta["present"])
        sub_states: list[GBDTMemberState | None] = []
        for i, (name, is_present) in enumerate(zip(group_names, present)):
            if not is_present:
                sub_states.append(None)
                continue
            sub_states.append(
                GBDTMemberState.load(d / f"group_{i:02d}_{name}")
            )
        return cls(
            sub_states=sub_states,
            group_slices=group_slices,
            group_names=group_names,
            feature_dim=int(meta["feature_dim"]),
            fit_method=str(meta.get("fit_method", "featroute-gbdt-logit-mean")),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _sigmoid_stable_one(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _group_logit_batch(
    state: FeatRouteState, features_matrix: np.ndarray
) -> np.ndarray:
    """Mean group logit per row, ``[N]`` float64.

    For each present group ``g``, slice the shared matrix to that group's
    columns and take ``predict_raw`` (the sub-GBDT's logit). Average over
    the present groups. Rows with no present group get logit 0.0.
    """
    N = int(features_matrix.shape[0])
    fm = np.ascontiguousarray(features_matrix, dtype=np.float64)
    acc = np.zeros(N, dtype=np.float64)
    n_present = 0
    for name, sub in zip(state.group_names, state.sub_states):
        if sub is None:
            continue
        sl = state.group_slices[name]
        Xg = fm[:, sl]
        # Defensive: a degenerate empty slice contributes nothing.
        if int(Xg.shape[1]) == 0:
            continue
        acc += predict_raw(sub, Xg)
        n_present += 1
    if n_present == 0:
        # No usable group -> neutral logit (p = 0.5).
        return np.zeros(N, dtype=np.float64)
    return acc / float(n_present)


def apply_batch(
    state: FeatRouteState, features_matrix: np.ndarray
) -> np.ndarray:
    """Vectorized inference -> float32 probabilities ``[N]``.

    ``features_matrix`` is the FULL shared feature matrix ``[N, feature_dim]``;
    each group's sub-model is fed its own column slice. The blend is the
    logit-mean over present groups, finished with a stable sigmoid + clamp.
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

    z = _group_logit_batch(state, features_matrix)
    z = np.where(np.isfinite(z), z, 0.0)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    out = np.clip(out, _EPS, 1.0 - _EPS)
    return out.astype(np.float32, copy=False)


def apply_one(state: FeatRouteState, features: np.ndarray) -> float:
    """Single-row inference -> Python ``float`` in (eps, 1-eps).

    Numerically identical to ``apply_batch`` on a 1-row matrix: the same
    per-group ``predict_raw`` over the same column slices, the same mean,
    the same sigmoid + clamp.
    """
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got shape {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim "
            f"{state.feature_dim}"
        )
    z = float(_group_logit_batch(state, features.reshape(1, -1))[0])
    if not math.isfinite(z):
        return 0.5
    p = _sigmoid_stable_one(z)
    return float(min(max(p, _EPS), 1.0 - _EPS))


# ---------------------------------------------------------------------------
# Offline trainer
# ---------------------------------------------------------------------------


def fit_featroute(
    *,
    X: np.ndarray,                          # [N, F] float32
    y: np.ndarray,                          # [N] float32 in [0, 1]
    group_slices: Mapping[str, slice],
    fold_train_idx: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    group_names: Sequence[str] | None = None,
    seed: int = 0,
    **gbdt_kwargs,
) -> FeatRouteState:
    """Train one GBDT per feature group and wrap them as a FeatRoute.

    Parameters
    ----------
    X, y:
        Full shared feature matrix and (soft) labels. Each sub-model
        ``g`` is trained on ``X[:, group_slices[g]]``.
    group_slices:
        Mapping ``group_name -> slice`` into ``X``'s columns. Must cover
        every name in ``group_names``.
    fold_train_idx:
        Optional row indices to train on (the rest are held out by the
        caller's fold logic). When ``None``, all rows are used. This is
        the OOF hook: the per-fold caller passes ``fold.train_item_keys``
        rows so the sub-models never see the held-out fold.
    sample_weight:
        Optional per-row weights (sliced by ``fold_train_idx`` alongside
        ``X``/``y``).
    group_names:
        Order of groups (defaults to :data:`FEATROUTE_GROUP_NAMES`,
        filtered to those present in ``group_slices``). A group whose
        slice selects zero columns is recorded as absent (``None``
        sub-model) and skipped in the logit-mean.
    seed:
        Base RNG seed; group ``i`` uses ``seed + i`` so the sub-models
        are not identically-seeded.
    **gbdt_kwargs:
        Forwarded to :func:`src.gbdt_member.fit_gbdt_member` for EVERY
        sub-model (e.g. ``n_estimators``, ``learning_rate``,
        ``num_leaves``, ``holdout_group_id``).

    Returns
    -------
    FeatRouteState
    """
    from src.gbdt_member import fit_gbdt_member  # offline-only entry

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.shape != (int(X.shape[0]),):
        raise ValueError(f"y shape {y.shape} != ({X.shape[0]},)")

    full_dim = int(X.shape[1])
    if group_names is None:
        names = [g for g in FEATROUTE_GROUP_NAMES if g in group_slices]
    else:
        names = list(group_names)
    for g in names:
        if g not in group_slices:
            raise ValueError(f"group {g!r} missing from group_slices")

    # Subset rows for this fold if requested.
    if fold_train_idx is not None:
        idx = np.asarray(fold_train_idx).reshape(-1)
        X_fit = X[idx]
        y_fit = y[idx]
        w_fit = sample_weight[idx] if sample_weight is not None else None
        holdout_gid = gbdt_kwargs.get("holdout_group_id", None)
        if holdout_gid is not None:
            gbdt_kwargs = dict(gbdt_kwargs)
            gbdt_kwargs["holdout_group_id"] = np.asarray(holdout_gid)[idx]
    else:
        X_fit = X
        y_fit = y
        w_fit = sample_weight

    sub_states: list[GBDTMemberState | None] = []
    kept_slices: dict[str, slice] = {}
    for i, name in enumerate(names):
        sl = group_slices[name]
        kept_slices[name] = sl
        width = _slice_width(sl, full_dim)
        if width == 0:
            # Empty / missing group: record absence, skip training.
            LOG.info("featroute: group %r selects 0 columns; skipping", name)
            sub_states.append(None)
            continue
        Xg = np.ascontiguousarray(X_fit[:, sl], dtype=np.float32)
        feat_names = tuple(f"{name}__{j}" for j in range(width))
        sub = fit_gbdt_member(
            X=Xg,
            y=y_fit,
            feature_names=feat_names,
            sample_weights=w_fit,
            seed=int(seed) + i,
            **gbdt_kwargs,
        )
        sub_states.append(sub)

    state = FeatRouteState(
        sub_states=sub_states,
        group_slices=kept_slices,
        group_names=list(names),
        feature_dim=full_dim,
    )
    LOG.info(
        "featroute fit OK: %d/%d groups present (feature_dim=%d)",
        state.n_present_groups, state.n_groups, full_dim,
    )
    return state


__all__ = [
    "FeatRouteState",
    "FEATROUTE_GROUP_NAMES",
    "apply_one",
    "apply_batch",
    "fit_featroute",
]
