"""Family-agnostic 3x-bag wrapper for ensemble members (Gap (b) of the
principled-export pipeline).

Every Layer-1 roster member (gbdt / xgb / cat / forest / knn / mlp / fm /
irt / featroute / logreg) is trained as a small *bag* of K independently
seeded sub-models, and the bag prediction is the **logit-space mean** of the
sub-model probabilities. Bagging lowers the variance of each base learner
before it ever reaches the Layer-2 meta, which is the cheapest honest
diversity lever we have (no new feature engineering, no new hyperparameters).

This module is deliberately member-agnostic: it knows nothing about *what*
a member is, only that it satisfies the repo's member contract:

    fit_fn(*, X, y, sample_weight=..., **fit_kwargs) -> member_state
    member_state.save(dir) / type(member_state).load(dir)
    apply_fn(member_state, X) -> np.ndarray of probabilities in (0, 1)

The three public entry points are:

* :func:`train_bagged_member` -- train K seeded sub-models over a fold's
  train rows, optionally bootstrap-resampling rows within the fold.
* :func:`bagged_predict_logit_mean` -- average K sub-model probabilities in
  logit space (numerically clipped) and return probabilities.
* :class:`BaggedMemberState` -- a member-contract-shaped container that holds
  the K sub-states and round-trips through ``save`` / ``load`` by delegating
  to each sub-state's own ``save`` / ``load`` plus a small index json.

Runtime contract
----------------
``BaggedMemberState.load(dir)`` reloads the K sub-states and the apply path
(:meth:`BaggedMemberState.apply_one` / :meth:`apply_batch`) is **pure numpy**:
it calls the per-member apply function on each sub-state and logit-averages.
No torch / sklearn / lightgbm is imported here; the (offline) ``fit_fn`` may
of course use whatever it likes, but nothing in this module does.

Seeding semantics
-----------------
``seeds`` enters the model RNG: each sub-model is fit with ``seed=<s>`` passed
through to ``fit_fn`` (members in this repo accept a ``seed`` kwarg). When
``bag_kind == "bootstrap"`` the *same* seed also drives a row resample (with
replacement) over ``fold_train_idx`` -- so a fixed ``seeds`` tuple yields a
fully deterministic bag, both in the model RNG and in the bootstrap draw.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

LOG = logging.getLogger("bagging")

# Reuse the stacker's logit convention so bag-averaging and the meta agree
# on the same clipped logit transform (single source of truth for eps).
from src.stacker import _EPS, logit_clipped


# ---------------------------------------------------------------------------
# Numpy logit-mean combine (pure runtime path)
# ---------------------------------------------------------------------------


def _sigmoid_stable_arr(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid (never NaN on huge |z|)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit_mean_combine(probs: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Average a stack of probability vectors in **logit space**.

    Parameters
    ----------
    probs
        ``[K, N]`` (or ``[K]`` for a single row) array of probabilities in
        ``(0, 1)`` -- one row per bag sub-model.
    eps
        Clip applied before the logit and to the returned probability, so the
        transform stays finite on saturated inputs (matches
        :func:`src.stacker.logit_clipped`).

    Returns
    -------
    np.ndarray
        ``[N]`` (or scalar-shaped ``[]``) probabilities = sigmoid(mean over K
        of logit(probs)). The geometric mean of odds; order-invariant in K.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        # A single row across K members -> treat as [K, 1].
        arr = arr.reshape(arr.shape[0], 1)
        squeeze = True
    elif arr.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"probs must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("probs must contain at least one bag member (K>=1)")
    logits = logit_clipped(arr, eps=eps)          # [K, N]
    mean_logit = logits.mean(axis=0)              # [N]
    p = _sigmoid_stable_arr(mean_logit)
    p = np.clip(p, eps, 1.0 - eps)
    if squeeze:
        return p.reshape(())  # scalar-shaped
    return p.astype(np.float64)


def bagged_predict_logit_mean(
    states: Sequence[Any],
    apply_fn: Callable[[Any, np.ndarray], np.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    """Logit-space mean of ``apply_fn(state, X)`` over the bag ``states``.

    ``apply_fn`` is the member's batch apply (e.g.
    ``gbdt_member.apply_state_batch``); it must return ``[N]`` probabilities
    in ``(0, 1)`` for each sub-state. The result is the geometric mean of the
    odds across the K sub-models, clipped to ``(eps, 1-eps)``.

    Order-invariant in ``states`` (averaging commutes); deterministic.
    """
    if len(states) < 1:
        raise ValueError("states must contain at least one bag member")
    preds = []
    for st in states:
        p = np.asarray(apply_fn(st, X), dtype=np.float64).reshape(-1)
        preds.append(p)
    n = preds[0].shape[0]
    for i, p in enumerate(preds):
        if p.shape[0] != n:
            raise ValueError(
                f"bag member {i} returned {p.shape[0]} preds, expected {n}"
            )
    stacked = np.stack(preds, axis=0)             # [K, N]
    return logit_mean_combine(stacked).astype(np.float32)


# ---------------------------------------------------------------------------
# Bagged training
# ---------------------------------------------------------------------------


def _bootstrap_indices(
    fold_train_idx: np.ndarray, seed: int
) -> np.ndarray:
    """Sample ``len(fold_train_idx)`` rows WITH replacement from the fold.

    The draw is seeded so a fixed ``seed`` reproduces the exact resample.
    Returns absolute row indices (a resampled view of ``fold_train_idx``).
    """
    rng = np.random.default_rng(int(seed))
    m = int(fold_train_idx.shape[0])
    picks = rng.integers(0, m, size=m)
    return fold_train_idx[picks]


def train_bagged_member(
    fit_fn: Callable[..., Any],
    *,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    fold_train_idx: np.ndarray,
    seeds: Sequence[int] = (0, 1, 2),
    bag_kind: str = "seed_only",
    **fit_kwargs: Any,
) -> list[Any]:
    """Train a bag of K seeded sub-models of one member, return their states.

    Parameters
    ----------
    fit_fn
        The member's trainer, e.g. ``gbdt_member.fit_gbdt_member``. It is
        called as ``fit_fn(X=..., y=..., sample_weight=..., seed=<s>,
        **fit_kwargs)`` and must return a member-state with ``.save`` /
        ``.load`` and an apply function. ``seed`` enters the model RNG.
    X, y
        Full design matrix / labels. ``fold_train_idx`` selects the fold's
        training rows; the member is only ever shown those rows (honest OOF
        discipline -- the held-out fold is predicted, never trained on).
    sample_weight
        Optional ``[N]`` per-row weights, sliced alongside ``X`` / ``y``.
        Forwarded as ``sample_weight=`` to ``fit_fn``.
    fold_train_idx
        Absolute row indices of this fold's training rows.
    seeds
        One seed per bag sub-model. ``len(seeds)`` == bag size K. Each seed
        is forwarded to ``fit_fn`` as ``seed=``.
    bag_kind
        ``"seed_only"``  -- every sub-model sees the same ``fold_train_idx``
                           rows, differing only in the model RNG seed.
        ``"bootstrap"`` -- additionally resample the fold rows WITH
                           replacement (seeded by the same seed) before
                           fitting, for stronger bag diversity.
    **fit_kwargs
        Forwarded verbatim to ``fit_fn`` (hyperparameters, feature names...).
        Must NOT contain ``seed`` or ``sample_weight`` (those are managed
        here) -- a duplicate would raise ``TypeError`` at the ``fit_fn`` call.

    Returns
    -------
    list[member_state]
        One fitted sub-state per seed, in the order of ``seeds``.
    """
    if bag_kind not in ("seed_only", "bootstrap"):
        raise ValueError(
            f"bag_kind must be 'seed_only' or 'bootstrap', got {bag_kind!r}"
        )
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) < 1:
        raise ValueError("seeds must contain at least one seed (K>=1)")
    fold_train_idx = np.asarray(fold_train_idx).reshape(-1)
    if fold_train_idx.shape[0] < 1:
        raise ValueError("fold_train_idx is empty")
    if "seed" in fit_kwargs:
        raise ValueError(
            "do not pass 'seed' in fit_kwargs; train_bagged_member manages "
            "the per-bag seed"
        )

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    sw = None if sample_weight is None else np.asarray(sample_weight).reshape(-1)

    states: list[Any] = []
    for s in seeds:
        if bag_kind == "bootstrap":
            rows = _bootstrap_indices(fold_train_idx, seed=s)
        else:
            rows = fold_train_idx
        kwargs = dict(fit_kwargs)
        kwargs["seed"] = int(s)
        if sw is not None:
            kwargs["sample_weight"] = sw[rows]
        LOG.info(
            "train_bagged_member: fitting bag member seed=%d  bag_kind=%s  "
            "n_rows=%d", s, bag_kind, int(rows.shape[0]),
        )
        state = fit_fn(X=X[rows], y=y[rows], **kwargs)
        states.append(state)
    return states


# ---------------------------------------------------------------------------
# BaggedMemberState (member-contract container)
# ---------------------------------------------------------------------------


def _state_class_path(state: Any) -> str:
    """Fully-qualified ``module:ClassName`` of a member-state instance."""
    cls = type(state)
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_state_class(path: str) -> type:
    """Inverse of :func:`_state_class_path` -- import and return the class."""
    module_name, _, qualname = path.partition(":")
    if not qualname:
        raise ValueError(f"malformed state class path {path!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"{path!r} did not resolve to a class")
    return obj


@dataclass
class BaggedMemberState:
    """A bag of K member sub-states combined by logit-mean.

    Satisfies the member contract (``save`` / ``load``, and pure-numpy
    ``apply_one`` / ``apply_batch``) so the rest of the pipeline can treat a
    bagged member exactly like a single member.

    Attributes
    ----------
    members
        The K fitted sub-states (each a member-state with ``save`` / ``load``).
    apply_fn
        The member's batch apply (``apply_fn(state, X) -> [N] probs``). Held
        on the dataclass so :meth:`apply_batch` is self-contained. It is NOT
        serialized; on :meth:`load` it is re-resolved from the recorded
        member-module's ``apply_state_batch`` (or supplied by the caller).
    combine
        Combine rule; only ``"logit_mean"`` is supported.
    member_class_path
        ``module:ClassName`` of the sub-states, recorded so :meth:`load` can
        reconstruct them without the caller naming the class.
    apply_fn_name
        Name of the member-module attribute used as the batch apply
        (default ``"apply_state_batch"``), recorded for :meth:`load`.
    """

    members: list[Any]
    apply_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None
    combine: str = "logit_mean"
    member_class_path: str = ""
    apply_fn_name: str = "apply_state_batch"
    format_version: int = field(default=1)

    def __post_init__(self) -> None:
        if self.combine != "logit_mean":
            raise ValueError(
                f"only combine='logit_mean' is supported, got {self.combine!r}"
            )
        if len(self.members) < 1:
            raise ValueError("BaggedMemberState requires >= 1 member")
        if not self.member_class_path:
            self.member_class_path = _state_class_path(self.members[0])

    # ---- inference (pure numpy) ----
    def _apply(self) -> Callable[[Any, np.ndarray], np.ndarray]:
        if self.apply_fn is not None:
            return self.apply_fn
        # Re-resolve from the member module (e.g. gbdt_member.apply_state_batch).
        module_name = self.member_class_path.partition(":")[0]
        module = importlib.import_module(module_name)
        fn = getattr(module, self.apply_fn_name, None)
        if fn is None:
            raise AttributeError(
                f"module {module_name!r} has no attribute "
                f"{self.apply_fn_name!r}; pass apply_fn explicitly"
            )
        self.apply_fn = fn
        return fn

    def apply_batch(self, X: np.ndarray) -> np.ndarray:
        """``[N]`` logit-mean probabilities over the bag (float32)."""
        return bagged_predict_logit_mean(self.members, self._apply(), X)

    def apply_one(self, features: np.ndarray) -> float:
        """Single-row logit-mean probability -> Python float in (eps, 1-eps)."""
        feats = np.asarray(features)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        p = self.apply_batch(feats)
        return float(p.reshape(-1)[0])

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        """Save K sub-states under ``member_00/ .. member_KK/`` + index json.

        Each sub-state is saved via its own ``save`` (delegation), and a small
        ``bag_index.json`` records the bag size, combine rule, sub-state class
        path, and apply-fn name so :meth:`load` can round-trip with no caller
        knowledge of the member type.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sub_dirs: list[str] = []
        for i, st in enumerate(self.members):
            name = f"member_{i:02d}"
            st.save(out / name)
            sub_dirs.append(name)
        index = {
            "combine": self.combine,
            "n_members": int(len(self.members)),
            "member_dirs": sub_dirs,
            "member_class_path": self.member_class_path,
            "apply_fn_name": self.apply_fn_name,
            "format_version": int(self.format_version),
        }
        (out / "bag_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(
        cls,
        in_dir: Path | str,
        *,
        member_class: type | None = None,
        apply_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    ) -> "BaggedMemberState":
        """Reload a bag: read the index, ``load`` each sub-state, re-wire apply.

        Parameters
        ----------
        in_dir
            Directory previously written by :meth:`save`.
        member_class
            Override for the sub-state class. By default the class recorded in
            the index (``member_class_path``) is imported and used.
        apply_fn
            Override for the batch apply. By default the member module's
            ``apply_fn_name`` attribute (recorded in the index) is used.
        """
        d = Path(in_dir)
        index = json.loads((d / "bag_index.json").read_text(encoding="utf-8"))
        path = str(index["member_class_path"])
        state_cls = member_class if member_class is not None else _resolve_state_class(path)
        member_dirs = list(index["member_dirs"])
        members = [state_cls.load(d / name) for name in member_dirs]
        return cls(
            members=members,
            apply_fn=apply_fn,
            combine=str(index.get("combine", "logit_mean")),
            member_class_path=path,
            apply_fn_name=str(index.get("apply_fn_name", "apply_state_batch")),
            format_version=int(index.get("format_version", 1)),
        )


__all__ = [
    "BaggedMemberState",
    "train_bagged_member",
    "bagged_predict_logit_mean",
    "logit_mean_combine",
]
