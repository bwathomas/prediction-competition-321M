"""Architecture registry + the one-fold smoke gate every architecture must pass before
AIDE may compose it (finite NLL, correct OOF shape, no exception, leakage probes green
via the harness)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..harness.eval import evaluate
from .architectures import LogisticArchitecture, MLPArchitecture, lazy_lightgbm
from .linear_stacker import LinearStacker

# name -> builder(**kw) -> model. Pure-numpy entries train anywhere; lazy_* entries
# raise a clear RuntimeError off-Colab (see architectures._lazy).
REGISTRY = {
    "logistic": lambda **kw: LogisticArchitecture(**kw),
    "mlp": lambda **kw: MLPArchitecture(**kw),
    "linear_stacker": lambda **kw: LinearStacker(**kw),
    "gbdt_lightgbm": lazy_lightgbm,
}


def get(name: str, **kw):
    """Return a zero-arg factory for the named architecture with bound hyperparameters."""
    if name not in REGISTRY:
        raise KeyError(f"unknown architecture {name!r}; known: {sorted(REGISTRY)}")
    builder = REGISTRY[name]
    return lambda: builder(**kw)


@dataclass
class SmokeResult:
    ok: bool
    nll: float
    auc: object
    error: object  # str on failure, else None


def smoke_test(model_factory, ds, manifest, *, dropout=None, neutral_prefixes=None) -> SmokeResult:
    """Run one OOF evaluation and verify a finite NLL + correct OOF shape. Any exception
    (including a missing heavy library) is captured into ok=False."""
    try:
        res = evaluate(model_factory, ds, manifest, dropout=dropout,
                       neutral_prefixes=neutral_prefixes)
        in_range = bool(np.nanmin(res.oof) >= 0.0 and np.nanmax(res.oof) <= 1.0)
        ok = bool(math.isfinite(res.nll) and res.oof.shape == (len(ds.y),) and in_range)
        return SmokeResult(ok=ok, nll=res.nll, auc=res.auc, error=None)
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(ok=False, nll=float("nan"), auc=None, error=repr(exc))
