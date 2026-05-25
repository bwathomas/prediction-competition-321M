"""Training-time dropout for metadata + benchmark-id channels.

Two complementary regularizers, both designed to make a trained model
robust to the cold-start regime that the hosted competition will see
(some test rows on benchmarks unseen during training, with no metadata
available for those benchmarks):

* ``p_bench`` -- per training row, with this probability replace the
  row's benchmark metadata (``bc_cat`` + ``bc_num``) with the
  ``__MISSING__`` pattern stored at row 0 of
  ``MetaHybridIRTKFactorGatedMLP.bc_meta_*``. This trains the
  ``__MISSING__`` token embedding and the gated MLP's missingness gate
  on benchmark-cold-start patterns. ONLY meaningful for
  metadata-aware models.

* ``p_subj`` -- analogously masks subject metadata
  (``subj_cat`` + ``subj_num``). Independent of ``p_bench``, so the
  model also sees joint-missing patterns at frequency
  ``p_bench * p_subj``. ONLY meaningful for metadata-aware models.

* ``q_bc`` -- per training row, with this probability replace the
  row's ``bc_idx`` with ``0`` (the UNK benchmark slot). This trains
  the per-bc bias ``beta[0]`` and the residual MLP to predict
  robustly without the per-benchmark intercept signal. Useful for
  *both* metadata-aware AND metadata-less models, but most relevant
  for the no-meta model used in the cold-start ensemble blend.

Implementation
--------------

We use a single ``forward_pre_hook`` registered on the model. The
hook is a no-op when ``module.training is False`` (so eval / val pass
through unchanged). When training, it samples Bernoullis from a
seedable ``torch.Generator`` and:

  1. Replaces the requested fraction of ``bc_idx`` rows with ``0``.
  2. (Meta models only) Builds a per-row ``meta_override`` dict whose
     fields equal the model's normal buffer lookup for unmasked rows
     and the ``__MISSING__`` row 0 for masked rows, then injects it
     as the ``meta_override=...`` keyword argument of the forward
     call. The hybrid (non-meta) variant doesn't accept this kwarg
     and is skipped.

Because the hook is removed by ``handle.remove()`` after training,
the saved ``state_dict`` is bit-identical to a model trained without
the hook, and the hosted runtime never sees the hook.

Usage
-----

>>> from src.train_dropout import install_train_dropout, TrainDropoutConfig
>>> handle = install_train_dropout(
...     model,
...     TrainDropoutConfig(p_bench=0.20, p_subj=0.10, q_bc=0.0, seed=17),
... )
>>> # ... training loop ...
>>> handle.remove()

Or as a context manager::

>>> with install_train_dropout(model, cfg):
...     train_one(...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainDropoutConfig:
    """Per-row Bernoulli probabilities applied at training time only.

    All probabilities default to 0 (no-op). Independent samples per
    row, so the joint-missing rate is the product of the two
    metadata probabilities.
    """

    p_bench: float = 0.0
    p_subj: float = 0.0
    q_bc: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("p_bench", "p_subj", "q_bc"):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"TrainDropoutConfig.{name}={v} must be in [0, 1]"
                )
            setattr(self, name, v)


def _has_meta_buffers(model: torch.nn.Module) -> bool:
    """True iff the model exposes the ``MetaHybrid*`` buffer layout."""
    for name in (
        "bc_meta_cat_ids",
        "bc_meta_num",
        "subject_meta_cat_ids",
        "subject_meta_num",
    ):
        if not hasattr(model, name):
            return False
        if getattr(model, name) is None:
            return False
    return True


def _accepts_meta_override(model: torch.nn.Module) -> bool:
    """True iff ``model.forward`` declares a ``meta_override`` kwarg."""
    import inspect

    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "meta_override" in sig.parameters


class _DropoutPreHook:
    """Closure-as-class so we can introspect / remove cleanly.

    Held as a module attribute; the underlying ``RemovableHandle`` is
    inside ``self._handle`` so callers don't have to track two
    objects.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: TrainDropoutConfig,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.has_meta = _has_meta_buffers(model)
        self.accepts_override = _accepts_meta_override(model)
        self.generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        self.n_calls = 0
        self.n_train_calls = 0
        self.n_rows_seen = 0
        self.n_rows_bench_masked = 0
        self.n_rows_subj_masked = 0
        self.n_rows_bc_idx_masked = 0
        self._handle: Any | None = None

    # -------- pre-hook entry point --------

    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        self.n_calls += 1
        if not module.training:
            return args, kwargs
        self.n_train_calls += 1

        # subject_idx and bc_idx are positional args 0 and 1 of every
        # supported variant (HybridIRTItemKFactorGatedMLP and
        # MetaHybridIRTKFactorGatedMLP). We rebuild ``args`` only when
        # we mutate ``bc_idx``.
        if len(args) < 2:
            return args, kwargs
        subject_idx = args[0]
        bc_idx = args[1]
        if not (
            isinstance(subject_idx, torch.Tensor)
            and isinstance(bc_idx, torch.Tensor)
        ):
            return args, kwargs

        B = int(bc_idx.shape[0])
        if B == 0:
            return args, kwargs
        self.n_rows_seen += B

        new_args = list(args)
        new_kwargs = dict(kwargs)

        # ---- bc-idx dropout ----
        # bc_idx is replaced with 0 with prob q_bc. This affects BOTH
        # the per-bc bias lookup AND (for meta models) the buffer
        # lookup of bc_meta_*; we run the bc_idx replacement BEFORE
        # the metadata override below so the override sees the
        # post-mask bc_idx.
        if self.cfg.q_bc > 0.0:
            mask_bc = (
                torch.rand(B, generator=self.generator) < self.cfg.q_bc
            )
            n_masked = int(mask_bc.sum().item())
            if n_masked > 0:
                self.n_rows_bc_idx_masked += n_masked
                bc_idx = torch.where(
                    mask_bc.to(bc_idx.device),
                    torch.zeros_like(bc_idx),
                    bc_idx,
                )
                new_args[1] = bc_idx

        # ---- metadata dropout (meta models only) ----
        if (
            self.has_meta
            and self.accepts_override
            and (self.cfg.p_bench > 0.0 or self.cfg.p_subj > 0.0)
        ):
            override = self._build_meta_override(
                module, subject_idx=subject_idx, bc_idx=bc_idx, B=B
            )
            if override is not None:
                new_kwargs["meta_override"] = override

        return tuple(new_args), new_kwargs

    # -------- helpers --------

    def _build_meta_override(
        self,
        module: torch.nn.Module,
        *,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        B: int,
    ) -> dict[str, torch.Tensor] | None:
        """Return a per-row meta_override that mixes buffer values
        with the all-MISSING row 0 according to the dropout masks.
        """
        sub_cat_buf = getattr(module, "subject_meta_cat_ids", None)
        sub_num_buf = getattr(module, "subject_meta_num", None)
        bc_cat_buf = getattr(module, "bc_meta_cat_ids", None)
        bc_num_buf = getattr(module, "bc_meta_num", None)
        if sub_cat_buf is None or bc_cat_buf is None:
            return None

        # Standard buffer lookup -- the same one ``_gather_metadata``
        # would do internally. We replicate it here so we can splice
        # in the MISSING rows below, and so the override is the only
        # path to metadata for this batch.
        sub_cat = sub_cat_buf[subject_idx]
        sub_num = sub_num_buf[subject_idx]
        bc_cat = bc_cat_buf[bc_idx]
        bc_num = bc_num_buf[bc_idx]

        # Row 0 is the all-MISSING row by construction (see
        # build_metadata_id_tables: zeros = MISSING token, zero
        # numerics, missingness=1).
        bc_cat_missing = bc_cat_buf[0:1]
        bc_num_missing = bc_num_buf[0:1]
        sub_cat_missing = sub_cat_buf[0:1]
        sub_num_missing = sub_num_buf[0:1]

        if self.cfg.p_bench > 0.0:
            mask = (torch.rand(B, generator=self.generator) < self.cfg.p_bench)
            n = int(mask.sum().item())
            if n > 0:
                self.n_rows_bench_masked += n
                m = mask.to(bc_cat.device).unsqueeze(-1)
                bc_cat = torch.where(m, bc_cat_missing.expand_as(bc_cat), bc_cat)
                bc_num = torch.where(m, bc_num_missing.expand_as(bc_num), bc_num)

        if self.cfg.p_subj > 0.0:
            mask = (torch.rand(B, generator=self.generator) < self.cfg.p_subj)
            n = int(mask.sum().item())
            if n > 0:
                self.n_rows_subj_masked += n
                m = mask.to(sub_cat.device).unsqueeze(-1)
                sub_cat = torch.where(m, sub_cat_missing.expand_as(sub_cat), sub_cat)
                sub_num = torch.where(m, sub_num_missing.expand_as(sub_num), sub_num)

        return {
            "subj_cat": sub_cat,
            "subj_num": sub_num,
            "bc_cat": bc_cat,
            "bc_num": bc_num,
        }

    # -------- lifecycle --------

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "_DropoutPreHook":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def install_train_dropout(
    model: torch.nn.Module,
    cfg: TrainDropoutConfig,
) -> _DropoutPreHook:
    """Register the dropout pre-hook on ``model`` and return its handle.

    Calling ``handle.remove()`` (or using the handle as a context
    manager) detaches the hook. After detach, the model is bit-identical
    to one trained without the hook.

    The hook is a no-op when:

      * ``model.training`` is False (so val passes through), or
      * the model has no metadata buffers AND ``q_bc == 0`` (nothing
        to do for the no-meta variant without bc dropout).

    Stats counters (``n_rows_seen``, ``n_rows_*_masked``) are exposed on
    the returned handle so the caller can log effective dropout rates
    after training.
    """
    hook = _DropoutPreHook(model, cfg)
    if (
        not hook.has_meta
        and cfg.q_bc <= 0.0
    ):
        # Nothing to do -- caller likely passed the wrong cfg, but we
        # still install a no-op hook so the lifecycle (.remove()) is
        # symmetric. The hook just returns args, kwargs unchanged.
        pass
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    hook._handle = handle
    return hook


__all__ = [
    "TrainDropoutConfig",
    "install_train_dropout",
]
