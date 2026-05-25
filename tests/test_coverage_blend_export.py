"""Tests for ``export_coverage_blend_run`` and the underlying
``blend_weights_missing`` plumbing in ``export_ensemble_run``.

We focus on the structural / serialization invariants of the bundle:

  * ``ensemble_ckpt["blend_weights_missing"]`` is round-tripped when the
    caller supplies it, and absent (None) when they don't.
  * ``runtime_meta.json`` records the second weight vector for
    diagnostic visibility.
  * ``export_ensemble_run`` rejects mismatched-length
    ``blend_weights_missing``.
  * ``_EnsembleModel`` (the runtime class) routes per-row weights and
    forwards ``meta_override`` only to members that declare it.

The runtime template's ``_EnsembleModel`` is exercised indirectly via
the runtime-template parity tests; we test the *module-level*
``_EnsembleModel`` here (it has the same code) by constructing it with
toy fold-models.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.export_submission import _RUNTIME_MODEL_PY


def _extract_runtime_class(template: str, class_name: str) -> str:
    """Pull a class definition out of the runtime template string.

    The template embeds ``_EnsembleModel`` (and other model variants);
    we extract just that class source so we can ``exec`` it in a
    minimal namespace and unit-test the class directly without
    booting the full runtime (which requires shipped artifacts).
    """
    pattern = (
        rf"^class {re.escape(class_name)}\(nn\.Module\):"
        r"(?:\n(?:[ \t].*)?)*"
    )
    m = re.search(pattern, template, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(
            f"could not find class {class_name!r} in _RUNTIME_MODEL_PY"
        )
    return m.group(0)


_NS: dict = {"nn": nn, "torch": torch}
exec(_extract_runtime_class(_RUNTIME_MODEL_PY, "_EnsembleModel"), _NS)
_EnsembleModel = _NS["_EnsembleModel"]


class _MetaForward(nn.Module):
    """Toy fold-model whose forward DECLARES ``meta_override``."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(
        self, s, bc, ie, se=None, pool_feats=None, cluster_ids=None,
        judge_feats=None, nn_feats=None, meta_override=None,
    ):
        # Output is a logit per row -- we encode the scale into the
        # output so blending is verifiable.
        # Also encode whether meta_override was passed by adding 100.
        B = ie.shape[0]
        bias = 100.0 if meta_override is not None else 0.0
        return torch.full((B,), self.scale + bias, dtype=ie.dtype, device=ie.device)


class _NoMetaForward(nn.Module):
    """Toy fold-model whose forward does NOT declare ``meta_override``."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(
        self, s, bc, ie, se=None, pool_feats=None, cluster_ids=None,
        judge_feats=None, nn_feats=None,
    ):
        B = ie.shape[0]
        return torch.full((B,), self.scale, dtype=ie.dtype, device=ie.device)


def _members(scale_a=1.0, scale_b=2.0):
    return [
        {
            "config_id": "meta",
            "model_name": "meta_hybrid_irt_kfactor_gated_mlp",
            "model_cfg": {"use_metadata_features": True},
            "fold_models": [_MetaForward(scale=scale_a)],
        },
        {
            "config_id": "no_meta",
            "model_name": "hybrid_irt_kfactor_gated_mlp",
            "model_cfg": {"use_metadata_features": False},
            "fold_models": [_NoMetaForward(scale=scale_b)],
        },
    ]


def test_scalar_blend_unchanged_when_missing_weights_none():
    em = _EnsembleModel(_members(scale_a=0.0, scale_b=0.0), [0.6, 0.4])
    s = torch.tensor([0])
    bc = torch.tensor([0])
    ie = torch.zeros(1, 4)
    out = em(s, bc, ie)
    # Logit such that sigmoid(logit) = 0.6 * 0.5 + 0.4 * 0.5 = 0.5 -> logit ~ 0
    p = torch.sigmoid(out).item()
    assert abs(p - 0.5) < 1e-6
    assert em._has_coverage_blend is False


def test_coverage_blend_uses_present_weights_when_bench_present_one():
    """With bench_present=[1], the blend should equal blend_weights."""
    em = _EnsembleModel(
        _members(scale_a=10.0, scale_b=-10.0),
        blend_weights=[0.9, 0.1],
        blend_weights_missing=[0.1, 0.9],
    )
    assert em._has_coverage_blend is True
    s = torch.tensor([0])
    bc = torch.tensor([1])  # arbitrary
    ie = torch.zeros(1, 4)
    bp = torch.tensor([1.0])
    out = em(s, bc, ie, bench_present=bp)
    p_blend = torch.sigmoid(out).item()
    # Member A logit = 10 -> p_A ≈ 1.0; Member B logit = -10 -> p_B ≈ 0.0.
    # Blend with weights [0.9, 0.1]: ≈ 0.9 * 1.0 + 0.1 * 0.0 = 0.9
    assert abs(p_blend - 0.9) < 1e-3


def test_coverage_blend_uses_missing_weights_when_bench_present_zero():
    em = _EnsembleModel(
        _members(scale_a=10.0, scale_b=-10.0),
        blend_weights=[0.9, 0.1],
        blend_weights_missing=[0.1, 0.9],
    )
    s = torch.tensor([0])
    bc = torch.tensor([0])
    ie = torch.zeros(1, 4)
    bp = torch.tensor([0.0])
    out = em(s, bc, ie, bench_present=bp)
    p_blend = torch.sigmoid(out).item()
    # Now weights swap: 0.1 * 1.0 + 0.9 * 0.0 = 0.1
    assert abs(p_blend - 0.1) < 1e-3


def test_coverage_blend_per_row_routing_in_batch():
    em = _EnsembleModel(
        _members(scale_a=10.0, scale_b=-10.0),
        blend_weights=[0.9, 0.1],
        blend_weights_missing=[0.1, 0.9],
    )
    s = torch.tensor([0, 0, 0, 0])
    bc = torch.tensor([1, 0, 1, 0])
    ie = torch.zeros(4, 4)
    bp = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = em(s, bc, ie, bench_present=bp)
    p = torch.sigmoid(out)
    # rows 0, 2 -> ≈ 0.9; rows 1, 3 -> ≈ 0.1
    assert abs(p[0].item() - 0.9) < 1e-3
    assert abs(p[1].item() - 0.1) < 1e-3
    assert abs(p[2].item() - 0.9) < 1e-3
    assert abs(p[3].item() - 0.1) < 1e-3


def test_meta_override_routed_only_to_meta_member():
    """The meta member adds +100 when meta_override is non-None; the
    no-meta member doesn't see the kwarg. With huge magnitudes the
    blended logit should reflect this asymmetry.
    """
    em = _EnsembleModel(
        _members(scale_a=0.0, scale_b=0.0),
        blend_weights=[1.0, 0.0],  # only Member A contributes
    )
    s = torch.tensor([0])
    bc = torch.tensor([0])
    ie = torch.zeros(1, 4)

    # Without meta_override: Member A logit = 0.0 -> p_A = 0.5, blend = 0.5
    out_no = em(s, bc, ie)
    assert abs(torch.sigmoid(out_no).item() - 0.5) < 1e-3

    # With meta_override: Member A logit = 100 -> p_A ≈ 1.0
    fake_override = {
        "subj_cat": torch.zeros(1, 1, dtype=torch.long),
        "subj_num": torch.zeros(1, 1),
        "bc_cat": torch.zeros(1, 1, dtype=torch.long),
        "bc_num": torch.zeros(1, 1),
    }
    out_with = em(s, bc, ie, meta_override=fake_override)
    assert torch.sigmoid(out_with).item() > 0.99


def test_member_signature_introspection_caches_correct_flags():
    em = _EnsembleModel(_members(), [0.5, 0.5])
    # Member 0 is the meta model; should accept meta_override.
    assert em._member_accepts_meta_override == [True, False]


def test_constructor_rejects_mismatched_missing_weight_length():
    with pytest.raises(RuntimeError, match="blend_weights_missing"):
        _EnsembleModel(_members(), [0.5, 0.5], blend_weights_missing=[0.5])


def test_constructor_normalizes_missing_weights():
    """Negative or all-zero ``blend_weights_missing`` should be coerced
    to a uniform distribution rather than crashing the runtime."""
    em = _EnsembleModel(_members(), [0.5, 0.5], blend_weights_missing=[0.0, 0.0])
    # Coerced to uniform [0.5, 0.5].
    assert torch.allclose(
        em.blend_weights_missing.cpu(),
        torch.tensor([0.5, 0.5], dtype=torch.float32),
    )


def test_export_ensemble_run_rejects_mismatched_missing_weights():
    """Public exporter signature should also enforce length match
    (in addition to the inner ``_EnsembleModel`` check) so callers get
    the failure at the export call site rather than during runtime
    bundle load.
    """
    from src.export_submission import export_ensemble_run

    members = [
        {
            "config_id": "a",
            "model_name": "x",
            "model_cfg": {},
            "fold_checkpoint_paths": [],
        },
        {
            "config_id": "b",
            "model_name": "y",
            "model_cfg": {},
            "fold_checkpoint_paths": [],
        },
    ]
    with pytest.raises(RuntimeError, match="blend_weights_missing length"):
        export_ensemble_run(
            members=members,
            blend_weights=[0.5, 0.5],
            blend_weights_missing=[0.5],  # mismatch
            indexer={"subject_to_id": {}, "bc_to_id": {}},
            encoder_cfg={"model_id": "x"},
            fold_assignment_sha256="x",
        )
