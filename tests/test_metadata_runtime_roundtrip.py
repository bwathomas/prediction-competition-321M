"""Runtime mirror parity test for the metadata model variant.

The runtime ``model.py`` shipped to Codabench inlines parallel
versions of every training-side ``nn.Module`` (see
``src/export_submission.py::_RUNTIME_MODEL_PY``). The state-dict layout
between the runtime mirror and the training class MUST agree exactly
or every shipped checkpoint will crash on import.

Rather than trying to execute the entire runtime template in-process
(it pulls in transformers, encoders, etc., and is not designed for
unit-test isolation), we ast-walk the rendered runtime template and
extract just the metadata model class + its small dependency chain,
then exec that subset and compare its state_dict key set to the
training class's. Any mismatch is a real bug -- the runtime would
fail at ``load_state_dict(..., strict=True)``.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.export_submission import _RUNTIME_MODEL_PY
from src.metadata_features import (
    MetadataPreprocessor,
    build_metadata_id_tables,
)
from src.models import Indexer, MetaHybridIRTKFactorGatedMLP, ModelConfig


# ---------------------------------------------------------------------------
# Runtime-template extractor
# ---------------------------------------------------------------------------


# Class names + their (transitive) helpers in the runtime template that
# the metadata mirror depends on. We hand-list them rather than try to
# resolve symbol dependencies automatically -- this is a tight, targeted
# subset.
_RUNTIME_CLASSES = (
    "_ItemParameterMap",
    "_ItemIRTHeads",
    "_GatedResidual",
    "_IRTItemBase",
    "_RuntimePerFieldCategoricalEmbeddings",
    "_RuntimeMetaTower",
    "_RuntimeFactorizationMachineCross",
    "_RuntimeExplicitCrossEmbeddings",
    "_MetaHybridIRTKFactorGatedMLP",
)
_RUNTIME_HELPERS = (
    "_cluster_emb_runtime",
    "_runtime_auto_emb_dim",
    "_residual_input_dim_hybrid_irt_kfactor",
    "_residual_features_hybrid_irt_kfactor",
    "_residual_input_dim_meta_hybrid",
)


def _extract_runtime_module() -> types.ModuleType:
    """Extract just the metadata-mirror class hierarchy + minimal helpers.

    Returns an in-memory module with torch + numpy + math + nn / F imported
    and the named classes defined. Reads from the live
    ``_RUNTIME_MODEL_PY`` string so any structural change to the runtime
    template is caught by this test.
    """
    tree = ast.parse(_RUNTIME_MODEL_PY)
    wanted_names = set(_RUNTIME_CLASSES) | set(_RUNTIME_HELPERS)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted_names:
            kept.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_names:
            kept.append(node)
    new_mod = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(new_mod)
    code = compile(new_mod, filename="<runtime-meta-subset>", mode="exec")

    mod = types.ModuleType("runtime_meta_subset")
    mod.__dict__.update(
        {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "np": np,
            "math": __import__("math"),
        }
    )
    exec(code, mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runtime_template_contains_meta_class_set() -> None:
    """All metadata-related classes / helpers must exist in the runtime."""
    tree = ast.parse(_RUNTIME_MODEL_PY)
    present_classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    present_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for cls in _RUNTIME_CLASSES:
        assert cls in present_classes, f"runtime template missing class {cls}"
    for fn in _RUNTIME_HELPERS:
        assert fn in present_funcs, f"runtime template missing helper {fn}"


@pytest.fixture
def small_meta_setup() -> tuple[ModelConfig, MetaHybridIRTKFactorGatedMLP]:
    model_info = pd.DataFrame(
        [
            {"name": "Mistral-7B", "organization": "Mistral", "family": "Mistral", "macro_family": "Mistral", "parameters": 7.0, "release_date": 2024},
            {"name": "Llama-3-8B", "organization": "Meta", "family": "Llama 3", "macro_family": "Llama", "parameters": 8.0, "release_date": 2024},
        ]
    )
    benchmark_info = pd.DataFrame(
        [
            {"benchmark": "afrimedqa", "topic": "Medicine", "age": 500},
            {"benchmark": "hle", "topic": "Reasoning", "age": 200},
        ]
    )
    indexer = Indexer.fit(["s_mistral", "s_llama"], ["afrimedqa::none", "hle::none"])
    subject_content_by_key = {
        "s_mistral": "Name: Mistral-7B",
        "s_llama": "Name: Llama-3-8B",
    }
    mp = MetadataPreprocessor.fit(model_info, benchmark_info)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=indexer.subject_to_id,
        bc_to_id=indexer.bc_to_id,
        subject_content_by_key=subject_content_by_key,
    )

    cfg = ModelConfig(
        k=4,
        item_embed_dim=8,
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        dropout=0.0,
        n_subjects=indexer.n_subjects,
        n_benchmark_conditions=indexer.n_bc,
        use_metadata_features=True,
    )
    torch.manual_seed(0)
    model = MetaHybridIRTKFactorGatedMLP(cfg)
    model.attach_metadata_tables(tables)
    return cfg, model


def test_runtime_meta_state_dict_keys_match_training(small_meta_setup) -> None:
    """The runtime class' state_dict key set MUST equal the training class'.

    This is the load-time invariant: a key set mismatch would make
    ``load_state_dict(state, strict=True)`` raise on the platform.
    """
    cfg, train_model = small_meta_setup

    rt = _extract_runtime_module()
    runtime_cls = rt._MetaHybridIRTKFactorGatedMLP

    cfg_dict = dict(cfg.__dict__)
    rt_model = runtime_cls(cfg_dict)
    # The runtime model has to rebuild its meta submodules from the
    # saved state so its widths agree with what the training side
    # actually wrote. Exercise that rebuild path here.
    train_state = train_model.state_dict()
    rt_model.rebuild_from_state_dict(train_state)
    rt_state = rt_model.state_dict()

    train_keys = set(train_state.keys())
    rt_keys = set(rt_state.keys())

    missing_in_runtime = train_keys - rt_keys
    extra_in_runtime = rt_keys - train_keys
    assert not missing_in_runtime, (
        f"state_dict keys present in training class but missing in runtime: "
        f"{sorted(missing_in_runtime)}"
    )
    assert not extra_in_runtime, (
        f"state_dict keys present in runtime but missing in training: "
        f"{sorted(extra_in_runtime)}"
    )


def test_runtime_meta_state_dict_shapes_match_training(small_meta_setup) -> None:
    """Per-key tensor shapes MUST match between training and runtime."""
    cfg, train_model = small_meta_setup
    rt = _extract_runtime_module()
    runtime_cls = rt._MetaHybridIRTKFactorGatedMLP
    rt_model = runtime_cls(dict(cfg.__dict__))
    train_state = train_model.state_dict()
    rt_model.rebuild_from_state_dict(train_state)
    rt_state = rt_model.state_dict()
    mismatches = []
    for k, t in train_state.items():
        if k not in rt_state:
            continue
        if tuple(t.shape) != tuple(rt_state[k].shape):
            mismatches.append((k, tuple(t.shape), tuple(rt_state[k].shape)))
    assert not mismatches, (
        "state_dict shape mismatches between training and runtime: "
        + "; ".join(f"{k}: train={a} vs runtime={b}" for k, a, b in mismatches)
    )


def test_runtime_meta_forward_matches_training(small_meta_setup) -> None:
    """End-to-end parity: same inputs through training and runtime model.

    After a few SGD steps on the training side (so weights are
    non-trivial), the runtime side -- after ``rebuild_from_state_dict``
    + ``load_state_dict`` -- must produce identical logits.
    """
    cfg, train_model = small_meta_setup
    # Take a few SGD steps so the metadata weights diverge from
    # zero-init; this catches save/load bugs that masquerade as
    # parity when everything is identically zero.
    s = torch.tensor([1, 2, 1, 2])
    bc = torch.tensor([1, 1, 2, 2])
    ie = torch.randn(4, cfg.item_embed_dim, generator=torch.Generator().manual_seed(11))
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    opt = torch.optim.SGD(train_model.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        logits = train_model(s, bc, ie)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    train_model.eval()
    with torch.no_grad():
        y_train = train_model(s, bc, ie)

    rt = _extract_runtime_module()
    runtime_cls = rt._MetaHybridIRTKFactorGatedMLP
    rt_model = runtime_cls(dict(cfg.__dict__))
    state = train_model.state_dict()
    rt_model.rebuild_from_state_dict(state)
    missing, unexpected = rt_model.load_state_dict(state, strict=True)
    assert not missing, f"runtime missing keys: {missing}"
    assert not unexpected, f"runtime unexpected keys: {unexpected}"
    rt_model.eval()
    with torch.no_grad():
        y_rt = rt_model(s, bc, ie)
    torch.testing.assert_close(y_train, y_rt, rtol=1e-5, atol=1e-6)


def test_runtime_meta_override_path_runs() -> None:
    """meta_override path is wired correctly and returns a finite logit."""
    cfg = ModelConfig(
        k=4,
        item_embed_dim=8,
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        dropout=0.0,
        n_subjects=2,
        n_benchmark_conditions=2,
        use_metadata_features=True,
    )
    torch.manual_seed(0)
    train_model = MetaHybridIRTKFactorGatedMLP(cfg)
    # No attach_metadata_tables -- we want to exercise the override
    # path explicitly. ``MetadataIdTables`` placeholders are zero-init
    # so the per-id lookup returns the UNK row.

    cat_dim = len(cfg.meta_subject_categorical)
    num_dim = 2 * len(cfg.meta_subject_numeric)
    bench_cat_dim = len(cfg.meta_benchmark_categorical)
    bench_num_dim = 2 * len(cfg.meta_benchmark_numeric)
    override = {
        "subj_cat": torch.zeros((1, cat_dim), dtype=torch.long),
        "subj_num": torch.zeros((1, num_dim), dtype=torch.float32),
        "bc_cat": torch.zeros((1, bench_cat_dim), dtype=torch.long),
        "bc_num": torch.zeros((1, bench_num_dim), dtype=torch.float32),
    }
    s = torch.tensor([0])
    bc = torch.tensor([0])
    ie = torch.randn(1, cfg.item_embed_dim)
    train_model.eval()
    with torch.no_grad():
        logit = train_model(s, bc, ie, meta_override=override)
    assert logit.shape == (1,)
    assert torch.isfinite(logit).all()
