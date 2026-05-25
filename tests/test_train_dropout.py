"""Tests for ``src.train_dropout``.

Verifies:
  * The pre-hook is a no-op during ``model.eval()``.
  * ``q_bc=1.0`` replaces every bc_idx with 0 during training.
  * ``p_bench=1.0`` replaces every bc_meta_* row with the MISSING row.
  * ``p_subj=1.0`` replaces every subject_meta_* row with the MISSING row.
  * The hook is seedable (same seed -> same masks).
  * After ``handle.remove()``, the model behaves identically to never
    having installed the hook (state_dict-level equality of forward).
  * The hook works on both ``MetaHybrid*`` (full meta path) and
    ``Hybrid*`` (bc-only path).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models import ModelConfig, build_model
from src.train_dropout import TrainDropoutConfig, install_train_dropout


def _make_meta_id_tables(n_subjects: int, n_bc: int):
    """Tiny synthetic metadata tables compatible with ModelConfig defaults."""
    from src.metadata_features import MetadataIdTables

    n_sub_cat = 2
    n_bc_cat = 2
    n_sub_num = 2
    n_bc_num = 2

    sub_cat = torch.zeros((n_subjects, n_sub_cat), dtype=torch.long)
    sub_num = torch.zeros((n_subjects, 2 * n_sub_num), dtype=torch.float32)
    bc_cat = torch.zeros((n_bc, n_bc_cat), dtype=torch.long)
    bc_num = torch.zeros((n_bc, 2 * n_bc_num), dtype=torch.float32)

    # Row 0 = MISSING by construction (0 categorical token, 0 numeric,
    # missingness=1 in odd channels).
    sub_num[0, 1::2] = 1.0
    bc_num[0, 1::2] = 1.0

    # Subsequent rows: category id = row index (mod 5 to stay in vocab),
    # numerics = small non-zero values with missingness=0.
    for r in range(1, n_subjects):
        for j in range(n_sub_cat):
            sub_cat[r, j] = (r + j) % 5
        for j in range(n_sub_num):
            sub_num[r, 2 * j] = float(r) * 0.1
            sub_num[r, 2 * j + 1] = 0.0
    for r in range(1, n_bc):
        for j in range(n_bc_cat):
            bc_cat[r, j] = (r + j) % 5
        for j in range(n_bc_num):
            bc_num[r, 2 * j] = float(r) * 0.1
            bc_num[r, 2 * j + 1] = 0.0

    return MetadataIdTables(
        subject_cat_ids=sub_cat,
        subject_num=sub_num,
        bc_cat_ids=bc_cat,
        bc_num=bc_num,
        subject_cat_cardinalities=(8, 8),
        benchmark_cat_cardinalities=(8, 8),
        subject_num_dim=2 * n_sub_num,
        benchmark_num_dim=2 * n_bc_num,
    )


def _meta_cfg(n_subjects: int, n_bc: int) -> ModelConfig:
    return ModelConfig(
        k=4,
        item_embed_dim=8,
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        dropout=0.0,
        n_subjects=n_subjects,
        n_benchmark_conditions=n_bc,
        use_subject_text_embedding=False,
        subject_embed_dim=0,
        lambda_resid_init=0.1,
        lambda_resid_trainable=True,
        use_pool_features=False,
        pool_feature_dim=0,
        use_cluster_features=False,
        n_clusters=0,
        cluster_embed_dim=0,
        use_judge_features=False,
        judge_feature_dim=0,
        use_nn_features=False,
        nn_feature_dim=0,
        use_metadata_features=True,
        meta_subject_categorical=("subj_a", "subj_b"),
        meta_subject_numeric=("snum1", "snum2"),
        meta_benchmark_categorical=("bench_a", "bench_b"),
        meta_benchmark_numeric=("bnum1", "bnum2"),
        meta_explicit_crosses=(),
    )


def _no_meta_cfg(n_subjects: int, n_bc: int) -> ModelConfig:
    return ModelConfig(
        k=4,
        item_embed_dim=8,
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        dropout=0.0,
        n_subjects=n_subjects,
        n_benchmark_conditions=n_bc,
        use_subject_text_embedding=False,
        subject_embed_dim=0,
        lambda_resid_init=0.1,
        lambda_resid_trainable=True,
        use_pool_features=False,
        pool_feature_dim=0,
        use_cluster_features=False,
        n_clusters=0,
        cluster_embed_dim=0,
        use_judge_features=False,
        judge_feature_dim=0,
        use_nn_features=False,
        nn_feature_dim=0,
        use_metadata_features=False,
    )


def _make_meta_model(n_subjects: int = 6, n_bc: int = 5):
    cfg = _meta_cfg(n_subjects, n_bc)
    model = build_model("meta_hybrid_irt_kfactor_gated_mlp", cfg)
    model.attach_metadata_tables(_make_meta_id_tables(n_subjects, n_bc))
    return model


def _make_no_meta_model(n_subjects: int = 6, n_bc: int = 5):
    cfg = _no_meta_cfg(n_subjects, n_bc)
    return build_model("hybrid_irt_kfactor_gated_mlp", cfg)


def _make_inputs(B: int, item_dim: int, n_subjects: int, n_bc: int):
    torch.manual_seed(0)
    subject_idx = torch.randint(1, n_subjects, (B,), dtype=torch.long)
    bc_idx = torch.randint(1, n_bc, (B,), dtype=torch.long)
    item_emb = torch.randn(B, item_dim)
    return subject_idx, bc_idx, item_emb


def test_dropout_is_noop_in_eval_mode():
    model = _make_meta_model()
    cfg = TrainDropoutConfig(p_bench=1.0, p_subj=1.0, q_bc=1.0, seed=0)
    handle = install_train_dropout(model, cfg)
    model.eval()
    s, bc, ie = _make_inputs(B=4, item_dim=8, n_subjects=6, n_bc=5)
    with torch.no_grad():
        out_with = model(s, bc, ie)
    handle.remove()
    with torch.no_grad():
        out_without = model(s, bc, ie)
    assert torch.allclose(out_with, out_without, atol=1e-6)
    assert handle.n_train_calls == 0
    assert handle.n_rows_bench_masked == 0
    assert handle.n_rows_bc_idx_masked == 0


def test_q_bc_one_replaces_every_bc_idx_with_zero():
    model = _make_meta_model()
    cfg = TrainDropoutConfig(q_bc=1.0, seed=0)
    s, bc, ie = _make_inputs(B=8, item_dim=8, n_subjects=6, n_bc=5)
    bc_zero = torch.zeros_like(bc)

    model.eval()
    with torch.no_grad():
        out_with_zero_bc = model(s, bc_zero, ie)

    handle = install_train_dropout(model, cfg)
    model.train()
    out_dropout = model(s, bc, ie)
    handle.remove()
    model.eval()

    # ``q_bc=1.0`` -> every row's bc_idx becomes 0, so output should
    # match the explicit bc=0 forward.
    assert torch.allclose(out_dropout, out_with_zero_bc, atol=1e-5)
    assert handle.n_rows_bc_idx_masked == 8


def test_p_bench_one_replaces_bc_metadata_with_missing_row():
    """With p_bench=1.0 and q_bc=0, every row's bc_meta_* should equal
    row 0 of the buffer (the MISSING row). We verify by gathering the
    metadata tensors via _gather_metadata and comparing to row 0.
    """
    model = _make_meta_model()
    cfg = TrainDropoutConfig(p_bench=1.0, p_subj=0.0, q_bc=0.0, seed=42)
    s, bc, ie = _make_inputs(B=6, item_dim=8, n_subjects=6, n_bc=5)

    handle = install_train_dropout(model, cfg)
    model.train()

    captured: dict = {}

    def capture_hook(module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Stop further forward execution by raising; we only care
        # about what gets passed in.
        raise RuntimeError("__stop__")

    h2 = model.register_forward_pre_hook(capture_hook, with_kwargs=True)
    try:
        model(s, bc, ie)
    except RuntimeError as e:
        assert "__stop__" in str(e)
    finally:
        h2.remove()
        handle.remove()

    override = captured["kwargs"].get("meta_override")
    assert override is not None, "meta_override should be injected with p_bench=1"
    expected_bc_cat = model.bc_meta_cat_ids[0:1].expand(6, -1)
    expected_bc_num = model.bc_meta_num[0:1].expand(6, -1)
    assert torch.equal(override["bc_cat"], expected_bc_cat)
    assert torch.equal(override["bc_num"], expected_bc_num)
    # Subject side should NOT be masked (p_subj=0).
    expected_subj_cat = model.subject_meta_cat_ids[s]
    expected_subj_num = model.subject_meta_num[s]
    assert torch.equal(override["subj_cat"], expected_subj_cat)
    assert torch.equal(override["subj_num"], expected_subj_num)


def test_p_subj_one_replaces_subject_metadata():
    model = _make_meta_model()
    cfg = TrainDropoutConfig(p_bench=0.0, p_subj=1.0, q_bc=0.0, seed=42)
    s, bc, ie = _make_inputs(B=6, item_dim=8, n_subjects=6, n_bc=5)

    handle = install_train_dropout(model, cfg)
    model.train()
    captured: dict = {}

    def capture_hook(module, args, kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("__stop__")

    h2 = model.register_forward_pre_hook(capture_hook, with_kwargs=True)
    try:
        model(s, bc, ie)
    except RuntimeError:
        pass
    finally:
        h2.remove()
        handle.remove()

    override = captured["kwargs"].get("meta_override")
    assert override is not None
    expected_subj_cat = model.subject_meta_cat_ids[0:1].expand(6, -1)
    expected_subj_num = model.subject_meta_num[0:1].expand(6, -1)
    assert torch.equal(override["subj_cat"], expected_subj_cat)
    assert torch.equal(override["subj_num"], expected_subj_num)
    # Bench side should NOT be masked.
    expected_bc_cat = model.bc_meta_cat_ids[bc]
    assert torch.equal(override["bc_cat"], expected_bc_cat)


def test_partial_dropout_is_seedable():
    """Same seed -> same per-row masks, so same forward output."""
    cfg = TrainDropoutConfig(p_bench=0.5, p_subj=0.3, q_bc=0.4, seed=123)
    s, bc, ie = _make_inputs(B=64, item_dim=8, n_subjects=6, n_bc=5)

    model_a = _make_meta_model()
    model_b = _make_meta_model()
    # Sync weights across the two models so any output difference is
    # purely from the dropout RNG.
    model_b.load_state_dict(model_a.state_dict())

    h_a = install_train_dropout(model_a, cfg)
    h_b = install_train_dropout(model_b, cfg)
    model_a.train()
    model_b.train()
    out_a = model_a(s, bc, ie)
    out_b = model_b(s, bc, ie)
    h_a.remove()
    h_b.remove()
    assert torch.allclose(out_a, out_b, atol=1e-6)
    # Counters should also match.
    assert h_a.n_rows_bench_masked == h_b.n_rows_bench_masked
    assert h_a.n_rows_subj_masked == h_b.n_rows_subj_masked
    assert h_a.n_rows_bc_idx_masked == h_b.n_rows_bc_idx_masked


def test_remove_restores_unmasked_forward():
    model = _make_meta_model()
    s, bc, ie = _make_inputs(B=6, item_dim=8, n_subjects=6, n_bc=5)
    model.eval()
    with torch.no_grad():
        baseline = model(s, bc, ie)
    handle = install_train_dropout(
        model, TrainDropoutConfig(p_bench=1.0, p_subj=1.0, q_bc=1.0, seed=0)
    )
    handle.remove()
    with torch.no_grad():
        after_remove = model(s, bc, ie)
    assert torch.allclose(baseline, after_remove, atol=1e-6)


def test_no_meta_model_supports_q_bc_only():
    """``HybridIRTItemKFactorGatedMLP`` has no metadata; the hook must
    still apply ``q_bc`` and skip metadata override (since the
    forward signature has no ``meta_override`` kwarg).
    """
    model = _make_no_meta_model()
    cfg = TrainDropoutConfig(p_bench=0.5, p_subj=0.5, q_bc=1.0, seed=0)
    s, bc, ie = _make_inputs(B=8, item_dim=8, n_subjects=6, n_bc=5)

    bc_zero = torch.zeros_like(bc)
    model.eval()
    with torch.no_grad():
        out_zero = model(s, bc_zero, ie)

    handle = install_train_dropout(model, cfg)
    assert handle.has_meta is False
    assert handle.accepts_override is False
    model.train()
    out_dropout = model(s, bc, ie)
    handle.remove()

    # q_bc=1.0 -> all bc_idx replaced with 0, so output matches the
    # explicit bc=0 forward. p_bench / p_subj are silently ignored
    # because the hybrid model has no meta_override path.
    assert torch.allclose(out_dropout, out_zero, atol=1e-5)
    assert handle.n_rows_bench_masked == 0
    assert handle.n_rows_subj_masked == 0
    assert handle.n_rows_bc_idx_masked == 8


def test_zero_probabilities_are_passthrough():
    model = _make_meta_model()
    cfg = TrainDropoutConfig(p_bench=0.0, p_subj=0.0, q_bc=0.0, seed=0)
    s, bc, ie = _make_inputs(B=8, item_dim=8, n_subjects=6, n_bc=5)
    model.eval()
    with torch.no_grad():
        baseline = model(s, bc, ie)
    handle = install_train_dropout(model, cfg)
    model.train()
    # Even in train mode, all-zero probs means no masking happens.
    out = model(s, bc, ie)
    handle.remove()
    # Forward includes randomness from nothing dropout-side, but the
    # model has dropout=0 in cfg so this should be exact.
    assert torch.allclose(out, baseline, atol=1e-5)
    assert handle.n_rows_bench_masked == 0
    assert handle.n_rows_subj_masked == 0
    assert handle.n_rows_bc_idx_masked == 0


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        TrainDropoutConfig(p_bench=1.5)
    with pytest.raises(ValueError):
        TrainDropoutConfig(q_bc=-0.1)
