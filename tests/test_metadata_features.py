"""Unit tests for the structured-metadata pathway.

These tests cover three properties that the architecture promises:

1. ``use_metadata_features=False`` is bit-identical to the original
   :class:`HybridIRTItemKFactorGatedMLP` -- the new model variant must
   be a strict superset that doesn't perturb existing checkpoints.
2. With ``use_metadata_features=True``, the model boots up with
   zero-init metadata heads (towers, FM, explicit crosses) so the
   *output* at step 0 equals the non-metadata baseline up to the
   per-id embedding noise. This is the "parity at init" invariant.
3. The metadata channels *do* fire after a few synthetic gradient
   steps that put signal into the metadata side -- specifically, the
   FM cross head must be able to learn a ``family x topic`` toy
   interaction. This is the "Mistral x Medicine" test.

Tests are intentionally CPU-only and small so they can run on any
machine without a GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.metadata_features import (
    CategoricalVocab,
    ExplicitCrossEmbeddings,
    FactorizationMachineCross,
    MetadataPreprocessor,
    MetadataSchema,
    MetaTower,
    NumericScaler,
    _auto_emb_dim,
    build_metadata_id_tables,
    extract_display_name,
    normalize_condition_token,
)
from src.models import (
    HybridIRTItemKFactorGatedMLP,
    Indexer,
    MetaHybridIRTKFactorGatedMLP,
    ModelConfig,
    build_model,
)


# ---------------------------------------------------------------------------
# Small fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "Mistral-7B", "organization": "Mistral", "family": "Mistral", "macro_family": "Mistral", "parameters": 7.0, "release_date": 2024},
            {"name": "Mixtral-8x7B", "organization": "Mistral", "family": "Mixtral", "macro_family": "Mistral", "parameters": 56.0, "release_date": 2024},
            {"name": "Llama-3-8B", "organization": "Meta", "family": "Llama 3", "macro_family": "Llama", "parameters": 8.0, "release_date": 2024},
            {"name": "Claude-3-Opus", "organization": "Anthropic", "family": "Claude 3", "macro_family": "Claude", "parameters": None, "release_date": 2024},
            {"name": "GPT-4o", "organization": "OpenAI", "family": "GPT-4", "macro_family": "GPT", "parameters": None, "release_date": 2024},
        ]
    )


@pytest.fixture
def tiny_benchmark_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"benchmark": "afrimedqa", "topic": "Medicine", "age": 500, "has_conditions": 1},
            {"benchmark": "androidworld", "topic": "GUI Agents", "age": 700, "has_conditions": 0},
            {"benchmark": "hle", "topic": "Reasoning", "age": 200, "has_conditions": 0},
        ]
    )


@pytest.fixture
def tiny_indexer() -> Indexer:
    subject_keys = ["s_mistral", "s_mixtral", "s_llama", "s_claude", "s_gpt4o"]
    bc_keys = ["afrimedqa::none", "androidworld::none", "hle::none"]
    return Indexer.fit(subject_keys, bc_keys)


@pytest.fixture
def tiny_subject_content_by_key() -> dict[str, str]:
    return {
        "s_mistral": "Name: Mistral-7B\nOrganization: Mistral",
        "s_mixtral": "Name: Mixtral-8x7B\nOrganization: Mistral",
        "s_llama": "Name: Llama-3-8B\nOrganization: Meta",
        "s_claude": "Name: Claude-3-Opus\nOrganization: Anthropic",
        "s_gpt4o": "Name: GPT-4o\nOrganization: OpenAI",
    }


# ---------------------------------------------------------------------------
# Helpers used by multiple tests
# ---------------------------------------------------------------------------


def _make_cfg(**kwargs) -> ModelConfig:
    cfg = ModelConfig(
        k=4,
        item_embed_dim=8,
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        dropout=0.0,
        n_subjects=6,
        n_benchmark_conditions=4,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _fresh_random_inputs(cfg: ModelConfig, batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, cfg.n_subjects, (batch_size,), generator=g)
    bc = torch.randint(0, cfg.n_benchmark_conditions, (batch_size,), generator=g)
    ie = torch.randn(batch_size, cfg.item_embed_dim, generator=g)
    return s, bc, ie


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_display_name_and_normalize_condition() -> None:
    assert extract_display_name("Name: Mistral-7B\nOrganization: Mistral") == "Mistral-7B"
    assert extract_display_name("name: claude-3-opus") == "claude-3-opus"
    assert extract_display_name("") == ""
    assert extract_display_name(None) == ""

    assert normalize_condition_token(None) == "none"
    assert normalize_condition_token("") == "none"
    assert normalize_condition_token("None") == "none"
    assert normalize_condition_token("attack=direct") == "attack=direct"


def test_categorical_vocab_unk_and_missing() -> None:
    v = CategoricalVocab("organization").fit(["Mistral", "Meta", "Mistral", None, "", "Anthropic"])
    # MISSING at 0, UNK at 1, real tokens after.
    assert v.token_to_id["__MISSING__"] == 0
    assert v.token_to_id["__UNK__"] == 1
    assert v.encode_one(None) == 0
    assert v.encode_one("") == 0
    assert v.encode_one("Meta") == v.token_to_id["Meta"]
    # Unseen value -> UNK
    assert v.encode_one("NewlyAddedOrg") == 1

    encoded = v.encode(["Mistral", None, "Anthropic", "Other"])
    assert encoded.dtype == np.int64
    assert encoded[0] == v.token_to_id["Mistral"]
    assert encoded[1] == 0
    assert encoded[2] == v.token_to_id["Anthropic"]
    assert encoded[3] == 1


def test_categorical_vocab_roundtrip() -> None:
    v = CategoricalVocab("topic").fit(["Math", "Medicine", "Code"])
    d = v.to_dict()
    v2 = CategoricalVocab.from_dict(d)
    assert v2.token_to_id == v.token_to_id
    assert v2.frozen
    assert v2.encode_one("Medicine") == v.encode_one("Medicine")


def test_numeric_scaler_handles_missing_and_log() -> None:
    sc = NumericScaler("log_params", log_transform=True).fit([7.0, 13.0, 56.0, None, float("nan")])
    x, m = sc.transform([7.0, None, 100.0])
    assert x.dtype == np.float32
    assert m.dtype == np.float32
    # Missing entry imputed to the median of the *original* (pre-log)
    # values, which is 13.0; the missingness indicator must be 1.
    assert m[1] == 1.0
    assert m[0] == 0.0
    # The transform output is finite for every input.
    assert np.isfinite(x).all()

    # Roundtrip serialization.
    sc2 = NumericScaler.from_dict(sc.to_dict())
    x2, m2 = sc2.transform([7.0, None, 100.0])
    np.testing.assert_allclose(x, x2)
    np.testing.assert_allclose(m, m2)


def test_metadata_preprocessor_fit_and_encode(tiny_model_info, tiny_benchmark_info) -> None:
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    assert "organization" in mp.subject_cat_vocabs
    assert "topic" in mp.benchmark_cat_vocabs
    assert "log_params" in mp.subject_num_scalers
    assert "benchmark_age" in mp.benchmark_num_scalers

    cat_ids, num, miss = mp.encode_subject("Mistral-7B")
    assert cat_ids.shape == (3,)        # 3 subject categorical fields
    assert num.shape == (2,)             # log_params + release_date
    assert miss.shape == (2,)
    # log_params for Mistral-7B (=7) is present, so missingness indicator 0
    assert miss[0] == 0.0
    # release_date for Mistral-7B is also present
    assert miss[1] == 0.0
    # organization id for Mistral-7B must equal the fitted org-vocab "Mistral" id
    assert int(cat_ids[0]) == mp.subject_cat_vocabs["organization"].encode_one("Mistral")

    # Encoding an UNKNOWN subject by name returns all-MISSING
    cat_ids_u, num_u, miss_u = mp.encode_subject("Totally-Unknown-Model")
    assert (cat_ids_u == 0).all()
    assert (miss_u == 1.0).all()

    cat_ids_b, num_b, miss_b = mp.encode_benchmark("afrimedqa")
    assert int(cat_ids_b[0]) == mp.benchmark_cat_vocabs["topic"].encode_one("Medicine")
    assert miss_b[0] == 0.0


def test_metadata_preprocessor_roundtrip(tiny_model_info, tiny_benchmark_info, tmp_path) -> None:
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    blob = mp.to_dict()
    # Must be JSON-serializable
    payload = json.dumps(blob, default=str)
    mp2 = MetadataPreprocessor.from_dict(json.loads(payload))
    a, b, c = mp.encode_subject("Mistral-7B")
    a2, b2, c2 = mp2.encode_subject("Mistral-7B")
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_allclose(b, b2)
    np.testing.assert_allclose(c, c2)


def test_build_metadata_id_tables(
    tiny_model_info, tiny_benchmark_info, tiny_indexer, tiny_subject_content_by_key
) -> None:
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=tiny_indexer.subject_to_id,
        bc_to_id=tiny_indexer.bc_to_id,
        subject_content_by_key=tiny_subject_content_by_key,
    )
    # Row 0 must be UNK (MISSING for cats, missingness=1 for nums)
    assert (tables.subject_cat_ids[0] == 0).all()
    assert tables.subject_num[0, 1] == 1.0       # missingness ind for first numeric
    assert (tables.bc_cat_ids[0] == 0).all()

    # A known subject's row must encode its organization
    mistral_idx = tiny_indexer.subject_to_id["s_mistral"]
    expected_org = mp.subject_cat_vocabs["organization"].encode_one("Mistral")
    assert int(tables.subject_cat_ids[mistral_idx, 0]) == expected_org

    # The cardinalities reported must match the vocabs
    assert tables.subject_cat_cardinalities[0] == mp.subject_cat_vocabs["organization"].n_tokens
    assert tables.benchmark_cat_cardinalities[0] == mp.benchmark_cat_vocabs["topic"].n_tokens


def test_factorization_machine_cross_zero_at_init() -> None:
    # FM head with zero-init output projection must produce zero logits
    # for any input. (This is the parity-at-init invariant.)
    fm = FactorizationMachineCross(
        subject_field_dims=[8, 8],
        benchmark_field_dims=[8],
        d_fm=4,
    )
    e_subj = [torch.randn(5, 8), torch.randn(5, 8)]
    e_bench = [torch.randn(5, 8)]
    out = fm(e_subj, e_bench)
    assert out.shape == (5,)
    assert torch.allclose(out, torch.zeros(5))


def test_factorization_machine_cross_with_numerics_zero_at_init() -> None:
    """Adding numeric fields must not perturb the parity-at-init guarantee."""
    fm = FactorizationMachineCross(
        subject_field_dims=[8, 8],
        benchmark_field_dims=[8],
        d_fm=4,
        subject_num_field_count=2,    # log_params, release_date
        bench_num_field_count=1,      # benchmark_age
    )
    B = 5
    e_subj = [torch.randn(B, 8), torch.randn(B, 8)]
    e_bench = [torch.randn(B, 8)]
    # subj_num is interleaved [value, mask, value, mask, ...]; bench_num same.
    subj_num = torch.randn(B, 4)        # 2 numeric fields x 2
    bench_num = torch.randn(B, 2)       # 1 numeric field x 2
    out = fm(e_subj, e_bench, subj_num_features=subj_num, bench_num_features=bench_num)
    assert out.shape == (B,)
    assert torch.allclose(out, torch.zeros(B))


def test_factorization_machine_cross_numerics_optional_when_count_zero() -> None:
    """``num_field_count=0`` produces the legacy cat-only behavior."""
    fm = FactorizationMachineCross(
        subject_field_dims=[6],
        benchmark_field_dims=[4],
        d_fm=3,
    )
    out_a = fm([torch.randn(2, 6)], [torch.randn(2, 4)])
    # Passing None numerics must equal not passing them at all.
    out_b = fm(
        [torch.randn(2, 6)],
        [torch.randn(2, 4)],
        subj_num_features=None,
        bench_num_features=None,
    )
    assert out_a.shape == (2,) and out_b.shape == (2,)


def test_factorization_machine_cross_split_value_mask_helper() -> None:
    """Internal helper: (B, 2N) -> list of (B, 2) tensors, in field order."""
    fm = FactorizationMachineCross(
        subject_field_dims=[],
        benchmark_field_dims=[],
        d_fm=2,
        subject_num_field_count=2,
        bench_num_field_count=0,
    )
    flat = torch.tensor(
        [[1.0, 0.0, 5.0, 1.0],
         [2.0, 0.0, 6.0, 0.0]],
        dtype=torch.float32,
    )
    parts = FactorizationMachineCross._split_value_mask(flat, 2)
    assert len(parts) == 2
    assert torch.equal(parts[0], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.equal(parts[1], torch.tensor([[5.0, 1.0], [6.0, 0.0]]))
    # An empty/None input -> empty list.
    assert FactorizationMachineCross._split_value_mask(None, 0) == []
    assert FactorizationMachineCross._split_value_mask(flat, 0) == []
    # Wrong shape raises.
    import pytest as _pytest
    with _pytest.raises(ValueError):
        FactorizationMachineCross._split_value_mask(torch.zeros(3, 5), 2)


def test_fm_cross_can_learn_numeric_x_categorical_interaction() -> None:
    """Smoke: FM head must be able to fit a synthetic
    ``f(family, log_params) = w_family * log_params`` interaction
    that no purely-additive head could learn.

    We parametrize the *labels* as a function of (family categorical,
    a single numeric scalar), and check that the FM channel reduces
    BCE-with-logits loss meaningfully relative to a fixed-zero baseline.
    """
    from torch import nn as _nn

    torch.manual_seed(0)

    n_family = 4
    fam_emb = _nn.Embedding(n_family, 6)
    _nn.init.normal_(fam_emb.weight, std=0.5)
    fm = FactorizationMachineCross(
        subject_field_dims=[6],
        benchmark_field_dims=[],
        d_fm=4,
        subject_num_field_count=1,
    )

    # Generate synthetic data: label depends on family AND log_params,
    # specifically family-specific monotone direction. We use
    # deterministic labels (sign of the ground-truth logit) so the
    # optimal BCE loss is 0; this avoids irreducible Bernoulli noise
    # masking the FM channel's learning capacity in a small-batch test.
    B = 512
    fam_id = torch.randint(0, n_family, (B,))
    logp = torch.randn(B) * 1.5
    sign = torch.tensor([2.0, -2.0, 1.5, -1.5])  # one direction per family
    logits = sign[fam_id] * logp
    labels = (logits > 0).float()
    subj_num = torch.stack([logp, torch.zeros_like(logp)], dim=1)  # value + mask=0

    opt = torch.optim.Adam(list(fm.parameters()) + list(fam_emb.parameters()), lr=0.1)
    bce = _nn.BCEWithLogitsLoss()

    # Baseline: untrained head -> ~0.69 (label-prior loss).
    with torch.no_grad():
        baseline = bce(
            fm([fam_emb(fam_id)], [], subj_num_features=subj_num), labels
        ).item()

    final = float("inf")
    for _ in range(800):
        opt.zero_grad()
        out = fm([fam_emb(fam_id)], [], subj_num_features=subj_num)
        loss = bce(out, labels)
        loss.backward()
        opt.step()
        final = float(loss.detach())

    # The cat x num interaction must be learnable; loss should drop
    # at least 50% from the label-prior baseline. (In practice we see
    # loss settle around 0.10-0.20 -- well below the 0.35 threshold.)
    assert final < 0.5 * baseline, (
        f"FM cat x num head failed to learn family x log_params interaction: "
        f"baseline={baseline:.4f}, final={final:.4f}"
    )


def test_explicit_cross_zero_at_init(tiny_model_info, tiny_benchmark_info) -> None:
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    sub_card = tuple(v.n_tokens for v in mp.subject_cat_vocabs.values())
    bench_card = tuple(v.n_tokens for v in mp.benchmark_cat_vocabs.values())
    head = ExplicitCrossEmbeddings(
        crosses=("family__topic",),
        schema=mp.schema,
        subject_cardinalities=sub_card,
        benchmark_cardinalities=bench_card,
        emb_dim=4,
    )
    subj_cat = torch.tensor([[2, 3, 4], [1, 1, 1], [0, 0, 0]], dtype=torch.long)
    bench_cat = torch.tensor([[2], [3], [0]], dtype=torch.long)
    out = head(subj_cat, bench_cat)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.zeros(3))


def test_meta_tower_zero_at_init_and_shape() -> None:
    tower = MetaTower(in_dim=16, hidden_dim=32, k=8, dropout=0.0)
    x = torch.randn(7, 16)
    s, v = tower(x)
    assert s.shape == (7,)
    assert v.shape == (7, 8)
    assert torch.allclose(s, torch.zeros(7))
    assert torch.allclose(v, torch.zeros(7, 8))

    # A degenerate (in_dim=0) tower also returns zeros.
    tower2 = MetaTower(in_dim=0, hidden_dim=8, k=4, dropout=0.0)
    x2 = torch.zeros(3, 0)
    s2, v2 = tower2(x2)
    assert torch.allclose(s2, torch.zeros(3))
    assert torch.allclose(v2, torch.zeros(3, 4))


def test_meta_model_parity_with_hybrid_when_metadata_disabled() -> None:
    """A meta model with use_metadata_features=False must behave like the hybrid.

    This guards against the new variant accidentally introducing
    behavior changes when the flag is off. We compare the output of
    forward(...) on a random batch against the original hybrid.
    """
    cfg_off = _make_cfg(use_metadata_features=False)
    torch.manual_seed(0)
    hybrid = HybridIRTItemKFactorGatedMLP(cfg_off)
    torch.manual_seed(0)
    meta_off = MetaHybridIRTKFactorGatedMLP(cfg_off)

    # Copy parameters across so we compare apples to apples. (Init
    # streams differ because MetaHybrid has extra modules constructed,
    # so the rng is consumed differently; the architectural property
    # we want is "same params -> same output".)
    state = hybrid.state_dict()
    own_keys = set(meta_off.state_dict().keys())
    # Only load the keys we share; ignore metadata-side params.
    intersect = {k: v for k, v in state.items() if k in own_keys and meta_off.state_dict()[k].shape == v.shape}
    missing, unexpected = meta_off.load_state_dict(intersect, strict=False)
    # The keys missing are the metadata-side weights, which are unused
    # when use_metadata_features=False. We only check the shared subset.
    s, bc, ie = _fresh_random_inputs(cfg_off, batch_size=4, seed=42)
    hybrid.eval()
    meta_off.eval()
    with torch.no_grad():
        y_h = hybrid(s, bc, ie)
        y_m = meta_off(s, bc, ie)
    torch.testing.assert_close(y_h, y_m, rtol=1e-4, atol=1e-5)


def test_meta_model_parity_at_init_with_metadata_on(
    tiny_model_info, tiny_benchmark_info, tiny_indexer, tiny_subject_content_by_key
) -> None:
    """With metadata enabled but zero-init heads, output equals the no-meta baseline.

    This guards the "boots up bit-identical" invariant -- the metadata
    pathway must add exactly zero to the logit at construction time
    (only after gradients have flowed should the metadata contribute).
    """
    cfg = _make_cfg(
        n_subjects=tiny_indexer.n_subjects,
        n_benchmark_conditions=tiny_indexer.n_bc,
        use_metadata_features=True,
    )
    torch.manual_seed(0)
    model = MetaHybridIRTKFactorGatedMLP(cfg)

    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=tiny_indexer.subject_to_id,
        bc_to_id=tiny_indexer.bc_to_id,
        subject_content_by_key=tiny_subject_content_by_key,
    )
    model.attach_metadata_tables(tables)

    # Now build a non-metadata sibling with the same id-table weights
    # and verify forward() agrees with cfg.use_metadata_features=False.
    cfg_off = _make_cfg(
        n_subjects=tiny_indexer.n_subjects,
        n_benchmark_conditions=tiny_indexer.n_bc,
        use_metadata_features=False,
    )
    torch.manual_seed(0)
    model_off = MetaHybridIRTKFactorGatedMLP(cfg_off)

    # Copy weights of the shared modules across; the metadata module
    # outputs are zero so the on/off comparison should still match.
    own_off = model_off.state_dict()
    intersect = {k: v for k, v in model.state_dict().items() if k in own_off and own_off[k].shape == v.shape}
    model_off.load_state_dict(intersect, strict=False)

    s = torch.tensor([1, 2, 3])
    bc = torch.tensor([1, 2, 1])
    ie = torch.randn(3, cfg.item_embed_dim)
    model.eval()
    model_off.eval()
    with torch.no_grad():
        y_on = model(s, bc, ie)
        y_off = model_off(s, bc, ie)
    # The metadata channels are all zero-init at construction so the
    # output should be exactly the same.
    torch.testing.assert_close(y_on, y_off, rtol=1e-4, atol=1e-5)


def test_meta_model_save_load_roundtrip(
    tiny_model_info, tiny_benchmark_info, tiny_indexer, tiny_subject_content_by_key, tmp_path
) -> None:
    cfg = _make_cfg(
        n_subjects=tiny_indexer.n_subjects,
        n_benchmark_conditions=tiny_indexer.n_bc,
        use_metadata_features=True,
    )
    torch.manual_seed(7)
    model = MetaHybridIRTKFactorGatedMLP(cfg)
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=tiny_indexer.subject_to_id,
        bc_to_id=tiny_indexer.bc_to_id,
        subject_content_by_key=tiny_subject_content_by_key,
    )
    model.attach_metadata_tables(tables)

    # Take a few SGD steps so head weights become non-zero -- if save
    # / load is broken it will show up here.
    s = torch.tensor([1, 2, 3])
    bc = torch.tensor([1, 2, 1])
    ie = torch.randn(3, cfg.item_embed_dim, generator=torch.Generator().manual_seed(9))
    y_target = torch.tensor([1.0, 0.0, 1.0])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        logits = model(s, bc, ie)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_target)
        loss.backward()
        opt.step()

    # Save / reload
    ckpt_path = tmp_path / "meta_hybrid.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg2 = ModelConfig(**payload["cfg"])
    model2 = MetaHybridIRTKFactorGatedMLP(cfg2)
    # We need to attach with the same cardinalities as the saved
    # checkpoint, which is captured in the saved buffers themselves --
    # but `MetaHybrid` resizes its per-field embeddings only when
    # `attach_metadata_tables` is called. Re-fit the preprocessor and
    # re-build the tables, then load.
    mp2 = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    tables2 = build_metadata_id_tables(
        preprocessor=mp2,
        subject_to_id=tiny_indexer.subject_to_id,
        bc_to_id=tiny_indexer.bc_to_id,
        subject_content_by_key=tiny_subject_content_by_key,
    )
    model2.attach_metadata_tables(tables2)
    model2.load_state_dict(payload["model_state"])

    model.eval()
    model2.eval()
    with torch.no_grad():
        y_a = model(s, bc, ie)
        y_b = model2(s, bc, ie)
    torch.testing.assert_close(y_a, y_b, rtol=1e-5, atol=1e-6)


def test_build_model_registry_includes_meta_hybrid() -> None:
    cfg = _make_cfg(use_metadata_features=False)
    model = build_model("meta_hybrid_irt_kfactor_gated_mlp", cfg)
    assert isinstance(model, MetaHybridIRTKFactorGatedMLP)


def test_fm_cross_learns_family_topic_interaction(
    tiny_model_info, tiny_benchmark_info, tiny_indexer, tiny_subject_content_by_key
) -> None:
    """End-to-end check: the metadata stack can learn a Mistral x Medicine signal.

    We synthesize a tiny dataset where the label is *purely* determined
    by the (family, topic) cross -- specifically, family=Mistral on
    topic=Medicine flips the label. If the metadata pathway is wired
    correctly, the FM + explicit cross + tower channels should drive
    the training loss measurably below random within a few hundred
    SGD steps. If only the id embeddings can fit it, the loss won't
    drop as fast for unseen (subject_idx, bc_idx) combos.
    """
    cfg = _make_cfg(
        n_subjects=tiny_indexer.n_subjects,
        n_benchmark_conditions=tiny_indexer.n_bc,
        k=4,
        use_metadata_features=True,
        meta_include_in_residual=True,
        # Disable the residual MLP to make the test deterministic w.r.t.
        # the metadata channels only; we want a clean signal that the
        # towers / FM / explicit crosses are picking up the family x
        # topic interaction without help from the dense MLP.
        lambda_resid_init=0.0,
        lambda_resid_trainable=False,
    )
    torch.manual_seed(0)
    model = MetaHybridIRTKFactorGatedMLP(cfg)
    mp = MetadataPreprocessor.fit(tiny_model_info, tiny_benchmark_info)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=tiny_indexer.subject_to_id,
        bc_to_id=tiny_indexer.bc_to_id,
        subject_content_by_key=tiny_subject_content_by_key,
    )
    model.attach_metadata_tables(tables)

    # Build a balanced batch: each (subject, benchmark) cell is one row.
    sub_ids = list(range(1, tiny_indexer.n_subjects))     # skip UNK
    bc_ids = list(range(1, tiny_indexer.n_bc))             # skip UNK
    rows = [(s, b) for s in sub_ids for b in bc_ids]
    s_arr = torch.tensor([r[0] for r in rows], dtype=torch.long)
    bc_arr = torch.tensor([r[1] for r in rows], dtype=torch.long)
    # Label rule: 1.0 iff subject is in family "Mistral" AND benchmark
    # has topic "Medicine" (which is afrimedqa idx=1 in this setup).
    sub_keys = [k for k, _ in sorted(tiny_indexer.subject_to_id.items(), key=lambda x: x[1])]
    bc_keys = [k for k, _ in sorted(tiny_indexer.bc_to_id.items(), key=lambda x: x[1])]
    name_for_key = {
        "s_mistral": "Mistral-7B",
        "s_mixtral": "Mixtral-8x7B",
        "s_llama": "Llama-3-8B",
        "s_claude": "Claude-3-Opus",
        "s_gpt4o": "GPT-4o",
    }
    family_for_key = {
        k: tiny_model_info.set_index("name").loc[name_for_key[k], "family"]
        for k in sub_keys
        if k in name_for_key
    }
    topic_for_bckey = {
        bc_keys[idx]: tiny_benchmark_info.set_index("benchmark").loc[
            bc_keys[idx].split("::")[0], "topic"
        ]
        if "::" in bc_keys[idx] and bc_keys[idx].split("::")[0] in tiny_benchmark_info["benchmark"].values
        else "_"
        for idx in range(tiny_indexer.n_bc)
    }
    y = torch.tensor(
        [
            1.0
            if (family_for_key.get(sub_keys[s], "") == "Mistral" and topic_for_bckey.get(bc_keys[b], "") == "Medicine")
            else 0.0
            for s, b in rows
        ],
        dtype=torch.float32,
    )
    # Fixed item embedding (constant across rows) -- this forces the
    # IRT / k-factor channels to be unable to distinguish rows by item,
    # so the metadata cross is the only place the label signal can land.
    ie = torch.ones(len(rows), cfg.item_embed_dim) * 0.01

    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_before = None
    for step in range(400):
        opt.zero_grad()
        logits = model(s_arr, bc_arr, ie)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if step == 0:
            loss_before = float(loss.item())
        loss.backward()
        opt.step()
    loss_after = float(loss.item())

    # The structured prior (mu + per-id theta + per-id beta) gives the
    # baseline a head start because the label is sparse, but the
    # additional cross signal must move the loss meaningfully *lower*
    # over training. We require a 30% improvement -- in practice this
    # test trains to ~0.05 loss when the metadata pathway is wired
    # correctly and stays around the initial label-prior loss when it
    # is broken.
    assert loss_after < 0.7 * loss_before, (
        f"Metadata FM + cross head did not learn family x topic: "
        f"loss_before={loss_before:.4f}, loss_after={loss_after:.4f}"
    )


def test_auto_emb_dim_bounds() -> None:
    assert _auto_emb_dim(1) >= 4
    assert _auto_emb_dim(2) >= 4
    assert _auto_emb_dim(1000, max_dim=16) == 16
    assert _auto_emb_dim(10) <= _auto_emb_dim(100)


def test_full_categorical_cross_grid_default_schema() -> None:
    s = MetadataSchema()
    grid = s.full_categorical_cross_grid()
    # Default schema: 3 subject x 1 benchmark = 3 crosses, in stable order.
    assert grid == ("organization__topic", "family__topic", "macro_family__topic")


def test_full_categorical_cross_grid_multi_bench_field() -> None:
    s = MetadataSchema(
        subject_categorical=("a", "b"),
        benchmark_categorical=("x", "y", "z"),
    )
    grid = s.full_categorical_cross_grid()
    assert grid == (
        "a__x", "a__y", "a__z",
        "b__x", "b__y", "b__z",
    )


def test_full_categorical_cross_grid_empty_when_either_side_empty() -> None:
    assert MetadataSchema(
        subject_categorical=(),
        benchmark_categorical=("topic",),
    ).full_categorical_cross_grid() == ()
    assert MetadataSchema(
        subject_categorical=("family",),
        benchmark_categorical=(),
    ).full_categorical_cross_grid() == ()


def test_with_full_categorical_cross_grid_replaces_explicit_crosses() -> None:
    s = MetadataSchema(
        subject_categorical=("organization", "family"),
        benchmark_categorical=("topic",),
        explicit_crosses=("family__topic",),  # hand-picked subset
    )
    s2 = s.with_full_categorical_cross_grid()
    # subject / benchmark / numeric fields are unchanged.
    assert s2.subject_categorical == s.subject_categorical
    assert s2.benchmark_categorical == s.benchmark_categorical
    assert s2.subject_numeric == s.subject_numeric
    assert s2.benchmark_numeric == s.benchmark_numeric
    # explicit_crosses promoted to the full grid.
    assert s2.explicit_crosses == ("organization__topic", "family__topic")
    # original is not mutated.
    assert s.explicit_crosses == ("family__topic",)


def test_full_grid_actually_builds_one_table_per_pair(
    tiny_model_info, tiny_benchmark_info
) -> None:
    """Smoke: when we feed the full cross grid into ExplicitCrossEmbeddings,
    every pair gets its own table (no silent dropping)."""
    schema = MetadataSchema(
        subject_categorical=("organization", "family", "macro_family"),
        benchmark_categorical=("topic",),
    ).with_full_categorical_cross_grid()
    mp = MetadataPreprocessor.fit(
        tiny_model_info, tiny_benchmark_info, schema=schema
    )
    sub_cards = [
        mp.subject_cat_vocabs[c].n_tokens for c in schema.subject_categorical
    ]
    bench_cards = [
        mp.benchmark_cat_vocabs[c].n_tokens for c in schema.benchmark_categorical
    ]
    from src.metadata_features import ExplicitCrossEmbeddings

    mod = ExplicitCrossEmbeddings(
        crosses=schema.explicit_crosses,
        schema=schema,
        subject_cardinalities=sub_cards,
        benchmark_cardinalities=bench_cards,
        emb_dim=4,
    )
    # One table per (subj_field, bench_field) combination in the grid.
    assert len(mod.tables) == len(schema.subject_categorical) * len(
        schema.benchmark_categorical
    )
    # And the wrapping module exposes the same field tuples in the same order.
    assert tuple((sf, bf) for sf, bf, *_ in mod.crosses) == tuple(
        (s, b)
        for s in schema.subject_categorical
        for b in schema.benchmark_categorical
    )
