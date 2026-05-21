"""Smoke test for the new HybridIRTItemKFactorGatedMLP variant.

Run from the repo root with:

    python scripts/smoke_hybrid_irt_kfactor.py

Verifies:
- ``build_model("hybrid_irt_kfactor_gated_mlp", cfg)`` works.
- Forward pass returns finite logits of shape ``[batch]``.
- ``decompose()`` returns separate ``irt``, ``offset``, ``factor``, ``mlp``
  components (and ``theta``, ``beta_i``, ``alpha_i``).
- All existing model registry entries still build.
- The model honors ``override_mlp_zero``, ``force_judge_zero``, and
  ``force_nn_zero`` ablation flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models import MODEL_REGISTRY, ModelConfig, build_model


def main() -> None:
    cfg = ModelConfig(
        n_subjects=3,
        n_benchmark_conditions=2,
        item_embed_dim=8,
        k=4,
        use_pool_features=True,
        pool_feature_dim=9,
        use_cluster_features=True,
        n_clusters=5,
        cluster_embed_dim=3,
        use_judge_features=True,
        judge_feature_dim=4,
        use_nn_features=True,
        nn_feature_dim=8,
    )

    model = build_model("hybrid_irt_kfactor_gated_mlp", cfg)
    model.eval()  # disable dropout so decompose() / forward() are deterministic
    print("Model class:", type(model).__name__)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Param count: {n_params:,}")

    assert model.has_residual is True
    assert model.has_irt_heads is True
    assert model.has_judge_features is True
    assert model.has_nn_features is True

    bsz = 7
    subject_idx = torch.randint(0, cfg.n_subjects, (bsz,))
    bc_idx = torch.randint(0, cfg.n_benchmark_conditions, (bsz,))
    item_emb = torch.randn(bsz, cfg.item_embed_dim)
    pool_feats = torch.randn(bsz, cfg.pool_feature_dim)
    cluster_ids = torch.randint(0, cfg.n_clusters + 1, (bsz,))
    judge_feats = torch.randn(bsz, cfg.judge_feature_dim)
    nn_feats = torch.randn(bsz, cfg.nn_feature_dim)

    out = model(
        subject_idx,
        bc_idx,
        item_emb,
        subject_emb=None,
        pool_feats=pool_feats,
        cluster_ids=cluster_ids,
        judge_feats=judge_feats,
        nn_feats=nn_feats,
    )
    assert out.shape == (bsz,), out.shape
    assert torch.isfinite(out).all().item(), "non-finite logits in forward()"
    print("forward() ok, shape =", tuple(out.shape))

    out_mlp_zero = model(
        subject_idx, bc_idx, item_emb,
        pool_feats=pool_feats, cluster_ids=cluster_ids,
        judge_feats=judge_feats, nn_feats=nn_feats,
        override_mlp_zero=True,
    )
    assert torch.isfinite(out_mlp_zero).all().item()
    print("override_mlp_zero ok")

    out_jzero = model(
        subject_idx, bc_idx, item_emb,
        pool_feats=pool_feats, cluster_ids=cluster_ids,
        judge_feats=judge_feats, nn_feats=nn_feats,
        force_judge_zero=True, force_nn_zero=True,
    )
    assert torch.isfinite(out_jzero).all().item()
    print("force_judge_zero / force_nn_zero ok")

    decomp = model.decompose(
        subject_idx, bc_idx, item_emb,
        pool_feats=pool_feats, cluster_ids=cluster_ids,
        judge_feats=judge_feats, nn_feats=nn_feats,
    )
    for k in ("irt", "offset", "factor", "mlp", "theta", "beta_i", "alpha_i"):
        assert k in decomp, f"missing key in decompose(): {k}"
        assert torch.isfinite(decomp[k]).all().item(), f"non-finite {k}"
    print("decompose keys:", sorted(decomp.keys()))

    # decompose components sum to the unablated forward
    full = decomp["irt"] + decomp["offset"] + decomp["factor"] + decomp["mlp"]
    out_full = model(
        subject_idx, bc_idx, item_emb,
        pool_feats=pool_feats, cluster_ids=cluster_ids,
        judge_feats=judge_feats, nn_feats=nn_feats,
    )
    assert torch.allclose(full, out_full, atol=1e-5), (full - out_full).abs().max()
    print("decompose components sum == forward(): ok")

    # Backwards compatibility: every existing key still builds.
    minimal_cfg = ModelConfig(
        n_subjects=3, n_benchmark_conditions=2, item_embed_dim=8, k=4
    )
    for name in MODEL_REGISTRY:
        m = build_model(name, minimal_cfg)
        print(f"  built {name}: {type(m).__name__}")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
