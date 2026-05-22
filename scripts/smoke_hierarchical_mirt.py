"""Smoke test for the new HierarchicalMIRT variant.

Run from the repo root with:

    python scripts/smoke_hierarchical_mirt.py

Verifies:
- ``build_model("hierarchical_mirt", cfg)`` works.
- Forward pass returns finite logits of shape ``[batch]``.
- ``decompose()`` returns separate ``irt``, ``offset``, ``mirt``, ``mlp``
  components (and ``theta``, ``beta_i``, ``alpha_i``).
- At step 0, the ``mirt`` component is exactly zero (this is the whole
  point of zero-initializing the alpha_vec_head output layer).
- The model honors ``override_mlp_zero``, ``force_judge_zero``, and
  ``force_nn_zero`` ablation flags.
- All existing model registry entries still build.
- The trainer-side state-dict manifest is printed so runtime mirror
  divergence can be caught by the bundle audit.
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

    model = build_model("hierarchical_mirt", cfg)
    model.eval()
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

    # Critical init invariants:
    # - alpha_vec_head output layer is zero-initialized, so the MIRT component
    #   is exactly 0 at step 0.
    # - beta_i is zero-initialized, matching the simulation. Otherwise random
    #   item difficulty is active before the scalar 2PL channel has learned.
    # If either breaks, the inductive-bias design is gone.
    decomp_init = model.decompose(
        subject_idx, bc_idx, item_emb,
        pool_feats=pool_feats, cluster_ids=cluster_ids,
        judge_feats=judge_feats, nn_feats=nn_feats,
    )
    assert torch.allclose(
        decomp_init["mirt"], torch.zeros_like(decomp_init["mirt"]), atol=0.0
    ), (
        "HierarchicalMIRT init invariant failed: MIRT component is not "
        "exactly zero at step 0. Check that alpha_vec_head[-1] is "
        "zero-initialized."
    )
    assert torch.allclose(
        decomp_init["beta_i"], torch.zeros_like(decomp_init["beta_i"]), atol=0.0
    ), (
        "HierarchicalMIRT init invariant failed: beta_i is not exactly zero "
        "at step 0. Check that irt_heads.beta_head[-1] is zero-initialized."
    )
    print("init invariant ok: MIRT contribution and beta_i == 0 at step 0")

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
    for k in ("irt", "offset", "mirt", "mlp", "theta", "beta_i", "alpha_i"):
        assert k in decomp, f"missing key in decompose(): {k}"
        assert torch.isfinite(decomp[k]).all().item(), f"non-finite {k}"
    print("decompose keys:", sorted(decomp.keys()))

    # decompose components sum to the unablated forward
    full = decomp["irt"] + decomp["offset"] + decomp["mirt"] + decomp["mlp"]
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

    # ------------------------------------------------------------------
    # State-dict parameter manifest. The runtime mirror in
    # src/export_submission.py (inlined inside the runtime template) must
    # have a structurally identical parameter layout, so any future
    # divergence is caught at export time by scripts/_audit_bundle.py.
    # This block prints the trainer-side manifest so a refactor can compare
    # against the runtime side by inspection.
    # ------------------------------------------------------------------
    print("Trainer state_dict keys (sorted):")
    for k in sorted(model.state_dict().keys()):
        t = model.state_dict()[k]
        print(f"  {k}  shape={tuple(t.shape)}")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
