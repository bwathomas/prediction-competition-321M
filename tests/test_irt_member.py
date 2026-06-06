"""Tests for the standalone amortized item-IRT member (``src.irt_member``).

Covers the standard member contract -- ``apply_one == apply_batch`` and a
``save``/``load`` round-trip -- plus a FAITHFUL-LIFT check that the lifted
runtime architecture reproduces the inlined IRT classes in
``src.export_submission`` exactly (same forward on identical weights),
proving the lift is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.irt_member import (
    IRTMemberState,
    apply_batch,
    apply_one,
    build_irt_model,
    fit_irt_member,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_irt_synth(n=3000, n_subj=25, n_items=150, n_bcs=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    item_emb = rng.normal(size=(n_items, d)).astype(np.float32)
    theta_true = rng.normal(size=n_subj).astype(np.float32)
    diff_true = rng.normal(size=n_items).astype(np.float32)
    bc_off_true = 0.3 * rng.normal(size=n_bcs).astype(np.float32)

    row_to_uniq = rng.integers(0, n_items, size=n).astype(np.int64)
    subject_ids = rng.integers(0, n_subj, size=n).astype(np.int64)
    bc_ids = rng.integers(0, n_bcs, size=n).astype(np.int64)

    # 1-PL-ish signal: theta_s - diff_i + bc offset.
    logit = (
        theta_true[subject_ids]
        - diff_true[row_to_uniq]
        + bc_off_true[bc_ids]
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return item_emb, row_to_uniq, subject_ids, bc_ids, y, n_subj, n_bcs


# ---------------------------------------------------------------------------
# Contract: fit / apply
# ---------------------------------------------------------------------------


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def test_fit_apply_no_nan_and_learns():
    item_emb, r2u, sids, bcs, y, n_subj, n_bcs = _make_irt_synth()
    state = fit_irt_member(
        labels=y,
        subject_ids=sids,
        n_subjects=n_subj,
        item_emb_unique=item_emb,
        row_to_uniq=r2u,
        bc_ids=bcs,
        n_bcs=n_bcs,
        variant="irt_mlp",
        item_map_hidden_dim=32,
        residual_hidden_dim=32,
        epochs=15,
        batch_size=512,
        seed=1,
        show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p = apply_batch(state, subject_ids=sids, item_emb=per_row_item, bc_ids=bcs)
    assert p.shape == (y.shape[0],)
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0 and p.max() <= 1.0
    prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(prior * np.log(prior) + (1 - prior) * np.log(1 - prior))
    assert _nll(p, y) < nll_prior


def test_apply_one_matches_batch():
    item_emb, r2u, sids, bcs, y, n_subj, n_bcs = _make_irt_synth(n=800)
    state = fit_irt_member(
        labels=y,
        subject_ids=sids,
        n_subjects=n_subj,
        item_emb_unique=item_emb,
        row_to_uniq=r2u,
        bc_ids=bcs,
        n_bcs=n_bcs,
        variant="irt_mlp",
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        epochs=5,
        batch_size=256,
        seed=2,
        show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p_batch = apply_batch(state, subject_ids=sids, item_emb=per_row_item, bc_ids=bcs)
    for i in range(6):
        p1 = apply_one(
            state,
            subject_id=int(sids[i]),
            item_emb=per_row_item[i],
            bc_id=int(bcs[i]),
        )
        assert abs(p1 - float(p_batch[i])) < 1e-6


def test_save_load_roundtrip(tmp_path):
    item_emb, r2u, sids, bcs, y, n_subj, n_bcs = _make_irt_synth(n=1200)
    state = fit_irt_member(
        labels=y,
        subject_ids=sids,
        n_subjects=n_subj,
        item_emb_unique=item_emb,
        row_to_uniq=r2u,
        bc_ids=bcs,
        n_bcs=n_bcs,
        variant="irt_mlp",
        item_map_hidden_dim=16,
        residual_hidden_dim=16,
        epochs=4,
        batch_size=256,
        seed=3,
        show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p0 = apply_batch(state, subject_ids=sids, item_emb=per_row_item, bc_ids=bcs)
    state.save(tmp_path / "irt")
    loaded = IRTMemberState.load(tmp_path / "irt")
    p1 = apply_batch(loaded, subject_ids=sids, item_emb=per_row_item, bc_ids=bcs)
    assert np.max(np.abs(p0 - p1)) < 1e-6


@pytest.mark.parametrize("variant", ["irt", "irt_mlp", "irt_gated_mlp"])
def test_all_variants_save_load(tmp_path, variant):
    item_emb, r2u, sids, bcs, y, n_subj, n_bcs = _make_irt_synth(n=600)
    state = fit_irt_member(
        labels=y,
        subject_ids=sids,
        n_subjects=n_subj,
        item_emb_unique=item_emb,
        row_to_uniq=r2u,
        bc_ids=bcs,
        n_bcs=n_bcs,
        variant=variant,
        item_map_hidden_dim=8,
        residual_hidden_dim=8,
        epochs=2,
        batch_size=256,
        seed=4,
        show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p0 = apply_batch(state, subject_ids=sids, item_emb=per_row_item, bc_ids=bcs)
    out = state.save(tmp_path / variant)
    p1 = apply_batch(
        IRTMemberState.load(out), subject_ids=sids, item_emb=per_row_item, bc_ids=bcs
    )
    assert np.max(np.abs(p0 - p1)) < 1e-6


# ---------------------------------------------------------------------------
# FAITHFUL-LIFT: irt_member forward == export_submission IRT forward
# ---------------------------------------------------------------------------


def _export_irt_namespace():
    """Exec the *exact* runtime model source that export_submission ships.

    The IRT classes live inside the ``_RUNTIME_MODEL_PY`` source string that
    export_submission renders into the package-free ``model.py`` (they are
    not module-level attributes, and the full string runs artifact I/O at
    import time). We slice out the self-contained model-class block -- from
    ``class _ItemParameterMap`` through the last IRT class -- which is pure
    class/function definitions, prepend the torch imports, and exec it. The
    faithful-lift check then compares against the exact bytes that ship in
    the runtime ``model.py``.
    """
    es = pytest.importorskip("src.export_submission")
    src_text = getattr(es, "_RUNTIME_MODEL_PY", None)
    if not isinstance(src_text, str):
        pytest.skip("export_submission._RUNTIME_MODEL_PY unavailable")
    lines = src_text.splitlines()

    def _find(prefix):
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                return i
        return None

    start = _find("class _ItemParameterMap")
    # The IRT model classes end right before the hybrid-k-factor class.
    end = _find("class _HybridIRTItemKFactorGatedMLP")
    if start is None or end is None or end <= start:
        pytest.skip("could not locate IRT class block in runtime model source")
    block = "\n".join(lines[start:end])
    header = (
        "import math\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
    )
    ns: dict = {}
    try:
        exec(compile(header + block, "<export_submission IRT classes>", "exec"), ns)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not exec runtime IRT class block: {exc!r}")
    if "_IRTItemKFactorMLP" not in ns or "_IRTItemKFactor" not in ns:
        pytest.skip("runtime IRT classes not found after exec")
    return ns


def test_faithful_lift_matches_export_submission():
    """Instantiate the export_submission IRT-MLP class on identical random
    weights and assert irt_member's forward agrees to <1e-4 on a synthetic
    batch -- this proves the lift out of export_submission is correct."""
    es = _export_irt_namespace()

    n_subj, n_bcs, n_items, d = 12, 5, 60, 16
    cfg = {
        "n_subjects": n_subj,
        "n_benchmark_conditions": n_bcs,
        "item_embed_dim": d,
        "item_map_hidden_dim": 24,
        "residual_hidden_dim": 24,
        "k": 1,
        "dropout": 0.0,
        "lambda_resid_init": 0.2,
        "lambda_resid_trainable": True,
    }

    torch.manual_seed(123)
    lifted = build_irt_model(cfg, variant="irt_mlp").eval()
    export_model = es["_IRTItemKFactorMLP"](cfg).eval()

    # Copy lifted weights into the export model -> identical parameters.
    export_model.load_state_dict(lifted.state_dict(), strict=True)

    rng = np.random.default_rng(7)
    n = 40
    s = torch.as_tensor(rng.integers(0, n_subj, size=n), dtype=torch.long)
    bc = torch.as_tensor(rng.integers(0, n_bcs, size=n), dtype=torch.long)
    ie = torch.as_tensor(rng.normal(size=(n, d)).astype(np.float32))

    with torch.no_grad():
        eta_lift = lifted(s, bc, ie).detach().cpu().numpy()
        eta_exp = export_model(s, bc, ie).detach().cpu().numpy()

    assert eta_lift.shape == eta_exp.shape == (n,)
    assert np.max(np.abs(eta_lift - eta_exp)) < 1e-4


def test_faithful_lift_plain_irt_matches_export_submission():
    """Same faithful-lift check for the no-residual plain IRT variant."""
    es = _export_irt_namespace()

    n_subj, n_bcs, n_items, d = 10, 3, 40, 12
    cfg = {
        "n_subjects": n_subj,
        "n_benchmark_conditions": n_bcs,
        "item_embed_dim": d,
        "item_map_hidden_dim": 20,
        "residual_hidden_dim": 20,
        "k": 1,
        "dropout": 0.0,
    }
    torch.manual_seed(321)
    lifted = build_irt_model(cfg, variant="irt").eval()
    export_model = es["_IRTItemKFactor"](cfg).eval()
    export_model.load_state_dict(lifted.state_dict(), strict=True)

    rng = np.random.default_rng(11)
    n = 32
    s = torch.as_tensor(rng.integers(0, n_subj, size=n), dtype=torch.long)
    bc = torch.as_tensor(rng.integers(0, n_bcs, size=n), dtype=torch.long)
    ie = torch.as_tensor(rng.normal(size=(n, d)).astype(np.float32))
    with torch.no_grad():
        eta_lift = lifted(s, bc, ie).detach().cpu().numpy()
        eta_exp = export_model(s, bc, ie).detach().cpu().numpy()
    assert np.max(np.abs(eta_lift - eta_exp)) < 1e-4
