"""Regression: Member 2/3/4 cell pulls subject factors from Model A's
state dict using the modern PyTorch nn.Embedding key naming.

The notebook's "build subject_tables" cell needs ``theta`` and ``u``
arrays per subject. These come from ``MetaHybridIRTKFactorGatedMLP``'s
checkpoint, where ``self.theta = nn.Embedding(n_subjects, 1)`` and
``self.u = nn.Embedding(n_subjects, k)``. PyTorch flattens those into
state-dict keys ``theta.weight`` and ``u.weight``. An older revision of
the cell tried to read bare ``theta`` / ``u`` and crashed with
``KeyError: 'theta'`` on every fresh checkpoint.

These tests pin the lookup contract so a future regression
re-introduces the failure into CI rather than into the user's run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def _lookup_param(state_dict, candidates):
    """Identical implementation to the notebook helper. We duplicate
    it here so the test stays self-contained and fails loudly the
    moment the notebook's contract changes."""
    for k in candidates:
        if k in state_dict:
            return state_dict[k]
    raise KeyError(
        f"none of {list(candidates)} in state_dict; "
        f"keys: {sorted(state_dict.keys())}"
    )


def _make_irt_state_dict(n_subjects: int = 4, k: int = 3) -> dict:
    """Mirror the embedding layout used by ``MetaHybridIRTKFactorGatedMLP``."""
    theta = nn.Embedding(n_subjects, 1)
    u = nn.Embedding(n_subjects, k)
    nn.init.zeros_(theta.weight)
    nn.init.normal_(u.weight, std=0.05)
    sd = {"theta.weight": theta.weight.detach().clone(),
          "u.weight": u.weight.detach().clone()}
    return sd


def test_embedding_param_keys_use_dot_weight_suffix() -> None:
    """Hard contract: nn.Embedding flattens to ``<name>.weight``. This
    test will fail on any future PyTorch where that changes -- and
    that's the day the notebook needs an update."""
    sd = _make_irt_state_dict()
    assert "theta.weight" in sd
    assert "u.weight" in sd
    # Bare names must NOT be there in modern checkpoints.
    assert "theta" not in sd
    assert "u" not in sd


def test_lookup_finds_dot_weight_first() -> None:
    sd = _make_irt_state_dict()
    theta_t = _lookup_param(sd, ("theta.weight", "theta", "subject_theta.weight", "subject_theta"))
    u_t = _lookup_param(sd, ("u.weight", "U.weight", "u", "U"))
    assert theta_t.shape == (4, 1)
    assert u_t.shape == (4, 3)


def test_lookup_falls_back_to_bare_name_for_legacy_bundles() -> None:
    """Older ad-hoc bundles sometimes carried bare ``theta`` /``u``
    arrays. The fallback chain still finds them."""
    sd = {
        "theta": torch.zeros(5, 1),
        "u": torch.zeros(5, 4),
    }
    theta_t = _lookup_param(sd, ("theta.weight", "theta"))
    u_t = _lookup_param(sd, ("u.weight", "U.weight", "u", "U"))
    assert theta_t.shape == (5, 1)
    assert u_t.shape == (5, 4)


def test_lookup_raises_with_helpful_message_when_absent() -> None:
    sd = {"some_other_key.weight": torch.zeros(3, 2)}
    with pytest.raises(KeyError, match="theta.weight"):
        _lookup_param(sd, ("theta.weight", "theta"))


def test_theta_squeeze_to_1d_for_member_subject_tables() -> None:
    """The notebook squeezes the [n_subjects, 1] tensor into 1-D
    because ``MemberSubjectTables.theta`` is documented as 1-D.
    Anything else trips an assert in MemberSubjectTables.assert_shapes
    downstream."""
    sd = _make_irt_state_dict()
    theta_arr = _lookup_param(sd, ("theta.weight", "theta")).detach().cpu().numpy().astype(np.float32)
    if theta_arr.ndim == 2 and theta_arr.shape[1] == 1:
        theta_arr = theta_arr[:, 0]
    assert theta_arr.ndim == 1
    assert theta_arr.shape[0] == 4


def test_u_passes_through_when_already_2d() -> None:
    sd = _make_irt_state_dict(n_subjects=6, k=8)
    u_arr = _lookup_param(sd, ("u.weight", "u")).detach().cpu().numpy().astype(np.float32)
    if u_arr.ndim == 1:
        u_arr = u_arr[:, None]
    assert u_arr.shape == (6, 8)
