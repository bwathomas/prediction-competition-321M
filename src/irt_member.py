"""Amortized item-IRT ensemble member (standalone torch member).

This module lifts the inlined IRT classes out of
:mod:`src.export_submission` (``_IRTItemBase``, ``_ItemIRTHeads``,
``_ResidualIRTItem`` and the K-factor variant ``_IRTItemKFactorMLP``)
into a standalone ensemble member with the project's standard contract:

    fit_irt_member(...) -> IRTMemberState  # offline (torch) trainer
    IRTMemberState.save / .load            # round-trips the fitted model
    apply_batch(state, ...) -> np.ndarray  # runtime probabilities
    apply_one(state, ...)   -> float       # single-row convenience

Architecture (amortized item-IRT)
---------------------------------
Each (subject ``s``, benchmark-condition ``bc``, item-embedding ``ie``)
triple maps to a logit::

    beta_i, alpha_i = item_heads(ie)        # item tower over the live item emb
    c_irt           = alpha_i * (theta_s - beta_i)
    c_offset        = beta_bc + mu
    eta             = c_irt + c_offset
                      [ + lambda_resid * residual([ie, theta, beta_i, alpha_i]) ]
                      [ + rho_factor   * (u_s . v_i) / sqrt(k)  (k-factor) ]

``theta`` is a *per-subject* embedding -- a pure lookup table -- so it is
shipped as a plain ``.npy`` and gathered with numpy at runtime. The
benchmark-condition offset ``beta_bc`` is likewise a per-condition lookup
shipped as ``.npy``. Everything that consumes the *live* item embedding
(the item tower ``irt_heads``, the optional residual MLP, and the optional
k-factor ``item_map``/``u``) is a small torch module whose ``state_dict``
is shipped (safetensors preferred, ``.pt`` fallback) and re-instantiated at
inference. Runtime therefore touches only ``torch`` + ``numpy`` (+ optional
``safetensors``), all on the Codabench whitelist.

Faithful lift
-------------
The runtime torch sub-modules below (:class:`_ItemIRTHeads`,
:class:`_ItemParameterMap`, :class:`_DenseResidual`, :class:`_GatedResidual`
and the ``_IRTItem*`` model classes) reproduce the export_submission
classes parameter-for-parameter with the *same* ``state_dict`` key layout,
so a checkpoint trained here loads cleanly into the export_submission
runtime and the two forwards agree to floating-point tolerance. The unit
test ``tests/test_irt_member.py`` asserts this directly on shared weights.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("irt_member")

_EPS = 1.0e-6


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Runtime torch sub-modules -- faithful mirrors of src/export_submission.py
#
# These are intentionally byte-for-byte equivalent (same submodule names,
# same parameter shapes, same forward math) to the inlined classes in
# export_submission.py so that state_dicts are interchangeable.
# ---------------------------------------------------------------------------


def _build_item_modules():
    """Build the torch modules lazily (torch is offline+runtime, not import-time)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ItemIRTHeads(nn.Module):
        """Mirror of ``export_submission._ItemIRTHeads`` (item tower)."""

        def __init__(self, item_dim: int, hidden: int, dropout: float = 0.0):
            super().__init__()
            self.beta_head = nn.Sequential(
                nn.LayerNorm(item_dim),
                nn.Linear(item_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.alpha_head = nn.Sequential(
                nn.LayerNorm(item_dim),
                nn.Linear(item_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self._alpha_pre_bias = 0.54

        def forward(self, ie):
            beta = self.beta_head(ie).squeeze(-1)
            alpha_pre = self.alpha_head(ie).squeeze(-1) + self._alpha_pre_bias
            return beta, F.softplus(alpha_pre)

    class _ItemParameterMap(nn.Module):
        """Mirror of ``export_submission._ItemParameterMap`` (k-factor tower)."""

        def __init__(self, item_embed_dim: int, k: int, hidden: int, dropout: float = 0.1):
            super().__init__()
            self.norm = nn.LayerNorm(item_embed_dim)
            self.fc1 = nn.Linear(item_embed_dim, hidden)
            self.drop = nn.Dropout(dropout)
            self.head_factor = nn.Linear(hidden, k)
            self.head_diff = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.silu(self.fc1(self.norm(x)))
            h = self.drop(h)
            return self.head_factor(h), self.head_diff(h).squeeze(-1)

    class _DenseResidual(nn.Module):
        """Mirror of ``export_submission._DenseResidual``."""

        def __init__(self, in_dim: int, hidden: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, hidden)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.silu(self.fc1(self.norm(x)))
            h = self.drop(h)
            return self.fc2(h).squeeze(-1)

    class _GatedResidual(nn.Module):
        """Mirror of ``export_submission._GatedResidual``."""

        def __init__(self, in_dim: int, hidden: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(in_dim)
            self.gate = nn.Linear(in_dim, hidden)
            self.up = nn.Linear(in_dim, hidden)
            self.down = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            x = self.norm(x)
            h = F.silu(self.gate(x)) * self.up(x)
            h = self.drop(h)
            return self.down(h).squeeze(-1)

    def _residual_input_dim_irt(cfg: dict) -> int:
        # [ie, theta, beta_i, alpha_i] -> item_embed_dim + 3
        return int(cfg["item_embed_dim"]) + 3

    def _residual_features_irt(comps, ie):
        return torch.cat(
            [
                ie,
                comps["theta"].unsqueeze(-1),
                comps["beta_i"].unsqueeze(-1),
                comps["alpha_i"].unsqueeze(-1),
            ],
            dim=-1,
        )

    class _IRTItemBase(nn.Module):
        """Mirror of ``export_submission._IRTItemBase``.

        ``theta`` (per-subject) and ``beta`` (per-benchmark-condition) are
        ``nn.Embedding`` lookup tables; ``irt_heads`` is the item tower over
        the live item embedding.
        """

        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg
            self.mu = nn.Parameter(torch.zeros(1))
            self.theta = nn.Embedding(cfg["n_subjects"], 1)
            self.beta = nn.Embedding(cfg["n_benchmark_conditions"], 1)
            self.irt_heads = _ItemIRTHeads(
                item_dim=cfg["item_embed_dim"],
                hidden=cfg["item_map_hidden_dim"],
                dropout=cfg.get("dropout", 0.0),
            )

        def _irt_components(self, s, bc, ie):
            theta = self.theta(s).squeeze(-1)
            bc_off = self.beta(bc).squeeze(-1)
            beta_i, alpha_i = self.irt_heads(ie)
            c_irt = alpha_i * (theta - beta_i)
            c_offset = bc_off + self.mu
            return {
                "irt": c_irt,
                "offset": c_offset,
                "theta": theta,
                "beta_i": beta_i,
                "alpha_i": alpha_i,
            }

    class _IRTItemKFactor(_IRTItemBase):
        """Plain amortized IRT (no residual)."""

        def forward(self, s, bc, ie):
            comps = self._irt_components(s, bc, ie)
            return comps["irt"] + comps["offset"]

    class _ResidualIRTItem(_IRTItemBase):
        """Mirror of ``export_submission._ResidualIRTItem`` (IRT + residual MLP)."""

        residual_cls = _DenseResidual

        def __init__(self, cfg: dict):
            super().__init__(cfg)
            in_dim = _residual_input_dim_irt(cfg)
            self.residual = self.residual_cls(
                in_dim=in_dim,
                hidden=cfg["residual_hidden_dim"],
                dropout=cfg.get("dropout", 0.0),
            )
            self.lambda_resid = nn.Parameter(
                torch.tensor(float(cfg.get("lambda_resid_init", 0.1))),
                requires_grad=bool(cfg.get("lambda_resid_trainable", True)),
            )

        def forward(self, s, bc, ie):
            comps = self._irt_components(s, bc, ie)
            x = _residual_features_irt(comps, ie)
            return comps["irt"] + comps["offset"] + self.lambda_resid * self.residual(x)

    class _IRTItemKFactorMLP(_ResidualIRTItem):
        """Dense-residual variant (mirror of ``_IRTItemKFactorMLP``)."""

        residual_cls = _DenseResidual

    class _IRTItemKFactorGatedMLP(_ResidualIRTItem):
        """Gated-residual variant (mirror of ``_IRTItemKFactorGatedMLP``)."""

        residual_cls = _GatedResidual

    return {
        "ItemIRTHeads": _ItemIRTHeads,
        "ItemParameterMap": _ItemParameterMap,
        "DenseResidual": _DenseResidual,
        "GatedResidual": _GatedResidual,
        "IRTItemBase": _IRTItemBase,
        "IRTItemKFactor": _IRTItemKFactor,
        "ResidualIRTItem": _ResidualIRTItem,
        "IRTItemKFactorMLP": _IRTItemKFactorMLP,
        "IRTItemKFactorGatedMLP": _IRTItemKFactorGatedMLP,
    }


# Map a variant name -> model class key. Default is the dense-residual IRT,
# which matches the AIDE "M1 IRT-MLP" head.
_VARIANT_TO_KEY = {
    "irt": "IRTItemKFactor",
    "irt_mlp": "IRTItemKFactorMLP",
    "irt_gated_mlp": "IRTItemKFactorGatedMLP",
}


def build_irt_model(cfg: dict, variant: str = "irt_mlp"):
    """Instantiate the runtime torch model for ``variant`` given ``cfg``.

    Shared by the trainer, the runtime path and the faithful-lift test so
    there is a single source of truth for the architecture.
    """
    mods = _build_item_modules()
    key = _VARIANT_TO_KEY.get(str(variant))
    if key is None:
        raise ValueError(
            f"unknown IRT variant {variant!r}; expected one of "
            f"{sorted(_VARIANT_TO_KEY)}"
        )
    return mods[key](cfg)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class IRTMemberState:
    """Fitted amortized item-IRT member.

    Shipped artifacts (written by :meth:`save`):

    * ``theta.npy``        -- ``[n_subjects]`` per-subject ability lookup.
    * ``beta_bc.npy``      -- ``[n_benchmark_conditions]`` per-condition offset.
    * ``item_tower.safetensors`` (or ``item_tower.pt``) -- the ``state_dict``
      of everything that consumes the live item embedding (item tower +
      optional residual MLP + scalars ``mu``/``lambda_resid``). The full
      model ``state_dict`` is stored; the ``theta``/``beta`` embedding
      weights are *also* present there (for an exact export_submission
      round-trip) but the ``.npy`` copies are the runtime source of truth.
    * ``config.json``      -- model cfg + variant + provenance.
    """

    cfg: dict
    variant: str
    theta: np.ndarray            # [n_subjects] float32
    beta_bc: np.ndarray          # [n_benchmark_conditions] float32
    state_dict: dict             # full torch state_dict as numpy arrays
    item_embed_dim: int
    n_subjects: int
    n_benchmark_conditions: int

    # ---- provenance ----
    n_train: int = 0
    n_pos: int = 0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    _use_safetensors: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=np.float32).reshape(-1)
        self.beta_bc = np.asarray(self.beta_bc, dtype=np.float32).reshape(-1)
        if self.theta.shape[0] != int(self.n_subjects):
            raise ValueError(
                f"theta len {self.theta.shape[0]} != n_subjects {self.n_subjects}"
            )
        if self.beta_bc.shape[0] != int(self.n_benchmark_conditions):
            raise ValueError(
                f"beta_bc len {self.beta_bc.shape[0]} != "
                f"n_benchmark_conditions {self.n_benchmark_conditions}"
            )
        if not np.all(np.isfinite(self.theta)):
            raise ValueError("IRTMemberState: theta contains NaN/Inf")
        if not np.all(np.isfinite(self.beta_bc)):
            raise ValueError("IRTMemberState: beta_bc contains NaN/Inf")

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "theta.npy", self.theta.astype(np.float32))
        np.save(out / "beta_bc.npy", self.beta_bc.astype(np.float32))

        sd = {k: np.asarray(v, dtype=np.float32) for k, v in self.state_dict.items()}
        wrote_safetensors = False
        if self._use_safetensors:
            try:
                from safetensors.numpy import save_file

                # safetensors requires contiguous arrays.
                save_file(
                    {k: np.ascontiguousarray(v) for k, v in sd.items()},
                    str(out / "item_tower.safetensors"),
                )
                wrote_safetensors = True
            except Exception:  # pragma: no cover - exercised only w/o safetensors
                wrote_safetensors = False
        if not wrote_safetensors:
            # ``.pt`` fallback: a plain npz under a .pt name keeps the runtime
            # numpy-only (no torch.load needed to read weights back). Pass
            # ``allow_pickle=False`` explicitly so the ``**sd`` keys bind to
            # ``kwds`` (not the keyword-only ``allow_pickle`` param); the saved
            # arrays are plain float32, so pickling is never needed and the
            # loader (np.load default allow_pickle=False) reads them back by key.
            np.savez(out / "item_tower.pt", allow_pickle=False, **sd)

        cfg = {
            "cfg": _jsonable(self.cfg),
            "variant": str(self.variant),
            "item_embed_dim": int(self.item_embed_dim),
            "n_subjects": int(self.n_subjects),
            "n_benchmark_conditions": int(self.n_benchmark_conditions),
            "state_dict_keys": list(sd.keys()),
            "weights_file": (
                "item_tower.safetensors" if wrote_safetensors else "item_tower.pt.npz"
            ),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "format_version": 1,
        }
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "IRTMemberState":
        d = Path(in_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        theta = np.load(d / "theta.npy").astype(np.float32, copy=False)
        beta_bc = np.load(d / "beta_bc.npy").astype(np.float32, copy=False)

        weights_file = cfg.get("weights_file", "item_tower.safetensors")
        st_path = d / "item_tower.safetensors"
        npz_path = d / "item_tower.pt.npz"
        if weights_file == "item_tower.safetensors" and st_path.exists():
            from safetensors.numpy import load_file

            sd = {k: np.asarray(v, dtype=np.float32) for k, v in load_file(str(st_path)).items()}
        else:
            # ``np.savez`` appends .npz to "item_tower.pt"; tolerate either name.
            load_path = npz_path if npz_path.exists() else (d / "item_tower.pt")
            with np.load(load_path) as npz:
                sd = {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}

        return cls(
            cfg=cfg["cfg"],
            variant=str(cfg["variant"]),
            theta=theta,
            beta_bc=beta_bc,
            state_dict=sd,
            item_embed_dim=int(cfg["item_embed_dim"]),
            n_subjects=int(cfg["n_subjects"]),
            n_benchmark_conditions=int(cfg["n_benchmark_conditions"]),
            n_train=int(cfg.get("n_train", 0)),
            n_pos=int(cfg.get("n_pos", 0)),
            train_loss=float(cfg.get("train_loss", float("nan"))),
            val_loss=float(cfg.get("val_loss", float("nan"))),
        )

    # ---- runtime torch model reconstruction ----
    def _build_model(self):
        """Re-instantiate the torch model and load the shipped weights."""
        import torch

        model = build_irt_model(self.cfg, self.variant)
        sd = {
            k: torch.as_tensor(np.asarray(v, dtype=np.float32))
            for k, v in self.state_dict.items()
        }
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Runtime inference (torch + numpy only)
# ---------------------------------------------------------------------------


def apply_batch(
    state: IRTMemberState,
    *,
    subject_ids: np.ndarray,
    item_emb: np.ndarray,
    bc_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Pure torch+numpy forward. Returns ``[N]`` float32 probabilities.

    ``item_emb`` is the ``[N, D]`` *per-row* item-embedding matrix (the
    caller has already gathered each row's item embedding). ``subject_ids``
    and ``bc_ids`` are ``[N]`` integer id arrays; out-of-range ids route to
    0 (subjects) / 0 (benchmark condition) and contribute the corresponding
    table entry, matching the embedding-lookup semantics.
    """
    import torch

    s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    n_rows = int(s.shape[0])
    e = np.asarray(item_emb, dtype=np.float32)
    if e.ndim != 2 or int(e.shape[1]) != int(state.item_embed_dim):
        raise ValueError(
            f"item_emb shape {e.shape} must be (N, {state.item_embed_dim})"
        )
    if int(e.shape[0]) != n_rows:
        raise ValueError(
            f"item_emb has {e.shape[0]} rows, expected {n_rows}"
        )
    if bc_ids is None:
        bc = np.zeros(n_rows, dtype=np.int64)
    else:
        bc = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
        if int(bc.shape[0]) != n_rows:
            raise ValueError(f"bc_ids has {bc.shape[0]} rows, expected {n_rows}")

    # Clamp ids into valid lookup range (graceful for cold subjects/conditions).
    s = np.where((s >= 0) & (s < state.n_subjects), s, 0)
    bc = np.where((bc >= 0) & (bc < state.n_benchmark_conditions), bc, 0)

    model = state._build_model()
    with torch.no_grad():
        s_t = torch.as_tensor(s, dtype=torch.long)
        bc_t = torch.as_tensor(bc, dtype=torch.long)
        ie_t = torch.as_tensor(e, dtype=torch.float32)
        eta = model(s_t, bc_t, ie_t)
        z = eta.detach().cpu().numpy().astype(np.float64).reshape(-1)
    p = _sigmoid_stable(z)
    return np.clip(p, _EPS, 1.0 - _EPS).astype(np.float32)


def apply_one(
    state: IRTMemberState,
    *,
    subject_id: int,
    item_emb: np.ndarray,
    bc_id: int | None = None,
) -> float:
    """Single-row convenience; equals ``apply_batch`` on a 1-row input."""
    sid = np.array([int(subject_id)], dtype=np.int64)
    ie = np.asarray(item_emb, dtype=np.float32).reshape(1, -1)
    bc = None if bc_id is None else np.array([int(bc_id)], dtype=np.int64)
    return float(apply_batch(state, subject_ids=sid, item_emb=ie, bc_ids=bc)[0])


def apply_state_batch(state: IRTMemberState, **kwargs) -> np.ndarray:
    return apply_batch(state, **kwargs)


# ---------------------------------------------------------------------------
# Offline trainer (torch; notebook/Colab-only)
# ---------------------------------------------------------------------------


def fit_irt_member(
    *,
    labels: np.ndarray,                       # [N] in {0, 1}
    subject_ids: np.ndarray,                  # [N] int subject id
    n_subjects: int,
    item_emb_unique: np.ndarray,              # [n_uniq, D] float32
    row_to_uniq: np.ndarray,                  # [N] int into item_emb_unique
    bc_ids: np.ndarray | None = None,         # [N] int benchmark-condition id
    n_bcs: int = 0,
    variant: str = "irt_mlp",
    k: int = 1,
    item_map_hidden_dim: int = 64,
    residual_hidden_dim: int = 64,
    dropout: float = 0.0,
    lambda_resid_init: float = 0.1,
    lambda_resid_trainable: bool = True,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-5,
    epochs: int = 40,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    early_stopping_patience: int = 6,
    seed: int = 0,
    device: str | None = None,
    show_progress: bool = True,
) -> IRTMemberState:
    """Train an amortized item-IRT member via Adam + early stopping.

    The model gathers each minibatch's item rows from ``item_emb_unique``
    using ``row_to_uniq`` (it never materialises a dense ``[N, D]`` matrix),
    mirroring the memory contract of :func:`src.mlp_member.fit_mlp_member`.

    Returns an :class:`IRTMemberState` whose per-subject ``theta`` and
    per-condition ``beta_bc`` are extracted from the trained embeddings, and
    whose full ``state_dict`` is retained for the package-free runtime path.
    """
    import torch
    import torch.nn as nn

    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    N = int(y.shape[0])
    if N == 0:
        raise ValueError("fit_irt_member: empty label array")

    s_all_np = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    if s_all_np.shape[0] != N:
        raise ValueError(f"subject_ids len {s_all_np.shape[0]} != N={N}")
    r2u_np = np.asarray(row_to_uniq, dtype=np.int64).reshape(-1)
    if r2u_np.shape[0] != N:
        raise ValueError(f"row_to_uniq len {r2u_np.shape[0]} != N={N}")
    item_emb_unique = np.asarray(item_emb_unique, dtype=np.float32)
    if item_emb_unique.ndim != 2:
        raise ValueError("item_emb_unique must be 2-D [n_uniq, D]")
    item_embed_dim = int(item_emb_unique.shape[1])

    n_bcs = int(n_bcs)
    if bc_ids is None:
        bc_np = np.zeros(N, dtype=np.int64)
        n_bcs = max(1, n_bcs)
    else:
        bc_np = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
        if bc_np.shape[0] != N:
            raise ValueError(f"bc_ids len {bc_np.shape[0]} != N={N}")
        n_bcs = max(int(n_bcs), int(bc_np.max()) + 1) if N else max(1, n_bcs)
    n_bcs = max(1, n_bcs)
    n_subjects = max(int(n_subjects), int(s_all_np.max()) + 1) if N else int(n_subjects)
    n_subjects = max(1, int(n_subjects))

    cfg = {
        "n_subjects": int(n_subjects),
        "n_benchmark_conditions": int(n_bcs),
        "item_embed_dim": int(item_embed_dim),
        "item_map_hidden_dim": int(item_map_hidden_dim),
        "residual_hidden_dim": int(residual_hidden_dim),
        "k": int(k),
        "dropout": float(dropout),
        "lambda_resid_init": float(lambda_resid_init),
        "lambda_resid_trainable": bool(lambda_resid_trainable),
    }

    rng = np.random.default_rng(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- internal train/val split ----
    perm = rng.permutation(N)
    n_val = max(1, int(round(val_fraction * N)))
    val_mask = np.zeros(N, dtype=bool)
    val_mask[perm[:n_val]] = True
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    if tr_idx.size == 0 or va_idx.size == 0:
        tr_idx = np.arange(N)
        va_idx = np.arange(N)

    torch.manual_seed(int(seed))
    model = build_irt_model(cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    item_t = torch.from_numpy(item_emb_unique).to(device)
    s_t = torch.from_numpy(s_all_np).to(device)
    bc_t = torch.from_numpy(bc_np).to(device)
    r2u_t = torch.from_numpy(r2u_np).to(device)
    y_t = torch.from_numpy(y).to(device)

    def _run_batch(idx_t):
        s_b = s_t[idx_t]
        bc_b = bc_t[idx_t]
        ie_b = item_t[r2u_t[idx_t]]
        return model(s_b, bc_b, ie_b)

    tr_idx_t = torch.from_numpy(tr_idx.astype(np.int64)).to(device)
    va_idx_t = torch.from_numpy(va_idx.astype(np.int64)).to(device)

    best_val = float("inf")
    best_state: dict | None = None
    patience = 0
    for ep in range(int(epochs)):
        model.train()
        bperm = torch.randperm(tr_idx.size, device=device)
        tr_shuf = tr_idx_t[bperm]
        for bstart in range(0, tr_idx.size, batch_size):
            b = tr_shuf[bstart:bstart + batch_size]
            logits = _run_batch(b)
            loss = bce(logits, y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # ---- validate ----
        model.eval()
        with torch.no_grad():
            vloss_sum = 0.0
            vn = 0
            for bstart in range(0, va_idx.size, batch_size):
                b = va_idx_t[bstart:bstart + batch_size]
                logits = _run_batch(b)
                vloss_sum += float(bce(logits, y_t[b])) * int(b.shape[0])
                vn += int(b.shape[0])
            vloss = vloss_sum / max(1, vn)
        if show_progress:
            LOG.info("irt epoch %d/%d val_loss=%.5f", ep + 1, epochs, vloss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- export ----
    sd = {k_: v.detach().cpu().numpy().astype(np.float32) for k_, v in model.state_dict().items()}
    theta = sd["theta.weight"].reshape(-1).astype(np.float32)
    beta_bc = sd["beta.weight"].reshape(-1).astype(np.float32)

    # ---- final train loss (sampled) ----
    net_train_loss = float("nan")
    with torch.no_grad():
        try:
            s_n = min(tr_idx.size, 200_000)
            s_b = tr_idx_t[:s_n]
            logits = _run_batch(s_b)
            net_train_loss = float(bce(logits, y_t[s_b]))
        except Exception:  # pragma: no cover - diagnostics only
            pass

    return IRTMemberState(
        cfg=cfg,
        variant=str(variant),
        theta=theta,
        beta_bc=beta_bc,
        state_dict=sd,
        item_embed_dim=int(item_embed_dim),
        n_subjects=int(n_subjects),
        n_benchmark_conditions=int(n_bcs),
        n_train=int(tr_idx.size),
        n_pos=int(y[tr_idx].sum()),
        train_loss=float(net_train_loss),
        val_loss=float(best_val),
    )


__all__ = [
    "IRTMemberState",
    "fit_irt_member",
    "build_irt_model",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
]
