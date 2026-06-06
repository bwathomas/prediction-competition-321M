"""Tier-2 neural base learners: 1D-CNN, DAE-MLP, FT-Transformer.

These are the user's Tier-2 archetypes (slowest, run last). Each consumes the
same dense+PCA feature matrix the tree-style members use ([N, F] float32 over
the 8 AIDE feature groups + the PCA item-embedding kind), trains with soft
binary cross-entropy (continuous label in [0,1] == the metric), and produces an
OOF probability vector. Unlike the linear/tree members, neural-net inference is
not cheaply compile-able to pure numpy, so the saved state is a torch
``state_dict`` (``model.pt``) plus the standardization stats + the architecture
config needed to rebuild the module; ``apply_state_batch`` reloads and runs a
GPU forward. (Package-free runtime compilation is a separate, later concern; for
the OOF-library / greedy-ES experiment we only need reloadable train-time
models that regenerate the same predictions.)

Uniform contract (mirrors logreg_member / fwfm_member / forest_member):
    fit_<arch>_member(*, X, y, feature_names, ...) -> NeuralMemberState
    state.save(out_dir) / NeuralMemberState.load(out_dir)
    apply_state_batch(state, X) -> [N] float32 probabilities in (eps, 1-eps)

All three share ``_train_torch_member`` (TRAIN-only standardization, group-
stratified internal val for early stopping, Adam, BCEWithLogits on soft labels).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("neural_members")
_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# State (arch config + std stats + torch state_dict on disk)
# ---------------------------------------------------------------------------
@dataclass
class NeuralMemberState:
    arch: str                       # "cnn1d" | "dae_mlp" | "ft_transformer"
    config: dict[str, Any]          # kwargs to rebuild the nn.Module
    feature_names: tuple[str, ...]
    feat_mean: np.ndarray           # [F] float32 (TRAIN-only)
    feat_std: np.ndarray            # [F] float32 (TRAIN-only, zero->1)
    train_loss: float
    val_loss: float
    n_train: int
    # populated only after a fit (held in-memory) or a load (rebuilt lazily):
    _state_dict: Any = field(default=None, repr=False)

    @property
    def feature_dim(self) -> int:
        return int(self.feat_mean.shape[0])

    def save(self, out_dir: Path | str) -> Path:
        import torch
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict, out / "model.pt")
        np.savez_compressed(out / "std.npz",
                            feat_mean=self.feat_mean.astype(np.float32),
                            feat_std=self.feat_std.astype(np.float32))
        (out / "meta.json").write_text(json.dumps({
            "arch": self.arch, "config": self.config,
            "feature_names": list(self.feature_names),
            "feature_dim": int(self.feature_dim),
            "train_loss": float(self.train_loss), "val_loss": float(self.val_loss),
            "n_train": int(self.n_train), "format_version": 1,
        }, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "NeuralMemberState":
        import torch
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "std.npz") as npz:
            fm = npz["feat_mean"].astype(np.float32, copy=False)
            fs = npz["feat_std"].astype(np.float32, copy=False)
        sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
        return cls(arch=str(meta["arch"]), config=dict(meta["config"]),
                   feature_names=tuple(meta["feature_names"]),
                   feat_mean=fm, feat_std=fs,
                   train_loss=float(meta.get("train_loss", 0.0)),
                   val_loss=float(meta.get("val_loss", 0.0)),
                   n_train=int(meta.get("n_train", 0)), _state_dict=sd)


# ---------------------------------------------------------------------------
# nn.Module architectures (built lazily so the module imports without torch)
# ---------------------------------------------------------------------------
def _build_module(arch: str, config: dict[str, Any]):
    import torch
    import torch.nn as nn

    F = int(config["feature_dim"])

    if arch == "cnn1d":
        ch = int(config.get("channels", 32)); k = int(config.get("kernel", 5))
        hid = int(config.get("hid", 64)); pdrop = float(config.get("dropout", 0.1))

        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                pad = k // 2
                self.conv = nn.Sequential(
                    nn.Conv1d(1, ch, k, padding=pad), nn.ReLU(),
                    nn.Conv1d(ch, ch, k, padding=pad), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1))
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Linear(ch, hid), nn.ReLU(),
                    nn.Dropout(pdrop), nn.Linear(hid, 1))

            def forward(self, x):                       # x: [N, F]
                return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)

        return CNN1D()

    if arch == "dae_mlp":
        enc1 = int(config.get("enc1", 256)); bott = int(config.get("bottleneck", 128))
        pdrop = float(config.get("dropout", 0.1))

        class DAEMLP(nn.Module):
            """Joint denoising-autoencoder + supervised head. The encoder feeds
            BOTH a reconstruction decoder (aux MSE on the clean input) and a
            prediction head (BCE). Inference uses head(encoder(x)) with no noise."""
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(F, enc1), nn.ReLU(),
                                         nn.Linear(enc1, bott), nn.ReLU())
                self.dec = nn.Sequential(nn.Linear(bott, enc1), nn.ReLU(),
                                         nn.Linear(enc1, F))
                self.head = nn.Sequential(nn.Dropout(pdrop), nn.Linear(bott, 1))

            def forward(self, x):
                z = self.enc(x)
                return self.head(z).squeeze(-1)

            def forward_train(self, x_noisy, x_clean):
                z = self.enc(x_noisy)
                logit = self.head(z).squeeze(-1)
                recon = self.dec(z)
                return logit, recon

        return DAEMLP()

    if arch == "ft_transformer":
        d = int(config.get("d_token", 16)); nl = int(config.get("n_layers", 2))
        nh = int(config.get("n_heads", 4)); pdrop = float(config.get("dropout", 0.1))

        class FTTransformer(nn.Module):
            """Feature-tokenizer transformer (Gorishniy et al. 2021). Each scalar
            feature -> a d-dim token via a per-feature affine map; a CLS token is
            prepended; a transformer encoder mixes tokens; CLS -> prediction."""
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(F, d)); nn.init.normal_(self.weight, std=0.02)
                self.bias = nn.Parameter(torch.zeros(F, d))
                self.cls = nn.Parameter(torch.empty(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=nh, dim_feedforward=d * 4, dropout=pdrop,
                    batch_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(layer, num_layers=nl)
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

            def forward(self, x):                       # x: [N, F]
                tok = x.unsqueeze(-1) * self.weight + self.bias    # [N, F, d]
                cls = self.cls.expand(x.shape[0], -1, -1)          # [N, 1, d]
                h = self.enc(torch.cat([cls, tok], dim=1))         # [N, F+1, d]
                return self.head(h[:, 0]).squeeze(-1)

        return FTTransformer()

    raise ValueError(f"unknown arch {arch!r}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def apply_state_batch(state: NeuralMemberState, X: np.ndarray,
                      device: str | None = None, batch_size: int = 32768) -> np.ndarray:
    """Standardize with the saved stats, GPU-forward, sigmoid -> [N] float32."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_module(state.arch, state.config).to(device)
    model.load_state_dict(state._state_dict)
    model.eval()
    mu = state.feat_mean.astype(np.float32); sd = state.feat_std.astype(np.float32)
    out = np.empty(int(X.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, X.shape[0], int(batch_size)):
            e = min(s + int(batch_size), X.shape[0])
            xb = (X[s:e].astype(np.float32) - mu) / sd
            logit = model(torch.as_tensor(xb, dtype=torch.float32, device=device))
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
    return np.clip(out, _EPS, 1.0 - _EPS).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared trainer (TRAIN-only standardization + group-stratified early-stop val)
# ---------------------------------------------------------------------------
def _split_idx(N, holdout_group_id, val_fraction, rng):
    if holdout_group_id is None:
        perm = rng.permutation(N); n_val = max(64, int(round(val_fraction * N)))
        return perm[n_val:], perm[:n_val]
    gids = np.asarray(holdout_group_id).reshape(-1)
    ug = np.unique(gids)
    if ug.shape[0] < 2:
        raise ValueError("holdout_group_id needs >=2 groups")
    held = set(int(g) for g in rng.choice(ug, size=max(1, int(round(val_fraction * ug.shape[0]))),
                                          replace=False))
    vmask = np.fromiter((int(g) in held for g in gids), count=N, dtype=bool)
    return np.where(~vmask)[0], np.where(vmask)[0]


def _train_torch_member(*, arch, config, X, y, feature_names, holdout_group_id,
                        learning_rate, weight_decay, epochs, batch_size, val_fraction,
                        early_stopping_patience, seed, device, log_every,
                        step_fn=None, dae_noise=0.0, dae_recon_lambda=0.0):
    """Standardize on TRAIN rows, fit ``arch`` with Adam + BCEWithLogits + early
    stopping; return a populated NeuralMemberState. ``step_fn`` (optional) lets a
    member customize the per-batch loss (used by dae_mlp for the recon aux loss)."""
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    if X.ndim != 2 or y.shape != (int(X.shape[0]),):
        raise ValueError(f"bad shapes X={X.shape} y={y.shape}")
    if int(len(feature_names)) != int(X.shape[1]):
        raise ValueError(f"feature_names {len(feature_names)} != F {X.shape[1]}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(int(seed))
    N, F = int(X.shape[0]), int(X.shape[1])
    tr, va = _split_idx(N, holdout_group_id, val_fraction, rng)

    mu = X[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = X[tr].std(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.where(sd < 1e-9, np.float32(1.0), sd)
    Xtr = ((X[tr] - mu) / sd).astype(np.float32)
    Xva = ((X[va] - mu) / sd).astype(np.float32)

    config = dict(config); config["feature_dim"] = F
    torch.manual_seed(int(seed))
    model = _build_module(arch, config).to(device)
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(y[tr].astype(np.float32), device=device)
    Xva_t = torch.as_tensor(Xva, device=device)
    yva_t = torch.as_tensor(y[va].astype(np.float32), device=device)
    opt = Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    n_tr = int(Xtr_t.shape[0])

    def _val_loss():
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for s in range(0, int(Xva_t.shape[0]), int(batch_size)):
                e = min(s + int(batch_size), int(Xva_t.shape[0]))
                tot += float(bce(model(Xva_t[s:e]), yva_t[s:e]).item()) * (e - s); n += (e - s)
            return tot / max(n, 1)

    best_val, best_sd, since = float("inf"), None, 0
    for ep in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, int(batch_size)):
            bi = perm[s:s + int(batch_size)]
            xb, yb = Xtr_t.index_select(0, bi), ytr_t.index_select(0, bi)
            opt.zero_grad(set_to_none=True)
            if arch == "dae_mlp" and (dae_noise > 0 or dae_recon_lambda > 0):
                # swap-noise: replace a fraction of entries with values resampled
                # from random rows in the batch (Porto-Seguro DAE recipe).
                mask = (torch.rand_like(xb) < float(dae_noise))
                src = xb[torch.randint(0, xb.shape[0], (xb.shape[0],), device=device)]
                x_noisy = torch.where(mask, src, xb)
                logit, recon = model.forward_train(x_noisy, xb)
                loss = bce(logit, yb) + float(dae_recon_lambda) * mse(recon, xb)
            else:
                loss = bce(model(xb), yb)
            loss.backward(); opt.step()
        vl = _val_loss()
        if vl < best_val - 1e-6:
            best_val, since = vl, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if int(log_every) > 0 and ((ep + 1) % int(log_every) == 0 or ep == 0):
            LOG.info("%s: epoch %d/%d val=%.5f best=%.5f (no-improve %d/%d)",
                     arch, ep + 1, int(epochs), vl, best_val, since, int(early_stopping_patience))
        if step_fn is not None:
            step_fn(ep + 1, int(epochs), vl, best_val)
        if since >= int(early_stopping_patience):
            break
    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return NeuralMemberState(
        arch=arch, config=config, feature_names=tuple(str(s) for s in feature_names),
        feat_mean=mu, feat_std=sd, train_loss=float("nan"), val_loss=float(best_val),
        n_train=int(n_tr), _state_dict=best_sd)


# ---------------------------------------------------------------------------
# Public fit_* entry points
# ---------------------------------------------------------------------------
def fit_cnn1d_member(*, X, y, feature_names, channels=32, kernel=5, hid=64, dropout=0.1,
                     learning_rate=1e-3, weight_decay=1e-5, epochs=30, batch_size=8192,
                     val_fraction=0.1, early_stopping_patience=5, seed=0, device=None,
                     holdout_group_id=None, log_every=5, **_ignored) -> NeuralMemberState:
    return _train_torch_member(
        arch="cnn1d",
        config={"channels": channels, "kernel": kernel, "hid": hid, "dropout": dropout},
        X=X, y=y, feature_names=feature_names, holdout_group_id=holdout_group_id,
        learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs,
        batch_size=batch_size, val_fraction=val_fraction,
        early_stopping_patience=early_stopping_patience, seed=seed, device=device,
        log_every=log_every)


def fit_dae_mlp_member(*, X, y, feature_names, enc1=256, bottleneck=128, dropout=0.1,
                       dae_noise=0.15, dae_recon_lambda=1.0, learning_rate=1e-3,
                       weight_decay=1e-5, epochs=40, batch_size=8192, val_fraction=0.1,
                       early_stopping_patience=6, seed=0, device=None,
                       holdout_group_id=None, log_every=5, **_ignored) -> NeuralMemberState:
    return _train_torch_member(
        arch="dae_mlp",
        config={"enc1": enc1, "bottleneck": bottleneck, "dropout": dropout},
        X=X, y=y, feature_names=feature_names, holdout_group_id=holdout_group_id,
        learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs,
        batch_size=batch_size, val_fraction=val_fraction,
        early_stopping_patience=early_stopping_patience, seed=seed, device=device,
        log_every=log_every, dae_noise=dae_noise, dae_recon_lambda=dae_recon_lambda)


def fit_ft_transformer_member(*, X, y, feature_names, d_token=16, n_layers=2, n_heads=4,
                              dropout=0.1, learning_rate=1e-3, weight_decay=1e-5, epochs=25,
                              batch_size=2048, val_fraction=0.1, early_stopping_patience=4,
                              seed=0, device=None, holdout_group_id=None, log_every=2,
                              **_ignored) -> NeuralMemberState:
    return _train_torch_member(
        arch="ft_transformer",
        config={"d_token": d_token, "n_layers": n_layers, "n_heads": n_heads, "dropout": dropout},
        X=X, y=y, feature_names=feature_names, holdout_group_id=holdout_group_id,
        learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs,
        batch_size=batch_size, val_fraction=val_fraction,
        early_stopping_patience=early_stopping_patience, seed=seed, device=device,
        log_every=log_every)


__all__ = ["NeuralMemberState", "apply_state_batch", "fit_cnn1d_member",
           "fit_dae_mlp_member", "fit_ft_transformer_member"]
