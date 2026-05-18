"""Training loop for the k-factor / residual ablation.

The trainer is intentionally small and deterministic:

- BCEWithLogitsLoss.
- AdamW with linear warmup + cosine decay (or plain linear) scheduler.
- Mixed precision: bf16 when the GPU supports it, otherwise fp32.
- Early stopping on validation BCE.
- Best checkpoint saved by item-cold-start validation log-loss.
- Optional multi-seed runs returning a per-seed result list.

The trainer consumes pre-aligned numpy arrays (subject_ids, bc_ids,
item_emb, subject_emb, labels). The notebook is responsible for building
those arrays from the dataframe + embedding lookups.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from .models import (
    Indexer,
    LookupDataset,
    ModelConfig,
    build_model,
)

LOG = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Best-effort determinism across numpy / torch / python.

    Note: with bf16 + cudnn benchmark there's a small amount of run-to-run
    nondeterminism remaining. We accept that in exchange for A100 throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 30
    warmup_steps: int = 200
    scheduler: str = "cosine"   # cosine | linear | none
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    bf16: bool = True
    num_workers: int = 0


@dataclass
class TrainResult:
    run_id: str
    model_name: str
    seed: int
    k: int
    epoch_best: int
    best_val_log_loss: float
    best_val_brier: float
    best_val_auc: float | None
    history: list[dict] = field(default_factory=list)
    checkpoint_path: str = ""
    metadata_path: str = ""
    n_train: int = 0
    n_val: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def _make_scheduler(opt, total_steps: int, warmup: int, kind: str):
    """Linear-warmup + (cosine|linear|constant) decay."""
    kind = (kind or "none").lower()
    warmup = max(0, int(warmup))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if kind == "none":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = float(min(max(progress, 0.0), 1.0))
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        # cosine
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics used inline by the trainer (full metrics live in src/eval.py)
# ---------------------------------------------------------------------------


def _log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y_true - p) ** 2))


def _auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    yb = (y_true >= 0.5).astype(int)
    if yb.sum() == 0 or (1 - yb).sum() == 0:
        return None
    if not np.allclose(y_true, yb, atol=1e-6):
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = yb == 1
    return float(
        (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0)
        / max(1, pos.sum() * (~pos).sum())
    )


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------


def _move_batch(batch, device: str):
    return [b.to(device, non_blocking=True) for b in batch]


def evaluate_model(
    model: nn.Module,
    val_ds: LookupDataset,
    *,
    device: str,
    batch_size: int,
    bf16: bool,
) -> tuple[float, float, float | None, np.ndarray, np.ndarray]:
    """Run val_ds through model and return (log_loss, brier, auc, p, y)."""
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.inference_mode():
        with autocast_ctx:
            for batch in loader:
                s, bc, ie, se, y = _move_batch(batch, device)
                logits = model(s, bc, ie, se if se.shape[-1] > 0 else None)
                p = torch.sigmoid(logits).float().cpu().numpy()
                preds.append(p)
                targets.append(y.float().cpu().numpy())
    p = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    y = np.concatenate(targets) if targets else np.zeros(0, dtype=np.float32)
    return _log_loss(y, p), _brier(y, p), _auc(y, p), p, y


def train_one(
    *,
    model_name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_ds: LookupDataset,
    val_ds: LookupDataset,
    indexer: Indexer,
    seed: int,
    run_id: str,
    checkpoint_dir: str | os.PathLike[str],
    device: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Train a single (model_name, seed) configuration."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, model_cfg).to(device)
    LOG.info("Built %s with %s parameters", model_name, _count_params(model))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    opt = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    total_steps = max(1, len(train_loader) * train_cfg.epochs)
    sched = _make_scheduler(
        opt, total_steps, train_cfg.warmup_steps, train_cfg.scheduler
    )
    loss_fn = nn.BCEWithLogitsLoss()
    bf16 = train_cfg.bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast_ctx_factory = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16))
        if device.startswith("cuda")
        else (lambda: torch.amp.autocast("cpu", enabled=False))
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{run_id}.pt"
    meta_path = checkpoint_dir / f"{run_id}.json"

    best_ll = float("inf")
    best_brier = float("nan")
    best_auc: float | None = None
    best_epoch = -1
    history: list[dict] = []
    patience = 0
    start = time.time()
    step = 0
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            s, bc, ie, se, y = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx_factory():
                logits = model(s, bc, ie, se if se.shape[-1] > 0 else None)
                loss = loss_fn(logits, y)
            loss.backward()
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            loss_sum += float(loss.item()) * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / max(1, n)
        val_ll, val_brier, val_auc, _val_p, _val_y = evaluate_model(
            model, val_ds,
            device=device,
            batch_size=max(1024, train_cfg.batch_size),
            bf16=bf16,
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_log_loss": val_ll,
                "val_brier": val_brier,
                "val_auc": val_auc,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        improved = val_ll < best_ll - 1e-6
        if improved:
            best_ll = val_ll
            best_brier = val_brier
            best_auc = val_auc
            best_epoch = epoch
            patience = 0
            _save_checkpoint(model, model_cfg, train_cfg, indexer, ckpt_path)
        else:
            patience += 1
        LOG.info(
            "epoch %d train=%.5f val_ll=%.5f val_brier=%.5f val_auc=%s lr=%.5g %s",
            epoch,
            train_loss,
            val_ll,
            val_brier,
            f"{val_auc:.4f}" if val_auc is not None else "n/a",
            opt.param_groups[0]["lr"],
            "(best)" if improved else "",
        )
        if patience >= train_cfg.early_stopping_patience:
            LOG.info("Early stopping at epoch %d", epoch)
            break

    elapsed = time.time() - start
    result = TrainResult(
        run_id=run_id,
        model_name=model_name,
        seed=seed,
        k=model_cfg.k,
        epoch_best=best_epoch,
        best_val_log_loss=best_ll,
        best_val_brier=best_brier,
        best_val_auc=best_auc,
        history=history,
        checkpoint_path=str(ckpt_path),
        metadata_path=str(meta_path),
        n_train=len(train_ds),
        n_val=len(val_ds),
        elapsed_seconds=elapsed,
    )
    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "seed": seed,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "indexer": indexer.to_dict(),
        "result": {
            **asdict(result),
            "history": history,
        },
        "extra": dict(extra_metadata or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default))
    return result


def _save_checkpoint(model, model_cfg: ModelConfig, train_cfg: TrainConfig, indexer: Indexer, path: Path) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "indexer": indexer.to_dict(),
        },
        path,
    )


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unserializable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Multi-seed loop
# ---------------------------------------------------------------------------


def train_many(
    *,
    model_name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_ds: LookupDataset,
    val_ds: LookupDataset,
    indexer: Indexer,
    seeds: list[int],
    checkpoint_dir: str | os.PathLike[str],
    run_id_prefix: str = "",
    device: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> list[TrainResult]:
    """Run `train_one` across multiple seeds. Returns a list of TrainResults."""
    out: list[TrainResult] = []
    for seed in seeds:
        rid = f"{run_id_prefix}{model_name}_k{model_cfg.k}_seed{seed}"
        LOG.info("=== Training %s ===", rid)
        res = train_one(
            model_name=model_name,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_ds=train_ds,
            val_ds=val_ds,
            indexer=indexer,
            seed=seed,
            run_id=rid,
            checkpoint_dir=checkpoint_dir,
            device=device,
            extra_metadata=extra_metadata,
        )
        out.append(res)
    return out


__all__ = [
    "TrainConfig",
    "TrainResult",
    "evaluate_model",
    "set_seed",
    "train_many",
    "train_one",
]
