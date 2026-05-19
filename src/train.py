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

This version adds:
- tqdm batch-level progress bars for train and validation.
- live rows/sec, ETA, loss, LR, and GPU memory display.
- JSONL progress-file logging for worker/notebook environments where
  terminal output is hidden.
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
from tqdm.auto import tqdm

from .models import (
    Indexer,
    LookupDataset,
    ModelConfig,
    build_model,
    irt_regularization,
    model_has_irt_heads,
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

    # Progress / diagnostics.
    progress: bool = True
    log_every_batches: int = 10

    # If nonempty, append JSONL progress events here.
    # This is useful when stdout from workers is hidden.
    progress_file: str = ""


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
# Formatting / progress helpers
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _gpu_mem_string() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{allocated:.1f}GB/{reserved:.1f}GB"


def _gpu_mem_dict() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_gb": None,
            "gpu_reserved_gb": None,
            "gpu_max_allocated_gb": None,
        }
    return {
        "gpu_allocated_gb": float(torch.cuda.memory_allocated() / 1024**3),
        "gpu_reserved_gb": float(torch.cuda.memory_reserved() / 1024**3),
        "gpu_max_allocated_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
    }


def _write_progress_file(progress_file: str, event: dict) -> None:
    """Append one JSONL progress event.

    This should never crash training. It intentionally catches all exceptions.
    """
    if not progress_file:
        return

    try:
        path = Path(progress_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        event = dict(event)
        event["time"] = time.time()
        event["time_readable"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=_json_default) + "\n")
            f.flush()
    except Exception:
        pass


def _write_progress(train_cfg: TrainConfig, event: dict) -> None:
    _write_progress_file(train_cfg.progress_file, event)


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
# Train / eval helpers
# ---------------------------------------------------------------------------


def _move_batch(batch, device: str):
    return [b.to(device, non_blocking=True) for b in batch]


def _unpack_batch(batch, device: str):
    """Unpack the 8-tuple yielded by ``LookupDataset`` onto ``device``.

    Returns ``(s, bc, ie, se_or_none, pf_or_none, ci_or_none, y, jf_or_none)``.
    Empty optional channels are returned as ``None`` so the model can fast-path
    around them. The judge-features tensor ``jf`` has ``shape[-1] == 0`` when
    judge scoring is disabled or no scores were attached.
    """
    moved = _move_batch(batch, device)
    s, bc, ie, se, pf, ci, y, jf = moved
    se_use = se if se.shape[-1] > 0 else None
    pf_use = pf if pf.shape[-1] > 0 else None
    ci_use = ci if (ci.numel() > 0 and ci.dim() >= 1) else None
    jf_use = jf if jf.shape[-1] > 0 else None
    return s, bc, ie, se_use, pf_use, ci_use, y, jf_use


def _forward_model(model, s, bc, ie, se, pf, ci, jf=None):
    """Call ``model.forward`` with the channels the model supports.

    All current variants accept the full kwargs signature; the wrapper exists
    so callers can stay agnostic if we later add variants with different
    signatures. ``jf`` is the judge feature tensor (or ``None`` when judge
    features are disabled).
    """
    return model(s, bc, ie, se, pf, ci, jf)


def evaluate_model(
    model: nn.Module,
    val_ds: LookupDataset,
    *,
    device: str,
    batch_size: int,
    bf16: bool,
    progress: bool = True,
    desc: str = "val",
    progress_file: str = "",
    log_every_batches: int = 10,
    run_id: str = "",
    epoch: int | None = None,
    epochs: int | None = None,
) -> tuple[float, float, float | None, np.ndarray, np.ndarray]:
    """Run val_ds through model and return (log_loss, brier, auc, p, y)."""
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast("cpu", enabled=False)
    )

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )

    t0 = time.time()
    n_seen = 0

    _write_progress_file(
        progress_file,
        {
            "type": "val_start",
            "run_id": run_id,
            "epoch": epoch,
            "epochs": epochs,
            "n_val": int(len(val_ds)),
            "batch_size": int(batch_size),
            "n_batches": int(len(loader)),
            "device": device,
            **_gpu_mem_dict(),
        },
    )

    with torch.inference_mode():
        with autocast_ctx:
            for batch_idx, batch in enumerate(iterator, start=1):
                s, bc, ie, se, pf, ci, y, jf = _unpack_batch(batch, device)
                logits = _forward_model(model, s, bc, ie, se, pf, ci, jf)
                p = torch.sigmoid(logits).float().cpu().numpy()

                preds.append(p)
                targets.append(y.float().cpu().numpy())

                n_seen += y.shape[0]
                elapsed = time.time() - t0
                rows_per_sec = n_seen / max(elapsed, 1e-9)
                remaining = (len(val_ds) - n_seen) / max(rows_per_sec, 1e-9)

                if batch_idx % max(1, log_every_batches) == 0 or batch_idx == len(loader):
                    iterator.set_postfix(
                        {
                            "rows/s": f"{rows_per_sec:,.0f}",
                            "seen": f"{n_seen:,}/{len(val_ds):,}",
                            "ETA": _fmt_seconds(remaining),
                        }
                    )

                    _write_progress_file(
                        progress_file,
                        {
                            "type": "val_batch",
                            "run_id": run_id,
                            "epoch": epoch,
                            "epochs": epochs,
                            "batch_idx": int(batch_idx),
                            "n_batches": int(len(loader)),
                            "seen": int(n_seen),
                            "n_val": int(len(val_ds)),
                            "rows_per_sec": float(rows_per_sec),
                            "eta_seconds": float(remaining),
                            **_gpu_mem_dict(),
                        },
                    )

    p = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    y = np.concatenate(targets) if targets else np.zeros(0, dtype=np.float32)

    val_ll = _log_loss(y, p)
    val_brier = _brier(y, p)
    val_auc = _auc(y, p)

    _write_progress_file(
        progress_file,
        {
            "type": "val_end",
            "run_id": run_id,
            "epoch": epoch,
            "epochs": epochs,
            "n_val": int(len(val_ds)),
            "elapsed_seconds": float(time.time() - t0),
            "val_log_loss": float(val_ll),
            "val_brier": float(val_brier),
            "val_auc": float(val_auc) if val_auc is not None else None,
            **_gpu_mem_dict(),
        },
    )

    return val_ll, val_brier, val_auc, p, y


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------


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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

    LOG.info(
        "Starting %s | n_train=%d n_val=%d batch_size=%d train_batches=%d "
        "device=%s bf16=%s gpu_mem=%s",
        run_id,
        len(train_ds),
        len(val_ds),
        train_cfg.batch_size,
        len(train_loader),
        device,
        bf16,
        _gpu_mem_string(),
    )

    _write_progress(
        train_cfg,
        {
            "type": "run_start",
            "run_id": run_id,
            "model_name": model_name,
            "seed": int(seed),
            "k": int(model_cfg.k),
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "batch_size": int(train_cfg.batch_size),
            "train_batches": int(len(train_loader)),
            "epochs": int(train_cfg.epochs),
            "device": device,
            "bf16": bool(bf16),
            "param_count": int(_count_params(model)),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            **_gpu_mem_dict(),
        },
    )

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        epoch_t0 = time.time()

        _write_progress(
            train_cfg,
            {
                "type": "epoch_start",
                "run_id": run_id,
                "model_name": model_name,
                "epoch": int(epoch),
                "epochs": int(train_cfg.epochs),
                "n_train": int(len(train_ds)),
                "batch_size": int(train_cfg.batch_size),
                "n_batches": int(len(train_loader)),
                "lr": float(opt.param_groups[0]["lr"]),
                **_gpu_mem_dict(),
            },
        )

        iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"{run_id} epoch {epoch}/{train_cfg.epochs} train",
            leave=False,
            dynamic_ncols=True,
            disable=not train_cfg.progress,
        )

        for batch_idx, batch in enumerate(iterator, start=1):
            s, bc, ie, se, pf, ci, y, jf = _unpack_batch(batch, device)

            opt.zero_grad(set_to_none=True)

            with autocast_ctx_factory():
                logits = _forward_model(model, s, bc, ie, se, pf, ci, jf)
                loss = loss_fn(logits, y)
                # Soft IRT regularization: keep beta from exploding and
                # log(alpha) close to 0 so the IRT head and the residual MLP
                # don't collude in degenerate ways.
                if model_has_irt_heads(model):
                    beta_i, alpha_i = model.irt_heads(ie)
                    loss = loss + irt_regularization(
                        beta_i,
                        alpha_i,
                        lambda_beta=float(model_cfg.irt_lambda_beta),
                        lambda_alpha=float(model_cfg.irt_lambda_alpha),
                    )

            loss.backward()

            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            opt.step()
            sched.step()
            step += 1

            bs = y.shape[0]
            loss_sum += float(loss.item()) * bs
            n += bs

            if (
                batch_idx % max(1, train_cfg.log_every_batches) == 0
                or batch_idx == len(train_loader)
            ):
                elapsed = time.time() - epoch_t0
                rows_per_sec = n / max(elapsed, 1e-9)
                remaining_rows = len(train_ds) - n
                eta = remaining_rows / max(rows_per_sec, 1e-9)
                train_loss_so_far = loss_sum / max(1, n)

                postfix = {
                    "loss": f"{train_loss_so_far:.5f}",
                    "rows/s": f"{rows_per_sec:,.0f}",
                    "seen": f"{n:,}/{len(train_ds):,}",
                    "ETA": _fmt_seconds(eta),
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                }

                if device.startswith("cuda") and torch.cuda.is_available():
                    postfix["gpu_mem"] = _gpu_mem_string()

                iterator.set_postfix(postfix)

                _write_progress(
                    train_cfg,
                    {
                        "type": "train_batch",
                        "run_id": run_id,
                        "model_name": model_name,
                        "epoch": int(epoch),
                        "epochs": int(train_cfg.epochs),
                        "batch_idx": int(batch_idx),
                        "n_batches": int(len(train_loader)),
                        "seen": int(n),
                        "n_train": int(len(train_ds)),
                        "batch_size": int(train_cfg.batch_size),
                        "train_loss_so_far": float(train_loss_so_far),
                        "rows_per_sec": float(rows_per_sec),
                        "eta_seconds": float(eta),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "global_step": int(step),
                        **_gpu_mem_dict(),
                    },
                )

        train_loss = loss_sum / max(1, n)
        train_elapsed = time.time() - epoch_t0

        val_t0 = time.time()
        val_ll, val_brier, val_auc, _val_p, _val_y = evaluate_model(
            model,
            val_ds,
            device=device,
            batch_size=max(1024, train_cfg.batch_size),
            bf16=bf16,
            progress=train_cfg.progress,
            desc=f"{run_id} epoch {epoch}/{train_cfg.epochs} val",
            progress_file=train_cfg.progress_file,
            log_every_batches=train_cfg.log_every_batches,
            run_id=run_id,
            epoch=epoch,
            epochs=train_cfg.epochs,
        )
        val_elapsed = time.time() - val_t0

        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_log_loss": val_ll,
                "val_brier": val_brier,
                "val_auc": val_auc,
                "lr": opt.param_groups[0]["lr"],
                "train_elapsed_seconds": train_elapsed,
                "val_elapsed_seconds": val_elapsed,
                "gpu_mem": _gpu_mem_string() if device.startswith("cuda") else "n/a",
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

        _write_progress(
            train_cfg,
            {
                "type": "epoch_end",
                "run_id": run_id,
                "model_name": model_name,
                "epoch": int(epoch),
                "epochs": int(train_cfg.epochs),
                "train_loss": float(train_loss),
                "val_log_loss": float(val_ll),
                "val_brier": float(val_brier),
                "val_auc": float(val_auc) if val_auc is not None else None,
                "best_val_log_loss": float(best_ll),
                "best_epoch": int(best_epoch),
                "improved": bool(improved),
                "patience": int(patience),
                "early_stopping_patience": int(train_cfg.early_stopping_patience),
                "elapsed_seconds": float(time.time() - start),
                "train_elapsed_seconds": float(train_elapsed),
                "val_elapsed_seconds": float(val_elapsed),
                "lr": float(opt.param_groups[0]["lr"]),
                **_gpu_mem_dict(),
            },
        )

        LOG.info(
            "epoch %d/%d train=%.5f val_ll=%.5f val_brier=%.5f val_auc=%s "
            "lr=%.5g train_time=%s val_time=%s gpu_mem=%s %s",
            epoch,
            train_cfg.epochs,
            train_loss,
            val_ll,
            val_brier,
            f"{val_auc:.4f}" if val_auc is not None else "n/a",
            opt.param_groups[0]["lr"],
            _fmt_seconds(train_elapsed),
            _fmt_seconds(val_elapsed),
            _gpu_mem_string() if device.startswith("cuda") else "n/a",
            "(best)" if improved else "",
        )

        if patience >= train_cfg.early_stopping_patience:
            LOG.info("Early stopping at epoch %d", epoch)
            _write_progress(
                train_cfg,
                {
                    "type": "early_stop",
                    "run_id": run_id,
                    "model_name": model_name,
                    "epoch": int(epoch),
                    "epochs": int(train_cfg.epochs),
                    "best_val_log_loss": float(best_ll),
                    "best_epoch": int(best_epoch),
                    "patience": int(patience),
                    "elapsed_seconds": float(time.time() - start),
                    **_gpu_mem_dict(),
                },
            )
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

    _write_progress(
        train_cfg,
        {
            "type": "run_end",
            "run_id": run_id,
            "model_name": model_name,
            "seed": int(seed),
            "k": int(model_cfg.k),
            "epoch_best": int(best_epoch),
            "best_val_log_loss": float(best_ll),
            "best_val_brier": float(best_brier),
            "best_val_auc": float(best_auc) if best_auc is not None else None,
            "elapsed_seconds": float(elapsed),
            "checkpoint_path": str(ckpt_path),
            "metadata_path": str(meta_path),
            **_gpu_mem_dict(),
        },
    )

    return result


def _save_checkpoint(
    model,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    indexer: Indexer,
    path: Path,
) -> None:
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
    if isinstance(obj, Path):
        return str(obj)
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

    for seed_idx, seed in enumerate(seeds, start=1):
        rid = f"{run_id_prefix}{model_name}_k{model_cfg.k}_seed{seed}"
        LOG.info("=== Training %s ===", rid)

        _write_progress(
            train_cfg,
            {
                "type": "seed_start",
                "run_id": rid,
                "model_name": model_name,
                "seed": int(seed),
                "seed_idx": int(seed_idx),
                "n_seeds": int(len(seeds)),
                "k": int(model_cfg.k),
            },
        )

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

        _write_progress(
            train_cfg,
            {
                "type": "seed_end",
                "run_id": rid,
                "model_name": model_name,
                "seed": int(seed),
                "seed_idx": int(seed_idx),
                "n_seeds": int(len(seeds)),
                "k": int(model_cfg.k),
                "best_val_log_loss": float(res.best_val_log_loss),
                "elapsed_seconds": float(res.elapsed_seconds),
            },
        )

    return out


__all__ = [
    "TrainConfig",
    "TrainResult",
    "evaluate_model",
    "set_seed",
    "train_many",
    "train_one",
]
