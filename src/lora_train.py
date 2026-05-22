"""LoRA fine-tuning of the item-side encoder, joint with the head.

This is a separate training loop on top of the existing head-only trainer
(``src.train``). It is fundamentally different in two ways:

1. The Qwen3-Embedding-4B encoder is *no longer frozen*. We wrap it with
   PEFT LoRA adapters in its attention projections (q/k/v/o) and train
   those adapters jointly with the existing head. Subjects stay cached
   lookups (no LoRA on the subject side) -- subjects are not cold-start.

2. The cached item embeddings are no longer authoritative inputs. Every
   step forwards raw item tokens through the adapter-augmented encoder so
   the produced item embedding reflects the current adapter weights. The
   judge / NN / pool / cluster features remain frozen inputs computed once
   against the original frozen embeddings -- this is intentional, they are
   stable signals that ground the head while the encoder is perturbed.

A single epoch over ~300k unique items can be 1-3 hours on a single A100.
The run must be overnight-safe: we step-checkpoint to Google Drive every
``checkpoint_every_steps`` (atomic tmp+rename), eval every
``eval_every_steps`` against the *current* adapter (no cache), and on
restart auto-resume from the latest Drive checkpoint with optimizer,
scheduler, and RNG state restored.

Out of scope: LoRA on subject text, multi-GPU/FSDP, hyperparameter sweeps,
recomputing judge / NN features against fine-tuned embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .colab_tqdm import get_tqdm

tqdm = get_tqdm()

from .embeddings import (
    EncoderConfig,
    TransformerEmbedder,
    _last_token_pool,
    _mean_pool,
)
from .models import (
    Indexer,
    ModelConfig,
    build_model,
    irt_regularization,
    model_has_irt_heads,
)
from .tokenized_items import TokenizedItemCache

LOG = logging.getLogger("lora_train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainConfig:
    """Configuration for the LoRA fine-tuning loop.

    See ``configs/default.yaml -> lora:`` for prose explanations of each
    field. The defaults here intentionally match the YAML so a caller can
    instantiate with no args for diagnostic runs.
    """

    enabled: bool = False
    base_checkpoint: str | None = None
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    layers_to_transform: list[int] | None = None
    gradient_checkpointing: bool = True

    encoder_lr: float = 5.0e-6
    head_lr: float = 5.0e-4
    weight_decay_head: float = 0.01
    epochs: int = 1
    batch_size_items: int = 8
    grad_accum_steps: int = 4
    max_length: int = 1024
    warmup_steps: int = 50
    # Planned-budget scheduler knobs. ``max_train_steps = 0`` recovers the
    # full-epoch schedule; ``> 0`` schedules over the planned short-run
    # budget so a cosine/linear decay actually decays during a 1k-10k
    # LoRA pass instead of being effectively constant-LR.
    scheduler: str = "cosine"
    max_train_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.05
    bf16: bool = True
    use_8bit_optimizer: bool = False

    checkpoint_every_steps: int = 200
    eval_every_steps: int = 1000
    drive_checkpoint_dir: str = ""
    keep_last_n_checkpoints: int = 3
    resume: bool = True
    max_runtime_minutes: int = 600
    init_check_tol: float = 1.0e-3
    val_batch_size_items: int = 16
    # Cap on val batches used by the periodic / final / init eval pass.
    # 0 = full val; > 0 = a deterministic random subset of
    # ``val_eval_max_batches * val_batch_size_items`` rows.
    val_eval_max_batches: int = 0
    val_eval_seed: int = 12345
    oom_fallback: bool = True

    export_mode: str = "adapter_only"
    hf_upload_repo: str = ""

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "LoRATrainConfig":
        d = dict(d or {})
        targets = d.get("target_modules") or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        layers = d.get("layers_to_transform", None)
        if layers is not None and len(layers) == 0:
            layers = None
        return cls(
            enabled=bool(d.get("enabled", False)),
            base_checkpoint=d.get("base_checkpoint") or None,
            r=int(d.get("r", 16)),
            alpha=int(d.get("alpha", 32)),
            dropout=float(d.get("dropout", 0.05)),
            target_modules=list(targets),
            layers_to_transform=(
                [int(x) for x in layers] if layers is not None else None
            ),
            gradient_checkpointing=bool(d.get("gradient_checkpointing", True)),
            encoder_lr=float(d.get("encoder_lr", 5.0e-6)),
            head_lr=float(d.get("head_lr", 5.0e-4)),
            weight_decay_head=float(d.get("weight_decay_head", 0.01)),
            epochs=int(d.get("epochs", 1)),
            batch_size_items=int(d.get("batch_size_items", 8)),
            grad_accum_steps=int(d.get("grad_accum_steps", 4)),
            max_length=int(d.get("max_length", 1024)),
            warmup_steps=int(d.get("warmup_steps", 50)),
            scheduler=str(d.get("scheduler", "cosine")),
            max_train_steps=int(d.get("max_train_steps", 0)),
            warmup_ratio=float(d.get("warmup_ratio", 0.0)),
            min_lr_ratio=float(d.get("min_lr_ratio", 0.05)),
            bf16=bool(d.get("bf16", True)),
            use_8bit_optimizer=bool(d.get("use_8bit_optimizer", False)),
            checkpoint_every_steps=int(d.get("checkpoint_every_steps", 200)),
            eval_every_steps=int(d.get("eval_every_steps", 1000)),
            drive_checkpoint_dir=str(d.get("drive_checkpoint_dir", "")),
            keep_last_n_checkpoints=int(d.get("keep_last_n_checkpoints", 3)),
            resume=bool(d.get("resume", True)),
            max_runtime_minutes=int(d.get("max_runtime_minutes", 600)),
            init_check_tol=float(d.get("init_check_tol", 1.0e-3)),
            val_batch_size_items=int(d.get("val_batch_size_items", 16)),
            val_eval_max_batches=int(d.get("val_eval_max_batches", 0)),
            val_eval_seed=int(d.get("val_eval_seed", 12345)),
            oom_fallback=bool(d.get("oom_fallback", True)),
            export_mode=str(d.get("export_mode", "adapter_only")),
            hf_upload_repo=str(d.get("hf_upload_repo", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset: indices into the precomputed tokenized item cache + frozen feats
# ---------------------------------------------------------------------------


class LoRARowDataset(torch.utils.data.Dataset):
    """One row per training example.

    Stores integer indices into ``TokenizedItemCache.input_ids`` plus the
    already-aligned subject / bc / pool / cluster / judge / nn / label
    arrays. The collator fetches the variable-length ``input_ids`` arrays
    by index at batch construction time so we never materialize a padded
    ``[N, max_length]`` tensor in memory.
    """

    def __init__(
        self,
        *,
        item_token_idx: np.ndarray,    # [N] int32, indices into cache.input_ids
        subject_idx: np.ndarray,       # [N] int64
        bc_idx: np.ndarray,            # [N] int64
        labels: np.ndarray,            # [N] float32
        subject_emb: np.ndarray | None = None,   # [N, D] or None
        pool_feats: np.ndarray | None = None,    # [N, P] or None
        cluster_ids: np.ndarray | None = None,   # [N] or None
        judge_feats: np.ndarray | None = None,   # [N, 4] or None
        nn_feats: np.ndarray | None = None,      # [N, 8] or None
        token_lens: np.ndarray | None = None,    # [N] int32 for bucketing
    ):
        self.item_token_idx = np.asarray(item_token_idx, dtype=np.int64)
        self.subject_idx = np.asarray(subject_idx, dtype=np.int64)
        self.bc_idx = np.asarray(bc_idx, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        n = self.labels.shape[0]
        self.subject_emb = (
            np.asarray(subject_emb, dtype=np.float32)
            if subject_emb is not None
            else None
        )
        self.pool_feats = (
            np.asarray(pool_feats, dtype=np.float32)
            if pool_feats is not None
            else None
        )
        self.cluster_ids = (
            np.asarray(cluster_ids, dtype=np.int64)
            if cluster_ids is not None
            else None
        )
        self.judge_feats = (
            np.asarray(judge_feats, dtype=np.float32)
            if judge_feats is not None
            else None
        )
        self.nn_feats = (
            np.asarray(nn_feats, dtype=np.float32)
            if nn_feats is not None
            else None
        )
        self.token_lens = (
            np.asarray(token_lens, dtype=np.int32)
            if token_lens is not None
            else None
        )
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        out = {
            "item_token_idx": int(self.item_token_idx[idx]),
            "subject_idx": int(self.subject_idx[idx]),
            "bc_idx": int(self.bc_idx[idx]),
            "label": float(self.labels[idx]),
        }
        if self.subject_emb is not None:
            out["subject_emb"] = self.subject_emb[idx]
        if self.pool_feats is not None:
            out["pool_feats"] = self.pool_feats[idx]
        if self.cluster_ids is not None:
            out["cluster_id"] = int(self.cluster_ids[idx])
        if self.judge_feats is not None:
            out["judge_feats"] = self.judge_feats[idx]
        if self.nn_feats is not None:
            out["nn_feats"] = self.nn_feats[idx]
        return out


def _make_collator(
    *,
    token_cache: TokenizedItemCache,
    pad_token_id: int,
):
    """Return a collator that pads ``input_ids`` to per-batch max length.

    The model's pad token id is reused from the encoder tokenizer; Qwen3
    uses ``<|endoftext|>`` as both eos and pad. Pad positions have
    ``attention_mask=0``.
    """

    def _collate(batch: list[dict]) -> dict:
        idx_arr = np.fromiter(
            (ex["item_token_idx"] for ex in batch), dtype=np.int64
        )
        ids_list = [token_cache.input_ids[i] for i in idx_arr]
        max_len = int(max((a.size for a in ids_list), default=1))
        max_len = max(1, max_len)
        bsz = len(ids_list)
        input_ids = np.full((bsz, max_len), pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, max_len), dtype=np.int64)
        for i, a in enumerate(ids_list):
            n = int(a.size)
            if n > 0:
                input_ids[i, :n] = a
                attention_mask[i, :n] = 1
        out: dict[str, torch.Tensor] = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "subject_idx": torch.tensor(
                [ex["subject_idx"] for ex in batch], dtype=torch.long
            ),
            "bc_idx": torch.tensor(
                [ex["bc_idx"] for ex in batch], dtype=torch.long
            ),
            "label": torch.tensor(
                [ex["label"] for ex in batch], dtype=torch.float32
            ),
        }
        if "subject_emb" in batch[0]:
            out["subject_emb"] = torch.from_numpy(
                np.stack([ex["subject_emb"] for ex in batch], axis=0)
            ).to(torch.float32)
        if "pool_feats" in batch[0]:
            out["pool_feats"] = torch.from_numpy(
                np.stack([ex["pool_feats"] for ex in batch], axis=0)
            ).to(torch.float32)
        if "cluster_id" in batch[0]:
            out["cluster_id"] = torch.tensor(
                [ex["cluster_id"] for ex in batch], dtype=torch.long
            )
        if "judge_feats" in batch[0]:
            out["judge_feats"] = torch.from_numpy(
                np.stack([ex["judge_feats"] for ex in batch], axis=0)
            ).to(torch.float32)
        if "nn_feats" in batch[0]:
            out["nn_feats"] = torch.from_numpy(
                np.stack([ex["nn_feats"] for ex in batch], axis=0)
            ).to(torch.float32)
        return out

    return _collate


def _fixed_random_subset_indices(n: int, m: int, *, seed: int) -> np.ndarray:
    """Return a deterministic random subset of row indices.

    The returned indices are sorted after sampling so DataLoader order is
    deterministic and stable across resumes, but the subset itself is a
    random sample of the full set rather than the first ``m`` rows. This
    is how the capped validation pass picks which rows it evaluates: if
    the val frame is sorted in any meaningful way (subject, length,
    benchmark), evaluating the first N rows would bias the cap.
    """
    n = int(n)
    m = int(min(max(0, m), n))
    if m <= 0 or m >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=m, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Length-bucketed BatchSampler. Shuffles bucket order but keeps in-batch
# token lengths close so padding waste is bounded.
# ---------------------------------------------------------------------------


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        *,
        token_lens: np.ndarray,
        batch_size: int,
        n_buckets: int = 32,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.token_lens = np.asarray(token_lens, dtype=np.int32)
        self.batch_size = max(1, int(batch_size))
        self.n_buckets = max(1, int(n_buckets))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        n = self.token_lens.shape[0]
        # Pre-sort indices once by token length for bucket assignment.
        self._sorted = np.argsort(self.token_lens, kind="stable")
        self._n = n
        # Precompute bucket id per (sorted) position.
        bucket_size = max(1, n // self.n_buckets)
        self._bucket_ids = np.minimum(
            (np.arange(n, dtype=np.int64) // bucket_size),
            self.n_buckets - 1,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        # Build per-bucket index lists in the original dataset index space.
        buckets: list[list[int]] = [[] for _ in range(self.n_buckets)]
        for sorted_pos, idx in enumerate(self._sorted):
            buckets[int(self._bucket_ids[sorted_pos])].append(int(idx))

        if self.shuffle:
            for b in buckets:
                rng.shuffle(b)

        # Form batches within each length bucket so in-batch padding remains
        # bounded. Then shuffle the completed batches globally so training
        # does not spend thousands of consecutive microbatches in one
        # token-length regime -- previously the loop concatenated entire
        # buckets back-to-back, which produced long runs of one length
        # regime and surfaced as degenerate per-batch loss spikes.
        batches: list[list[int]] = []
        for b in buckets:
            for start in range(0, len(b), self.batch_size):
                batch = b[start : start + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self._n / self.batch_size)


# ---------------------------------------------------------------------------
# Pooling helper shared with the embedder. Importing the private helpers
# keeps the LoRA path bit-identical to the frozen-encoder path.
# ---------------------------------------------------------------------------


def _pool(
    last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden[:, 0]
    if pooling == "last_token":
        return _last_token_pool(last_hidden, attention_mask)
    return _mean_pool(last_hidden, attention_mask)


# ---------------------------------------------------------------------------
# Encoder wrapping
# ---------------------------------------------------------------------------


def _force_disable_gradient_checkpointing(encoder, base_encoder=None) -> None:
    """Force-disable HF/PEFT gradient checkpointing on every nested module.

    Some HF/PEFT versions retain checkpointing state even after the
    user-level config says ``gradient_checkpointing=False`` -- notably
    after ``get_peft_model(...)`` wraps a base encoder that was loaded
    while a prior session left checkpointing enabled. The downstream
    symptom is "RuntimeError: Checkpointing requires checkpoint_fn..." or
    silent zero adapter gradients depending on the PEFT version.

    To make ``cfg.gradient_checkpointing = False`` actually mean what it
    says, we walk every submodule of the wrapper, the underlying base
    model, and the original ``base_encoder`` reference, and clear the
    flag everywhere. We also turn off ``use_cache`` on configs that have
    it (it interacts badly with checkpointing).
    """
    modules = [encoder]
    if base_encoder is not None:
        modules.append(base_encoder)

    try:
        if hasattr(encoder, "get_base_model"):
            modules.append(encoder.get_base_model())
    except Exception:
        pass

    for module in modules:
        if module is None:
            continue

        try:
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
        except Exception:
            LOG.warning("gradient_checkpointing_disable failed", exc_info=True)

        try:
            if hasattr(module, "disable_input_require_grads"):
                module.disable_input_require_grads()
        except Exception:
            pass

        try:
            if hasattr(module, "config") and hasattr(module.config, "use_cache"):
                module.config.use_cache = False
        except Exception:
            pass

        try:
            for sub in module.modules():
                if hasattr(sub, "gradient_checkpointing"):
                    try:
                        sub.gradient_checkpointing = False
                    except Exception:
                        pass
                if hasattr(sub, "config") and hasattr(sub.config, "use_cache"):
                    try:
                        sub.config.use_cache = False
                    except Exception:
                        pass
        except Exception:
            pass


def _wrap_encoder_with_lora(
    base_encoder: nn.Module,
    *,
    cfg: LoRATrainConfig,
) -> nn.Module:
    """Wrap a HF AutoModel with PEFT LoRA adapters on the attention proj.

    Returns the wrapped PeftModel. Adapters initialize to zero contribution
    (PEFT default) so step 0 is numerically identical to the frozen encoder.
    Gradient checkpointing + ``enable_input_require_grads`` is mandatory
    when combining grad ckpt with PEFT or the encoder receives no gradient.
    """
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LoRA training requires the `peft` package: pip install peft. "
            f"Import failed with {type(exc).__name__}: {exc}"
        ) from exc

    layer_kwargs: dict = {}
    if cfg.layers_to_transform is not None:
        layer_kwargs["layers_to_transform"] = list(cfg.layers_to_transform)

    lora_cfg = LoraConfig(
        r=int(cfg.r),
        lora_alpha=int(cfg.alpha),
        lora_dropout=float(cfg.dropout),
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="FEATURE_EXTRACTION",
        **layer_kwargs,
    )
    encoder = get_peft_model(base_encoder, lora_cfg)

    if not cfg.gradient_checkpointing:
        LOG.info("Force-disabling gradient checkpointing after PEFT wrapping")
        _force_disable_gradient_checkpointing(encoder, base_encoder=base_encoder)

    if cfg.gradient_checkpointing:
        # Order matters: the input-require-grads hook must be installed
        # BEFORE the first forward, and AFTER gradient checkpointing is
        # enabled on the underlying encoder. Both PEFT and HF gradient
        # checkpointing share this requirement; without it the encoder's
        # input embedding output doesn't have requires_grad=True, the
        # checkpointing wrapper bails out of its backward path, and the
        # adapter parameters receive no gradient.
        try:
            encoder.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"gradient_checkpointing_enable() failed ({type(exc).__name__}: "
                f"{exc}); proceeding without it (memory may not fit)."
            )
        try:
            encoder.enable_input_require_grads()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"enable_input_require_grads() failed ({type(exc).__name__}: "
                f"{exc}); LoRA gradients may be zero -- check the run."
            )
    try:
        encoder.print_trainable_parameters()
    except Exception:
        pass
    return encoder


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


_TRAIN_STATE_FILE = "training_state.json"
_HEAD_STATE_FILE = "head.pt"
_OPT_STATE_FILE = "optimizer.pt"
_ADAPTER_DIR = "adapter"


def _rng_state() -> dict:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu().numpy().tolist(),
    }
    if torch.cuda.is_available():
        try:
            out["cuda"] = [s.cpu().numpy().tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            out["cuda"] = None
    return out


def _restore_rng_state(state: dict) -> None:
    try:
        py = state.get("python")
        if py is not None:
            # JSON tuples come back as lists; tuple() the nested elements.
            py = tuple(
                tuple(x) if isinstance(x, list) else x for x in py
            )
            random.setstate(py)
    except Exception:
        LOG.warning("python RNG restore failed", exc_info=True)
    try:
        npy = state.get("numpy")
        if npy is not None:
            npy = tuple(npy)  # state is a 5-tuple
            # numpy's state has a bytes array nested deep -- if it round-tripped
            # through JSON it became a list of ints; rebuild as needed.
            np.random.set_state(_coerce_numpy_state(npy))
    except Exception:
        LOG.warning("numpy RNG restore failed", exc_info=True)
    try:
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(
                torch.tensor(torch_state, dtype=torch.uint8)
            )
    except Exception:
        LOG.warning("torch RNG restore failed", exc_info=True)
    try:
        cuda_state = state.get("cuda")
        if cuda_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.tensor(s, dtype=torch.uint8) for s in cuda_state]
            )
    except Exception:
        LOG.warning("cuda RNG restore failed", exc_info=True)


def _coerce_numpy_state(state: tuple) -> tuple:
    """numpy's get_state returns (str, np.ndarray uint32[624], int, int, float).
    After json round-tripping the array is a list of ints; reconstitute.
    """
    if not isinstance(state, tuple):
        return state
    out_list = list(state)
    if len(out_list) >= 2 and not isinstance(out_list[1], np.ndarray):
        try:
            out_list[1] = np.asarray(out_list[1], dtype=np.uint32)
        except Exception:
            pass
    return tuple(out_list)


def _atomic_save_dir(staging: Path, final: Path) -> None:
    """Promote ``staging`` -> ``final`` via os.replace (cross-FS fallback)."""
    if final.exists():
        shutil.rmtree(final, ignore_errors=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging, final)
    except OSError:
        shutil.move(str(staging), str(final))


@dataclass
class _CheckpointPaths:
    root: Path
    step: int

    @property
    def staging(self) -> Path:
        return self.root / f"step_{self.step}.tmp"

    @property
    def final(self) -> Path:
        return self.root / f"step_{self.step}"


def _save_checkpoint(
    *,
    root: Path,
    step: int,
    epoch: int,
    encoder,
    head_payload: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    train_state: dict,
    is_best: bool,
) -> Path:
    """Atomically write the per-step checkpoint to Drive (or any FS).

    Layout::

        {root}/step_{step}/
            adapter/                  # encoder.save_pretrained(...)
            head.pt                   # runtime-compatible head checkpoint:
                                      #   {model_state, model_cfg, train_cfg,
                                      #    indexer, model_name}
            optimizer.pt              # optimizer + scheduler state
            training_state.json       # global step, epoch, best val NLL, RNG, hashes

    The ``head.pt`` deliberately matches the schema produced by
    ``src.train._save_checkpoint`` so the existing exporter can ingest it
    directly via ``lora_head_checkpoint``. The companion ``training_state.json``
    carries the LoRA-specific bookkeeping the exporter does not need.
    """
    paths = _CheckpointPaths(root=Path(root), step=int(step))
    if paths.staging.exists():
        shutil.rmtree(paths.staging, ignore_errors=True)
    paths.staging.mkdir(parents=True, exist_ok=True)

    adapter_dir = paths.staging / _ADAPTER_DIR
    # PEFT's save_pretrained writes ~10-50MB of adapter weights.
    encoder.save_pretrained(str(adapter_dir), safe_serialization=True)

    torch.save(head_payload, paths.staging / _HEAD_STATE_FILE)
    torch.save(
        {
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
        },
        paths.staging / _OPT_STATE_FILE,
    )
    (paths.staging / _TRAIN_STATE_FILE).write_text(
        json.dumps(train_state, indent=2, default=_json_default),
        encoding="utf-8",
    )

    _atomic_save_dir(paths.staging, paths.final)

    if is_best:
        best_dir = paths.root / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir, ignore_errors=True)
        # Re-copy contents -- the original final dir must remain for the
        # "last_n" prune step below.
        shutil.copytree(paths.final, best_dir)
    return paths.final


def _prune_old_checkpoints(root: Path, *, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    steps: list[tuple[int, Path]] = []
    for p in root.glob("step_*"):
        if p.is_dir() and not p.name.endswith(".tmp"):
            try:
                steps.append((int(p.name.split("_", 1)[1]), p))
            except ValueError:
                continue
    steps.sort(key=lambda x: x[0])
    excess = max(0, len(steps) - keep_last_n)
    for _, p in steps[:excess]:
        shutil.rmtree(p, ignore_errors=True)


def _find_latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    best_step = -1
    best_path: Path | None = None
    for p in root.glob("step_*"):
        if not p.is_dir() or p.name.endswith(".tmp"):
            continue
        try:
            s = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        if (p / _TRAIN_STATE_FILE).exists() and s > best_step:
            best_step = s
            best_path = p
    return best_path


def _load_checkpoint(
    ckpt_dir: Path,
    *,
    map_location: str = "cpu",
) -> dict:
    state_path = ckpt_dir / _TRAIN_STATE_FILE
    head_path = ckpt_dir / _HEAD_STATE_FILE
    opt_path = ckpt_dir / _OPT_STATE_FILE
    if not (state_path.exists() and head_path.exists() and opt_path.exists()):
        raise FileNotFoundError(
            f"Incomplete checkpoint at {ckpt_dir}: state={state_path.exists()} "
            f"head={head_path.exists()} opt={opt_path.exists()}"
        )
    train_state = json.loads(state_path.read_text(encoding="utf-8"))
    # ``head.pt`` is the runtime-compatible checkpoint dict (model_state +
    # model_cfg + indexer + train_cfg). For resume we only need
    # ``model_state``; for export we use the whole dict directly.
    head_payload = torch.load(head_path, map_location=map_location)
    head_state = (
        head_payload["model_state"]
        if isinstance(head_payload, dict) and "model_state" in head_payload
        else head_payload
    )
    opt_pkg = torch.load(opt_path, map_location=map_location)
    return {
        "training_state": train_state,
        "head": head_state,
        "head_payload": head_payload,
        "optimizer": opt_pkg.get("optimizer"),
        "scheduler": opt_pkg.get("scheduler"),
        "adapter_dir": ckpt_dir / _ADAPTER_DIR,
    }


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, bytes):
        return list(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Scheduler (same shape as src.train)
# ---------------------------------------------------------------------------


def _make_scheduler(
    opt,
    total_steps: int,
    warmup: int,
    *,
    scheduler: str = "cosine",
    min_lr_ratio: float = 0.05,
):
    """Warmup + decay scheduler over the planned training budget.

    The learning rates in the optimizer are interpreted as PEAK learning
    rates. The scheduler then:
      1. linearly warms from 0 to peak LR over ``warmup`` steps;
      2. decays from peak LR to ``peak_lr * min_lr_ratio`` over the
         remaining ``total_steps - warmup`` steps.

    Using ``total_steps`` derived from the planned ``max_train_steps``
    instead of the full 150k-step epoch matters a lot for short LoRA
    runs: scheduling cosine decay over a full epoch turns a 1k-5k-step
    run into an effectively constant-LR / warmup-only schedule, which
    defeats the point of choosing cosine or linear.
    """
    total_steps = max(1, int(total_steps))
    warmup = max(0, int(warmup))
    if total_steps <= 1:
        warmup = 0
    else:
        warmup = min(warmup, total_steps - 1)

    min_lr_ratio = float(min(max(float(min_lr_ratio), 0.0), 1.0))
    sched = str(scheduler or "cosine").lower().strip()

    def lr_lambda(step: int) -> float:
        step = int(step)

        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))

        if sched in {"constant", "constant_with_warmup"}:
            return 1.0

        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = float(min(max(progress, 0.0), 1.0))

        if sched in {"linear", "linear_decay"}:
            decay = 1.0 - progress
        elif sched in {"cosine", "cosine_decay"}:
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(
                f"Unknown LoRA scheduler={scheduler!r}; "
                "use cosine, linear, or constant"
            )

        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics (avoid coupling to src.train's private helpers)
# ---------------------------------------------------------------------------


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


# ---------------------------------------------------------------------------
# Train step + eval pass
# ---------------------------------------------------------------------------


def _move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _encoder_embed(
    encoder, *, input_ids, attention_mask, pooling: str
) -> torch.Tensor:
    """Forward raw tokens through the (adapter-augmented) encoder + pool.

    Mirrors ``TransformerEmbedder._forward_batch`` -- but the surrounding
    autocast / inference_mode is set up by the caller because we need
    gradients here.
    """
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    return _pool(last_hidden, attention_mask, pooling)


def _forward_head(model, *, item_emb, batch: dict) -> torch.Tensor:
    s = batch["subject_idx"]
    bc = batch["bc_idx"]
    se = batch.get("subject_emb")
    if se is not None and se.shape[-1] == 0:
        se = None
    pf = batch.get("pool_feats")
    if pf is not None and pf.shape[-1] == 0:
        pf = None
    ci = batch.get("cluster_id")
    if ci is not None and ci.numel() == 0:
        ci = None
    jf = batch.get("judge_feats")
    if jf is not None and jf.shape[-1] == 0:
        jf = None
    nf = batch.get("nn_feats")
    if nf is not None and nf.shape[-1] == 0:
        nf = None
    return model(s, bc, item_emb, se, pf, ci, jf, nf)


@torch.no_grad()
def _evaluate_lora(
    *,
    encoder,
    head_model,
    val_loader,
    device: str,
    pooling: str,
    bf16: bool,
    desc: str = "lora-val",
    max_batches: int | None = None,
    log_every: int = 100,
) -> dict:
    """Run a single eval pass through the (adapter-augmented) encoder.

    Args:
        max_batches: optional cap on the number of validation batches to
            consume. ``None`` or ``<= 0`` evaluates every batch yielded by
            ``val_loader``. Used by the periodic in-loop eval so a single
            eval pass stays bounded even when the val split is large.
        log_every: emit a per-batch progress log every N batches. ``0``
            disables progress logging; the default is friendly to runs
            that hide tqdm output (Colab background tabs, CI).
    """
    encoder.eval()
    head_model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast("cpu", enabled=False)
    )

    pbar = tqdm(
        val_loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        total=len(val_loader),
    )
    eval_t0 = time.time()
    seen_rows = 0
    with autocast_ctx:
        for batch_i, batch in enumerate(pbar, start=1):
            if max_batches is not None and max_batches > 0 and batch_i > max_batches:
                LOG.info(
                    "%s stopping early after %d batches / %d rows because max_batches=%d",
                    desc,
                    batch_i - 1,
                    seen_rows,
                    max_batches,
                )
                break

            batch = _move_batch(batch, device)
            item_emb = _encoder_embed(
                encoder,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pooling=pooling,
            )
            # The head is fp32 (see _build_head_optimizer); cast back so
            # nn.Linear inputs match parameter dtype.
            item_emb_fp = item_emb.float()
            logits = _forward_head(head_model, item_emb=item_emb_fp, batch=batch)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            preds.append(p)
            targets.append(batch["label"].float().cpu().numpy())

            seen_rows += int(batch["label"].shape[0])
            if log_every and batch_i % int(log_every) == 0:
                elapsed = time.time() - eval_t0
                LOG.info(
                    "%s progress: batch %d/%d rows=%d elapsed=%.1f min rate=%.1f rows/s",
                    desc,
                    batch_i,
                    len(val_loader),
                    seen_rows,
                    elapsed / 60.0,
                    seen_rows / max(elapsed, 1e-6),
                )

    p = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    y = np.concatenate(targets) if targets else np.zeros(0, dtype=np.float32)
    return {
        "log_loss": _log_loss(y, p),
        "brier": _brier(y, p),
        "n": int(y.shape[0]),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@dataclass
class LoRARunResult:
    """What ``run`` returns to the notebook for the export cell."""

    enabled: bool
    completed: bool
    base_checkpoint: str
    n_train: int
    n_val: int
    global_step: int
    best_step: int
    best_val_log_loss: float
    best_val_brier: float
    base_val_log_loss: float
    base_val_brier: float
    init_check_passed: bool
    drive_checkpoint_dir: str
    best_checkpoint_dir: str
    latest_checkpoint_dir: str
    elapsed_seconds: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def run(
    *,
    cfg: LoRATrainConfig,
    encoder_cfg: EncoderConfig,
    embedder: TransformerEmbedder,
    token_cache: TokenizedItemCache,
    train_ds: LoRARowDataset,
    val_ds: LoRARowDataset,
    base_checkpoint_path: str | os.PathLike,
    indexer: Indexer,
    device: str | None = None,
    progress_file: str | os.PathLike | None = None,
) -> LoRARunResult:
    """Run the overnight-safe LoRA fine-tuning loop.

    Inputs:
      ``cfg`` -- ``LoRATrainConfig.from_dict(CFG["lora"])``.
      ``encoder_cfg`` -- the same ``EncoderConfig`` used to build the
        frozen-cache embedder; we reuse its ``pooling`` / ``bf16`` /
        ``model_id``. The encoder weights are loaded fresh here so the
        LoRA wrapping doesn't perturb the embedder used elsewhere.
      ``embedder`` -- the frozen-cache embedder. We only borrow its
        tokenizer (already loaded) and its dim for sanity asserts.
      ``token_cache`` -- pre-tokenized item cache (cell 8e). Mandatory.
      ``train_ds`` / ``val_ds`` -- :class:`LoRARowDataset` instances with
        ``item_token_idx`` already mapped to ``token_cache`` row indices.
      ``base_checkpoint_path`` -- the head-only checkpoint we initialize
        from. Loaded for head + IRT + subject embed + feature consumers.
      ``indexer`` -- subject / bc indexer (matches the base checkpoint).

    Returns a :class:`LoRARunResult` with paths and metrics for the
    export cell.
    """
    if not cfg.enabled:
        return LoRARunResult(
            enabled=False,
            completed=False,
            base_checkpoint=str(base_checkpoint_path or ""),
            n_train=len(train_ds),
            n_val=len(val_ds),
            global_step=0,
            best_step=0,
            best_val_log_loss=float("nan"),
            best_val_brier=float("nan"),
            base_val_log_loss=float("nan"),
            base_val_brier=float("nan"),
            init_check_passed=False,
            drive_checkpoint_dir="",
            best_checkpoint_dir="",
            latest_checkpoint_dir="",
            elapsed_seconds=0.0,
            reason="lora.enabled = false",
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        warnings.warn(
            "lora_train.run: running on CPU is supported only for smoke "
            "tests; a full LoRA pass requires a 40GB+ GPU."
        )

    # ---- Resolve runtime knobs --------------------------------------------
    pooling = str(encoder_cfg.pooling or "last_token")
    bf16 = bool(cfg.bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported())
    drive_root = Path(cfg.drive_checkpoint_dir) if cfg.drive_checkpoint_dir else None
    if drive_root is not None:
        drive_root.mkdir(parents=True, exist_ok=True)

    # ---- Load base checkpoint (head only) ---------------------------------
    base_path = Path(base_checkpoint_path)
    LOG.info("Loading base head-only checkpoint: %s", base_path)
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model_cfg_dict = dict(base_ckpt.get("model_cfg") or {})
    base_model_name = base_ckpt.get("model_name") or base_ckpt.get("model_cfg", {}).get(
        "model_name"
    )
    if base_model_name is None:
        # The training-time checkpoint dict doesn't include model_name (it
        # lives in the metadata json); the LoRA cell passes the name in via
        # the higher-level wrapper, but as a fallback we accept it on the
        # training_state we emit.
        base_model_name = base_ckpt.get("model_name", "")
    model_cfg = ModelConfig(**base_model_cfg_dict)
    model_cfg_dict = asdict(model_cfg)

    head_model = build_model(
        base_model_name or "hierarchical_mirt", model_cfg
    )
    missing, unexpected = head_model.load_state_dict(
        base_ckpt["model_state"], strict=False
    )
    if missing or unexpected:
        LOG.warning(
            "Base checkpoint load: missing=%d unexpected=%d (acceptable for "
            "feature-toggle drift, abort if shape-related)",
            len(missing),
            len(unexpected),
        )
    head_model = head_model.to(device).float()  # head stays fp32

    # ---- Build a fresh base encoder, then wrap with LoRA ------------------
    LOG.info("Loading fresh base encoder %s for LoRA fine-tuning", encoder_cfg.model_id)
    from transformers import AutoModel  # local: keep module import cheap

    enc_dtype = torch.bfloat16 if bf16 else torch.float32
    base_encoder_kwargs: dict[str, Any] = {
        "torch_dtype": enc_dtype,
        "trust_remote_code": bool(encoder_cfg.trust_remote_code),
    }
    if encoder_cfg.use_flash_attention and device.startswith("cuda"):
        base_encoder_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        base_encoder = AutoModel.from_pretrained(
            encoder_cfg.model_id, **base_encoder_kwargs
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        LOG.warning(
            "FA2 unavailable for LoRA base encoder (%s); falling back to SDPA",
            type(exc).__name__,
        )
        base_encoder_kwargs["attn_implementation"] = "sdpa"
        base_encoder = AutoModel.from_pretrained(
            encoder_cfg.model_id, **base_encoder_kwargs
        )
    encoder = _wrap_encoder_with_lora(base_encoder, cfg=cfg).to(device)

    if not cfg.gradient_checkpointing:
        LOG.info("Force-disabling gradient checkpointing after encoder.to(device)")
        _force_disable_gradient_checkpointing(encoder, base_encoder=base_encoder)

    # ---- Tokenizer for pad token ------------------------------------------
    embedder._load()  # noqa: SLF001
    tokenizer = embedder._tok  # noqa: SLF001
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )
    collate = _make_collator(token_cache=token_cache, pad_token_id=int(pad_id))

    # ---- Build dataloaders ------------------------------------------------
    train_sampler = LengthBucketBatchSampler(
        token_lens=np.asarray(
            [token_cache.token_lens[int(i)] for i in train_ds.item_token_idx],
            dtype=np.int32,
        ),
        batch_size=cfg.batch_size_items,
        shuffle=True,
        seed=0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    # Validation loader. If ``val_eval_max_batches > 0``, evaluate a fixed
    # deterministic random subset instead of the first N validation rows
    # -- ordering matters because the val frame is sorted by subject /
    # benchmark and the first-N slice was systematically biased.
    val_eval_ds: torch.utils.data.Dataset = val_ds
    requested_val_rows = (
        int(cfg.val_eval_max_batches) * int(cfg.val_batch_size_items)
        if int(getattr(cfg, "val_eval_max_batches", 0)) > 0
        else 0
    )
    if requested_val_rows > 0 and requested_val_rows < len(val_ds):
        val_eval_seed = int(getattr(cfg, "val_eval_seed", 12345))
        val_subset_idx = _fixed_random_subset_indices(
            len(val_ds),
            requested_val_rows,
            seed=val_eval_seed,
        )
        val_eval_ds = torch.utils.data.Subset(val_ds, val_subset_idx.tolist())
        LOG.info(
            "Using fixed random validation subset: rows=%d/%d seed=%d "
            "batch_size=%d batches=%d",
            len(val_eval_ds),
            len(val_ds),
            val_eval_seed,
            cfg.val_batch_size_items,
            math.ceil(len(val_eval_ds) / max(1, cfg.val_batch_size_items)),
        )
    else:
        LOG.info(
            "Using full validation set: rows=%d batch_size=%d batches=%d",
            len(val_ds),
            cfg.val_batch_size_items,
            math.ceil(len(val_ds) / max(1, cfg.val_batch_size_items)),
        )

    # Length-bucketed batch sampler with shuffle=True so validation batches
    # are not consumed in subject-sorted order. The aggregate val log-loss
    # is invariant to batch order, but tqdm/progress logs and any future
    # streaming consumer benefit from the shuffled order, and (more
    # importantly) it parallels the training-side shuffle so debugging
    # batch-level behavior reflects the same regime.
    val_token_lens_for_sampler = (
        np.asarray(
            [
                token_cache.token_lens[int(val_ds.item_token_idx[i])]
                for i in val_eval_ds.indices
            ],
            dtype=np.int32,
        )
        if isinstance(val_eval_ds, torch.utils.data.Subset)
        else np.asarray(
            [token_cache.token_lens[int(i)] for i in val_ds.item_token_idx],
            dtype=np.int32,
        )
    )
    val_sampler = LengthBucketBatchSampler(
        token_lens=val_token_lens_for_sampler,
        batch_size=cfg.val_batch_size_items,
        shuffle=True,
        seed=int(getattr(cfg, "val_eval_seed", 12345)),
    )
    val_loader = torch.utils.data.DataLoader(
        val_eval_ds,
        batch_sampler=val_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    # ---- Optimizer (two parameter groups) ---------------------------------
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in head_model.parameters() if p.requires_grad]
    n_enc = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in head_params)
    LOG.info(
        "Optimizer param groups: encoder=%d (%.3f%%) head=%d",
        n_enc,
        100.0 * n_enc / max(1, n_enc + sum(p.numel() for p in encoder.parameters())),
        n_head,
    )

    opt_kwargs: dict = {}
    if cfg.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb  # type: ignore

            opt_cls = bnb.optim.AdamW8bit
        except Exception:
            LOG.warning("bitsandbytes unavailable; falling back to torch.optim.AdamW")
            opt_cls = torch.optim.AdamW
    else:
        opt_cls = torch.optim.AdamW

    optimizer = opt_cls(
        [
            {
                "params": encoder_params,
                "lr": float(cfg.encoder_lr),
                "weight_decay": 0.0,
            },
            {
                "params": head_params,
                "lr": float(cfg.head_lr),
                "weight_decay": float(cfg.weight_decay_head),
            },
        ],
        **opt_kwargs,
    )

    total_steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    epoch_total_steps = max(1, int(total_steps_per_epoch * cfg.epochs))

    # If ``max_train_steps`` is set, schedule over the intended short-run
    # budget, not over the full epoch. This is critical for cosine/linear
    # decays to actually fire during a 1k-10k LoRA pass.
    total_steps = (
        int(cfg.max_train_steps)
        if int(getattr(cfg, "max_train_steps", 0)) > 0
        else epoch_total_steps
    )
    total_steps = max(1, int(total_steps))

    if float(getattr(cfg, "warmup_ratio", 0.0)) > 0:
        warmup_steps = int(round(total_steps * float(cfg.warmup_ratio)))
    else:
        warmup_steps = int(cfg.warmup_steps)

    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))

    LOG.info(
        "LoRA LR schedule: scheduler=%s planned_steps=%d epoch_total_steps=%d "
        "warmup_steps=%d warmup_ratio=%.4f min_lr_ratio=%.4f "
        "peak_encoder_lr=%.3g peak_head_lr=%.3g",
        getattr(cfg, "scheduler", "cosine"),
        total_steps,
        epoch_total_steps,
        warmup_steps,
        float(getattr(cfg, "warmup_ratio", 0.0)),
        float(getattr(cfg, "min_lr_ratio", 0.05)),
        float(cfg.encoder_lr),
        float(cfg.head_lr),
    )

    scheduler = _make_scheduler(
        optimizer,
        total_steps,
        warmup_steps,
        scheduler=getattr(cfg, "scheduler", "cosine"),
        min_lr_ratio=float(getattr(cfg, "min_lr_ratio", 0.05)),
    )

    # ---- Resume? -----------------------------------------------------------
    resumed = False
    global_step = 0
    epoch_start = 0
    best_val_ll = float("inf")
    best_val_brier = float("nan")
    best_step = 0
    elapsed_prior = 0.0
    if cfg.resume and drive_root is not None:
        latest = _find_latest_checkpoint(drive_root)
        if latest is not None:
            LOG.info("Resuming from %s", latest)
            payload = _load_checkpoint(latest)
            ts = payload["training_state"]
            global_step = int(ts.get("global_step", 0))
            epoch_start = int(ts.get("epoch", 0))
            best_val_ll = float(ts.get("best_val_log_loss", best_val_ll))
            best_val_brier = float(ts.get("best_val_brier", best_val_brier))
            best_step = int(ts.get("best_step", 0))
            elapsed_prior = float(ts.get("elapsed_seconds", 0.0))

            # Reload adapter weights via PEFT into the existing module.
            try:
                from peft import PeftModel  # type: ignore

                adapter_dir = payload["adapter_dir"]
                if adapter_dir.exists():
                    # PEFT's load_adapter mutates the existing PeftModel in-place.
                    encoder.load_adapter(
                        str(adapter_dir), adapter_name="default", is_trainable=True
                    )
                    LOG.info("Reloaded adapter weights from %s", adapter_dir)
                    if not cfg.gradient_checkpointing:
                        LOG.info(
                            "Force-disabling gradient checkpointing after adapter reload"
                        )
                        _force_disable_gradient_checkpointing(encoder)
                else:
                    LOG.warning("Adapter dir missing at %s; using fresh PEFT init", adapter_dir)
                _ = PeftModel  # silence unused-import lint
            except Exception:
                LOG.exception("Failed to reload LoRA adapter; aborting resume")
                raise

            # Reload head/optimizer/scheduler/RNG.
            head_model.load_state_dict(payload["head"], strict=False)
            head_model = head_model.to(device)
            if payload["optimizer"] is not None:
                try:
                    optimizer.load_state_dict(payload["optimizer"])
                except Exception:
                    LOG.exception("Optimizer state restore failed; continuing with fresh optimizer")
            if payload["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(payload["scheduler"])
                except Exception:
                    LOG.exception("Scheduler state restore failed")
            rng = ts.get("rng_state")
            if rng:
                _restore_rng_state(rng)
            resumed = True

    # ---- Step-0 init sanity check ----------------------------------------
    # The init eval defaults to ``cfg.val_eval_max_batches`` so an init
    # check on a 1M-row val split does not stall a 15-min LoRA experiment
    # for 30 minutes. ``LORA_INIT_EVAL_BATCHES`` and ``LORA_EVAL_LOG_EVERY``
    # env vars override the per-cfg defaults without re-editing the config.
    init_eval_batches = int(
        os.environ.get(
            "LORA_INIT_EVAL_BATCHES",
            str(int(getattr(cfg, "val_eval_max_batches", 0) or 0)),
        )
    )
    eval_log_every = int(os.environ.get("LORA_EVAL_LOG_EVERY", "25"))
    LOG.info(
        "Computing base + step-0 val NLL for init sanity check on first %d val batches "
        "(set LORA_INIT_EVAL_BATCHES=0 for full init eval).",
        init_eval_batches,
    )
    base_val = _evaluate_lora(
        encoder=encoder,
        head_model=head_model,
        val_loader=val_loader,
        device=device,
        pooling=pooling,
        bf16=bf16,
        desc="lora-val (init)",
        max_batches=init_eval_batches if init_eval_batches > 0 else None,
        log_every=eval_log_every,
    )
    init_check_passed = True
    if not resumed:
        # If we did not resume, the adapter is fresh (zero contribution).
        # Sanity-check that against the base checkpoint's val NLL.
        base_ckpt_val_ll = float(
            (base_ckpt.get("result") or {}).get("best_val_log_loss", float("nan"))
            if isinstance(base_ckpt.get("result"), dict)
            else float("nan")
        )
        if not math.isfinite(base_ckpt_val_ll):
            # Older checkpoints did not embed the result block; we still
            # compare against the *current* val pass through the frozen
            # encoder embedding, which the wrapped encoder reproduces.
            base_ckpt_val_ll = base_val["log_loss"]
        diff = abs(base_val["log_loss"] - base_ckpt_val_ll)
        LOG.info(
            "Init check: step-0 val_ll=%.5f  base_ckpt_val_ll=%.5f  diff=%.5f  tol=%.5f",
            base_val["log_loss"],
            base_ckpt_val_ll,
            diff,
            cfg.init_check_tol,
        )
        if diff > cfg.init_check_tol:
            init_check_passed = False
            warnings.warn(
                f"LoRA init check FAILED: step-0 val NLL ({base_val['log_loss']:.5f}) "
                f"deviates from base checkpoint val NLL ({base_ckpt_val_ll:.5f}) by "
                f"{diff:.5f} > tol {cfg.init_check_tol:.5f}. Adapters should "
                "initialize to zero contribution; check head/encoder wiring."
            )
        if best_val_ll == float("inf"):
            best_val_ll = base_val["log_loss"]
            best_val_brier = base_val["brier"]
            best_step = global_step

    LOG.info(
        "LoRA setup: n_train=%d n_val=%d batches_per_epoch=%d grad_accum=%d "
        "effective_item_batch=%d steps_per_epoch=%d total_steps=%d bf16=%s",
        len(train_ds),
        len(val_ds),
        len(train_loader),
        cfg.grad_accum_steps,
        cfg.batch_size_items * cfg.grad_accum_steps,
        total_steps_per_epoch,
        total_steps,
        bf16,
    )

    # ---- Training loop ----------------------------------------------------
    loss_fn = nn.BCEWithLogitsLoss()
    autocast_ctx_factory = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16))
        if device.startswith("cuda")
        else (lambda: torch.amp.autocast("cpu", enabled=False))
    )

    start = time.time()
    deadline = (
        start + 60.0 * cfg.max_runtime_minutes
        if cfg.max_runtime_minutes and cfg.max_runtime_minutes > 0
        else None
    )

    # The runtime exporter expects ``head.pt`` to be a torch.save dict with
    # the same schema as the head-only trainer's checkpoint: model_state,
    # model_cfg, train_cfg, indexer, plus (optionally) model_name. We
    # surface that schema here so a LoRA checkpoint can be exported via
    # ``export_run(..., lora_head_checkpoint=step_N/head.pt)`` without any
    # post-processing.
    lora_pseudo_train_cfg = {
        "learning_rate": float(cfg.head_lr),
        "weight_decay": float(cfg.weight_decay_head),
        "batch_size": int(cfg.batch_size_items * cfg.grad_accum_steps),
        "epochs": int(cfg.epochs),
        "scheduler": str(getattr(cfg, "scheduler", "cosine")),
        "grad_clip": 1.0,
        "bf16": bool(bf16),
        "_provenance": "lora_train.run",
    }

    def _save_current(*, step: int, epoch: int, reason: str, is_best: bool) -> Path | None:
        if drive_root is None:
            return None
        head_payload = {
            "model_state": head_model.state_dict(),
            "model_cfg": dict(model_cfg_dict),
            "train_cfg": dict(lora_pseudo_train_cfg),
            "indexer": indexer.to_dict(),
            "model_name": str(base_model_name or ""),
            "result": {
                "best_val_log_loss": float(best_val_ll),
                "best_val_brier": float(best_val_brier),
                "best_step": int(best_step),
                "base_val_log_loss": float(base_val["log_loss"]),
                "base_val_brier": float(base_val["brier"]),
            },
        }
        opt_state = optimizer.state_dict()
        sched_state = scheduler.state_dict()
        train_state = {
            "global_step": int(step),
            "epoch": int(epoch),
            "best_val_log_loss": float(best_val_ll),
            "best_val_brier": float(best_val_brier),
            "best_step": int(best_step),
            "base_val_log_loss": float(base_val["log_loss"]),
            "base_val_brier": float(base_val["brier"]),
            "init_check_passed": bool(init_check_passed),
            "elapsed_seconds": float(elapsed_prior + (time.time() - start)),
            "reason": str(reason),
            "encoder_model_id": str(encoder_cfg.model_id),
            "lora_cfg": cfg.to_dict(),
            "model_cfg": dict(model_cfg_dict),
            "model_name": str(base_model_name or ""),
            "rng_state": _rng_state(),
            "base_checkpoint": str(base_path),
        }
        path = _save_checkpoint(
            root=drive_root,
            step=step,
            epoch=epoch,
            encoder=encoder,
            head_payload=head_payload,
            optimizer_state=opt_state,
            scheduler_state=sched_state,
            train_state=train_state,
            is_best=is_best,
        )
        _prune_old_checkpoints(drive_root, keep_last_n=cfg.keep_last_n_checkpoints)
        return path

    if not resumed:
        # Drop a step-0 checkpoint so the resume path is exercised even if
        # the runtime dies before the first checkpoint_every_steps boundary.
        try:
            _save_current(step=0, epoch=0, reason="init", is_best=False)
        except Exception:
            LOG.exception("init checkpoint failed; continuing")

    encoder.train()
    head_model.train()
    last_loss_str = "--"
    # ``epoch`` is referenced from the except handlers below if we never
    # entered the for-loop body (e.g. resume with epoch_start == cfg.epochs).
    epoch = epoch_start

    try:
        for epoch in range(epoch_start, cfg.epochs):
            train_sampler.set_epoch(epoch)
            pbar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"LoRA epoch {epoch + 1}/{cfg.epochs}",
                dynamic_ncols=True,
                leave=False,
            )
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            t_step = time.time()
            for batch_idx, batch in enumerate(pbar, start=1):
                batch = _move_batch(batch, device)
                try:
                    with autocast_ctx_factory():
                        item_emb = _encoder_embed(
                            encoder,
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pooling=pooling,
                        )
                        item_emb_fp = item_emb.float()
                        logits = _forward_head(
                            head_model, item_emb=item_emb_fp, batch=batch
                        )
                        loss = loss_fn(logits, batch["label"])
                        if model_has_irt_heads(head_model):
                            beta_i, alpha_i = head_model.irt_heads(item_emb_fp)
                            loss = loss + irt_regularization(
                                beta_i,
                                alpha_i,
                                lambda_beta=float(model_cfg.irt_lambda_beta),
                                lambda_alpha=float(model_cfg.irt_lambda_alpha),
                            )
                    (loss / max(1, cfg.grad_accum_steps)).backward()
                except torch.cuda.OutOfMemoryError:
                    if not cfg.oom_fallback:
                        raise
                    LOG.warning(
                        "LoRA OOM at batch_idx=%d bs=%d max_len=%d; halving "
                        "batch_size_items, doubling grad_accum_steps once",
                        batch_idx,
                        cfg.batch_size_items,
                        cfg.max_length,
                    )
                    torch.cuda.empty_cache()
                    cfg.batch_size_items = max(1, cfg.batch_size_items // 2)
                    cfg.grad_accum_steps = cfg.grad_accum_steps * 2
                    cfg.oom_fallback = False
                    raise RuntimeError(
                        "OOM during LoRA forward; halved batch_size_items to "
                        f"{cfg.batch_size_items}, doubled grad_accum_steps to "
                        f"{cfg.grad_accum_steps}. Re-run the cell to retry."
                    )

                accum_count += 1
                if accum_count >= cfg.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(head_model.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_count = 0

                    step_elapsed = time.time() - t_step
                    t_step = time.time()
                    steps_per_s = 1.0 / max(step_elapsed, 1e-6)
                    last_loss_str = f"{loss.item():.4f}"

                    # ---- checkpoint --------------------------------------
                    if (
                        cfg.checkpoint_every_steps > 0
                        and global_step % cfg.checkpoint_every_steps == 0
                        and drive_root is not None
                    ):
                        try:
                            _save_current(
                                step=global_step,
                                epoch=epoch,
                                reason="periodic",
                                is_best=False,
                            )
                        except Exception:
                            LOG.exception(
                                "periodic checkpoint failed at step %d", global_step
                            )

                    # ---- eval --------------------------------------------
                    if (
                        cfg.eval_every_steps > 0
                        and global_step % cfg.eval_every_steps == 0
                    ):
                        metrics = _evaluate_lora(
                            encoder=encoder,
                            head_model=head_model,
                            val_loader=val_loader,
                            device=device,
                            pooling=pooling,
                            bf16=bf16,
                            desc=f"lora-val step={global_step}",
                            max_batches=(
                                int(cfg.val_eval_max_batches)
                                if int(getattr(cfg, "val_eval_max_batches", 0)) > 0
                                else None
                            ),
                            log_every=int(
                                os.environ.get("LORA_EVAL_LOG_EVERY", "100")
                            ),
                        )
                        encoder.train()
                        head_model.train()
                        improved = metrics["log_loss"] < best_val_ll - 1e-6
                        if improved:
                            best_val_ll = float(metrics["log_loss"])
                            best_val_brier = float(metrics["brier"])
                            best_step = int(global_step)
                        LOG.info(
                            "step=%d val_ll=%.5f val_brier=%.5f best=%.5f@step%d%s",
                            global_step,
                            metrics["log_loss"],
                            metrics["brier"],
                            best_val_ll,
                            best_step,
                            " (improved)" if improved else "",
                        )
                        try:
                            _save_current(
                                step=global_step,
                                epoch=epoch,
                                reason="eval",
                                is_best=improved,
                            )
                        except Exception:
                            LOG.exception(
                                "eval-time checkpoint failed at step %d", global_step
                            )

                    # ---- deadline ----------------------------------------
                    if deadline is not None and time.time() > deadline:
                        LOG.warning(
                            "max_runtime_minutes (%d) hit at step %d; "
                            "checkpointing and exiting cleanly",
                            cfg.max_runtime_minutes,
                            global_step,
                        )
                        _save_current(
                            step=global_step,
                            epoch=epoch,
                            reason="max_runtime_hit",
                            is_best=False,
                        )
                        return _build_result(
                            cfg=cfg,
                            base_checkpoint=str(base_path),
                            n_train=len(train_ds),
                            n_val=len(val_ds),
                            global_step=global_step,
                            best_step=best_step,
                            best_val_ll=best_val_ll,
                            best_val_brier=best_val_brier,
                            base_val=base_val,
                            init_check_passed=init_check_passed,
                            drive_root=drive_root,
                            elapsed=time.time() - start + elapsed_prior,
                            reason="hit max_runtime; re-run to continue",
                            completed=False,
                        )

                    pbar.set_postfix(
                        {
                            "loss": last_loss_str,
                            "best_ll": f"{best_val_ll:.5f}",
                            "step": global_step,
                            "lr_enc": f"{optimizer.param_groups[0]['lr']:.1e}",
                            "lr_head": f"{optimizer.param_groups[1]['lr']:.1e}",
                            "steps/s": f"{steps_per_s:.2f}",
                            "gpu_mem": _gpu_mem_string(),
                        }
                    )

                    # ---- clean stop at max_train_steps ----------------
                    # This avoids relying on manual interrupt or
                    # ``max_runtime_minutes`` to stop a short, scheduled
                    # LoRA run -- when ``max_train_steps`` is set the
                    # schedule has already decayed all the way down to
                    # ``min_lr_ratio * peak_lr`` so continuing past it
                    # only burns compute.
                    if (
                        int(getattr(cfg, "max_train_steps", 0)) > 0
                        and global_step >= int(cfg.max_train_steps)
                    ):
                        LOG.info(
                            "max_train_steps=%d hit at global_step=%d; "
                            "checkpointing and exiting",
                            int(cfg.max_train_steps),
                            global_step,
                        )

                        if not (
                            cfg.eval_every_steps > 0
                            and global_step % cfg.eval_every_steps == 0
                        ):
                            metrics = _evaluate_lora(
                                encoder=encoder,
                                head_model=head_model,
                                val_loader=val_loader,
                                device=device,
                                pooling=pooling,
                                bf16=bf16,
                                desc=f"lora-val step={global_step} final-budget",
                                max_batches=(
                                    int(cfg.val_eval_max_batches)
                                    if int(getattr(cfg, "val_eval_max_batches", 0)) > 0
                                    else None
                                ),
                                log_every=int(
                                    os.environ.get("LORA_EVAL_LOG_EVERY", "100")
                                ),
                            )
                            encoder.train()
                            head_model.train()
                            improved = metrics["log_loss"] < best_val_ll - 1e-6
                            if improved:
                                best_val_ll = float(metrics["log_loss"])
                                best_val_brier = float(metrics["brier"])
                                best_step = int(global_step)
                            LOG.info(
                                "step=%d val_ll=%.5f val_brier=%.5f best=%.5f@step%d%s",
                                global_step,
                                metrics["log_loss"],
                                metrics["brier"],
                                best_val_ll,
                                best_step,
                                " (improved)" if improved else "",
                            )

                        _save_current(
                            step=global_step,
                            epoch=epoch,
                            reason="max_train_steps",
                            is_best=False,
                        )
                        return _build_result(
                            cfg=cfg,
                            base_checkpoint=str(base_path),
                            n_train=len(train_ds),
                            n_val=len(val_ds),
                            global_step=global_step,
                            best_step=best_step,
                            best_val_ll=best_val_ll,
                            best_val_brier=best_val_brier,
                            base_val=base_val,
                            init_check_passed=init_check_passed,
                            drive_root=drive_root,
                            elapsed=time.time() - start + elapsed_prior,
                            reason=f"hit max_train_steps={int(cfg.max_train_steps)}",
                            completed=True,
                        )

    except KeyboardInterrupt:
        LOG.warning("LoRA training interrupted by KeyboardInterrupt")
        try:
            _save_current(
                step=global_step,
                epoch=epoch,
                reason="keyboard_interrupt",
                is_best=False,
            )
        except Exception:
            LOG.exception("emergency-checkpoint failed")
        raise
    except Exception:
        LOG.exception("LoRA training raised; saving emergency checkpoint")
        try:
            _save_current(
                step=global_step,
                epoch=epoch,
                reason="exception",
                is_best=False,
            )
        except Exception:
            LOG.exception("emergency-checkpoint failed")
        raise

    # ---- Final eval + checkpoint -----------------------------------------
    metrics = _evaluate_lora(
        encoder=encoder,
        head_model=head_model,
        val_loader=val_loader,
        device=device,
        pooling=pooling,
        bf16=bf16,
        desc="lora-val final",
        max_batches=(
            int(cfg.val_eval_max_batches)
            if int(getattr(cfg, "val_eval_max_batches", 0)) > 0
            else None
        ),
        log_every=int(os.environ.get("LORA_EVAL_LOG_EVERY", "100")),
    )
    improved = metrics["log_loss"] < best_val_ll - 1e-6
    if improved:
        best_val_ll = float(metrics["log_loss"])
        best_val_brier = float(metrics["brier"])
        best_step = int(global_step)
    try:
        _save_current(
            step=global_step,
            epoch=cfg.epochs,
            reason="final",
            is_best=improved,
        )
    except Exception:
        LOG.exception("final checkpoint failed")

    return _build_result(
        cfg=cfg,
        base_checkpoint=str(base_path),
        n_train=len(train_ds),
        n_val=len(val_ds),
        global_step=global_step,
        best_step=best_step,
        best_val_ll=best_val_ll,
        best_val_brier=best_val_brier,
        base_val=base_val,
        init_check_passed=init_check_passed,
        drive_root=drive_root,
        elapsed=time.time() - start + elapsed_prior,
        reason="completed",
        completed=True,
    )


def _build_result(
    *,
    cfg: LoRATrainConfig,
    base_checkpoint: str,
    n_train: int,
    n_val: int,
    global_step: int,
    best_step: int,
    best_val_ll: float,
    best_val_brier: float,
    base_val: dict,
    init_check_passed: bool,
    drive_root: Path | None,
    elapsed: float,
    reason: str,
    completed: bool,
) -> LoRARunResult:
    best_dir = ""
    latest_dir = ""
    if drive_root is not None:
        bd = drive_root / "best"
        if bd.exists():
            best_dir = str(bd)
        latest = _find_latest_checkpoint(drive_root)
        if latest is not None:
            latest_dir = str(latest)
    return LoRARunResult(
        enabled=True,
        completed=completed,
        base_checkpoint=str(base_checkpoint),
        n_train=int(n_train),
        n_val=int(n_val),
        global_step=int(global_step),
        best_step=int(best_step),
        best_val_log_loss=float(best_val_ll),
        best_val_brier=float(best_val_brier),
        base_val_log_loss=float(base_val.get("log_loss", float("nan"))),
        base_val_brier=float(base_val.get("brier", float("nan"))),
        init_check_passed=bool(init_check_passed),
        drive_checkpoint_dir=str(drive_root) if drive_root else "",
        best_checkpoint_dir=best_dir,
        latest_checkpoint_dir=latest_dir,
        elapsed_seconds=float(elapsed),
        reason=str(reason),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_mem_string() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    return f"{a:.1f}/{r:.1f}GB"


__all__ = [
    "LoRARowDataset",
    "LoRARunResult",
    "LoRATrainConfig",
    "LengthBucketBatchSampler",
    "run",
]
