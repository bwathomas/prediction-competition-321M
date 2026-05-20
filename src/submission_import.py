"""Import a previously-built submission.zip back into the working tree.

After a Colab disconnect, the local ``artifacts/`` directory is wiped but
a previously-exported submission ZIP -- which already contains the
trained checkpoint, the runtime metadata, the pool-feature z-score stats,
the cluster centroids, the quantized training-item cache, and (when LoRA
was applied) the LoRA adapter -- is the cheapest way to skip training
and go straight to either:

1. The smoke test / re-upload to Codabench
   (just unpack the ZIP into ``submission/`` and re-run cell 20).

2. A new LoRA fine-tuning run seeded from the previously-trained head.
   (Use :func:`materialize_as_run` to write the imported checkpoint into
   ``artifacts/checkpoints/{run_id}.pt`` + sibling JSON so it shows up in
   the rest of the notebook's machinery exactly like a fresh training run.)

This module is intentionally small and dependency-free beyond the
standard library + ``torch`` (needed to peek at the checkpoint's
``model_cfg`` / ``train_cfg`` / ``indexer`` blocks).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("submission_import")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImportedSubmission:
    """Paths + metadata extracted from a previously-built submission.

    All paths point inside ``out_dir``. The dataclass intentionally
    mirrors what cell 19's exporter would have produced, so the
    downstream cells (smoke test, calibrator load, etc.) can consume an
    imported submission without distinguishing it from a fresh export.
    """

    src: Path                                  # original .zip or dir
    out_dir: Path                              # unpacked submission/ root
    checkpoint_path: Path                      # artifacts/checkpoint.pt
    runtime_meta_path: Path                    # artifacts/runtime_meta.json
    runtime_meta: dict = field(default_factory=dict)

    # Convenience pulled from runtime_meta + the checkpoint:
    run_id: str = ""
    model_name: str = ""
    encoder_model_id: str = ""
    model_cfg: dict = field(default_factory=dict)
    train_cfg: dict = field(default_factory=dict)
    best_val_log_loss: float = float("nan")
    best_val_brier: float = float("nan")
    epoch_best: int = -1

    # Optional sibling artifacts (None if not shipped):
    pool_stats_path: Path | None = None
    cluster_centroids_path: Path | None = None
    training_cache_dir: Path | None = None
    lora_adapter_dir: Path | None = None
    lora_mode: str = "none"

    def has_lora(self) -> bool:
        return self.lora_adapter_dir is not None and self.lora_adapter_dir.exists()

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in (
            "src",
            "out_dir",
            "checkpoint_path",
            "runtime_meta_path",
            "pool_stats_path",
            "cluster_centroids_path",
            "training_cache_dir",
            "lora_adapter_dir",
        ):
            v = d.get(k)
            d[k] = str(v) if v is not None else None
        return d


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract ``zip_path`` into ``dest``, rejecting path-traversal entries.

    Submission ZIPs produced by :func:`src.export_submission.make_submission_zip`
    contain only POSIX-relative entries (e.g. ``model.py``,
    ``artifacts/checkpoint.pt``). We still defend against malicious or
    accidentally-absolute paths because this helper is meant to be
    pointed at arbitrary user-uploaded files.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            # Reject absolute paths and parent-traversal entries.
            if name.startswith("/") or name.startswith("\\"):
                raise RuntimeError(
                    f"submission zip {zip_path} contains an absolute path "
                    f"entry ({name!r}); refusing to extract."
                )
            if ".." in Path(name).parts:
                raise RuntimeError(
                    f"submission zip {zip_path} contains a parent-traversal "
                    f"entry ({name!r}); refusing to extract."
                )
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(
                    f"submission zip {zip_path} would extract {name!r} "
                    f"outside {dest_resolved}; refusing."
                )
            zf.extract(member, dest)


def _looks_like_submission_root(p: Path) -> bool:
    """A submission root has ``model.py`` next to an ``artifacts/`` dir."""
    return (p / "model.py").is_file() and (p / "artifacts").is_dir()


def _find_submission_root(unpack_dir: Path) -> Path:
    """Locate the submission root inside ``unpack_dir``.

    Most ZIPs put the runtime files directly at the root; some users
    re-zip with a wrapper folder. We accept either layout.
    """
    if _looks_like_submission_root(unpack_dir):
        return unpack_dir
    # Single top-level dir that wraps the submission?
    children = [p for p in unpack_dir.iterdir() if p.is_dir()]
    if len(children) == 1 and _looks_like_submission_root(children[0]):
        return children[0]
    # Fall back to any deeper match.
    for cand in unpack_dir.rglob("model.py"):
        root = cand.parent
        if _looks_like_submission_root(root):
            return root
    raise RuntimeError(
        f"could not locate a submission root inside {unpack_dir}; expected "
        "a directory containing model.py + artifacts/. Make sure the ZIP "
        "was produced by src.export_submission.make_submission_zip."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_submission(
    src: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] = "submission",
    overwrite: bool = True,
) -> ImportedSubmission:
    """Unpack a previously-built submission ZIP (or dir) into ``out_dir``.

    ``src`` may be:

    - A ``.zip`` file path (typically the file the user re-uploaded to
      Colab from local disk or Drive).
    - An already-unpacked submission directory -- it is copied into
      ``out_dir`` so the working tree owns its own writable copy.

    When ``overwrite=True`` (default) any pre-existing ``out_dir`` is
    deleted first so a stale partial unpack cannot poison the result.

    Returns an :class:`ImportedSubmission` carrying the canonical paths
    and parsed metadata. The submission is immediately usable by:

    - ``scripts/smoke_test_submission.py --submission <out_dir>``
    - The notebook's cell 20 / cell 21 / Codabench re-upload flow
    - :func:`materialize_as_run` for a LoRA-resume seed
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"import_submission: src does not exist: {src_path}")

    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"import_submission: out_dir={out} already exists; pass "
                "overwrite=True to replace it."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if src_path.is_dir():
        if not _looks_like_submission_root(src_path):
            raise RuntimeError(
                f"import_submission: {src_path} is a directory but does not "
                "look like a submission root (missing model.py or artifacts/)."
            )
        # Copy contents so the out_dir is independent of the source.
        for entry in src_path.iterdir():
            tgt = out / entry.name
            if entry.is_dir():
                shutil.copytree(entry, tgt)
            else:
                shutil.copy2(entry, tgt)
        sub_root = out
    elif src_path.suffix.lower() == ".zip":
        # Extract into a staging dir first to handle ZIPs that wrap the
        # submission in a top-level folder; then copy the resolved root
        # into ``out``.
        with tempfile.TemporaryDirectory(prefix="submission_import_") as tmp:
            tmp_path = Path(tmp)
            _safe_extract_zip(src_path, tmp_path)
            staged_root = _find_submission_root(tmp_path)
            if staged_root != tmp_path:
                # Move the wrapped contents up so ``out`` is the submission root.
                for entry in staged_root.iterdir():
                    tgt = out / entry.name
                    if entry.is_dir():
                        shutil.copytree(entry, tgt)
                    else:
                        shutil.copy2(entry, tgt)
            else:
                for entry in tmp_path.iterdir():
                    tgt = out / entry.name
                    if entry.is_dir():
                        shutil.copytree(entry, tgt)
                    else:
                        shutil.copy2(entry, tgt)
        sub_root = out
    else:
        raise ValueError(
            f"import_submission: unsupported source {src_path} -- pass a "
            ".zip file or an unpacked submission directory."
        )

    if not _looks_like_submission_root(sub_root):
        raise RuntimeError(
            f"import_submission: extracted directory {sub_root} is missing "
            "model.py or artifacts/. The source ZIP is probably not a "
            "submission produced by make_submission_zip."
        )

    artifacts = sub_root / "artifacts"
    ckpt = artifacts / "checkpoint.pt"
    meta_path = artifacts / "runtime_meta.json"
    if not ckpt.exists():
        raise RuntimeError(
            f"import_submission: {ckpt} is missing -- the ZIP is not a "
            "valid submission bundle."
        )
    if not meta_path.exists():
        raise RuntimeError(
            f"import_submission: {meta_path} is missing -- the ZIP is not "
            "a valid submission bundle."
        )

    try:
        runtime_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"import_submission: could not parse {meta_path}: {exc}"
        ) from exc

    # Peek inside checkpoint.pt for model_cfg / train_cfg / metrics. We
    # do not need the full state_dict here, but ``torch.load`` does load
    # everything onto CPU; the cost is one-time at import.
    import torch as _torch  # local import to keep module import light

    try:
        ckpt_blob: dict = _torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"import_submission: could not torch.load {ckpt}: {exc}"
        ) from exc

    model_cfg = dict(ckpt_blob.get("model_cfg") or {})
    train_cfg = dict(ckpt_blob.get("train_cfg") or {})
    result_blob = dict(ckpt_blob.get("result") or {})

    # Optional sibling artifacts: only surfaced if the file actually exists.
    pool_stats_path = artifacts / "pool_features_stats.json"
    cluster_centroids_path = artifacts / "cluster_centroids.npy"
    training_cache_dir = sub_root / "cache"
    lora_adapter_dir = sub_root / "lora_adapter"

    lora_block = dict(runtime_meta.get("lora") or {})
    lora_mode = str(lora_block.get("mode", "none"))

    imported = ImportedSubmission(
        src=src_path,
        out_dir=sub_root,
        checkpoint_path=ckpt,
        runtime_meta_path=meta_path,
        runtime_meta=runtime_meta,
        run_id=str(runtime_meta.get("run_id") or ""),
        model_name=str(
            runtime_meta.get("model_name")
            or ckpt_blob.get("model_name")
            or ""
        ),
        encoder_model_id=str(runtime_meta.get("encoder_model_id") or ""),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        best_val_log_loss=float(
            result_blob.get("best_val_log_loss", float("nan"))
        ),
        best_val_brier=float(
            result_blob.get("best_val_brier", float("nan"))
        ),
        epoch_best=int(result_blob.get("epoch_best", -1) or -1),
        pool_stats_path=pool_stats_path if pool_stats_path.exists() else None,
        cluster_centroids_path=(
            cluster_centroids_path if cluster_centroids_path.exists() else None
        ),
        training_cache_dir=training_cache_dir if training_cache_dir.exists() else None,
        lora_adapter_dir=(
            lora_adapter_dir
            if (lora_mode == "adapter_only" and lora_adapter_dir.exists())
            else None
        ),
        lora_mode=lora_mode,
    )

    LOG.info(
        "Imported submission from %s -> %s (run_id=%s model=%s lora=%s)",
        src_path,
        sub_root,
        imported.run_id,
        imported.model_name,
        lora_mode,
    )
    return imported


# ---------------------------------------------------------------------------
# Re-materialize as a trainable run (for LoRA resume / cell-13 reruns)
# ---------------------------------------------------------------------------


def materialize_as_run(
    imported: ImportedSubmission,
    *,
    checkpoints_dir: str | os.PathLike[str],
    run_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Write the imported checkpoint into ``artifacts/checkpoints/``.

    This is what you call when you want to **continue training** from an
    imported submission (e.g. seed a fresh LoRA run with a previously-
    trained head). It writes two files alongside the existing head-only
    runs so the rest of the notebook -- ``runs_df``, ``SELECTED_RUN_ID``,
    the LoRA cell's ``base_checkpoint`` resolver -- sees the imported run
    as just another row.

    - ``{checkpoints_dir}/{run_id}.pt``   -- the head checkpoint
    - ``{checkpoints_dir}/{run_id}.json`` -- a minimal metadata sidecar
      with model_cfg / train_cfg / result block / source provenance

    The run_id defaults to ``imported.run_id``, suffixed with
    ``_imported`` to make it visually obvious in the runs table that
    this row came from a re-upload.

    Returns a small dict suitable for appending to ``ALL_RUNS`` /
    ``runs_df``::

        {
            "run_id": "...",
            "model_name": "...",
            "k": 16,
            "seed": 0,
            "best_val_log_loss": 0.444,
            "best_val_brier": 0.118,
            "best_val_auc": None,
            "epoch_best": 5,
            "checkpoint_path": "...",
            "metadata_path": "...",
            "elapsed_seconds": 0.0,
            "feature_tag": "imported",
            "use_pool_features": ...,
            ...
        }
    """
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rid = run_id or (imported.run_id + "_imported")
    target_ckpt = ckpt_dir / f"{rid}.pt"
    target_meta = ckpt_dir / f"{rid}.json"

    if target_ckpt.exists() and not overwrite:
        raise FileExistsError(
            f"materialize_as_run: {target_ckpt} already exists; pass "
            "overwrite=True to replace it."
        )

    shutil.copy2(imported.checkpoint_path, target_ckpt)

    sidecar: dict[str, Any] = {
        "run_id": rid,
        "model_name": imported.model_name,
        "model_cfg": imported.model_cfg,
        "train_cfg": imported.train_cfg,
        "result": {
            "best_val_log_loss": imported.best_val_log_loss,
            "best_val_brier": imported.best_val_brier,
            "best_val_auc": None,
            "epoch_best": imported.epoch_best,
            "history": [],
            "n_train": 0,
            "n_val": 0,
        },
        "extra": {
            "encoder_model_id": imported.encoder_model_id,
            "source_zip": str(imported.src),
            "source_run_id": imported.run_id,
            "imported": True,
            "lora_mode": imported.lora_mode,
        },
    }
    target_meta.write_text(json.dumps(sidecar, indent=2, default=str))

    k = int((imported.model_cfg or {}).get("k", 0) or 0)
    return {
        "run_id": rid,
        "model_name": imported.model_name,
        "k": k,
        "seed": 0,
        "feature_tag": "imported",
        "use_pool_features": bool(
            (imported.model_cfg or {}).get("use_pool_features", False)
        ),
        "use_cluster_features": bool(
            (imported.model_cfg or {}).get("use_cluster_features", False)
        ),
        "use_judge_features": bool(
            (imported.model_cfg or {}).get("use_judge_features", False)
        ),
        "use_nn_features": bool(
            (imported.model_cfg or {}).get("use_nn_features", False)
        ),
        "epoch_best": int(imported.epoch_best),
        "best_val_log_loss": float(imported.best_val_log_loss),
        "best_val_brier": float(imported.best_val_brier),
        "best_val_auc": None,
        "checkpoint_path": str(target_ckpt),
        "metadata_path": str(target_meta),
        "elapsed_seconds": 0.0,
    }


__all__ = [
    "ImportedSubmission",
    "import_submission",
    "materialize_as_run",
]
