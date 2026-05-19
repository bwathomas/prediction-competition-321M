"""Google Drive persistence for the encoder embedding cache.

The encoder pass is the most expensive step of the pipeline. Persisting its
output to Google Drive means subsequent Colab sessions skip encoding
entirely whenever the underlying text data has not changed.

Behavior:

- Mounts Google Drive in Colab (no-op if already mounted, no-op outside Colab).
- Downloads ``{drive_folder}/{encoder_slug}/`` into the local cache directory
  if it exists and the persisted ``content_hash`` matches the expected hash
  for the current dataset.
- Uploads atomically: writes to ``{drive_folder}/{encoder_slug}.tmp/`` first,
  then renames into place. The web UI can show stale state during long
  uploads; the rename avoids consumers reading partial caches.
- Supports partial invalidation: if the on-disk cache has 90% of the items
  but 100 new items were added since last run, only those 100 get encoded
  and the merged cache is uploaded back.

Outside Colab the module is a no-op; ``enabled: false`` in the YAML is the
recommended way to disable it explicitly, but the helpers also degrade
gracefully if ``google.colab`` cannot be imported.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

LOG = logging.getLogger("drive_cache")


# ---------------------------------------------------------------------------
# Dataclass for the resolved cache state
# ---------------------------------------------------------------------------


@dataclass
class DriveCacheStatus:
    """The result of trying to populate the local cache from Drive."""

    enabled: bool
    mounted: bool
    drive_folder: Path | None
    local_folder: Path
    cache_hit: bool                  # local cache fully matches expected hash
    partial_hit: bool                # cache present but content hash differs
    expected_hash: str
    cached_hash: str | None
    encoded_n_items_cached: int
    encoded_n_subjects_cached: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mounted": self.mounted,
            "drive_folder": str(self.drive_folder) if self.drive_folder else None,
            "local_folder": str(self.local_folder),
            "cache_hit": self.cache_hit,
            "partial_hit": self.partial_hit,
            "expected_hash": self.expected_hash,
            "cached_hash": self.cached_hash,
            "encoded_n_items_cached": self.encoded_n_items_cached,
            "encoded_n_subjects_cached": self.encoded_n_subjects_cached,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Mount helpers
# ---------------------------------------------------------------------------


def mount_drive_if_needed() -> bool:
    """Mount Google Drive in Colab if it isn't already mounted.

    Returns True on success (or if already mounted), False outside Colab or
    on mount failure. Never raises -- callers downgrade to a local-only run.
    """
    if "google.colab" not in sys.modules and not _looks_like_colab():
        return False
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False
    if os.path.exists("/content/drive/MyDrive"):
        LOG.info("Google Drive already mounted at /content/drive/MyDrive")
        return True
    try:
        drive.mount("/content/drive")
        return os.path.exists("/content/drive/MyDrive")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("drive.mount failed: %s", exc)
        return False


def _looks_like_colab() -> bool:
    return any(k in os.environ for k in ("COLAB_GPU", "COLAB_RELEASE_TAG"))


# ---------------------------------------------------------------------------
# Atomic copy primitives (rename-after-write)
# ---------------------------------------------------------------------------


def _copy_dir_atomic(src: Path, dst: Path) -> None:
    """Copy `src` into `dst` via a ``.tmp`` staging directory + rename.

    `dst` is replaced atomically (rmtree the old, rename in the new).
    Works across filesystems because we shutil.copytree to staging and then
    use os.replace for the final move when on the same FS, falling back to
    a plain shutil.move otherwise.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.with_name(dst.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    if dst.exists():
        shutil.rmtree(dst)
    try:
        os.replace(staging, dst)
    except OSError:
        shutil.move(str(staging), str(dst))


def _copy_files_atomic(src_dir: Path, dst_dir: Path, *, files: list[str]) -> None:
    """Copy a fixed file list from src_dir to dst_dir atomically."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    staging = dst_dir.with_name(dst_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for fname in files:
            src_path = src_dir / fname
            if src_path.exists():
                shutil.copy2(src_path, staging / fname)
        # Promote staging -> dst (replace existing). We can't os.replace a
        # directory in all cases; do an in-place merge.
        for f in staging.iterdir():
            target = dst_dir / f.name
            if target.exists():
                target.unlink()
            shutil.move(str(f), str(target))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sync entry points
# ---------------------------------------------------------------------------


_CACHE_FILES = ("meta.json", "items.parquet", "subjects.parquet", "encoding_log.json")


def download_to_local(
    *,
    drive_folder: Path,
    local_folder: Path,
    expected_hash: str,
) -> DriveCacheStatus:
    """Try to populate the local cache from a Drive folder.

    The "cache hit" predicate is: the Drive folder exists, contains a
    ``meta.json`` with a ``content_hash`` field matching ``expected_hash``,
    and at least one of ``items.parquet`` or ``subjects.parquet`` exists.

    On a strict hit we copy the parquet + meta files into ``local_folder``.
    On a hash mismatch we still copy whatever's there so subsequent
    encoding only fills in the missing rows.
    """
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    drive_meta_path = drive_folder / "meta.json"
    if not drive_meta_path.exists():
        return DriveCacheStatus(
            enabled=True,
            mounted=True,
            drive_folder=drive_folder,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="drive cache empty (no meta.json)",
        )

    try:
        meta = json.loads(drive_meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    cached_hash = str(meta.get("content_hash") or "") or None
    n_items_cached = int(meta.get("n_items", 0) or 0)
    n_subjects_cached = int(meta.get("n_subjects", 0) or 0)

    # Copy whatever is there into the local folder so encoding can resume.
    files_present = [f for f in _CACHE_FILES if (drive_folder / f).exists()]
    if files_present:
        _copy_files_atomic(drive_folder, local_folder, files=files_present)

    full_hit = cached_hash == expected_hash and (
        (local_folder / "items.parquet").exists()
        or (local_folder / "subjects.parquet").exists()
    )
    partial_hit = (cached_hash is not None) and not full_hit and files_present
    if full_hit:
        reason = "drive cache HIT (content hash matches; skipping encoding)"
    elif partial_hit:
        reason = (
            "drive cache PARTIAL hit (content hash differs; will encode only "
            "the missing items / subjects then upload the merged cache)"
        )
    else:
        reason = "drive cache MISS"

    return DriveCacheStatus(
        enabled=True,
        mounted=True,
        drive_folder=drive_folder,
        local_folder=local_folder,
        cache_hit=bool(full_hit),
        partial_hit=bool(partial_hit),
        expected_hash=expected_hash,
        cached_hash=cached_hash,
        encoded_n_items_cached=n_items_cached,
        encoded_n_subjects_cached=n_subjects_cached,
        reason=reason,
    )


def upload_from_local(
    *,
    local_folder: Path,
    drive_folder: Path,
    files: list[str] | None = None,
) -> dict:
    """Atomically publish the local cache to a Drive folder.

    Writes into ``{drive_folder}.tmp/`` first, then renames into place to
    avoid consumers reading a half-uploaded cache.
    """
    files = list(files or _CACHE_FILES)
    drive_folder = Path(drive_folder)
    drive_folder.parent.mkdir(parents=True, exist_ok=True)
    staging = drive_folder.with_name(drive_folder.name + ".tmp")

    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    t0 = time.time()
    for fname in files:
        src_path = local_folder / fname
        if src_path.exists():
            shutil.copy2(src_path, staging / fname)
            written.append(fname)
        else:
            skipped.append(fname)

    # Promote staging -> drive_folder.
    if drive_folder.exists():
        shutil.rmtree(drive_folder, ignore_errors=True)
    try:
        os.replace(staging, drive_folder)
    except OSError:
        shutil.move(str(staging), str(drive_folder))

    elapsed = time.time() - t0
    LOG.info(
        "drive upload: wrote %d files to %s in %.1fs (skipped %d)",
        len(written),
        drive_folder,
        elapsed,
        len(skipped),
    )
    return {
        "drive_folder": str(drive_folder),
        "written": written,
        "skipped": skipped,
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# High-level resolve
# ---------------------------------------------------------------------------


def resolve_cache(
    *,
    cfg: Mapping,
    encoder_slug: str,
    local_cache_root: Path,
    expected_hash: str,
) -> DriveCacheStatus:
    """One-shot helper: mount + download if cfg.drive_cache enables it.

    Returns a DriveCacheStatus describing what happened so the notebook can
    log the decision (hit / partial / miss) and act accordingly.
    """
    local_folder = Path(local_cache_root) / encoder_slug
    local_folder.mkdir(parents=True, exist_ok=True)

    dc = cfg.get("drive_cache") if cfg else None
    enabled = bool(dc and dc.get("enabled", False))
    if not enabled:
        return DriveCacheStatus(
            enabled=False,
            mounted=False,
            drive_folder=None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="drive_cache.enabled = false",
        )

    if not bool(dc.get("download_on_start", True)):
        # User explicitly disabled downloads; still try to mount so upload works.
        mounted = mount_drive_if_needed()
        return DriveCacheStatus(
            enabled=True,
            mounted=bool(mounted),
            drive_folder=Path(dc.get("folder", "")) / encoder_slug if dc.get("folder") else None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="download_on_start = false (skipping drive download)",
        )

    mounted = mount_drive_if_needed()
    if not mounted:
        return DriveCacheStatus(
            enabled=True,
            mounted=False,
            drive_folder=None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="drive mount unavailable (not in Colab or mount failed)",
        )

    drive_root = Path(dc.get("folder", ""))
    drive_folder = drive_root / encoder_slug
    return download_to_local(
        drive_folder=drive_folder,
        local_folder=local_folder,
        expected_hash=expected_hash,
    )


# ---------------------------------------------------------------------------
# Judge cache helpers (symmetric to the encoder embedding cache)
#
# The judge cache lives at:
#     local:  {cache_dir}/{judge_slug}/scores.parquet  (+ meta.json)
#     drive:  {drive_folder}/{judge_slug}/scores.parquet
#
# The slug already encodes (model_id, prompt_version), so a hit on the slug
# is a strict hit -- but we still verify ``content_hash`` (= prompt_version)
# in meta.json before promoting a partial-overlap parquet as authoritative.
# ---------------------------------------------------------------------------


_JUDGE_CACHE_FILES = ("scores.parquet", "meta.json")


def download_judge_to_local(
    *,
    drive_folder: Path,
    local_folder: Path,
    expected_hash: str,
) -> DriveCacheStatus:
    """Try to populate the local judge cache from a Drive folder.

    The "cache hit" predicate is: the Drive folder exists, contains a
    ``meta.json`` whose ``content_hash`` field equals ``expected_hash``
    (which the caller computes as the judge prompt-version hash), and
    ``scores.parquet`` exists. We always copy whatever files are present so
    incremental encoding can resume from a partial cache.
    """
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    drive_meta_path = drive_folder / "meta.json"
    if not drive_meta_path.exists():
        return DriveCacheStatus(
            enabled=True,
            mounted=True,
            drive_folder=drive_folder,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="judge drive cache empty (no meta.json)",
        )
    try:
        meta = json.loads(drive_meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    cached_hash = str(meta.get("content_hash") or "") or None
    n_rows_cached = int(meta.get("n_rows", 0) or 0)

    files_present = [f for f in _JUDGE_CACHE_FILES if (drive_folder / f).exists()]
    if files_present:
        _copy_files_atomic(drive_folder, local_folder, files=files_present)

    has_scores = (local_folder / "scores.parquet").exists()
    full_hit = bool(cached_hash == expected_hash and has_scores)
    partial_hit = (cached_hash is not None) and not full_hit and files_present

    if full_hit:
        reason = "judge drive cache HIT (prompt_version matches; skipping re-score)"
    elif partial_hit:
        reason = (
            "judge drive cache PARTIAL hit (prompt_version differs; will append "
            "missing pairs then upload the merged cache)"
        )
    else:
        reason = "judge drive cache MISS"

    return DriveCacheStatus(
        enabled=True,
        mounted=True,
        drive_folder=drive_folder,
        local_folder=local_folder,
        cache_hit=full_hit,
        partial_hit=partial_hit,
        expected_hash=expected_hash,
        cached_hash=cached_hash,
        encoded_n_items_cached=n_rows_cached,
        encoded_n_subjects_cached=0,
        reason=reason,
    )


def resolve_judge_cache(
    *,
    cfg: Mapping,
    judge_slug: str,
    local_cache_root: Path,
    expected_hash: str,
) -> DriveCacheStatus:
    """One-shot helper: mount Drive + download the judge cache if available.

    ``cfg`` is the full config dict (we look at ``cfg["judge_drive_cache"]``).
    ``judge_slug`` is the directory name produced by ``src.judge.judge_slug``
    so a hit on the slug already implies a (model_id, prompt_version) match.
    """
    local_folder = Path(local_cache_root) / judge_slug
    local_folder.mkdir(parents=True, exist_ok=True)

    dc = cfg.get("judge_drive_cache") if cfg else None
    enabled = bool(dc and dc.get("enabled", False))
    if not enabled:
        return DriveCacheStatus(
            enabled=False,
            mounted=False,
            drive_folder=None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="judge_drive_cache.enabled = false",
        )

    if not bool(dc.get("download_on_start", True)):
        mounted = mount_drive_if_needed()
        return DriveCacheStatus(
            enabled=True,
            mounted=bool(mounted),
            drive_folder=Path(dc.get("folder", "")) / judge_slug if dc.get("folder") else None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="judge_drive_cache.download_on_start = false",
        )

    mounted = mount_drive_if_needed()
    if not mounted:
        return DriveCacheStatus(
            enabled=True,
            mounted=False,
            drive_folder=None,
            local_folder=local_folder,
            cache_hit=False,
            partial_hit=False,
            expected_hash=expected_hash,
            cached_hash=None,
            encoded_n_items_cached=0,
            encoded_n_subjects_cached=0,
            reason="judge drive mount unavailable (not in Colab or mount failed)",
        )

    drive_root = Path(dc.get("folder", ""))
    drive_folder = drive_root / judge_slug
    return download_judge_to_local(
        drive_folder=drive_folder,
        local_folder=local_folder,
        expected_hash=expected_hash,
    )


def upload_judge_cache(
    *,
    local_folder: Path,
    drive_folder: Path,
) -> dict:
    """Publish the local judge cache (scores.parquet + meta.json) to Drive.

    Atomic via ``{drive_folder}.tmp/`` staging + rename to avoid consumers
    reading a half-written parquet.
    """
    return upload_from_local(
        local_folder=local_folder,
        drive_folder=drive_folder,
        files=list(_JUDGE_CACHE_FILES),
    )


__all__ = [
    "DriveCacheStatus",
    "download_judge_to_local",
    "download_to_local",
    "mount_drive_if_needed",
    "resolve_cache",
    "resolve_judge_cache",
    "upload_from_local",
    "upload_judge_cache",
]
