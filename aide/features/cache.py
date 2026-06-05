"""Derive-once feature-shard cache: fold-aware keys, atomic idempotent writes, manifest.

A *shard* is the cached output of one ``(embedding_family x feature_group x fold)``
derivation. Per Plan 4 §A the cache key

    (embedding_family, feature_group, outer_fold, split_seed, n_folds, code_version[, inner_fold])

makes OOF discipline part of the identity: a label-derived group fit on fold ``f``'s
*train* items is a different shard from fold ``g``'s, so the funnel can never assemble a
mismatched-fold (leaky) feature by accident. ``outer_fold="all"`` marks a fold-invariant
neutral/geometry group (derived once per family).

The key/path/atomicity/manifest logic here is backend-agnostic and numpy-only, so the
correctness-critical cache contract is fully locally unit-testable. The on-disk container
is **pluggable**: ``NpzBackend`` (numpy-native, the local default + test backend) vs an
optional ``ParquetBackend`` (pyarrow, the Colab default) — per the "Colab-only, contracts
local" decision. Writes are atomic (``.tmp`` in the same dir → ``os.replace``) and
idempotent (a shard whose key exists is never recomputed), so a Colab timeout can never
leave a half-written shard and a resumed run skips what is already present.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aide.harness.funnel import CacheMissError, FeatureBlock

_SAFE = re.compile(r"[^0-9A-Za-z._-]")


def _slug(value) -> str:
    """Filesystem-safe token for a key component (keeps the readable §A scheme)."""
    return _SAFE.sub("-", str(value))


def _short_hash(value) -> str:
    """8-hex digest of the RAW value — disambiguates tokens that ``_slug`` collapses
    (e.g. ``"a/b"`` and ``"a-b"`` both slug to ``"a-b"`` but hash differently)."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]


def _safe_component(name, what: str) -> str:
    """Validate a value used as a literal PATH component (family/group): reject path
    separators and other unsafe chars so a stray ``/`` can't escape the cache root."""
    s = str(name)
    if not re.fullmatch(r"[0-9A-Za-z._-]+", s):
        raise ValueError(
            f"unsafe {what} {name!r}: must match [0-9A-Za-z._-]+ (no path separators)")
    return s


def content_hash(*parts) -> str:
    """Stable, order-sensitive SHA-256 over the given parts (``str`` or ``bytes``).

    Callers hash the *inputs* of a derivation (embedding file bytes/path, CSR digest,
    git rev) and pass the result as ``inputs_hash`` so a stale shard can be detected.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x1f")  # unit separator so ("a","b") != ("ab",)
        h.update(p if isinstance(p, (bytes, bytearray)) else str(p).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class ShardKey:
    embedding_family: str
    feature_group: str
    outer_fold: object       # int fold id, or the string "all" (fold-invariant)
    split_seed: int
    n_folds: int
    code_version: str
    inner_fold: object = None  # int (layer-2 recursive variant) or None

    def stem(self) -> str:
        fold = "all" if self.outer_fold == "all" else int(self.outer_fold)  # type: ignore[arg-type]
        parts = [f"fold{_slug(fold)}"]
        if self.inner_fold is not None:
            parts.append(f"inner{_slug(int(self.inner_fold))}")  # type: ignore[arg-type]
        parts += [f"seed{_slug(self.split_seed)}",
                  f"nf{_slug(self.n_folds)}",
                  f"{_slug(self.code_version)}-{_short_hash(self.code_version)}"]
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {
            "embedding_family": self.embedding_family,
            "feature_group": self.feature_group,
            "outer_fold": self.outer_fold,
            "split_seed": self.split_seed,
            "n_folds": self.n_folds,
            "code_version": self.code_version,
            "inner_fold": self.inner_fold,
        }


class ShardBackend:
    """Pluggable shard container. ``ext`` has no leading dot."""

    ext = ""

    def write(self, path: Path, block: FeatureBlock) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self, path: Path) -> FeatureBlock:  # pragma: no cover - interface
        raise NotImplementedError


class NpzBackend(ShardBackend):
    """numpy-native ``.npz`` container — the local default and the funnel's format.

    ``allow_pickle`` stays at the numpy default (False): shards hold only float and
    unicode-string arrays, so the load path never touches the pickle code-exec surface.
    """

    ext = "npz"

    def write(self, path: Path, block: FeatureBlock) -> None:
        with open(path, "wb") as fh:  # explicit handle so np.savez can't re-append ".npz"
            np.savez(fh,
                     X=np.asarray(block.X, dtype=np.float32),
                     columns=np.asarray(block.columns, dtype=str),
                     row_ids=np.asarray(block.row_ids).astype(str))

    def read(self, path: Path) -> FeatureBlock:
        d = np.load(path)
        return FeatureBlock(X=np.asarray(d["X"], dtype=np.float32),
                            columns=[str(c) for c in d["columns"]],
                            row_ids=np.asarray(d["row_ids"]).astype(str))


def _atomic(path: Path, write_fn) -> None:
    """Write via a same-dir temp file then ``os.replace`` (atomic on POSIX/NTFS).

    On any failure the temp file is removed so no ``.tmp`` litter or partial final file
    survives — the property the derive-once resume relies on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _validate(block: FeatureBlock) -> None:
    n_rows = int(np.asarray(block.X).shape[0])
    n_cols = int(np.asarray(block.X).shape[1]) if np.asarray(block.X).ndim == 2 else -1
    cols = list(block.columns)
    if n_cols != len(cols):
        raise ValueError(f"block has {n_cols} X columns but {len(cols)} column names")
    if len(set(cols)) != len(cols):
        raise ValueError("duplicate feature column names in block — assembly would be ambiguous")
    if len(np.asarray(block.row_ids)) != n_rows:
        raise ValueError(f"block has {n_rows} X rows but {len(np.asarray(block.row_ids))} row_ids")


class FeatureCache:
    def __init__(self, root, backend: ShardBackend | None = None, code_version: str = "dev"):
        self.root = Path(root)
        self.backend = backend or NpzBackend()
        self.code_version = code_version

    # --- key construction ------------------------------------------------------------
    def key(self, embedding_family: str, feature_group: str, *, fold, seed: int,
            n_folds: int, inner_fold=None) -> ShardKey:
        return ShardKey(embedding_family=embedding_family, feature_group=feature_group,
                        outer_fold=fold, split_seed=seed, n_folds=n_folds,
                        code_version=self.code_version, inner_fold=inner_fold)

    # --- paths -----------------------------------------------------------------------
    def shard_path(self, key: ShardKey) -> Path:
        family = _safe_component(key.embedding_family, "embedding_family")
        group = _safe_component(key.feature_group, "feature_group")
        return self.root / family / group / f"{key.stem()}.{self.backend.ext}"

    def meta_path(self, key: ShardKey) -> Path:
        return self.shard_path(key).with_suffix("").with_suffix(".meta.json")

    @property
    def index_path(self) -> Path:
        return self.root / "INDEX.json"

    # --- existence / read ------------------------------------------------------------
    def exists(self, key: ShardKey) -> bool:
        return self.shard_path(key).exists()

    def read_shard(self, key: ShardKey) -> FeatureBlock:
        p = self.shard_path(key)
        if not p.exists():
            raise CacheMissError(
                f"feature shard not cached: family={key.embedding_family!r} "
                f"group={key.feature_group!r} fold={key.outer_fold!r} "
                f"inner={key.inner_fold!r} seed={key.split_seed} at {p} — the cache is "
                f"load-only and never recomputes inside a training run; derive it offline")
        return self.backend.read(p)

    def read_meta(self, key: ShardKey) -> dict:
        mp = self.meta_path(key)
        if not mp.exists():
            raise CacheMissError(f"no shard meta at {mp}")
        return json.loads(mp.read_text(encoding="utf-8"))

    # --- write (the ONLY writer) -----------------------------------------------------
    def write_shard(self, key: ShardKey, block: FeatureBlock, *, inputs_hash: str,
                    overwrite: bool = False) -> str:
        """Atomically persist one shard. Returns ``"written"`` or ``"skipped"``.

        Derive-once: an existing shard for ``key`` is skipped (never recomputed) unless
        ``overwrite=True``. Shard then meta are written atomically; the manifest is
        updated last so the INDEX only ever references a fully-written shard.
        """
        if self.exists(key) and not overwrite:
            return "skipped"
        _validate(block)
        shard_p = self.shard_path(key)
        _atomic(shard_p, lambda tmp: self.backend.write(tmp, block))
        meta = {
            **key.as_dict(),
            "inputs_hash": inputs_hash,
            "columns": list(block.columns),
            "n_rows": int(np.asarray(block.X).shape[0]),
            "n_cols": len(list(block.columns)),
            "container": self.backend.ext,
            "rel_path": str(shard_p.relative_to(self.root)),
        }
        _atomic(self.meta_path(key),
                lambda tmp: Path(tmp).write_text(json.dumps(meta, sort_keys=True),
                                                 encoding="utf-8"))
        self._index_upsert(meta)
        return "written"

    # --- manifest --------------------------------------------------------------------
    def load_index(self) -> dict:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _index_upsert(self, meta: dict) -> None:
        idx = self.load_index()
        idx[meta["rel_path"]] = meta
        _atomic(self.index_path,
                lambda tmp: Path(tmp).write_text(json.dumps(idx, sort_keys=True),
                                                 encoding="utf-8"))

    def rebuild_index(self) -> dict:
        """Reconstruct INDEX.json by scanning all ``*.meta.json`` under the root."""
        idx = {}
        for mp in self.root.rglob("*.meta.json"):
            meta = json.loads(mp.read_text(encoding="utf-8"))
            idx[meta["rel_path"]] = meta
        _atomic(self.index_path,
                lambda tmp: Path(tmp).write_text(json.dumps(idx, sort_keys=True),
                                                 encoding="utf-8"))
        return idx
