"""Embedding I/O — one-time float16 ``.npy`` conversion + fast load (Plan 4 Prep C / §B.2).

Reading the 311k×4096 float64 parquet over Drive FUSE is the single biggest time sink in a
run (observed minutes). ``convert_embeddings_to_npy`` does that read ONCE, writing a compact
float16 ``.npy`` (+ a keys JSON) next to the parquet; subsequent runs ``np.load(mmap_mode)``
it in a blink. ``driver.load_embeddings`` prefers the ``.npy`` sibling automatically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def flat_to_matrix(flat, n_rows, dtype=np.float16):
    """Reshape a flat list<float> values buffer to ``[n_rows, dim]`` at ``dtype``."""
    flat = np.asarray(flat)
    dim = flat.shape[0] // n_rows
    return flat.reshape(n_rows, dim).astype(dtype)


def npy_paths(parquet_path):
    p = Path(parquet_path)
    return p.with_suffix(".f16.npy"), p.with_suffix(".keys.json")


def convert_embeddings_to_npy(parquet_path, *, overwrite=False):
    """Read an items/subjects parquet once → write a float16 ``.npy`` + keys JSON sibling.
    Returns (npy_path, keys_path). Colab (needs pyarrow)."""
    import pyarrow.parquet as pq
    npy, keys_path = npy_paths(parquet_path)
    if npy.exists() and keys_path.exists() and not overwrite:
        return npy, keys_path
    t = pq.read_table(parquet_path)
    keys = [str(k) for k in t.column(t.column_names[0]).to_pylist()]
    flat = t.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
    np.save(npy, flat_to_matrix(flat, len(keys), np.float16))
    keys_path.write_text(json.dumps(keys), encoding="utf-8")
    return npy, keys_path


def load_embeddings_npy(npy_path, keys_path, *, mmap=True):
    """Load (keys, emb) from the float16 ``.npy`` cache. ``emb`` is memmapped if ``mmap``."""
    keys = json.loads(Path(keys_path).read_text(encoding="utf-8"))
    emb = np.load(npy_path, mmap_mode="r" if mmap else None)
    return keys, emb
