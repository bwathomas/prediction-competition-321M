"""Tests for the float16 .npy embedding cache (Prep C). The parquet→npy conversion is
Colab (pyarrow); the reshape helper + the load round-trip are tested locally."""
import json

import numpy as np

from aide.features.embed_io import flat_to_matrix, load_embeddings_npy, npy_paths


def test_flat_to_matrix_reshapes_and_casts():
    m = flat_to_matrix(np.arange(6.0), 2, dtype=np.float16)
    assert m.shape == (2, 3) and m.dtype == np.float16
    assert m.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_npy_paths_siblings(tmp_path):
    npy, keys = npy_paths(tmp_path / "items.parquet")
    assert npy.name == "items.f16.npy" and keys.name == "items.keys.json"


def test_load_embeddings_npy_roundtrip(tmp_path):
    emb = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float16)
    np.save(tmp_path / "e.npy", emb)
    (tmp_path / "e.keys.json").write_text(json.dumps(["a", "b", "c"]))
    keys, loaded = load_embeddings_npy(tmp_path / "e.npy", tmp_path / "e.keys.json", mmap=False)
    assert keys == ["a", "b", "c"]
    assert np.array_equal(np.asarray(loaded), emb)
