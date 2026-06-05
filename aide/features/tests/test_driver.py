"""Local tests for the driver's numpy-pure helpers. The data-loading / passrate / chunk
derivation paths are heavy (pyarrow/scipy/faiss) and are validated on Colab against a real
embedding+label slice; here we lock the import-safe pieces."""
import numpy as np

from aide.features.driver import FAMILY_SLUG, _concat_blocks, content_inputs_hash, unit_rows
from aide.harness.funnel import FeatureBlock


def test_family_slugs_cover_three_families():
    assert set(FAMILY_SLUG) == {"llama", "qwen", "mistral"}
    assert FAMILY_SLUG["qwen"] == "Qwen__Qwen3-Embedding-8B"


def test_unit_rows_normalizes():
    x = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


def test_unit_rows_handles_zero_vector():
    x = unit_rows(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(x))   # no div-by-zero nan/inf


def test_concat_blocks_accumulates_all_chunk_rows():
    """Chunked derivation must accumulate into ONE shard — the cache is write-once-per-key,
    so writing each chunk separately would keep only the first (the bug this guards)."""
    b1 = FeatureBlock(X=np.array([[1.0, 2.0]], np.float32), columns=["a", "b"],
                      row_ids=np.array(["r0"]))
    b2 = FeatureBlock(X=np.array([[3.0, 4.0], [5.0, 6.0]], np.float32), columns=["a", "b"],
                      row_ids=np.array(["r1", "r2"]))
    out = _concat_blocks([b1, b2])
    assert out.X.shape == (3, 2)                 # all 3 rows, not just the first chunk's 1
    assert list(out.row_ids) == ["r0", "r1", "r2"]
    assert out.columns == ["a", "b"]
    assert out.X.dtype == np.float32


def test_content_inputs_hash_stable_and_sensitive():
    a = content_inputs_hash("qwen", "nn_passrate", 0, "v1")
    assert a == content_inputs_hash("qwen", "nn_passrate", 0, "v1")
    assert a != content_inputs_hash("qwen", "nn_passrate", 1, "v1")
    assert a != content_inputs_hash("llama", "nn_passrate", 0, "v1")
