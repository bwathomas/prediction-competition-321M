"""Local tests for the driver's numpy-pure helpers. The data-loading / passrate / chunk
derivation paths are heavy (pyarrow/scipy/faiss) and are validated on Colab against a real
embedding+label slice; here we lock the import-safe pieces."""
import numpy as np

from aide.features.driver import FAMILY_SLUG, content_inputs_hash, unit_rows


def test_family_slugs_cover_three_families():
    assert set(FAMILY_SLUG) == {"llama", "qwen", "mistral"}
    assert FAMILY_SLUG["qwen"] == "Qwen__Qwen3-Embedding-8B"


def test_unit_rows_normalizes():
    x = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


def test_unit_rows_handles_zero_vector():
    x = unit_rows(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(x))   # no div-by-zero nan/inf


def test_content_inputs_hash_stable_and_sensitive():
    a = content_inputs_hash("qwen", "nn_passrate", 0, "v1")
    assert a == content_inputs_hash("qwen", "nn_passrate", 0, "v1")
    assert a != content_inputs_hash("qwen", "nn_passrate", 1, "v1")
    assert a != content_inputs_hash("llama", "nn_passrate", 0, "v1")
