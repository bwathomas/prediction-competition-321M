"""Runtime-template parity tests for centroid distances + NN calibration.

These tests don't actually run the full submission pipeline (that
requires the encoder + cache stack). Instead they extract specific
helper functions from the rendered ``_RUNTIME_MODEL_PY`` string,
re-execute them in a controlled namespace with synthetic inputs, and
compare them to the source-of-truth implementations in :mod:`src`.

The point is to lock down the contract that the runtime template
implements the *same* math as training -- a previous bug that
silently shipped zero-filled features came from drift between these
two paths.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from src.clustering import compute_top_m_distances
from src.export_submission import _RUNTIME_MODEL_PY
from src.nn_calibration import NNCalibrator, NNCalibratorState, SubjectResidualTable


# ---------------------------------------------------------------------------
# Helpers: extract a top-level def / class block from the runtime template.
# ---------------------------------------------------------------------------


def _extract_top_level_block(name: str, source: str) -> str:
    """Return the source of the top-level function or class ``name``."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            start = node.lineno - 1  # 1-based -> 0-based
            end = node.end_lineno  # already 1-based exclusive when sliced
            lines = source.splitlines()
            return "\n".join(lines[start:end])
    raise AssertionError(f"top-level {name!r} not found in runtime template")


def _make_template_namespace() -> dict:
    """Build a namespace with the minimal globals the extracted blocks need."""
    return {
        "__name__": "_runtime_test",
        "np": np,
        "math": __import__("math"),
        "json": __import__("json"),
        "Path": Path,
        # We don't want the runtime to actually try to load anything.
        "_CLUSTER_CENTROIDS": None,
        "_CENTROID_NORM2_CACHE": None,
        "POOL_FEATURE_NAMES": (),
        "_POOL_STATS": {},
        "_MODEL_CFG": {},
        "EPS": 1e-6,
        "DEFAULT_PROB": 0.5,
        # _NNCalibrator's load_table uses this; we'll route via a local LOG.
        "LOG": __import__("logging").getLogger("runtime_test"),
        "compute_pool_features_runtime": lambda text: {},
    }


# ---------------------------------------------------------------------------
# Centroid distance parity: training vs runtime
# ---------------------------------------------------------------------------


def _compile_centroid_helpers() -> dict:
    ns = _make_template_namespace()
    blocks = [
        # _centroid_norm2 references _CLUSTER_CENTROIDS + the cache.
        # _compute_centroid_distance_vec uses _centroid_norm2.
        # Both are top-level defs in the runtime template.
    ]
    src_norm2 = _extract_top_level_block("_centroid_norm2", _RUNTIME_MODEL_PY)
    src_dist = _extract_top_level_block(
        "_compute_centroid_distance_vec", _RUNTIME_MODEL_PY
    )
    exec(compile(src_norm2, "<rt-_centroid_norm2>", "exec"), ns)
    exec(compile(src_dist, "<rt-_compute_centroid_distance_vec>", "exec"), ns)
    return ns


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_runtime_centroid_distance_matches_training(seed):
    """Runtime path emits the same per-item top-m distance vector as
    :func:`src.clustering.compute_top_m_distances`."""
    rng = np.random.default_rng(seed)
    K, D, M = 16, 32, 5
    centroids = rng.standard_normal((K, D)).astype(np.float32)
    queries = rng.standard_normal((40, D)).astype(np.float32)

    ns = _compile_centroid_helpers()
    ns["_CLUSTER_CENTROIDS"] = centroids
    # Runtime computes a single query at a time.
    runtime_out = np.stack(
        [ns["_compute_centroid_distance_vec"](q, M) for q in queries], axis=0
    )

    # Training path computes the whole batch.
    _, train_out = compute_top_m_distances(centroids, queries, top_m=M)
    np.testing.assert_allclose(runtime_out, train_out, rtol=1e-3, atol=1e-3)


def test_runtime_centroid_distance_zero_when_centroids_missing():
    ns = _compile_centroid_helpers()
    # _CLUSTER_CENTROIDS stays None; runtime should return zeros.
    out = ns["_compute_centroid_distance_vec"](
        np.zeros(8, dtype=np.float32), 4
    )
    assert out.shape == (4,)
    assert (out == 0).all()


def test_runtime_centroid_distance_pads_when_top_m_exceeds_k():
    """When top_m > k, the runtime pads with the last available value
    instead of crashing."""
    rng = np.random.default_rng(0)
    K, D = 3, 8
    centroids = rng.standard_normal((K, D)).astype(np.float32)
    q = rng.standard_normal(D).astype(np.float32)
    ns = _compile_centroid_helpers()
    ns["_CLUSTER_CENTROIDS"] = centroids
    out = ns["_compute_centroid_distance_vec"](q, 5)
    # First K values must be the sorted top-K distances; padding fills
    # positions K..top_m-1 with the value at position K-1.
    assert out.shape == (5,)
    np.testing.assert_allclose(out[3], out[2])
    np.testing.assert_allclose(out[4], out[2])


# ---------------------------------------------------------------------------
# NN calibrator parity: training vs runtime
# ---------------------------------------------------------------------------


def _compile_runtime_nncalibrator() -> dict:
    """Compile just the runtime ``_NNCalibrator`` class into a sandbox.

    The runtime class duplicates the math from
    :class:`src.nn_calibration.NNCalibrator` so we can run it without
    pulling in the encoder / training-cache machinery.
    """
    ns = _make_template_namespace()
    src = _extract_top_level_block("_NNCalibrator", _RUNTIME_MODEL_PY)
    # Avoid pulling math from the parent module twice; the namespace
    # already has ``math``.
    exec(compile(src, "<rt-_NNCalibrator>", "exec"), ns)
    return ns


def _runtime_apply(ns, cal, table: SubjectResidualTable, subj: int,
                   nbrs: np.ndarray, sims: np.ndarray, p: float) -> float:
    """Bridge the in-memory training-side table into the runtime
    calibrator instance and call ``apply``."""
    cal.passrate_indptr = table.passrate_indptr
    cal.passrate_indices = table.passrate_indices
    cal.passrate_data = table.passrate_data
    cal.uncal_prob_data = table.uncal_prob_data
    cal.n_subjects = table.n_subjects
    cal.n_training_items = table.n_training_items
    return float(cal.apply(p, subj, nbrs, sims))


def test_runtime_nncal_alpha_zero_is_identity():
    ns = _compile_runtime_nncalibrator()
    cal = ns["_NNCalibrator"]({"alpha": 0.0})
    out = cal.apply(0.42, 0, np.array([0, 1]), np.array([0.9, 0.8]))
    assert out == pytest.approx(0.42)


def test_runtime_nncal_matches_training_calibrator_apply():
    rng = np.random.default_rng(42)
    N_subj, N_train = 6, 50
    bias = rng.uniform(-0.5, 0.5, size=N_subj).astype(np.float32)

    # Build the residual table with the training-side helper.
    s_ids = np.repeat(np.arange(N_subj), N_train)
    t_rows = np.tile(np.arange(N_train), N_subj)
    p_uncal = rng.uniform(0.05, 0.95, size=s_ids.size).astype(np.float32)
    logit_true = np.log(p_uncal / (1 - p_uncal)) + bias[s_ids]
    labels = (1.0 / (1 + np.exp(-logit_true)) > rng.uniform(size=s_ids.size)).astype(
        np.float32
    )
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids, training_item_rows=t_rows,
        labels=labels, uncal_probs=p_uncal,
        n_subjects=N_subj, n_training_items=N_train,
    )

    state = NNCalibratorState(alpha=0.5, k=8, similarity="cosine")

    # Training-side calibrator with the same state.
    train_cal = NNCalibrator(state)

    # Runtime-side calibrator with the same state, hot-loaded with the
    # same table arrays in-memory.
    ns = _compile_runtime_nncalibrator()
    rt_cal = ns["_NNCalibrator"](state.to_dict())

    # Sample queries.
    K = 8
    for _ in range(20):
        s = int(rng.integers(0, N_subj))
        nbrs = rng.integers(0, N_train, size=K).astype(np.int64)
        sims = rng.uniform(0.5, 1.0, size=K).astype(np.float32)
        p = float(rng.uniform(0.05, 0.95))

        train_p = float(
            train_cal.apply(
                residual_table=table,
                subject_ids=np.array([s], dtype=np.int64),
                neighbor_rows=nbrs.reshape(1, -1),
                neighbor_sims=sims.reshape(1, -1),
                p_uncal=np.array([p], dtype=np.float32),
            )[0]
        )
        rt_p = _runtime_apply(ns, rt_cal, table, s, nbrs, sims, p)
        assert abs(train_p - rt_p) < 1e-5, (train_p, rt_p)


def test_runtime_nncal_load_table_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(1)
    table = SubjectResidualTable.from_rows(
        subject_ids=rng.integers(0, 4, size=30),
        training_item_rows=rng.integers(0, 20, size=30),
        labels=rng.uniform(0, 1, size=30).astype(np.float32),
        uncal_probs=rng.uniform(0, 1, size=30).astype(np.float32),
        n_subjects=4, n_training_items=20,
    )
    table.save(tmp_path)

    ns = _compile_runtime_nncalibrator()
    cal = ns["_NNCalibrator"]({"alpha": 0.5, "k": 4})
    cal.load_table(tmp_path)
    assert cal.has_table
    assert cal.n_subjects == 4
    assert cal.n_training_items == 20

    # Empty / missing dir -> table stays unloaded, has_table False.
    cal2 = ns["_NNCalibrator"]({"alpha": 0.5})
    cal2.load_table(tmp_path / "does_not_exist")
    assert not cal2.has_table
