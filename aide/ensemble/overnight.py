"""Self-contained overnight qwen optimization (launched via run_bg on Colab).

Three resumable stages, each idempotent so a Colab disconnect resumes cheaply:
  1. complete the feature cache — ``derive_tabular_global`` writes the metadata-groupby +
     subject-encoding groups (cache-skips if already present);
  2. assemble the full per-row training ``Dataset`` from the cache, cached to a Drive ``.npy``;
  3. ``run_search`` — checkpointed ensemble search to minimize OOF item-cold-start NLL.

Every stage finite-checks and aborts loudly rather than silently producing garbage overnight.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

GEOMETRY_GROUPS = ["nn_geometry", "cluster_geometry", "centroid_distance", "item_cluster"]
LABEL_GROUPS = ["nn_label_derivatives", "counts_subject", "cluster_passrate", "cluster_subject",
                "groupby_subject_metadata", "groupby_benchmark_metadata", "mean_encoded_subject"]


def _load_labels(drive_root):
    import glob
    import pandas as pd
    db = glob.glob(f"{drive_root}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    df = pd.read_parquet(db, columns=["subject_key", "item_key", "label",
                                      "subject_content", "benchmark"])
    for c in ("subject_key", "item_key", "benchmark"):
        df[c] = df[c].astype(str)
    return df


def _assemble_or_load(store, manifest, labels_df, cache_path, progress):
    from aide.ensemble.assemble import assemble_training_matrix
    from aide.harness.eval import Dataset
    # cache as a plain .npz of arrays (no pickle / allow_pickle) — first-party data, but we
    # keep the load path off the pickle code-exec surface, matching the feature cache.
    p = Path(cache_path)
    if p.exists():
        d = np.load(p)
        progress(f"loaded cached Dataset {d['X'].shape}")
        return Dataset(X=np.asarray(d["X"], np.float32), feature_columns=[str(c) for c in d["cols"]],
                       y=d["y"], item_keys=d["items"].astype(str),
                       subjects=d["subjects"].astype(str), benchmarks=d["bench"].astype(str))
    ds = assemble_training_matrix(
        store, manifest, item_keys=labels_df["item_key"].to_numpy(),
        subject_keys=labels_df["subject_key"].to_numpy(), labels=labels_df["label"].to_numpy(),
        benchmarks=labels_df["benchmark"].to_numpy(), geometry_groups=GEOMETRY_GROUPS,
        label_groups=LABEL_GROUPS, progress=progress)
    if not np.isfinite(ds.X).all():
        raise ValueError("assembled feature matrix has non-finite entries — aborting")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        np.savez(fh, X=ds.X, cols=np.asarray(ds.feature_columns, dtype=str), y=ds.y,
                 items=np.asarray(ds.item_keys).astype(str),
                 subjects=np.asarray(ds.subjects).astype(str),
                 bench=np.asarray(ds.benchmarks).astype(str))
    progress(f"assembled + cached Dataset {ds.X.shape}")
    return ds


def run_overnight(*, drive_root, repo_root, family="qwen", code_version="v2",
                  time_budget_s=28800, subsample=200_000, progress=None):
    from aide.features.cache import FeatureCache
    from aide.features.metadata import derive_tabular_global, load_metadata
    from aide.features.store import FoldFeatureStore
    from aide.ensemble.optimize import run_search
    from aide.hygiene.manifest import build_manifest
    if progress is None:
        progress = lambda *a, **k: None  # noqa: E731
    work = f"{drive_root}/optimize/{family}"
    Path(work).mkdir(parents=True, exist_ok=True)
    store = FoldFeatureStore(FeatureCache(f"{drive_root}/features", code_version=code_version),
                             embedding_family=family, seed=0, n_folds=3)

    # restrict labels to items that actually have features (the derivation's emb_set)
    geo0 = store.cache.read_shard(store._key(GEOMETRY_GROUPS[0], "all"))
    feat_items = set(str(k) for k in geo0.row_ids)
    labels = _load_labels(drive_root)
    labels = labels[labels["item_key"].isin(feat_items)].reset_index(drop=True)
    manifest = build_manifest(sorted(feat_items), n_folds=3, seed=0)
    progress(f"labels={len(labels)} rows over {len(feat_items)} featured items")

    # Stage 1: tabular groups (idempotent)
    progress("stage1: tabular groups")
    mi, bi = load_metadata(repo_root)
    cov = derive_tabular_global(store=store, labels_df=labels, manifest=manifest,
                                model_info=mi, benchmark_info=bi, family=family,
                                code_version=code_version, progress=progress)
    progress(f"stage1 done; subject-join coverage={cov['subject_join_coverage']:.3f}")

    # Stage 2: assemble
    progress("stage2: assemble")
    ds = _assemble_or_load(store, manifest, labels, f"{work}/dataset_{code_version}.npz", progress)

    # Stage 3: search
    progress("stage3: search")
    res = run_search(ds, manifest, checkpoint_path=f"{work}/search_ckpt.json",
                     subsample=subsample, time_budget_s=time_budget_s, now_fn=time.time,
                     progress=progress)
    Path(f"{work}/result.json").write_text(json.dumps(res, indent=2))
    return res
