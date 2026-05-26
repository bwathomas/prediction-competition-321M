"""End-to-end pipeline smoke test for the qwen8b_minimalist notebook.

We don't actually run the notebook (it pulls Qwen3-Embedding-8B and
takes ~60 GB of GPU memory) but we exercise the *exact same wiring*
the notebook does, post-encoding, on a tiny synthetic dataset:

    item embedding (random)
        -> k-means + centroid distances
        -> pool features + z-score
        -> FAISS NN index + (subject, item) passrate table
        -> compute_nn_features() over train + val
        -> ModelConfig with metadata + NN + cluster + pool channels
        -> meta_hybrid_irt_kfactor_gated_mlp build_model + attach_metadata_tables
        -> single forward pass (the trainer would loop, we just need
           the shapes to align)
        -> SubjectResidualTable.from_rows + NNCalibrator.fit_alpha_on_val
        -> export_run() with nn_calibrator_state + nn_calibrator_table_dir
        -> verify runtime_meta.json + cache/nn_residual/ in the bundle

If any of the cell-to-cell handoffs in the notebook is wrong (a kwarg
name mismatch, a missing column, a forward arg renamed) this test
fails. The notebook's ~10 logical cells each map to a phase below.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.clustering import fit_and_assign
from src.data import prepare_metadata_artifacts
from src.embeddings import stack_lookup
from src.export_submission import (
    bundle_training_cache,
    compute_train_counts,
    export_run,
    make_submission_zip,
)
from src.item_features import (
    apply_zscore,
    build_centroid_distance_features,
    build_feature_matrix,
    centroid_distance_feature_names,
    compute_features_for_items,
    fit_zscore_stats,
    merge_pool_and_centroid_features,
)
from src.models import Indexer, LookupDataset, ModelConfig, build_model
from src.nn_calibration import NNCalibrator, SubjectResidualTable
from src.nn_features import (
    NNFeaturesConfig,
    TrainingNNIndex,
    build_passrate_table,
    compute_nn_features,
)


def _synthetic_df(n_items: int = 64, n_subjects: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    benches = ["bench_a", "bench_b", "bench_c"]
    rows = []
    for i in range(n_items):
        rows.append(
            {
                "item_key": f"item_{i:04d}",
                "item_split_key": f"item_{i:04d}",
                "benchmark": benches[i % len(benches)],
                "condition": "default",
                "benchmark_condition_key": f"{benches[i % len(benches)]}::default",
                "item_content": f"item content {i} alpha beta",
            }
        )
    items = pd.DataFrame(rows)

    subjects = pd.DataFrame(
        {
            "subject_key": [f"subj_{j:03d}" for j in range(n_subjects)],
            "subject_content": [f"subject {j}" for j in range(n_subjects)],
        }
    )

    out_rows = []
    for j in range(n_subjects):
        # Each subject sees a random ~70% slice of items.
        for it in items.itertuples(index=False):
            if rng.random() > 0.7:
                continue
            # Item-difficulty + subject-bias logistic.
            logit = float(
                rng.standard_normal() * 0.3
                + 0.05 * (j - n_subjects / 2)
                + 0.05 * (int(it.item_key[-3:]) - n_items / 2)
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            label = int(rng.random() < prob)
            out_rows.append(
                {
                    "item_key": it.item_key,
                    "item_split_key": it.item_split_key,
                    "benchmark": it.benchmark,
                    "condition": it.condition,
                    "benchmark_condition_key": it.benchmark_condition_key,
                    "item_content": it.item_content,
                    "subject_key": subjects["subject_key"].iloc[j],
                    "subject_content": subjects["subject_content"].iloc[j],
                    "label": label,
                }
            )
    return pd.DataFrame(out_rows)


def _split(df: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = df["item_key"].drop_duplicates().tolist()
    rng.shuffle(keys)
    n_val = max(2, int(0.2 * len(keys)))
    val_keys = set(keys[:n_val])
    val_mask = df["item_key"].isin(val_keys)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def _random_emb_lookup(item_keys, dim: int = 64, seed: int = 1):
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for k in item_keys:
        v = rng.standard_normal(dim).astype(np.float32)
        v /= max(1e-6, float(np.linalg.norm(v)))
        out[str(k)] = v
    return out


@pytest.fixture(scope="module")
def fake_metadata(tmp_path_factory) -> tuple[Path, Path]:
    """Write tiny model_info.csv / benchmark_info.csv files matching the
    schema MetadataPreprocessor expects, and patch the loader to use them.
    """
    md = tmp_path_factory.mktemp("md")
    (md / "model_info.csv").write_text(
        "name,organization,family,macro_family,parameters,release_date\n"
        + "\n".join(
            f"subj_{j:03d},org_{j%3},fam_{j%2},macro_{j%2},{(j+1)*1e9},2023-0{1+j%9}-01"
            for j in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    (md / "benchmark_info.csv").write_text(
        "benchmark,topic,benchmark_age\n"
        + "\n".join(f"bench_{c},topic_{c},{i*0.1:.2f}" for i, c in enumerate("abc"))
        + "\n",
        encoding="utf-8",
    )
    return md / "model_info.csv", md / "benchmark_info.csv"


def test_full_minimalist_pipeline(tmp_path: Path, fake_metadata) -> None:
    model_info_path, benchmark_info_path = fake_metadata
    df = _synthetic_df(n_items=48, n_subjects=8, seed=0)
    train_df, val_df = _split(df, seed=0)

    item_df = (
        df[["item_key", "benchmark", "condition", "item_content"]]
        .drop_duplicates(subset=["item_key"])
        .reset_index(drop=True)
    )

    item_emb_lookup = _random_emb_lookup(item_df["item_key"].tolist(), dim=32)
    item_emb_dim = next(iter(item_emb_lookup.values())).shape[0]

    # ------------------------------------------------------------- pool + clusters
    pool_df = compute_features_for_items(item_df, progress=False)

    centroids_path = tmp_path / "centroids.npy"
    assignments_path = tmp_path / "assignments.parquet"
    all_emb = np.stack([item_emb_lookup[k] for k in item_df["item_key"]], axis=0)
    centroids, cluster_assignments = fit_and_assign(
        item_df["item_key"].astype(str).tolist(),
        all_emb,
        k=4,
        seed=0,
        centroids_path=centroids_path,
        assignments_path=assignments_path,
        overwrite=True,
        backend="sklearn",
    )

    top_m = 3
    centroid_df = build_centroid_distance_features(
        item_keys=pool_df["item_key"].astype(str).tolist(),
        item_emb_lookup=item_emb_lookup,
        centroids=centroids,
        top_m=top_m,
    )
    expected_cols = list(centroid_distance_feature_names(top_m))
    assert all(c in centroid_df.columns for c in expected_cols)

    combined_df, combined_cols = merge_pool_and_centroid_features(pool_df, centroid_df)
    train_keys = set(train_df["item_key"].astype(str).tolist())
    train_features = combined_df[combined_df["item_key"].astype(str).isin(train_keys)]
    pool_stats = fit_zscore_stats(train_features, feature_cols=combined_cols)
    pool_features_z = apply_zscore(combined_df, pool_stats)
    pool_stats_path = tmp_path / "pool_features_stats.json"
    pool_stats_path.write_text(json.dumps(pool_stats), encoding="utf-8")

    # ------------------------------------------------------------- indexer + NN
    indexer = Indexer.fit(
        subject_keys=train_df["subject_key"].tolist(),
        bc_keys=train_df["benchmark_condition_key"].tolist(),
    )

    nn_cfg = NNFeaturesConfig.from_dict(
        {"enabled": True, "k": 4, "similarity": "cosine", "prefer_gpu": False}
    )
    nn_dir = tmp_path / "nn_features" / "training"
    train_item_keys = sorted({k for k in train_df["item_key"].astype(str)})
    train_item_keys = [k for k in train_item_keys if k in item_emb_lookup]
    nn_index = TrainingNNIndex.build_from_lookup(
        item_emb_lookup={k: item_emb_lookup[k] for k in train_item_keys},
        out_dir=nn_dir,
        cfg=nn_cfg,
        item_keys=train_item_keys,
    )

    nn_item_index_map = {k: i for i, k in enumerate(train_item_keys)}
    passrate_csr, passrate_mask_csr = build_passrate_table(
        train_df=train_df,
        item_index_map=nn_item_index_map,
        subject_index_map=indexer.subject_to_id,
    )

    def _stack_query(rows_df):
        keys = rows_df["item_key"].astype(str).tolist()
        embs = np.stack([item_emb_lookup[k] for k in keys], axis=0).astype(np.float32)
        sids = np.array(
            [indexer.subject_id(str(s)) for s in rows_df["subject_key"]],
            dtype=np.int64,
        )
        return embs, keys, sids

    train_emb, train_keys_for_nn, train_sid = _stack_query(train_df)
    val_emb, val_keys_for_nn, val_sid = _stack_query(val_df)

    nn_train_mat = compute_nn_features(
        query_embeds=train_emb,
        query_item_keys=train_keys_for_nn,
        subject_ids=train_sid,
        nn_index=nn_index,
        passrate_csr=passrate_csr,
        passrate_mask_csr=passrate_mask_csr,
        cfg=nn_cfg,
        exclude_self=True,
    )
    nn_val_mat = compute_nn_features(
        query_embeds=val_emb,
        query_item_keys=val_keys_for_nn,
        subject_ids=val_sid,
        nn_index=nn_index,
        passrate_csr=passrate_csr,
        passrate_mask_csr=passrate_mask_csr,
        cfg=nn_cfg,
        exclude_self=False,
    )
    from src.nn_features import NN_FEATURE_DIM as _NN_DIM

    assert nn_train_mat.shape == (len(train_df), _NN_DIM)
    assert nn_val_mat.shape == (len(val_df), _NN_DIM)

    # ------------------------------------------------------------- training tensors
    def _pool_matrix(keys):
        return build_feature_matrix(
            [str(k) for k in keys],
            pool_features_z,
            feature_cols=list(combined_cols),
            key_col="item_key",
        )

    def _cluster_vector(keys):
        return np.array(
            [int(cluster_assignments.get(str(k), 0)) for k in keys],
            dtype=np.int64,
        )

    def _build(part, nn_mat):
        s = np.array(
            [indexer.subject_id(k) for k in part["subject_key"]], dtype=np.int64
        )
        bc = np.array(
            [indexer.bc_id(k) for k in part["benchmark_condition_key"]],
            dtype=np.int64,
        )
        return LookupDataset(
            subject_ids=s,
            bc_ids=bc,
            item_emb=stack_lookup(part["item_key"], item_emb_lookup),
            labels=part["label"].astype(float).to_numpy(),
            subject_emb=None,
            pool_feats=_pool_matrix(part["item_key"]),
            cluster_ids=_cluster_vector(part["item_key"]),
            judge_feats=None,
            nn_feats=nn_mat.astype(np.float32),
            sample_weights=None,
        )

    train_ds = _build(train_df, nn_train_mat)
    val_ds = _build(val_df, nn_val_mat)
    assert len(train_ds) == len(train_df)
    assert len(val_ds) == len(val_df)

    # ----------------------------------------- meta artifacts (need real loader)
    import src.data as data_mod
    from src.metadata_features import MetadataSchema

    real_loader = data_mod.load_metadata_frames

    def _fake_loader():
        return (
            pd.read_csv(model_info_path),
            pd.read_csv(benchmark_info_path),
        )

    # Use a reduced schema that matches the ModelConfig below; otherwise
    # the towers built from cfg.meta_subject_categorical etc. will not
    # match the buffer widths produced from the preprocessor's schema.
    fixture_schema = MetadataSchema(
        subject_categorical=("organization", "family"),
        subject_numeric=("log_params",),
        benchmark_categorical=("topic",),
        benchmark_numeric=("benchmark_age",),
        explicit_crosses=("family__topic",),
    )
    data_mod.load_metadata_frames = _fake_loader  # type: ignore[assignment]
    try:
        meta_preprocessor, meta_id_tables = prepare_metadata_artifacts(
            train_df, indexer, schema=fixture_schema,
        )
    finally:
        data_mod.load_metadata_frames = real_loader  # type: ignore[assignment]

    # ------------------------------------------------------------- model
    model_cfg = ModelConfig(
        k=4,
        item_embed_dim=item_emb_dim,
        item_map_hidden_dim=32,
        residual_hidden_dim=32,
        dropout=0.0,
        n_subjects=indexer.n_subjects,
        n_benchmark_conditions=indexer.n_bc,
        use_pool_features=True,
        pool_feature_dim=len(combined_cols),
        use_cluster_features=True,
        n_clusters=int(centroids.shape[0]),
        cluster_embed_dim=4,
        use_judge_features=False,
        judge_feature_dim=0,
        use_nn_features=True,
        nn_feature_dim=int(nn_train_mat.shape[1]),
        use_metadata_features=True,
        meta_subject_categorical=("organization", "family"),
        meta_subject_numeric=("log_params",),
        meta_benchmark_categorical=("topic",),
        meta_benchmark_numeric=("benchmark_age",),
        meta_explicit_crosses=("family__topic",),
    )

    model = build_model("meta_hybrid_irt_kfactor_gated_mlp", model_cfg)
    model.attach_metadata_tables(meta_id_tables)

    def _score(ds: LookupDataset) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(
                subject_idx=ds.subject_ids,
                bc_idx=ds.bc_ids,
                item_emb=ds.item_emb,
                subject_emb=None,
                pool_feats=ds.pool_feats,
                cluster_ids=ds.cluster_ids,
                nn_feats=ds.nn_feats,
            )
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    p_uncal_train = _score(train_ds)
    p_uncal_val = _score(val_ds)
    assert p_uncal_train.shape == (len(train_df),)
    assert np.isfinite(p_uncal_train).all()
    assert ((p_uncal_train >= 0) & (p_uncal_train <= 1)).all()

    # ------------------------------------------------------------- NN calibrator
    key_to_train_row = {k: i for i, k in enumerate(train_item_keys)}
    train_rows = np.array(
        [key_to_train_row.get(str(k), -1) for k in train_df["item_key"]],
        dtype=np.int64,
    )
    ok = train_rows >= 0
    residual_table = SubjectResidualTable.from_rows(
        subject_ids=train_sid[ok],
        training_item_rows=train_rows[ok],
        labels=train_df["label"].astype(float).to_numpy()[ok],
        uncal_probs=p_uncal_train[ok],
        n_subjects=indexer.n_subjects,
        n_training_items=len(train_item_keys),
    )

    val_neighbor_rows, val_neighbor_sims = nn_index.nearest(
        val_emb, k=4, exclude_self=False
    )
    calibrator = NNCalibrator.fit_alpha_on_val(
        residual_table=residual_table,
        val_subject_ids=val_sid,
        val_neighbor_rows=val_neighbor_rows,
        val_neighbor_sims=val_neighbor_sims,
        val_uncal_probs=p_uncal_val,
        val_labels=val_df["label"].astype(float).to_numpy(),
        k=4,
        similarity="cosine",
    )
    state = calibrator.to_dict()
    assert {"alpha", "k", "similarity"} <= set(state)

    residual_dir = tmp_path / "nn_calibration"
    residual_table.save(residual_dir)
    assert (residual_dir / "passrate_indptr.npy").exists()

    # ------------------------------------------------------------- export
    # Save a synthetic checkpoint that matches export_run's expectations.
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "qwen8b_minimalist.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "indexer": indexer.to_dict(),
            "encoder_cfg": {"model_id": "stub-encoder"},
        },
        ckpt_path,
    )

    # Build a TrainResult-shaped object with just enough fields for export_run.
    from src.train import TrainResult

    train_result = TrainResult(
        run_id="qwen8b_minimalist",
        model_name="meta_hybrid_irt_kfactor_gated_mlp",
        seed=0,
        k=int(model_cfg.k),
        epoch_best=1,
        best_val_log_loss=0.5,
        best_val_brier=0.25,
        best_val_auc=0.5,
        history=[],
        checkpoint_path=str(ckpt_path),
        metadata_path="",
        n_train=len(train_ds),
        n_val=len(val_ds),
    )

    # Build the training cache (writes the items.parquet first, then the
    # int8 + faiss + nn passrate bundle).
    items_parquet = tmp_path / "items.parquet"
    pd.DataFrame(
        {
            "item_key": list(item_emb_lookup.keys()),
            "benchmark": [item_df.set_index("item_key").loc[k, "benchmark"] for k in item_emb_lookup],
            "condition": ["default"] * len(item_emb_lookup),
            "embedding": [v.astype(np.float16) for v in item_emb_lookup.values()],
        }
    ).to_parquet(items_parquet, index=False)

    encoder_cfg = {
        "model_id": "stub-encoder",
        "max_length": 64,
        "use_contextual_item_text": True,
        "qwen3_instruction": "",
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "mean",
        "bf16": False,
    }
    submission_cache_cfg = {
        "enabled": True,
        "quantize": "fp16",
        "pca_dim": None,
        "include_faiss_index": False,
        "passrate_format": "sparse",
        "max_bundle_size_mb": 200,
        "runtime_k": int(nn_cfg.k),
    }
    training_cache_dir = tmp_path / "training_cache"
    bundle_training_cache(
        items_parquet_path=items_parquet,
        out_dir=training_cache_dir,
        submission_cache_cfg=submission_cache_cfg,
        encoder_cfg=encoder_cfg,
        items_meta_df=item_df,
        cluster_assignments=cluster_assignments,
        n_clusters=int(centroids.shape[0]),
        train_df=train_df,
        nn_features_cfg=nn_cfg.to_dict(),
        subject_to_id=indexer.subject_to_id,
    )

    submission_dir = tmp_path / "submission"
    sub = export_run(
        result=train_result,
        encoder_cfg=encoder_cfg,
        submission_dir=submission_dir,
        include_labeling=False,
        pool_stats_path=pool_stats_path,
        cluster_centroids_path=centroids_path,
        pool_feature_names=list(combined_cols),
        training_cache_dir=training_cache_dir,
        judge_cfg=None,
        nn_features_cfg=nn_cfg.to_dict(),
        ship_training_cache=True,
        ship_requirements_txt=False,
        meta_preprocessor=meta_preprocessor,
        nn_calibrator_state=state,
        nn_calibrator_table_dir=residual_dir,
    )

    # 1. The runtime bundle has model.py, runtime_meta.json, the centroids,
    # the meta preprocessor JSON, and the cluster centroid file.
    assert (sub / "model.py").exists()
    runtime_meta = json.loads((sub / "artifacts" / "runtime_meta.json").read_text(encoding="utf-8"))
    assert "nn_calibrator" in runtime_meta
    assert runtime_meta["nn_calibrator"]["k"] == int(state["k"])
    assert runtime_meta["nn_calibrator"]["similarity"] == state["similarity"]
    assert (sub / "artifacts" / "meta_preprocessor.json").exists()
    assert (sub / "artifacts" / "cluster_centroids.npy").exists()

    # 2. NN residual table copied into cache/nn_residual/ when alpha != 0.
    if float(state.get("alpha", 0.0)) != 0.0:
        assert (sub / "cache" / "nn_residual" / "passrate_indptr.npy").exists()
        assert (sub / "cache" / "nn_residual" / "uncal_prob_data.npy").exists()
        assert (sub / "cache" / "nn_residual" / "meta.json").exists()

    # 3. Zip step builds a non-empty file under the cap.
    zip_out = make_submission_zip(
        sub, zip_path=tmp_path / "submission.zip", max_zip_size_mb=200
    )
    assert zip_out.exists()
    assert zip_out.stat().st_size > 1024
