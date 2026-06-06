"""Per-family NN-infra prep + live 23-dim AIDE-feature computation (the USE_AIDE_FEATS engine).

SHIP_PLAN_3WAY §CORRECTION (2026-06-06): "The SUBMISSION computes the 23-dim NN
features LIVE at predict-time from a shipped train-neighbor index (exactly what the
sent submodels already do, 8-dim). PREP per-family neighbor infra ONCE: load family
train item embeddings -> build TrainingNNIndex over train items -> build passrate CSR
(build_passrate_table) + build_conditional_passrate_context -> k-means centroids. Save
to DR/ship/nn_infra/<fam>/. Then COMPUTE 23-dim AIDE feats for train folds{0,1}, fold2
(honest val), AND holdout (135650) -- all cheap once the index exists."

This module is the bridge that makes ``USE_AIDE_FEATS=True`` viable end-to-end:

  * ``prep_nn_infra(fam, ...)``   -- build (or load) the per-family neighbor infra and
    persist it to DR/ship/nn_infra/<fam>/ in the EXACT on-disk layout the runtime
    ``nn23_runtime.ConditionalContextRuntime.maybe_load`` + ``TrainingNNIndex.load``
    expect. Idempotent: re-running loads the cached infra.
  * ``load_nn_infra(fam, ...)``   -- reload a prepped infra bundle into a ``NNInfra``.
  * ``compute_features(infra, rows_item, rows_subj, ...)`` -- the 23-dim AIDE NN feature
    matrix for ANY row set (train-fold / fold2 / holdout), computed live via
    ``src.nn_features.compute_nn_features_streaming`` (the BATCH mirror of the runtime's
    per-item ``nn23_runtime.compute_nn_features_23``; both bottom out on the SAME
    ``_aggregate_nn_features`` numerics, so train and submission never drift).
  * ``compute_dense_block(...)`` -- the full dense_X = [23-dim AIDE NN | centroid-distance
    block | metadata numeric+categorical-onehot] for a row set, leak-free (every table
    fit on the TRAIN-index rows only).

=================================================================================
WHY THE HOLDOUT IS NOW SCORABLE (the correction to train_aide_mlp's old VERDICT)
=================================================================================
The old fallback verdict said "holdout AIDE shards are not on Drive". That conflated
two things: pre-DERIVED OOF shards (which truly only cover the nf3 train structure) vs.
COMPUTING the 23-dim vector live from a neighbor index. The 23-dim NN feature is a pure
function of:  (query item embedding) x (TrainingNNIndex over train items) x (subject's
passrate CSR row) x (conditional context).  None of those need a precomputed holdout
shard -- the holdout item's embedding is on Drive (DR/embeddings/<slug>/items.parquet),
and the index/CSR/context are built ONCE from train. So we compute the holdout's 23-dim
feats with the SAME call we use for train, and the shipped submodel recomputes them
identically at predict time from the bundled infra + nn23_runtime.py aggregator.

=================================================================================
NEIGHBOR-LEAKAGE DISCIPLINE (critical -- the honest-val contract)
=================================================================================
The neighbor index + passrate CSR + conditional context define the "memory" a row's
features are read from. For the features to be HONEST we must respect item-disjointness:

  * For the SHIPPED model and for HOLDOUT features: the index/CSR/context are built over
    ALL train items (folds {0,1,2}). Holdout items are a disjoint universe (never in the
    train index) so there is no self-leak; the runtime ships exactly this all-train infra.
  * For the honest-VAL run (predict fold2 from a folds{0,1}-trained model): fold2 items
    must NOT be in the index/CSR/context, else a fold2 row would retrieve ITSELF (label
    leak). So ``prep_nn_infra(..., index_fold_subset=(0,1))`` builds a SECOND, folds{0,1}-
    only infra; fold2 features (and the folds{0,1} training features) are computed against
    THAT. ``exclude_self`` in ``nearest()`` additionally drops a query item that happens to
    be in the index (defensive; for fold2-vs-f01 the item simply isn't there).

So a full family prep produces TWO infra bundles:
    DR/ship/nn_infra/<fam>/all/     (index over folds {0,1,2}) -> holdout + ship-train feats
    DR/ship/nn_infra/<fam>/f01/     (index over folds {0,1})    -> fold2 + f01-train feats
``train_aide_mlp`` uses 'f01' for the honest-val run feature matrices and 'all' for the
shipped-model + holdout feature matrices.

=================================================================================
ON-DISK LAYOUT  (DR/ship/nn_infra/<fam>/<subset>/)
=================================================================================
    training_index.faiss / training_index_embeddings.npy / training_index_keys.json /
        training_index_meta.json              (TrainingNNIndex.save artifacts)
    subject_passrate.npz / subject_passrate_mask.npz            (legacy 8-dim CSR; cells 0..7)
    subject_index.json                        (subject_key -> CSR row id; UNK conventions)
    item_index.json                           (item_key   -> CSR col / index row id)
    centroids.npy                             (k-means centroids over index items)
    conditional_meta.json + the conditional bag (ConditionalPassrateContext.save layout,
        consumed by nn23_runtime.ConditionalContextRuntime.maybe_load)
    benchmark_to_id.json                      (raw benchmark string -> item_benchmark_id;
        lets runtime cell 19 resolve the query benchmark)
    infra_meta.json                           (fam, subset, k, sims, cardinalities, dims)

The qwen family already has a compatible bundle at DR/artifacts/nn_features (+ cluster
_centroids.npy / item_clusters.parquet). ``prep_nn_infra('qwen', reuse_qwen_artifacts=
True)`` copies/loads that instead of rebuilding. nemotron + lgai are built fresh in their
own embedding space.

=================================================================================
RUNTIME PARITY CONTRACT (how the shipped submission recomputes the SAME dense_X)
=================================================================================
1. The submodel bundle ships DR/ship/nn_infra/<fam>/all/  (the index keys+embeddings,
   subject CSR, conditional bag, centroids, benchmark_to_id, infra_meta).
2. At predict time the submodel builds its ``_TrainingItemCache`` from those files (the
   index = training_index.faiss, nn_passrate = subject_passrate.npz) and calls
   ``nn23_runtime.compute_nn_features_23(cache, item_emb, subject_id, cond_ctx=...,
   query_benchmark_id=..., query_cluster_id=..., k=NN_RUNTIME_K)`` PER ROW -> 23-dim.
3. The centroid/metadata dense columns are recomputed at runtime from centroids.npy +
   the metadata CSVs (also shipped) with the SAME ``compute_dense_block`` ordering baked
   into ``meta.json`` (``dense_feature_names`` from this module). The concatenation order
   [nn23 | centroid | metadata] is LOCKED here and asserted at apply time.
The only train/runtime difference is batched-vs-per-row execution; the numerics are the
shared ``_aggregate_nn_features`` + a deterministic centroid argmin + a leak-free CSV join.

=================================================================================
ASSUMPTIONS (verify before trusting outputs) -- mirror train_aide_mlp A1..A7
  P1. NN_RUNTIME_K must match k used here (default 16; src.nn_features default). Recorded
      in infra_meta.json; the submodel reads it.
  P2. Subject CSR row ids: built over the FULL train+holdout subject vocabulary (A4 item
      cold-start) so a holdout subject resolves to its real passrate row, not UNK. The
      passrate VALUES are train-only (CSR built from train rows), so this is leak-free.
  P3. Conditional context cardinalities (families / macro / orgs / clusters) come from the
      metadata join over the TRAIN-index rows; auto-grown by build_conditional_passrate_
      context if the declared vocab is short. n_clusters = k-means K (default 64).
  P4. benchmark string<->id: item_benchmark_id is assigned by first-seen order over the
      index items' benchmarks; benchmark_to_id.json inverts it so runtime resolves a
      query's benchmark id. Unknown -> -1 (cell 19 redacts).
=================================================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# default neighbor K -- MUST equal the submodel's NN_RUNTIME_K (P1).
DEFAULT_K = 16
DEFAULT_KMEANS_K = 64
DEFAULT_SIM = "cosine"

FAM_ALIAS = {
    "qwen": "qwen",
    "nemotron": "llama",
    "llama": "llama",
    "lgai": "mistral",
    "mistral": "mistral",
}


# ----------------------------------------------------------------------------------
# NNInfra: a prepped (or loaded) per-family neighbor bundle for ONE index subset.
# ----------------------------------------------------------------------------------
@dataclass
class NNInfra:
    fam: str
    subset: str                       # 'all' | 'f01'
    root: str                         # DR/ship/nn_infra/<fam>/<subset>
    k: int
    similarity: str

    # neighbor index over the index-subset's TRAIN items
    nn_index: object = None           # src.nn_features.TrainingNNIndex
    cfg: object = None                # src.nn_features.NNFeaturesConfig

    # legacy subject x item passrate (cells 0..7)
    passrate_csr: object = None
    passrate_mask_csr: object = None

    # id maps
    item_index_map: dict = field(default_factory=dict)   # item_key -> col/index id
    subject_index_map: dict = field(default_factory=dict)  # subject_key -> CSR row id
    index_item_keys: list = field(default_factory=list)    # ordered index items

    # conditional context (cells 15..22) + benchmark string<->id
    cond_ctx: object = None           # src.nn_features.ConditionalPassrateContext
    benchmark_to_id: dict = field(default_factory=dict)

    # geometry
    centroids: np.ndarray | None = None  # [K, D] k-means centroids

    d_emb: int = 0


# ----------------------------------------------------------------------------------
# small IO helpers
# ----------------------------------------------------------------------------------
def _drive_root() -> str:
    return os.environ.get(
        "SHIP_DRIVE_ROOT",
        "/content/drive/MyDrive/prediction-competition-321M",
    )


def _repo_root() -> str:
    return os.environ.get("SHIP_REPO_ROOT", "/content/pc321")


def _emb_dir(driver_fam: str) -> str:
    from aide.features.driver import FAMILY_SLUG
    return f"{_drive_root()}/embeddings/{FAMILY_SLUG[driver_fam]}"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------
# metadata id resolution over a set of items/subjects (leak-free: maps only)
# ----------------------------------------------------------------------------------
def _resolve_metadata_ids(
    *,
    repo_root: str,
    drive_root: str,
    index_item_keys: list[str],
    subject_keys: list[str],
    prepared_parquet: str,
):
    """Per-(item/subject) metadata ids needed by build_conditional_passrate_context.

    Returns a dict with: subject_to_{family,macro_family,organization}_id (over the
    subject vocab), item_benchmark_id / item_benchmark_age (over the index items),
    benchmark_to_id (string->id), and trait cardinalities. All ids use 0 = MISSING.
    Built from the prepared parquet + data/metadata CSVs via aide.features.metadata
    (the same fragile-but-leak-free join the rest of the pipeline uses).
    """
    import pandas as pd

    from aide.features.metadata import (
        extract_subject_name,
        row_subject_meta,
    )

    # load the two metadata CSVs (repo-vendored; Drive copies also fine)
    def _load_csv(name):
        for p in (f"{drive_root}/data/metadata/{name}", f"{repo_root}/data/metadata/{name}"):
            if Path(p).exists():
                return pd.read_csv(p)
        return None

    model_info = _load_csv("model_info.csv")
    benchmark_info = _load_csv("benchmark_info.csv")

    prep = pd.read_parquet(
        prepared_parquet,
        columns=["subject_key", "item_key", "benchmark", "subject_content"],
    )
    prep["subject_key"] = prep["subject_key"].astype(str)
    prep["item_key"] = prep["item_key"].astype(str)
    subj_content = dict(
        zip(prep.drop_duplicates("subject_key")["subject_key"],
            prep.drop_duplicates("subject_key")["subject_content"])
    )
    bench_by_item = dict(
        zip(prep.drop_duplicates("item_key")["item_key"],
            prep.drop_duplicates("item_key")["benchmark"])
    )

    # ----- subject trait ids (0 = MISSING) -----
    names = [extract_subject_name(subj_content.get(s, "")) for s in subject_keys]
    smeta, subj_cov = (
        row_subject_meta(names, model_info)
        if model_info is not None
        else ({"organization": np.array(["UNK"] * len(names)),
               "family": np.array(["UNK"] * len(names)),
               "macro_family": np.array(["UNK"] * len(names))}, 0.0)
    )

    def _ids_from(values):
        vocab = {"__MISSING__": 0}
        out = np.zeros(len(values), dtype=np.int32)
        for i, v in enumerate(values):
            v = str(v)
            if v in ("UNK", "nan", ""):
                out[i] = 0
                continue
            out[i] = vocab.setdefault(v, len(vocab))
        return out, len(vocab)

    fam_id, n_fam = _ids_from(smeta["family"])
    macro_id, n_macro = _ids_from(smeta["macro_family"])
    org_id, n_org = _ids_from(smeta["organization"])

    # ----- per-index-item benchmark id + age -----
    benches = [str(bench_by_item.get(k, "UNK")) for k in index_item_keys]
    # RAW age (in days) per benchmark string, straight off the CSV. We do NOT route
    # this through row_benchmark_meta: that helper returns the COARSE age_bin
    # (floor(age/180)) as a string, which would store a bin index in
    # item_benchmark_age and corrupt the freshness feature (cell 20)'s scale.
    raw_age_by_bench: dict[str, float] = {}
    if benchmark_info is not None:
        bi = benchmark_info.copy()
        bi.columns = [str(c) for c in bi.columns]
        if "benchmark" not in bi.columns:
            for cand in ("benchmark_id", "name"):
                if cand in bi.columns:
                    bi = bi.rename(columns={cand: "benchmark"})
                    break
        if "benchmark" in bi.columns and "age" in bi.columns:
            for _, r in bi.iterrows():
                a = float(pd.to_numeric(r.get("age"), errors="coerce"))
                raw_age_by_bench[str(r["benchmark"])] = a if np.isfinite(a) else np.nan
    # benchmark string -> id (0-based dense), -1 for UNK
    bench_to_id: dict[str, int] = {}
    item_bench_id = np.full(len(index_item_keys), -1, dtype=np.int32)
    for i, b in enumerate(benches):
        if b in ("UNK", "nan", ""):
            continue
        item_bench_id[i] = bench_to_id.setdefault(b, len(bench_to_id))
    age = np.array(
        [raw_age_by_bench.get(str(b), np.nan) for b in benches],
        dtype=np.float32,
    )

    return {
        "subject_to_family_id": fam_id,
        "subject_to_macro_family_id": macro_id,
        "subject_to_organization_id": org_id,
        "n_families": int(n_fam),
        "n_macro_families": int(n_macro),
        "n_organizations": int(n_org),
        "item_benchmark_id": item_bench_id,
        "item_benchmark_age": age,
        "benchmark_to_id": bench_to_id,
        "subject_coverage": float(subj_cov),
    }


# ----------------------------------------------------------------------------------
# prep: build (or load) one index-subset infra bundle and persist it.
# ----------------------------------------------------------------------------------
def prep_nn_infra(
    fam: str,
    *,
    index_item_keys: list[str],
    index_subject_keys: list[str],
    all_subject_keys: list[str],
    train_df,                          # pandas: subject_key,item_key,label over INDEX rows
    subset: str = "all",
    k: int = DEFAULT_K,
    kmeans_k: int = DEFAULT_KMEANS_K,
    similarity: str = DEFAULT_SIM,
    reuse_qwen_artifacts: bool = False,
    overwrite: bool = False,
    prepared_parquet: str | None = None,
) -> NNInfra:
    """Build/persist the per-family neighbor infra for ONE index subset.

    Args:
      index_item_keys: the TRAIN items that form the neighbor index (folds{0,1,2} for
        subset='all'; folds{0,1} for subset='f01'). MUST be a subset of the family's
        embedding cache item_keys.
      index_subject_keys: subjects to give CSR rows (>= all subjects appearing in train).
      all_subject_keys: full train+holdout subject vocab (the CSR/context get rows for
        all of them so holdout subjects resolve to a real row -- P2).
      train_df: the (subject_key,item_key,label) rows whose ITEM is in index_item_keys
        (this is what populates passrate + conditional context; leak-free for the subset).
      subset: 'all' or 'f01' (controls the output dir + leakage scope; see module docstring).
    Returns a ready-to-use NNInfra. Idempotent unless overwrite=True.
    """
    import sys
    if _repo_root() not in sys.path:
        sys.path.insert(0, _repo_root())

    from src.clustering import assign_clusters, fit_kmeans
    from src.nn_features import (
        NNFeaturesConfig,
        TrainingNNIndex,
        build_conditional_passrate_context,
        build_passrate_table,
    )

    driver_fam = FAM_ALIAS[fam]
    root = Path(f"{_drive_root()}/ship/nn_infra/{fam}/{subset}")
    cfg = NNFeaturesConfig(k=int(k), similarity=str(similarity),
                           cache_dir=str(root), prefer_gpu=True)

    index_item_keys = [str(x) for x in index_item_keys]
    item_index_map = {k_: i for i, k_ in enumerate(index_item_keys)}
    # CSR/context subject rows over the FULL vocab (P2). UNK is the last row by convention
    # used elsewhere; here we just give every train+holdout subject a stable id.
    subj_vocab = {}
    for s in all_subject_keys:
        subj_vocab.setdefault(str(s), len(subj_vocab))
    subject_index_map = subj_vocab

    # ---- (a) embeddings for the index items (+ optional qwen reuse short-circuit) ----
    from aide.features.driver import load_embeddings
    item_keys_emb, item_emb = load_embeddings(f"{_emb_dir(driver_fam)}/items.parquet")
    item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
    emb_idx = {str(k_): i for i, k_ in enumerate(item_keys_emb)}
    d_emb = int(item_emb.shape[1])
    miss = [k_ for k_ in index_item_keys if k_ not in emb_idx]
    if miss:
        raise ValueError(f"{len(miss)} index items missing from {fam} emb cache "
                         f"(e.g. {miss[:3]})")
    idx_rows = np.fromiter((emb_idx[k_] for k_ in index_item_keys), dtype=np.int64,
                           count=len(index_item_keys))
    index_emb_lookup = {k_: item_emb[emb_idx[k_]] for k_ in index_item_keys}

    if root.exists() and not overwrite and (root / "infra_meta.json").exists():
        # reload the cached bundle
        return load_nn_infra(fam, subset=subset, k=k, similarity=similarity)
    root.mkdir(parents=True, exist_ok=True)

    # ---- (b) TrainingNNIndex over the index items ----
    nn_index = TrainingNNIndex.build_from_lookup(
        index_emb_lookup, root, cfg, item_keys=index_item_keys
    )

    # ---- (c) legacy subject x item passrate CSR (cells 0..7) ----
    pr_csr, pr_mask = build_passrate_table(train_df, item_index_map, subject_index_map)
    from scipy import sparse
    sparse.save_npz(root / "subject_passrate.npz", pr_csr)
    sparse.save_npz(root / "subject_passrate_mask.npz", pr_mask)

    # ---- (d) k-means centroids over the index items (or reuse qwen) ----
    if reuse_qwen_artifacts and fam == "qwen":
        qc = Path(f"{_drive_root()}/artifacts/cluster_centroids.npy")
        centroids = (np.load(qc).astype(np.float32)
                     if qc.exists() else fit_kmeans(item_emb[idx_rows], int(kmeans_k)))
    else:
        centroids = fit_kmeans(item_emb[idx_rows], int(kmeans_k))
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    np.save(root / "centroids.npy", centroids)
    # item cluster id (1-based; 0=UNK) over the index items, for the conditional context
    item_cluster_id_1based = assign_clusters(centroids, item_emb[idx_rows])
    # context expects -1 = unclustered; shift the 1-based assign (>=1) to 0-based here
    item_cluster_id = (item_cluster_id_1based - 1).astype(np.int32)

    # ---- (e) metadata ids for the conditional context ----
    prepared_parquet = prepared_parquet or _glob_prepared()
    meta = _resolve_metadata_ids(
        repo_root=_repo_root(), drive_root=_drive_root(),
        index_item_keys=index_item_keys,
        subject_keys=list(subject_index_map.keys()),
        prepared_parquet=prepared_parquet,
    )

    # ---- (f) conditional passrate context (cells 15..22) ----
    cond_ctx = build_conditional_passrate_context(
        train_df=train_df,
        item_index_map=item_index_map,
        subject_index_map=subject_index_map,
        subject_to_family_id=meta["subject_to_family_id"],
        subject_to_macro_family_id=meta["subject_to_macro_family_id"],
        subject_to_organization_id=meta["subject_to_organization_id"],
        item_benchmark_id=meta["item_benchmark_id"],
        item_benchmark_age=meta["item_benchmark_age"],
        item_cluster_id=item_cluster_id,
        n_families=meta["n_families"],
        n_macro_families=meta["n_macro_families"],
        n_organizations=meta["n_organizations"],
        n_clusters=int(centroids.shape[0]),
    )
    cond_ctx.save(root)  # writes the bag + conditional_meta.json (runtime layout)
    _write_json(root / "benchmark_to_id.json", meta["benchmark_to_id"])

    # ---- (g) id maps + infra meta ----
    _write_json(root / "item_index.json", index_item_keys)
    _write_json(root / "subject_index.json", subject_index_map)
    _write_json(root / "infra_meta.json", {
        "fam": fam, "driver_family": driver_fam, "subset": subset,
        "k": int(k), "kmeans_k": int(centroids.shape[0]), "similarity": str(similarity),
        "d_emb": d_emb, "n_index_items": len(index_item_keys),
        "n_subjects": len(subject_index_map),
        "n_families": meta["n_families"], "n_macro_families": meta["n_macro_families"],
        "n_organizations": meta["n_organizations"],
        "subject_meta_coverage": meta["subject_coverage"],
        "NN_FEATURE_DIM": 23,
    })

    return NNInfra(
        fam=fam, subset=subset, root=str(root), k=int(k), similarity=str(similarity),
        nn_index=nn_index, cfg=cfg,
        passrate_csr=pr_csr, passrate_mask_csr=pr_mask,
        item_index_map=item_index_map, subject_index_map=subject_index_map,
        index_item_keys=index_item_keys,
        cond_ctx=cond_ctx, benchmark_to_id=meta["benchmark_to_id"],
        centroids=centroids, d_emb=d_emb,
    )


def _glob_prepared() -> str:
    import glob
    hits = glob.glob(f"{_drive_root()}/prepared_datasets/*measurement_db_prepared*.parquet")
    if not hits:
        raise FileNotFoundError("prepared parquet not found on Drive")
    return hits[0]


def load_nn_infra(fam: str, *, subset: str = "all", k: int = DEFAULT_K,
                  similarity: str = DEFAULT_SIM) -> NNInfra:
    """Reload a prepped infra bundle from DR/ship/nn_infra/<fam>/<subset>/."""
    import sys
    if _repo_root() not in sys.path:
        sys.path.insert(0, _repo_root())
    from scipy import sparse

    from src.nn_features import (
        ConditionalPassrateContext,
        NNFeaturesConfig,
        TrainingNNIndex,
    )

    root = Path(f"{_drive_root()}/ship/nn_infra/{fam}/{subset}")
    im = _read_json(root / "infra_meta.json")
    cfg = NNFeaturesConfig(k=int(im.get("k", k)), similarity=str(im.get("similarity", similarity)),
                           cache_dir=str(root), prefer_gpu=True)
    nn_index = TrainingNNIndex.load(root, cfg)
    pr_csr = sparse.load_npz(root / "subject_passrate.npz").tocsr()
    pr_mask = sparse.load_npz(root / "subject_passrate_mask.npz").tocsr()
    index_item_keys = _read_json(root / "item_index.json")
    subject_index_map = _read_json(root / "subject_index.json")
    item_index_map = {str(k_): i for i, k_ in enumerate(index_item_keys)}
    cond_ctx = ConditionalPassrateContext.load(root)
    benchmark_to_id = (_read_json(root / "benchmark_to_id.json")
                       if (root / "benchmark_to_id.json").exists() else {})
    centroids = (np.load(root / "centroids.npy").astype(np.float32)
                 if (root / "centroids.npy").exists() else None)
    return NNInfra(
        fam=fam, subset=subset, root=str(root), k=int(im.get("k", k)),
        similarity=str(im.get("similarity", similarity)),
        nn_index=nn_index, cfg=cfg, passrate_csr=pr_csr, passrate_mask_csr=pr_mask,
        item_index_map=item_index_map, subject_index_map=subject_index_map,
        index_item_keys=index_item_keys, cond_ctx=cond_ctx,
        benchmark_to_id={str(k_): int(v) for k_, v in benchmark_to_id.items()},
        centroids=centroids, d_emb=int(im.get("d_emb", 0)),
    )


# ----------------------------------------------------------------------------------
# compute_features: the 23-dim AIDE NN matrix for ANY row set (the holdout-scorer).
# ----------------------------------------------------------------------------------
def compute_features(
    infra: NNInfra,
    rows_item: list[str],
    rows_subj: list[str],
    *,
    item_emb_lookup,                  # maps item_key -> 1-D float32 (family emb cache)
    query_benchmark_ids=None,         # [N] int32 (-1 unknown); optional cell-19 driver
    query_benchmark_age=None,         # [N] float32 (NaN unknown); optional cell-20 driver
    query_cluster_ids=None,           # [N] int32 (-1 unknown); optional cell-22 driver
    exclude_self: bool | None = None,
    chunk: int = 4096,
) -> np.ndarray:
    """Return the [N, 23] AIDE NN feature matrix for the given rows, LIVE.

    Thin wrapper over ``src.nn_features.compute_nn_features_streaming`` so train-time
    feature computation is the BATCH twin of the runtime per-item path
    (``nn23_runtime.compute_nn_features_23``); both call ``_aggregate_nn_features``, so
    cells 0..22 are bit-identical for a given (item, subject, neighbor) configuration.

    For honest val (fold2) pass an ``infra`` built over folds{0,1} only -- then fold2
    items are absent from the index and cannot self-retrieve. ``exclude_self`` defaults
    to the cfg value (True) and additionally drops any query item that IS in the index.
    """
    from src.nn_features import compute_nn_features_streaming

    rows_item = [str(x) for x in rows_item]
    sids = np.fromiter(
        (infra.subject_index_map.get(str(s), len(infra.subject_index_map))
         for s in rows_subj),
        dtype=np.int64, count=len(rows_subj),
    )
    feats = compute_nn_features_streaming(
        query_item_keys=rows_item,
        item_emb_lookup=item_emb_lookup,
        subject_ids=sids,
        nn_index=infra.nn_index,
        passrate_csr=infra.passrate_csr,
        passrate_mask_csr=infra.passrate_mask_csr,
        cfg=infra.cfg,
        exclude_self=exclude_self,
        query_chunk_size=int(chunk),
        conditional_context=infra.cond_ctx,
        query_benchmark_ids=query_benchmark_ids,
        query_benchmark_age=query_benchmark_age,
        query_cluster_ids=query_cluster_ids,
    )
    return np.ascontiguousarray(feats, dtype=np.float32)


# ----------------------------------------------------------------------------------
# compute_dense_block: [23-dim AIDE NN | centroid-distance | metadata] for a row set.
# ----------------------------------------------------------------------------------
def compute_dense_block(
    infra: NNInfra,
    rows_item: list[str],
    rows_subj: list[str],
    *,
    item_emb_lookup,
    metadata_out: dict | None = None,   # slice of metadata_tables.build_metadata_tables
    metadata_slice: slice | None = None,  # which rows of metadata_out map to THESE rows
    top_m_centroids: int = 4,
    chunk: int = 4096,
):
    """Assemble the full dense_X = [nn23 | centroid_distance(top_m) | metadata_numeric+onehot].

    Returns (dense_X [N, F] float32, feature_names tuple[str]). The column order is LOCKED
    and recorded so the shipped runtime reproduces it exactly (see RUNTIME PARITY CONTRACT):
        cols  0..22                : 23-dim AIDE NN feats (compute_features)
        next  top_m                : sorted top-m squared-L2 distances to k-means centroids
        next  +1                   : nearest-centroid distance gap (d2 - d1) peakedness
        then  metadata_numeric     : metadata_tables 'numerical' block (z-scored, leak-free)
        then  metadata onehots     : family / macro_family / organization / bench_topic
                                     (low-card categoricals, train-fit vocab; UNK column)
    ``metadata_out`` is the dict from ``scripts.ship.metadata_tables.build_metadata_tables``
    (fit on the train split, encoding both train+holdout); ``metadata_slice`` selects the
    rows for THIS call (e.g. slice(0, n_first) for train, slice(n_first, None) for holdout).
    Pass metadata_out=None to ship the NN+centroid block only.
    """
    from src.clustering import compute_top_m_distances

    nn23 = compute_features(infra, rows_item, rows_subj,
                            item_emb_lookup=item_emb_lookup, chunk=chunk)
    names = list(_NN23_NAMES)
    blocks = [nn23]

    # ---- centroid distance block ----
    if infra.centroids is not None:
        q = np.empty((len(rows_item), infra.d_emb), dtype=np.float32)
        for j, k_ in enumerate(rows_item):
            q[j] = item_emb_lookup[str(k_)]
        _ids, d2 = compute_top_m_distances(infra.centroids, q, top_m=int(top_m_centroids))  # ([N,m],[N,m])
        d2 = np.ascontiguousarray(d2, dtype=np.float32)
        gap = (d2[:, 1] - d2[:, 0]).reshape(-1, 1) if d2.shape[1] >= 2 else np.zeros((d2.shape[0], 1), np.float32)
        blocks.append(d2)
        blocks.append(gap.astype(np.float32))
        names += [f"centroid_d2_{i}" for i in range(d2.shape[1])] + ["centroid_gap"]

    # ---- metadata block (numeric + low-card one-hots), leak-free ----
    if metadata_out is not None:
        sl = metadata_slice if metadata_slice is not None else slice(0, len(rows_item))
        num = np.asarray(metadata_out["numerical"], dtype=np.float32)[sl]
        blocks.append(num)
        names += [f"meta_num__{n}" for n in metadata_out["num_feature_names"]]
        for key, card_key in (("family_ids", "n_families"),
                              ("macro_family_ids", "n_macro_families"),
                              ("organization_ids", "n_organizations"),
                              ("bench_topic_ids", "n_bench_topics")):
            ids = np.asarray(metadata_out[key], dtype=np.int64)[sl]
            card = int(metadata_out[card_key])
            oh = np.zeros((ids.shape[0], card), dtype=np.float32)
            valid = (ids >= 0) & (ids < card)
            oh[np.arange(ids.shape[0])[valid], ids[valid]] = 1.0
            blocks.append(oh)
            names += [f"meta_{key}__{j}" for j in range(card)]

    dense = np.concatenate(blocks, axis=1).astype(np.float32)
    return np.ascontiguousarray(dense), tuple(names)


_NN23_NAMES = (
    "passrate_mean", "passrate_weighted_mean", "passrate_std", "coverage",
    "top1_label", "top1_similarity", "mean_similarity", "n_labeled_neighbors_log1p",
    "effective_neighbor_count", "top1_minus_topk_similarity", "bootstrap_se_passrate",
    "neighbor_label_entropy", "top1_label_match", "sim_distribution_skew",
    "distance_to_kth_neighbor", "passrate_subject_conditional",
    "passrate_family_conditional", "passrate_macro_family_conditional",
    "passrate_organization_conditional", "passrate_benchmark_conditional",
    "neighbor_freshness_diff", "n_distinct_subjects_in_neighborhood",
    "cluster_passrate_subject_query",
)


__all__ = [
    "NNInfra",
    "prep_nn_infra",
    "load_nn_infra",
    "compute_features",
    "compute_dense_block",
    "FAM_ALIAS",
    "DEFAULT_K",
    "DEFAULT_KMEANS_K",
]
