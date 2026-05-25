# %% [markdown]
# # Qwen3-Embedding-8B minimalist notebook
#
# **Goal**: train one metadata-aware head on cached `Qwen/Qwen3-Embedding-8B`
# vectors, with FAISS-GPU nearest-neighbor features + multi-centroid soft
# cluster distances ON by default, fit a Netflix-Prize-style nearest-
# neighbor calibrator on validation, and export a Codabench-ready
# submission. No judge LLM, no LoRA fine-tuning, no K-fold ablation grid.
#
# What ships in the bundle:
#
# - `artifacts/checkpoint.pt` -- trained `meta_hybrid_irt_kfactor_gated_mlp`.
# - `artifacts/cluster_centroids.npy` -- runtime computes top-m centroid
#   distances on every prediction. Same math as training, see
#   `src/clustering.py:compute_top_m_distances`.
# - `artifacts/pool_features_stats.json` -- z-score stats including the
#   `centroid_dist_*` columns.
# - `artifacts/meta_preprocessor.json` -- cold-start subject / benchmark
#   metadata encoder.
# - `artifacts/runtime_meta.json` -- includes `nn_calibrator` block (alpha,
#   k, similarity, ...).
# - `cache/` -- training-item cache (PCA + int8 + FAISS) for runtime NN
#   feature lookup. **Required** when NN features are on (without it the
#   trained NN slot would receive zeros at test time and the head would
#   regress).
# - `cache/nn_residual/` -- `(subject_id, training_item_row) ->
#   (label, p_uncal)` sparse table for the NN calibrator. **Required**
#   whenever `nn_calibrator.alpha != 0`.
#
# Future ensemble work consumes the same `cache/nn_residual/` artifact:
# every member ships its own table (alongside its own checkpoint), the
# orchestrator can then either average the per-member NN-calibrated
# probabilities or fit a per-member alpha on a held-out blend split.

# %% [markdown]
# ## 1. Setup (Colab + local both supported)

# %%
import os
import subprocess
import sys
from pathlib import Path

# Edit these two lines if you fork the repo or want to keep a working
# copy on Drive instead of /content. ``REPO_URL`` is only used when the
# directory at ``REPO_DIR`` does not already exist.
REPO_URL = "https://github.com/bwathomas/prediction-competition-321M.git"
REPO_DIR = Path("/content/Prediction-Competition-321M")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        print(f"{REPO_DIR} already cloned; pulling latest...")
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
            check=False,
        )
    elif not REPO_DIR.exists():
        print(f"Cloning {REPO_URL} -> {REPO_DIR} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            check=True,  # loud failure on bad URL / network issue
        )
    if not (REPO_DIR / "configs" / "default.yaml").exists():
        raise FileNotFoundError(
            f"{REPO_DIR} does not look like the prediction-competition repo "
            f"(no configs/default.yaml). Edit REPO_URL/REPO_DIR above and "
            "rerun this cell."
        )
    os.chdir(REPO_DIR)
    subprocess.run(
        ["pip", "install", "-q", "faiss-gpu-cu12", "sentence-transformers", "tqdm"],
        check=False,
    )

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
print(f"Working directory: {ROOT}")
print(f"In Colab: {IN_COLAB}")

# Configure Python's root logger so per-epoch validation metrics from
# ``src.train`` (and other ``LOG.info(...)`` calls across the package)
# actually show up in the notebook. Without this, Jupyter / Colab
# defaults the root logger to WARNING and the trainer's per-epoch
# `val_log_loss / val_brier / val_auc` lines are silently dropped --
# you only see the final ``best val log-loss`` print at the end of
# the cell. ``force=True`` (Python 3.8+) replaces any pre-installed
# Colab handlers so we get a single, predictable formatter.
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
# The ``train`` logger is the one that emits per-epoch summaries.
logging.getLogger("train").setLevel(logging.INFO)

# %% [markdown]
# ## 2. Configuration: Qwen8B + metadata + NN + centroid distances ON

# %%
import json
import yaml

with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

# --- Encoder: Qwen3-Embedding-8B with content-hash drive cache.
# Qwen3-Embedding requires last_token pooling per the model card.
CFG["encoder"].update({
    "model_id": "Qwen/Qwen3-Embedding-8B",
    "max_length": 512,
    "batch_size": 8,
    "use_flash_attention": True,
    "trust_remote_code": False,
    "pooling": "last_token",
    "use_contextual_item_text": True,
})

# --- Model: metadata-mode hybrid IRT (cold-start aware).
CFG.setdefault("metadata", {})
CFG["metadata"]["enabled"] = True
CFG.setdefault("train", {})
CFG["train"]["models"] = ["meta_hybrid_irt_kfactor_gated_mlp"]

# --- Item-side feature channels. NN + centroid distances ON, judge OFF.
CFG.setdefault("item_features", {})
CFG["item_features"]["use_pool"] = True
CFG.setdefault("clustering", {})
CFG["clustering"]["k"] = int(CFG["clustering"].get("k", 64))
CFG.setdefault("centroid_distances", {})
CFG["centroid_distances"]["enabled"] = True
CFG["centroid_distances"]["top_m"] = 8
CFG.setdefault("nn_features", {})
CFG["nn_features"]["enabled"] = True
CFG["nn_features"]["k"] = 16
CFG["nn_features"]["similarity"] = "cosine"
CFG["nn_features"]["prefer_gpu"] = True
CFG.setdefault("judge", {})
CFG["judge"]["enabled"] = False
CFG.setdefault("lora", {})
CFG["lora"]["enabled"] = False
CFG.setdefault("kfold", {})
CFG["kfold"]["enabled"] = False
CFG.setdefault("submission", {})
CFG["submission"]["ship_training_cache"] = True
CFG["submission"]["ship_requirements_txt"] = False
CFG.setdefault("nn_calibration", {})
CFG["nn_calibration"]["enabled"] = True
CFG["nn_calibration"]["k"] = 16
CFG["nn_calibration"]["temperature"] = 1.0
CFG["nn_calibration"]["min_weight_sum"] = 1e-3

print("Encoder:", CFG["encoder"]["model_id"])
print("Model variant:", CFG["train"]["models"])
print("Metadata enabled:", CFG["metadata"]["enabled"])
print("NN features:", CFG["nn_features"]["enabled"], "k=", CFG["nn_features"]["k"])
print("Centroid distances:", CFG["centroid_distances"]["enabled"],
      "top_m=", CFG["centroid_distances"]["top_m"])
print("Judge:", CFG["judge"]["enabled"])

# %%
from src.embeddings import login_huggingface, resolve_hf_token

HF_TOKEN = resolve_hf_token()
if HF_TOKEN:
    login_huggingface(HF_TOKEN)
    print("Hugging Face: logged in")
else:
    print("Hugging Face: no token resolved (cache-only run assumed)")

# %% [markdown]
# ## 3. Data + cold-start splits

# %%
import numpy as np
import pandas as pd

from src.data import (
    compute_dataset_stats,
    make_item_cold_start_split,
    prepare_dataset,
    print_dataset_stats,
)

df = prepare_dataset(CFG["data"], token=HF_TOKEN, download=True)
print(f"Dataset rows: {len(df):,}")
print_dataset_stats(compute_dataset_stats(df))

SEED = int(CFG["seed"])
primary = make_item_cold_start_split(
    df,
    val_fraction=float(CFG["splits"]["val_fraction"]),
    seed=SEED,
)
print(f"train rows: {len(primary.train):,}  val rows: {len(primary.val):,}")

# %% [markdown]
# ## 4. Encode (or load from cache) with Qwen3-Embedding-8B

# %%
from dataclasses import fields as _dc_fields

from src import drive_cache as drive_cache_mod
from src.embeddings import (
    EncoderConfig,
    TransformerEmbedder,
    assert_deduplicated,
    build_unique_items,
    build_unique_subjects,
    content_hash_for_items,
    encoder_slug,
    verify_flash_attention,
)

fa_active, fa_msg = verify_flash_attention(
    bool(CFG["encoder"].get("use_flash_attention", False))
)
print("Flash Attention 2:", "ACTIVE" if fa_active else "OFF", "--", fa_msg)
if not fa_active and CFG["encoder"].get("use_flash_attention"):
    CFG["encoder"]["use_flash_attention"] = False

# Filter CFG["encoder"] to only the keys EncoderConfig accepts; the YAML
# also holds runtime-only knobs like ``runtime_batch_size`` that would
# otherwise blow up the dataclass constructor with a TypeError.
_enc_known = {f.name for f in _dc_fields(EncoderConfig)}
_enc_kwargs = {k: v for k, v in CFG["encoder"].items() if k in _enc_known}
enc_cfg = EncoderConfig(**_enc_kwargs)
embedder = TransformerEmbedder(enc_cfg)
slug = encoder_slug(enc_cfg.model_id)

print(f"Encoder            : {enc_cfg.model_id}")
print(f"Pooling            : {enc_cfg.pooling}")
print(f"Cache dir          : {embedder.base}")

required_item_cols = {"item_key", "benchmark", "condition", "item_content"}
missing = required_item_cols - set(df.columns)
if missing:
    raise ValueError(f"df is missing required item columns: {sorted(missing)}")
required_subject_cols = {"subject_key", "subject_content"}
missing = required_subject_cols - set(df.columns)
if missing:
    raise ValueError(f"df is missing required subject columns: {sorted(missing)}")

item_df = (
    df[["item_key", "benchmark", "condition", "item_content"]]
    .drop_duplicates(subset=["item_key"])
    .reset_index(drop=True)
)
item_keys_list, item_texts_list, item_benches_list = build_unique_items(
    item_df,
    contextual=enc_cfg.use_contextual_item_text,
    passage_prefix=embedder._resolve_passage_prefix(),
)
assert_deduplicated(item_keys_list, kind="item")

subject_df = (
    df[["subject_key", "subject_content"]]
    .drop_duplicates(subset=["subject_key"])
    .reset_index(drop=True)
)
subject_keys_list, subject_texts_list = build_unique_subjects(
    subject_df, query_prefix=embedder._resolve_query_prefix()
)
assert_deduplicated(subject_keys_list, kind="subject")

CONTENT_HASH = content_hash_for_items(
    list(zip(item_keys_list, item_texts_list))
    + list(zip(subject_keys_list, subject_texts_list))
)
print(f"Content hash: {CONTENT_HASH[:16]}...")

cache_root = ROOT / CFG["encoder"]["cache_dir"]
drive_status = drive_cache_mod.resolve_cache(
    cfg=CFG, encoder_slug=slug,
    local_cache_root=cache_root, expected_hash=CONTENT_HASH,
)
print("Cache decision:", drive_status.reason,
      "(hit:", bool(drive_status.cache_hit), ")")

embedder.warm_caches_from_disk()

print("Encoding items (cache-aware)...")
item_emb_lookup, item_log = embedder.embed_unique(
    kind="item",
    keys=item_keys_list,
    texts=item_texts_list,
    benchmarks=item_benches_list,
)
print(f"  cached={item_log['n_cache_hits']}  encoded={item_log['n_encoded']}")

print("Encoding subjects (cache-aware)...")
subject_emb_lookup, subject_log = embedder.embed_unique(
    kind="subject", keys=subject_keys_list, texts=subject_texts_list,
)
print(f"  cached={subject_log['n_cache_hits']}  encoded={subject_log['n_encoded']}")

embedder.finalize(
    content_hash=CONTENT_HASH,
    n_items=len(item_keys_list),
    n_subjects=len(subject_keys_list),
    extra_log={
        "items": item_log,
        "subjects": subject_log,
        "drive_cache": drive_status.as_dict(),
    },
)

drive_cfg = CFG.get("drive_cache") or {}
if (
    drive_cfg.get("enabled")
    and drive_cfg.get("upload_on_completion", True)
    and getattr(drive_status, "mounted", False)
    and (item_log["n_encoded"] or subject_log["n_encoded"])
):
    drive_folder = Path(drive_cfg["folder"]) / slug
    print(drive_cache_mod.upload_from_local(
        local_folder=embedder.base, drive_folder=drive_folder,
    ))

ITEM_EMB_DIM = embedder.embedding_dim
print(f"Item embedding dim: {ITEM_EMB_DIM}")

# %% [markdown]
# ## 5. Pool features + k-means + multi-centroid distances (FAISS GPU)

# %%
from src.clustering import fit_and_assign
from src.item_features import (
    apply_zscore,
    build_centroid_distance_features,
    centroid_distance_feature_names,
    compute_features_for_items,
    fit_zscore_stats,
    load_pool_features,
    merge_pool_and_centroid_features,
    save_pool_features,
    save_zscore_stats,
)

ART_FEATURES = ROOT / "artifacts" / "item_features"
POOL_PATH = ART_FEATURES / "pool_features.parquet"
POOL_STATS_PATH = ART_FEATURES / "pool_features_stats.json"
CENTROIDS_PATH = ROOT / CFG["clustering"]["centroids_path"]
ASSIGN_PATH = ROOT / CFG["clustering"]["assignments_path"]

pool_df = load_pool_features(POOL_PATH)
if pool_df is None:
    pool_df = compute_features_for_items(item_df, progress=True)
    save_pool_features(pool_df, POOL_PATH)
    print(f"Computed pool features for {len(pool_df):,} unique items.")
else:
    print(f"Loaded pool features for {len(pool_df):,} unique items.")

n_clusters = int(CFG["clustering"]["k"])
all_emb_keys = list(item_emb_lookup.keys())
all_emb = np.stack([item_emb_lookup[k] for k in all_emb_keys], axis=0)
centroids, cluster_assignments = fit_and_assign(
    all_emb_keys,
    all_emb,
    k=n_clusters,
    seed=int(CFG["clustering"]["seed"]),
    centroids_path=CENTROIDS_PATH,
    assignments_path=ASSIGN_PATH,
    overwrite=False,
    backend=str(CFG["clustering"].get("backend", "auto")),
    niter=int(CFG["clustering"].get("faiss_niter", 50)),
    nredo=int(CFG["clustering"].get("faiss_nredo", 1)),
    gpu_id=int(CFG["clustering"].get("gpu_id", 0)),
    assign_batch_size=int(
        CFG["clustering"].get("faiss_assign_batch", 65536)
    ),
)
print(f"Clustering: {centroids.shape[0]} centroids, "
      f"{len(cluster_assignments)} item assignments")

top_m = int(CFG["centroid_distances"]["top_m"])
centroid_distance_df = build_centroid_distance_features(
    item_keys=pool_df["item_key"].astype(str).tolist(),
    item_emb_lookup=item_emb_lookup,
    centroids=centroids,
    top_m=top_m,
)
print(f"Centroid-distance columns: {centroid_distance_feature_names(top_m)}")

combined_df, combined_cols = merge_pool_and_centroid_features(
    pool_df, centroid_distance_df,
)
train_keys = set(primary.train["item_key"].astype(str).tolist())
train_features = combined_df[combined_df["item_key"].astype(str).isin(train_keys)]
pool_stats = fit_zscore_stats(train_features, feature_cols=combined_cols)
save_zscore_stats(pool_stats, POOL_STATS_PATH)
pool_features_z = apply_zscore(combined_df, pool_stats)
POOL_FEATURE_NAMES_EXT = tuple(combined_cols)
print(f"Pool feature width: {len(POOL_FEATURE_NAMES_EXT)}  "
      f"(9 text + {top_m} centroid_dist)")

# %% [markdown]
# ## 6. Indexer + nearest-neighbor features (FAISS GPU; on by default)

# %%
import gc

from src.models import Indexer
from src.nn_features import (
    NNFeaturesConfig,
    TrainingNNIndex,
    build_passrate_table,
    compute_nn_features_streaming,
)


def _ram_gb() -> float:
    """Return current process RSS in GiB (psutil falls back to 0.0)."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024.0**3)
    except Exception:
        return 0.0


def _log_ram(label: str) -> None:
    rss = _ram_gb()
    if rss > 0.0:
        print(f"  [RAM] {label}: rss={rss:.2f} GiB")


_log_ram("cell 6 start")

# Estimate ``item_emb_lookup`` resident bytes so a future OOM can be
# attributed quickly. The dict overhead itself is ~100 MB per million
# keys; the dominant cost is the underlying float32 ndarrays.
_lookup_n = len(item_emb_lookup)
_lookup_bytes = _lookup_n * int(ITEM_EMB_DIM) * 4
print(
    f"item_emb_lookup: N={_lookup_n:,}  D={ITEM_EMB_DIM}  "
    f"~{_lookup_bytes / (1024.0**3):.2f} GiB of float32 tensors"
)

# Single source of truth for subject_id / bc_id semantics. Reused below
# by the NN passrate table, training tensors, and the NN calibrator.
indexer = Indexer.fit(
    subject_keys=primary.train["subject_key"].tolist(),
    bc_keys=primary.train["benchmark_condition_key"].tolist(),
)
print(f"Indexer: n_subjects={indexer.n_subjects}  n_bc={indexer.n_bc}")
_log_ram("after Indexer.fit")

nn_cfg_dict = dict(CFG["nn_features"])
# Force the FAISS-only memory-saving path: after the IndexFlat is built
# we can drop ``self.embeddings`` (FAISS keeps its own internal copy)
# and reclaim ``N * D * 4`` bytes of RSS. With Qwen3-Embedding-8B this
# is ~1.6 GB per 100k training items.
nn_cfg_dict.setdefault("free_embeddings_after_faiss", True)
nn_cfg = NNFeaturesConfig.from_dict(nn_cfg_dict)
NN_DIR = ROOT / nn_cfg.cache_dir / "training"
NN_DIR.mkdir(parents=True, exist_ok=True)

train_item_keys = sorted(set(primary.train["item_key"].astype(str)))
train_item_keys = [k for k in train_item_keys if k in item_emb_lookup]

n_train_items = len(train_item_keys)
predicted_peak_gb = 2.0 * n_train_items * int(ITEM_EMB_DIM) * 4.0 / (1024.0**3)
print(
    f"NN index build: N={n_train_items:,}  D={ITEM_EMB_DIM}  "
    f"predicted peak ~{predicted_peak_gb:.2f} GiB during cosine norm + FAISS add"
)

# Pass ``item_emb_lookup`` directly -- ``build_from_lookup`` only reads
# the keys we list in ``item_keys``, so the previous dict-comprehension
# subset was a wasteful temporary.
nn_index = TrainingNNIndex.build_from_lookup(
    item_emb_lookup=item_emb_lookup,
    out_dir=NN_DIR,
    cfg=nn_cfg,
    item_keys=train_item_keys,
)
gc.collect()
_log_ram("after TrainingNNIndex.build_from_lookup")
nn_index_kind = type(nn_index._faiss_index).__name__ if nn_index._faiss_index is not None else "numpy"
print(f"NN index: N={len(train_item_keys):,}  D={ITEM_EMB_DIM}  "
      f"backend={nn_index_kind}  similarity={nn_cfg.similarity}")

nn_item_index_map = {k: i for i, k in enumerate(train_item_keys)}
nn_passrate_csr, nn_passrate_mask_csr = build_passrate_table(
    train_df=primary.train,
    item_index_map=nn_item_index_map,
    subject_index_map=indexer.subject_to_id,
)
print(f"Passrate matrix: shape={nn_passrate_csr.shape}  "
      f"nnz={nn_passrate_csr.nnz:,}")

# `compute_nn_features_streaming` dedupes by item_key (neighbor structure
# is subject-independent) and stacks queries one chunk at a time, so peak
# RAM stays at ``NN_QUERY_CHUNK * D * 4`` bytes (~67 MB at chunk=4096,
# D=4096) instead of the ~14 GB the row-by-row np.stack path would peak
# at on a 12 GB Colab with Qwen3-Embedding-8B.
NN_QUERY_CHUNK = int(CFG["nn_features"].get("query_chunk_size", 4096))


def _split_query(rows_df):
    keys = rows_df["item_key"].astype(str).tolist()
    sids = np.array(
        [indexer.subject_id(str(s)) for s in rows_df["subject_key"]],
        dtype=np.int64,
    )
    return keys, sids


train_keys_for_nn, train_sid = _split_query(primary.train)
val_keys_for_nn, val_sid = _split_query(primary.val)

_log_ram("before compute_nn_features_streaming(train)")
nn_train_mat = compute_nn_features_streaming(
    query_item_keys=train_keys_for_nn,
    item_emb_lookup=item_emb_lookup,
    subject_ids=train_sid,
    nn_index=nn_index,
    passrate_csr=nn_passrate_csr,
    passrate_mask_csr=nn_passrate_mask_csr,
    cfg=nn_cfg,
    exclude_self=True,
    query_chunk_size=NN_QUERY_CHUNK,
)
gc.collect()
_log_ram("after compute_nn_features_streaming(train)")
nn_val_mat = compute_nn_features_streaming(
    query_item_keys=val_keys_for_nn,
    item_emb_lookup=item_emb_lookup,
    subject_ids=val_sid,
    nn_index=nn_index,
    passrate_csr=nn_passrate_csr,
    passrate_mask_csr=nn_passrate_mask_csr,
    cfg=nn_cfg,
    exclude_self=False,
    query_chunk_size=NN_QUERY_CHUNK,
)
gc.collect()
_log_ram("after compute_nn_features_streaming(val)")
print(f"NN feature matrices: train={nn_train_mat.shape}  val={nn_val_mat.shape}")

# Sanity: passrate_mean must positively correlate with the val label,
# else NN features are wired up wrong and we should fail loudly here
# rather than train on garbage.
pmean = nn_val_mat[:, 0].astype(np.float64)
ylab = primary.val["label"].astype(float).to_numpy()
mask = (np.abs(pmean - float(nn_cfg.fallback_value)) > 1e-9)
if mask.sum() >= 100:
    corr = float(np.corrcoef(pmean[mask], ylab[mask])[0, 1])
    print(f"NN sanity: corr(passrate_mean, val_label)={corr:+.3f}  on n={int(mask.sum()):,}")
    if corr <= 0.0:
        raise RuntimeError(
            f"NN features failed sanity check: corr={corr:+.3f}. "
            "Fix the (subject, item) keying or pool / index dim mismatch "
            "before training."
        )

# %% [markdown]
# ## 7. Build training tensors + metadata artifacts

# %%
from src.data import prepare_metadata_artifacts
from src.embeddings import stack_lookup
from src.item_features import build_feature_matrix
from src.metadata_features import MetadataSchema
from src.models import LookupDataset


def _pool_matrix(keys):
    return build_feature_matrix(
        [str(k) for k in keys],
        pool_features_z,
        feature_cols=list(POOL_FEATURE_NAMES_EXT),
        key_col="item_key",
    )


def _cluster_vector(keys):
    return np.array(
        [int(cluster_assignments.get(str(k), 0)) for k in keys],
        dtype=np.int64,
    )


def _build(split_part, nn_mat):
    s = np.array([indexer.subject_id(k) for k in split_part["subject_key"]], dtype=np.int64)
    bc = np.array(
        [indexer.bc_id(k) for k in split_part["benchmark_condition_key"]],
        dtype=np.int64,
    )
    ie = stack_lookup(split_part["item_key"], item_emb_lookup)
    pf = _pool_matrix(split_part["item_key"])
    ci = _cluster_vector(split_part["item_key"])
    y = split_part["label"].astype(float).to_numpy()
    return LookupDataset(
        subject_ids=s,
        bc_ids=bc,
        item_emb=ie,
        labels=y,
        subject_emb=None,
        pool_feats=pf,
        cluster_ids=ci,
        judge_feats=None,
        nn_feats=nn_mat.astype(np.float32),
        sample_weights=None,
    )


train_ds = _build(primary.train, nn_train_mat)
val_ds = _build(primary.val, nn_val_mat)
print(f"train tensors: {len(train_ds)}  val tensors: {len(val_ds)}")

# Pin the schema explicitly so the metadata preprocessor's vocab + scaler
# layout always matches the ModelConfig fields below. Without this, the
# preprocessor falls back to MetadataSchema's hard-coded defaults and the
# resulting buffer widths can disagree with the model's tower input dims
# whenever the YAML diverges from those defaults.
META_SECTION = CFG["metadata"]
_meta_schema = MetadataSchema(
    subject_categorical=tuple(META_SECTION.get("subject_categorical", ()) or ()),
    subject_numeric=tuple(META_SECTION.get("subject_numeric", ()) or ()),
    benchmark_categorical=tuple(META_SECTION.get("benchmark_categorical", ()) or ()),
    benchmark_numeric=tuple(META_SECTION.get("benchmark_numeric", ()) or ()),
    explicit_crosses=tuple(META_SECTION.get("explicit_crosses", ()) or ()),
)
meta_preprocessor, meta_id_tables = prepare_metadata_artifacts(
    primary.train, indexer, schema=_meta_schema,
)
print(
    f"MetadataPreprocessor: "
    f"subj_cat={list(meta_preprocessor.subject_cat_vocabs)}  "
    f"subj_num={list(meta_preprocessor.subject_num_scalers)}  "
    f"bench_cat={list(meta_preprocessor.benchmark_cat_vocabs)}  "
    f"bench_num={list(meta_preprocessor.benchmark_num_scalers)}"
)

# %% [markdown]
# ## 8. Train the metadata-aware head

# %%
from dataclasses import asdict

import src.train as train_mod
from src.models import ModelConfig
from src.train import TrainConfig, train_one

CKPT_DIR = ROOT / "artifacts" / "checkpoints" / "qwen8b_minimalist"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

K_LATENT = int((CFG["train"].get("k_factors") or [16])[0])
MODEL_NAME = "meta_hybrid_irt_kfactor_gated_mlp"

model_cfg = ModelConfig(
    k=K_LATENT,
    item_embed_dim=ITEM_EMB_DIM,
    item_map_hidden_dim=int(CFG["train"].get("item_map_hidden_dim", 512)),
    residual_hidden_dim=int(CFG["train"].get("residual_hidden_dim", 256)),
    dropout=float(CFG["train"].get("dropout", 0.1)),
    n_subjects=indexer.n_subjects,
    n_benchmark_conditions=indexer.n_bc,
    use_subject_text_embedding=False,
    subject_embed_dim=0,
    lambda_resid_init=float(CFG["train"].get("lambda_resid_init", 0.1)),
    lambda_resid_trainable=bool(CFG["train"].get("lambda_resid_trainable", True)),
    use_pool_features=True,
    pool_feature_dim=len(POOL_FEATURE_NAMES_EXT),
    use_cluster_features=True,
    n_clusters=int(CFG["clustering"]["k"]),
    cluster_embed_dim=int(CFG["item_features"].get("cluster_embed_dim", 16)),
    use_judge_features=False,
    judge_feature_dim=0,
    use_nn_features=True,
    nn_feature_dim=int(nn_train_mat.shape[1]),
    use_metadata_features=True,
    meta_subject_categorical=_meta_schema.subject_categorical,
    meta_subject_numeric=_meta_schema.subject_numeric,
    meta_benchmark_categorical=_meta_schema.benchmark_categorical,
    meta_benchmark_numeric=_meta_schema.benchmark_numeric,
    meta_explicit_crosses=_meta_schema.explicit_crosses,
)
print("ModelConfig:")
print(json.dumps(asdict(model_cfg), default=str, indent=2)[:1200])

# Monkey-patch ``build_model`` so train_one's internal model construction
# also gets the metadata id tables attached. The trainer doesn't know
# about meta tables, so without this hook the model trains with a
# zero-init metadata channel -- behavior identical to the non-metadata
# baseline. We restore the original in a finally block.
_orig_build_model = train_mod.build_model


def _build_with_meta(name, cfg):
    m = _orig_build_model(name, cfg)
    if name == MODEL_NAME and bool(getattr(cfg, "use_metadata_features", False)):
        m.attach_metadata_tables(meta_id_tables)
    return m


train_mod.build_model = _build_with_meta
try:
    train_cfg = TrainConfig(
        learning_rate=float(CFG["train"].get("learning_rate", 3.0e-3)),
        weight_decay=float(CFG["train"].get("weight_decay", 1.0e-4)),
        batch_size=int(CFG["train"].get("batch_size", 65536)),
        epochs=int(CFG["train"].get("epochs", 5)),
        warmup_steps=int(CFG["train"].get("warmup_steps", 30)),
        scheduler=str(CFG["train"].get("scheduler", "cosine")),
        grad_clip=float(CFG["train"].get("grad_clip", 1.0)),
        early_stopping_patience=int(CFG["train"].get("early_stopping_patience", 5)),
        bf16=bool(CFG["encoder"].get("bf16", True)),
        num_workers=int(CFG["train"].get("num_workers", 0)),
    )
    result = train_one(
        model_name=MODEL_NAME,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        indexer=indexer,
        seed=SEED,
        run_id="qwen8b_minimalist",
        checkpoint_dir=CKPT_DIR,
        extra_metadata={
            "encoder_model_id": CFG["encoder"]["model_id"],
            "use_metadata_features": True,
            "use_nn_features": True,
            "centroid_distances_top_m": int(top_m),
        },
    )
finally:
    train_mod.build_model = _orig_build_model

print(
    f"\nbest val log-loss: {result.best_val_log_loss:.6f}  "
    f"brier: {result.best_val_brier:.6f}  "
    f"epoch_best: {result.epoch_best}"
)

# %% [markdown]
# ## 9. Score uncalibrated train + val, fit the NN calibrator
#
# The runtime NN calibrator needs `(subject_id, training_item_row) ->
# (label, p_uncal)` for every training pair. We re-run the trained
# model over the training set in inference mode to record `p_uncal`,
# then fit the shrinkage `alpha` on the validation split.

# %%
import torch
from tqdm.auto import tqdm

from src.models import build_model as _build_model_for_inf
from src.nn_calibration import (
    NNCalibrator,
    SubjectResidualTable,
)


def _score_dataset(ds: LookupDataset, model, batch_size: int = 8192) -> np.ndarray:
    device = next(model.parameters()).device
    n = len(ds)
    out = np.zeros(n, dtype=np.float32)
    has_pool = ds.pool_feats.shape[-1] > 0
    has_cluster = bool(getattr(model.cfg, "has_cluster_embedding", False))
    has_nn = ds.nn_feats.shape[-1] > 0
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="score", leave=False):
            end = min(start + batch_size, n)
            kw = {}
            if has_pool:
                kw["pool_feats"] = ds.pool_feats[start:end].to(device)
            if has_cluster:
                kw["cluster_ids"] = ds.cluster_ids[start:end].to(device)
            if has_nn:
                kw["nn_feats"] = ds.nn_feats[start:end].to(device)
            logits = model(
                subject_idx=ds.subject_ids[start:end].to(device),
                bc_idx=ds.bc_ids[start:end].to(device),
                item_emb=ds.item_emb[start:end].to(device),
                subject_emb=None,
                **kw,
            )
            out[start:end] = torch.sigmoid(logits).detach().cpu().numpy()
    return out


device = "cuda" if torch.cuda.is_available() else "cpu"
trained = _build_model_for_inf(MODEL_NAME, model_cfg)
# attach_metadata_tables rebuilds the metadata embedding modules with
# the correct cardinalities, so we MUST attach BEFORE loading the
# checkpoint (otherwise the newly-built tables overwrite the trained
# weights with random init).
trained.attach_metadata_tables(meta_id_tables)
ckpt = torch.load(result.checkpoint_path, map_location=device, weights_only=False)
trained.load_state_dict(ckpt["model_state"])
trained = trained.to(device).eval()

print("Scoring train rows for residual table...")
p_uncal_train = _score_dataset(train_ds, trained)
print("Scoring val rows for calibrator fit...")
p_uncal_val = _score_dataset(val_ds, trained)

# (subject_id, training_item_row, label, p_uncal) -> CSR table.
key_to_train_row = {k: i for i, k in enumerate(train_item_keys)}
train_subj_ids = np.array(
    [indexer.subject_id(str(s)) for s in primary.train["subject_key"]],
    dtype=np.int64,
)
train_item_rows = np.array(
    [key_to_train_row.get(str(k), -1) for k in primary.train["item_key"]],
    dtype=np.int64,
)
ok = train_item_rows >= 0
residual_table = SubjectResidualTable.from_rows(
    subject_ids=train_subj_ids[ok],
    training_item_rows=train_item_rows[ok],
    labels=primary.train["label"].astype(float).to_numpy()[ok],
    uncal_probs=p_uncal_train[ok],
    n_subjects=indexer.n_subjects,
    n_training_items=len(train_item_keys),
)

val_subj_ids_np = np.array(
    [indexer.subject_id(str(s)) for s in primary.val["subject_key"]],
    dtype=np.int64,
)

# Same dedupe + chunked search trick as the NN feature build above:
# stack per-unique-item embeddings only, in NN_QUERY_CHUNK chunks, then
# expand back to per-row. Avoids the [N_val, D] peak that would crash
# the kernel on Qwen3-Embedding-8B.
val_keys_arr = np.asarray([str(k) for k in primary.val["item_key"]])
val_unique_keys, val_inverse = np.unique(val_keys_arr, return_inverse=True)
K_CAL = int(CFG["nn_calibration"]["k"])
val_uniq_idx = np.empty((len(val_unique_keys), K_CAL), dtype=np.int64)
val_uniq_sims = np.empty((len(val_unique_keys), K_CAL), dtype=np.float32)
for _s in range(0, len(val_unique_keys), NN_QUERY_CHUNK):
    _e = min(_s + NN_QUERY_CHUNK, len(val_unique_keys))
    _chunk_keys = list(val_unique_keys[_s:_e])
    _chunk_emb = np.stack(
        [item_emb_lookup[k] for k in _chunk_keys], axis=0
    ).astype(np.float32, copy=False)
    _idx, _sims = nn_index.nearest(
        _chunk_emb, k=K_CAL, exclude_self=False, query_keys=_chunk_keys,
    )
    val_uniq_idx[_s:_e] = _idx
    val_uniq_sims[_s:_e] = _sims
    del _chunk_emb, _idx, _sims
val_neighbor_rows = val_uniq_idx[val_inverse]
val_neighbor_sims = val_uniq_sims[val_inverse]

calibrator = NNCalibrator.fit_alpha_on_val(
    residual_table=residual_table,
    val_subject_ids=val_subj_ids_np,
    val_neighbor_rows=val_neighbor_rows,
    val_neighbor_sims=val_neighbor_sims,
    val_uncal_probs=p_uncal_val,
    val_labels=primary.val["label"].astype(float).to_numpy(),
    k=int(CFG["nn_calibration"]["k"]),
    similarity=str(CFG["nn_calibration"].get("similarity", "cosine")),
    temperature=float(CFG["nn_calibration"].get("temperature", 1.0)),
    min_weight_sum=float(CFG["nn_calibration"].get("min_weight_sum", 1e-3)),
)
print("Calibrator state:", calibrator.state)

RESIDUAL_DIR = ROOT / "artifacts" / "nn_calibration"
RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
residual_table.save(RESIDUAL_DIR)
print(f"Residual table saved to {RESIDUAL_DIR}")

# %% [markdown]
# ## 10. Export submission (with NN cache + NN calibrator)

# %%
from src.export_submission import (
    bundle_training_cache,
    compute_train_counts,
    export_run,
    make_submission_zip,
)

SUBMISSION_DIR = ROOT / "submission" / "qwen8b_minimalist"
TRAINING_CACHE_DIR = ROOT / "artifacts" / "training_cache"

training_cache_result = bundle_training_cache(
    items_parquet_path=embedder.items_path,
    out_dir=TRAINING_CACHE_DIR,
    submission_cache_cfg=CFG["submission_cache"],
    encoder_cfg=CFG["encoder"],
    items_meta_df=item_df,
    cluster_assignments=cluster_assignments,
    n_clusters=int(CFG["clustering"]["k"]),
    train_df=primary.train,
    nn_features_cfg=nn_cfg.to_dict(),
    subject_to_id=indexer.subject_to_id,
)
print(
    f"Training cache size: {training_cache_result.size_mb:.1f} MB  "
    f"(soft cap {CFG['submission_cache'].get('max_bundle_size_mb', 200)} MB)"
)

sub_dir = export_run(
    result=result,
    encoder_cfg=CFG["encoder"],
    submission_dir=SUBMISSION_DIR,
    include_labeling=True,
    pool_stats_path=POOL_STATS_PATH,
    cluster_centroids_path=CENTROIDS_PATH,
    pool_feature_names=list(POOL_FEATURE_NAMES_EXT),
    training_cache_dir=TRAINING_CACHE_DIR,
    judge_cfg=None,
    nn_features_cfg=nn_cfg.to_dict(),
    ship_training_cache=True,
    ship_requirements_txt=False,
    train_counts=compute_train_counts(primary.train),
    meta_preprocessor=meta_preprocessor,
    nn_calibrator_state=calibrator.to_dict(),
    nn_calibrator_table_dir=RESIDUAL_DIR,
)
print(f"Submission dir: {sub_dir}")

zip_cap_mb = float(CFG["submission"].get("max_zip_size_mb", 70))
zip_path = make_submission_zip(
    sub_dir,
    zip_path=sub_dir.with_suffix(".zip"),
    max_zip_size_mb=zip_cap_mb,
)
size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"Zip: {zip_path}  size: {size_mb:.1f} MB  (cap {zip_cap_mb:.0f} MB)")

if IN_COLAB:
    try:
        from google.colab import files  # type: ignore
        files.download(str(zip_path))
    except Exception as exc:
        print("Auto-download failed (%s); copy the zip manually." % exc)
