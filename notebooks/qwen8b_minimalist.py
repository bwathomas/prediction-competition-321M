# %% [markdown]
# # Qwen3-Embedding-8B minimalist notebook -- coverage-blend ensemble
#
# **Goal**: train two heads on cached `Qwen/Qwen3-Embedding-8B` vectors --
#
#   * **Model A**: metadata-aware (`meta_hybrid_irt_kfactor_gated_mlp`)
#     trained with **metadata dropout** (per-row Bernoulli masking of
#     subject and benchmark metadata so the `__MISSING__` embeddings
#     and the missingness-gate get gradient signal).
#   * **Model B**: no-metadata (`hybrid_irt_kfactor_gated_mlp`) trained
#     with **bc-id dropout** (per-row Bernoulli replacement of `bc_idx`
#     with the UNK slot so the residual MLP learns to predict without
#     the per-benchmark bias signal).
#
# After training, fit a **coverage-conditional blend** on a synthetic
# benchmark-cold-start val slice: two scalars `w_present` and
# `w_missing` such that the runtime blends per-row by whether the
# row's benchmark is in the trained indexer. Fit a Netflix-Prize-style
# NN calibrator on the blended predictions, and export a Codabench-
# ready ensemble bundle that ships both checkpoints + both weight
# vectors.
#
# **Why blend with coverage gating?** On hosted test rows for unseen
# benchmarks, Model A's metadata gate sees `__MISSING__` patterns it
# has been explicitly trained on (good), and Model A's per-bc bias
# `beta[0]` is the well-trained UNK slot (good); Model B is trained
# specifically to predict robustly without per-bc bias (good). Either
# model alone is suboptimal: A's metadata channel can dominate when
# it shouldn't, B has no access to subject/benchmark metadata signal
# even when the test row HAS that metadata. The blend lets each
# model contribute where it's strongest. Math: Jensen's slack from
# blending is `(1/2) * w * (1-w) * Var(p_A - p_B) / (p * (1-p))`,
# maximized when `w_A` matches the relative reliability per regime.
#
# What ships in the bundle:
#
# - `artifacts/checkpoint.pt` -- ensemble bundle with both members'
#   state dicts and the per-coverage blend weights. Loaded by the
#   runtime via the existing `_EnsembleModel` path
#   (heterogeneous meta+nometa members supported).
# - `artifacts/cluster_centroids.npy` -- runtime computes top-m
#   centroid distances on every prediction.
# - `artifacts/pool_features_stats.json` -- z-score stats including
#   the `centroid_dist_*` columns.
# - `artifacts/meta_preprocessor.json` -- cold-start subject /
#   benchmark metadata encoder, used by Member A's `meta_override`.
# - `artifacts/runtime_meta.json` -- includes `nn_calibrator` block
#   (alpha, k, similarity, ...) AND `ensemble.blend_weights[_missing]`
#   for diagnostic visibility.
# - `cache/` -- training-item cache (PCA + int8 + FAISS) for runtime
#   NN feature lookup. **Required** when NN features are on.
# - `cache/nn_residual/` -- `(subject_id, training_item_row) ->
#   (label, p_uncal)` sparse table for the NN calibrator. The
#   `p_uncal` column here is the BLENDED uncalibrated prediction
#   (so the calibrator's residuals are taken w.r.t. what the runtime
#   actually emits before NN calibration).
#
# Design rationale (dropout rates, seeds, blend tuning) is documented
# at the top of each cell below.

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

# --- Coverage-blend ensemble: two models trained with complementary
#     dropouts, blended per-row by benchmark-coverage at runtime.
#
#  Why these specific defaults?
#
#  * Model A's ``p_bench=0.20`` matches a defensible upper bound on the
#    test-time fraction of cold-start benchmarks (the hosted comp's
#    leaderboard is heavily new-benchmark). Train-time mask rate ≈
#    test-time mask rate is the bias-variance optimum: too low and the
#    MISSING embedding stays under-trained; too high and the in-
#    distribution metadata signal is starved.
#
#  * Model A's ``p_subj=0.10`` is half of p_bench because subject-side
#    cold-start is rarer than benchmark-side (the comp re-uses many
#    model orgs across benchmarks). Independence between the two flags
#    (no joint-mask dependency) means we get joint-missing rows at the
#    product probability ``0.20 * 0.10 = 0.02`` -- enough to let the
#    gate learn the worst-case "no metadata at all" pattern but not
#    enough to dominate.
#
#  * Model B's ``q_bc=0.15`` regularizes the per-benchmark bias
#    ``beta[bc_idx]``. With prob 0.15 we replace ``bc_idx`` with 0
#    (the UNK slot), so the residual MLP and the IRT channel learn to
#    predict robustly even when the per-bc intercept signal is
#    unavailable. 15% is conservative: any higher and we starve the
#    in-distribution bc bias; any lower and ``beta[0]`` stays under-
#    trained for hosted cold-start rows.
#
#  Seeding strategy (also a deliberate choice):
#
#  * SPLIT_SEED is shared across both models so the val rows are
#    identical -- needed for blend tuning to be on a comparable slice.
#  * MODEL_A_SEED and MODEL_B_SEED differ from SEED and from each
#    other: different inits + batch order + dropout RNG = larger
#    ``Var(p_A - p_B)`` = larger Jensen slack from blending. Empirical
#    ensembling literature consistently reports ~1-3% log-loss gain
#    per seed-diversification on top of architectural diversity.
CFG["coverage_blend"] = {
    "model_a": {
        "p_bench": 0.20,
        "p_subj": 0.10,
        "q_bc": 0.0,
    },
    "model_b": {
        "p_bench": 0.0,
        "p_subj": 0.0,
        "q_bc": 0.15,
    },
    "synthetic_cold_start_frac": 0.15,
    "model_a_seed_offset": 11,
    "model_b_seed_offset": 23,
}

print("Encoder:", CFG["encoder"]["model_id"])
print("Model variant:", CFG["train"]["models"])
print("Metadata enabled:", CFG["metadata"]["enabled"])
print("NN features:", CFG["nn_features"]["enabled"], "k=", CFG["nn_features"]["k"])
print("Centroid distances:", CFG["centroid_distances"]["enabled"],
      "top_m=", CFG["centroid_distances"]["top_m"])
print("Judge:", CFG["judge"]["enabled"])
print(
    "Coverage blend: A(p_bench={p_bench}, p_subj={p_subj}, q_bc={q_bc}), "
    "B(q_bc={q_bc_b})  synth cold-start={cs:.0%}".format(
        p_bench=CFG["coverage_blend"]["model_a"]["p_bench"],
        p_subj=CFG["coverage_blend"]["model_a"]["p_subj"],
        q_bc=CFG["coverage_blend"]["model_a"]["q_bc"],
        q_bc_b=CFG["coverage_blend"]["model_b"]["q_bc"],
        cs=CFG["coverage_blend"]["synthetic_cold_start_frac"],
    )
)

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
# ## 8a. Train Model A: metadata-aware + metadata dropout
#
# **Why metadata dropout, not vanilla training?** Model A's
# `MetaHybridIRTKFactorGatedMLP` reserves row 0 of every metadata
# buffer for the all-`__MISSING__` pattern. Without dropout, every
# training row uses real metadata, so row 0's embedding receives ZERO
# gradients during training. At test time on a cold-start benchmark
# the runtime substitutes row 0 via `meta_override`, the model sees
# a never-trained embedding, and the metadata channel emits
# essentially-random logits. Metadata dropout fixes this: with prob
# `p_bench` we replace bench metadata with row 0 during the forward
# pass, so the row 0 embedding (and the missingness gate) get gradient
# signal proportional to expected test-time exposure.
#
# **Implementation**: a forward-pre-hook installed on the model. The
# hook is no-op during eval (so val metrics are honest) and seeds
# its RNG from `MODEL_A_SEED` so every training run is reproducible
# bit-for-bit.

# %%
from dataclasses import asdict

import src.train as train_mod
from src.models import ModelConfig
from src.train import TrainConfig, train_one
from src.train_dropout import TrainDropoutConfig, install_train_dropout

CKPT_DIR = ROOT / "artifacts" / "checkpoints" / "qwen8b_minimalist"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

K_LATENT = int((CFG["train"].get("k_factors") or [16])[0])
MODEL_A_NAME = "meta_hybrid_irt_kfactor_gated_mlp"
MODEL_B_NAME = "hybrid_irt_kfactor_gated_mlp"

SPLIT_SEED = SEED
MODEL_A_SEED = SEED + int(CFG["coverage_blend"]["model_a_seed_offset"])
MODEL_B_SEED = SEED + int(CFG["coverage_blend"]["model_b_seed_offset"])
print(
    f"Seeds: SPLIT={SPLIT_SEED}  MODEL_A={MODEL_A_SEED}  "
    f"MODEL_B={MODEL_B_SEED}"
)

# Common ModelConfig fields shared between A and B. Only the metadata
# block + the model variant differs across the two members.
_common_kwargs = dict(
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
)

model_a_cfg = ModelConfig(
    use_metadata_features=True,
    meta_subject_categorical=_meta_schema.subject_categorical,
    meta_subject_numeric=_meta_schema.subject_numeric,
    meta_benchmark_categorical=_meta_schema.benchmark_categorical,
    meta_benchmark_numeric=_meta_schema.benchmark_numeric,
    meta_explicit_crosses=_meta_schema.explicit_crosses,
    **_common_kwargs,
)
print("Model A (meta + meta-dropout):")
print(json.dumps(asdict(model_a_cfg), default=str, indent=2)[:1200])

# Build TrainConfig once -- reused for both members so any wall-clock
# comparison between A and B reflects only the architecture / dropout
# delta, not optimizer hyperparameters.
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

# Monkey-patch ``build_model`` so train_one's internal model
# construction also gets:
#   1. the metadata id tables attached (for Model A)
#   2. the dropout pre-hook installed (for whichever model we're training)
# We track the active dropout config in a closure so the patch can be
# parametric. Restored in finally blocks.
_orig_build_model = train_mod.build_model
_active_dropout_cfg: dict = {"cfg": None, "name": None, "installed_handles": []}


def _build_with_overrides(name, cfg):
    m = _orig_build_model(name, cfg)
    if name == MODEL_A_NAME and bool(getattr(cfg, "use_metadata_features", False)):
        m.attach_metadata_tables(meta_id_tables)
    drop_cfg = _active_dropout_cfg["cfg"]
    if drop_cfg is not None and name == _active_dropout_cfg["name"]:
        h = install_train_dropout(m, drop_cfg)
        _active_dropout_cfg["installed_handles"].append(h)
    return m


train_mod.build_model = _build_with_overrides
try:
    a_drop = TrainDropoutConfig(
        p_bench=float(CFG["coverage_blend"]["model_a"]["p_bench"]),
        p_subj=float(CFG["coverage_blend"]["model_a"]["p_subj"]),
        q_bc=float(CFG["coverage_blend"]["model_a"]["q_bc"]),
        seed=MODEL_A_SEED,
    )
    _active_dropout_cfg["cfg"] = a_drop
    _active_dropout_cfg["name"] = MODEL_A_NAME
    _active_dropout_cfg["installed_handles"] = []
    result_a = train_one(
        model_name=MODEL_A_NAME,
        model_cfg=model_a_cfg,
        train_cfg=train_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        indexer=indexer,
        seed=MODEL_A_SEED,
        run_id="qwen8b_minimalist_A_meta_dropout",
        checkpoint_dir=CKPT_DIR,
        extra_metadata={
            "encoder_model_id": CFG["encoder"]["model_id"],
            "use_metadata_features": True,
            "use_nn_features": True,
            "centroid_distances_top_m": int(top_m),
            "meta_dropout": asdict(a_drop),
        },
    )
    for h in _active_dropout_cfg["installed_handles"]:
        print(
            f"  Model A dropout stats: train_calls={h.n_train_calls}  "
            f"rows={h.n_rows_seen}  bench_masked={h.n_rows_bench_masked}  "
            f"subj_masked={h.n_rows_subj_masked}  "
            f"bc_idx_masked={h.n_rows_bc_idx_masked}"
        )
        h.remove()
finally:
    _active_dropout_cfg["cfg"] = None
    _active_dropout_cfg["name"] = None

print(
    f"\nModel A best val log-loss: {result_a.best_val_log_loss:.6f}  "
    f"brier: {result_a.best_val_brier:.6f}  "
    f"epoch_best: {result_a.epoch_best}"
)

# %% [markdown]
# ## 8b. Train Model B: no-metadata + benchmark-id dropout
#
# **Why bc-idx dropout for the no-metadata model?** Model B's
# `HybridIRTItemKFactorGatedMLP` predicts via
# `mu + beta[bc_idx] + alpha_i*(theta - beta_i) + factor + residual`.
# At hosted-test time on a cold-start benchmark, `bc_idx=0` (the UNK
# slot), so `beta[0]` is the only piece of the per-bc bias channel
# the runtime sees. Without dropout `beta[0]` receives no gradient
# during training (no row uses bc_idx=0), so at test time it returns
# whatever the random initializer gave us, plus the residual MLP has
# never had to predict without a per-bc bias signal. With prob
# `q_bc=0.15` we replace `bc_idx` with 0 during the forward pass,
# training `beta[0]` to be a useful "average benchmark" prior and
# forcing the MLP to learn bias-free predictions.
#
# **Why no metadata at all?** Model B's role in the blend is to be
# the "metadata-free fallback". Even if we COULD wire metadata into
# this model variant, doing so would defeat the purpose of having
# two complementary members. The blend math (Jensen's slack) is
# maximized when `Var(p_A - p_B)` is large, which requires Model B
# to use a different feature subset.

# %%
model_b_cfg = ModelConfig(
    use_metadata_features=False,
    **_common_kwargs,
)
print("Model B (no_meta + bc-dropout):")
print(json.dumps(asdict(model_b_cfg), default=str, indent=2)[:1200])

train_mod.build_model = _build_with_overrides
try:
    b_drop = TrainDropoutConfig(
        p_bench=float(CFG["coverage_blend"]["model_b"]["p_bench"]),
        p_subj=float(CFG["coverage_blend"]["model_b"]["p_subj"]),
        q_bc=float(CFG["coverage_blend"]["model_b"]["q_bc"]),
        seed=MODEL_B_SEED,
    )
    _active_dropout_cfg["cfg"] = b_drop
    _active_dropout_cfg["name"] = MODEL_B_NAME
    _active_dropout_cfg["installed_handles"] = []
    result_b = train_one(
        model_name=MODEL_B_NAME,
        model_cfg=model_b_cfg,
        train_cfg=train_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        indexer=indexer,
        seed=MODEL_B_SEED,
        run_id="qwen8b_minimalist_B_no_meta_bc_dropout",
        checkpoint_dir=CKPT_DIR,
        extra_metadata={
            "encoder_model_id": CFG["encoder"]["model_id"],
            "use_metadata_features": False,
            "use_nn_features": True,
            "centroid_distances_top_m": int(top_m),
            "bc_dropout": asdict(b_drop),
        },
    )
    for h in _active_dropout_cfg["installed_handles"]:
        print(
            f"  Model B dropout stats: train_calls={h.n_train_calls}  "
            f"rows={h.n_rows_seen}  bc_idx_masked={h.n_rows_bc_idx_masked}"
        )
        h.remove()
finally:
    _active_dropout_cfg["cfg"] = None
    _active_dropout_cfg["name"] = None
    train_mod.build_model = _orig_build_model

print(
    f"\nModel B best val log-loss: {result_b.best_val_log_loss:.6f}  "
    f"brier: {result_b.best_val_brier:.6f}  "
    f"epoch_best: {result_b.epoch_best}"
)

# %% [markdown]
# ## 9. Score both models + fit per-coverage blend weights
#
# **Blend tuning needs a benchmark-cold-start val signal.** Our val
# split is item-cold-start (item-disjoint from train), but every val
# row's benchmark IS in `_BC_TO_ID`. Naive blend tuning would push
# `w_present == w_missing` because we never measure log-loss on
# rows where `bench_present=0`.
#
# We synthesize cold-start at scoring time: pick a random
# `synthetic_cold_start_frac` of val benchmarks, and for those
# benchmarks' val rows force `bc_idx -> 0` AND `meta_override`
# to all-MISSING when scoring Model A. The model has actually seen
# those benchmarks during training, so beta[that_bc] is well-tuned;
# at scoring time we strip both signals to simulate the test-time
# regime. Remaining (1 - frac) of val rows use real bc_idx + real
# metadata. Now we have two predictions per row + a `bench_present`
# flag, and can fit `w_present` on present rows + `w_missing` on
# missing rows independently.
#
# Each is a 1D convex log-loss minimization. We solve it via golden-
# section search over [0, 1].

# %%
import torch
from tqdm.auto import tqdm

from src.models import build_model as _build_model_for_inf

device = "cuda" if torch.cuda.is_available() else "cpu"


def _score_dataset(
    ds: LookupDataset,
    model,
    *,
    bc_override: torch.Tensor | None = None,
    meta_override_template: dict | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run ``model`` over ``ds`` and return per-row uncalibrated
    probabilities.

    ``bc_override`` (when not None) is a (N,) int64 tensor that
    replaces ``ds.bc_ids`` for this scoring pass -- used to force
    `bc_idx -> 0` on the synthetic cold-start slice.

    ``meta_override_template`` (when not None) is a dict with the
    same keys as ``MetadataIdTables.row(0)``; the same MISSING row
    is broadcast to every row of every batch. Only consumed by
    metadata-aware models.
    """
    n = len(ds)
    out = np.zeros(n, dtype=np.float32)
    has_pool = ds.pool_feats.shape[-1] > 0
    has_cluster = bool(getattr(model.cfg, "has_cluster_embedding", False))
    has_nn = ds.nn_feats.shape[-1] > 0
    accepts_meta = bool(getattr(model.cfg, "use_metadata_features", False))
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
            if accepts_meta and meta_override_template is not None:
                B_chunk = end - start
                kw["meta_override"] = {
                    k: v.expand(B_chunk, -1).to(device).contiguous()
                    for k, v in meta_override_template.items()
                }
            bc_chunk = (
                bc_override[start:end] if bc_override is not None
                else ds.bc_ids[start:end]
            )
            logits = model(
                subject_idx=ds.subject_ids[start:end].to(device),
                bc_idx=bc_chunk.to(device),
                item_emb=ds.item_emb[start:end].to(device),
                subject_emb=None,
                **kw,
            )
            out[start:end] = torch.sigmoid(logits).detach().cpu().numpy()
    return out


# Re-instantiate Model A and load its weights. attach_metadata_tables
# rebuilds the embedding modules with the right cardinalities, so we
# MUST attach BEFORE loading the checkpoint.
trained_a = _build_model_for_inf(MODEL_A_NAME, model_a_cfg)
trained_a.attach_metadata_tables(meta_id_tables)
ckpt_a = torch.load(result_a.checkpoint_path, map_location=device, weights_only=False)
trained_a.load_state_dict(ckpt_a["model_state"])
trained_a = trained_a.to(device).eval()

trained_b = _build_model_for_inf(MODEL_B_NAME, model_b_cfg)
ckpt_b = torch.load(result_b.checkpoint_path, map_location=device, weights_only=False)
trained_b.load_state_dict(ckpt_b["model_state"])
trained_b = trained_b.to(device).eval()

# Pick the synthetic cold-start subset: a deterministic random
# selection of val benchmarks. Reproducible via SPLIT_SEED so re-runs
# of this cell hit the same partition.
val_bc_keys = primary.val["benchmark_condition_key"].astype(str).to_numpy()
unique_val_bc = np.unique(val_bc_keys)
_rng = np.random.default_rng(SPLIT_SEED + 7919)
n_synth = max(
    1,
    int(round(
        float(CFG["coverage_blend"]["synthetic_cold_start_frac"])
        * len(unique_val_bc)
    )),
)
synth_bc_set = set(_rng.choice(unique_val_bc, size=n_synth, replace=False).tolist())
synth_mask = np.array(
    [k in synth_bc_set for k in val_bc_keys],
    dtype=bool,
)
print(
    f"Synthetic cold-start: held out {len(synth_bc_set):,} of "
    f"{len(unique_val_bc):,} val benchmarks  "
    f"({synth_mask.sum():,}/{len(synth_mask):,} val rows masked)"
)
bench_present_val = (~synth_mask).astype(np.float32)

# Build the bc_override and meta_override for the synthetic-cold rows.
# For non-synth rows we keep the real bc_idx and let the model do
# its normal buffer lookup (no meta override).
val_bc_synth = val_ds.bc_ids.clone()
val_bc_synth[torch.from_numpy(synth_mask)] = 0
miss_row = {
    "subj_cat": meta_id_tables.subject_cat_ids[0:1].clone(),
    "subj_num": meta_id_tables.subject_num[0:1].clone(),
    "bc_cat": meta_id_tables.bc_cat_ids[0:1].clone(),
    "bc_num": meta_id_tables.bc_num[0:1].clone(),
}

# We need 4 score passes for blend tuning:
#   p_a_present = Model A on (real bc, real meta)              -> rows where bench_present=1
#   p_a_missing = Model A on (bc=0, meta=MISSING)              -> rows where bench_present=0
#   p_b_present = Model B on (real bc)                          -> rows where bench_present=1
#   p_b_missing = Model B on (bc=0)                             -> rows where bench_present=0
# For each model we do TWO passes (one all-real, one all-missing) and
# splice into a single per-row vector via synth_mask.
print("Scoring Model A: real-meta pass...")
p_a_real = _score_dataset(val_ds, trained_a)
print("Scoring Model A: cold-start (bc=0, meta=MISSING) pass...")
p_a_cold = _score_dataset(
    val_ds, trained_a, bc_override=val_bc_synth, meta_override_template=miss_row
)
p_a_val = np.where(synth_mask, p_a_cold, p_a_real)

print("Scoring Model B: real-bc pass...")
p_b_real = _score_dataset(val_ds, trained_b)
print("Scoring Model B: cold-start (bc=0) pass...")
p_b_cold = _score_dataset(val_ds, trained_b, bc_override=val_bc_synth)
p_b_val = np.where(synth_mask, p_b_cold, p_b_real)

ylab_val = primary.val["label"].astype(float).to_numpy()


def _logloss_blend(w: float, p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> float:
    """Log-loss of ``w * p_a + (1 - w) * p_b`` clipped to safe range."""
    eps = 1e-7
    p = np.clip(w * p_a + (1.0 - w) * p_b, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _golden_section_min(f, lo: float, hi: float, tol: float = 1e-4, max_iter: int = 64) -> float:
    """Golden-section search on a unimodal scalar f over [lo, hi]."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return float((a + b) / 2.0)


# Fit the two weights on disjoint slices of the val set.
present_idx = np.where(~synth_mask)[0]
missing_idx = np.where(synth_mask)[0]

if len(present_idx) > 0:
    w_present = _golden_section_min(
        lambda w: _logloss_blend(
            w, p_a_val[present_idx], p_b_val[present_idx], ylab_val[present_idx]
        ),
        0.0, 1.0,
    )
else:
    w_present = 0.5
if len(missing_idx) > 0:
    w_missing = _golden_section_min(
        lambda w: _logloss_blend(
            w, p_a_val[missing_idx], p_b_val[missing_idx], ylab_val[missing_idx]
        ),
        0.0, 1.0,
    )
else:
    w_missing = 0.5

ll_a_p = _logloss_blend(1.0, p_a_val[present_idx], p_b_val[present_idx], ylab_val[present_idx]) if len(present_idx) else float("nan")
ll_b_p = _logloss_blend(0.0, p_a_val[present_idx], p_b_val[present_idx], ylab_val[present_idx]) if len(present_idx) else float("nan")
ll_blend_p = _logloss_blend(w_present, p_a_val[present_idx], p_b_val[present_idx], ylab_val[present_idx]) if len(present_idx) else float("nan")
ll_a_m = _logloss_blend(1.0, p_a_val[missing_idx], p_b_val[missing_idx], ylab_val[missing_idx]) if len(missing_idx) else float("nan")
ll_b_m = _logloss_blend(0.0, p_a_val[missing_idx], p_b_val[missing_idx], ylab_val[missing_idx]) if len(missing_idx) else float("nan")
ll_blend_m = _logloss_blend(w_missing, p_a_val[missing_idx], p_b_val[missing_idx], ylab_val[missing_idx]) if len(missing_idx) else float("nan")
print(
    f"Blend weights:\n"
    f"  bench_present=1: w_A={w_present:.3f}  "
    f"ll_A={ll_a_p:.5f}  ll_B={ll_b_p:.5f}  ll_blend={ll_blend_p:.5f}\n"
    f"  bench_present=0: w_A={w_missing:.3f}  "
    f"ll_A={ll_a_m:.5f}  ll_B={ll_b_m:.5f}  ll_blend={ll_blend_m:.5f}"
)

BLEND_PRESENT = (float(w_present), float(1.0 - w_present))
BLEND_MISSING = (float(w_missing), float(1.0 - w_missing))

# Per-row blended val prediction (for the NN calibrator below).
w_per_row = np.where(synth_mask, w_missing, w_present).astype(np.float32)
p_blend_val = w_per_row * p_a_val + (1.0 - w_per_row) * p_b_val

# %% [markdown]
# ## 10. NN calibrator on the BLENDED predictions
#
# The runtime applies the NN calibrator AFTER the per-row blend, so
# the residual table must store residuals computed against the same
# blended `p_uncal` the runtime emits. We score Model A and Model B
# on TRAIN (no synthetic cold-start: train-time we always use real
# bc + real meta) and combine them with `w_present` (because train
# rows are by construction NOT cold-start). The val side reuses the
# blended predictions computed in cell 9.

# %%
from src.nn_calibration import NNCalibrator, SubjectResidualTable

print("Scoring Model A on train (real meta)...")
p_a_train = _score_dataset(train_ds, trained_a)
print("Scoring Model B on train (real bc)...")
p_b_train = _score_dataset(train_ds, trained_b)
p_uncal_train_blend = (
    BLEND_PRESENT[0] * p_a_train + BLEND_PRESENT[1] * p_b_train
).astype(np.float32)

# Build the residual table from blended train predictions.
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
    uncal_probs=p_uncal_train_blend[ok],
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
    val_uncal_probs=p_blend_val,
    val_labels=ylab_val,
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
# ## 11. Export coverage-blend ensemble bundle
#
# Single-call wrapper that packages both members + both blend weight
# vectors into a Codabench-ready zip. The runtime's `_EnsembleModel`
# loads the bundle, introspects each member's `forward` signature
# to decide whether to forward `meta_override`, and uses the per-row
# `bench_present` flag to route to `blend_weights` (present) or
# `blend_weights_missing` (missing).

# %%
from src.export_submission import (
    bundle_training_cache,
    compute_train_counts,
    export_coverage_blend_run,
    make_submission_zip,
)

SUBMISSION_DIR = ROOT / "submission" / "qwen8b_minimalist_coverage_blend"
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

sub_dir = export_coverage_blend_run(
    member_a_state_dict=ckpt_a["model_state"],
    member_a_model_name=MODEL_A_NAME,
    member_a_model_cfg=asdict(model_a_cfg),
    member_a_config_id="meta_dropout",
    member_b_state_dict=ckpt_b["model_state"],
    member_b_model_name=MODEL_B_NAME,
    member_b_model_cfg=asdict(model_b_cfg),
    member_b_config_id="no_meta_bc_dropout",
    blend_weights_present=BLEND_PRESENT,
    blend_weights_missing=BLEND_MISSING,
    indexer={
        "subject_to_id": dict(indexer.subject_to_id),
        "bc_to_id": dict(indexer.bc_to_id),
    },
    encoder_cfg=CFG["encoder"],
    submission_dir=SUBMISSION_DIR,
    representative_result=result_a,  # for runtime_meta cosmetics
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
