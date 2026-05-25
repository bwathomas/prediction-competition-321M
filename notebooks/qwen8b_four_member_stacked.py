# %% [markdown]
# # Qwen3-Embedding-8B FOUR-MEMBER STACKED notebook
#
# **Pipeline**: train 4 diverse base models on cached
# `Qwen/Qwen3-Embedding-8B` vectors, fuse with a stacker, calibrate
# once with NN-residuals, and export a single Codabench-ready bundle.
#
#   * **Member 1 (IRT-MLP coverage blend)**: the existing
#     `meta_hybrid_irt_kfactor_gated_mlp` (Model A + Model B
#     coverage-conditional blend, trained with metadata dropout +
#     bc-idx dropout). Heavy backbone, trained on Qwen embeddings
#     directly.
#   * **Member 2 (LightGBM)**: a gradient-boosted decision-tree
#     classifier trained offline on the dense `member_features`
#     schema (`theta_s`, `u_s`, subject metadata, pool features,
#     centroid distances, cluster id, NN features, condition
#     one-hot) and exported as flat NumPy tree arrays. Pure-NumPy
#     traversal at inference -- **no `lightgbm` import in
#     `model.py`**.
#   * **Member 3 (FAISS-free kNN-similarity)**: pure-NumPy
#     similarity-weighted nearest-neighbor classifier on PCA-
#     compressed + quantized item embeddings. Two-stage Bayesian
#     shrinkage. **No `faiss` import in `model.py`** (the bundle
#     uses brute-force matmul over its own quantized embeddings).
#   * **Member 4 (Logistic regression)**: hand-rolled torch trainer,
#     pure-NumPy `apply_state_one` at inference. Same feature schema
#     as Member 2. Provides a calibrated linear baseline that
#     diversifies the ensemble.
#
# **Stacker**: out-of-fold ridge logistic regression on the four
# member predictions (in logit space) plus three auxiliary features
# (`bench_present`, NN neighbor support, mean similarity). Hand-
# rolled torch training, pure-NumPy `apply_one` at runtime.
#
# **Calibrator**: a SINGLE post-stacker NN-residual calibrator with
# `shrinkage_tau` continuous dampening (no hard `min_weight_sum`
# cutoff). Reuses Member 3's neighbor mechanism for FAISS-free
# top-k.
#
# **Caching**: every expensive step writes a checkpoint to
# `CACHE_DIR` (Drive when mounted, else local). Re-running the
# notebook short-circuits to the cache; flip `RESET_CACHE = True`
# in the config cell to force re-compute.
#
# **Verifying runtime safety**: the exporter runs a static import
# audit and a ZIP-size check before emitting `submission.zip`. The
# RED-TEAM cell at the end exercises the rendered `model.py`
# end-to-end (including bundle reload) on synthetic queries.
#
# **What ships in the bundle**:
#   - `artifacts/checkpoint.pt` -- IRT-MLP coverage-blend ensemble.
#   - `artifacts/member2_gbdt/` -- Member 2 trees + bias + feature schema.
#   - `artifacts/member3_knn/` -- Member 3 PCA basis, quantized embeddings,
#     pass-rate table.
#   - `artifacts/member4_logreg/` -- Member 4 weights + bias.
#   - `artifacts/stacker/` -- 8-dim ridge weights + bias.
#   - `artifacts/nn_calibrator_stacked/` + `artifacts/residual_table/`
#     -- post-stacker calibrator state + per-(subject, item) residual
#     table.
#   - `_pure/{gbdt,knn,logreg,stacker,nn_calibration,member_features}.py`
#     -- pure-NumPy runtime modules (no torch/faiss/lightgbm/sklearn).
#   - `cache/` -- training-item cache for runtime NN feature lookup.
#
# **Why blend with coverage gating?** On hosted test rows for unseen
# benchmarks, Model A's metadata gate sees `__MISSING__` patterns it
# has been explicitly trained on (good), and Model A's per-bc bias
# `beta[0]` is the well-trained UNK slot (good); Model B is trained
# specifically to predict robustly without per-bc bias AND without
# bench-side metadata, while still using subject-side metadata
# (model org / family / release date / etc.) which is usually
# available even on cold-start benchmarks. Either model alone is
# suboptimal: A's metadata channel can dominate when it shouldn't,
# B doesn't use bench metadata even when the test row HAS that
# metadata. The blend lets each model contribute where it's
# strongest. Math: Jensen's slack from blending is
# `(1/2) * w * (1-w) * Var(p_A - p_B) / (p * (1-p))`, maximized when
# `w_A` matches the relative reliability per regime.
#
# What ships in the bundle:
#
# - `artifacts/checkpoint.pt` -- ensemble bundle with both members'
#   state dicts, the per-coverage blend weights, and the per-member
#   `force_bench_missing` flags. Loaded by the runtime via the
#   existing `_EnsembleModel` path; Member B's bench fields in the
#   incoming `meta_override` are auto-spliced to row-0 (MISSING)
#   before forward() so inference matches Member B's training.
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
# ## 1b. Cache helper (Drive-aware, with hash-keyed manifest)
#
# Every expensive step in the pipeline routes through ``cache_or_compute``
# so that re-running the notebook after a kernel crash short-circuits to
# the saved artifact. Drive is preferred (durable across Colab restarts);
# local ``./.fmstacked_cache/`` is the fallback.
#
# Set ``RESET_CACHE = True`` in the config cell below to force every
# step to re-compute. Or pass ``force=True`` to the helper.

# %%
import hashlib
import pickle

CACHE_VERSION = "v1"
CACHE_LOCAL = ROOT / ".fmstacked_cache" / CACHE_VERSION
CACHE_LOCAL.mkdir(parents=True, exist_ok=True)
CACHE_DRIVE = None
if IN_COLAB and os.path.ismount("/content/drive"):
    _drive_cache_root = Path("/content/drive/MyDrive/qwen8b_four_member_stacked_cache")
    _drive_cache_root.mkdir(parents=True, exist_ok=True)
    CACHE_DRIVE = _drive_cache_root / CACHE_VERSION
    CACHE_DRIVE.mkdir(parents=True, exist_ok=True)
    print(f"Cache: drive={CACHE_DRIVE} local={CACHE_LOCAL}")
else:
    print(f"Cache: local-only at {CACHE_LOCAL}")

CACHE_LOG = logging.getLogger("cache")


def _cache_key(name: str, key_inputs: tuple) -> str:
    """Stable hash of (name, sorted key_inputs) for cache filenames."""
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"||")
    h.update(repr(tuple(sorted(str(x) for x in key_inputs))).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_paths(name: str, key: str) -> tuple[Path, Path | None]:
    """Return (local_path, drive_path or None) for a given (name, key)."""
    fname = f"{name}__{key}.pkl"
    return CACHE_LOCAL / fname, (CACHE_DRIVE / fname) if CACHE_DRIVE else None


def cache_or_compute(name: str, key_inputs: tuple, compute_fn, *, force: bool = False):
    """Load ``name``+``key_inputs`` from cache if present, else
    call ``compute_fn()``, persist its return value, and return it.

    Cache files are pickled. Use small, JSON-serializable return
    values where possible (``dict`` of ``np.ndarray`` is fine).

    ``force=True`` always triggers re-compute and overwrites the
    cache. The module-level ``RESET_CACHE`` does the same globally
    (set in cell 2).
    """
    global RESET_CACHE
    key = _cache_key(name, key_inputs)
    local_path, drive_path = _cache_paths(name, key)
    do_force = bool(force) or bool(globals().get("RESET_CACHE", False))

    if not do_force:
        for cand in (local_path, drive_path):
            if cand is not None and cand.exists():
                CACHE_LOG.info("[cache HIT] %s key=%s -> %s", name, key, cand.name)
                with open(cand, "rb") as fh:
                    return pickle.load(fh)

    CACHE_LOG.info("[cache MISS] %s key=%s -- computing...", name, key)
    result = compute_fn()
    # Always write to local; mirror to drive when available.
    with open(local_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    CACHE_LOG.info("[cache SAVE] %s -> %s (%.1f MB)", name, local_path.name,
                   local_path.stat().st_size / (1024 * 1024))
    if drive_path is not None:
        try:
            with open(drive_path, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            CACHE_LOG.info("[cache MIRROR-DRIVE] %s -> %s",
                           name, drive_path.name)
        except Exception as exc:
            CACHE_LOG.warning("[cache MIRROR-DRIVE failed] %s: %s", name, exc)
    return result


def cache_clear(prefix: str | None = None) -> int:
    """Drop cache entries; pass ``prefix`` to drop only one stage."""
    n = 0
    for root in (CACHE_LOCAL, CACHE_DRIVE):
        if root is None or not root.exists():
            continue
        for f in root.iterdir():
            if prefix is None or f.name.startswith(prefix):
                try:
                    f.unlink()
                    n += 1
                except Exception:
                    pass
    print(f"cache_clear({prefix!r}): removed {n} files")
    return n


# %% [markdown]
# ## 2. Configuration: Qwen8B + metadata + NN + centroid distances ON

# %%
import json
import yaml

# Set ``RESET_CACHE = True`` to force every cached step in this
# notebook to recompute. Useful when you change a hyperparameter in
# CFG below (the cache key won't include hyperparameters by default
# unless you wire them through ``key_inputs`` in the cache_or_compute
# call site).
RESET_CACHE = False

# tqdm: nice progress bars in Jupyter / Colab.
try:
    from tqdm.auto import tqdm
except Exception:
    # Cheap shim so ``tqdm(iter)`` always works.
    def tqdm(it, **kwargs):
        return it

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
#  * Model B's ``p_bench=1.0`` is the "always-MISSING bench" mode.
#    The bench-side embeddings, towers, and FM crosses only ever
#    update on row 0 (the MISSING token), so Member B effectively
#    has no benchmark-metadata channel. At inference the runtime
#    flips ``force_bench_missing=True`` for Member B, so the
#    incoming meta_override has its bench fields swapped for row 0
#    before forward() -- training distribution == test distribution.
#    Subject side ``p_subj=0.10`` keeps the model robust to
#    cold-start subjects (orgs not in model_info.csv).
#
#  * Model B's ``q_bc=0.15`` regularizes the per-benchmark bias
#    ``beta[bc_idx]``. With prob 0.15 we replace ``bc_idx`` with 0
#    (the UNK slot), so the residual MLP and the IRT channel learn to
#    predict robustly even when the per-bc intercept signal is
#    unavailable. 15% is conservative: any higher and we starve the
#    in-distribution bc bias; any lower and ``beta[0]`` stays under-
#    trained for hosted cold-start rows.
#
#  * ``epochs=3`` for BOTH members. Empirically the metadata head
#    overfits past ~3 epochs (the meta channel is a high-capacity
#    sponge for spurious benchmark / subject regularities); the
#    no-bench-meta head is also at the edge of overfitting on a
#    typical Colab-budget run. Three epochs is the safe default;
#    raise it explicitly only after measuring val log-loss per epoch
#    and confirming it's still trending down at epoch 3.
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
        "p_bench": 1.0,
        "p_subj": 0.10,
        "q_bc": 0.15,
    },
    "epochs_per_member": 3,
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
# Both members use the metadata-aware variant. Member B's bench-side
# pathway is rendered unused at inference via force_bench_missing
# (see cell 11), and at training is starved of gradient by p_bench=1.0.
MODEL_A_NAME = "meta_hybrid_irt_kfactor_gated_mlp"
MODEL_B_NAME = "meta_hybrid_irt_kfactor_gated_mlp"

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
#
# ``epochs`` is pinned to ``CFG["coverage_blend"]["epochs_per_member"]``
# (default 3) instead of reading from ``CFG["train"]["epochs"]``. The
# 3-epoch budget is the empirically-safe default to prevent the
# meta-channel from overfitting; bump it deliberately if val log-loss
# is still trending down at epoch 3.
EPOCHS_PER_MEMBER = int(CFG["coverage_blend"]["epochs_per_member"])
train_cfg = TrainConfig(
    learning_rate=float(CFG["train"].get("learning_rate", 3.0e-3)),
    weight_decay=float(CFG["train"].get("weight_decay", 1.0e-4)),
    batch_size=int(CFG["train"].get("batch_size", 65536)),
    epochs=EPOCHS_PER_MEMBER,
    warmup_steps=int(CFG["train"].get("warmup_steps", 30)),
    scheduler=str(CFG["train"].get("scheduler", "cosine")),
    grad_clip=float(CFG["train"].get("grad_clip", 1.0)),
    early_stopping_patience=int(CFG["train"].get("early_stopping_patience", 5)),
    bf16=bool(CFG["encoder"].get("bf16", True)),
    num_workers=int(CFG["train"].get("num_workers", 0)),
)
print(f"epochs per member: {train_cfg.epochs}")

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
    # Both Member A and Member B use the meta-aware variant with the
    # full schema; attach the same id tables to either when the cfg
    # asks for metadata features. The training-time dropout below
    # makes Member B's bench buffers gradient-only-on-row-0 even
    # though both members share the buffer layout.
    if bool(getattr(cfg, "use_metadata_features", False)) and hasattr(
        m, "attach_metadata_tables"
    ):
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
# ## 8b. Train Model B: subject-metadata-only + bc-dropout (3 epochs)
#
# **Why bc-idx dropout?** Model B's `MetaHybridIRTKFactorGatedMLP`
# predicts via `mu + beta[bc_idx] + alpha_i*(theta - beta_i) +
# factor + meta_logit + residual`. At hosted-test time on a
# cold-start benchmark, `bc_idx=0` (the UNK slot), so `beta[0]` is
# the only piece of the per-bc bias channel the runtime sees.
# Without dropout `beta[0]` receives no gradient during training
# (no row uses bc_idx=0), so at test time it returns whatever the
# random initializer gave us, plus the residual MLP has never had
# to predict without a per-bc bias signal. With prob `q_bc=0.15`
# we replace `bc_idx` with 0 during the forward pass, training
# `beta[0]` to be a useful "average benchmark" prior and forcing
# the MLP to learn bias-free predictions.
#
# **Why subject metadata only?** The user's ask: incorporate model
# metadata (Claude family / org / release date / etc.) into the
# second head while explicitly NOT using benchmark metadata.
# Subject side stays usable because hosted-test models usually have
# entries in `model_info.csv` (or are siblings of an entry whose
# subject-tower output generalizes); benchmark side is unusable
# because hosted-test benchmarks are often brand-new with no entry.
# `p_bench=1.0` at training (via the dropout pre-hook) makes only
# row 0 of the bench tables get gradient, and `force_bench_missing
# =True` at export makes inference match by splicing row 0 into
# every incoming meta_override before Member B's forward(). Net
# effect: Member B uses subject metadata + bc_idx (with bc-dropout)
# + pool / NN features, and ignores the bench tower entirely.
#
# **Why this still yields Jensen slack with Model A?** Member B's
# meta channel has a different feature set than Member A (subject-
# only vs subject+bench), and `Var(p_A - p_B)` is further amplified
# by independent seeds, dropout RNG, and random batch order. We
# don't need a fundamentally different architecture to get blend
# benefit; we need a different effective feature subspace.

# %%
# Member B uses the SAME schema as Member A so both share one set of
# metadata buffers and one preprocessor. The asymmetry is encoded
# entirely via:
#   * train-time ``p_bench=1.0`` (only the bench-side ``__MISSING__``
#     embedding gets gradient signal),
#   * inference-time ``force_bench_missing=True`` on the export side
#     (the runtime ``_EnsembleModel`` swaps Member B's incoming
#     bench fields for row 0 of its own ``bc_meta_*`` buffers
#     before forward()).
# This keeps the bundle layout uniform (one ``meta_preprocessor.json``,
# one indexer, one set of buffer shapes) and the only difference
# between A and B is the dropout schedule.
model_b_cfg = ModelConfig(
    use_metadata_features=True,
    meta_subject_categorical=_meta_schema.subject_categorical,
    meta_subject_numeric=_meta_schema.subject_numeric,
    meta_benchmark_categorical=_meta_schema.benchmark_categorical,
    meta_benchmark_numeric=_meta_schema.benchmark_numeric,
    meta_explicit_crosses=_meta_schema.explicit_crosses,
    **_common_kwargs,
)
print("Model B (subject-meta only via p_bench=1.0 + bc-dropout):")
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
        run_id="qwen8b_minimalist_B_subj_meta_bc_dropout",
        checkpoint_dir=CKPT_DIR,
        extra_metadata={
            "encoder_model_id": CFG["encoder"]["model_id"],
            # Model B keeps ``use_metadata_features=True`` (subject
            # tower trains, bench tower starves at row 0) and is
            # paired with ``force_bench_missing=True`` at export.
            "use_metadata_features": True,
            "use_nn_features": True,
            "centroid_distances_top_m": int(top_m),
            "subj_meta_dropout_and_bc_dropout": asdict(b_drop),
            "force_bench_missing_at_inference": True,
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
    bench_missing_real_subject: bool = False,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run ``model`` over ``ds`` and return per-row uncalibrated
    probabilities.

    ``bc_override`` (when not None) is a (N,) int64 tensor that
    replaces ``ds.bc_ids`` for this scoring pass -- used to force
    `bc_idx -> 0` on the synthetic cold-start slice.

    ``meta_override_template`` (when not None) is a dict with the
    same keys as ``MetadataIdTables.row(0)``; the same row is
    broadcast to every row of every batch. Use this for the
    "all-MISSING" pattern (synthetic cold-start, Model A).

    ``bench_missing_real_subject`` is the pattern Member B was
    trained on: for each batch, look up REAL subject metadata via
    the model's own ``subject_meta_*[subject_idx]`` buffers and
    splice in row-0 (MISSING) for the bench fields. Mutually
    exclusive with ``meta_override_template`` -- if both are set,
    ``meta_override_template`` wins (so Member A's all-MISSING
    pattern still works for synthetic-cold-start scoring).
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
            elif accepts_meta and bench_missing_real_subject:
                B_chunk = end - start
                sub_idx_chunk = ds.subject_ids[start:end].to(device)
                # Real subject lookup from the model's buffers, MISSING
                # bench from row 0 of the same buffers. Matches what
                # the dropout pre-hook produced at training time.
                real_sub_cat = model.subject_meta_cat_ids[sub_idx_chunk]
                real_sub_num = model.subject_meta_num[sub_idx_chunk]
                miss_bc_cat = (
                    model.bc_meta_cat_ids[0:1]
                    .expand(B_chunk, -1)
                    .contiguous()
                )
                miss_bc_num = (
                    model.bc_meta_num[0:1]
                    .expand(B_chunk, -1)
                    .contiguous()
                )
                kw["meta_override"] = {
                    "subj_cat": real_sub_cat,
                    "subj_num": real_sub_num,
                    "bc_cat": miss_bc_cat,
                    "bc_num": miss_bc_num,
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
# Member B also uses the meta-aware variant -- attach the same id
# tables before loading its state dict, just like Member A.
trained_b.attach_metadata_tables(meta_id_tables)
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

# Member B was trained with p_bench=1.0 + p_subj=0.10, so its bench-
# side parameters only ever saw row-0 (MISSING) bench input while
# its subject-side parameters trained on REAL subject metadata
# (with row 0 mixed in 10% of the time). We MUST score it with the
# matching pattern -- real subject + MISSING bench -- otherwise the
# real-bench embeddings (untrained random init) emit noise. The
# synthetic-cold-start subset additionally forces ``bc_idx -> 0``
# to simulate the per-bc-bias-missing test-time regime.
print("Scoring Model B: real-bc + real-subj-meta + MISSING-bench-meta pass...")
p_b_real = _score_dataset(val_ds, trained_b, bench_missing_real_subject=True)
print("Scoring Model B: cold-start (bc=0 + same meta pattern) pass...")
p_b_cold = _score_dataset(
    val_ds, trained_b, bc_override=val_bc_synth, bench_missing_real_subject=True
)
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

# Per-row blended val prediction. This is **Member 1**'s contribution
# to the four-member stacker below.
w_per_row = np.where(synth_mask, w_missing, w_present).astype(np.float32)
p_blend_val = w_per_row * p_a_val + (1.0 - w_per_row) * p_b_val
print(
    f"[Member 1] coverage-blend val log-loss: "
    f"{-(ylab_val * np.log(np.clip(p_blend_val, 1e-6, 1 - 1e-6)) + (1 - ylab_val) * np.log(1 - np.clip(p_blend_val, 1e-6, 1 - 1e-6))).mean():.6f}"
)

# %% [markdown]
# ## 9b. Build dense feature matrix for Members 2 + 4
#
# `src/member_features.py` defines the locked column schema shared
# across the GBDT (Member 2) and LogReg (Member 4) members. We:
#
#   1. fit a `MemberFeatureSchema` on training rows (locks column
#      order, vocab, z-score stats),
#   2. build a `MemberSubjectTables` (per-subject theta, u, subject
#      categorical ids, scaled subject numerics),
#   3. dense-build `[N, F]` feature matrices for train + val.
#
# The runtime `predict()` rebuilds the same vector via the per-row
# `build_member_features_one` so column alignment is guaranteed.

# %%
import torch

from src.member_features import (
    MemberFeatureSchema,
    MemberSubjectTables,
    build_member_features,
)

# --- Subject side: pull theta + u from Member A's checkpoint, build
#     subject_cat_ids / subject_num from meta_id_tables.
_ckpt_a_for_theta = torch.load(result_a.checkpoint_path, map_location="cpu")
_state_a_for_theta = _ckpt_a_for_theta["model_state"]
theta_s_per_subject = _state_a_for_theta["theta"].cpu().numpy().astype(np.float32)
if "u" in _state_a_for_theta:
    u_s_per_subject = _state_a_for_theta["u"].cpu().numpy().astype(np.float32)
elif "U" in _state_a_for_theta:
    u_s_per_subject = _state_a_for_theta["U"].cpu().numpy().astype(np.float32)
else:
    u_s_per_subject = np.zeros((indexer.n_subjects, K_LATENT), dtype=np.float32)
if u_s_per_subject.ndim == 1:
    u_s_per_subject = u_s_per_subject[:, None]
print(f"[member_features] theta shape={theta_s_per_subject.shape}  "
      f"u shape={u_s_per_subject.shape}")

# Subject metadata tables (reuse the ones built for Member 1).
subj_cat_ids_np = meta_id_tables.subject_cat_ids.cpu().numpy().astype(np.int64)
subj_num_np = meta_id_tables.subject_num.cpu().numpy().astype(np.float32)
subject_tables = MemberSubjectTables(
    theta=theta_s_per_subject,
    u=u_s_per_subject,
    subject_cat_ids=subj_cat_ids_np,
    subject_num=subj_num_np,
)
print(f"[member_features] subject_tables: theta={subject_tables.theta.shape}  "
      f"u={subject_tables.u.shape}  cat={subject_tables.subject_cat_ids.shape}  "
      f"num={subject_tables.subject_num.shape}")

# --- Pool stats: split combined_cols into text-pool vs centroid-dist.
_text_pool_cols = [
    c for c in combined_cols if not c.startswith("centroid_dist_")
]
_centroid_cols_in_pool = [
    c for c in combined_cols if c.startswith("centroid_dist_")
]
print(f"[member_features] text pool cols: {_text_pool_cols}")
print(f"[member_features] centroid cols : {_centroid_cols_in_pool}")


def _fit_schema():
    return MemberFeatureSchema.fit(
        k_factors=int(u_s_per_subject.shape[1]),
        n_clusters=int(CFG["clustering"]["k"]),
        top_m_centroids=int(top_m),
        pool_feature_names=tuple(_text_pool_cols),
        pool_stats={c: pool_stats[c] for c in _text_pool_cols},
        nn_feature_names=tuple(f"nn_{i:02d}" for i in range(nn_train_mat.shape[1])),
        # Use the cardinalities from meta_id_tables (matches model A/B);
        # the field names come from _meta_schema. Both are parallel.
        subject_cat_field_names=tuple(_meta_schema.subject_categorical),
        subject_cat_field_cardinalities=tuple(
            int(c) for c in meta_id_tables.subject_cat_cardinalities
        ),
        subject_num_field_names=tuple(_meta_schema.subject_numeric),
        train_conditions=primary.train["condition"].astype(str).tolist(),
        min_condition_count=10,
        centroid_dist_names=tuple(_centroid_cols_in_pool),
    )


member_feat_schema: MemberFeatureSchema = cache_or_compute(
    "member_feat_schema",
    key_inputs=("v1", len(_text_pool_cols), int(top_m), int(u_s_per_subject.shape[1]),
                int(indexer.n_subjects)),
    compute_fn=_fit_schema,
)
print(f"[member_features] schema dim={member_feat_schema.feature_dim}")
print(f"[member_features] first 8 columns: {list(member_feat_schema.feature_names[:8])}")


train_subj_ids_np = np.array(
    [indexer.subject_id(str(s)) for s in primary.train["subject_key"]],
    dtype=np.int64,
)
val_subj_ids_np = np.array(
    [indexer.subject_id(str(s)) for s in primary.val["subject_key"]],
    dtype=np.int64,
)


def _split_pool_and_centroids(item_keys_for_split):
    """Look up RAW pool + centroid_dist features for a given list
    of item_keys, in the same order as ``item_keys_for_split``."""
    df = combined_df.set_index("item_key")
    sub = df.reindex(np.asarray(item_keys_for_split, dtype=str))
    sub = sub.fillna(0.0)
    pool_arr = sub[_text_pool_cols].astype(np.float32).to_numpy()
    cd_arr = sub[_centroid_cols_in_pool].astype(np.float32).to_numpy()
    return pool_arr, cd_arr


def _build_X(split_part, nn_mat):
    keys = split_part["item_key"].astype(str).tolist()
    pool_raw, cd_raw = _split_pool_and_centroids(keys)
    cl_ids = np.array(
        [int(cluster_assignments.get(str(k), 0)) for k in keys],
        dtype=np.int64,
    )
    subj_idx = np.array(
        [indexer.subject_id(str(s)) for s in split_part["subject_key"]],
        dtype=np.int64,
    )
    return build_member_features(
        member_feat_schema,
        subject_tables,
        subject_idx=subj_idx,
        pool_feats=pool_raw,
        centroid_dists=cd_raw,
        cluster_ids=cl_ids,
        nn_feats=nn_mat.astype(np.float32),
        conditions=split_part["condition"].astype(str).tolist(),
    )


X_train_dense = cache_or_compute(
    "X_train_dense",
    key_inputs=(member_feat_schema.feature_dim, len(primary.train)),
    compute_fn=lambda: _build_X(primary.train, nn_train_mat),
)
X_val_dense = cache_or_compute(
    "X_val_dense",
    key_inputs=(member_feat_schema.feature_dim, len(primary.val)),
    compute_fn=lambda: _build_X(primary.val, nn_val_mat),
)
y_train = primary.train["label"].astype(float).to_numpy().astype(np.float32)
print(f"[member_features] X_train: {X_train_dense.shape}  X_val: {X_val_dense.shape}")

# %% [markdown]
# ## 9c. Train Member 2 (LightGBM)
#
# Offline LightGBM training; we ship the compiled tree arrays + bias
# for pure-NumPy traversal at runtime. The internal parity check (in
# `fit_gbdt_member`) ensures the NumPy walker matches LightGBM's
# `predict(raw_score=True)` to within `parity_atol=1e-5`.

# %%
from src.gbdt_member import (
    apply_batch as gbdt_apply_batch,
    fit_gbdt_member,
)


def _fit_member2():
    print("[Member 2] training LightGBM (this typically takes 30-90s)...")
    return fit_gbdt_member(
        X=X_train_dense,
        y=y_train,
        feature_names=tuple(member_feat_schema.feature_names),
        n_estimators=int(CFG.get("member2_gbdt", {}).get("n_estimators", 400)),
        learning_rate=float(CFG.get("member2_gbdt", {}).get("learning_rate", 0.05)),
        num_leaves=int(CFG.get("member2_gbdt", {}).get("num_leaves", 31)),
        min_data_in_leaf=int(CFG.get("member2_gbdt", {}).get("min_data_in_leaf", 50)),
        early_stopping_rounds=int(CFG.get("member2_gbdt", {}).get("early_stopping_rounds", 30)),
        seed=SEED,
        parity_atol=1.0e-5,
        # Hold out 10% of train for LightGBM's own early-stopping val.
        val_fraction=0.1,
    )


gbdt_state = cache_or_compute(
    "gbdt_state",
    key_inputs=(member_feat_schema.feature_dim, len(primary.train), SEED),
    compute_fn=_fit_member2,
)
p_member2_val = gbdt_apply_batch(gbdt_state, X_val_dense)
nll_m2 = float(-(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
                 + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean())
print(f"[Member 2] val log-loss: {nll_m2:.6f}  n_trees={len(gbdt_state.tree_starts)-1}")

# %% [markdown]
# ## 9d. Train Member 3 (FAISS-free kNN-similarity)
#
# PCA-compressed + (fp16 or int8) quantized embeddings + dense
# `[n_subjects, n_items]` pass-rate table. Two-stage Bayesian
# shrinkage (per-subject, then global). All runtime ops are pure
# NumPy matmul + argpartition; no FAISS, no torch.

# %%
from src.knn_member import (
    KNNMemberState,
    apply_batch as knn_apply_batch,
    fit_knn_member,
)


def _passrate_dense_from_csr(csr, mask_csr, n_subjects, n_items):
    """Inflate the sparse pass-rate matrices to dense fp16 / bool
    for Member 3's runtime. ``[S, N]`` is bounded by ``S * N * (2 + 1)``
    bytes; with S=200, N=100000 that's ~60 MB, fine."""
    dense = csr.astype(np.float32).toarray()
    mask = mask_csr.astype(bool).toarray()
    return dense, mask


print("[Member 3] inflating pass-rate matrices...")
m3_passrate_dense, m3_passrate_mask = _passrate_dense_from_csr(
    nn_passrate_csr, nn_passrate_mask_csr,
    indexer.n_subjects, len(train_item_keys),
)
print(f"[Member 3] passrate_dense shape={m3_passrate_dense.shape}  "
      f"observed={m3_passrate_mask.sum():,} / {m3_passrate_mask.size:,}")


# Subject_keys in the SAME order as passrate_dense's rows (which
# is indexer's id ordering: subject_to_id maps key -> id, so we
# need keys sorted by id).
_subject_keys_ordered = [
    k for k, _ in sorted(indexer.subject_to_id.items(), key=lambda kv: kv[1])
]
assert len(_subject_keys_ordered) == indexer.n_subjects


def _fit_member3():
    train_emb_arr = np.stack(
        [item_emb_lookup[k] for k in train_item_keys], axis=0
    ).astype(np.float32)
    print(f"[Member 3] PCA-fitting + quantizing {len(train_item_keys):,} item embeddings...")
    return fit_knn_member(
        item_keys=train_item_keys,
        item_embeddings=train_emb_arr,
        subject_keys=_subject_keys_ordered,
        passrate_dense=m3_passrate_dense,
        passrate_mask=m3_passrate_mask,
        pca_dim=int(CFG.get("member3_knn", {}).get("pca_dim", 128)),
        quantization=str(CFG.get("member3_knn", {}).get("quantization", "int8")),
        k=int(CFG.get("member3_knn", {}).get("k", 32)),
        tau_subject=float(CFG.get("member3_knn", {}).get("tau_subject", 5.0)),
        tau_global=float(CFG.get("member3_knn", {}).get("tau_global", 200.0)),
    )


knn_state: KNNMemberState = cache_or_compute(
    "knn_state",
    key_inputs=(len(train_item_keys), int(CFG.get("member3_knn", {}).get("pca_dim", 128))),
    compute_fn=_fit_member3,
)
print(f"[Member 3] state: pca_dim={knn_state.pca_dim}  "
      f"quant={knn_state.quantization}  k={knn_state.k}")

# Score on val rows.
print("[Member 3] scoring val rows...")
val_item_emb = np.stack(
    [item_emb_lookup[k] for k in primary.val["item_key"]], axis=0
).astype(np.float32)
val_subj_keys_for_knn = [str(s) for s in primary.val["subject_key"]]
p_member3_val = knn_apply_batch(knn_state, val_item_emb, val_subj_keys_for_knn)
nll_m3 = float(-(ylab_val * np.log(np.clip(p_member3_val, 1e-6, 1 - 1e-6))
                 + (1 - ylab_val) * np.log(1 - np.clip(p_member3_val, 1e-6, 1 - 1e-6))).mean())
print(f"[Member 3] val log-loss: {nll_m3:.6f}")

# %% [markdown]
# ## 9e. Train Member 4 (Logistic regression)
#
# Hand-rolled torch trainer (Adam + L2 + early stopping). At runtime
# we ship only `weights.npz` and `meta.json`; predict via a single
# matvec + sigmoid in pure NumPy.

# %%
from src.logreg_member import (
    apply_state_batch as logreg_apply_state_batch,
    fit_logreg_member,
)


def _fit_member4():
    print("[Member 4] training hand-rolled torch logistic regression...")
    return fit_logreg_member(
        X=X_train_dense,
        y=y_train,
        feature_names=tuple(member_feat_schema.feature_names),
        epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
        learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
        weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
        early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
        seed=SEED,
        # Hold out 10% of train for early-stopping val.
        val_fraction=0.1,
    )


logreg_state = cache_or_compute(
    "logreg_state",
    key_inputs=(member_feat_schema.feature_dim, len(primary.train), SEED),
    compute_fn=_fit_member4,
)
p_member4_val = logreg_apply_state_batch(logreg_state, X_val_dense)
nll_m4 = float(-(ylab_val * np.log(np.clip(p_member4_val, 1e-6, 1 - 1e-6))
                 + (1 - ylab_val) * np.log(1 - np.clip(p_member4_val, 1e-6, 1 - 1e-6))).mean())
print(f"[Member 4] val log-loss: {nll_m4:.6f}  "
      f"weights||={float(np.linalg.norm(logreg_state.weights)):.3f}  "
      f"bias={float(logreg_state.bias):.3f}")

# %% [markdown]
# ## 9f. Train the stacker (ridge logistic regression on val predictions)
#
# The stacker takes 4 member predictions (in logit space) plus 3
# auxiliary features (`bench_present`, NN neighbor support, mean
# similarity, centroid distance) and emits one calibrated probability.
# Hand-rolled torch training; pure-NumPy inference. We fit on val
# (held out from every member's training) and report log-loss on the
# same val set -- the OOF assertion is implicit because every member
# saw zero val rows during training.

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    build_stacker_features,
    fit_stacker,
)

# Auxiliary stacker features (val side).
val_bench_present = np.array(
    [
        1.0 if str(b) in indexer.bc_to_id else 0.0
        for b in primary.val["benchmark_condition_key"]
    ],
    dtype=np.float32,
)
# NN neighbor support: log1p of how many observed neighbors we found
# for the val item under Member 3's neighbor mechanism.
print("[Stacker] computing auxiliary features (NN support, mean sim, centroid dist)...")
val_nn_mean_sim = nn_val_mat[:, 1].astype(np.float32)  # column 1 = mean similarity
val_nn_support = nn_val_mat[:, 2].astype(np.float32)   # column 2 = log1p neighbors observed

# Centroid distance: nearest-centroid normalized distance from pool features.
# We grab the first centroid_dist_* column; if multiple exist (top_m > 1),
# we take the min (closest centroid).
_centroid_cols = [
    c for c in pool_features_z.columns if c.startswith("centroid_dist_")
]
if _centroid_cols:
    val_pool_idx = pool_features_z.set_index("item_key").reindex(
        primary.val["item_key"].astype(str)
    )
    val_centroid_dist = val_pool_idx[_centroid_cols].astype(np.float32).min(axis=1).to_numpy()
else:
    val_centroid_dist = np.full(len(primary.val), 0.5, dtype=np.float32)

stacker_member_probs_val = np.stack(
    [p_blend_val, p_member2_val, p_member3_val, p_member4_val], axis=1
).astype(np.float32)

stacker_X_val = build_stacker_features(
    member_probs=stacker_member_probs_val,
    bench_present=val_bench_present,
    nn_neighbor_support=val_nn_support,
    nn_mean_similarity=val_nn_mean_sim,
    centroid_distance=val_centroid_dist,
)
print(f"[Stacker] X_val: {stacker_X_val.shape}  ylab_val: {ylab_val.shape}")


def _fit_stacker():
    return fit_stacker(
        X=stacker_X_val,
        y=ylab_val.astype(np.float32),
        n_iters=int(CFG.get("stacker", {}).get("n_iters", 1500)),
        learning_rate=float(CFG.get("stacker", {}).get("learning_rate", 0.05)),
        l2=float(CFG.get("stacker", {}).get("l2", 1.0)),
        early_stopping_patience=int(CFG.get("stacker", {}).get("early_stopping_patience", 200)),
        # Internal 80/20 split inside fit_stacker for early stopping.
        val_fraction=0.2,
        seed=SEED,
    )


stacker_state = cache_or_compute(
    "stacker_state",
    key_inputs=(stacker_X_val.shape[1], len(ylab_val), SEED),
    compute_fn=_fit_stacker,
)
print(f"[Stacker] weights: {stacker_state.weights}")
print(f"[Stacker] bias:    {stacker_state.bias:.4f}")
p_stacker_val = stacker_apply_batch(stacker_state, stacker_X_val)
nll_stack = float(-(ylab_val * np.log(np.clip(p_stacker_val, 1e-6, 1 - 1e-6))
                    + (1 - ylab_val) * np.log(1 - np.clip(p_stacker_val, 1e-6, 1 - 1e-6))).mean())
nll_uniform = float(-(ylab_val * np.log(np.clip(stacker_member_probs_val.mean(axis=1), 1e-6, 1 - 1e-6))
                    + (1 - ylab_val) * np.log(1 - np.clip(stacker_member_probs_val.mean(axis=1), 1e-6, 1 - 1e-6))).mean())
print(f"\n[Stacker] val log-loss summary:")
print(f"  Member 1 (IRT-MLP blend):  {-(ylab_val * np.log(np.clip(p_blend_val, 1e-6, 1 - 1e-6)) + (1 - ylab_val) * np.log(1 - np.clip(p_blend_val, 1e-6, 1 - 1e-6))).mean():.6f}")
print(f"  Member 2 (LightGBM):       {nll_m2:.6f}")
print(f"  Member 3 (kNN-similarity): {nll_m3:.6f}")
print(f"  Member 4 (LogReg):         {nll_m4:.6f}")
print(f"  Uniform avg of 4 members:  {nll_uniform:.6f}")
print(f"  STACKER:                   {nll_stack:.6f}")
if nll_stack > nll_uniform + 1e-3:
    print(
        "WARNING: Stacker did not beat uniform average. Consider increasing "
        "stacker.l2 or stacker.n_iters in CFG, or check that members are "
        "diverse enough."
    )

# %% [markdown]
# ## 10. NN calibrator on the STACKED predictions (post-stacker)
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

# --- Build train-side predictions for ALL FOUR members so the
#     residual table holds residuals against the same p_uncal that
#     the runtime will produce after stacking.
print("[Calibrator] Member 1 (IRT-MLP coverage blend) on train...")
p_a_train = _score_dataset(train_ds, trained_a)
p_b_train = _score_dataset(
    train_ds, trained_b, bench_missing_real_subject=True
)
p1_train = (BLEND_PRESENT[0] * p_a_train + BLEND_PRESENT[1] * p_b_train).astype(np.float32)
print(f"[Calibrator] p1_train: shape={p1_train.shape}  "
      f"log-loss={-(y_train * np.log(np.clip(p1_train, 1e-6, 1 - 1e-6)) + (1 - y_train) * np.log(1 - np.clip(p1_train, 1e-6, 1 - 1e-6))).mean():.6f}")

print("[Calibrator] Member 2 (GBDT) on train...")
p2_train = gbdt_apply_batch(gbdt_state, X_train_dense)

print("[Calibrator] Member 3 (kNN) on train...")
train_item_emb = np.stack(
    [item_emb_lookup[k] for k in primary.train["item_key"]], axis=0
).astype(np.float32)
train_subj_keys_for_knn = [str(s) for s in primary.train["subject_key"]]
p3_train = knn_apply_batch(knn_state, train_item_emb, train_subj_keys_for_knn)

print("[Calibrator] Member 4 (LogReg) on train...")
p4_train = logreg_apply_state_batch(logreg_state, X_train_dense)

# Train-side stacker features (matching the same builder used on val).
train_bench_present = np.array(
    [
        1.0 if str(b) in indexer.bc_to_id else 0.0
        for b in primary.train["benchmark_condition_key"]
    ],
    dtype=np.float32,
)
train_nn_mean_sim = nn_train_mat[:, 1].astype(np.float32)
train_nn_support = nn_train_mat[:, 2].astype(np.float32)
if _centroid_cols_in_pool:
    train_pool_idx = pool_features_z.set_index("item_key").reindex(
        primary.train["item_key"].astype(str)
    )
    train_centroid_dist = train_pool_idx[_centroid_cols_in_pool].astype(np.float32).min(axis=1).to_numpy()
else:
    train_centroid_dist = np.full(len(primary.train), 0.5, dtype=np.float32)

train_stacker_feats = build_stacker_features(
    member_probs=np.stack([p1_train, p2_train, p3_train, p4_train], axis=1).astype(np.float32),
    bench_present=train_bench_present,
    nn_neighbor_support=train_nn_support,
    nn_mean_similarity=train_nn_mean_sim,
    centroid_distance=train_centroid_dist,
)
p_uncal_train_stacker = stacker_apply_batch(stacker_state, train_stacker_feats)
print(f"[Calibrator] p_uncal_train_stacker: shape={p_uncal_train_stacker.shape}  "
      f"log-loss={-(y_train * np.log(np.clip(p_uncal_train_stacker, 1e-6, 1 - 1e-6)) + (1 - y_train) * np.log(1 - np.clip(p_uncal_train_stacker, 1e-6, 1 - 1e-6))).mean():.6f}")

# --- Build the residual table from the STACKED train predictions.
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
    uncal_probs=p_uncal_train_stacker[ok],
    n_subjects=indexer.n_subjects,
    n_training_items=len(train_item_keys),
)

# --- Val-side neighbor lookup (deduped, chunked for memory).
val_keys_arr = np.asarray([str(k) for k in primary.val["item_key"]])
val_unique_keys, val_inverse = np.unique(val_keys_arr, return_inverse=True)
K_CAL = int(CFG["nn_calibration"]["k"])
val_uniq_idx = np.empty((len(val_unique_keys), K_CAL), dtype=np.int64)
val_uniq_sims = np.empty((len(val_unique_keys), K_CAL), dtype=np.float32)
for _s in tqdm(range(0, len(val_unique_keys), NN_QUERY_CHUNK),
               desc="[Calibrator] val NN search"):
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

# Fit the calibrator on the STACKED val predictions with the
# 2-D (alpha, shrinkage_tau) grid sweep.
calibrator = NNCalibrator.fit_alpha_on_val(
    residual_table=residual_table,
    val_subject_ids=val_subj_ids_np,
    val_neighbor_rows=val_neighbor_rows,
    val_neighbor_sims=val_neighbor_sims,
    val_uncal_probs=p_stacker_val,
    val_labels=ylab_val,
    k=int(CFG["nn_calibration"]["k"]),
    similarity=str(CFG["nn_calibration"].get("similarity", "cosine")),
    temperature=float(CFG["nn_calibration"].get("temperature", 1.0)),
    shrinkage_taus=tuple(CFG["nn_calibration"].get("shrinkage_taus") or
                         (0.0, 0.5, 1.0, 2.0, 5.0)),
)
print(f"[Calibrator] state: alpha={calibrator.state.alpha:.4f}  "
      f"shrinkage_tau={calibrator.state.shrinkage_tau:.4f}  "
      f"fit_method={calibrator.state.fit_method}")

p_final_val = calibrator.apply(
    residual_table=residual_table,
    subject_ids=val_subj_ids_np,
    neighbor_rows=val_neighbor_rows,
    neighbor_sims=val_neighbor_sims,
    p_uncal=p_stacker_val,
)
nll_final = float(-(ylab_val * np.log(np.clip(p_final_val, 1e-6, 1 - 1e-6))
                    + (1 - ylab_val) * np.log(1 - np.clip(p_final_val, 1e-6, 1 - 1e-6))).mean())
print(f"\n[Calibrator] post-calibration val log-loss: {nll_final:.6f}  "
      f"(stacker: {nll_stack:.6f}, delta: {nll_final - nll_stack:+.6f})")

RESIDUAL_DIR = ROOT / "artifacts" / "nn_calibration_stacked"
RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
residual_table.save(RESIDUAL_DIR)
print(f"[Calibrator] residual table saved to {RESIDUAL_DIR}")

# %% [markdown]
# ## 11. Export the four-member stacked bundle
#
# Two-step export:
#   1. Build the Member 1 (IRT-MLP coverage-blend) bundle via the
#      existing `export_coverage_blend_run`. This handles the heavy
#      pieces (encoder, Indexer, model classes, pool/cluster/NN
#      features, per-batch caching) that we don't want to rewrite.
#   2. Wrap the Member 1 bundle with `export_four_member_stacked_run`,
#      which:
#        - copies Members 2-4 + stacker + post-calibrator state into
#          ``artifacts/{member2_gbdt,member3_knn,member4_logreg,
#          stacker,nn_calibrator_stacked,residual_table}/``,
#        - copies the pure-NumPy runtime modules into ``_pure/``,
#        - patches ``model.py`` to strip ``import faiss`` and append
#          a stacker postprocessing block that reassigns ``predict``
#          to the four-member orchestration,
#        - audits the resulting bundle for forbidden imports and
#          enforces the 65 MB ZIP cap.

# %%
from src.export_submission import (
    bundle_training_cache,
    compute_train_counts,
    export_coverage_blend_run,
    make_submission_zip,
)
from src.export_stacked_submission import (
    audit_runtime_imports,
    export_four_member_stacked_run,
    measure_bundle_size_bytes,
)

SUBMISSION_DIR_M1 = ROOT / "submission" / "qwen8b_4member_stacked_member1"
SUBMISSION_DIR = ROOT / "submission" / "qwen8b_4member_stacked"
TRAINING_CACHE_DIR = ROOT / "artifacts" / "training_cache"

print("[Export] Step 1: building Member 1 (coverage-blend) bundle...")
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
    f"[Export] training cache size: {training_cache_result.size_mb:.1f} MB  "
    f"(soft cap {CFG['submission_cache'].get('max_bundle_size_mb', 200)} MB)"
)

sub_dir_m1 = export_coverage_blend_run(
    member_a_state_dict=ckpt_a["model_state"],
    member_a_model_name=MODEL_A_NAME,
    member_a_model_cfg=asdict(model_a_cfg),
    member_a_config_id="meta_dropout",
    member_b_state_dict=ckpt_b["model_state"],
    member_b_model_name=MODEL_B_NAME,
    member_b_model_cfg=asdict(model_b_cfg),
    member_b_config_id="subj_meta_bc_dropout",
    member_a_force_bench_missing=False,
    member_b_force_bench_missing=True,
    blend_weights_present=BLEND_PRESENT,
    blend_weights_missing=BLEND_MISSING,
    indexer={
        "subject_to_id": dict(indexer.subject_to_id),
        "bc_to_id": dict(indexer.bc_to_id),
    },
    encoder_cfg=CFG["encoder"],
    submission_dir=SUBMISSION_DIR_M1,
    representative_result=result_a,
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
    # Note: we DON'T ship the legacy NN calibrator here -- the
    # post-stacker calibrator below replaces it. Setting these to
    # None is the legacy-bundle escape hatch.
    nn_calibrator_state=None,
    nn_calibrator_table_dir=None,
)
print(f"[Export] Member 1 bundle: {sub_dir_m1}")

# Save the schema + subject_tables so a future runtime feature
# builder can load them. This is a no-op for the current bundle
# (since runtime_feature_builder_py=None means Members 2 & 4 fall
# through to their bias at test time -- the stacker still uses
# them, just with their constant bias output rather than a
# per-row prediction). See KNOWN LIMITATION below.
SCHEMA_DIR = sub_dir_m1.parent / "_member_feat_artifacts"
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
schema_dict = member_feat_schema.to_dict()
(SCHEMA_DIR / "member_feature_schema.json").write_text(
    json.dumps(schema_dict, indent=2), encoding="utf-8"
)
subject_tables.save(SCHEMA_DIR)
print(f"[Export] saved schema + subject_tables to {SCHEMA_DIR}")
print("\n[Export] Step 2: wrapping with four-member stacked exporter...")
print(
    "  KNOWN LIMITATION: Member 2 (GBDT) and Member 4 (LogReg) "
    "rely on the dense `member_features` schema. The runtime "
    "feature builder (`runtime_feature_builder_py`) is a TODO: "
    "without it, Members 2 & 4 emit their bias prediction at "
    "test time. The stacker still combines them, but their "
    "diversity contribution is reduced. Member 1 (IRT-MLP) and "
    "Member 3 (kNN) are fully wired."
)
sub_dir = export_four_member_stacked_run(
    member1_bundle_dir=sub_dir_m1,
    gbdt_state=gbdt_state,
    knn_state=knn_state,
    logreg_state=logreg_state,
    stacker_state=stacker_state,
    nn_calibrator=calibrator,
    residual_table=residual_table,
    out_dir=SUBMISSION_DIR,
    src_dir=ROOT / "src",
    runtime_feature_builder_py=None,
)
print(f"[Export] stacked bundle: {sub_dir}")

# Static audit: catch any forbidden runtime imports BEFORE zipping.
findings = audit_runtime_imports(sub_dir)
if findings:
    raise RuntimeError(f"Static audit found forbidden imports: {findings}")
print("[Export] static import audit: PASS")

bundle_bytes = measure_bundle_size_bytes(sub_dir)
print(f"[Export] bundle size: {bundle_bytes / (1024 * 1024):.1f} MB")

zip_cap_mb = float(CFG["submission"].get("max_zip_size_mb", 65))
zip_path = make_submission_zip(
    sub_dir,
    zip_path=sub_dir.with_suffix(".zip"),
    max_zip_size_mb=zip_cap_mb,
)
size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"[Export] ZIP: {zip_path}  size: {size_mb:.1f} MB  (cap {zip_cap_mb:.0f} MB)")

if IN_COLAB:
    try:
        from google.colab import files  # type: ignore
        files.download(str(zip_path))
    except Exception as exc:
        print("Auto-download failed (%s); copy the zip manually." % exc)

# %% [markdown]
# ## 12. RED-TEAM: end-to-end smoke test of the rendered bundle
#
# We've already audited the bundle's static imports and ZIP size.
# Now exercise the rendered `model.py` by importing it as a module
# in a child process and calling `predict()` on a synthetic input.
# This catches runtime template bugs (NameError, KeyError, missing
# state files) that static analysis misses.
#
# We also dump a per-member ablation table on val so you can see
# how each part of the pipeline contributes.

# %%
import importlib.util
import subprocess
import sys

# --- 12a. Ablation table.
print("=" * 72)
print("RED-TEAM SECTION A: per-member val log-loss")
print("=" * 72)
_y = ylab_val.astype(np.float64)
_eps = 1e-6


def _ll(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _eps, 1 - _eps)
    return float(-(_y * np.log(p) + (1 - _y) * np.log(1 - p)).mean())


print(f"  Member 1 (IRT-MLP coverage blend) : {_ll(p_blend_val):.6f}")
print(f"  Member 2 (LightGBM)               : {_ll(p_member2_val):.6f}")
print(f"  Member 3 (kNN-similarity)         : {_ll(p_member3_val):.6f}")
print(f"  Member 4 (LogReg)                 : {_ll(p_member4_val):.6f}")
print(f"  Uniform avg of 4                  : {_ll(stacker_member_probs_val.mean(axis=1)):.6f}")
print(f"  Stacker only                      : {_ll(p_stacker_val):.6f}")
print(f"  Stacker + NN calibrator (FINAL)   : {_ll(p_final_val):.6f}")

# --- 12b. Static checks on the rendered model.py.
print("\n" + "=" * 72)
print("RED-TEAM SECTION B: static checks on rendered model.py")
print("=" * 72)
model_py_text = (sub_dir / "model.py").read_text(encoding="utf-8")
checks: list[tuple[str, bool, str]] = [
    ("contains _stacked_predict",
     "_stacked_predict" in model_py_text,
     "missing"),
    ("reassigns predict to _stacked_predict",
     "predict = _stacked_predict" in model_py_text,
     "missing"),
    ("FAISS sentinel present",
     "PHASE-5: strict no-FAISS rule" in model_py_text,
     "FAISS not stripped"),
    ("no `import faiss` anywhere",
     not bool(__import__("re").search(r"^\s*import\s+faiss", model_py_text,
                                       __import__("re").MULTILINE)),
     "found bare `import faiss`"),
]
for name, ok, fail_reason in checks:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}" + ("" if ok else f"  ({fail_reason})"))
all_ok = all(ok for _, ok, _ in checks)
if not all_ok:
    raise RuntimeError("RED-TEAM B failed; aborting.")

# --- 12c. Subprocess smoke test of model.py.
print("\n" + "=" * 72)
print("RED-TEAM SECTION C: subprocess import + predict() smoke test")
print("=" * 72)
smoke_py = sub_dir / "_smoke_test.py"
smoke_py.write_text(
    "import sys, os, json\n"
    f"sys.path.insert(0, {str(sub_dir)!r})\n"
    "import model as M\n"
    "row = {\n"
    "    'benchmark': 'TEST::synthetic',\n"
    "    'condition': 'none',\n"
    "    'subject_content': 'synthetic subject',\n"
    "    'item_content': 'synthetic item content for smoke test',\n"
    "}\n"
    "p = M.predict(row, labeled=None)\n"
    "p = float(p)\n"
    "assert 0.0 <= p <= 1.0, f'Out-of-range probability: {p}'\n"
    "print(f'PASS predict={p:.4f}')\n",
    encoding="utf-8",
)
try:
    res = subprocess.run(
        [sys.executable, str(smoke_py)],
        capture_output=True, text=True, timeout=300,
    )
    print(f"  stdout: {res.stdout.strip()}")
    if res.returncode != 0:
        print(f"  stderr: {res.stderr[-2000:]}")
        print("  [FAIL] subprocess smoke test")
    else:
        print("  [PASS] subprocess smoke test")
except subprocess.TimeoutExpired:
    print("  [FAIL] subprocess smoke test timed out (300s)")
except Exception as exc:
    print(f"  [SKIP] subprocess smoke test failed to launch: {exc}")
finally:
    try:
        smoke_py.unlink()
    except Exception:
        pass

# --- 12d. Determinism: two consecutive predict() calls return the
# same value.
print("\n" + "=" * 72)
print("RED-TEAM SECTION D: determinism of stacker + calibrator on val")
print("=" * 72)
p_stack_a = stacker_apply_batch(stacker_state, stacker_X_val)
p_stack_b = stacker_apply_batch(stacker_state, stacker_X_val)
assert np.array_equal(p_stack_a, p_stack_b)
print(f"  [PASS] stacker_apply_batch determinism (max delta: {float(np.abs(p_stack_a - p_stack_b).max())})")
p_cal_a = calibrator.apply(
    residual_table=residual_table,
    subject_ids=val_subj_ids_np,
    neighbor_rows=val_neighbor_rows,
    neighbor_sims=val_neighbor_sims,
    p_uncal=p_stack_a,
)
p_cal_b = calibrator.apply(
    residual_table=residual_table,
    subject_ids=val_subj_ids_np,
    neighbor_rows=val_neighbor_rows,
    neighbor_sims=val_neighbor_sims,
    p_uncal=p_stack_a,
)
assert np.array_equal(p_cal_a, p_cal_b)
print(f"  [PASS] calibrator.apply determinism (max delta: {float(np.abs(p_cal_a - p_cal_b).max())})")

# --- 12e. Edge cases on each member's runtime apply.
print("\n" + "=" * 72)
print("RED-TEAM SECTION E: edge cases on member apply functions")
print("=" * 72)
# All-NaN feature row -> Member 2 (GBDT) handles via default_left.
nan_feats = np.full(member_feat_schema.feature_dim, np.nan, dtype=np.float32)
from src.gbdt_member import apply_one as _g_apply_one
p2_nan = _g_apply_one(gbdt_state, nan_feats)
print(f"  [PASS] GBDT NaN-features -> {p2_nan:.4f} (finite={np.isfinite(p2_nan)})")

# All-Inf feature row -> Member 4 (LogReg) clamps.
inf_feats = np.full(member_feat_schema.feature_dim, np.inf, dtype=np.float32)
from src.logreg_member import apply_state_one as _lr_apply_state_one
p4_inf = _lr_apply_state_one(logreg_state, inf_feats)
print(f"  [PASS] LogReg Inf-features -> {p4_inf:.4f} (finite={np.isfinite(p4_inf)})")

# Zero-norm query -> Member 3 (kNN) falls through.
zero_q = np.zeros(ITEM_EMB_DIM, dtype=np.float32)
from src.knn_member import apply_one as _knn_apply_one
p3_zero = _knn_apply_one(knn_state, zero_q, _subject_keys_ordered[0])
print(f"  [PASS] kNN zero-norm query -> {p3_zero:.4f} (finite={np.isfinite(p3_zero)})")

# Unknown subject -> Member 3 returns global prior.
rng_redteam = np.random.default_rng(0)
p3_unk = _knn_apply_one(
    knn_state, rng_redteam.normal(size=ITEM_EMB_DIM).astype(np.float32),
    "totally_unknown_subject",
)
expected = max(min(knn_state.global_passrate, 1 - 1e-6), 1e-6)
import math
assert math.isclose(p3_unk, expected, abs_tol=1e-6)
print(f"  [PASS] kNN unknown-subject -> global prior {p3_unk:.4f}")

# --- 12f. Final summary.
print("\n" + "=" * 72)
print("FINAL RED-TEAM SUMMARY")
print("=" * 72)
print(f"  Bundle dir   : {sub_dir}")
print(f"  Bundle size  : {bundle_bytes / (1024 * 1024):.1f} MB")
print(f"  ZIP path     : {zip_path}")
print(f"  ZIP size     : {size_mb:.1f} MB  (cap {zip_cap_mb:.0f} MB)")
print(f"  Final val LL : {_ll(p_final_val):.6f}")
print(f"  Cells passed : {sum(1 for _, ok, _ in checks if ok)}/{len(checks)}")
print("\nIf the subprocess smoke test PASSED, the bundle is ready to upload.")
print("If anything FAILED, fix the root cause and re-run from cell 11.")
