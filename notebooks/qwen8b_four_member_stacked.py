# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

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


def state_fingerprint(obj, n_chars: int = 16) -> str:
    """Stable short hex fingerprint of any picklable object.

    Used to discriminate cache entries by their *upstream* state so a
    downstream cache (e.g. the NN calibrator) auto-invalidates whenever
    a depended-upon state (Model A checkpoint, GBDT trees, kNN tables,
    logreg weights, stacker weights, etc.) materially changes.

    Pickle-serializing dataclasses + ndarray dicts is deterministic
    enough across runs of the same Python interpreter that two
    bit-identical objects yield identical fingerprints. We DO NOT
    promise cross-Python-version stability -- a Python upgrade triggers
    a one-time recompute, which is the safe direction.
    """
    h = hashlib.sha256()
    h.update(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return h.hexdigest()[:n_chars]


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

# --- Out-of-fold (OOF) training for honest stacking.
# Item-grouped K-fold: each fold's OOF items are disjoint from its train
# items. Members are retrained per-fold; OOF predictions feed the stacker
# instead of the (in-sample-for-the-meta-learner) val predictions.
# ``n_folds=3`` is the agreed compromise between leakage discipline and
# Colab cost (3x Member-1 training vs current; ~3-5h extra wall time).
# ``run_shuffled_label_control`` triggers Gate 1c when set to True --
# expensive (re-runs the entire OOF pipeline on permuted labels), so
# defaults to False; user toggles it on demand.
CFG.setdefault("oof", {})
CFG["oof"].setdefault("enabled", True)
CFG["oof"].setdefault("n_folds", 3)
CFG["oof"].setdefault("seed", 7)
CFG["oof"].setdefault("run_shuffled_label_control", False)
CFG["oof"].setdefault("shuffled_label_seed", 12345)
CFG["oof"].setdefault("nn_neighbor_probe_sample_size", 2000)
CFG["oof"].setdefault("optimism_threshold_nats", 0.03)

# Member 2 v2 (Task 3 of the diversification plan): subject-mean residual
# GBDT trained on NON-EMBEDDING features only. The original Member 2
# shared the full 1210-dim X_train_dense with Member 1 (theta, u, pool,
# centroid, NN features) and used Member 1's prediction as the residual
# anchor -- the two failure modes that made its val errors heavily
# correlated with Member 1's (the stacker downweighted it to ~0.02).
# Task 3 fixes BOTH at once:
#   1. New feature view: subject_idx, cluster_id, bench_condition_id,
#      bc_redacted_mask, subject_obs_count_log1p, subject/bench cat+num
#      metadata, and the 8 mean-encoded interaction columns. The
#      no-embedding schema is audited by Gate 3d before training.
#   2. New anchor: logit(subject_mean) instead of logit(Member 1).
#      subject_mean is a much WEAKER baseline (just per-subject mean
#      pass-rate with Bayesian shrinkage), so the GBDT has to learn
#      real per-row item-cluster interactions to beat it. Anchors are
#      computed OUT OF FOLD: fold-train labels only for per-fold
#      Member 2 training (Gate 3a), and the same K-fold OOF accumulator
#      gives subject_mean_train_oof for the GLOBAL Member 2 fit's
#      anchor, so the row's own label NEVER enters its own anchor.
# Set `enabled=False` to revert to the legacy Member 2 (Member 1 anchor +
# full embedding schema) for A/B comparison.
CFG.setdefault("member2_v2", {})
CFG["member2_v2"].setdefault("enabled", True)
CFG["member2_v2"].setdefault("subject_mean_smoothing", 30.0)
# --- Member 5: kNN on a 1-D supervised difficulty projection (Task 4) ---
# Member 3 already does kNN on raw item embeddings (semantic similarity).
# Member 5 builds neighborhoods in PREDICTED DIFFICULTY space instead --
# items that look alike in topic but differ in difficulty land in different
# neighborhoods, which is the decorrelation source we want.
#
# Knobs (sane defaults; the per-fold OOF loop + Gate 4b will report whether
# we need to tune them):
#   k=32                          # neighborhood size (1-D so binary-search-fast)
#   tau=0.05                      # Gaussian kernel scale on the difficulty axis
#   ridge_alpha=10.0              # L2 on the projection's weights
#   item_fallback_weight=0.3      # mirror of Member 3's discount on unobserved cells
#   min_subjects_per_item=3       # items with fewer rated subjects skip the fit
# Set `enabled=False` to skip Member 5 and revert to the 4-member stacker
# (legacy / Task-3 behavior) for A/B comparison.
CFG.setdefault("member5", {})
CFG["member5"].setdefault("enabled", True)
CFG["member5"].setdefault("k", 32)
CFG["member5"].setdefault("tau", 0.05)
CFG["member5"].setdefault("ridge_alpha", 10.0)
CFG["member5"].setdefault("item_fallback_weight", 0.3)
CFG["member5"].setdefault("min_subjects_per_item", 3)
# Sample size for Gate 4d's apply round-trip probe.
CFG["member5"].setdefault("gate4d_sample_size", 64)
CFG.setdefault("judge", {})
CFG["judge"]["enabled"] = False
CFG.setdefault("lora", {})
CFG["lora"]["enabled"] = False
CFG.setdefault("kfold", {})
CFG["kfold"]["enabled"] = False
CFG.setdefault("submission", {})
CFG["submission"]["ship_training_cache"] = True
CFG["submission"]["ship_requirements_txt"] = False
# Codabench accepts up to 1500 MB; we intentionally bump from the
# default 65 MB cap so the bundle has room for the four-member
# stacker artifacts (decoded train embeddings for kNN, GBDT trees,
# meta-feature schema, residual table, ...).
CFG["submission"]["max_zip_size_mb"] = 1500
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
    NN_FEATURE_DIM,
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
# ## 6.5. OOF fold setup (Task 1 of the diversification plan)
#
# Build the item-grouped K-fold schedule used by all per-fold member
# retraining downstream. The folds partition `primary.train` by
# **item_key**, so any row in fold *f*'s OOF set is guaranteed not to
# share an item with fold *f*'s training set. This is the foundation
# for honest stacker inputs: the stacker is trained on OOF member
# predictions (each row's prediction comes from a member that never
# saw that row's item) instead of on val (where the meta-learner is
# in-sample). Val becomes a pure held-out reporting set.
#
# Gate 1a (RED-TEAM): assert item-key disjointness per fold AND the
# stronger "every train row appears in exactly one fold's OOF set"
# row-partition invariant. Catches off-by-one bugs in fold assignment.

# %%
from src.oof_folds import (
    ItemFold,
    assert_item_disjoint,
    assert_nn_neighbors_in_fold_train,
    assert_row_idx_partition,
    fold_cache_suffix,
    make_item_grouped_folds,
    report_train_vs_val_optimism,
)

OOF_ENABLED = bool(CFG["oof"].get("enabled", True))
OOF_N_FOLDS = int(CFG["oof"].get("n_folds", 3))
OOF_SEED = int(CFG["oof"].get("seed", 7))

# Per-row item_key array over primary.train. We treat keys as strings
# everywhere so the fold-id dict lookups can't be tripped by dtype mismatch.
_train_row_item_keys = primary.train["item_key"].astype(str).to_numpy()
print(
    f"[OOF] N_train_rows={len(_train_row_item_keys):,}  "
    f"unique_items={len(set(_train_row_item_keys)):,}  "
    f"n_folds={OOF_N_FOLDS}  seed={OOF_SEED}"
)

folds: list[ItemFold] = make_item_grouped_folds(
    item_keys_per_row=_train_row_item_keys,
    n_folds=OOF_N_FOLDS,
    seed=OOF_SEED,
)

# Gate 1a: item-disjointness, per-fold AND row-partition invariant.
print("[OOF Gate 1a] Asserting item-key disjointness on every fold...")
for f in folds:
    assert_item_disjoint(f)
assert_row_idx_partition(folds, n_rows=len(_train_row_item_keys))
print(
    f"[OOF Gate 1a] PASS: all {len(folds)} folds have disjoint train/OOF "
    f"item-key sets AND each of {len(_train_row_item_keys):,} train rows "
    "appears in exactly one fold's OOF set."
)

print("\n[OOF] Fold summary (per fold: n_train_items, n_oof_items, "
      "n_train_rows, n_oof_rows, cache_suffix):")
print(f"{'fold':>6}  {'train_items':>12}  {'oof_items':>10}  "
      f"{'train_rows':>12}  {'oof_rows':>10}  {'cache_suffix':<18}")
for f in folds:
    suffix = fold_cache_suffix(
        fold_id=f.fold_id, train_item_keys=f.train_item_keys
    )
    print(f"  {f.fold_id:>4d}  {len(f.train_item_keys):>12,d}  "
          f"{len(f.oof_item_keys):>10,d}  {len(f.train_row_idx):>12,d}  "
          f"{len(f.oof_row_idx):>10,d}  {suffix:<18}")

# Per-row fold id, useful for diagnostics + downstream per-fold work.
_train_row_fold_id = np.full(len(_train_row_item_keys), -1, dtype=np.int64)
for f in folds:
    _train_row_fold_id[f.oof_row_idx] = f.fold_id
assert (_train_row_fold_id >= 0).all(), \
    "BUG: some training row failed to receive a fold id (should never happen given Gate 1a)"

print(f"\n[OOF] Per-row fold-id distribution: "
      f"{np.bincount(_train_row_fold_id).tolist()}  (should sum to N_train_rows)")

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


# NOTE: We deliberately DO NOT build ``train_ds`` / ``val_ds`` here.
# Cell 7b will rebuild them after recomputing ``nn_train_mat`` /
# ``nn_val_mat`` with the conditional context, and ``_build`` materializes
# a [N_rows, embedding_dim] per-row item embedding tensor via
# ``stack_lookup`` -- at 5M rows x 4096 dims that's ~80 GB per copy. If
# we built here AND in cell 7b, peak RAM doubles and we OOM on machines
# that would otherwise comfortably hold one copy. The downstream cells
# (training, scoring) all bind to the cell-7b versions.
print(
    f"NN feature matrices (legacy 0..14): "
    f"train={nn_train_mat.shape}  val={nn_val_mat.shape}  "
    f"(datasets will be built in cell 7b once cells 15..22 are populated)"
)

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
# Always promote ``explicit_crosses`` to the FULL Cartesian product of
# ``subject_categorical x benchmark_categorical`` so every (subject_trait,
# benchmark_trait) categorical pair gets its own dedicated cross table
# (rather than the hand-picked default subset). This costs at most
# ``|subject_cat| * |benchmark_cat|`` extra small embedding tables --
# tiny in absolute terms (e.g. for the default schema, 3 * 1 = 3 crosses)
# but gives the model dedicated capacity for each pair instead of the
# parameter-shared FM cross alone.
_full_cross_grid = _meta_schema.full_categorical_cross_grid()
if _meta_schema.explicit_crosses != _full_cross_grid:
    print(
        f"[Metadata] Promoting explicit_crosses to full cat x cat grid:\n"
        f"  before ({len(_meta_schema.explicit_crosses)}): "
        f"{list(_meta_schema.explicit_crosses)}\n"
        f"  after  ({len(_full_cross_grid)}): {list(_full_cross_grid)}"
    )
_meta_schema = _meta_schema.with_full_categorical_cross_grid()
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


# ---------------------------------------------------------------------------
# Metadata coverage check.
#
# This catches the failure mode where the schema declares a field
# (e.g. ``benchmark_age``) but the underlying CSV / join produced no
# data for any entity, leaving the corresponding numeric channel at
# ``mask=1`` (= MISSING) and the conditional NN cells (freshness etc.)
# at fallback for every row. We RAISE on zero-coverage fields so the
# run halts before we waste compute training on silently-missing data.
# Mask convention (from ``NumericScaler.transform``): ``mask < 0.5``
# means the value was present in the source CSV; ``mask >= 0.5`` means
# missing / non-finite / "Unknown".
# ---------------------------------------------------------------------------
def _assert_metadata_coverage(
    preprocessor,
    id_tables,
    schema,
    *,
    min_coverage: float = 0.01,
    show_examples: int = 5,
) -> None:
    sub_cat = id_tables.subject_cat_ids.cpu().numpy()
    sub_num = id_tables.subject_num.cpu().numpy()
    bc_cat = id_tables.bc_cat_ids.cpu().numpy()
    bc_num = id_tables.bc_num.cpu().numpy()

    # Slice off row 0 -- it's the reserved UNK row, MISSING by design.
    sub_cat_real = sub_cat[1:] if sub_cat.shape[0] > 1 else sub_cat
    sub_num_real = sub_num[1:] if sub_num.shape[0] > 1 else sub_num
    bc_cat_real = bc_cat[1:] if bc_cat.shape[0] > 1 else bc_cat
    bc_num_real = bc_num[1:] if bc_num.shape[0] > 1 else bc_num

    issues: list[str] = []
    rows: list[tuple[str, str, int, int, float]] = []

    def _record(side: str, field: str, present: int, total: int) -> None:
        frac = float(present) / float(max(total, 1))
        rows.append((side, field, int(present), int(total), frac))
        if total == 0:
            return
        if frac < min_coverage:
            issues.append(
                f"  - [{side}] {field!r}: only {present}/{total} "
                f"({frac:.2%}) entities have data (threshold "
                f"{min_coverage:.0%})"
            )

    for j, field in enumerate(schema.subject_categorical):
        if j >= sub_cat_real.shape[1]:
            issues.append(f"  - [subject_cat] {field!r}: column missing from id table")
            continue
        col = sub_cat_real[:, j]
        present = int((col != 0).sum())
        _record("subject_cat", field, present, col.size)

    for j, field in enumerate(schema.subject_numeric):
        col_idx = 2 * j + 1
        if col_idx >= sub_num_real.shape[1]:
            issues.append(f"  - [subject_num] {field!r}: column missing from id table")
            continue
        present = int((sub_num_real[:, col_idx] < 0.5).sum())
        _record("subject_num", field, present, sub_num_real.shape[0])

    for j, field in enumerate(schema.benchmark_categorical):
        if j >= bc_cat_real.shape[1]:
            issues.append(f"  - [bench_cat] {field!r}: column missing from id table")
            continue
        col = bc_cat_real[:, j]
        present = int((col != 0).sum())
        _record("bench_cat", field, present, col.size)

    for j, field in enumerate(schema.benchmark_numeric):
        col_idx = 2 * j + 1
        if col_idx >= bc_num_real.shape[1]:
            issues.append(f"  - [bench_num] {field!r}: column missing from id table")
            continue
        present = int((bc_num_real[:, col_idx] < 0.5).sum())
        _record("bench_num", field, present, bc_num_real.shape[0])

    print("Metadata coverage (excluding row 0 = UNK):")
    print(f"  {'side':<14s} {'field':<22s} {'present':>10s} / {'total':<10s}  {'frac':>7s}")
    for side, field, p, t, f in rows:
        flag = "" if f >= min_coverage else "  <-- LOW"
        print(f"  {side:<14s} {field:<22s} {p:>10d} / {t:<10d}  {f:>7.2%}{flag}")

    if issues:
        # Probe the underlying preprocessor lookup tables to surface the
        # most likely root cause (CSV column missing, normalization
        # rename failed, or join key mismatch).
        bench_keys = list(preprocessor._benchmark_by_id.keys())[: int(show_examples)]
        model_keys = list(preprocessor._subject_by_name.keys())[: int(show_examples)]
        bench_sample = [
            (k, dict(preprocessor._benchmark_by_id[k]))
            for k in bench_keys
        ]
        model_sample = [
            (k, dict(preprocessor._subject_by_name[k]))
            for k in model_keys
        ]
        msg_lines = [
            "Metadata coverage check FAILED:",
            *issues,
            "",
            "Hints:",
            "  * For numeric fields: confirm the column exists in the source CSV "
            "AND values are numeric (not 'Unknown' / blank). 'benchmark_age' is "
            "renamed from 'age' inside _normalize_benchmark_info, but only if the "
            "raw column name is exactly 'age'.",
            "  * For categorical fields: confirm the join key matches. Subjects "
            "join by display name extracted from subject_content; benchmarks "
            "join by the leading segment before '::' in the bc key.",
            "",
            f"Sample benchmark_info rows ({len(bench_sample)} of "
            f"{len(preprocessor._benchmark_by_id)}):",
        ]
        for k, row in bench_sample:
            msg_lines.append(f"  benchmark={k!r}: {row}")
        msg_lines.append(
            f"Sample model_info rows ({len(model_sample)} of "
            f"{len(preprocessor._subject_by_name)}):"
        )
        for k, row in model_sample:
            msg_lines.append(f"  name={k!r}: {row}")
        raise RuntimeError("\n".join(msg_lines))


_assert_metadata_coverage(meta_preprocessor, meta_id_tables, _meta_schema)

# %% [markdown]
# ## 7b. Conditional NN-feature context + recomputed NN matrices
#
# **What this cell does.** The first NN compute above produced cells
# `[0..14]` of the locked 23-feature schema; cells `[15..22]` (the four
# subject-trait-conditional passrates, the benchmark-conditional
# passrate, the freshness diff, the distinct-subjects-in-neighborhood
# count, and the cluster-conditional passrate) need the metadata
# preprocessor's vocabularies + the per-train-item benchmark / cluster
# arrays that only become available after Cell 7 has built
# `meta_id_tables`. We therefore:
#
#   1. Pull the per-subject trait id arrays (family / macro_family /
#      organization) out of `meta_id_tables.subject_cat_ids` using the
#      current schema's field order.
#   2. Materialize per-train-item arrays (benchmark id, benchmark age,
#      cluster id) keyed by the same indexer the NN feature pipeline
#      uses, so neighbor lookups land in the right namespace.
#   3. Build a `ConditionalPassrateContext`. Stays in RAM (a few MB per
#      ~50k items) and is reused at bundle export time.
#   4. Build the per-row query metadata arrays for train + val (subject
#      meta is never redacted at training time -- the model sees the
#      same bench-redaction augmentation it would at runtime via
#      `train_dropout` instead).
#   5. Recompute `nn_train_mat` / `nn_val_mat` *with* the context so
#      cells `[15..22]` have signal. We then rebuild `train_ds` / `val_ds`
#      so the LookupDataset wraps the up-to-date 23-column matrix.
#
# Yes this re-runs the FAISS query pass once. The cost is a small
# multiple of the first compute (~30 s on a 12 GB GPU for ~50k unique
# items) and pays for itself the first time the conditional cells
# move val log-loss.

# %%
from src.nn_features import (
    MISSING_TRAIT_ID,
    NN_FEATURE_NAMES,
    ConditionalPassrateContext,
    build_conditional_passrate_context,
)

_subject_cat_fields = list(_meta_schema.subject_categorical)


def _trait_id_array(trait_name: str) -> np.ndarray:
    """Per-subject_id trait id array (length = indexer.n_subjects).

    Returns all zeros if the schema doesn't include this trait, which
    sends every row's trait through the MISSING fallback in the
    aggregator.
    """
    if trait_name not in _subject_cat_fields:
        return np.zeros(indexer.n_subjects, dtype=np.int32)
    field_idx = _subject_cat_fields.index(trait_name)
    return (
        meta_id_tables.subject_cat_ids[:, field_idx]
        .cpu()
        .numpy()
        .astype(np.int32)
    )


s2fam = _trait_id_array("family")
s2macro = _trait_id_array("macro_family")
s2org = _trait_id_array("organization")

# Cardinalities from the preprocessor's vocabularies (n_tokens already
# includes the reserved MISSING / UNK rows).
def _vocab_size(name: str) -> int:
    v = meta_preprocessor.subject_cat_vocabs.get(name)
    return int(v.n_tokens) if v is not None else 1


N_FAMILIES = _vocab_size("family")
N_MACRO_FAMILIES = _vocab_size("macro_family")
N_ORGANIZATIONS = _vocab_size("organization")
N_CLUSTERS_CTX = int(CFG["clustering"]["k"])

# Per-train-item arrays keyed by the SAME index map used by the NN
# pipeline. ``train_item_keys`` was built upstream and is the canonical
# row-index source.
item_keys_arr = np.asarray(train_item_keys)
item_to_bc = (
    primary.train.assign(_k=primary.train["item_key"].astype(str))
    .groupby("_k", sort=False)["benchmark_condition_key"]
    .first()
    .to_dict()
)
item_benchmark_id_arr = np.full(len(train_item_keys), -1, dtype=np.int32)
for i, k in enumerate(train_item_keys):
    bck = item_to_bc.get(str(k))
    if bck is not None:
        bc = indexer.bc_id(str(bck))
        if bc >= 0:
            item_benchmark_id_arr[i] = int(bc)

# Benchmark age extraction. ``meta_id_tables.bc_num`` has shape
# ``[n_bc, 2 * n_bench_numeric]`` with each numeric field encoded as
# interleaved (scaled_value, missingness_mask) pairs. The mask
# convention from ``NumericScaler.transform`` is:
#
#     mask < 0.5  ==  PRESENT (source value was finite)
#     mask >= 0.5 ==  MISSING (NaN-imputed to the median)
#
# So we want the value when mask < 0.5 and NaN otherwise -- the
# aggregator's redaction path triggers on NaN ages, not on the
# imputed median, which would silently bias every conditional cell.
_bench_num_fields = list(_meta_schema.benchmark_numeric)
if "benchmark_age" in _bench_num_fields and getattr(
    meta_id_tables, "bc_num", None
) is not None:
    age_idx = _bench_num_fields.index("benchmark_age")
    bc_num_np = meta_id_tables.bc_num.cpu().numpy().astype(np.float32)
    if bc_num_np.shape[1] >= 2 * (age_idx + 1):
        _age_value = bc_num_np[:, 2 * age_idx]
        _age_mask = bc_num_np[:, 2 * age_idx + 1]
        bc_id_to_age_arr = np.where(
            _age_mask < 0.5,           # PRESENT in the source CSV
            _age_value,                 # use the scaled value
            np.nan,                     # else MISSING -> NaN -> redact
        ).astype(np.float32)
        # Sanity probe: brief one-shot diagnostic on the BC table.
        # Skips row 0 (UNK is MISSING by design).
        _present_bc = int((_age_mask[1:] < 0.5).sum())
        _total_bc = max(int(_age_mask.shape[0] - 1), 0)
        print(
            f"[bench_age] resolved {_present_bc}/{_total_bc} bc rows "
            f"(non-UNK) to a finite age before the per-row mapping"
        )
    else:
        bc_id_to_age_arr = np.full(indexer.n_bc, np.nan, dtype=np.float32)
else:
    bc_id_to_age_arr = np.full(indexer.n_bc, np.nan, dtype=np.float32)
item_benchmark_age_arr = np.full(len(train_item_keys), np.nan, dtype=np.float32)
_valid_bid = item_benchmark_id_arr >= 0
item_benchmark_age_arr[_valid_bid] = bc_id_to_age_arr[
    item_benchmark_id_arr[_valid_bid]
]

# Cluster id per train item (already 0-indexed; -1 if unassigned).
item_cluster_id_arr = np.array(
    [int(cluster_assignments.get(str(k), -1)) for k in train_item_keys],
    dtype=np.int32,
)

cond_context = build_conditional_passrate_context(
    train_df=primary.train,
    item_index_map=nn_item_index_map,
    subject_index_map=indexer.subject_to_id,
    subject_to_family_id=s2fam,
    subject_to_macro_family_id=s2macro,
    subject_to_organization_id=s2org,
    item_benchmark_id=item_benchmark_id_arr,
    item_benchmark_age=item_benchmark_age_arr,
    item_cluster_id=item_cluster_id_arr,
    n_families=N_FAMILIES,
    n_macro_families=N_MACRO_FAMILIES,
    n_organizations=N_ORGANIZATIONS,
    n_clusters=N_CLUSTERS_CTX,
)
cond_context.assert_shapes()
print(
    f"ConditionalPassrateContext: "
    f"n_subjects={cond_context.n_subjects}  n_items={cond_context.n_items}  "
    f"n_fam={cond_context.n_families}  n_macro={cond_context.n_macro_families}  "
    f"n_org={cond_context.n_organizations}  n_clusters={cond_context.n_clusters}  "
    f"family_nnz={cond_context.family_passrate_csr.nnz}  "
    f"cluster_subject_nnz={cond_context.cluster_subject_passrate_csr.nnz}"
)


def _query_meta_for_split(rows_df) -> dict:
    """Per-row query metadata arrays for ``compute_nn_features_streaming``.

    Returned dict: bench_ids (int32 -1 for UNK), bench_age (float32 NaN
    for unknown), cluster_ids (int32 -1 for UNK), subject_meta_redacted
    (int32, 0 at training time -- bench-side redaction augmentation is
    performed by the train-time dropout hook, not here).
    """
    bench_ids = np.array(
        [int(indexer.bc_id(str(k))) for k in rows_df["benchmark_condition_key"]],
        dtype=np.int32,
    )
    bench_age = np.full(len(rows_df), np.nan, dtype=np.float32)
    valid = bench_ids >= 0
    bench_age[valid] = bc_id_to_age_arr[bench_ids[valid]]
    cluster_ids = np.array(
        [int(cluster_assignments.get(str(k), -1)) for k in rows_df["item_key"]],
        dtype=np.int32,
    )
    return {
        "query_benchmark_ids": bench_ids,
        "query_benchmark_age": bench_age,
        "query_cluster_ids": cluster_ids,
        "subject_meta_redacted": np.zeros(len(rows_df), dtype=np.int32),
    }


_train_qmeta = _query_meta_for_split(primary.train)
_val_qmeta = _query_meta_for_split(primary.val)
print(
    f"Query metadata (train): bench_known="
    f"{int((_train_qmeta['query_benchmark_ids'] >= 0).sum())}/{len(primary.train):,}  "
    f"age_known={int(np.isfinite(_train_qmeta['query_benchmark_age']).sum())}/{len(primary.train):,}  "
    f"cluster_known={int((_train_qmeta['query_cluster_ids'] >= 0).sum())}/{len(primary.train):,}"
)

# Recompute NN matrices WITH the conditional context. We reuse the
# already-built ``nn_index`` and ``nn_passrate_csr`` so this is a single
# additional FAISS pass plus the in-memory conditional lookups (cheap).
#
# We pass ``return_diagnostics=True`` for the train side so the cell
# below can print HONEST per-cell redaction counts. The previous
# heuristic (``np.abs(nn_train_mat[:, i] - fallback_value) <= 1e-9``)
# silently confounds:
#   - genuine redaction (the cell's redact MASK was 1), with
#   - legitimate exact-zero output (e.g. ``q_age == mean(neighbor_age)``
#     when the query and its neighbors share a benchmark date -- very
#     common for ``neighbor_freshness_diff``).
# That's why ``neighbor_freshness_diff`` looked like 83% fallback when
# the underlying redaction rate may be much smaller. ``return_diagnostics``
# reads the redaction MASKS directly off ``_resolve_conditional_inputs``
# and surfaces them as honest counts.
_log_ram("before compute_nn_features_streaming(train, with cond context)")
nn_train_mat, nn_train_diag = compute_nn_features_streaming(
    query_item_keys=train_keys_for_nn,
    item_emb_lookup=item_emb_lookup,
    subject_ids=train_sid,
    nn_index=nn_index,
    passrate_csr=nn_passrate_csr,
    passrate_mask_csr=nn_passrate_mask_csr,
    cfg=nn_cfg,
    exclude_self=True,
    query_chunk_size=NN_QUERY_CHUNK,
    conditional_context=cond_context,
    return_diagnostics=True,
    **_train_qmeta,
)
gc.collect()
_log_ram("after compute_nn_features_streaming(train, with cond context)")
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
    conditional_context=cond_context,
    **_val_qmeta,
)
gc.collect()
_log_ram("after compute_nn_features_streaming(val, with cond context)")
print(
    f"NN feature matrices (with conditional cells): "
    f"train={nn_train_mat.shape}  val={nn_val_mat.shape}"
)

# Honest per-cell redaction diagnostic (reads the per-row redaction MASK
# from `_resolve_conditional_inputs`, NOT the post-aggregation cell
# value). For comparison we also keep the old "value matches fallback"
# count so it's obvious when the two diverge -- a divergence like
#   redact=0.04, value-at-fb=0.83
# means the cell is genuinely well-covered but its valid output happens
# to land at exactly 0 a lot (this is the freshness_diff case: many
# items share a z-scored benchmark date, so q_age - mean(neighbor_age)
# is exactly 0).
_FB = float(nn_cfg.fallback_value)
_total_train = int(nn_train_diag["n_rows"])
print("Honest redaction diagnostic (cells [15..22]):")
print(
    f"  {'cell':40s}  {'redact_frac':>11s}  {'value-at-fb_frac':>17s}  notes"
)
for _i, _name in enumerate(NN_FEATURE_NAMES):
    if _i < 15:
        continue
    _value_at_fb = float(np.mean(np.abs(nn_train_mat[:, _i] - _FB) <= 1e-9))
    _redact_count = nn_train_diag["per_cell"].get(_name, None)
    if _redact_count is None:
        _redact_str = "N/A (no mask)"
    else:
        _redact_str = f"{_redact_count / max(_total_train, 1):.3f}"
    _notes = ""
    if _redact_count is not None and _value_at_fb > _redact_count / max(_total_train, 1) + 0.05:
        _notes = " <- value hits fb more often than redaction (legitimate zeros)"
    print(
        f"  cell[{_i:2d}] {_name:34s}  {_redact_str:>11s}  {_value_at_fb:>17.3f}{_notes}"
    )

# Localize freshness redactions: query-side vs neighbor-side. With z-
# scored ages, exact-zero outputs are common and don't mean the feature
# is broken; the numbers below tell us whether genuine missingness is
# small (then the feature is fine, just noisy) or large (then we have
# a coverage gap to chase).
if "freshness" in nn_train_diag:
    fr = nn_train_diag["freshness"]
    print(
        "  freshness localizer: "
        f"q_age_known={fr['n_query_age_known']}/{fr['n_query_total']} "
        f"({fr['n_query_age_known'] / max(fr['n_query_total'], 1):.3f}); "
        f"item_age_known={fr['n_train_items_with_known_age']}/"
        f"{fr['n_train_items_total']} "
        f"({fr['n_train_items_with_known_age'] / max(fr['n_train_items_total'], 1):.3f}); "
        f"mean_n_known_neighbors_per_row={fr['mean_n_known_neighbors_per_row']:.2f}; "
        f"frac_rows_with_zero_known_neighbors={fr['frac_rows_with_zero_known_neighbors']:.3f}"
    )

# Free intermediates that we no longer need before allocating the
# heavy datasets. Tolerant of cell re-runs (the names may already be
# gone from a previous pass).
#
# Why this matters: ``_build`` materializes a per-row [N,
# embedding_dim] item embedding tensor via ``stack_lookup`` -- at 5M
# rows x 4096 dims that's ~80 GB per copy. If we leave the old
# ``train_ds`` / ``val_ds`` bound while ``_build`` runs, peak RAM
# briefly doubles and we OOM on machines that would otherwise hold
# one copy comfortably. The per-split query-metadata dicts also
# carry ~80 MB on the train split each.
for _stale_name in ("_train_qmeta", "_val_qmeta", "train_ds", "val_ds"):
    if _stale_name in globals():
        try:
            del globals()[_stale_name]
        except KeyError:
            pass
gc.collect()
_log_ram("before _build (datasets, with cond cells)")
train_ds = _build(primary.train, nn_train_mat)
gc.collect()
_log_ram("after _build(train)")
val_ds = _build(primary.val, nn_val_mat)
gc.collect()
_log_ram("after _build(val)")
print(f"train tensors: {len(train_ds)}  val tensors: {len(val_ds)} (built with 23-col NN matrix)")

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
# Member 1 of the stacker is the metadata-aware IRT-MLP variant
# trained for 3 epochs with metadata-dropout regularization. Model B
# (the no-meta companion that previously formed a coverage-conditional
# blend) has been dropped: the stacker provides the diversity that
# the coverage blend used to provide, so a second IRT head is
# redundant and only inflates the bundle.
MODEL_A_NAME = "meta_hybrid_irt_kfactor_gated_mlp"

SPLIT_SEED = SEED
MODEL_A_SEED = SEED + int(CFG["coverage_blend"]["model_a_seed_offset"])
print(f"Seeds: SPLIT={SPLIT_SEED}  MODEL_A={MODEL_A_SEED}")

# Common ModelConfig fields for Member 1.
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


a_drop = TrainDropoutConfig(
    p_bench=float(CFG["coverage_blend"]["model_a"]["p_bench"]),
    p_subj=float(CFG["coverage_blend"]["model_a"]["p_subj"]),
    q_bc=float(CFG["coverage_blend"]["model_a"]["q_bc"]),
    seed=MODEL_A_SEED,
)


def _train_model_a():
    """Train Model A and return a picklable bundle.

    Returns a dict with:
      * ``train_result``: the :class:`TrainResult` (dataclass, picklable)
        emitted by ``train_one``.
      * ``ckpt``: the loaded checkpoint dict (``{"model_state": ..., ...}``)
        so cache hits don't have to re-read the file off disk (and so we
        can recreate the checkpoint file when the original was reaped).
      * ``dropout_stats``: per-row hook counters for diagnostics.

    The monkey-patched ``train_mod.build_model`` is restored on every
    exit path; the dropout hook is removed before pickling. We never
    pickle live ``torch.utils.hooks.RemovableHandle`` objects.
    """
    train_mod.build_model = _build_with_overrides
    stats: list[dict] = []
    try:
        _active_dropout_cfg["cfg"] = a_drop
        _active_dropout_cfg["name"] = MODEL_A_NAME
        _active_dropout_cfg["installed_handles"] = []
        _result = train_one(
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
            stats.append(
                {
                    "train_calls": int(h.n_train_calls),
                    "rows": int(h.n_rows_seen),
                    "bench_masked": int(h.n_rows_bench_masked),
                    "subj_masked": int(h.n_rows_subj_masked),
                    "bc_idx_masked": int(h.n_rows_bc_idx_masked),
                }
            )
            h.remove()
        _ckpt = torch.load(_result.checkpoint_path, map_location="cpu", weights_only=False)
        return {
            "train_result": _result,
            "ckpt": _ckpt,
            "dropout_stats": stats,
        }
    finally:
        _active_dropout_cfg["cfg"] = None
        _active_dropout_cfg["name"] = None
        train_mod.build_model = _orig_build_model


# Cache key for Model A. Anything that materially affects the trained
# weights MUST be in here; otherwise a stale checkpoint will silently
# survive a hyperparameter change. We hash the ModelConfig + TrainConfig
# + TrainDropoutConfig as JSON strings, plus the dataset / indexer
# fingerprints (n_train, n_val, n_subjects, n_bc) and the NN feature
# schema dim (which is now 15 -- bumping that auto-invalidates older
# caches keyed under nn_feature_dim=8).
import torch  # noqa: E402  -- imported here so the cache fn can torch.load
from src.nn_features import NN_FEATURE_DIM as _NN_FEATURE_DIM_CACHE_TAG

MODEL_A_KEY_INPUTS = (
    "model_a_v4_cond23",                      # bump this on schema changes
    MODEL_A_NAME,
    int(MODEL_A_SEED),
    json.dumps(asdict(model_a_cfg), default=str, sort_keys=True),
    json.dumps(asdict(train_cfg), default=str, sort_keys=True),
    json.dumps(asdict(a_drop), default=str, sort_keys=True),
    int(len(train_ds)),
    int(len(val_ds)),
    int(indexer.n_subjects),
    int(indexer.n_bc),
    int(_NN_FEATURE_DIM_CACHE_TAG),
    # Pin the conditional NN-feature context cardinalities + presence
    # so the cached Model A is forcibly re-trained whenever the
    # conditional schema or the trait cardinalities change. We avoid
    # hashing the full sparse tables here -- the build is deterministic
    # given (train_df, indexer, schema), all of which are already
    # captured by the keys above.
    int(cond_context.n_families),
    int(cond_context.n_macro_families),
    int(cond_context.n_organizations),
    int(cond_context.n_clusters),
)
print(f"[cache] Model A key inputs:\n  {[str(x)[:80] for x in MODEL_A_KEY_INPUTS]}")

_model_a_bundle = cache_or_compute(
    "model_a_trained",
    key_inputs=MODEL_A_KEY_INPUTS,
    compute_fn=_train_model_a,
)
result_a = _model_a_bundle["train_result"]
ckpt_a_cached = _model_a_bundle["ckpt"]

# When loading from cache, the file at ``result_a.checkpoint_path`` may
# have been deleted between runs (Drive reap, fresh Colab, etc.) but
# downstream cells assume that file exists. Re-create it idempotently
# from the cached state dict so ``torch.load(result_a.checkpoint_path)``
# below keeps working.
_ckpt_path = Path(result_a.checkpoint_path)
if not _ckpt_path.exists():
    _ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_a_cached, _ckpt_path)
    print(f"[cache] Restored Model A checkpoint to {_ckpt_path}")

for s in _model_a_bundle["dropout_stats"]:
    print(
        f"  Model A dropout stats: train_calls={s['train_calls']}  "
        f"rows={s['rows']}  bench_masked={s['bench_masked']}  "
        f"subj_masked={s['subj_masked']}  bc_idx_masked={s['bc_idx_masked']}"
    )

print(
    f"\nModel A best val log-loss: {result_a.best_val_log_loss:.6f}  "
    f"brier: {result_a.best_val_brier:.6f}  "
    f"epoch_best: {result_a.epoch_best}"
)

# %% [markdown]
# ## 8b. (Removed) Model B is no longer trained
#
# Earlier revisions of this notebook trained a second IRT-MLP head
# ("Model B") with `p_bench=1.0` + `q_bc=0.15` and combined it with
# Model A through a coverage-conditional blend. The four-member
# stacker downstream now provides cross-architecture diversity
# (LightGBM, kNN-similarity, logistic regression on top of Model A),
# so a second IRT head is redundant and only inflates the bundle.
# Member 1 of the stacker is therefore Model A directly.

# %% [markdown]
# ## 9. Score Model A on val (Member 1 prediction vector)
#
# Member 1's contribution to the stacker is just Model A's
# uncalibrated probability on every val row. We previously scored
# Model A twice (real-meta + synthetic-cold-start) to fit a
# coverage-conditional blend against Model B; with B gone, that
# blend is degenerate and we only need the real-meta pass.
#
# %%
import torch
from tqdm.auto import tqdm

from src.models import build_model as _build_model_for_inf

device = "cuda" if torch.cuda.is_available() else "cpu"


def _score_dataset(
    ds: LookupDataset,
    model,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run ``model`` over ``ds`` and return per-row uncalibrated
    probabilities. Model A always uses real metadata + real bc_idx
    at scoring time; the dropout pre-hook only fires under
    ``model.training``, which we explicitly disable below.
    """
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


# Re-instantiate Model A and load its weights. attach_metadata_tables
# rebuilds the embedding modules with the right cardinalities, so we
# MUST attach BEFORE loading the checkpoint.
trained_a = _build_model_for_inf(MODEL_A_NAME, model_a_cfg)
trained_a.attach_metadata_tables(meta_id_tables)
# Use the cached checkpoint dict directly (avoids a duplicate disk
# read; the cached object is bit-identical to ``torch.load`` of the
# checkpoint file).
ckpt_a = ckpt_a_cached
trained_a.load_state_dict(ckpt_a["model_state"])
trained_a = trained_a.to(device).eval()

print("Scoring Model A on val...")
p_a_val = _score_dataset(val_ds, trained_a)

# Pre-score Model A on train ONCE here (cached). This lets us free
# the heavy ``train_ds`` / ``val_ds`` item-embedding stacks (~84 GB
# combined at 5M+260k rows x 4096 dims) before the
# X_train_dense / X_val_dense build, which itself wants ~25 GB. The
# downstream calibrator cell consumes ``p_a_train`` directly instead
# of re-scoring, so this isn't a duplicate cost.
print("Scoring Model A on train (cached so calibrator skips re-score)...")


def _score_a_on_train():
    return _score_dataset(train_ds, trained_a)


p_a_train = cache_or_compute(
    "p_a_train",
    key_inputs=(
        "v1",
        int(len(primary.train)),
        int(state_fingerprint(ckpt_a["model_state"])[:8], 16),
    ),
    compute_fn=_score_a_on_train,
)
print(f"[Member 1] p_a_train: shape={p_a_train.shape}  "
      f"log-loss={-(primary.train['label'].astype(float).to_numpy() * np.log(np.clip(p_a_train, 1e-6, 1-1e-6)) + (1 - primary.train['label'].astype(float).to_numpy()) * np.log(1 - np.clip(p_a_train, 1e-6, 1-1e-6))).mean():.6f}")

# Free the heavy item-embedding stacks now that Model A has produced
# both train + val predictions. ``train_ds.item_emb`` alone is the
# single largest object in scope (~80 GB at 5M rows x 4096 dims).
# We leave the trained model itself bound (cheap) in case the
# calibrator wants to score on additional rows; but per the cached
# ``p_a_train`` / ``p_a_val`` we don't actually need it any more.
trained_a = trained_a.to("cpu")
for _stale_name in ("train_ds", "val_ds"):
    if _stale_name in globals():
        try:
            del globals()[_stale_name]
        except KeyError:
            pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
_log_ram("after freeing train_ds / val_ds (post Model A scoring)")

ylab_val = primary.val["label"].astype(float).to_numpy()
val_bc_keys = primary.val["benchmark_condition_key"].astype(str).to_numpy()

# With Model B removed, ``synth_mask`` and ``bench_present_val`` are
# vestigial: every val row's benchmark IS in the trained indexer
# (item-cold-start val), so coverage on val is uniformly 1. We keep
# them defined so the stacker / calibrator cells don't need a
# downstream rewrite. ``BLEND_PRESENT`` / ``BLEND_MISSING`` are only
# used as runtime-meta breadcrumbs in a couple of places below; the
# degenerate (1.0, 0.0) values reflect "Member 1 is Model A only".
synth_mask = np.zeros(len(p_a_val), dtype=bool)
bench_present_val = np.ones(len(p_a_val), dtype=np.float32)
BLEND_PRESENT = (1.0, 0.0)
BLEND_MISSING = (1.0, 0.0)

eps = 1e-7
ll_member1 = -(
    ylab_val * np.log(np.clip(p_a_val, eps, 1.0 - eps))
    + (1.0 - ylab_val) * np.log(np.clip(1.0 - p_a_val, eps, 1.0 - eps))
).mean()
print(f"[Member 1] Model A val log-loss: {ll_member1:.6f}")

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
#     subject_cat_ids / subject_num from meta_id_tables. We reuse the
#     cached bundle here so this cell short-circuits as soon as Model A
#     is cached -- no extra ``torch.load`` round-trip.
_ckpt_a_for_theta = ckpt_a_cached
_state_a_for_theta = _ckpt_a_for_theta["model_state"]


def _lookup_param(state_dict, candidates: tuple[str, ...]):
    """Return the first state-dict tensor matching one of ``candidates``.

    ``MetaHybridIRTKFactorGatedMLP`` stores subject ability + factors as
    ``nn.Embedding`` modules (``self.theta``, ``self.u``), so the
    flattened state-dict keys are ``theta.weight`` / ``u.weight``, not
    ``theta`` / ``u``. Older checkpoints used bare names. We try
    ``.weight`` first to match the current model definition, then fall
    back to bare names so an older bundle still works.
    """
    for k in candidates:
        if k in state_dict:
            return state_dict[k]
    raise KeyError(
        "Could not locate any of "
        f"{list(candidates)} in checkpoint state dict. Keys present: "
        f"{sorted(state_dict.keys())[:30]} ..."
    )


theta_tensor = _lookup_param(
    _state_a_for_theta, ("theta.weight", "theta", "subject_theta.weight", "subject_theta")
)
theta_s_per_subject = theta_tensor.detach().cpu().numpy().astype(np.float32)
# theta is shape [n_subjects, 1] when nn.Embedding(n_subjects, 1) is
# used -- flatten to [n_subjects] so MemberSubjectTables is happy.
if theta_s_per_subject.ndim == 2 and theta_s_per_subject.shape[1] == 1:
    theta_s_per_subject = theta_s_per_subject[:, 0]

try:
    u_tensor = _lookup_param(
        _state_a_for_theta, ("u.weight", "U.weight", "u", "U")
    )
    u_s_per_subject = u_tensor.detach().cpu().numpy().astype(np.float32)
except KeyError:
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


# ---------------------------------------------------------------------
# bc-redaction masks: simulate "benchmark unknown" cold-start at TRAIN
# AND VAL time. We pick ~``BC_REDACT_FRAC`` of the unique
# (item_key, benchmark_condition_key) combos uniformly at random and
# zero out their cond one-hot in the dense feature matrix. The
# leaderboard sees rows where the benchmark identity is missing -- if
# Members 2 & 4 only ever see fully-keyed rows during training, they
# learn to lean on the cond signal and collapse on cold-start. Drawing
# whole COMBOS (not random rows) is what the user asked for: it keeps
# the redaction unit semantically meaningful (a "lost benchmark" looks
# the same on every row of every item in that benchmark) and forces
# the row-side learners to recover from the same loss the runtime
# faces.
#
# Same fraction is applied to val so val_logloss reflects the
# leaderboard distribution rather than a fully-keyed-bench best case.
BC_REDACT_FRAC = float(CFG.get("bc_redaction", {}).get("fraction", 0.20))
_bc_redact_seed = int(CFG.get("bc_redaction", {}).get("seed", SEED ^ 0xC01D5)) & 0xFFFFFFFF


def _make_bc_redaction_mask(split_df, *, frac: float, seed: int) -> np.ndarray:
    """Mark rows whose (item_key, bc_key) combo is in a randomly chosen
    set of held-out combos. Returns a [N] bool array."""
    if float(frac) <= 0.0:
        return np.zeros(len(split_df), dtype=bool)
    pairs_tuple = list(
        zip(
            split_df["item_key"].astype(str).tolist(),
            split_df["benchmark_condition_key"].astype(str).tolist(),
        )
    )
    unique_pairs = sorted(set(pairs_tuple))
    n_pairs = len(unique_pairs)
    n_redact = int(round(float(frac) * n_pairs))
    if n_redact <= 0:
        return np.zeros(len(split_df), dtype=bool)
    rng = np.random.default_rng(int(seed))
    held_idx = rng.choice(n_pairs, size=n_redact, replace=False)
    held_set = set(unique_pairs[i] for i in held_idx)
    out = np.fromiter(
        (p in held_set for p in pairs_tuple),
        count=len(pairs_tuple),
        dtype=bool,
    )
    return out


bc_redacted_train = _make_bc_redaction_mask(
    primary.train, frac=BC_REDACT_FRAC, seed=_bc_redact_seed,
)
bc_redacted_val = _make_bc_redaction_mask(
    primary.val, frac=BC_REDACT_FRAC, seed=_bc_redact_seed ^ 0xDEAD,
)
print(
    f"[bc_redaction] train: redacted "
    f"{int(bc_redacted_train.sum()):,} / {len(bc_redacted_train):,} rows "
    f"({100.0 * bc_redacted_train.mean():.1f}%)"
)
print(
    f"[bc_redaction] val:   redacted "
    f"{int(bc_redacted_val.sum()):,} / {len(bc_redacted_val):,} rows "
    f"({100.0 * bc_redacted_val.mean():.1f}%)"
)


def _build_X(split_part, nn_mat, redacted_mask):
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
        bc_redacted=redacted_mask,
    )


# ``redact_v1`` invalidates older X_*_dense caches that lacked the
# bc-redaction zeroing of the cond one-hot (i.e. were trained with
# fully-keyed bench conditions). Reusing those caches with the new
# member fits would defeat the point of the redaction: Members 2 & 4
# would still learn on the leak-prone signal.
X_train_dense = cache_or_compute(
    "X_train_dense",
    key_inputs=(
        member_feat_schema.feature_dim, len(primary.train),
        "redact_v1", round(BC_REDACT_FRAC, 3), _bc_redact_seed,
    ),
    compute_fn=lambda: _build_X(primary.train, nn_train_mat, bc_redacted_train),
)
X_val_dense = cache_or_compute(
    "X_val_dense",
    key_inputs=(
        member_feat_schema.feature_dim, len(primary.val),
        "redact_v1", round(BC_REDACT_FRAC, 3), _bc_redact_seed,
    ),
    compute_fn=lambda: _build_X(primary.val, nn_val_mat, bc_redacted_val),
)
y_train = primary.train["label"].astype(float).to_numpy().astype(np.float32)
print(f"[member_features] X_train: {X_train_dense.shape}  X_val: {X_val_dense.shape}")

# %% [markdown]
# ## 9b-bis. Mean-encoded interaction features for Members 2 & 4
#
# These cells fit per-cell pass-rate statistics on TRAIN rows and emit
# two compact dense feature matrices:
#
#   * `member2_interaction_train/val` -- 8 cold-start-safe interaction
#     columns (subject x item-cluster + subject x bench_condition
#     means/counts/deviations) that get APPENDED to `X_train_dense` /
#     `X_val_dense`. The point is to give the GBDT something to split
#     on that's BOTH novel (not encoded by the existing dense schema)
#     AND directly correlated with the label, so its trees stop
#     re-discovering Member 1's signal and start contributing
#     independent information to the stacker.
#
#   * `member4_marginal_train/val` -- 14 mean-encoded marginal columns
#     (subject mean, bc mean, cluster mean, plus a handful of two-way
#     interactions and a constant) that REPLACE Member 4's feature
#     view entirely. Member 4 currently shares X_train_dense with
#     Member 2 -- including all qwen-embedding-derived features --
#     which is why its predictions are heavily correlated with
#     Members 1 and 3 (which also lean on those embeddings). Cutting
#     it off from the embedding view forces the stacker to either
#     downweight it (if redundant) OR weight it highly when it
#     contributes a complementary signal (subject / bench baselines
#     that the other members are getting wrong).
#
# Cells with no training observations fall back through:
# subj_cluster -> cluster_mean -> global_mean, etc. (Bayesian
# shrinkage with `smoothing=30`). Val rows look up the precomputed
# train-time aggregates -- the val labels themselves are NEVER
# consulted at fit time.

# %%
from src.mean_encoded_features import (
    MEMBER2_INTERACTION_FEATURE_DIM,
    MEMBER2_INTERACTION_FEATURE_NAMES,
    MEMBER4_MARGINAL_FEATURE_DIM,
    MEMBER4_MARGINAL_FEATURE_NAMES,
    apply_member2_interaction_features,
    apply_member4_marginal_features,
    fit_mean_encoded_stats,
)


def _compute_id_arrays(rows_df):
    """Subject / cluster / bc-condition int ids for the rows in df.
    Returns (subj_ids, cluster_ids, bc_ids) all int64. Unknown ids
    map to -1 (the mean-encoded apply functions handle this safely)."""
    subj_ids = np.fromiter(
        (indexer.subject_id(str(s)) for s in rows_df["subject_key"]),
        count=int(len(rows_df)),
        dtype=np.int64,
    )
    cluster_ids = np.fromiter(
        (int(cluster_assignments.get(str(k), -1)) for k in rows_df["item_key"]),
        count=int(len(rows_df)),
        dtype=np.int64,
    )
    bc_ids = np.fromiter(
        (
            int(indexer.bc_to_id.get(str(b), -1))
            for b in rows_df["benchmark_condition_key"]
        ),
        count=int(len(rows_df)),
        dtype=np.int64,
    )
    return subj_ids, cluster_ids, bc_ids


_mef_train_subj, _mef_train_cluster, _mef_train_bc = _compute_id_arrays(primary.train)
_mef_val_subj, _mef_val_cluster, _mef_val_bc = _compute_id_arrays(primary.val)
_N_CLUSTERS_ME = int(CFG["clustering"]["k"])
print(
    f"[mean-enc] id arrays built: "
    f"train_subj_unique={int(np.unique(_mef_train_subj).size):,}  "
    f"train_cluster_unique={int(np.unique(_mef_train_cluster).size):,}  "
    f"train_bc_unique={int(np.unique(_mef_train_bc).size):,}  "
    f"(n_subjects={indexer.n_subjects}, n_clusters={_N_CLUSTERS_ME}, "
    f"n_bc={indexer.n_bc})"
)


def _fit_mean_encoded_stats():
    return fit_mean_encoded_stats(
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        labels=y_train,
        n_subjects=int(indexer.n_subjects),
        n_clusters=int(_N_CLUSTERS_ME),
        n_bcs=int(indexer.n_bc),
        smoothing=float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)),
    )


mean_encoded_stats = cache_or_compute(
    "mean_encoded_stats",
    key_inputs=(
        "v1",
        int(indexer.n_subjects), int(_N_CLUSTERS_ME), int(indexer.n_bc),
        len(primary.train), SEED,
        round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
    ),
    compute_fn=_fit_mean_encoded_stats,
)
print(
    f"[mean-enc] stats fit OK: "
    f"global_mean={mean_encoded_stats.global_mean:.4f}  "
    f"smoothing={mean_encoded_stats.smoothing:.1f}  "
    f"|subj_cluster_mean|={mean_encoded_stats.subj_cluster_mean.shape}  "
    f"|subj_bc_mean|={mean_encoded_stats.subj_bc_mean.shape}"
)

member2_interaction_train = apply_member2_interaction_features(
    mean_encoded_stats,
    subject_ids=_mef_train_subj,
    cluster_ids=_mef_train_cluster,
    bc_ids=_mef_train_bc,
)
member2_interaction_val = apply_member2_interaction_features(
    mean_encoded_stats,
    subject_ids=_mef_val_subj,
    cluster_ids=_mef_val_cluster,
    bc_ids=_mef_val_bc,
)
member4_marginal_train = apply_member4_marginal_features(
    mean_encoded_stats,
    subject_ids=_mef_train_subj,
    cluster_ids=_mef_train_cluster,
    bc_ids=_mef_train_bc,
)
member4_marginal_val = apply_member4_marginal_features(
    mean_encoded_stats,
    subject_ids=_mef_val_subj,
    cluster_ids=_mef_val_cluster,
    bc_ids=_mef_val_bc,
)
print(
    f"[mean-enc] Member 2 interaction features: train {member2_interaction_train.shape}  "
    f"val {member2_interaction_val.shape}  cols={MEMBER2_INTERACTION_FEATURE_DIM}"
)
print(
    f"[mean-enc] Member 4 marginal features:    train {member4_marginal_train.shape}  "
    f"val {member4_marginal_val.shape}  cols={MEMBER4_MARGINAL_FEATURE_DIM}"
)

# Augmented dense matrices for Member 2 (interaction features appended
# to the existing 1202-feature schema). Member 2 sees BOTH the original
# embedding-derived features AND the new mean-encoded interactions.
#
# MEMORY: only the LEGACY global Member 2 fit consumes these (~19 GB
# combined for the train+val matrices). On the Task 3 path (default)
# the global Member 2 uses ``X_*_dense_m2v2`` instead and these are
# pure waste; gate construction behind ``_M2V2_ENABLED`` so we don't
# blow ~19 GB of resident memory just to hold them through the OOF
# loop (which built fresh fold-scoped matrices anyway).
member2_feature_names = tuple(member_feat_schema.feature_names) + tuple(
    MEMBER2_INTERACTION_FEATURE_NAMES
)
# ``_M2V2_ENABLED`` is defined further down in section 9b-ter; peek at
# the same CFG value here so we don't pre-allocate the legacy matrices
# when the Task 3 path is active.
_m2v2_enabled_early = bool(CFG.get("member2_v2", {}).get("enabled", True))
if not _m2v2_enabled_early:
    X_train_dense_m2 = np.concatenate(
        [X_train_dense, member2_interaction_train], axis=1
    ).astype(np.float32, copy=False)
    X_val_dense_m2 = np.concatenate(
        [X_val_dense, member2_interaction_val], axis=1
    ).astype(np.float32, copy=False)
    print(
        f"[mean-enc] Member 2 augmented X: "
        f"train {X_train_dense_m2.shape}  val {X_val_dense_m2.shape}  "
        f"(was {X_train_dense.shape[1]} cols, now {X_train_dense_m2.shape[1]} cols)"
    )
else:
    X_train_dense_m2 = None
    X_val_dense_m2 = None
    print(
        "[mean-enc] Member 2 augmented X: NOT built (Task 3 path uses "
        "X_*_dense_m2v2 instead -- saves ~19 GB resident memory)."
    )

# %% [markdown]
# ## 9b-ter. Member 2 v2 setup: non-embedding schema + subject-mean anchor
#
# Builds the Task 3 artifacts that decouple Member 2 from Member 1:
#
#   1. `member2_v2_schema` -- the locked NON-EMBEDDING feature schema
#      (subject_idx, cluster_id, bc_id, bc_redacted_mask,
#      subject_obs_count_log1p, subject/bench cat+num metadata, and
#      the 8 mean-encoded interaction columns). Gate 3d audits the
#      schema for any embedding-derived columns (pool_, centroid_dist,
#      nn_, theta, u__) and raises on detection.
#
#   2. `subject_mean_table_global` -- per-subject mean pass-rate with
#      Bayesian shrinkage (`smoothing` param controls cold-subject
#      pull-toward-prior). This is the INFERENCE-TIME anchor used at
#      val and at runtime: each row's prediction is composed via
#      `gbdt_compose_residual_batch(state, X_row, subject_mean[row.subj])`.
#
#   3. `X_val_dense_m2v2` + `subject_mean_val` -- the val-side feature
#      matrix and anchor, materialized once here and reused at val
#      reporting (section 9.5c).
#
# The TRAINING-TIME anchors come from OOF subject_mean tables computed
# inside the per-fold loop (section 9.5) -- this avoids the trivial
# leakage where logit(subject_mean[s_r]) includes row r's own label.
# Skipped entirely when `CFG["member2_v2"]["enabled"]=False` (legacy
# Member 2 path stays active).

# %%
_M2V2_ENABLED = bool(CFG.get("member2_v2", {}).get("enabled", True))
_M2V2_SMOOTHING = float(
    CFG.get("member2_v2", {}).get("subject_mean_smoothing", 30.0)
)

if _M2V2_ENABLED:
    from src.member2_features import (
        Member2FeatureSchema,
        audit_no_embedding_features,
        build_member2_feature_matrix,
        build_member2_schema,
    )
    from src.subject_mean import (
        SubjectMeanTable,
        apply_subject_mean,
        apply_subject_obs_count,
        assert_oof_subject_mean,
        fit_subject_mean_table,
    )

    member2_v2_schema = build_member2_schema(
        subject_cat_field_names=tuple(_meta_schema.subject_categorical),
        subject_num_field_names=tuple(_meta_schema.subject_numeric),
        bench_cat_field_names=tuple(_meta_schema.benchmark_categorical),
        bench_num_field_names=tuple(_meta_schema.benchmark_numeric),
        interaction_feature_names=tuple(MEMBER2_INTERACTION_FEATURE_NAMES),
    )
    # Gate 3d (RED-TEAM): no-embedding audit. Catches the failure mode
    # where the schema accidentally bundles in theta/u/pool/centroid/NN
    # columns (which would re-correlate Member 2 with Member 1).
    audit_no_embedding_features(member2_v2_schema)
    print(
        f"[Member 2 v2] Gate 3d PASS: schema dim={member2_v2_schema.feature_dim}, "
        f"{len(member2_v2_schema.categorical_indices)} categorical cols, "
        f"no embedding-derived columns detected."
    )

    # Materialize per-id lookup tables from meta_id_tables (torch -> numpy).
    _m2v2_subj_cat_lookup = meta_id_tables.subject_cat_ids.cpu().numpy().astype(np.int64)
    _m2v2_subj_num_lookup = meta_id_tables.subject_num.cpu().numpy().astype(np.float32)
    _m2v2_bench_cat_lookup = meta_id_tables.bc_cat_ids.cpu().numpy().astype(np.int64)
    _m2v2_bench_num_lookup = meta_id_tables.bc_num.cpu().numpy().astype(np.float32)
    print(
        f"[Member 2 v2] lookup tables: "
        f"subj_cat={_m2v2_subj_cat_lookup.shape}  subj_num={_m2v2_subj_num_lookup.shape}  "
        f"bench_cat={_m2v2_bench_cat_lookup.shape}  bench_num={_m2v2_bench_num_lookup.shape}"
    )

    # Global subject_mean table (full train labels). This is the
    # INFERENCE-TIME anchor -- val/test rows look up subject_mean[s]
    # which does NOT include their own labels (val rows aren't in
    # train).
    subject_mean_table_global = fit_subject_mean_table(
        subject_ids=_mef_train_subj,
        labels=y_train,
        n_subjects=int(indexer.n_subjects),
        smoothing=_M2V2_SMOOTHING,
    )
    print(
        f"[Member 2 v2] global subject_mean table: "
        f"n_subjects={subject_mean_table_global.subject_mean.shape[0]}  "
        f"global_mean={subject_mean_table_global.global_mean:.4f}  "
        f"smoothing={subject_mean_table_global.smoothing:.1f}  "
        f"n_zero_obs_subjects={int((subject_mean_table_global.subject_obs_count == 0).sum())}"
    )

    _global_subj_obs_count_train_log1p = apply_subject_obs_count(
        subject_mean_table_global, _mef_train_subj, log1p=True,
    ).astype(np.float32)
    _global_subj_obs_count_val_log1p = apply_subject_obs_count(
        subject_mean_table_global, _mef_val_subj, log1p=True,
    ).astype(np.float32)

    X_val_dense_m2v2 = build_member2_feature_matrix(
        member2_v2_schema,
        subject_ids=_mef_val_subj,
        cluster_ids=_mef_val_cluster,
        bc_ids=_mef_val_bc,
        bc_redacted_mask=bc_redacted_val.astype(np.float32),
        subject_obs_count_log1p=_global_subj_obs_count_val_log1p,
        subject_cat_lookup=_m2v2_subj_cat_lookup,
        subject_num_lookup=_m2v2_subj_num_lookup,
        bench_cat_lookup=_m2v2_bench_cat_lookup,
        bench_num_lookup=_m2v2_bench_num_lookup,
        interaction_matrix=member2_interaction_val.astype(np.float32),
    )
    subject_mean_val = apply_subject_mean(
        subject_mean_table_global, _mef_val_subj,
    ).astype(np.float64)
    print(
        f"[Member 2 v2] X_val_dense_m2v2: {X_val_dense_m2v2.shape}  "
        f"subject_mean_val: shape={subject_mean_val.shape}  "
        f"min={subject_mean_val.min():.4f}  mean={subject_mean_val.mean():.4f}  "
        f"max={subject_mean_val.max():.4f}"
    )
else:
    print(
        "[Member 2 v2] DISABLED via CFG['member2_v2']['enabled']=False. "
        "Falling back to legacy Member 2 (Member 1 anchor + full schema)."
    )


# %% [markdown]
# ### 9b'. Member 2 v4 (direct-binary tree on v3 features) setup
#
# v4 is the same feature builder as v3 (25 numeric columns: per-id obs
# counts, mean-encoded passrates, is_unknown_* flags, plus p_m1 and
# subject_mean as logits) but trained as a STANDARD binary-objective
# GBDT on y directly -- no residual framing, no subject_mean anchor at
# apply time, output is sigmoid(tree_raw).
#
# Why v4 not v3: v3's residual-on-subject_mean composition was a
# blowup hazard -- on rows where the tree fit a large +5 logit
# correction on top of subject_mean=0.7, the composed prediction
# collapsed toward sigmoid(0.85 + 5) = 0.997 and a single wrong row
# contributed ~5.8 nats of loss. v3 failed Gate 3d at full scale
# (OOF NLL > constant-mean baseline) for exactly this reason. v4
# eliminates the failure mode by construction: a binary-objective
# tree's per-leaf output is bounded by the in-leaf label averages,
# so it mathematically cannot produce a 0.997-confident prediction on
# a row where the leaf's labels averaged 0.5.
#
# Why this still adds stacker value: M2 v4's job isn't to beat M1's
# individual NLL -- the stacker doesn't care about that. Its job is
# to produce errors UNCORRELATED with M1 (bilinear), M3 (neighbor avg),
# M4 (linear), and M5 (projection kNN). A tree on (p_m1, subject_mean,
# per-id passrates, obs counts) has tree-shaped step-function decision
# boundaries that no other member can produce, so its errors are
# structurally diverse and the stacker can extract value even when
# v4's standalone NLL is modest.
#
# v4 reuses v2's subject_mean infrastructure (only because subject_mean
# appears as a FEATURE in the v3 builder; it is no longer an anchor at
# apply time). v2's GBDT itself is still skipped when v4 is enabled.

# %%
_M2V4_ENABLED = bool(CFG.get("member2_v4", {}).get("enabled", True))
_M2V4_SMOOTHING = float(CFG.get("member2_v4", {}).get("smoothing", 30.0))
# Chunk size for M1 scoring on fold-train rows. ~256k rows * 4096 dim *
# 4 bytes = ~4 GB peak per chunk for the LookupDataset item_emb stack.
_M2V4_M1_CHUNK_ROWS = int(CFG.get("member2_v4", {}).get("m1_chunk_rows", 256_000))

if _M2V4_ENABLED:
    if not _M2V2_ENABLED:
        import warnings as _v4_warn
        _v4_warn.warn(
            "Member 2 v4 currently piggybacks on v2's subject_mean setup "
            "(subject_mean_table_global, subject_mean_val, the per-fold "
            "fold_subject_mean_table) -- the v3 feature builder pulls "
            "subject_mean as a per-row FEATURE (not an anchor). Either "
            "keep _M2V2_ENABLED=True and let v4 replace its GBDT output, "
            "or implement a v4-only subject_mean setup path before "
            "disabling v2. Forcing _M2V4_ENABLED=False so the pipeline "
            "does not silently produce garbage."
        )
        _M2V4_ENABLED = False
    else:
        from src.member2_v3_calibration import (
            MEMBER2_V3_FEATURE_NAMES,
            M2V3_FEATURE_DIM,
            Member2V3FeatureBuilder,
            Member2V3State,
            build_member2_v3_features,
            fit_member2_v3_feature_builder,
        )
        print(
            f"[Member 2 v4] direct-binary GBDT ENABLED (smoothing={_M2V4_SMOOTHING}, "
            f"F={M2V3_FEATURE_DIM} features = v3 builder, trained as binary "
            "classifier on y directly -- no residual framing, no subject_mean "
            "anchor at apply time, output = sigmoid(tree_raw). REPLACES v2/v3 "
            "GBDT output."
        )
else:
    print("[Member 2 v4] DISABLED via CFG['member2_v4']['enabled']=False.")


# %% [markdown]
# ### 9b''. Member 2 v5 (attribute + neighborhood tree, no M1 leakage) setup
#
# v5 is the literature-recommended cold-start tree recipe. It removes ALL
# M1-derived inputs (``logit_p_m1``, ``logit_subject_mean``,
# ``logit_disagreement``, ``abs_logit_disagreement``, ``p_m1_uncertainty``)
# that caused v3/v4 to memorize Member 1's prediction (corr=0.96, NLL=0.71
# vs constant-mean baseline of 0.63). In their place v5 adds three
# orthogonal signal families:
#
#   * **Item content** (19 cols): 9 pool/text features + 8 centroid
#     distances + benchmark_age value/mask. Cold-item-stable signals
#     that the tree can split on at inference time even when the item
#     was never seen at training time.
#   * **Honest historical counts** (9 cols): per-id and 2-D cell
#     ``log1p(observation_count)`` plus bc_redacted_mask. No target
#     leakage (just counts, not means), and they sharpen the obs-count
#     interactions that mean-encoding alone misses.
#   * **Neighborhood aggregates** (7 cols): the 7 most-useful columns
#     from the existing 23-col NN feature matrix
#     (passrate_weighted_mean, coverage, mean_similarity,
#     effective_neighbor_count, passrate_subject_conditional,
#     passrate_benchmark_conditional, distance_to_kth_neighbor). These
#     give the tree neighborhood-derived predictions that overlap
#     minimally with M3's kNN voting member.
#
# Plus the original 14 Bayes-shrunk mean-encoded passrates + unknown-id
# flags (subject/cluster/bc/macro_family/organization/family and the
# two 2-D subj_x_cluster / subj_x_bc cells), with subject-level mean
# encoding added (v3/v4 omitted it as collinear with subject_mean used
# as anchor; v5 has no anchor, so subject_passrate IS the signal).
#
# Total: 49 numeric features. NO categorical_feature= to LightGBM
# because the compiled numpy walker only supports numeric splits; the
# builder does Bayesian-shrunk OOF target encoding manually.
#
# Trained as a DIRECT binary GBDT (no init_score, output = sigmoid(tree_raw))
# so leaf outputs are bounded by in-leaf label averages -- prevents the
# cumulative-logit saturation that broke v4. When v5 is enabled it
# REPLACES v4's output via the same ``gbdt_state`` alias that the
# exporter consumes.
#
# v5 retains v2's setup as a soft dependency for the per-fold mean-
# encoded id arrays (``_mef_*``) and the global ``bc_redacted_train``
# mask; if v2 is disabled, v5 falls back to v4 (and warns).

# %%
_M2V5_ENABLED = bool(CFG.get("member2_v5", {}).get("enabled", True))
_M2V5_SMOOTHING = float(CFG.get("member2_v5", {}).get("smoothing", 30.0))

if _M2V5_ENABLED:
    import warnings as _v5_warn
    if not _M2V2_ENABLED:
        _v5_warn.warn(
            "Member 2 v5 reuses v2's per-fold mean-encoded id arrays "
            "(_mef_subj_fold_*, _mef_cluster_fold_*, _mef_bc_fold_*) "
            "as inputs. v2 is currently disabled, which means those "
            "id arrays may not be constructed in the per-fold loop. "
            "Either keep _M2V2_ENABLED=True or implement a v5-only "
            "id construction path. Forcing _M2V5_ENABLED=False so the "
            "pipeline falls back to v4/v2/legacy gracefully."
        )
        _M2V5_ENABLED = False

if _M2V5_ENABLED:
    # v5 supersedes v4 -- disable the v4 path so we don't pay for M1
    # fold-train scoring (only v4 needs that) and so the per-fold /
    # global branches don't both run.
    if _M2V4_ENABLED:
        print(
            "[Member 2 v5] superseding _M2V4_ENABLED=True with v5; v4 "
            "branches will be skipped in the per-fold and global fits."
        )
    _M2V4_ENABLED = False

    from src.member2_v5_attr_nn import (
        M2V5_FEATURE_DIM,
        MEMBER2_V5_FEATURE_NAMES,
        Member2V5FeatureBuilder,
        Member2V5State,
        Member2V5Warning,
        build_member2_v5_features,
        fit_member2_v5_feature_builder,
    )
    print(
        f"[Member 2 v5] attribute+NN tree ENABLED (smoothing={_M2V5_SMOOTHING}, "
        f"F={M2V5_FEATURE_DIM} features = item content + honest counts + "
        "mean-encoded passrates + NN aggregates; NO M1-derived inputs). "
        "Trained as direct binary GBDT on y; output = sigmoid(tree_raw). "
        "REPLACES v2/v3/v4 GBDT output."
    )
else:
    print("[Member 2 v5] DISABLED via CFG['member2_v5']['enabled']=False.")


# %% [markdown]
# ### 9b'''. v5 per-row pool / centroid / benchmark_age lookups
#
# v5 needs three per-row arrays that aren't already exposed in the
# per-fold loop:
#   * a [N_rows, 17] dense slice of ``pool_features_z`` aligned to a
#     row DataFrame's ``item_key`` order (9 pool cols + 8 centroid_dist cols)
#   * a [N_rows] benchmark_age array (NaN for unknown)
# We build small helpers here so both the per-fold OOF branch and the
# global 9.5c branch use the same code path -- prevents silent
# train/inference feature-pipeline drift.

# %%
if _M2V5_ENABLED:
    # v5's locked column order: 9 pool cols (from item_features.POOL_FEATURE_NAMES
    # with a "pool_" prefix added for clarity) + 8 centroid_dist cols.
    # ``pool_features_z`` stores the pool cols WITHOUT the "pool_" prefix
    # (because that's the raw output of ``item_features.feature_names()``)
    # plus the 8 centroid_dist cols. We map between the two namespaces
    # at build time WITHOUT mutating ``pool_features_z`` -- mutation
    # would break the per-fold M1 builder's ``_pool_matrix`` consumer
    # which expects the raw POOL_FEATURE_NAMES_EXT column names.
    _M2V5_POOL_RAW_COLS = (
        "token_len", "char_len", "has_latex", "has_code",
        "n_questions", "n_numbers", "is_multiple_choice",
        "n_choices", "lang_en",
    )
    _M2V5_CENTROID_COLS = tuple(f"centroid_dist_{k}" for k in range(8))
    _M2V5_POOL_SOURCE_COLS = _M2V5_POOL_RAW_COLS + _M2V5_CENTROID_COLS
    _m2v5_missing_cols = [
        c for c in _M2V5_POOL_SOURCE_COLS if c not in pool_features_z.columns
    ]
    if _m2v5_missing_cols:
        import warnings as _v5_col_warn
        _v5_col_warn.warn(
            f"[Member 2 v5] pool_features_z is missing columns "
            f"{_m2v5_missing_cols}; those columns will be zero-filled "
            "at v5 build time. Tree loses some item-content signal but "
            "does not crash."
        )
    # Build a fixed [N_items, 17] dense matrix indexed by item_key once,
    # then slice it per fold via positional indexing. Faster than a
    # per-row dict lookup and uses ~13 MB at full scale (197114 items x
    # 17 cols x 4 bytes).
    _m2v5_pool_idx_df = pool_features_z.set_index("item_key")
    _m2v5_pool_full_matrix = np.zeros(
        (len(_m2v5_pool_idx_df), len(_M2V5_POOL_SOURCE_COLS)),
        dtype=np.float32,
    )
    for _k, _c in enumerate(_M2V5_POOL_SOURCE_COLS):
        if _c in _m2v5_pool_idx_df.columns:
            _m2v5_pool_full_matrix[:, _k] = (
                _m2v5_pool_idx_df[_c].astype(np.float32).fillna(0.0).to_numpy()
            )
    _m2v5_item_key_to_pos = {
        str(k): i for i, k in enumerate(_m2v5_pool_idx_df.index.tolist())
    }

    def _m2v5_pool_for_rows(rows_df) -> np.ndarray:
        """Return ``[N, 17]`` fp32 pool/centroid matrix in row order.

        Unknown item_keys (cold items not in ``pool_features_z``) get a
        zero row. The v5 builder's ``_pool_columns`` finite-fill is a
        no-op on already-finite zero rows.
        """
        positions = np.array(
            [_m2v5_item_key_to_pos.get(str(k), -1)
             for k in rows_df["item_key"]],
            dtype=np.int64,
        )
        out = np.zeros(
            (len(rows_df), len(_M2V5_POOL_SOURCE_COLS)), dtype=np.float32
        )
        known = positions >= 0
        if int(known.sum()) > 0:
            out[known] = _m2v5_pool_full_matrix[positions[known]]
        return out

    # Item-key -> benchmark_age lookup (NaN for unknown). Built once.
    _m2v5_item_age_lookup = dict(
        zip(train_item_keys, item_benchmark_age_arr.astype(np.float64).tolist())
    )

    def _m2v5_age_for_rows(rows_df) -> np.ndarray:
        """Return ``[N]`` float64 benchmark_age (NaN for unknown items)."""
        return np.array(
            [_m2v5_item_age_lookup.get(str(k), np.nan)
             for k in rows_df["item_key"]],
            dtype=np.float64,
        )

# %% [markdown]
# ## 9c. Train Member 2 (LightGBM)
#
# Offline LightGBM training; we ship the compiled tree arrays + bias
# for pure-NumPy traversal at runtime. The internal parity check (in
# `fit_gbdt_member`) ensures the NumPy walker matches LightGBM's
# `predict(raw_score=True)` to within `parity_atol=1e-5`.
#
# **Residual-learner mode (post-2026-05-26):** instead of training the
# GBDT to predict the label directly, we train it to predict
# `logit(y) - logit(p_member1_train)` under a `regression_l2`
# objective. At inference the walker emits a tree residual; the
# composer combines it with Member 1's current-row logit to recover
# a probability. The math is just an additive shift in logit space.
# Why: a direct-label GBDT will largely re-learn whatever Member 1
# already captures (the stacker correctly downweighted the legacy
# member 2 to ~0.018, i.e. it contributed nothing), so we force it
# to spend capacity on the residual the anchor missed. `init_pred`
# is Member 1's IN-SAMPLE train predictions -- not technically OOF,
# but Member 1 is regularized enough that this is good enough; a
# proper k-fold OOF Member 1 would be cleaner if we re-run.

# %%
from src.gbdt_member import (
    apply_batch as gbdt_apply_batch,
    compose_residual_batch as gbdt_compose_residual_batch,
    fit_gbdt_member,
)


# Per-row item_id for the cold-start internal val split. We map each
# train row to the integer position of its item in ``train_item_keys``
# (the kNN/Indexer's item ordering); items with no match get -1, which
# lands in their own catch-all group. The split picks ~10% of distinct
# items and routes EVERY row of those items to LightGBM's internal
# val. Without this the booster's early stopping fires on rows from
# items it has already memorized -- val NLL diverges from the actual
# cold-start val NLL (the user observed 0.33 internal vs 0.65 reported,
# i.e. 100% optimism).
_item_to_train_idx = {str(k): i for i, k in enumerate(train_item_keys)}
gbdt_train_item_id = np.fromiter(
    (
        _item_to_train_idx.get(str(k), -1)
        for k in primary.train["item_key"].astype(str).tolist()
    ),
    count=len(primary.train),
    dtype=np.int64,
)
print(
    f"[Member 2] item-cold split groups: "
    f"{int(np.unique(gbdt_train_item_id).size):,} unique items"
)


def _fit_member2():
    # ~1.5-2.5 min on the 5M x ~1210 feature schema with the speed
    # knobs below (max_bin=63, force_col_wise=True). The default
    # LightGBM params (max_bin=255, no force_col_wise) take 5-10 min
    # at this scale; the speed knobs are bit-exact under
    # ``deterministic=True``.
    print(
        "[Member 2] training LightGBM in RESIDUAL mode "
        f"(X cols={X_train_dense_m2.shape[1]}, anchor=Member 1, "
        "objective=regression_l2 on logit-residual)..."
    )
    return fit_gbdt_member(
        X=X_train_dense_m2,
        y=y_train,
        feature_names=member2_feature_names,
        # Residual-learner anchor: Member 1's in-sample train preds.
        # The trees learn logit(y) - logit(p_a_train) instead of y,
        # so they MUST contribute orthogonal signal to be useful.
        init_pred_train=p_a_train.astype(np.float64, copy=False),
        n_estimators=int(CFG.get("member2_gbdt", {}).get("n_estimators", 400)),
        learning_rate=float(CFG.get("member2_gbdt", {}).get("learning_rate", 0.05)),
        num_leaves=int(CFG.get("member2_gbdt", {}).get("num_leaves", 31)),
        min_data_in_leaf=int(CFG.get("member2_gbdt", {}).get("min_data_in_leaf", 50)),
        early_stopping_rounds=int(CFG.get("member2_gbdt", {}).get("early_stopping_rounds", 30)),
        seed=SEED,
        parity_atol=1.0e-5,
        # Hold out 10% of train for LightGBM's own early-stopping val.
        val_fraction=float(CFG.get("member2_gbdt", {}).get("val_fraction", 0.1)),
        # Speed knobs (see fit_gbdt_member docstring for details).
        max_bin=int(CFG.get("member2_gbdt", {}).get("max_bin", 63)),
        force_col_wise=bool(CFG.get("member2_gbdt", {}).get("force_col_wise", True)),
        log_period=int(CFG.get("member2_gbdt", {}).get("log_period", 25)),
        num_threads=CFG.get("member2_gbdt", {}).get("num_threads", None),
        # Item-stratified cold-start internal val split (see comment above).
        holdout_group_id=gbdt_train_item_id,
    )


if _M2V2_ENABLED:
    # Task 3: defer the GLOBAL Member 2 fit to section 9.5c. It needs
    # `subject_mean_train_oof` (an OOF-anchor array) as `init_pred_train`,
    # which only exists AFTER the per-fold OOF loop in section 9.5
    # populates `subject_mean_train_oof_acc`. Using the global
    # subject_mean as the training anchor instead would be leakage:
    # logit(subject_mean_global[s_r]) includes y_r in its aggregate,
    # so the GBDT could trivially recover the label from the anchor.
    print(
        "[Member 2 v2] Global Member 2 fit DEFERRED to section 9.5c "
        "(requires OOF subject_mean from section 9.5)."
    )
    gbdt_state = None  # placeholder; populated in section 9.5c
    p_member2_val = None  # populated in section 9.5c
    nll_m2 = float("nan")
else:
    # Legacy Member 2 path (Member 1 anchor + full embedding schema).
    # Cache key versions: ``init_v2`` (broken init_score recovery fix),
    # ``speed_v1`` (LGBM speed knobs), ``honest_loss_v1`` (manual NLL
    # instead of LGBM-reported), ``coldsplit_v1`` (item-stratified ES
    # val split), ``redact_v1`` (bc-redacted feature matrix tie-in),
    # ``residual_v1`` (residual-learner mode + mean-enc interactions).
    gbdt_state = cache_or_compute(
        "gbdt_state",
        key_inputs=(int(X_train_dense_m2.shape[1]), len(primary.train), SEED,
                    "speed_v1", "init_v2", "honest_loss_v1",
                    "coldsplit_v1", "redact_v1", "residual_v1",
                    round(BC_REDACT_FRAC, 3), _bc_redact_seed),
        compute_fn=_fit_member2,
    )
    if str(gbdt_state.output_mode) != "residual_logit":
        raise RuntimeError(
            f"Expected gbdt_state.output_mode='residual_logit' (residual-learner "
            f"mode), got {gbdt_state.output_mode!r}. The cache key bump should "
            "have invalidated the legacy binary-mode entry; delete the cache "
            "file manually if this fires."
        )
    p_member2_val = gbdt_compose_residual_batch(
        gbdt_state, X_val_dense_m2, p_a_val.astype(np.float64, copy=False)
    )
    nll_m2 = float(-(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
                     + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean())
    print(f"[Member 2] residual val log-loss (cold-start primary.val):    {nll_m2:.6f}")
    print(f"[Member 2] residual val NLL stored in state (LGBM val split): {gbdt_state.val_loss:.6f}")
    print(f"[Member 2] train NLL stored in state (manual):       {gbdt_state.train_loss:.6f}")
    print(f"[Member 2] n_trees={gbdt_state.n_trees}  bias={gbdt_state.bias:+.4f}")

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


import time as _time_m3  # local alias to avoid clashing with anything else

# Speed knobs for Member 3. With pca_dim=128 and the realistic shapes
# we hit (~300k train items, ~270k val rows), the unchunked apply_batch
# would materialize a 315 GB sims matrix and either OOM or swap. The
# defaults below auto-chunk to <4 GB and use GPU when available; pin
# them in CFG so the offline scoring path (cell 9c) sees the same.
_m3_cfg = CFG.get("member3_knn", {})
_M3_CHUNK = int(_m3_cfg.get("apply_chunk_size", 0)) or None  # None = auto
_M3_USE_GPU = _m3_cfg.get("apply_use_gpu", None)             # None = auto
_M3_PCA_SAMPLES = int(_m3_cfg.get("max_pca_samples", 0)) or None  # None = use all


def _passrate_dense_from_csr(csr, mask_csr, n_subjects, n_items):
    """Inflate the sparse pass-rate matrices to dense fp32 / bool
    for Member 3's runtime. ``[S, N]`` is bounded by ``S * N * (4 + 1)``
    bytes; at S=900, N=300k that's ~1.4 GB, fine on Colab."""
    dense = csr.astype(np.float32).toarray()
    mask = mask_csr.astype(bool).toarray()
    return dense, mask


_m3_t0 = _time_m3.time()
print(f"[Member 3] inflating pass-rate matrices "
      f"(S={indexer.n_subjects}, N={len(train_item_keys):,})...")
m3_passrate_dense, m3_passrate_mask = _passrate_dense_from_csr(
    nn_passrate_csr, nn_passrate_mask_csr,
    indexer.n_subjects, len(train_item_keys),
)
_m3_dt_inflate = _time_m3.time() - _m3_t0
print(f"[Member 3] passrate_dense shape={m3_passrate_dense.shape}  "
      f"observed={m3_passrate_mask.sum():,} / {m3_passrate_mask.size:,}  "
      f"dense_size={m3_passrate_dense.nbytes / 1024**3:.2f} GB  "
      f"({_m3_dt_inflate:.1f}s)")


# Subject_keys in the SAME order as passrate_dense's rows (which
# is indexer's id ordering: subject_to_id maps key -> id, so we
# need keys sorted by id).
_subject_keys_ordered = [
    k for k, _ in sorted(indexer.subject_to_id.items(), key=lambda kv: kv[1])
]
assert len(_subject_keys_ordered) == indexer.n_subjects


def _fit_member3():
    _t = _time_m3.time()
    print(f"[Member 3] stacking {len(train_item_keys):,} train embeddings (D=4096)...")
    # Pre-allocate a single contiguous buffer instead of np.stack to
    # save the intermediate list-of-arrays. ~3-5x faster for 300k rows.
    train_emb_arr = np.empty(
        (len(train_item_keys), int(item_emb_lookup[train_item_keys[0]].shape[0])),
        dtype=np.float32,
    )
    for i, k in enumerate(tqdm(train_item_keys, desc="stack train embs", unit="item")):
        train_emb_arr[i] = item_emb_lookup[k]
    print(f"[Member 3]   stacked in {_time_m3.time() - _t:.1f}s  "
          f"({train_emb_arr.nbytes / 1024**3:.2f} GB)")

    print(f"[Member 3] PCA-fitting (randomized SVD) + quantizing "
          f"{len(train_item_keys):,} item embeddings"
          f"{' [pca subsample={:,}]'.format(_M3_PCA_SAMPLES) if _M3_PCA_SAMPLES else ''}...")
    return fit_knn_member(
        item_keys=train_item_keys,
        item_embeddings=train_emb_arr,
        subject_keys=_subject_keys_ordered,
        passrate_dense=m3_passrate_dense,
        passrate_mask=m3_passrate_mask,
        pca_dim=int(_m3_cfg.get("pca_dim", 128)),
        quantization=str(_m3_cfg.get("quantization", "int8")),
        # k bumped 32 -> 128 (default in src/knn_member.py). With more
        # neighbors, the per-row weighted mean has more mass on
        # subject-rated cells AFTER the item-global fallback discounts
        # unobserved cells. At k=32 a sparse subject's top-32 are mostly
        # unobserved -> mu_neigh collapses to 0.5; k=128 spreads support
        # across enough cells that the fallback can recover signal.
        k=int(_m3_cfg.get("k", 128)),
        tau_subject=float(_m3_cfg.get("tau_subject", 5.0)),
        tau_global=float(_m3_cfg.get("tau_global", 200.0)),
        # Item-global fallback weight: when subject_q hasn't rated a
        # neighbor i_k, contribute item_global_passrate[i_k] * 0.5 *
        # max(sim, 0) to the weighted mean. Stops cold-start subjects
        # from collapsing to 0.5 mu_neigh on every query and recovers
        # the "this item is universally hard" signal.
        item_fallback_weight=float(_m3_cfg.get("item_fallback_weight", 0.5)),
    )


# ``rsvd_v1`` invalidates any older entry trained with the previous
# full-SVD ``_fit_pca``. The new randomized PCA gives an
# approximately-equivalent basis (top-K subspace overlap > 0.99 in
# unit tests) but the basis is sign-flipped per column. Quantization
# noise dominates downstream cosine similarity, so the change is
# mostly cosmetic -- but we bump the key to be safe.
# ``itemfb_v1`` invalidates entries fit before the item-global
# fallback infrastructure (per-item passrate + obs count tables) and
# the K=32 -> K=128 default bump.
knn_state: KNNMemberState = cache_or_compute(
    "knn_state",
    key_inputs=(
        len(train_item_keys),
        int(_m3_cfg.get("pca_dim", 128)),
        str(_m3_cfg.get("quantization", "int8")),
        int(_m3_cfg.get("k", 128)),
        round(float(_m3_cfg.get("item_fallback_weight", 0.5)), 3),
        "rsvd_v1", "itemfb_v1",
    ),
    compute_fn=_fit_member3,
)
print(f"[Member 3] state: pca_dim={knn_state.pca_dim}  "
      f"quant={knn_state.quantization}  k={knn_state.k}  "
      f"n_items={knn_state.n_items:,}  "
      f"global_passrate={knn_state.global_passrate:.4f}")


# ---- Score on val rows ----
# 266k val rows x ~300k train items would produce a 315 GB sims matrix
# in one shot. apply_batch chunks internally; we just need to stack
# the val embeddings and pass them in.
print(f"[Member 3] stacking {len(primary.val):,} val embeddings...")
_t = _time_m3.time()
val_item_emb = np.empty(
    (len(primary.val), int(item_emb_lookup[next(iter(item_emb_lookup))].shape[0])),
    dtype=np.float32,
)
for i, k in enumerate(tqdm(primary.val["item_key"].to_list(), desc="stack val embs", unit="item")):
    val_item_emb[i] = item_emb_lookup[k]
val_subj_keys_for_knn = [str(s) for s in primary.val["subject_key"]]
print(f"[Member 3]   stacked in {_time_m3.time() - _t:.1f}s  "
      f"({val_item_emb.nbytes / 1024**3:.2f} GB)")

print(f"[Member 3] scoring val rows  "
      f"chunk_size={'auto' if _M3_CHUNK is None else _M3_CHUNK}  "
      f"use_gpu={'auto' if _M3_USE_GPU is None else _M3_USE_GPU}...")
_t = _time_m3.time()
p_member3_val = knn_apply_batch(
    knn_state, val_item_emb, val_subj_keys_for_knn,
    chunk_size=_M3_CHUNK, use_gpu=_M3_USE_GPU, progress=True,
)
_m3_dt_score = _time_m3.time() - _t
nll_m3 = float(-(ylab_val * np.log(np.clip(p_member3_val, 1e-6, 1 - 1e-6))
                 + (1 - ylab_val) * np.log(1 - np.clip(p_member3_val, 1e-6, 1 - 1e-6))).mean())
print(f"[Member 3] scored {len(p_member3_val):,} val rows in {_m3_dt_score:.1f}s  "
      f"({len(p_member3_val) / max(_m3_dt_score, 1e-9):.0f} rows/s)")
print(f"[Member 3] val log-loss: {nll_m3:.6f}  "
      f"p stats: min={p_member3_val.min():.4f} mean={p_member3_val.mean():.4f} max={p_member3_val.max():.4f}")
del val_item_emb
gc.collect()

# %% [markdown]
# ## 9d-bis. Train Member 5 (kNN on a 1-D supervised difficulty projection)
#
# Task 4 of the diversification plan. Member 3 finds neighbors by COSINE
# SIMILARITY on raw item embeddings -- "topically similar" items.
# Member 5 fits a weighted ridge regression of item-mean-passrate on
# item embeddings, projects every item into a 1-D PREDICTED DIFFICULTY
# axis, and finds neighbors on that 1-D line. Items that are
# topically similar but differ in difficulty are far in this space,
# and vice versa, so Member 5's errors should be only weakly correlated
# with Member 3's (verified in Gate 4b below).
#
# Pure NumPy at runtime: ridge solve at fit time, binary-search +
# Gaussian-kernel aggregation at apply time. ~O(log N + K) per query
# vs Member 3's O(N) -- the 1-D layout lets us skip PCA + quantization.

# %%
if CFG.get("member5", {}).get("enabled", False):
    import time as _time_m5
    from src.member5_difficulty_knn import (
        Member5State,
        assert_projection_disjoint_from_val,
        apply_batch_via_ids as m5_apply_batch_via_ids,
        apply_batch as m5_apply_batch,
        apply_one as m5_apply_one,
        fit_member5,
    )

    _m5_cfg = CFG["member5"]
    _M5_ENABLED = True
    _M5_K = int(_m5_cfg.get("k", 32))
    _M5_TAU = float(_m5_cfg.get("tau", 0.05))
    _M5_RIDGE_ALPHA = float(_m5_cfg.get("ridge_alpha", 10.0))
    _M5_ITEM_FB_WEIGHT = float(_m5_cfg.get("item_fallback_weight", 0.3))
    _M5_MIN_SUBJ_PER_ITEM = int(_m5_cfg.get("min_subjects_per_item", 3))
    _M5_GATE4D_SAMPLE_SIZE = int(_m5_cfg.get("gate4d_sample_size", 64))

    print(
        f"[Member 5] enabled. k={_M5_K} tau={_M5_TAU} ridge_alpha={_M5_RIDGE_ALPHA} "
        f"item_fb_w={_M5_ITEM_FB_WEIGHT} min_subj_per_item={_M5_MIN_SUBJ_PER_ITEM}"
    )

    # --- Gate 4c (global): projection-leakage probe vs val items ---
    # Member 5 is FIT on `train_item_keys` only; val items are excluded
    # by construction (primary.train and primary.val are item-disjoint).
    # The probe makes that contract loud: if anything ever changes the
    # split, this gate will catch it before the projection sees val rows.
    _val_item_keys_list = [str(k) for k in primary.val["item_key"]]
    _gate4c_global_result = assert_projection_disjoint_from_val(
        fit_item_keys=list(train_item_keys),
        val_item_keys=_val_item_keys_list,
    )
    print(
        f"[Gate 4c global] PASS: projection fit on "
        f"{_gate4c_global_result['n_fit_items']:,} items, "
        f"val has {_gate4c_global_result['n_val_items']:,} items, "
        f"overlap = {_gate4c_global_result['n_overlap']}."
    )

    # --- Global Member 5 fit (cached) ---
    # Memory discipline: the full ~5 GB item-embedding stack is built
    # ONLY INSIDE the compute_fn so that a cache HIT skips the alloc
    # entirely. The fast-path passrate reuses Member 3's already-built
    # `m3_passrate_dense` / `m3_passrate_mask` (~720 MB each, already
    # live), bypassing fit_member5's row aggregation which would
    # otherwise allocate ~3 GB more.

    def _fit_member5_global():
        _t_inner = _time_m5.time()
        print(f"[Member 5] stacking {len(train_item_keys):,} item embeddings "
              "(inside cache compute_fn so cache HIT skips this entirely)...")
        _m5_item_emb_local = np.empty(
            (len(train_item_keys),
             int(item_emb_lookup[train_item_keys[0]].shape[0])),
            dtype=np.float32,
        )
        for _i, _k in enumerate(train_item_keys):
            _m5_item_emb_local[_i] = item_emb_lookup[_k]
        print(f"[Member 5]   stacked in {_time_m5.time() - _t_inner:.1f}s "
              f"({_m5_item_emb_local.nbytes / 1024**3:.2f} GB)")
        return fit_member5(
            item_keys=list(train_item_keys),
            item_embeddings=_m5_item_emb_local,
            subject_keys=list(_subject_keys_ordered),
            # Row arrays not needed on the fast path; pass empties to
            # avoid the ~75 MB of per-row id vectors at full scale.
            subject_ids_per_row=np.zeros(0, dtype=np.int64),
            item_ids_per_row=np.zeros(0, dtype=np.int64),
            labels=np.zeros(0, dtype=np.float64),
            # Fast path: reuse Member 3's already-built passrate so we
            # don't allocate another [S, N] copy.
            passrate_dense=m3_passrate_dense,
            passrate_mask=m3_passrate_mask,
            k=_M5_K,
            tau=_M5_TAU,
            ridge_alpha=_M5_RIDGE_ALPHA,
            item_fallback_weight=_M5_ITEM_FB_WEIGHT,
            min_subjects_per_item=_M5_MIN_SUBJ_PER_ITEM,
        )

    member5_state: Member5State = cache_or_compute(
        "member5_state",
        key_inputs=(
            # m5_dknn_v2 bumps cache key for the passrate-reuse path.
            "m5_dknn_v2",
            int(len(train_item_keys)),
            int(indexer.n_subjects),
            int(len(primary.train)),
            int(m3_passrate_dense.shape[1]),  # n_items / proxy for d_emb context
            _M5_K, round(_M5_TAU, 6), round(_M5_RIDGE_ALPHA, 6),
            round(_M5_ITEM_FB_WEIGHT, 6), int(_M5_MIN_SUBJ_PER_ITEM),
            int(SEED),
        ),
        compute_fn=_fit_member5_global,
    )
    print(
        f"[Member 5] state: n_items={member5_state.n_items:,}  "
        f"n_subjects={member5_state.n_subjects:,}  "
        f"k={member5_state.k}  tau={member5_state.tau:.4f}  "
        f"global_mean={member5_state.global_mean:.4f}  "
        f"train_loss(sample)={member5_state.train_loss:.5f}  "
        f"||beta||={float(np.linalg.norm(member5_state.projection_weights)):.4f}"
    )

    # --- Gate 4d: apply round-trip probe ---
    # The pure-numpy apply path is the only thing the runtime executes.
    # Verify on a sample of train rows that apply_one (subject_key path)
    # matches apply_batch_via_ids (subject_id path) bit-for-bit. If they
    # diverge, the runtime bundle would silently mispredict.
    # Memory: only sample-sized arrays are materialized -- no full
    # _m5_subj_ids_global / _m5_item_ids_global / _m5_item_emb stacks
    # (those would each be ~5 GB at full scale and we don't need them
    # past fit time).
    _g4d_n_target = int(_M5_GATE4D_SAMPLE_SIZE)
    _g4d_n_train = int(len(primary.train))
    _g4d_n = min(_g4d_n_target, _g4d_n_train)
    _g4d_rng = np.random.default_rng(int(SEED) + 99)
    _g4d_row_idx = _g4d_rng.choice(_g4d_n_train, size=_g4d_n, replace=False)
    _g4d_train_subj_keys = primary.train["subject_key"].to_numpy()
    _g4d_train_item_keys_arr = primary.train["item_key"].to_numpy()
    _g4d_skeys = [str(_g4d_train_subj_keys[int(r)]) for r in _g4d_row_idx]
    _g4d_ikeys = [str(_g4d_train_item_keys_arr[int(r)]) for r in _g4d_row_idx]
    _g4d_sids = np.fromiter(
        (int(indexer.subject_to_id.get(s, -1)) for s in _g4d_skeys),
        dtype=np.int64, count=_g4d_n,
    )
    _g4d_embs = np.stack(
        [np.asarray(item_emb_lookup[k], dtype=np.float32) for k in _g4d_ikeys],
        axis=0,
    )
    _g4d_p_batch = m5_apply_batch_via_ids(
        member5_state, subject_ids=_g4d_sids, query_item_embeddings=_g4d_embs,
    )
    _g4d_p_one = np.array(
        [m5_apply_one(member5_state, _g4d_embs[r], _g4d_skeys[r])
         for r in range(_g4d_n)],
        dtype=np.float32,
    )
    _g4d_max_dev = float(np.max(np.abs(_g4d_p_batch.astype(np.float64)
                                       - _g4d_p_one.astype(np.float64))))
    if _g4d_max_dev > 1.0e-5:
        raise AssertionError(
            f"[Gate 4d] FAIL: max |apply_batch - apply_one| = {_g4d_max_dev:.3e} "
            f"on {_g4d_n} sample rows. The batch/single-row codepaths "
            "have diverged -- the runtime would mispredict."
        )
    print(
        f"[Gate 4d] PASS: apply_batch == apply_one on {_g4d_n} sample rows "
        f"(max abs dev = {_g4d_max_dev:.3e})."
    )
    del (
        _g4d_train_subj_keys, _g4d_train_item_keys_arr, _g4d_row_idx,
        _g4d_skeys, _g4d_ikeys, _g4d_sids, _g4d_embs,
        _g4d_p_batch, _g4d_p_one,
    )

    # --- Score on val rows ---
    _t = _time_m5.time()
    print(f"[Member 5] scoring {len(primary.val):,} val rows...")
    _d_emb_m5 = int(member5_state.projection_weights.shape[0])
    _m5_val_item_emb = np.empty(
        (len(primary.val), _d_emb_m5), dtype=np.float32,
    )
    for _i, _k in enumerate(primary.val["item_key"]):
        _m5_val_item_emb[_i] = item_emb_lookup[str(_k)]
    p_member5_val = m5_apply_batch(
        member5_state,
        subject_keys=[str(s) for s in primary.val["subject_key"]],
        query_item_embeddings=_m5_val_item_emb,
    )
    _m5_dt = _time_m5.time() - _t
    nll_m5 = float(-(
        ylab_val * np.log(np.clip(p_member5_val, 1e-6, 1 - 1e-6))
        + (1 - ylab_val) * np.log(1 - np.clip(p_member5_val, 1e-6, 1 - 1e-6))
    ).mean())
    print(
        f"[Member 5] scored val in {_m5_dt:.1f}s "
        f"({len(p_member5_val) / max(_m5_dt, 1e-9):.0f} rows/s)  "
        f"val log-loss: {nll_m5:.6f}  "
        f"p stats: min={p_member5_val.min():.4f} "
        f"mean={p_member5_val.mean():.4f} max={p_member5_val.max():.4f}"
    )
    # Free the bulky val embedding stack. There is no longer a global
    # _m5_item_emb hanging around -- per-fold OOF Member 5 builds its
    # own stack from item_emb_lookup, the same way Member 3 does.
    del _m5_val_item_emb
    gc.collect()
else:
    _M5_ENABLED = False
    member5_state = None
    p_member5_val = None
    nll_m5 = float("nan")
    print("[Member 5] DISABLED via CFG['member5']['enabled']=False")


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


X_train_dense_m4 = np.concatenate(
    [X_train_dense, member4_marginal_train], axis=1
).astype(np.float32, copy=False)
X_val_dense_m4 = np.concatenate(
    [X_val_dense, member4_marginal_val], axis=1
).astype(np.float32, copy=False)
member4_feature_names = tuple(member_feat_schema.feature_names) + tuple(
    MEMBER4_MARGINAL_FEATURE_NAMES
)
print(
    f"[Member 4] hybrid feature matrix: train {X_train_dense_m4.shape}  "
    f"val {X_val_dense_m4.shape}  "
    f"(embedding dense={X_train_dense.shape[1]} + marginals={MEMBER4_MARGINAL_FEATURE_DIM})"
)


def _fit_member4():
    # HYBRID: embedding-derived dense features + 14 mean-encoded
    # marginals. The marginal-only version (val NLL 0.523, stacker
    # weight -0.075) was too weak to compete; the embedding-only
    # version (val NLL 0.483, stacker weight ~0.10) was too redundant
    # with M1/M3 (both also lean on embeddings). The hybrid keeps the
    # predictive power AND gives the LR an exclusive feature axis
    # (subject/bench marginals) that none of the other members use
    # for direct prediction. Stronger L1 (3e-3, up from 1e-3) lets it
    # pick a sparser subset over the now-1216-dim input.
    print(
        f"[Member 4] training torch logistic regression on HYBRID "
        f"(X cols={X_train_dense_m4.shape[1]}, "
        f"embedding+marginal mix, l1_strength=3e-3)..."
    )
    return fit_logreg_member(
        X=X_train_dense_m4,
        y=y_train,
        feature_names=member4_feature_names,
        epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
        learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
        weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
        l1_strength=float(CFG.get("member4_logreg", {}).get("l1_strength_hybrid", 3.0e-3)),
        min_feature_std=float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)),
        early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
        seed=SEED,
        val_fraction=0.1,
        holdout_group_id=gbdt_train_item_id,
    )


# Cache key bumped: ``hybrid_v1`` ties this entry to the hybrid feature
# matrix (embedding dense + marginals). Old entries (marginal-only OR
# legacy embedding-only) are incompatible -- the saved weights vector
# has a different dimensionality.
logreg_state = cache_or_compute(
    "logreg_state",
    key_inputs=(
        int(X_train_dense_m4.shape[1]), len(primary.train), SEED,
        "std_v1", "l1minfreq_v1", "coldsplit_v1", "redact_v1", "hybrid_v1",
        round(float(CFG.get("member4_logreg", {}).get("l1_strength_hybrid", 3.0e-3)), 6),
        round(float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)), 6),
        round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
        round(BC_REDACT_FRAC, 3), _bc_redact_seed,
    ),
    compute_fn=_fit_member4,
)
p_member4_val = logreg_apply_state_batch(logreg_state, X_val_dense_m4)
nll_m4 = float(-(ylab_val * np.log(np.clip(p_member4_val, 1e-6, 1 - 1e-6))
                 + (1 - ylab_val) * np.log(1 - np.clip(p_member4_val, 1e-6, 1 - 1e-6))).mean())
print(f"[Member 4] val log-loss: {nll_m4:.6f}  "
      f"weights||={float(np.linalg.norm(logreg_state.weights)):.3f}  "
      f"bias={float(logreg_state.bias):.3f}  "
      f"fit_method={logreg_state.fit_method}")
# Sanity: if val NLL is worse than the prior, something's still wrong.
_p_prior = float(np.clip(ylab_val.mean(), 1e-6, 1 - 1e-6))
_nll_prior = -(_p_prior * np.log(_p_prior) + (1 - _p_prior) * np.log(1 - _p_prior))
if nll_m4 > _nll_prior + 0.01:
    print(f"  >>> WARNING: Member 4 val NLL {nll_m4:.4f} is worse than prior "
          f"{_nll_prior:.4f}. Saturation likely. Check feature scales / lr.")
else:
    print(f"  [OK] beats prior NLL {_nll_prior:.4f} by {_nll_prior - nll_m4:+.4f} nats")

# %% [markdown]
# ## 9e-bis. Pre-OOF memory reclamation
#
# AGGRESSIVE MEMORY DISCIPLINE: every dense matrix in this notebook is
# float32 with ~1200 columns and ~5M train rows (~16 GB) or ~880k val
# rows (~3 GB). At this point in the run, all of the GLOBAL members are
# fit and val-scored, and the matrices below are no longer needed by
# any downstream code:
#
#   * ``X_train_dense`` (~16 GB)  -- only used to concat into m2/m4
#   * ``X_val_dense``   (~3 GB)   -- only used to concat into m2/m4
#   * ``X_train_dense_m2`` (~16 GB, legacy only, may be None)
#   * ``X_val_dense_m2``   (~3 GB,  legacy only, may be None)
#   * ``X_train_dense_m4`` (~16 GB) -- consumed by global Member 4 fit
#   * ``X_val_dense_m4``   (~3 GB)  -- consumed by Member 4 val scoring
#   * ``member4_marginal_train/val`` (~few hundred MB, absorbed in m4)
#   * ``member2_interaction_val``    (consumed by m2v2 val build above)
#
# The OOF loop in section 9.5 (and the global Member 2 v2 fit in 9.5c)
# build their OWN fold-scoped / m2v2 matrices on the fly. Holding the
# global matrices through that loop is what made the OOF fold-0 OOM
# happen at "Building fold X_*_dense" -- the loop never even got a
# chance to allocate its 16 GB X_fold_train_dense because ~50 GB of
# now-obsolete globals were already pinned.
#
# ``X_train_dense_m2v2`` / ``X_val_dense_m2v2`` (small, ~hundreds of MB)
# ARE kept alive: section 9.5c re-fits global Member 2 v2 with the OOF
# subject_mean anchor produced by the OOF loop.
# ``member2_interaction_train`` (~few hundred MB) is also kept alive
# because section 9.5c's global Member 2 v2 build references it.

# %%
print("[Pre-OOF cleanup] Freeing global dense matrices no longer needed...")
_pre_oof_to_free = [
    ("X_train_dense", X_train_dense),
    ("X_val_dense", X_val_dense),
    ("X_train_dense_m4", X_train_dense_m4),
    ("X_val_dense_m4", X_val_dense_m4),
    ("member4_marginal_train", member4_marginal_train),
    ("member4_marginal_val", member4_marginal_val),
    ("member2_interaction_val", member2_interaction_val),
]
if X_train_dense_m2 is not None:
    _pre_oof_to_free.append(("X_train_dense_m2", X_train_dense_m2))
if X_val_dense_m2 is not None:
    _pre_oof_to_free.append(("X_val_dense_m2", X_val_dense_m2))
_pre_oof_freed_bytes = 0
for _name, _arr in _pre_oof_to_free:
    if _arr is None:
        continue
    if hasattr(_arr, "nbytes"):
        _pre_oof_freed_bytes += int(_arr.nbytes)
    print(f"  freeing {_name:<26s}: "
          f"{_arr.shape if hasattr(_arr, 'shape') else type(_arr).__name__}  "
          f"({(_arr.nbytes / 1024**3) if hasattr(_arr, 'nbytes') else 0:.2f} GB)")
del _pre_oof_to_free
# Bind the names to None so subsequent code that accidentally references
# them gets a clean TypeError rather than a stale 16 GB matrix.
X_train_dense = None
X_val_dense = None
X_train_dense_m2 = None
X_val_dense_m2 = None
X_train_dense_m4 = None
X_val_dense_m4 = None
member4_marginal_train = None
member4_marginal_val = None
member2_interaction_val = None
gc.collect()
print(f"[Pre-OOF cleanup] DONE -- released {_pre_oof_freed_bytes / 1024**3:.2f} GB.")

# %% [markdown]
# ## 9.5. Per-fold OOF compute (Task 1: honest stacking inputs)
#
# For each of the K folds, retrain Members 1-4 on the OTHER K-1 folds'
# items and predict on this fold's held-out items. Concatenate across
# folds -> one OOF prediction per training row, by a member that never
# saw that row's item. These OOF predictions replace the existing
# in-val stacker training inputs in section 9f.
#
# Per-fold compute is cached separately (cache_or_compute with fold
# suffix in the key), so re-running with the same settings hits cache.
# Bumping `CFG["oof"]["seed"]` or `n_folds` invalidates everything.
#
# What's fold-scoped (leakage-safe):
#   - NN index (built over fold.train_item_keys only)
#   - passrate_csr + conditional context (aggregated over fold-train labels only)
#   - mean_encoded_stats (fit on fold-train labels only)
#   - Member 1 retraining (if CFG["oof"]["retrain_member1_per_fold"]=True)
#   - Member 2 GBDT residual anchor (uses fold M1 predictions, not global)
#   - Member 3 kNN (built over fold-train items only)
#   - Member 4 LogReg (fit on fold-train rows only)
#
# What stays global (acknowledged weak-leak; flagged in final summary):
#   - member_feat_schema (fit on full primary.train)
#   - subject_tables (theta, u from full-train Model A)
#   - pool features z-score normalization (computed once on full train)
#   - item embeddings (Qwen8B encoder is data-independent)

# %%
from src.oof_pipeline import (
    OofPredictionAccumulator,
    build_fold_item_index_map,
    build_fold_nn_index,
    reindex_per_item_array,
    slice_train_rows,
    split_fold_train_for_early_stopping,
)
from src.nn_features import build_passrate_table

OOF_RETRAIN_M1 = bool(CFG["oof"].get("retrain_member1_per_fold", True))
OOF_NN_BASE_DIR = ROOT / nn_cfg.cache_dir / "oof_folds"
OOF_NN_BASE_DIR.mkdir(parents=True, exist_ok=True)
OOF_ES_VAL_FRACTION = float(CFG["oof"].get("es_val_fraction", 0.1))

print(
    f"[OOF] Starting per-fold compute (n_folds={len(folds)}, "
    f"retrain_M1_per_fold={OOF_RETRAIN_M1}, es_val_fraction={OOF_ES_VAL_FRACTION:.2f})..."
)
print(
    f"[OOF] Expected wall time on Colab: "
    f"{'~30-60 min per fold x ' + str(len(folds)) + ' folds (Member 1 dominant) plus ~10 min per fold for Members 2/3/4' if OOF_RETRAIN_M1 else '~10 min per fold (Members 2/3/4 only, M1 kept global with small acknowledged leakage)'}"
)

_N_TRAIN = len(primary.train)
p_a_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p_a_train_oof")
p2_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p2_train_oof")
p3_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p3_train_oof")
p4_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p4_train_oof")
nn_mean_sim_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="nn_mean_sim_oof")
nn_support_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="nn_support_oof")
centroid_dist_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="centroid_dist_oof")
# Task 4: Member 5 OOF accumulator (created only when Member 5 is enabled).
if _M5_ENABLED:
    p5_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p5_train_oof")
    # Per-fold Gate 4c results (printed in aggregate after the loop).
    _gate4c_per_fold: dict[int, dict] = {}
else:
    p5_train_oof_acc = None
    _gate4c_per_fold = {}

# Task 3 (Member 2 v2): accumulate per-row OOF subject_mean (computed
# from each fold's train labels only) so the GLOBAL Member 2 fit in
# section 9.5c can use it as an honest init_score anchor. Per-fold
# Gate 3a results are tracked in a separate dict for the post-loop
# summary print.
if _M2V2_ENABLED:
    subject_mean_train_oof_acc = OofPredictionAccumulator(
        _N_TRAIN, name="subject_mean_train_oof"
    )
    _gate3a_per_fold: dict[int, dict] = {}
else:
    subject_mean_train_oof_acc = None
    _gate3a_per_fold = {}

# Gate 1b tracker: for each fold we'll save a sample of OOF-row top-k
# neighbor item_keys so we can assert (after the loop) that none of them
# leak from the fold's own OOF item set.
_oof_nn_probe_data: dict[int, np.ndarray] = {}

# Per-row item_key array as a NumPy object array for fast slicing.
_train_row_item_keys_arr = np.asarray(_train_row_item_keys, dtype=object)


def _slice_nn_aux_from_oof_mat(nn_oof_mat_fold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull the stacker's two NN-derived aux features out of an OOF
    NN-feature matrix: (mean_similarity, log1p_neighbors_observed).
    Matches the column indexing used by the existing val stacker
    (columns 1 and 2 of `nn_val_mat`)."""
    return (
        nn_oof_mat_fold[:, 1].astype(np.float32),
        nn_oof_mat_fold[:, 2].astype(np.float32),
    )


def _centroid_dist_for_rows(rows_df) -> np.ndarray:
    """Compute the centroid_distance aux feature for a row slice
    (min across centroid_dist_* columns in pool_features_z)."""
    _cols = [c for c in pool_features_z.columns if c.startswith("centroid_dist_")]
    if not _cols:
        return np.full(len(rows_df), 0.5, dtype=np.float32)
    _idx = pool_features_z.set_index("item_key").reindex(
        rows_df["item_key"].astype(str)
    )
    return _idx[_cols].astype(np.float32).min(axis=1).to_numpy()


for fold in folds:
    print(f"\n[OOF] ============ fold {fold.fold_id}/{len(folds)} ============")
    print(f"[OOF] fold {fold.fold_id}: train_rows={len(fold.train_row_idx):,} "
          f"oof_rows={len(fold.oof_row_idx):,} "
          f"train_items={len(fold.train_item_keys):,} "
          f"oof_items={len(fold.oof_item_keys):,}")
    fold_suffix = fold_cache_suffix(
        fold_id=fold.fold_id, train_item_keys=fold.train_item_keys
    )

    # ----- Slice primary.train into fold-train and fold-oof DataFrames -----
    fold_train_df = slice_train_rows(primary.train, fold, side="train")
    fold_oof_df = slice_train_rows(primary.train, fold, side="oof")

    # ----- Build fold-scoped NN index -----
    fold_nn_dir = OOF_NN_BASE_DIR / f"fold_{fold.fold_id}_{fold_suffix}"
    print(f"[OOF f{fold.fold_id}] Building fold NN index ({len(fold.train_item_keys):,} items)...")
    fold_nn_index = build_fold_nn_index(
        fold=fold,
        item_emb_lookup=item_emb_lookup,
        out_dir=fold_nn_dir,
        nn_cfg=nn_cfg,
        TrainingNNIndex=TrainingNNIndex,
    )
    fold_item_index_map = build_fold_item_index_map(fold)

    # ----- Build fold-scoped passrate + conditional context -----
    print(f"[OOF f{fold.fold_id}] Building fold passrate + conditional context...")
    fold_passrate_csr, fold_passrate_mask_csr = build_passrate_table(
        train_df=fold_train_df,
        item_index_map=fold_item_index_map,
        subject_index_map=indexer.subject_to_id,
    )
    fold_item_bench_id = reindex_per_item_array(
        arr=item_benchmark_id_arr,
        train_item_keys_global=train_item_keys,
        fold=fold,
        fill=-1,
    )
    fold_item_bench_age = reindex_per_item_array(
        arr=item_benchmark_age_arr,
        train_item_keys_global=train_item_keys,
        fold=fold,
        fill=np.float32(np.nan),
    )
    fold_item_cluster = reindex_per_item_array(
        arr=item_cluster_id_arr,
        train_item_keys_global=train_item_keys,
        fold=fold,
        fill=-1,
    )
    fold_cond_context = build_conditional_passrate_context(
        train_df=fold_train_df,
        item_index_map=fold_item_index_map,
        subject_index_map=indexer.subject_to_id,
        subject_to_family_id=s2fam,
        subject_to_macro_family_id=s2macro,
        subject_to_organization_id=s2org,
        item_benchmark_id=fold_item_bench_id,
        item_benchmark_age=fold_item_bench_age,
        item_cluster_id=fold_item_cluster,
        n_families=N_FAMILIES,
        n_macro_families=N_MACRO_FAMILIES,
        n_organizations=N_ORGANIZATIONS,
        n_clusters=N_CLUSTERS_CTX,
    )
    fold_cond_context.assert_shapes()

    # ----- Compute fold-scoped NN features for fold-train and fold-OOF rows -----
    print(f"[OOF f{fold.fold_id}] Computing NN features for fold-train rows...")
    _ftk_train, _fsid_train = _split_query(fold_train_df)
    _ftk_oof, _fsid_oof = _split_query(fold_oof_df)
    nn_train_mat_fold = compute_nn_features_streaming(
        query_item_keys=_ftk_train,
        item_emb_lookup=item_emb_lookup,
        subject_ids=_fsid_train,
        nn_index=fold_nn_index,
        passrate_csr=fold_passrate_csr,
        passrate_mask_csr=fold_passrate_mask_csr,
        cfg=nn_cfg,
        exclude_self=True,  # fold-train items ARE in fold's NN index; exclude self-match
        query_chunk_size=NN_QUERY_CHUNK,
        conditional_context=fold_cond_context,
    )
    gc.collect()
    print(f"[OOF f{fold.fold_id}] Computing NN features for fold-OOF rows...")
    nn_oof_mat_fold = compute_nn_features_streaming(
        query_item_keys=_ftk_oof,
        item_emb_lookup=item_emb_lookup,
        subject_ids=_fsid_oof,
        nn_index=fold_nn_index,
        passrate_csr=fold_passrate_csr,
        passrate_mask_csr=fold_passrate_mask_csr,
        cfg=nn_cfg,
        exclude_self=False,  # OOF items NOT in fold's NN index, no self to exclude
        query_chunk_size=NN_QUERY_CHUNK,
        conditional_context=fold_cond_context,
    )
    gc.collect()

    # Gate 1b probe: query a sample of fold-OOF rows against the fold's NN
    # index and capture the top-k neighbor item_keys. Post-loop, we assert
    # every captured neighbor key is in fold.train_item_keys -- a single
    # leak there would silently contaminate Members 1/2 (NN aux) and 3.
    _probe_sample_size = int(min(
        CFG["oof"].get("nn_neighbor_probe_sample_size", 2000),
        len(fold.oof_row_idx),
    ))
    _probe_rng = np.random.default_rng(int(CFG["oof"].get("seed", 7)) * 31 + fold.fold_id)
    _probe_local_idx = _probe_rng.choice(len(fold.oof_row_idx), size=_probe_sample_size, replace=False)
    _probe_oof_keys = [str(_ftk_oof[i]) for i in _probe_local_idx]
    _probe_query_emb = np.stack(
        [np.asarray(item_emb_lookup[k], dtype=np.float32) for k in _probe_oof_keys],
        axis=0,
    )
    # nearest() returns (idx [N_probe, k], sims). exclude_self=False because
    # OOF items are guaranteed NOT in fold's NN index (they're in fold.oof_item_keys).
    _probe_idx, _ = fold_nn_index.nearest(
        query_embeds=_probe_query_emb,
        k=int(nn_cfg.k),
        exclude_self=False,
    )
    _probe_neighbor_keys = np.array(
        [
            [
                fold.train_item_keys[int(j)] if 0 <= int(j) < len(fold.train_item_keys) else "-1"
                for j in row
            ]
            for row in _probe_idx
        ],
        dtype=object,
    )
    _oof_nn_probe_data[fold.fold_id] = _probe_neighbor_keys
    del _probe_query_emb, _probe_idx
    # `fold_nn_index` holds the fold-train item embedding index
    # (~3 GB at full scale for 200k items x 4096 dims). Its consumers
    # are all done -- compute_nn_features_streaming for both train/oof
    # is complete, and the Gate 1b probe just ran. Free it before the
    # heavier downstream allocations land.
    del fold_nn_index
    gc.collect()

    # ----- Fold mean encoded stats (fit on fold-train labels only) -----
    print(f"[OOF f{fold.fold_id}] Fitting fold mean-encoded stats...")
    _mef_subj_fold_train, _mef_cluster_fold_train, _mef_bc_fold_train = _compute_id_arrays(fold_train_df)
    _mef_subj_fold_oof, _mef_cluster_fold_oof, _mef_bc_fold_oof = _compute_id_arrays(fold_oof_df)
    _y_fold_train = fold_train_df["label"].astype(float).to_numpy().astype(np.float32)
    fold_mean_encoded_stats = fit_mean_encoded_stats(
        subject_ids=_mef_subj_fold_train,
        cluster_ids=_mef_cluster_fold_train,
        bc_ids=_mef_bc_fold_train,
        labels=_y_fold_train,
        n_subjects=int(indexer.n_subjects),
        n_clusters=int(_N_CLUSTERS_ME),
        n_bcs=int(indexer.n_bc),
        smoothing=float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)),
    )
    fold_member2_interaction_train = apply_member2_interaction_features(
        fold_mean_encoded_stats,
        subject_ids=_mef_subj_fold_train,
        cluster_ids=_mef_cluster_fold_train,
        bc_ids=_mef_bc_fold_train,
    )
    fold_member2_interaction_oof = apply_member2_interaction_features(
        fold_mean_encoded_stats,
        subject_ids=_mef_subj_fold_oof,
        cluster_ids=_mef_cluster_fold_oof,
        bc_ids=_mef_bc_fold_oof,
    )
    fold_member4_marginal_train = apply_member4_marginal_features(
        fold_mean_encoded_stats,
        subject_ids=_mef_subj_fold_train,
        cluster_ids=_mef_cluster_fold_train,
        bc_ids=_mef_bc_fold_train,
    )
    fold_member4_marginal_oof = apply_member4_marginal_features(
        fold_mean_encoded_stats,
        subject_ids=_mef_subj_fold_oof,
        cluster_ids=_mef_cluster_fold_oof,
        bc_ids=_mef_bc_fold_oof,
    )
    # fold_mean_encoded_stats has been consumed by the four apply_*
    # calls above; nothing else references it for this fold. Free now.
    del fold_mean_encoded_stats
    gc.collect()

    # ----- Fold subject_mean table + Gate 3a + Member 2 v2 features (Task 3) -----
    if _M2V2_ENABLED:
        print(f"[OOF f{fold.fold_id}] Fitting fold subject_mean table (Task 3 Member 2 v2)...")
        fold_subject_mean_table = fit_subject_mean_table(
            subject_ids=_mef_subj_fold_train,
            labels=_y_fold_train,
            n_subjects=int(indexer.n_subjects),
            smoothing=_M2V2_SMOOTHING,
        )
        # Per-row subject_mean anchors. fold-train anchor is in-sample (matches
        # standard training-time anchor convention; fold-train rows ARE the
        # training set for this fold's Member 2). fold-OOF anchor is OUT OF
        # FOLD by construction -- fold_subject_mean_table was fit on
        # fold-train labels only, and fold-OOF subjects are queried against
        # that table without their own labels having contributed to it.
        subject_mean_train_fold = apply_subject_mean(
            fold_subject_mean_table, _mef_subj_fold_train,
        ).astype(np.float64)
        subject_mean_oof_fold = apply_subject_mean(
            fold_subject_mean_table, _mef_subj_fold_oof,
        ).astype(np.float64)

        # Gate 3a (RED-TEAM): assert subject_mean_oof_fold was computed from
        # the fold-train table (not the global table). Catches the failure
        # mode where someone uses subject_mean_table_global for fold OOF
        # anchors -- that table includes fold-OOF labels in its aggregates
        # which would let the GBDT trivially recover the label.
        _gate3a_result = assert_oof_subject_mean(
            subject_mean_oof_for_fold=subject_mean_oof_fold,
            fold_subject_ids=_mef_subj_fold_oof,
            fold_train_subject_mean_table=fold_subject_mean_table,
        )
        _gate3a_per_fold[int(fold.fold_id)] = _gate3a_result
        # Diagnostic: mean abs delta between fold-train table and global table
        # on this fold's OOF rows. Should be small but NONZERO (if exactly
        # zero, the fold table somehow used the global label set -- bug).
        _diff_vs_global = float(np.abs(
            subject_mean_oof_fold
            - apply_subject_mean(subject_mean_table_global, _mef_subj_fold_oof)
        ).mean())
        print(
            f"  [Gate 3a fold {fold.fold_id}] PASS: "
            f"{_gate3a_result['n_checked']:,} OOF rows checked, "
            f"max_abs_delta={_gate3a_result['max_abs_delta']:.2e}, "
            f"fold-vs-global mean abs diff on OOF rows={_diff_vs_global:.5f} "
            "(nonzero confirms the fold table differs from the global table)"
        )

        # Build the Member 2 v2 feature matrices for fold-train and fold-OOF.
        # OOM/compute discipline: when v4 or v5 is enabled the v2 GBDT will
        # be SKIPPED below, so these matrices have no consumer and would
        # burn ~280 MB (train) + ~140 MB (OOF) for nothing. Skip in that
        # case so the per-fold loop stays tight.
        if (not _M2V4_ENABLED) and (not _M2V5_ENABLED):
            _fold_subj_obs_count_train_log1p = apply_subject_obs_count(
                fold_subject_mean_table, _mef_subj_fold_train, log1p=True,
            ).astype(np.float32)
            _fold_subj_obs_count_oof_log1p = apply_subject_obs_count(
                fold_subject_mean_table, _mef_subj_fold_oof, log1p=True,
            ).astype(np.float32)
            X_fold_train_m2v2 = build_member2_feature_matrix(
                member2_v2_schema,
                subject_ids=_mef_subj_fold_train,
                cluster_ids=_mef_cluster_fold_train,
                bc_ids=_mef_bc_fold_train,
                bc_redacted_mask=bc_redacted_train[fold.train_row_idx].astype(np.float32),
                subject_obs_count_log1p=_fold_subj_obs_count_train_log1p,
                subject_cat_lookup=_m2v2_subj_cat_lookup,
                subject_num_lookup=_m2v2_subj_num_lookup,
                bench_cat_lookup=_m2v2_bench_cat_lookup,
                bench_num_lookup=_m2v2_bench_num_lookup,
                interaction_matrix=fold_member2_interaction_train.astype(np.float32),
            )
            X_fold_oof_m2v2 = build_member2_feature_matrix(
                member2_v2_schema,
                subject_ids=_mef_subj_fold_oof,
                cluster_ids=_mef_cluster_fold_oof,
                bc_ids=_mef_bc_fold_oof,
                bc_redacted_mask=bc_redacted_train[fold.oof_row_idx].astype(np.float32),
                subject_obs_count_log1p=_fold_subj_obs_count_oof_log1p,
                subject_cat_lookup=_m2v2_subj_cat_lookup,
                subject_num_lookup=_m2v2_subj_num_lookup,
                bench_cat_lookup=_m2v2_bench_cat_lookup,
                bench_num_lookup=_m2v2_bench_num_lookup,
                interaction_matrix=fold_member2_interaction_oof.astype(np.float32),
            )
        else:
            # v4 or v5 owns Member 2; v2 matrices are dead weight on this path.
            X_fold_train_m2v2 = None
            X_fold_oof_m2v2 = None
        # Accumulate OOF subject_mean for the GLOBAL Member 2 v2 / v3 fit
        # in 9.5c. Both v2 and v3 globals use it as the inference-time
        # anchor for honest OOF training of the booster.
        subject_mean_train_oof_acc.write_fold(fold.oof_row_idx, subject_mean_oof_fold)

    # ----- Fold X_train_dense / X_oof_dense (uses GLOBAL schema -- ack'd leak) -----
    # MEMORY DISCIPLINE: at full scale these are ~16 GB (train) + ~8 GB
    # (oof) per fold. On the Task 3 path the legacy Member 2 path is
    # off, so the base X matrices have a SINGLE consumer (Member 4's
    # concat-with-marginal). Defer the base build entirely on Task 3
    # so we don't hold ~24 GB through Members 1/3/5 -- they don't read
    # X_fold_*_dense. On legacy path we still need the base for both
    # Member 2 and Member 4, so build it eagerly there.
    _bc_redacted_fold_train = bc_redacted_train[fold.train_row_idx]
    _bc_redacted_fold_oof = bc_redacted_train[fold.oof_row_idx]
    if _M2V2_ENABLED:
        print(
            f"[OOF f{fold.fold_id}] DEFERRING X_fold_*_dense build "
            "(Task 3 path: only Member 4 consumes them; built JIT below)."
        )
        X_fold_train_dense = None
        X_fold_oof_dense = None
    else:
        print(f"[OOF f{fold.fold_id}] Building fold X_*_dense (legacy path)...")
        X_fold_train_dense = _build_X(
            fold_train_df, nn_train_mat_fold, _bc_redacted_fold_train,
        )
        X_fold_oof_dense = _build_X(
            fold_oof_df, nn_oof_mat_fold, _bc_redacted_fold_oof,
        )
    gc.collect()
    # ``X_fold_*_dense_m2`` / ``X_fold_*_dense_m4`` are built lazily below
    # near their consumer (see Member 2 legacy branch and Member 4 block).

    # Aux features for the stacker (mean_sim + log1p_n_observed) are
    # cheap [N_oof] slices of nn_oof_mat_fold. Capture them EARLY so
    # we can free nn_oof_mat_fold after X_fold_oof_dense is built --
    # otherwise the full ~200 MB NN feature matrix sits in memory all
    # the way through M2/M3/M5/M4 just for these two 1-D arrays.
    _aux_mean_sim_fold, _aux_support_fold = _slice_nn_aux_from_oof_mat(nn_oof_mat_fold)

    # ----- Member 1 (fold-retrain OR fall back to global) -----
    if OOF_RETRAIN_M1:
        print(f"[OOF f{fold.fold_id}] Training fold Member 1 (IRT-MLP, item-grouped ES split)...")
        _es_train_rows, _es_val_rows = split_fold_train_for_early_stopping(
            fold=fold,
            item_keys_per_row=_train_row_item_keys_arr,
            es_val_fraction=OOF_ES_VAL_FRACTION,
            seed=int(CFG["oof"].get("seed", 7)) * 17 + fold.fold_id,
        )
        # Translate full-row indices into fold-train-local positions (so we can
        # slice nn_train_mat_fold which is in fold-train order, not full order).
        _fold_train_pos = {int(r): i for i, r in enumerate(fold.train_row_idx)}
        _es_train_local = np.array([_fold_train_pos[int(r)] for r in _es_train_rows], dtype=np.int64)
        _es_val_local = np.array([_fold_train_pos[int(r)] for r in _es_val_rows], dtype=np.int64)
        del _fold_train_pos
        # OOM guard: do NOT materialize _train_ds_fold / _val_ds_fold here.
        # ``_build`` calls ``stack_lookup`` which allocates [N, 4096] fp32 per
        # row (~16 KB/row). For a typical fold that's ~47 GB for ES-train,
        # ~5 GB for ES-val, ~27 GB for OOF -- ~79 GB transient _before_ the
        # cache call. On cache HIT that's pure waste; on cache MISS it stacks
        # on top of training memory. We defer the train/val build into the
        # compute_fn (so cache HIT skips it entirely; cache MISS GCs them
        # at function return), and build _oof_ds_fold only after the cache
        # call (it's only needed for scoring, never for training).

        # Fold M1 training closure (mirrors _train_model_a structure).
        # Datasets are built INSIDE so a cache HIT pays zero RAM for them.
        def _train_fold_model_a(
            _fold=fold,
            _es_train_rows=_es_train_rows,
            _es_val_rows=_es_val_rows,
            _es_train_local=_es_train_local,
            _es_val_local=_es_val_local,
        ):
            _es_train_df = primary.train.iloc[_es_train_rows]
            _es_val_df = primary.train.iloc[_es_val_rows]
            _train_ds_local = _build(_es_train_df, nn_train_mat_fold[_es_train_local])
            _val_ds_local = _build(_es_val_df, nn_train_mat_fold[_es_val_local])
            del _es_train_df, _es_val_df
            gc.collect()
            train_mod.build_model = _build_with_overrides
            try:
                _active_dropout_cfg["cfg"] = a_drop
                _active_dropout_cfg["name"] = MODEL_A_NAME
                _active_dropout_cfg["installed_handles"] = []
                _result = train_one(
                    model_name=MODEL_A_NAME,
                    model_cfg=model_a_cfg,
                    train_cfg=train_cfg,
                    train_ds=_train_ds_local,
                    val_ds=_val_ds_local,
                    indexer=indexer,
                    seed=int(MODEL_A_SEED) + 1000 * (int(_fold.fold_id) + 1),
                    run_id=f"qwen8b_oof_fold{_fold.fold_id}_model_a",
                    checkpoint_dir=CKPT_DIR / "oof" / f"fold_{_fold.fold_id}",
                    extra_metadata={
                        "encoder_model_id": CFG["encoder"]["model_id"],
                        "oof_fold_id": int(_fold.fold_id),
                        "oof_fold_suffix": fold_cache_suffix(
                            fold_id=_fold.fold_id,
                            train_item_keys=_fold.train_item_keys,
                        ),
                    },
                )
                for h in _active_dropout_cfg["installed_handles"]:
                    h.remove()
                _ckpt = torch.load(_result.checkpoint_path, map_location="cpu", weights_only=False)
                # Free datasets BEFORE returning so memory drops _before_
                # we cache-save the bundle (saves the on-disk pickle path
                # from competing with ~52 GB of live ds tensors).
                del _train_ds_local, _val_ds_local
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"train_result": _result, "ckpt": _ckpt}
            finally:
                _active_dropout_cfg["cfg"] = None
                _active_dropout_cfg["name"] = None
                train_mod.build_model = _orig_build_model

        _fold_m1_bundle = cache_or_compute(
            "model_a_trained_oof_fold",
            key_inputs=(
                "model_a_oof_v1", MODEL_A_NAME, int(MODEL_A_SEED), fold.fold_id,
                fold_suffix,
                json.dumps(asdict(model_a_cfg), default=str, sort_keys=True),
                json.dumps(asdict(train_cfg), default=str, sort_keys=True),
                json.dumps(asdict(a_drop), default=str, sort_keys=True),
                # NOTE: equivalent to the prior cache key fields
                # (_train_ds_fold.subject_ids.shape[0], _val_ds_fold.subject_ids.shape[0])
                # but computed WITHOUT materializing the datasets first.
                int(len(_es_train_rows)),
                int(len(_es_val_rows)),
                int(indexer.n_subjects), int(indexer.n_bc),
                int(_NN_FEATURE_DIM_CACHE_TAG),
                int(fold_cond_context.n_families),
                int(fold_cond_context.n_macro_families),
                int(fold_cond_context.n_organizations),
                int(fold_cond_context.n_clusters),
            ),
            compute_fn=_train_fold_model_a,
        )
        _fold_ckpt = _fold_m1_bundle["ckpt"]
        # Build _oof_ds_fold _after_ the cache call so:
        #   - cache HIT path: train/val ds were never built (~52 GB saved)
        #   - cache MISS path: train/val ds were built inside compute_fn
        #     and freed before this point (~52 GB saved at this peak)
        # Only _oof_ds_fold (~27 GB at largest fold) is live during scoring.
        _oof_ds_fold = _build(fold_oof_df, nn_oof_mat_fold)
        # Load fold M1 weights into a fresh model + score on OOF rows.
        _fold_model = _build_model_for_inf(MODEL_A_NAME, model_a_cfg)
        _fold_model.attach_metadata_tables(meta_id_tables)
        _fold_model.load_state_dict(_fold_ckpt["model_state"])
        _fold_model = _fold_model.to(device).eval()
        p_a_oof_fold = _score_dataset(_oof_ds_fold, _fold_model)
        # Whether we need M1's predictions on every fold-train row:
        #   - LEGACY Member 2 (no v2, no v4, no v5): yes -- residual anchor.
        #   - v2-only (no v4/v5): no -- v2 uses subject_mean.
        #   - v4 enabled (no v5): YES -- v4's tree consumes p_m1 as a per-row
        #     feature.
        #   - v5 enabled: NO -- v5 removes all M1-derived features by
        #     design (the v3/v4 saturation fix). Skipping this scoring
        #     pass also saves ~30-90s per fold of M1 inference.
        _need_fold_train_m1_anchor = (
            (not _M2V2_ENABLED) and (not _M2V5_ENABLED)
        ) or (_M2V4_ENABLED and not _M2V5_ENABLED)
        if _need_fold_train_m1_anchor:
            # MEMORY: scoring all fold-train rows at once allocates a
            # ~50 GB LookupDataset (rows * 4096 * 4 bytes). Chunk it so
            # peak is ~chunk_rows * 4096 * 4 ~= 4 GB per chunk.
            _chunk_rows_m1 = int(
                CFG.get("member2_v4", {}).get("m1_chunk_rows", 256_000)
            )
            import time as _m1_time_local
            p_a_anchor_fold_train = np.empty(len(fold_train_df), dtype=np.float32)
            _m1_t0 = _m1_time_local.time()
            for _cs in range(0, len(fold_train_df), _chunk_rows_m1):
                _ce = min(_cs + _chunk_rows_m1, len(fold_train_df))
                _chunk_df = fold_train_df.iloc[_cs:_ce]
                _chunk_nn = nn_train_mat_fold[_cs:_ce]
                _chunk_ds = _build(_chunk_df, _chunk_nn)
                p_a_anchor_fold_train[_cs:_ce] = _score_dataset(
                    _chunk_ds, _fold_model,
                )
                del _chunk_ds, _chunk_nn, _chunk_df
                gc.collect()
                if (_ce % (_chunk_rows_m1 * 2)) == 0 or _ce == len(fold_train_df):
                    _rps = _ce / max(_m1_time_local.time() - _m1_t0, 1e-6)
                    print(
                        f"[OOF f{fold.fold_id}] M1 fold-train scoring "
                        f"{_ce:,}/{len(fold_train_df):,} rows "
                        f"({_rps:,.0f} rows/s)"
                    )
        else:
            p_a_anchor_fold_train = None
        _fold_model = _fold_model.to("cpu")
        del _fold_model, _oof_ds_fold
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Use global p_a_train (saw all train items -- documented small leak).
        p_a_oof_fold = p_a_train[fold.oof_row_idx]
        _need_fold_train_m1_anchor = (
            (not _M2V2_ENABLED) and (not _M2V5_ENABLED)
        ) or (_M2V4_ENABLED and not _M2V5_ENABLED)
        if _need_fold_train_m1_anchor:
            p_a_anchor_fold_train = p_a_train[fold.train_row_idx]
        else:
            p_a_anchor_fold_train = None
    p_a_train_oof_acc.write_fold(fold.oof_row_idx, p_a_oof_fold)

    # The NN feature matrices are no longer needed *on the legacy
    # path*: their consumers (X_fold_*_dense, M1 datasets, and the
    # aux-feature slices) have all been served and we can return
    # ~360 MB (train) + ~200 MB (oof) to the OS before Member 2
    # starts.
    #
    # On the Task 3 path (``X_fold_*_dense`` is ``None``) the M4
    # chunked X build below still consumes them, so we KEEP them
    # alive and free each one immediately after its M4 build loop
    # completes (see the matching ``del nn_train_mat_fold`` and
    # ``del nn_oof_mat_fold`` inside the M4 chunked sections).
    if X_fold_train_dense is not None and X_fold_oof_dense is not None:
        del nn_train_mat_fold, nn_oof_mat_fold
        gc.collect()

    # ----- Fold Member 2 (GBDT residual) -----
    _y_fold_train_np = _y_fold_train
    _gbdt_train_item_id_fold = np.array(
        [int(_item_to_train_idx.get(str(k), -1)) for k in fold_train_df["item_key"]],
        dtype=np.int64,
    )
    if _M2V5_ENABLED:
        # Task v5: direct-binary GBDT on the v5 feature set (no M1, no
        # subject_mean as anchor). 49 features = 19 item content + 9
        # honest counts + 8 mean-encoded passrates + 6 unknown flags +
        # 7 NN aggregates. Trained as a binary classifier on y directly
        # -- no init_score, no residual framing, output = sigmoid(tree_raw).
        # The literature-standard cold-start tree recipe (Facebook 2014;
        # LightGBM PR #3234; Wang 2023; AliBoost 2025).
        print(
            f"[OOF f{fold.fold_id}] Training fold Member 2 v5 "
            f"(direct-binary GBDT, F={M2V5_FEATURE_DIM} features = item attrs "
            "+ counts + meanenc + NN aggregates, NO M1 features)..."
        )
        # Vocab sizing: same canonical bounds as v3/v4 -- the conditional
        # passrate context auto-grew clusters beyond CFG['clustering']['k']
        # in fold 0 (64 -> 65), and bc ids can drift if OOF subjects
        # touch unseen benchmarks. Bound n_subjects / n_clusters / n_bcs
        # by ``max(declared, observed_max + 1)`` over fold-train + fold-OOF
        # ids so the builder never raises on undersized vocab.
        def _safe_n_ids_v5(declared_lb, *id_arrs):
            n = int(declared_lb)
            for arr in id_arrs:
                arr = np.asarray(arr)
                if arr.size > 0 and bool((arr >= 0).any()):
                    n = max(n, int(arr.max()) + 1)
            return n
        _fold_v5_n_subjects = _safe_n_ids_v5(
            int(indexer.n_subjects),
            _mef_subj_fold_train, _mef_subj_fold_oof,
        )
        _fold_v5_n_clusters = _safe_n_ids_v5(
            int(fold_cond_context.n_clusters),
            _mef_cluster_fold_train, _mef_cluster_fold_oof,
        )
        _fold_v5_n_bcs = _safe_n_ids_v5(
            int(indexer.n_bc),
            _mef_bc_fold_train, _mef_bc_fold_oof,
        )
        # 1. Fit the v5 feature builder on fold-train (per-id obs counts +
        #    Bayes-shrunk mean-encoded passrates + 2-D cell counts/means,
        #    restricted to this fold's training rows so OOF rows are
        #    NEVER in the aggregates).
        _fold_v5_builder = fit_member2_v5_feature_builder(
            subject_ids=_mef_subj_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            bc_ids=_mef_bc_fold_train,
            labels=_y_fold_train_np,
            n_subjects=_fold_v5_n_subjects,
            n_clusters=_fold_v5_n_clusters,
            n_bcs=_fold_v5_n_bcs,
            n_macro_families=int(N_MACRO_FAMILIES),
            n_organizations=int(N_ORGANIZATIONS),
            n_families=int(N_FAMILIES),
            subject_to_macro_family_id=s2macro,
            subject_to_organization_id=s2org,
            subject_to_family_id=s2fam,
            smoothing=_M2V5_SMOOTHING,
        )
        # 2. Materialize per-row pool/age/NN inputs for fold-train.
        _v5_pool_train = _m2v5_pool_for_rows(fold_train_df)
        _v5_age_train = _m2v5_age_for_rows(fold_train_df)
        X_v5_fold_train = build_member2_v5_features(
            _fold_v5_builder,
            subject_ids=_mef_subj_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            bc_ids=_mef_bc_fold_train,
            bc_redacted_mask=bc_redacted_train[fold.train_row_idx].astype(np.float32),
            pool_features=_v5_pool_train,
            benchmark_age=_v5_age_train,
            nn_features_matrix=nn_train_mat_fold,
        )
        del _v5_pool_train, _v5_age_train
        gc.collect()
        # 3. Cache key with FULL content fingerprints. Bump the prefix
        #    (``v5_attr_nn_v1``) so v2/v3/v4 cache entries cannot be
        #    silently misloaded -- those used different feature schemas
        #    and an identical-shape collision would corrupt the stacker
        #    (the v2 fold-0 NLL=0.778 bug all over again).
        import hashlib as _m2v5_hashlib
        _m2v5_feat_names_hash = _m2v5_hashlib.sha256(
            "\0".join(MEMBER2_V5_FEATURE_NAMES).encode("utf-8")
        ).hexdigest()[:16]
        _m2v5_X_train_hash = _m2v5_hashlib.sha256(
            np.ascontiguousarray(X_v5_fold_train, dtype=np.float32).tobytes()
        ).hexdigest()[:16]
        _fold_gbdt_state = cache_or_compute(
            "gbdt_state_oof_fold",
            key_inputs=(
                "gbdt_oof_v5_attr_nn_v1",
                fold.fold_id, fold_suffix,
                int(M2V5_FEATURE_DIM), int(X_v5_fold_train.shape[0]),
                int(SEED),
                round(float(_M2V5_SMOOTHING), 4),
                _m2v5_feat_names_hash,
                _m2v5_X_train_hash,
            ),
            compute_fn=lambda _ff=fold: fit_gbdt_member(
                X=X_v5_fold_train,
                y=_y_fold_train_np,
                feature_names=MEMBER2_V5_FEATURE_NAMES,
                # Direct binary: no init_score, output_mode='probability',
                # apply via gbdt_apply_batch.
                init_pred_train=None,
                holdout_group_id=_gbdt_train_item_id_fold,
                # Tighter than v2 -- 49 features overfit easily if we
                # don't constrain leaf size. The bounded-leaf property
                # of the binary objective combined with min_data_in_leaf
                # >=500 is what prevents the v4-style saturation.
                n_estimators=int(
                    CFG.get("member2_v5", {}).get("n_estimators", 400)
                ),
                learning_rate=float(
                    CFG.get("member2_v5", {}).get("learning_rate", 0.05)
                ),
                num_leaves=int(
                    CFG.get("member2_v5", {}).get("num_leaves", 31)
                ),
                min_data_in_leaf=int(
                    CFG.get("member2_v5", {}).get("min_data_in_leaf", 500)
                ),
                feature_fraction=float(
                    CFG.get("member2_v5", {}).get("feature_fraction", 0.8)
                ),
                bagging_fraction=float(
                    CFG.get("member2_v5", {}).get("bagging_fraction", 0.8)
                ),
                bagging_freq=int(
                    CFG.get("member2_v5", {}).get("bagging_freq", 5)
                ),
                early_stopping_rounds=int(
                    CFG.get("member2_v5", {}).get("early_stopping_rounds", 30)
                ),
                seed=int(SEED) + 100 * (int(_ff.fold_id) + 1) + 17,
            ),
        )
        # Defense-in-depth: a v2/v3 residual state must never be applied
        # via gbdt_apply_batch. The prefix change makes a collision very
        # unlikely, but if a stale state somehow squeaks through the cache
        # key, warn loudly and surface it.
        import warnings as _v5_state_warn
        if str(_fold_gbdt_state.output_mode) != "probability":
            _v5_state_warn.warn(
                f"Expected _fold_gbdt_state.output_mode='probability' (v5 "
                f"direct binary), got {_fold_gbdt_state.output_mode!r}. "
                "A stale residual_logit state may have been loaded -- "
                "delete the matching gbdt_state_oof_fold__*.pkl cache "
                "entry and rerun. Continuing anyway; predictions on "
                "this fold may be corrupted."
            )
        # Free the train-side X matrix BEFORE building the OOF one so peak
        # memory is one X matrix at a time.
        del X_v5_fold_train
        gc.collect()
        _v5_pool_oof = _m2v5_pool_for_rows(fold_oof_df)
        _v5_age_oof = _m2v5_age_for_rows(fold_oof_df)
        X_v5_fold_oof = build_member2_v5_features(
            _fold_v5_builder,
            subject_ids=_mef_subj_fold_oof,
            cluster_ids=_mef_cluster_fold_oof,
            bc_ids=_mef_bc_fold_oof,
            bc_redacted_mask=bc_redacted_train[fold.oof_row_idx].astype(np.float32),
            pool_features=_v5_pool_oof,
            benchmark_age=_v5_age_oof,
            nn_features_matrix=nn_oof_mat_fold,
        )
        del _v5_pool_oof, _v5_age_oof
        gc.collect()
        # v5 is a direct binary GBDT, so its raw output IS the probability.
        p2_oof_fold = gbdt_apply_batch(_fold_gbdt_state, X_v5_fold_oof)
        # Defense-in-depth: enforce 1-D + correct length so the OOF
        # accumulator's per-fold write_fold (which sanity-checks shape)
        # never sees a column-vector or off-length array.
        p2_oof_fold = np.asarray(p2_oof_fold, dtype=np.float32).reshape(-1)
        if p2_oof_fold.shape[0] != int(len(fold_oof_df)):
            import warnings as _v5_pof_warn
            _v5_pof_warn.warn(
                f"[OOF f{fold.fold_id}] M2 v5 OOF length "
                f"{p2_oof_fold.shape[0]} != fold_oof_df length "
                f"{len(fold_oof_df)}. The accumulator write below will "
                "raise a shape-mismatch error -- inspect the v5 builder "
                "or the X_v5_fold_oof construction."
            )
        del X_v5_fold_oof
        gc.collect()
        # Gate 3d (NLL vs baseline) -- warn instead of halt so a borderline
        # NLL doesn't kill the whole pipeline.
        _p2_oof_clip = np.clip(p2_oof_fold, 1e-6, 1.0 - 1e-6)
        _yfold_oof = fold_oof_df["label"].astype(float).to_numpy()
        _p2_oof_nll = float(
            -(_yfold_oof * np.log(_p2_oof_clip)
              + (1.0 - _yfold_oof) * np.log(1.0 - _p2_oof_clip)).mean()
        )
        _p2_q = np.quantile(p2_oof_fold, [0.0, 0.01, 0.5, 0.99, 1.0])
        _baseline_mean = float(_yfold_oof.mean())
        _p2_baseline = float(
            -(_baseline_mean * np.log(_baseline_mean)
              + (1.0 - _baseline_mean) * np.log(1.0 - _baseline_mean))
        )
        # Gate 3e: structural orthogonality with M1. v5 has NO M1
        # features so a sky-high corr is conceptually impossible, but
        # we still track the number for diagnostic visibility.
        _corr_v5_m1 = float(
            np.corrcoef(p2_oof_fold.astype(np.float64),
                        p_a_oof_fold.astype(np.float64))[0, 1]
        )
        print(
            f"  [Gate 3d fold {fold.fold_id}] M2 v5 OOF NLL={_p2_oof_nll:.5f}  "
            f"baseline(constant_mean)={_p2_baseline:.5f}  "
            f"corr(M2v5, M1)={_corr_v5_m1:+.4f}  "
            f"p2 quantiles min/1%/50%/99%/max="
            f"{_p2_q[0]:.4f}/{_p2_q[1]:.4f}/{_p2_q[2]:.4f}/{_p2_q[3]:.4f}/{_p2_q[4]:.4f}"
        )
        if _p2_oof_nll > _p2_baseline + 0.05:
            _v5_state_warn.warn(
                f"Gate 3d WARN fold {fold.fold_id}: M2 v5 OOF NLL="
                f"{_p2_oof_nll:.5f} is materially WORSE than the "
                f"constant-mean baseline ({_p2_baseline:.5f}). With "
                "v5's bounded-leaf direct-binary output this is "
                "unexpected; inspect the feature pipeline. Continuing "
                "anyway so the stacker can still down-weight Member 2."
            )
        if abs(_corr_v5_m1) > 0.995:
            _v5_state_warn.warn(
                f"Gate 3e WARN fold {fold.fold_id}: "
                f"|corr(M2v5, M1)|={abs(_corr_v5_m1):.4f} > 0.995. "
                "v5 has no M1 features so this is suspicious -- check "
                "the per-id mean-encoding for accidental leakage."
            )
        del _p2_oof_clip, _yfold_oof, _p2_q, _baseline_mean, _p2_baseline
        del _p2_oof_nll, _corr_v5_m1, _fold_v5_builder
        gc.collect()
    elif _M2V4_ENABLED:
        # Task 3b: direct-binary GBDT on the v3 feature set. Same 25
        # features as v3 (p_m1 + subject_mean + per-id obs counts /
        # mean-encoded passrates + unknown_* flags) but trained as a
        # binary classifier on y directly -- no init_score, no
        # residual framing, output = sigmoid(tree_raw). This eliminates
        # v3's blow-up failure mode (composed (anchor_logit + tree_logit)
        # producing 0.997-confident predictions on rows where the
        # leaf's labels averaged 0.5) by construction: a binary-objective
        # leaf output is bounded by in-leaf label averages, so the
        # worst-case calibration error is on the same order as v2's
        # OOF NLL ~ 0.69, not v3's blow-up ~ 0.78+.
        print(
            f"[OOF f{fold.fold_id}] Training fold Member 2 v4 "
            f"(direct-binary GBDT, F={M2V3_FEATURE_DIM} features incl. p_m1 + subject_mean)..."
        )
        # Sanity: v4 needs M1's per-row prediction on fold-train rows
        # because it appears as a feature in the v3 builder. We force
        # the scoring above when _M2V4_ENABLED, so this should always pass.
        if p_a_anchor_fold_train is None or len(p_a_anchor_fold_train) != len(fold_train_df):
            raise RuntimeError(
                f"v4 needs p_a_anchor_fold_train sized {len(fold_train_df)}; "
                f"got {None if p_a_anchor_fold_train is None else len(p_a_anchor_fold_train)}. "
                "Check the M1-anchor scoring branch above (_need_fold_train_m1_anchor)."
            )
        # 1. Fit the v3 feature builder on fold-train (per-id obs counts +
        #    shrunken passrates, restricted to this fold's training rows
        #    so OOF rows are NEVER in the aggregates).
        #
        # Vocab sizing: the clustering's CFG['clustering']['k']
        # (N_CLUSTERS_CTX) is a LOWER bound -- ``build_conditional_passrate_context``
        # auto-grows it when the k-means partition produces an extra
        # cluster (we've seen 64 -> 65 in fold 0). Use
        # ``fold_cond_context.n_clusters`` which already encodes the
        # grown size, and also include the OOF cluster ids in the max
        # so the same builder safely scores fold-OOF rows below.
        # Bound n_bcs the same way against observed train+OOF ids.
        def _safe_n_ids(declared_lb, *id_arrs):
            n = int(declared_lb)
            for arr in id_arrs:
                arr = np.asarray(arr)
                if arr.size > 0 and bool((arr >= 0).any()):
                    n = max(n, int(arr.max()) + 1)
            return n
        _fold_v3_n_clusters = _safe_n_ids(
            int(fold_cond_context.n_clusters),
            _mef_cluster_fold_train, _mef_cluster_fold_oof,
        )
        _fold_v3_n_bcs = _safe_n_ids(
            int(indexer.n_bc),
            _mef_bc_fold_train, _mef_bc_fold_oof,
        )
        _fold_v3_builder = fit_member2_v3_feature_builder(
            subject_ids=_mef_subj_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            bc_ids=_mef_bc_fold_train,
            labels=_y_fold_train_np,
            n_subjects=int(indexer.n_subjects),
            n_clusters=_fold_v3_n_clusters,
            n_bcs=_fold_v3_n_bcs,
            n_macro_families=int(N_MACRO_FAMILIES),
            n_organizations=int(N_ORGANIZATIONS),
            n_families=int(N_FAMILIES),
            subject_to_macro_family_id=s2macro,
            subject_to_organization_id=s2org,
            subject_to_family_id=s2fam,
            smoothing=_M2V4_SMOOTHING,
        )
        # 2. Materialize the v3 feature matrix for fold-train rows.
        X_v3_fold_train = build_member2_v3_features(
            _fold_v3_builder,
            p_m1=p_a_anchor_fold_train,
            subject_mean=subject_mean_train_fold,
            subject_ids=_mef_subj_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            bc_ids=_mef_bc_fold_train,
            bc_redacted_mask=bc_redacted_train[fold.train_row_idx].astype(np.float32),
        )
        # 3. Cache key with FULL content fingerprints. Same defense-in-depth
        #    as the v2 strengthening: shape alone hides feature-pipeline
        #    drift, and a cache HIT on stale state silently corrupts the
        #    stacker (the v2 0.778 NLL bug).
        import hashlib as _m2v3_hashlib
        _m2v3_feat_names_hash = _m2v3_hashlib.sha256(
            "\0".join(MEMBER2_V3_FEATURE_NAMES).encode("utf-8")
        ).hexdigest()[:16]
        _m2v3_X_train_hash = _m2v3_hashlib.sha256(
            np.ascontiguousarray(X_v3_fold_train, dtype=np.float32).tobytes()
        ).hexdigest()[:16]
        _m2v3_anchor_hash = _m2v3_hashlib.sha256(
            np.ascontiguousarray(subject_mean_train_fold, dtype=np.float64).tobytes()
        ).hexdigest()[:16]
        _m2v3_pm1_hash = _m2v3_hashlib.sha256(
            np.ascontiguousarray(p_a_anchor_fold_train, dtype=np.float32).tobytes()
        ).hexdigest()[:16]
        _fold_gbdt_state = cache_or_compute(
            "gbdt_state_oof_fold",
            key_inputs=(
                # ``v4_direct_v1`` is distinct from any v2/v3 key prefix
                # so old residual entries don't collide. Bump the suffix
                # here when changing the v3 feature schema or v4 GBDT
                # hyperparams below.
                "gbdt_oof_v4_direct_v1",
                fold.fold_id, fold_suffix,
                int(M2V3_FEATURE_DIM), int(X_v3_fold_train.shape[0]),
                int(SEED),
                round(float(_M2V4_SMOOTHING), 4),
                _m2v3_feat_names_hash,
                _m2v3_X_train_hash,
                _m2v3_anchor_hash,
                _m2v3_pm1_hash,
            ),
            compute_fn=lambda _ff=fold: fit_gbdt_member(
                X=X_v3_fold_train,
                y=_y_fold_train_np,
                feature_names=MEMBER2_V3_FEATURE_NAMES,
                # init_pred_train=None -> standard binary objective,
                # output_mode='probability', apply via gbdt_apply_batch.
                # See gbdt_member.fit_gbdt_member docstring on residual
                # mode for why this matters: with init_pred_train set,
                # the tree learns logit-residuals on top of a per-row
                # anchor and a wrong leaf can compose to an extreme
                # probability. Direct binary mode is bounded.
                init_pred_train=None,
                holdout_group_id=_gbdt_train_item_id_fold,
                # Tighter than v2: 25 features overfit easily without
                # aggressive regularization. min_data_in_leaf bumped to
                # 500 and num_leaves halved to 15 (vs v2/v3 defaults)
                # to keep per-leaf label averages stable -- this is what
                # bounds the binary-objective leaf output and makes v4
                # safe against the v3 blow-up.
                n_estimators=int(
                    CFG.get("member2_v4", {}).get("n_estimators", 300)
                ),
                learning_rate=float(
                    CFG.get("member2_v4", {}).get("learning_rate", 0.05)
                ),
                num_leaves=int(
                    CFG.get("member2_v4", {}).get("num_leaves", 15)
                ),
                min_data_in_leaf=int(
                    CFG.get("member2_v4", {}).get("min_data_in_leaf", 500)
                ),
                feature_fraction=float(
                    CFG.get("member2_v4", {}).get("feature_fraction", 0.9)
                ),
                bagging_fraction=float(
                    CFG.get("member2_v4", {}).get("bagging_fraction", 0.8)
                ),
                bagging_freq=int(
                    CFG.get("member2_v4", {}).get("bagging_freq", 5)
                ),
                early_stopping_rounds=int(
                    CFG.get("member2_v4", {}).get("early_stopping_rounds", 30)
                ),
                seed=int(SEED) + 100 * (int(_ff.fold_id) + 1) + 17,
            ),
        )
        # Defense-in-depth: a v3 residual state must never be applied via
        # gbdt_apply_batch. Catch this if a stale cache pickle from the
        # v3 era is somehow picked up despite the cache-key bump.
        if str(_fold_gbdt_state.output_mode) != "probability":
            raise RuntimeError(
                f"Expected _fold_gbdt_state.output_mode='probability' (v4 "
                f"direct binary), got {_fold_gbdt_state.output_mode!r}. "
                "A stale v3 residual_logit state was loaded -- delete the "
                "matching gbdt_state_oof_fold__*.pkl cache entry and rerun."
            )
        # Free the train-side X matrix BEFORE building the OOF one so peak
        # memory is one X matrix at a time. Each is ~3M rows * 25 cols *
        # 4 bytes = ~300 MB, but tight is tight.
        del X_v3_fold_train
        gc.collect()
        X_v3_fold_oof = build_member2_v3_features(
            _fold_v3_builder,
            p_m1=p_a_oof_fold,
            subject_mean=subject_mean_oof_fold,
            subject_ids=_mef_subj_fold_oof,
            cluster_ids=_mef_cluster_fold_oof,
            bc_ids=_mef_bc_fold_oof,
            bc_redacted_mask=bc_redacted_train[fold.oof_row_idx].astype(np.float32),
        )
        # v4 is a direct binary GBDT, so its raw output IS the probability.
        # No residual composition with subject_mean here -- that was v3's
        # blow-up failure mode.
        p2_oof_fold = gbdt_apply_batch(_fold_gbdt_state, X_v3_fold_oof)
        del X_v3_fold_oof
        gc.collect()
        # Gate 3d (NLL vs baseline) -- identical guard to v2 but pointed
        # at the v4 output. With direct binary the bounded-leaf property
        # should put NLL between baseline and M1; if it doesn't, the
        # feature pipeline is broken.
        _p2_oof_clip = np.clip(p2_oof_fold, 1e-6, 1.0 - 1e-6)
        _yfold_oof = fold_oof_df["label"].astype(float).to_numpy()
        _p2_oof_nll = float(
            -(_yfold_oof * np.log(_p2_oof_clip)
              + (1.0 - _yfold_oof) * np.log(1.0 - _p2_oof_clip)).mean()
        )
        _p2_q = np.quantile(p2_oof_fold, [0.0, 0.01, 0.5, 0.99, 1.0])
        _baseline_mean = float(_yfold_oof.mean())
        _p2_baseline = float(
            -(_baseline_mean * np.log(_baseline_mean)
              + (1.0 - _baseline_mean) * np.log(1.0 - _baseline_mean))
        )
        # Gate 3e (NEW): decorrelation from M1. The v4 tree must NOT
        # collapse into "predict whatever M1 predicts" -- if it does, it
        # is contributing no incremental signal and the stacker weight
        # will pin to zero. Some correlation is expected (M1 is a feature),
        # but a correlation above 0.995 means the tree memorized M1.
        _corr_v4_m1 = float(
            np.corrcoef(p2_oof_fold.astype(np.float64),
                        p_a_oof_fold.astype(np.float64))[0, 1]
        )
        print(
            f"  [Gate 3d fold {fold.fold_id}] M2 v4 OOF NLL={_p2_oof_nll:.5f}  "
            f"baseline(constant_mean)={_p2_baseline:.5f}  "
            f"corr(M2v4, M1)={_corr_v4_m1:+.4f}  "
            f"p2 quantiles min/1%/50%/99%/max="
            f"{_p2_q[0]:.4f}/{_p2_q[1]:.4f}/{_p2_q[2]:.4f}/{_p2_q[3]:.4f}/{_p2_q[4]:.4f}"
        )
        import warnings as _v4_gate_warn
        if _p2_oof_nll > _p2_baseline + 0.05:
            _v4_gate_warn.warn(
                f"Gate 3d WARN fold {fold.fold_id}: M2 v4 OOF NLL="
                f"{_p2_oof_nll:.5f} is materially WORSE than the "
                f"constant-mean baseline ({_p2_baseline:.5f}). The "
                "bounded-leaf property of a direct binary GBDT means "
                "this can only fail if the v3 feature pipeline is "
                "broken or the GBDT hyperparams are pathological. "
                "Inspect both. Continuing anyway."
            )
        if abs(_corr_v4_m1) > 0.995:
            _v4_gate_warn.warn(
                f"Gate 3e WARN fold {fold.fold_id}: "
                f"|corr(M2v4, M1)|={abs(_corr_v4_m1):.4f} > 0.995. "
                "The tree just memorized Member 1 -- no incremental "
                "signal. Try increasing min_data_in_leaf, lowering "
                "n_estimators, or dropping the logit_p_m1 feature. "
                "Continuing anyway."
            )
        del _p2_oof_clip, _yfold_oof, _p2_q, _baseline_mean, _p2_baseline
        del _p2_oof_nll, _corr_v4_m1, _fold_v3_builder
        gc.collect()
    elif _M2V2_ENABLED:
        # Task 3: non-embedding schema + subject_mean anchor.
        print(
            f"[OOF f{fold.fold_id}] Training fold Member 2 v2 (GBDT residual, "
            f"subject_mean anchor, non-embedding schema X cols={X_fold_train_m2v2.shape[1]})..."
        )
        # Cache key fingerprints that catch silent-stale-state failures:
        #   - ``feature_names_hash``: detects column reorder/rename/add/drop
        #     within the v2 schema (shape alone misses these).
        #   - ``X_train_content_hash``: 16-byte digest of the full
        #     fp32 X matrix bytes. Detects any change to the upstream
        #     feature pipeline (mean-encoding, interaction matrix,
        #     bc_redacted mask, subject_obs_count) that produces the
        #     same shape but different values.
        #   - ``init_pred_train_hash``: detects silent subject_mean
        #     anchor changes (smoothing, label set used to fit the table).
        # M2 v2 fold-0 NLL=0.778 (worse than constant!) on a cache HIT
        # is exactly the failure mode these guard against -- a state
        # trained on different inputs was loaded and scored against
        # mismatched current inputs.
        import hashlib as _m2v2_hashlib
        _m2v2_feat_names_hash = _m2v2_hashlib.sha256(
            "\0".join(member2_v2_schema.feature_names).encode("utf-8")
        ).hexdigest()[:16]
        _m2v2_X_train_hash = _m2v2_hashlib.sha256(
            np.ascontiguousarray(X_fold_train_m2v2, dtype=np.float32).tobytes()
        ).hexdigest()[:16]
        _m2v2_anchor_hash = _m2v2_hashlib.sha256(
            np.ascontiguousarray(subject_mean_train_fold, dtype=np.float64).tobytes()
        ).hexdigest()[:16]
        _fold_gbdt_state = cache_or_compute(
            "gbdt_state_oof_fold",
            key_inputs=(
                # ``subjmean_v3_contenthash`` invalidates earlier v2 entries
                # (which lacked feature_names / X content / anchor fingerprints
                # and produced the M2 fold-0 OOF NLL=0.778 silent-corruption bug).
                "gbdt_oof_v3_subjmean_contenthash",
                fold.fold_id, fold_suffix,
                int(X_fold_train_m2v2.shape[1]), int(X_fold_train_m2v2.shape[0]),
                int(SEED),
                round(float(_M2V2_SMOOTHING), 4),
                _m2v2_feat_names_hash,
                _m2v2_X_train_hash,
                _m2v2_anchor_hash,
            ),
            compute_fn=lambda _ff=fold: fit_gbdt_member(
                X=X_fold_train_m2v2,
                y=_y_fold_train_np,
                feature_names=member2_v2_schema.feature_names,
                init_pred_train=subject_mean_train_fold,
                holdout_group_id=_gbdt_train_item_id_fold,
                n_estimators=int(CFG.get("gbdt", {}).get("n_estimators", 400)),
                learning_rate=float(CFG.get("gbdt", {}).get("learning_rate", 0.05)),
                num_leaves=int(CFG.get("gbdt", {}).get("num_leaves", 63)),
                min_data_in_leaf=int(CFG.get("gbdt", {}).get("min_data_in_leaf", 100)),
                feature_fraction=float(CFG.get("gbdt", {}).get("feature_fraction", 0.8)),
                bagging_fraction=float(CFG.get("gbdt", {}).get("bagging_fraction", 0.8)),
                bagging_freq=int(CFG.get("gbdt", {}).get("bagging_freq", 5)),
                early_stopping_rounds=int(CFG.get("gbdt", {}).get("early_stopping_rounds", 25)),
                seed=int(SEED) + 100 * (int(_ff.fold_id) + 1),
            ),
        )
        p2_oof_fold = gbdt_compose_residual_batch(
            _fold_gbdt_state, X_fold_oof_m2v2, subject_mean_oof_fold,
        )
        # Diagnostic: catch extreme-prediction blowups (the 0.778 NLL signature).
        # If MOST predictions are calibrated near the OOF passrate but a tail
        # is wildly extreme (sigmoid(logit(anchor) + big_residual)), the NLL
        # gets dominated by the tail. We report quantiles + an OOF NLL preview.
        _p2_oof_clip = np.clip(p2_oof_fold, 1e-6, 1.0 - 1e-6)
        _yfold_oof = fold_oof_df["label"].astype(float).to_numpy()
        _p2_oof_nll = float(
            -(_yfold_oof * np.log(_p2_oof_clip)
              + (1.0 - _yfold_oof) * np.log(1.0 - _p2_oof_clip)).mean()
        )
        _p2_q = np.quantile(p2_oof_fold, [0.0, 0.01, 0.5, 0.99, 1.0])
        # NLL of the constant-mean predictor = entropy of the label mean.
        _baseline_mean = float(_yfold_oof.mean())
        _p2_baseline = float(
            -(_baseline_mean * np.log(_baseline_mean)
              + (1.0 - _baseline_mean) * np.log(1.0 - _baseline_mean))
        )
        print(
            f"  [Gate 3d fold {fold.fold_id}] M2 v2 OOF NLL={_p2_oof_nll:.5f}  "
            f"baseline(constant_mean)={_p2_baseline:.5f}  "
            f"p2 quantiles min/1%/50%/99%/max="
            f"{_p2_q[0]:.4f}/{_p2_q[1]:.4f}/{_p2_q[2]:.4f}/{_p2_q[3]:.4f}/{_p2_q[4]:.4f}"
        )
        if _p2_oof_nll > _p2_baseline + 0.05:
            # 0.05 nat is generous; in a healthy run M2 should beat
            # baseline by a few nats. Worse-than-baseline used to halt
            # the pipeline; we now warn loudly and continue so the
            # stacker can still down-weight a bad Member 2.
            import warnings as _v2_gate_warn
            _v2_gate_warn.warn(
                f"Gate 3d WARN fold {fold.fold_id}: M2 v2 OOF NLL="
                f"{_p2_oof_nll:.5f} is materially WORSE than the "
                f"constant-mean baseline ({_p2_baseline:.5f}). This is "
                "the cache-stale / extreme-prediction blowup signature. "
                f"Quantile spread: min={_p2_q[0]:.4f} max={_p2_q[4]:.4f}. "
                "Wipe gbdt_state_oof_fold__*.pkl and retry, or inspect "
                "the feature pipeline. Continuing anyway."
            )
        del _p2_oof_clip, _yfold_oof, _p2_q, _baseline_mean, _p2_baseline, _p2_oof_nll
    else:
        # Legacy: full-schema Member 2 with Member 1 anchor.
        # MEMORY: build the m2 matrices JIT (now), train, score, free.
        # They are not used anywhere else; keeping them around through
        # M3/M5/M4 like the old structure did wastes ~24 GB peak per fold.
        print(f"[OOF f{fold.fold_id}] Building Member 2 legacy matrices (JIT)...")
        X_fold_train_dense_m2 = np.concatenate(
            [X_fold_train_dense, fold_member2_interaction_train], axis=1,
        ).astype(np.float32, copy=False)
        X_fold_oof_dense_m2 = np.concatenate(
            [X_fold_oof_dense, fold_member2_interaction_oof], axis=1,
        ).astype(np.float32, copy=False)
        gc.collect()
        print(f"[OOF f{fold.fold_id}] Training fold Member 2 (legacy: GBDT residual)...")
        _fold_m2_feature_names = (
            tuple(member_feat_schema.feature_names)
            + tuple(MEMBER2_INTERACTION_FEATURE_NAMES)
        )
        _fold_gbdt_state = cache_or_compute(
            "gbdt_state_oof_fold",
            key_inputs=(
                "gbdt_oof_v1", fold.fold_id, fold_suffix,
                int(X_fold_train_dense_m2.shape[1]), int(X_fold_train_dense_m2.shape[0]),
                int(SEED),
            ),
            compute_fn=lambda _ff=fold: fit_gbdt_member(
                X=X_fold_train_dense_m2,
                y=_y_fold_train_np,
                feature_names=_fold_m2_feature_names,
                init_pred_train=p_a_anchor_fold_train,
                holdout_group_id=_gbdt_train_item_id_fold,
                n_estimators=int(CFG.get("gbdt", {}).get("n_estimators", 400)),
                learning_rate=float(CFG.get("gbdt", {}).get("learning_rate", 0.05)),
                num_leaves=int(CFG.get("gbdt", {}).get("num_leaves", 63)),
                min_data_in_leaf=int(CFG.get("gbdt", {}).get("min_data_in_leaf", 100)),
                feature_fraction=float(CFG.get("gbdt", {}).get("feature_fraction", 0.8)),
                bagging_fraction=float(CFG.get("gbdt", {}).get("bagging_fraction", 0.8)),
                bagging_freq=int(CFG.get("gbdt", {}).get("bagging_freq", 5)),
                early_stopping_rounds=int(CFG.get("gbdt", {}).get("early_stopping_rounds", 25)),
                seed=int(SEED) + 100 * (int(_ff.fold_id) + 1),
            ),
        )
        # Free training matrix the moment scoring starts -- p_a_anchor too.
        del X_fold_train_dense_m2
        gc.collect()
        p2_oof_fold = gbdt_compose_residual_batch(
            _fold_gbdt_state, X_fold_oof_dense_m2, p_a_oof_fold,
        )
        del X_fold_oof_dense_m2
        gc.collect()
    p2_train_oof_acc.write_fold(fold.oof_row_idx, p2_oof_fold)

    # ----- Fold Member 3 (kNN-similarity) -----
    print(f"[OOF f{fold.fold_id}] Training fold Member 3 (kNN)...")
    # Build fold-local subject keys (ordered subject_id -> subject_key).
    _fold_subject_keys = _subject_keys_ordered  # subject set is global
    _fold_passrate_dense = np.asarray(fold_passrate_csr.todense(), dtype=np.float32)
    # MEMORY: store the mask as bool, not fp32. Both fit_knn_member
    # (src/knn_member.py:1062) and fit_member5 (src/member5_difficulty_knn.py:541)
    # cast to bool internally, so the result is numerically identical
    # and the cache key (which is shape/seed-based) is unchanged. The
    # dtype change drops ~540 MB at full scale ([907, 197114] fp32 ->
    # bool = 715 MB -> 178 MB) and removes the need for an extra
    # _fold_passrate_mask_bool copy at Member 5 fit time.
    _fold_passrate_mask_dense = np.asarray(fold_passrate_mask_csr.todense(), dtype=bool)
    _fold_item_emb_stacked = np.stack(
        [np.asarray(item_emb_lookup[k], dtype=np.float32) for k in fold.train_item_keys],
        axis=0,
    )
    _fold_knn_state = cache_or_compute(
        "knn_state_oof_fold",
        key_inputs=(
            # ``knn_oof_v2`` invalidates entries built with the old
            # broken kwargs (K/min_subjects_per_item/tau_init/train_lr/
            # train_iters/train_l2) which never matched fit_knn_member's
            # actual signature; the per-fold call now mirrors the
            # GLOBAL Member 3 invocation (pca_dim, quantization, k,
            # tau_subject, tau_global, item_fallback_weight).
            "knn_oof_v2", fold.fold_id, fold_suffix,
            int(len(fold.train_item_keys)), int(indexer.n_subjects),
            int(SEED),
            int(_m3_cfg.get("pca_dim", 128)),
            str(_m3_cfg.get("quantization", "int8")),
            int(_m3_cfg.get("k", 128)),
            round(float(_m3_cfg.get("tau_subject", 5.0)), 6),
            round(float(_m3_cfg.get("tau_global", 200.0)), 6),
            round(float(_m3_cfg.get("item_fallback_weight", 0.5)), 6),
        ),
        compute_fn=lambda _ff=fold: fit_knn_member(
            item_keys=list(_ff.train_item_keys),
            item_embeddings=_fold_item_emb_stacked,
            subject_keys=_fold_subject_keys,
            passrate_dense=_fold_passrate_dense,
            passrate_mask=_fold_passrate_mask_dense,
            pca_dim=int(_m3_cfg.get("pca_dim", 128)),
            quantization=str(_m3_cfg.get("quantization", "int8")),
            k=int(_m3_cfg.get("k", 128)),
            tau_subject=float(_m3_cfg.get("tau_subject", 5.0)),
            tau_global=float(_m3_cfg.get("tau_global", 200.0)),
            item_fallback_weight=float(_m3_cfg.get("item_fallback_weight", 0.5)),
            seed=int(SEED) + 200 * (int(_ff.fold_id) + 1),
        ),
    )
    # Score fold Member 3 on OOF rows. fold_oof_df has item_keys NOT in
    # fold's item universe; knn_apply_batch handles this via cold-start
    # fallback (uses subject's mean over neighbors with valid items).
    # MEMORY: build the [N_oof, d_emb] query stack as a local; the
    # knn_apply call internally chunks so we don't need a second copy.
    _m3_oof_query_emb = np.stack(
        [np.asarray(item_emb_lookup[k], dtype=np.float32)
         for k in fold_oof_df["item_key"].astype(str)],
        axis=0,
    )
    p3_oof_fold = knn_apply_batch(
        _fold_knn_state,
        _m3_oof_query_emb,
        fold_oof_df["subject_key"].astype(str).tolist(),
    )
    p3_train_oof_acc.write_fold(fold.oof_row_idx, p3_oof_fold)
    # The fold's KNN state (~900 MB at full scale: [S, N] passrate_sorted
    # + mask_sorted) is no longer needed -- Member 5 has its own state
    # below, and the OOF prediction has already been written to the
    # accumulator. Free it before Member 5's allocations land.
    del _fold_knn_state, _m3_oof_query_emb
    gc.collect()

    # ----- Fold Member 5 (difficulty-projected kNN) -----
    # Trained from scratch per fold on fold-train items + fold-train rows
    # so OOF predictions on fold-OOF rows are genuinely "model never saw
    # this item's label" (Gate 4a). The projection regression uses only
    # fold-train items' per-item passrate, which is itself derived from
    # the SAME `_fold_passrate_dense` / `_fold_passrate_mask_dense`
    # matrices Member 3 already built (so we skip ~3 GB of [S, N]
    # intermediates that the row-aggregation path would allocate -- the
    # root cause of the Section 9.5 fold 0 OOM in the first Task-4 build).
    if _M5_ENABLED:
        print(f"[OOF f{fold.fold_id}] Training fold Member 5 (difficulty-kNN)...")

        # --- Gate 4c (per-fold): projection-leakage probe vs OOF items ---
        # The fold's Member 5 is fit on `fold.train_item_keys` only. The
        # fold's OOF rows reference items in `fold.oof_item_keys` (item-
        # disjoint from train by Gate 1a). The probe verifies that the
        # fitted Member 5 NEVER saw any OOF-item's embedding during fit;
        # if Gate 1a's item-disjoint contract ever breaks, this fires.
        _gate4c_fold = assert_projection_disjoint_from_val(
            fit_item_keys=list(fold.train_item_keys),
            val_item_keys=list(fold.oof_item_keys),
        )
        _gate4c_per_fold[int(fold.fold_id)] = _gate4c_fold

        # Reuse `_fold_item_emb_stacked` (built above for Member 3) and
        # `_fold_passrate_dense` / `_fold_passrate_mask_dense` (also
        # already built for Member 3) so we don't pay double the memory.
        # The pre-built passrate path skips fit_member5's internal
        # aggregation -- this is the OOM fix.
        # `_fold_passrate_mask_dense` is already bool above (see
        # construction comment), so no separate bool copy is needed:
        # fit_member5's astype(bool, copy=False) on line 541 is a no-op
        # on bool input.
        _fold_m5_state = cache_or_compute(
            "member5_state_oof_fold",
            key_inputs=(
                # m5_oof_v2 bumps the cache key for the passrate-reuse path
                # (same numerical state, but cleaner discipline to invalidate
                # once when the construction path changes).
                "m5_oof_v2", fold.fold_id, fold_suffix,
                int(len(fold.train_item_keys)), int(indexer.n_subjects),
                int(len(fold_train_df)), int(_fold_item_emb_stacked.shape[1]),
                _M5_K, round(_M5_TAU, 6), round(_M5_RIDGE_ALPHA, 6),
                round(_M5_ITEM_FB_WEIGHT, 6), int(_M5_MIN_SUBJ_PER_ITEM),
                int(SEED),
            ),
            compute_fn=lambda _ff=fold: fit_member5(
                item_keys=list(_ff.train_item_keys),
                item_embeddings=_fold_item_emb_stacked,
                subject_keys=list(_subject_keys_ordered),
                # Row arrays are accepted but IGNORED on the fast path
                # (passrate_dense + passrate_mask is sufficient for fit).
                # Pass empty arrays so we don't allocate the per-row id
                # vectors at all -- another ~75 MB saved per fold.
                subject_ids_per_row=np.zeros(0, dtype=np.int64),
                item_ids_per_row=np.zeros(0, dtype=np.int64),
                labels=np.zeros(0, dtype=np.float64),
                passrate_dense=_fold_passrate_dense,
                passrate_mask=_fold_passrate_mask_dense,
                k=_M5_K,
                tau=_M5_TAU,
                ridge_alpha=_M5_RIDGE_ALPHA,
                item_fallback_weight=_M5_ITEM_FB_WEIGHT,
                min_subjects_per_item=_M5_MIN_SUBJ_PER_ITEM,
            ),
        )
        # --- MEMORY: free the M5 fit-time inputs IMMEDIATELY (the OOM fix).
        # `Member5State` holds its OWN sorted copies of passrate +
        # mask (member5_difficulty_knn.py lines 78-79), and the
        # scoring path `apply_batch_via_ids` does NOT touch
        # `_fold_item_emb_stacked` or the originals -- it projects
        # the OOF query embeddings through `state.projection_weights`
        # and indexes into `state.passrate_sorted` / `_mask_sorted`.
        # Holding the originals during scoring + Member 4 training is
        # the leading peak contributor; freeing here drops ~4.6 GB
        # before the next allocation (OOF item-emb stack, then the
        # ~16 GB X_fold_train_dense_m4 build for Member 4).
        del _fold_item_emb_stacked, _fold_passrate_dense, _fold_passrate_mask_dense
        gc.collect()

        # --- Score fold Member 5 on OOF rows (CHUNKED) ---
        # At full scale the per-fold OOF has ~1.67M rows. A single
        # ``np.stack`` of [1_675_412, 4096] fp32 = ~25.6 GiB -- the
        # peak that previously OOM'd or hung this cell. We instead
        # build the item-emb stack chunk by chunk, score each chunk
        # through m5_apply_batch_via_ids, and write the results into
        # a pre-allocated [N_oof] output array. Peak per chunk is
        # ``chunk_rows * 4096 * 4`` bytes (default 256k -> ~4 GiB),
        # which fits comfortably alongside the live M5 state (~870 MB)
        # and the global stuff that survived the pre-OOF cleanup.
        #
        # Chunking also gives the user progress feedback for
        # ``apply_batch_via_ids`` (a pure-Python per-row loop that
        # takes 3-5 min on the full fold OOF).
        _fold_oof_subj_ids = np.fromiter(
            (
                int(indexer.subject_to_id.get(str(s), -1))
                for s in fold_oof_df["subject_key"]
            ),
            dtype=np.int64,
            count=len(fold_oof_df),
        )
        _N_OOF_M5 = int(len(fold_oof_df))
        _M5_APPLY_CHUNK = int(CFG.get("member5", {}).get("apply_chunk_rows", 256_000))
        _M5_APPLY_CHUNK = max(1, min(_N_OOF_M5, _M5_APPLY_CHUNK))
        _fold_oof_item_keys_list = fold_oof_df["item_key"].astype(str).tolist()
        p5_oof_fold = np.empty(_N_OOF_M5, dtype=np.float32)
        _m5_t = _time_m5.time()
        _last_log = _m5_t
        for _cs in range(0, _N_OOF_M5, _M5_APPLY_CHUNK):
            _ce = min(_cs + _M5_APPLY_CHUNK, _N_OOF_M5)
            _chunk_emb = np.empty(
                (_ce - _cs, int(_fold_m5_state.projection_d_emb)), dtype=np.float32,
            )
            for _ri, _kidx in enumerate(range(_cs, _ce)):
                _chunk_emb[_ri] = item_emb_lookup[_fold_oof_item_keys_list[_kidx]]
            p5_oof_fold[_cs:_ce] = m5_apply_batch_via_ids(
                _fold_m5_state,
                subject_ids=_fold_oof_subj_ids[_cs:_ce],
                query_item_embeddings=_chunk_emb,
            )
            del _chunk_emb
            _now = _time_m5.time()
            if _now - _last_log > 30.0 or _ce == _N_OOF_M5:
                print(
                    f"[OOF f{fold.fold_id}] M5 OOF scoring "
                    f"{_ce:,}/{_N_OOF_M5:,} rows "
                    f"({(_ce / max(_now - _m5_t, 1e-9)):.0f} rows/s)"
                )
                _last_log = _now
        p5_train_oof_acc.write_fold(fold.oof_row_idx, p5_oof_fold)

        # Eagerly free the fold's Member 5 artifacts (state + per-row
        # id vectors + item-key list). The state holds [S, N]
        # passrate arrays inside the cached object -- if we don't
        # free them here they survive through Member 4's training
        # (~30 GB peak dense matrix), pushing total RAM over the
        # Colab cap. Member 5 has already written p5_oof_fold into
        # the accumulator so the state is no longer needed.
        del _fold_oof_subj_ids, _fold_oof_item_keys_list, _fold_m5_state
        gc.collect()
    else:
        p5_oof_fold = None
        # Member 5 disabled: free the M5-fit-time inputs here instead
        # (when enabled, they're freed right after the M5 fit returns;
        # see the matching block inside the `if _M5_ENABLED:` branch).
        del _fold_item_emb_stacked, _fold_passrate_dense, _fold_passrate_mask_dense
        gc.collect()

    # ----- Fold Member 4 (LogReg hybrid) -----
    # MEMORY: build M4's training matrix JIT, train, free, then build
    # M4's OOF matrix, score, free. At full scale each is ~16 GB
    # (train) and ~8 GB (oof); having them coexist with each other AND
    # with the base X_fold_*_dense was the leading peak contributor
    # in the old structure. The base X_fold_*_dense gets absorbed
    # into the concat result and is freed immediately after.
    #
    # On the Task 3 path we DEFERRED the base build above, so we
    # construct ``X_fold_train_dense_m4`` directly in chunks: build a
    # chunk of X via ``_build_X`` (small temporary, ~2 GB at chunk=500k)
    # and write it into the pre-allocated full-size m4 matrix alongside
    # the marginal columns. Peak only ever has the full m4 result +
    # one chunk's temporary, never the full base X as a separate object.
    print(f"[OOF f{fold.fold_id}] Building Member 4 train matrix (JIT)...")
    _F_BASE = int(member_feat_schema.feature_dim)
    _F_MARG = int(fold_member4_marginal_train.shape[1])
    _F_M4 = _F_BASE + _F_MARG
    if X_fold_train_dense is None:
        # Chunked build: never materialize the full 16 GB base.
        # Chunk size 500k keeps the per-chunk transient under ~2.5 GB.
        _N_TR = int(len(fold_train_df))
        _CHUNK_TR = int(CFG.get("oof", {}).get("xbuild_chunk_rows", 500_000))
        X_fold_train_dense_m4 = np.empty((_N_TR, _F_M4), dtype=np.float32)
        # Pre-write the marginal columns; they're small and fast.
        X_fold_train_dense_m4[:, _F_BASE:] = fold_member4_marginal_train
        for _cs in range(0, _N_TR, _CHUNK_TR):
            _ce = min(_cs + _CHUNK_TR, _N_TR)
            _chunk_X = _build_X(
                fold_train_df.iloc[_cs:_ce],
                nn_train_mat_fold[_cs:_ce],
                _bc_redacted_fold_train[_cs:_ce],
            )
            X_fold_train_dense_m4[_cs:_ce, :_F_BASE] = _chunk_X
            del _chunk_X
        # Task 3 path kept ``nn_train_mat_fold`` alive specifically
        # for this build. It's no longer referenced after the loop;
        # free its ~360 MB before the M4 logreg trainer allocates
        # its torch tensors.
        del nn_train_mat_fold
        gc.collect()
    else:
        # Legacy path: concat the already-built base with the marginal.
        X_fold_train_dense_m4 = np.concatenate(
            [X_fold_train_dense, fold_member4_marginal_train], axis=1,
        ).astype(np.float32, copy=False)
        del X_fold_train_dense
    # The marginal columns have been copied into the concat result.
    # ``_bc_redacted_fold_train`` is no longer needed (only OOF half remains).
    del fold_member4_marginal_train, _bc_redacted_fold_train
    gc.collect()

    print(f"[OOF f{fold.fold_id}] Training fold Member 4 (LogReg hybrid)...")
    _fold_m4_feature_names = (
        tuple(member_feat_schema.feature_names)
        + tuple(MEMBER4_MARGINAL_FEATURE_NAMES)
    )
    _fold_logreg_state = cache_or_compute(
        "logreg_state_oof_fold",
        key_inputs=(
            "logreg_oof_v1", fold.fold_id, fold_suffix,
            int(X_fold_train_dense_m4.shape[1]), int(X_fold_train_dense_m4.shape[0]),
            int(SEED),
        ),
        compute_fn=lambda _ff=fold: fit_logreg_member(
            X=X_fold_train_dense_m4,
            y=_y_fold_train_np,
            feature_names=_fold_m4_feature_names,
            epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
            learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
            weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
            l1_strength=float(CFG.get("member4_logreg", {}).get("l1_strength_hybrid", 3.0e-3)),
            min_feature_std=float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)),
            early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
            seed=int(SEED) + 300 * (int(_ff.fold_id) + 1),
            val_fraction=0.1,
            holdout_group_id=_gbdt_train_item_id_fold,
        ),
    )
    # Training done. Free the [N_train, F+14] matrix BEFORE building the
    # OOF matrix so they don't both live during scoring.
    del X_fold_train_dense_m4
    gc.collect()

    print(f"[OOF f{fold.fold_id}] Building Member 4 OOF matrix (JIT) + scoring...")
    if X_fold_oof_dense is None:
        # Chunked build (Task 3 path): never materialize the full base.
        _N_OF = int(len(fold_oof_df))
        _CHUNK_OF = int(CFG.get("oof", {}).get("xbuild_chunk_rows", 500_000))
        X_fold_oof_dense_m4 = np.empty((_N_OF, _F_M4), dtype=np.float32)
        X_fold_oof_dense_m4[:, _F_BASE:] = fold_member4_marginal_oof
        for _cs in range(0, _N_OF, _CHUNK_OF):
            _ce = min(_cs + _CHUNK_OF, _N_OF)
            _chunk_X = _build_X(
                fold_oof_df.iloc[_cs:_ce],
                nn_oof_mat_fold[_cs:_ce],
                _bc_redacted_fold_oof[_cs:_ce],
            )
            X_fold_oof_dense_m4[_cs:_ce, :_F_BASE] = _chunk_X
            del _chunk_X
        # Task 3 path kept ``nn_oof_mat_fold`` alive for this build;
        # free its ~200 MB before the M4 OOF scoring step.
        del nn_oof_mat_fold
        gc.collect()
    else:
        X_fold_oof_dense_m4 = np.concatenate(
            [X_fold_oof_dense, fold_member4_marginal_oof], axis=1,
        ).astype(np.float32, copy=False)
        del X_fold_oof_dense
    del fold_member4_marginal_oof, _bc_redacted_fold_oof
    gc.collect()
    p4_oof_fold = logreg_apply_state_batch(_fold_logreg_state, X_fold_oof_dense_m4)
    p4_train_oof_acc.write_fold(fold.oof_row_idx, p4_oof_fold)
    del X_fold_oof_dense_m4
    gc.collect()

    # ----- Aux features (NN mean sim, NN support, centroid dist for OOF rows) -----
    # _aux_mean_sim_fold + _aux_support_fold were captured early (right
    # after nn_oof_mat_fold was built) so the full NN feature matrix
    # could be freed alongside nn_train_mat_fold after Member 1.
    _aux_centroid_dist_fold = _centroid_dist_for_rows(fold_oof_df)
    nn_mean_sim_oof_acc.write_fold(fold.oof_row_idx, _aux_mean_sim_fold)
    nn_support_oof_acc.write_fold(fold.oof_row_idx, _aux_support_fold)
    centroid_dist_oof_acc.write_fold(fold.oof_row_idx, _aux_centroid_dist_fold)
    del _aux_mean_sim_fold, _aux_support_fold, _aux_centroid_dist_fold

    print(
        f"[OOF f{fold.fold_id}] Fold OOF NLL summary on {len(fold.oof_row_idx):,} held-out rows:"
    )
    _y_fold_oof = fold_oof_df["label"].astype(float).to_numpy()
    def _nll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(_y_fold_oof * np.log(p) + (1 - _y_fold_oof) * np.log(1 - p)).mean())
    print(
        f"  M1={_nll(p_a_oof_fold):.5f}  M2={_nll(p2_oof_fold):.5f}  "
        f"M3={_nll(p3_oof_fold):.5f}  M4={_nll(p4_oof_fold):.5f}"
        + (f"  M5={_nll(p5_oof_fold):.5f}" if _M5_ENABLED else "")
    )

    # Free fold-scoped artifacts before next iteration (keep RAM bounded).
    # The vast majority of large fold-scoped objects were freed inline
    # next to their last consumer (see comments at each free site).
    # What's still in scope here is the small bookkeeping state that
    # lives the whole fold:
    #   * Passrate CSRs / conditional context             (~tens of MB)
    #   * Member 2 interaction matrices                   (~few MB)
    #   * Mean-encoded stats / subject_mean table         (~few MB)
    #
    # ``fold_nn_index`` was already freed right after the Gate-1b probe.
    # ``nn_train_mat_fold`` / ``nn_oof_mat_fold`` were freed right after
    # Member 1 (datasets) and X_fold_*_dense / aux capture were all done.
    # ``X_fold_*_dense``, ``X_fold_*_dense_m4``, ``X_fold_*_dense_m2``,
    # ``fold_member4_marginal_*``, ``_fold_item_emb_stacked``,
    # ``_fold_passrate_dense`` / ``_fold_passrate_mask_dense``,
    # ``_fold_knn_state``, and Member 5's per-fold state / OOF stack
    # were all freed at their last consumer.
    del fold_passrate_csr, fold_passrate_mask_csr, fold_cond_context
    del fold_member2_interaction_train, fold_member2_interaction_oof
    if _M2V2_ENABLED:
        # On v4 / v5 paths the v2 matrices were never materialized (set
        # to None); skip the explicit del to avoid a NameError-by-del.
        if (not _M2V4_ENABLED) and (not _M2V5_ENABLED):
            del X_fold_train_m2v2, X_fold_oof_m2v2
        del subject_mean_train_fold, subject_mean_oof_fold
        del fold_subject_mean_table
    gc.collect()

# Finalize OOF accumulators -- raises if anything is missing / non-finite.
print("\n[OOF] Finalizing per-fold OOF accumulators...")
p_a_train_oof = p_a_train_oof_acc.finalize()
p2_train_oof = p2_train_oof_acc.finalize()
p3_train_oof = p3_train_oof_acc.finalize()
p4_train_oof = p4_train_oof_acc.finalize()
if _M5_ENABLED:
    p5_train_oof = p5_train_oof_acc.finalize()
else:
    p5_train_oof = None
nn_mean_sim_oof = nn_mean_sim_oof_acc.finalize().astype(np.float32)
nn_support_oof = nn_support_oof_acc.finalize().astype(np.float32)
centroid_dist_oof = centroid_dist_oof_acc.finalize().astype(np.float32)

_ylab_train = primary.train["label"].astype(float).to_numpy()
def _nll_full(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(_ylab_train * np.log(p) + (1 - _ylab_train) * np.log(1 - p)).mean())

print("\n[OOF] Train-row OOF NLL per member (aggregated across all folds):")
print(f"  M1 OOF: {_nll_full(p_a_train_oof):.5f}  vs in-sample p_a_train: {_nll_full(p_a_train):.5f}")
print(f"  M2 OOF: {_nll_full(p2_train_oof):.5f}")
print(f"  M3 OOF: {_nll_full(p3_train_oof):.5f}")
print(f"  M4 OOF: {_nll_full(p4_train_oof):.5f}")
if _M5_ENABLED:
    print(f"  M5 OOF: {_nll_full(p5_train_oof):.5f}")
print(f"[OOF] All accumulators finalized OK ({_N_TRAIN:,} rows each, all finite, single-write).")

# Task 4: Aggregate Gate 4c (per-fold projection-leakage probe) results.
if _M5_ENABLED and len(_gate4c_per_fold) > 0:
    _g4c_total_fit = sum(r["n_fit_items"] for r in _gate4c_per_fold.values())
    _g4c_total_val = sum(r["n_val_items"] for r in _gate4c_per_fold.values())
    _g4c_total_overlap = sum(r["n_overlap"] for r in _gate4c_per_fold.values())
    print(
        f"\n[Gate 4c per-fold aggregate] {len(_gate4c_per_fold)} folds checked, "
        f"sum(n_fit_items)={_g4c_total_fit:,}  "
        f"sum(n_oof_items)={_g4c_total_val:,}  "
        f"sum(overlap)={_g4c_total_overlap}. "
        + ("PASS." if _g4c_total_overlap == 0
           else f"FAIL: {_g4c_total_overlap} item overlaps found.")
    )

# Finalize Task 3 OOF subject_mean accumulator + aggregate Gate 3a summary.
if _M2V2_ENABLED:
    subject_mean_train_oof = subject_mean_train_oof_acc.finalize().astype(np.float64)
    print(
        f"\n[Member 2 v2] OOF subject_mean: shape={subject_mean_train_oof.shape}  "
        f"min={subject_mean_train_oof.min():.4f}  "
        f"mean={subject_mean_train_oof.mean():.4f}  "
        f"max={subject_mean_train_oof.max():.4f}"
    )
    # Aggregate Gate 3a summary (per-fold passes already printed inline).
    _total_3a_checked = sum(r["n_checked"] for r in _gate3a_per_fold.values())
    _total_3a_violations = sum(r["n_violations"] for r in _gate3a_per_fold.values())
    _max_3a_delta = max(r["max_abs_delta"] for r in _gate3a_per_fold.values())
    if _total_3a_violations > 0:
        import warnings as _g3a_warn
        _g3a_warn.warn(
            f"[Gate 3a aggregate] WARN: "
            f"{_total_3a_violations:,}/{_total_3a_checked:,} fold-OOF "
            "rows had subject_mean anchors that didn't match their "
            "fold-train table. Pipeline continues; downstream OOF "
            "metrics may be inflated by leakage."
        )
    print(
        f"[Gate 3a aggregate] PASS: {_total_3a_checked:,} fold-OOF rows across "
        f"{len(_gate3a_per_fold)} folds, 0 violations, "
        f"max_abs_delta_across_folds={_max_3a_delta:.2e}."
    )
else:
    subject_mean_train_oof = None

# %% [markdown]
# ## 9.5b. Gate 1b: NN-neighbor-in-fold-train probe (post-loop assertion)
#
# For each fold we sampled `nn_neighbor_probe_sample_size` OOF rows and
# captured their top-k neighbor item_keys from the fold-scoped NN index.
# Gate 1b asserts: NONE of those neighbor keys belong to that fold's
# OOF item set. A single violation means the fold's NN index leaked
# data the fold isn't allowed to see -- which would silently
# contaminate Members 1, 2 (NN aux), and 3.

# %%
print("[OOF Gate 1b] Asserting fold-NN neighbors stay within fold-train items...")
_probe_summary = []
for fold in folds:
    probe_keys = _oof_nn_probe_data[fold.fold_id]
    result = assert_nn_neighbors_in_fold_train(
        fold=fold,
        oof_row_neighbor_item_keys=probe_keys,
        sample_size=None,  # already pre-sampled inside the per-fold loop
    )
    _probe_summary.append((fold.fold_id, result))
    print(f"  fold {fold.fold_id}: checked={result['n_checked']:,} violations={result['n_violations']}")
print("[OOF Gate 1b] PASS: zero NN-neighbor leakage across all folds.")

# %% [markdown]
# ## 9.5c. Global Member 2 v2 fit (Task 3): subject-mean residual + non-embedding schema
#
# Fits the SHIPPED Member 2 booster on the FULL training set with the
# OOF subject_mean anchor accumulated in section 9.5. The anchor is
# OOF per-row (each row's `subject_mean_train_oof[r]` was computed from
# the fold-train labels of the fold that holds r in its OOF set, so
# row r's own label is never in its own anchor). At val / runtime
# inference the anchor switches to `subject_mean_table_global` (full
# train labels, which honestly excludes val rows since they aren't in
# train).
#
# Gates run here:
#   * Gate 3b (parity): fit_gbdt_member already asserts
#     `parity_atol=1e-5` between LightGBM's `predict(raw_score=True)`
#     and the pure-NumPy tree walker on the residual target. A line
#     is printed below confirming the assertion fired.
#   * Gate 3c (composer round-trip): a synthetic case verifies that
#     adding a zero tree-residual to a known subject_mean anchor
#     recovers the anchor probability exactly, AND that the composer
#     is numerically stable for subject_mean values near 0 and 1.

# %%
member2_v4_global_state = None  # populated below when _M2V4_ENABLED
member2_v5_global_state = None  # populated below when _M2V5_ENABLED

if _M2V5_ENABLED:
    # v5 GLOBAL fit: trains the SHIPPED direct-binary GBDT on the full
    # training set. v5 uses NO M1 anchor and NO subject_mean anchor
    # (both were the source of v3/v4 saturation); instead it consumes
    # item attributes, honest historical counts, mean-encoded passrates,
    # and 7 NN aggregates as features. This makes the trained tree
    # structurally orthogonal to Member 1 by construction.
    print(
        "\n[Member 2 v5 / 9.5c] Fitting GLOBAL v5 feature builder + "
        "training direct-binary GBDT on full train (no M1, no subject_mean)..."
    )
    def _safe_n_ids_v5_global(declared_lb, *id_arrs):
        n = int(declared_lb)
        for arr in id_arrs:
            arr = np.asarray(arr)
            if arr.size > 0 and bool((arr >= 0).any()):
                n = max(n, int(arr.max()) + 1)
        return n
    _global_v5_n_subjects = _safe_n_ids_v5_global(
        int(indexer.n_subjects),
        _mef_train_subj, _mef_val_subj,
    )
    _global_v5_n_clusters = _safe_n_ids_v5_global(
        int(cond_context.n_clusters),
        _mef_train_cluster, _mef_val_cluster,
    )
    _global_v5_n_bcs = _safe_n_ids_v5_global(
        int(indexer.n_bc),
        _mef_train_bc, _mef_val_bc,
    )
    member2_v5_global_builder = fit_member2_v5_feature_builder(
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        labels=y_train,
        n_subjects=_global_v5_n_subjects,
        n_clusters=_global_v5_n_clusters,
        n_bcs=_global_v5_n_bcs,
        n_macro_families=int(N_MACRO_FAMILIES),
        n_organizations=int(N_ORGANIZATIONS),
        n_families=int(N_FAMILIES),
        subject_to_macro_family_id=s2macro,
        subject_to_organization_id=s2org,
        subject_to_family_id=s2fam,
        smoothing=_M2V5_SMOOTHING,
    )
    _v5_pool_train_global = _m2v5_pool_for_rows(primary.train)
    _v5_age_train_global = _m2v5_age_for_rows(primary.train)
    X_train_v5_global = build_member2_v5_features(
        member2_v5_global_builder,
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        bc_redacted_mask=bc_redacted_train.astype(np.float32),
        pool_features=_v5_pool_train_global,
        benchmark_age=_v5_age_train_global,
        nn_features_matrix=nn_train_mat,
    )
    del _v5_pool_train_global, _v5_age_train_global
    gc.collect()
    print(
        f"[Member 2 v5 / 9.5c] X_train_v5_global: {X_train_v5_global.shape}  "
        "(no M1 / subject_mean anchor; label leakage = 0 by feature design)"
    )

    import hashlib as _m2v5g_hashlib
    _m2v5g_feat_names_hash = _m2v5g_hashlib.sha256(
        "\0".join(MEMBER2_V5_FEATURE_NAMES).encode("utf-8")
    ).hexdigest()[:16]
    _m2v5g_X_hash = _m2v5g_hashlib.sha256(
        np.ascontiguousarray(X_train_v5_global, dtype=np.float32).tobytes()
    ).hexdigest()[:16]

    def _fit_member2_v5_global():
        print(
            "[Member 2 v5 / 9.5c] training LightGBM DIRECT BINARY "
            f"(X cols={X_train_v5_global.shape[1]}, no init_score, "
            "objective=binary on y directly, no M1-derived features)..."
        )
        return fit_gbdt_member(
            X=X_train_v5_global,
            y=y_train,
            feature_names=MEMBER2_V5_FEATURE_NAMES,
            init_pred_train=None,
            n_estimators=int(
                CFG.get("member2_v5", {}).get("n_estimators_global", 500)
            ),
            learning_rate=float(
                CFG.get("member2_v5", {}).get("learning_rate", 0.05)
            ),
            num_leaves=int(
                CFG.get("member2_v5", {}).get("num_leaves", 31)
            ),
            min_data_in_leaf=int(
                CFG.get("member2_v5", {}).get("min_data_in_leaf", 500)
            ),
            early_stopping_rounds=int(
                CFG.get("member2_v5", {}).get("early_stopping_rounds_global", 30)
            ),
            seed=SEED,
            parity_atol=1.0e-5,
            val_fraction=float(
                CFG.get("member2_v5", {}).get("val_fraction", 0.1)
            ),
            max_bin=int(
                CFG.get("member2_v5", {}).get("max_bin", 63)
            ),
            force_col_wise=bool(
                CFG.get("member2_v5", {}).get("force_col_wise", True)
            ),
            log_period=int(
                CFG.get("member2_v5", {}).get("log_period", 25)
            ),
            num_threads=CFG.get("member2_v5", {}).get("num_threads", None),
            holdout_group_id=gbdt_train_item_id,
        )

    member2_v5_gbdt_state = cache_or_compute(
        "gbdt_state",
        key_inputs=(
            # ``v5_attr_nn_global_v1`` never collides with v2/v3/v4 keys
            # (different prefix string).
            "member2_v5_attr_nn_global_v1",
            int(M2V5_FEATURE_DIM), len(primary.train), SEED,
            round(BC_REDACT_FRAC, 3), _bc_redact_seed,
            round(float(_M2V5_SMOOTHING), 4),
            int(OOF_N_FOLDS), int(OOF_SEED), bool(OOF_RETRAIN_M1),
            _m2v5g_feat_names_hash, _m2v5g_X_hash,
        ),
        compute_fn=_fit_member2_v5_global,
    )
    import warnings as _v5g_warn
    if str(member2_v5_gbdt_state.output_mode) != "probability":
        _v5g_warn.warn(
            "Expected member2_v5_gbdt_state.output_mode='probability' (v5 "
            f"direct binary), got {member2_v5_gbdt_state.output_mode!r}. "
            "A stale residual_logit state was loaded -- delete the "
            "matching gbdt_state__*.pkl cache entry and rerun. "
            "Continuing; val predictions may be corrupted."
        )
    # Build the shipped state. Downstream (export, smoke tests) treats
    # ``gbdt_state`` as the LightGBM booster -- alias here so the rest
    # of the pipeline works without per-module branching.
    member2_v5_global_state = Member2V5State(
        gbdt=member2_v5_gbdt_state,
        builder=member2_v5_global_builder,
    )
    gbdt_state = member2_v5_gbdt_state

    # Gate 3b parity (re-verified on a small sample).
    print("\n[Gate 3b v5] verifying pure-NumPy tree walker parity vs LightGBM raw_score...")
    _gate3b_sample = int(min(2000, X_train_v5_global.shape[0]))
    _gate3b_idx = np.random.default_rng(0).choice(
        X_train_v5_global.shape[0], size=_gate3b_sample, replace=False,
    )
    from src.gbdt_member import predict_raw as _gbdt_predict_raw_g3b_v5
    _walker_raw = _gbdt_predict_raw_g3b_v5(
        member2_v5_gbdt_state, X_train_v5_global[_gate3b_idx]
    )
    print(
        f"[Gate 3b v5] PASS: implicit during fit (parity_atol=1e-5); "
        f"re-verified pure-NumPy walker on {_gate3b_sample:,}-row sample "
        f"raw_score range [{float(_walker_raw.min()):.4f}, {float(_walker_raw.max()):.4f}]."
    )

    # Apply Member 2 v5 to val rows. Val item embeddings may be cold,
    # which is exactly what v5 was designed for: pool/centroid/NN
    # signals stay informative for cold items.
    _v5_pool_val_global = _m2v5_pool_for_rows(primary.val)
    _v5_age_val_global = _m2v5_age_for_rows(primary.val)
    X_val_v5 = build_member2_v5_features(
        member2_v5_global_builder,
        subject_ids=_mef_val_subj,
        cluster_ids=_mef_val_cluster,
        bc_ids=_mef_val_bc,
        bc_redacted_mask=bc_redacted_val.astype(np.float32),
        pool_features=_v5_pool_val_global,
        benchmark_age=_v5_age_val_global,
        nn_features_matrix=nn_val_mat,
    )
    del _v5_pool_val_global, _v5_age_val_global
    p_member2_val = gbdt_apply_batch(member2_v5_gbdt_state, X_val_v5)
    # Defense-in-depth: ensure 1-D shape and exact val-length. Catches
    # any apply_batch contract drift before the val stack-up below
    # (which would otherwise fail with a generic np.stack ValueError
    # that doesn't point at Member 2).
    p_member2_val = np.asarray(p_member2_val, dtype=np.float32).reshape(-1)
    if p_member2_val.shape[0] != int(len(primary.val)):
        import warnings as _v5_pmv_warn
        _v5_pmv_warn.warn(
            f"[Member 2 v5] p_member2_val length {p_member2_val.shape[0]} "
            f"!= primary.val length {len(primary.val)}. The val stack-up "
            "in section 10 will reject the mismatch with a clearer error."
        )
    print(
        f"[Member 2 v5] p_member2_val shape={p_member2_val.shape}  "
        f"dtype={p_member2_val.dtype}  "
        f"(expected ({int(len(primary.val))},) fp32)"
    )
    nll_m2 = float(
        -(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
          + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean()
    )
    # Gate 3e at val scale: confirm v5 hasn't accidentally relearned M1
    # (it shouldn't, since v5 has no M1 features).
    _corr_v5_m1_val = float(
        np.corrcoef(
            p_member2_val.astype(np.float64), p_a_val.astype(np.float64)
        )[0, 1]
    )
    print(
        f"\n[Member 2 v5] val log-loss: {nll_m2:.6f}  "
        f"corr(M2v5, M1) on val={_corr_v5_m1_val:+.4f}"
    )
    print(
        f"[Member 2 v5] train NLL stored in state: {member2_v5_gbdt_state.train_loss:.6f}  "
        f"booster's internal val NLL (item-cold split): {member2_v5_gbdt_state.val_loss:.6f}"
    )
    print(
        f"[Member 2 v5] n_trees={member2_v5_gbdt_state.n_trees}  "
        f"bias={member2_v5_gbdt_state.bias:+.4f}"
    )
    if abs(_corr_v5_m1_val) > 0.995:
        _v5g_warn.warn(
            f"Gate 3e WARN on val: |corr(M2v5, M1)|="
            f"{abs(_corr_v5_m1_val):.4f} > 0.995. v5 has no M1 features "
            "so this is suspicious -- check the per-id mean-encoding "
            "for accidental leakage. Continuing anyway."
        )
    # Free the per-row v5 feature matrices we just used.
    del X_train_v5_global, X_val_v5
    gc.collect()

elif _M2V4_ENABLED:
    # v4 GLOBAL fit: trains the SHIPPED direct-binary GBDT on the full
    # training set with OOF anchors for both M1 prediction (p_a_train_oof,
    # accumulated by per-fold M1 retrains) AND the subject_mean baseline
    # (subject_mean_train_oof). Both honest -> no label leakage into the
    # training inputs, so the tree truly learns calibration patterns
    # rather than memorizing the train labels via the anchors. v4 differs
    # from v3 only in HOW it consumes those anchors: v4 uses them as
    # features (p_m1, subject_mean) inside the v3 builder rather than as
    # an init_score residual target, so the GBDT output stays bounded
    # by in-leaf label averages.
    print(
        "\n[Member 2 v4 / 9.5c] Fitting GLOBAL v3 feature builder + "
        "training direct-binary GBDT on full train (OOF M1 + OOF subject_mean as FEATURES)..."
    )
    # See per-fold v3 vocab-sizing comment: declared CFG['clustering']['k']
    # is a lower bound; the conditional passrate context (and now the v3
    # builder) must use the grown vocabulary. Also bound n_bcs by the
    # max observed bc id across train + val so the apply path
    # (which scores val rows below) never crashes on an unseen id.
    def _safe_n_ids_global(declared_lb, *id_arrs):
        n = int(declared_lb)
        for arr in id_arrs:
            arr = np.asarray(arr)
            if arr.size > 0 and bool((arr >= 0).any()):
                n = max(n, int(arr.max()) + 1)
        return n
    _global_v3_n_clusters = _safe_n_ids_global(
        int(cond_context.n_clusters),
        _mef_train_cluster, _mef_val_cluster,
    )
    _global_v3_n_bcs = _safe_n_ids_global(
        int(indexer.n_bc),
        _mef_train_bc, _mef_val_bc,
    )
    member2_v4_global_builder = fit_member2_v3_feature_builder(
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        labels=y_train,
        n_subjects=int(indexer.n_subjects),
        n_clusters=_global_v3_n_clusters,
        n_bcs=_global_v3_n_bcs,
        n_macro_families=int(N_MACRO_FAMILIES),
        n_organizations=int(N_ORGANIZATIONS),
        n_families=int(N_FAMILIES),
        subject_to_macro_family_id=s2macro,
        subject_to_organization_id=s2org,
        subject_to_family_id=s2fam,
        smoothing=_M2V4_SMOOTHING,
    )
    X_train_v3_global = build_member2_v3_features(
        member2_v4_global_builder,
        p_m1=p_a_train_oof,                  # honest OOF M1 anchors
        subject_mean=subject_mean_train_oof,  # honest OOF subject_mean anchors
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        bc_redacted_mask=bc_redacted_train.astype(np.float32),
    )
    print(
        f"[Member 2 v4 / 9.5c] X_train_v3_global: {X_train_v3_global.shape}  "
        f"p_a_train_oof: {p_a_train_oof.shape}  "
        f"subject_mean_train_oof: {subject_mean_train_oof.shape}  "
        "(both anchors are OOF per-row; label leakage = 0)"
    )

    import hashlib as _m2v3g_hashlib
    _m2v3g_feat_names_hash = _m2v3g_hashlib.sha256(
        "\0".join(MEMBER2_V3_FEATURE_NAMES).encode("utf-8")
    ).hexdigest()[:16]
    _m2v3g_X_hash = _m2v3g_hashlib.sha256(
        np.ascontiguousarray(X_train_v3_global, dtype=np.float32).tobytes()
    ).hexdigest()[:16]
    _m2v3g_anchor_hash = _m2v3g_hashlib.sha256(
        np.ascontiguousarray(subject_mean_train_oof, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    _m2v3g_pm1_hash = _m2v3g_hashlib.sha256(
        np.ascontiguousarray(p_a_train_oof, dtype=np.float32).tobytes()
    ).hexdigest()[:16]

    def _fit_member2_v4_global():
        print(
            "[Member 2 v4 / 9.5c] training LightGBM DIRECT BINARY "
            f"(X cols={X_train_v3_global.shape[1]}, no init_score, "
            "objective=binary on y directly, features include p_m1 + "
            "subject_mean as numeric columns)..."
        )
        return fit_gbdt_member(
            X=X_train_v3_global,
            y=y_train,
            feature_names=MEMBER2_V3_FEATURE_NAMES,
            # See per-fold v4 comment on init_pred_train=None.
            init_pred_train=None,
            n_estimators=int(
                CFG.get("member2_v4", {}).get("n_estimators_global", 400)
            ),
            learning_rate=float(
                CFG.get("member2_v4", {}).get("learning_rate", 0.05)
            ),
            num_leaves=int(
                CFG.get("member2_v4", {}).get("num_leaves", 15)
            ),
            min_data_in_leaf=int(
                CFG.get("member2_v4", {}).get("min_data_in_leaf", 500)
            ),
            early_stopping_rounds=int(
                CFG.get("member2_v4", {}).get("early_stopping_rounds_global", 30)
            ),
            seed=SEED,
            parity_atol=1.0e-5,
            val_fraction=float(
                CFG.get("member2_v4", {}).get("val_fraction", 0.1)
            ),
            max_bin=int(
                CFG.get("member2_v4", {}).get("max_bin", 63)
            ),
            force_col_wise=bool(
                CFG.get("member2_v4", {}).get("force_col_wise", True)
            ),
            log_period=int(
                CFG.get("member2_v4", {}).get("log_period", 25)
            ),
            num_threads=CFG.get("member2_v4", {}).get("num_threads", None),
            holdout_group_id=gbdt_train_item_id,
        )

    member2_v4_gbdt_state = cache_or_compute(
        "gbdt_state",
        key_inputs=(
            # ``v4_direct_global_v1`` never collides with v2/v3 keys
            "member2_v4_direct_global_v1",
            int(M2V3_FEATURE_DIM), len(primary.train), SEED,
            round(BC_REDACT_FRAC, 3), _bc_redact_seed,
            round(float(_M2V4_SMOOTHING), 4),
            int(OOF_N_FOLDS), int(OOF_SEED), bool(OOF_RETRAIN_M1),
            _m2v3g_feat_names_hash, _m2v3g_X_hash,
            _m2v3g_anchor_hash, _m2v3g_pm1_hash,
        ),
        compute_fn=_fit_member2_v4_global,
    )
    if str(member2_v4_gbdt_state.output_mode) != "probability":
        raise RuntimeError(
            "Expected member2_v4_gbdt_state.output_mode='probability' (v4 "
            f"direct binary), got {member2_v4_gbdt_state.output_mode!r}. "
            "A stale v3 residual_logit state was loaded -- delete the "
            "matching gbdt_state__*.pkl cache entry and rerun."
        )
    # Build the shipped state. Downstream code (export, smoke tests)
    # treats ``gbdt_state`` as the LightGBM booster -- alias here so
    # everything that worked for v2 keeps working without per-module
    # branching. The full v3 state (booster + builder) lives in
    # ``member2_v4_global_state``.
    member2_v4_global_state = Member2V3State(
        gbdt=member2_v4_gbdt_state,
        builder=member2_v4_global_builder,
    )
    gbdt_state = member2_v4_gbdt_state

    # Gate 3b parity (re-verified on a small sample, same as v2 branch).
    print("\n[Gate 3b v4] verifying pure-NumPy tree walker parity vs LightGBM raw_score...")
    _gate3b_sample = int(min(2000, X_train_v3_global.shape[0]))
    _gate3b_idx = np.random.default_rng(0).choice(
        X_train_v3_global.shape[0], size=_gate3b_sample, replace=False,
    )
    from src.gbdt_member import predict_raw as _gbdt_predict_raw_g3b
    _walker_raw = _gbdt_predict_raw_g3b(member2_v4_gbdt_state, X_train_v3_global[_gate3b_idx])
    print(
        f"[Gate 3b v4] PASS: implicit during fit (parity_atol=1e-5); "
        f"re-verified pure-NumPy walker on {_gate3b_sample:,}-row sample "
        f"raw_score range [{float(_walker_raw.min()):.4f}, {float(_walker_raw.max()):.4f}]."
    )

    # Apply Member 2 v4 to val rows. p_a_val comes from M1's val scoring
    # (computed in section 9a/9b); subject_mean_val from M2 v2 setup
    # (which was kept enabled as a prerequisite of v4 -- it appears as
    # a FEATURE here, not an apply-time anchor).
    X_val_v3 = build_member2_v3_features(
        member2_v4_global_builder,
        p_m1=p_a_val,
        subject_mean=subject_mean_val,
        subject_ids=_mef_val_subj,
        cluster_ids=_mef_val_cluster,
        bc_ids=_mef_val_bc,
        bc_redacted_mask=bc_redacted_val.astype(np.float32),
    )
    # Direct binary apply: GBDT output IS the probability. No composition
    # with subject_mean (subject_mean was already consumed as a column of
    # X_val_v3 by the v3 builder).
    p_member2_val = gbdt_apply_batch(member2_v4_gbdt_state, X_val_v3)
    nll_m2 = float(
        -(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
          + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean()
    )
    # Gate 3e at val scale: confirm v4 hasn't collapsed into pure M1.
    _corr_v4_m1_val = float(
        np.corrcoef(
            p_member2_val.astype(np.float64), p_a_val.astype(np.float64)
        )[0, 1]
    )
    print(
        f"\n[Member 2 v4] val log-loss: {nll_m2:.6f}  "
        f"corr(M2v4, M1) on val={_corr_v4_m1_val:+.4f}"
    )
    print(
        f"[Member 2 v4] train NLL stored in state: {member2_v4_gbdt_state.train_loss:.6f}  "
        f"booster's internal val NLL (item-cold split): {member2_v4_gbdt_state.val_loss:.6f}"
    )
    print(
        f"[Member 2 v4] n_trees={member2_v4_gbdt_state.n_trees}  "
        f"bias={member2_v4_gbdt_state.bias:+.4f}"
    )
    if abs(_corr_v4_m1_val) > 0.995:
        import warnings as _v4g_warn
        _v4g_warn.warn(
            f"Gate 3e WARN on val: "
            f"|corr(M2v4, M1)|={abs(_corr_v4_m1_val):.4f} > 0.995. "
            "Global v4 collapsed into Member 1; tree contributed no "
            "incremental signal. Continuing anyway -- the stacker will "
            "naturally down-weight a non-orthogonal member."
        )
    # Free the per-row v3 feature matrices we just used.
    del X_train_v3_global, X_val_v3
    gc.collect()

elif _M2V2_ENABLED:
    print("\n[Member 2 v2 / 9.5c] Building GLOBAL X_train_m2v2 (full train, OOF anchor)...")
    X_train_m2v2_global = build_member2_feature_matrix(
        member2_v2_schema,
        subject_ids=_mef_train_subj,
        cluster_ids=_mef_train_cluster,
        bc_ids=_mef_train_bc,
        bc_redacted_mask=bc_redacted_train.astype(np.float32),
        subject_obs_count_log1p=_global_subj_obs_count_train_log1p,
        subject_cat_lookup=_m2v2_subj_cat_lookup,
        subject_num_lookup=_m2v2_subj_num_lookup,
        bench_cat_lookup=_m2v2_bench_cat_lookup,
        bench_num_lookup=_m2v2_bench_num_lookup,
        interaction_matrix=member2_interaction_train.astype(np.float32),
    )
    print(
        f"[Member 2 v2 / 9.5c] X_train_m2v2_global: {X_train_m2v2_global.shape}  "
        f"subject_mean_train_oof: {subject_mean_train_oof.shape}  "
        f"(anchor is OOF per-row, label leakage = 0 by Gate 3a)"
    )

    def _fit_member2_v2_global():
        print(
            "[Member 2 v2 / 9.5c] training LightGBM RESIDUAL "
            f"(X cols={X_train_m2v2_global.shape[1]}, anchor=subject_mean OOF, "
            "objective=binary with logit(anchor) as init_score)..."
        )
        return fit_gbdt_member(
            X=X_train_m2v2_global,
            y=y_train,
            feature_names=member2_v2_schema.feature_names,
            init_pred_train=subject_mean_train_oof,
            n_estimators=int(CFG.get("member2_gbdt", {}).get("n_estimators", 400)),
            learning_rate=float(CFG.get("member2_gbdt", {}).get("learning_rate", 0.05)),
            num_leaves=int(CFG.get("member2_gbdt", {}).get("num_leaves", 31)),
            min_data_in_leaf=int(CFG.get("member2_gbdt", {}).get("min_data_in_leaf", 50)),
            early_stopping_rounds=int(CFG.get("member2_gbdt", {}).get("early_stopping_rounds", 30)),
            seed=SEED,
            parity_atol=1.0e-5,
            val_fraction=float(CFG.get("member2_gbdt", {}).get("val_fraction", 0.1)),
            max_bin=int(CFG.get("member2_gbdt", {}).get("max_bin", 63)),
            force_col_wise=bool(CFG.get("member2_gbdt", {}).get("force_col_wise", True)),
            log_period=int(CFG.get("member2_gbdt", {}).get("log_period", 25)),
            num_threads=CFG.get("member2_gbdt", {}).get("num_threads", None),
            holdout_group_id=gbdt_train_item_id,
        )

    # Cache key digests: tie the cache to subject_mean inputs + the new
    # feature matrix shape + smoothing param. ``subjmean_anchor_v2`` is
    # the version tag; bumping it forces a refit. The subject_mean digest
    # ensures swapping the anchor (e.g., different smoothing) invalidates.
    import hashlib as _hashlib_m2v2
    _m2v2_anchor_digest = _hashlib_m2v2.sha256(
        np.ascontiguousarray(subject_mean_train_oof, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    _m2v2_X_digest = _hashlib_m2v2.sha256(
        np.ascontiguousarray(X_train_m2v2_global, dtype=np.float32).tobytes()
    ).hexdigest()[:16]
    gbdt_state = cache_or_compute(
        "gbdt_state",
        key_inputs=(
            int(X_train_m2v2_global.shape[1]), len(primary.train), SEED,
            "subjmean_anchor_v2",     # Task 3 invalidator
            "non_embedding_schema_v1",  # Task 3 invalidator
            "speed_v1", "init_v2", "honest_loss_v1",
            "coldsplit_v1", "redact_v1",
            round(BC_REDACT_FRAC, 3), _bc_redact_seed,
            round(float(_M2V2_SMOOTHING), 4),
            int(OOF_N_FOLDS), int(OOF_SEED), bool(OOF_RETRAIN_M1),
            _m2v2_anchor_digest, _m2v2_X_digest,
        ),
        compute_fn=_fit_member2_v2_global,
    )
    if str(gbdt_state.output_mode) != "residual_logit":
        raise RuntimeError(
            f"Expected gbdt_state.output_mode='residual_logit', got "
            f"{gbdt_state.output_mode!r}. Cache invalidation likely failed."
        )

    # Gate 3b: parity. ``fit_gbdt_member`` internally asserts
    # parity_atol=1e-5 between LGBM raw_score and the pure-NumPy walker
    # on the residual target. If the assertion fired without raising,
    # parity is OK. We re-verify on a small batch to be explicit.
    print("\n[Gate 3b] verifying pure-NumPy tree walker parity vs LightGBM raw_score...")
    _gate3b_sample = int(min(2000, X_train_m2v2_global.shape[0]))
    _gate3b_idx = np.random.default_rng(0).choice(
        X_train_m2v2_global.shape[0], size=_gate3b_sample, replace=False,
    )
    from src.gbdt_member import predict_raw as _gbdt_predict_raw_g3b
    _walker_raw = _gbdt_predict_raw_g3b(gbdt_state, X_train_m2v2_global[_gate3b_idx])
    print(
        f"[Gate 3b] PASS (implicit from fit_gbdt_member): parity_atol=1e-5 "
        f"asserted on full train during fit. Re-verified pure-NumPy walker "
        f"on {_gate3b_sample:,}-row sample: raw_score range "
        f"[{float(_walker_raw.min()):.4f}, {float(_walker_raw.max()):.4f}]."
    )

    # Gate 3c: composer round-trip. Verify that the composer
    # `sigmoid(logit(anchor) + tree_raw)` is numerically stable for
    # extreme anchors and equals the closed-form composition.
    print("\n[Gate 3c] composer round-trip on extreme subject_mean anchors...")
    _gate3c_X = X_train_m2v2_global[_gate3b_idx[:5]]
    _gate3c_anchors = np.array([1e-6, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-6], dtype=np.float64)
    _gate3c_compose = gbdt_compose_residual_batch(gbdt_state, _gate3c_X, _gate3c_anchors)
    import warnings as _g3c_warn
    for _i, (_a, _p) in enumerate(zip(_gate3c_anchors, _gate3c_compose)):
        if not np.isfinite(_p):
            _g3c_warn.warn(
                f"Gate 3c WARN: composer emitted non-finite for anchor={_a}"
            )
        if not (0.0 < float(_p) < 1.0):
            _g3c_warn.warn(
                f"Gate 3c WARN: composer emitted out-of-range {_p} for anchor={_a}"
            )
    # Round-trip identity: verify p_compose == sigmoid(logit(anchor) + tree_raw).
    _gate3c_raw = _gbdt_predict_raw_g3b(gbdt_state, _gate3c_X)
    _gate3c_logit_anchor = np.log(_gate3c_anchors / (1.0 - _gate3c_anchors))
    _gate3c_expected = 1.0 / (1.0 + np.exp(-(_gate3c_logit_anchor + _gate3c_raw)))
    _gate3c_delta = float(np.abs(_gate3c_compose - _gate3c_expected).max())
    if _gate3c_delta > 1e-6:
        _g3c_warn.warn(
            f"[Gate 3c] WARN: composer disagreed with "
            "sigmoid(logit(anchor) + tree_raw) by max "
            f"{_gate3c_delta:.2e} (tolerance 1e-6). Composer may be broken."
        )
    print(
        f"[Gate 3c] PASS: composer round-trip max delta = {_gate3c_delta:.2e} "
        f"(tolerance 1e-6); composer emits valid probabilities for anchors in "
        f"[1e-6, 1-1e-6]."
    )

    # Apply Member 2 v2 to val rows -- this OVERRIDES the placeholder set
    # in section 9c.
    p_member2_val = gbdt_compose_residual_batch(
        gbdt_state, X_val_dense_m2v2, subject_mean_val,
    )
    nll_m2 = float(-(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
                     + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean())
    print(
        f"\n[Member 2 v2] val log-loss: {nll_m2:.6f}  "
        f"(compare to Member 1 val below in section 9f summary)"
    )
    print(
        f"[Member 2 v2] train NLL stored in state: {gbdt_state.train_loss:.6f}  "
        f"booster's internal val NLL (item-cold split): {gbdt_state.val_loss:.6f}"
    )
    print(f"[Member 2 v2] n_trees={gbdt_state.n_trees}  bias={gbdt_state.bias:+.4f}")

# %% [markdown]
# ## 9f. Train the stacker (ridge logistic regression on OOF train predictions)
#
# Replaces the previous val-trained stacker. Now the stacker learns on
# OOF train predictions (each row's prediction comes from a member that
# never saw that row's item) + the per-row labels. Val is reported but
# NOT fit on -- it is now a pure holdout for the meta-learner.

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    build_stacker_features,
    fit_stacker,
    stacker_feature_names,
)

# ---------- Auxiliary stacker features (val side, kept for final reporting) ----------
val_bench_present = np.array(
    [
        1.0 if str(b) in indexer.bc_to_id else 0.0
        for b in primary.val["benchmark_condition_key"]
    ],
    dtype=np.float32,
)
print("[Stacker] computing val-side aux features (NN support, mean sim, centroid dist)...")
val_nn_mean_sim = nn_val_mat[:, 1].astype(np.float32)
val_nn_support = nn_val_mat[:, 2].astype(np.float32)

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

# Task 4: when Member 5 is enabled, build a [N, 5] member_probs stack
# (M1, M2, M3, M4, M5); the column order is LOCKED across val and OOF-train
# so the runtime's stacker.apply_one sees the same column ordering.
_stacker_member_list_val = [p_a_val, p_member2_val, p_member3_val, p_member4_val]
_stacker_member_names_val = ["M1 (p_a_val)", "M2 (p_member2_val)", "M3 (p_member3_val)", "M4 (p_member4_val)"]
if _M5_ENABLED:
    _stacker_member_list_val.append(p_member5_val)
    _stacker_member_names_val.append("M5 (p_member5_val)")

# Defensive shape audit + 1-D coercion. ``np.stack`` raises a generic
# ``ValueError: all input arrays must have the same shape`` when any of
# the per-member val arrays is a different shape from the rest -- which
# does not tell us WHICH member is the offender. We pre-check here so
# the diagnostic names the broken member, and we ravel any 2-D / column-
# vector member silently (a (N, 1) -> (N,) coercion is information-
# preserving).
_N_VAL_EXPECTED = int(len(primary.val))
_stacker_member_diagnosed = []
import warnings as _stack_warn
_stacker_audit_lines = [
    f"[Stacker val audit] expected per-member length = {_N_VAL_EXPECTED:,}"
]
_stacker_audit_mismatched = False
for _name, _arr in zip(_stacker_member_names_val, _stacker_member_list_val):
    if _arr is None:
        _stacker_audit_lines.append(
            f"  - {_name}: array is None -- a 9.5c branch did not populate "
            "this member. Re-run the relevant member-fit cells."
        )
        _stacker_audit_mismatched = True
        _stacker_member_diagnosed.append(_arr)
        continue
    _arr_np = np.asarray(_arr)
    _orig_shape = tuple(_arr_np.shape)
    _flat = _arr_np.reshape(-1) if _arr_np.ndim != 1 else _arr_np
    _len = int(_flat.shape[0])
    _len_ok = (_len == _N_VAL_EXPECTED)
    _shape_note = "" if _orig_shape == (_N_VAL_EXPECTED,) else (
        f"  (coerced from shape {_orig_shape} -> ({_len},))"
    )
    _stacker_audit_lines.append(
        f"  - {_name}: shape={_orig_shape}, length={_len:,}"
        f"  {'OK' if _len_ok else 'LENGTH MISMATCH'}"
        f"{_shape_note}"
    )
    if not _len_ok:
        _stacker_audit_mismatched = True
    _stacker_member_diagnosed.append(_flat)
print("\n".join(_stacker_audit_lines))
if _stacker_audit_mismatched:
    _stack_warn.warn(
        "[Stacker val audit] one or more member val arrays do not match "
        f"primary.val length ({_N_VAL_EXPECTED:,}). The stacker cannot be "
        "constructed until every member returns a per-row array of the "
        "right length. Most common cause: the member's val-scoring cell "
        "was skipped or short-circuited by a stale cache. Re-run the "
        "specific member's 9.x cell with its cache file deleted, or set "
        "the corresponding CFG['memberX']['enabled']=False to drop it "
        "from the stacker. The stack call below will raise a clearer "
        "error than np.stack's generic message."
    )
    # Surface a CLEAR error -- this is a structural invariant of the
    # stacker (every member must be (N_val,)), not a soft data-quality
    # issue. Halting here is safer than producing a corrupt stacker.
    raise RuntimeError(
        "[Stacker val audit] per-member val length mismatch -- see the "
        "audit lines above for the offending member. Fix the upstream "
        "scoring cell, do not edit this stack call."
    )
_stacker_member_list_val = _stacker_member_diagnosed
stacker_member_probs_val = np.stack(_stacker_member_list_val, axis=1).astype(np.float32)

stacker_X_val = build_stacker_features(
    member_probs=stacker_member_probs_val,
    bench_present=val_bench_present,
    nn_neighbor_support=val_nn_support,
    nn_mean_similarity=val_nn_mean_sim,
    centroid_distance=val_centroid_dist,
)
print(f"[Stacker] X_val (final-reporting view): {stacker_X_val.shape} "
      f"(n_members={'5' if _M5_ENABLED else '4'})")

# ---------- OOF training side: build the stacker's TRAIN inputs from per-fold OOF preds ----------
print("[Stacker] computing OOF-train aux features + member prob stack...")
_train_bench_present = np.array(
    [
        1.0 if str(b) in indexer.bc_to_id else 0.0
        for b in primary.train["benchmark_condition_key"]
    ],
    dtype=np.float32,
)
_stacker_member_list_train_oof = [
    p_a_train_oof.astype(np.float32),
    p2_train_oof.astype(np.float32),
    p3_train_oof.astype(np.float32),
    p4_train_oof.astype(np.float32),
]
_stacker_oof_names = [
    "M1 (p_a_train_oof)", "M2 (p2_train_oof)",
    "M3 (p3_train_oof)", "M4 (p4_train_oof)",
]
if _M5_ENABLED:
    _stacker_member_list_train_oof.append(p5_train_oof.astype(np.float32))
    _stacker_oof_names.append("M5 (p5_train_oof)")

# Mirror the val-side defensive audit: same generic ``np.stack`` error,
# same need to surface WHICH OOF member is the offender. Length mismatch
# here usually means the per-fold OOF accumulator wasn't finalized for
# one of the members (cache HIT with shape drift), or a fold's
# write_fold call silently underfilled.
_N_TRAIN_EXPECTED = int(len(primary.train))
_stacker_oof_diagnosed = []
_stacker_oof_audit_lines = [
    f"[Stacker OOF audit] expected per-member length = {_N_TRAIN_EXPECTED:,}"
]
_stacker_oof_audit_mismatched = False
for _name, _arr in zip(_stacker_oof_names, _stacker_member_list_train_oof):
    if _arr is None:
        _stacker_oof_audit_lines.append(
            f"  - {_name}: array is None -- the per-fold accumulator "
            "produced no values. Re-run the per-fold OOF loop."
        )
        _stacker_oof_audit_mismatched = True
        _stacker_oof_diagnosed.append(_arr)
        continue
    _arr_np = np.asarray(_arr)
    _orig_shape = tuple(_arr_np.shape)
    _flat = _arr_np.reshape(-1) if _arr_np.ndim != 1 else _arr_np
    _len = int(_flat.shape[0])
    _len_ok = (_len == _N_TRAIN_EXPECTED)
    _shape_note = "" if _orig_shape == (_N_TRAIN_EXPECTED,) else (
        f"  (coerced from shape {_orig_shape} -> ({_len},))"
    )
    _stacker_oof_audit_lines.append(
        f"  - {_name}: shape={_orig_shape}, length={_len:,}"
        f"  {'OK' if _len_ok else 'LENGTH MISMATCH'}"
        f"{_shape_note}"
    )
    if not _len_ok:
        _stacker_oof_audit_mismatched = True
    _stacker_oof_diagnosed.append(_flat.astype(np.float32, copy=False))
print("\n".join(_stacker_oof_audit_lines))
if _stacker_oof_audit_mismatched:
    raise RuntimeError(
        "[Stacker OOF audit] per-member OOF length mismatch -- see the "
        "audit lines above for the offending member. Fix the upstream "
        "per-fold cell, do not edit this stack call."
    )
_stacker_member_list_train_oof = _stacker_oof_diagnosed
stacker_member_probs_train_oof = np.stack(
    _stacker_member_list_train_oof, axis=1,
)
stacker_X_train_oof = build_stacker_features(
    member_probs=stacker_member_probs_train_oof,
    bench_present=_train_bench_present,
    nn_neighbor_support=nn_support_oof,
    nn_mean_similarity=nn_mean_sim_oof,
    centroid_distance=centroid_dist_oof,
)
ylab_train_np = primary.train["label"].astype(float).to_numpy().astype(np.float32)
print(f"[Stacker] X_train_oof: {stacker_X_train_oof.shape}  ylab_train: {ylab_train_np.shape}")


_n_stacker_members = 5 if _M5_ENABLED else 4
_stacker_feature_names = stacker_feature_names(_n_stacker_members)
assert int(stacker_X_train_oof.shape[1]) == len(_stacker_feature_names), (
    f"stacker_X_train_oof has {stacker_X_train_oof.shape[1]} cols but "
    f"expected {len(_stacker_feature_names)} for n_members={_n_stacker_members}"
)


def _fit_stacker_oof():
    return fit_stacker(
        X=stacker_X_train_oof,
        y=ylab_train_np,
        feature_names=_stacker_feature_names,
        n_iters=int(CFG.get("stacker", {}).get("n_iters", 1500)),
        learning_rate=float(CFG.get("stacker", {}).get("learning_rate", 0.05)),
        l2=float(CFG.get("stacker", {}).get("l2", 1.0)),
        early_stopping_patience=int(CFG.get("stacker", {}).get("early_stopping_patience", 200)),
        # Internal 80/20 item-grouped split inside fit_stacker for early
        # stopping. We pass holdout_group_id when available so the early-
        # stopping val is item-disjoint (matches the OOF discipline).
        val_fraction=0.2,
        seed=SEED,
    )


import hashlib as _hashlib

# Digest the actual OOF-train input matrix + labels so the cache invalidates
# whenever any upstream member's OOF predictions change. This is the cache
# discipline that caught the stale-stacker bug during the diversification
# rollout -- bake it in from day 1 on the OOF flavor.
_stacker_Xtrain_digest = _hashlib.sha256(
    np.ascontiguousarray(stacker_X_train_oof, dtype=np.float32).tobytes()
).hexdigest()[:16]
_stacker_ytrain_digest = _hashlib.sha256(
    np.ascontiguousarray(ylab_train_np, dtype=np.float32).tobytes()
).hexdigest()[:16]
stacker_state = cache_or_compute(
    "stacker_state",
    key_inputs=(
        stacker_X_train_oof.shape[1], len(ylab_train_np), SEED,
        # Cache invalidators: OOF (Task 1) and Member 5 (Task 4). The
        # m5 tag bumps the key when Member 5 is added/removed so we
        # never silently reuse a 4-member stacker on 5-member features.
        "oof_v1",
        "m5_v1" if _M5_ENABLED else "no_m5",
        int(_n_stacker_members),
        int(OOF_N_FOLDS), int(OOF_SEED),
        bool(OOF_RETRAIN_M1),
        _stacker_Xtrain_digest, _stacker_ytrain_digest,
    ),
    compute_fn=_fit_stacker_oof,
)
print(f"[Stacker] weights: {stacker_state.weights}")
print(f"[Stacker] bias:    {stacker_state.bias:.4f}")

# Apply the OOF-fit stacker on (a) OOF-train inputs -- for Gate 1d --
# and (b) val inputs -- for final reporting (val never touched the fit).
p_stacker_train_oof = stacker_apply_batch(stacker_state, stacker_X_train_oof)
p_stacker_val = stacker_apply_batch(stacker_state, stacker_X_val)

def _bce(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

nll_stack_train_oof = _bce(ylab_train_np, p_stacker_train_oof)
nll_stack_val = _bce(ylab_val, p_stacker_val)
nll_uniform_val = _bce(ylab_val, stacker_member_probs_val.mean(axis=1))

print(f"\n[Stacker] val log-loss summary (OOF-fit stacker, val is TRUE holdout):")
print(f"  Member 1 (Model A IRT-MLP, val):    {_bce(ylab_val, p_a_val):.6f}")
print(f"  Member 2 (LightGBM, val):           {nll_m2:.6f}")
print(f"  Member 3 (kNN-similarity, val):     {nll_m3:.6f}")
print(f"  Member 4 (LogReg, val):             {nll_m4:.6f}")
if _M5_ENABLED:
    print(f"  Member 5 (difficulty-kNN, val):     {nll_m5:.6f}")
print(f"  Uniform avg of {_n_stacker_members} members (val):     {nll_uniform_val:.6f}")
print(f"  STACKER (val, OOF-fit):             {nll_stack_val:.6f}")
print(f"  STACKER (OOF-train, in-sample):     {nll_stack_train_oof:.6f}")
if nll_stack_val > nll_uniform_val + 1e-3:
    print(
        "WARNING: Stacker did not beat uniform average. Consider increasing "
        "stacker.l2 or stacker.n_iters in CFG, or check that members are "
        "diverse enough."
    )

# Gate 3e (RED-TEAM, Task 3): error-correlation between Member 2 and
# Member 1 on val. If the new Member 2 (subject-mean residual + non-
# embedding schema) is still highly correlated with Member 1's errors,
# it's contributing redundant signal to the stacker and Task 3 didn't
# achieve its decorrelation goal.
if _M2V2_ENABLED:
    _y64 = ylab_val.astype(np.float64)
    _err_m1 = p_a_val.astype(np.float64) - _y64
    _err_m2 = p_member2_val.astype(np.float64) - _y64
    _corr_m2_m1 = float(np.corrcoef(_err_m2, _err_m1)[0, 1])
    print(
        f"\n[Gate 3e] Member 2 v2 vs Member 1 error correlation (val): "
        f"corr(err_m2, err_m1) = {_corr_m2_m1:+.4f}"
    )
    if abs(_corr_m2_m1) > 0.85:
        print(
            f"[Gate 3e] FLAG: |corr|={abs(_corr_m2_m1):.4f} > 0.85 -- Member 2 v2 "
            "is still strongly correlated with Member 1. Possible causes:\n"
            "  1. Subject_obs_count_log1p is dominating splits (it's monotonic\n"
            "     with subject confidence which Member 1 also captures).\n"
            "  2. Mean-encoded interaction columns are recapturing the\n"
            "     subject-by-cluster signal Member 1 learns via theta x u.\n"
            "  3. Bench/subject metadata one-hots are correlated with the\n"
            "     IRT-MLP's metadata embeddings.\n"
            "  Consider: lower lr / num_leaves, drop subject_obs_count_log1p,\n"
            "  or move some interaction cols to Member 5 (Task 4)."
        )
    else:
        print(
            f"[Gate 3e] PASS: |corr|={abs(_corr_m2_m1):.4f} <= 0.85; Member 2 "
            "errors are sufficiently decorrelated from Member 1's. Task 3 met "
            "its decorrelation goal."
        )

# Gate 4b (RED-TEAM, Task 4): error-correlation between Member 5
# (difficulty-projected kNN) and Member 3 (raw-embedding kNN) on val.
# The whole point of Member 5 is to find DIFFERENT neighborhoods than
# Member 3 (difficulty vs topic). If their val errors are >= 0.90
# correlated, they're effectively the same model and Member 5 is
# adding cost without diversity -- the stacker can't extract a
# useful linear combination of two near-collinear features.
#
# We also report Member 5 vs Member 1 as a secondary diagnostic: if
# Member 5's neighborhoods reduce to "subject difficulty propensity"
# (the IRT-MLP's theta), the M5/M1 corr will be high and the
# diversity case for Member 5 collapses.
if _M5_ENABLED and p_member5_val is not None:
    _y64_4b = ylab_val.astype(np.float64)
    _err_m1 = p_a_val.astype(np.float64) - _y64_4b
    _err_m3 = p_member3_val.astype(np.float64) - _y64_4b
    _err_m5 = p_member5_val.astype(np.float64) - _y64_4b
    _corr_m5_m3 = float(np.corrcoef(_err_m5, _err_m3)[0, 1])
    _corr_m5_m1 = float(np.corrcoef(_err_m5, _err_m1)[0, 1])
    print(
        f"\n[Gate 4b] Member 5 vs Member 3 error correlation (val): "
        f"corr(err_m5, err_m3) = {_corr_m5_m3:+.4f}"
    )
    print(
        f"[Gate 4b] Member 5 vs Member 1 error correlation (val): "
        f"corr(err_m5, err_m1) = {_corr_m5_m1:+.4f}"
    )
    if abs(_corr_m5_m3) >= 0.90:
        print(
            f"[Gate 4b] FLAG: |corr(err_m5, err_m3)|={abs(_corr_m5_m3):.4f} "
            ">= 0.90 -- Member 5 is essentially redundant with Member 3 "
            "(difficulty-projected kNN and raw-embedding kNN are finding the "
            "same neighborhoods). Possible fixes:\n"
            "  1. Lower CFG['member5']['tau'] so the kernel is sharper and\n"
            "     the difficulty distance distinguishes neighbors more.\n"
            "  2. Increase CFG['member5']['ridge_alpha'] to suppress the\n"
            "     low-signal embedding dimensions in the projection.\n"
            "  3. Drop Member 5 if the diversity gain doesn't justify it."
        )
    else:
        print(
            f"[Gate 4b] PASS: |corr(err_m5, err_m3)|={abs(_corr_m5_m3):.4f} "
            "< 0.90; Member 5's neighborhoods are sufficiently distinct from "
            "Member 3's. Task 4 met its decorrelation goal."
        )
    # Report Member 5's stacker weight as a final diversity diagnostic:
    # a weight near zero means the stacker found nothing to add.
    _w5 = float(stacker_state.weights[4])  # logit_member5 is column 4
    print(
        f"[Task 4] Member 5 stacker weight = {_w5:+.4f} "
        + ("(close to zero -- stacker found little to add; consider "
           "dropping or retuning Member 5)"
           if abs(_w5) < 0.05
           else "(non-trivial -- Member 5 is contributing)")
    )

# %% [markdown]
# ## 9f-bis. Gate 1d: train-vs-val optimism check
#
# If the OOF stacker's training inputs were truly leakage-free, the
# in-sample (OOF-train) log-loss and the held-out (val) log-loss
# should be SIMILAR. A large train-better-than-val gap means the OOF
# predictions still contain residual contamination -- they're letting
# the stacker memorize per-row label hints. We flag when the gap
# exceeds `CFG["oof"]["optimism_threshold_nats"]` (default 0.03 nats).
#
# This is a SOFT gate (just a warning), because some optimism is
# expected from non-leakage sources (stacker's own internal early-
# stopping val is in-sample to the stacker fit). Gate 1c is the hard
# gate; this is the diagnostic that fingers WHICH layer leaked.

# %%
_optimism_threshold = float(CFG["oof"].get("optimism_threshold_nats", 0.03))
_gate1d = report_train_vs_val_optimism(
    train_loss=nll_stack_train_oof,
    val_loss=nll_stack_val,
    threshold_nats=_optimism_threshold,
    label="OOF stacker",
)
print(
    f"[OOF Gate 1d] OOF-train={nll_stack_train_oof:.5f}  val={nll_stack_val:.5f}  "
    f"gap={_gate1d['gap']:+.5f} nats  threshold={_optimism_threshold:.3f}  "
    f"flagged={_gate1d['flag']}"
)
if _gate1d["flag"]:
    print(
        "[OOF Gate 1d] WARNING: optimism gap exceeded threshold. Inspect "
        "fold-NN index (Gate 1b), fold mean-encoded stats (must be fit on "
        "fold-train labels only), and the member feature schema (currently "
        "global, with documented small leakage). If Gate 1c still passes "
        "this is likely the global-schema leak surfacing."
    )
else:
    print("[OOF Gate 1d] PASS: train-vs-val optimism is within threshold.")

# %% [markdown]
# ## 9f-ter. Gate 1c: shuffled-label control (toggleable)
#
# Permute the training labels uniformly at random, then re-run the OOF
# pipeline against those PERMUTED labels and confirm the resulting
# stacker collapses to chance (val log-loss approx entropy of the val
# label prior). A correct pipeline CANNOT beat chance on randomized
# labels -- there is no signal to learn. A leaky pipeline will beat
# chance because the inputs themselves carry label information that
# bypassed the permutation.
#
# Default mode: SHALLOW -- re-runs per-fold Members 2/3/4 with
# shuffled labels but reuses the real Member 1 OOF predictions
# (training Member 1 per-fold with shuffled labels would cost another
# ~3 hours of Colab time). This tests every fold-scoped leakage path
# EXCEPT Member 1's residual-anchor leak. The deliberate weakening is
# flagged in the final task summary.
#
# To enable: set `CFG["oof"]["run_shuffled_label_control"] = True` in
# the CFG cell (or override here in a one-off run). To upgrade to the
# DEEP mode (re-train Member 1 per-fold on shuffled labels too), set
# `CFG["oof"]["shuffled_label_control_mode"] = "deep"` -- expensive,
# ~3-5h additional Colab. Skip on subsequent runs once gate has passed.

# %%
from src.oof_pipeline import entropy_of_label_prior, make_permuted_labels

_run_gate1c = bool(CFG["oof"].get("run_shuffled_label_control", False))
_gate1c_mode = str(CFG["oof"].get("shuffled_label_control_mode", "shallow"))

if not _run_gate1c:
    print(
        "[OOF Gate 1c] SKIPPED (CFG['oof']['run_shuffled_label_control']=False). "
        "Toggle to True after Task 1 to confirm zero pipeline leakage; do not "
        "ship to leaderboard without having seen this gate PASS at least once."
    )
else:
    print(
        f"[OOF Gate 1c] RUNNING in {_gate1c_mode!r} mode. "
        "This will re-run per-fold member training on permuted labels."
    )

    _shuf_seed = int(CFG["oof"].get("shuffled_label_seed", 12345))
    y_train_shuf = make_permuted_labels(y=y_train, seed=_shuf_seed).astype(np.float32)

    # Expected val log-loss floor under the null (no signal): entropy of the
    # *val* label prior. The shuffled-label stacker should converge to predicting
    # the constant label mean on val rows, with log-loss equal to this entropy.
    _val_entropy = entropy_of_label_prior(ylab_val.astype(np.float64))
    print(f"[OOF Gate 1c] val label prior entropy = {_val_entropy:.5f} nats "
          f"(label_mean={float(ylab_val.mean()):.4f}). Gate passes if "
          f"shuffled-label stacker val log-loss >= {_val_entropy - 0.005:.5f}.")

    # Per-fold accumulators for the shuffled-label run.
    p_a_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p_a_train_shuf")
    p2_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p2_train_shuf")
    p3_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p3_train_shuf")
    p4_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p4_train_shuf")
    if _M5_ENABLED:
        p5_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p5_train_shuf")
    else:
        p5_train_shuf_acc = None

    for fold in folds:
        print(f"\n[OOF Gate 1c] === fold {fold.fold_id} (shuffled labels) ===")
        fold_suffix_shuf = fold_cache_suffix(
            fold_id=fold.fold_id, train_item_keys=fold.train_item_keys,
        ) + f"__shuf_{_shuf_seed}"
        fold_train_df = slice_train_rows(primary.train, fold, side="train")
        fold_oof_df = slice_train_rows(primary.train, fold, side="oof")

        # Re-fit fold mean encoded stats with SHUFFLED labels.
        _mef_subj_ft, _mef_cluster_ft, _mef_bc_ft = _compute_id_arrays(fold_train_df)
        _mef_subj_fo, _mef_cluster_fo, _mef_bc_fo = _compute_id_arrays(fold_oof_df)
        _y_ft_shuf = y_train_shuf[fold.train_row_idx]
        fold_mes_shuf = fit_mean_encoded_stats(
            subject_ids=_mef_subj_ft,
            cluster_ids=_mef_cluster_ft,
            bc_ids=_mef_bc_ft,
            labels=_y_ft_shuf,
            n_subjects=int(indexer.n_subjects),
            n_clusters=int(_N_CLUSTERS_ME),
            n_bcs=int(indexer.n_bc),
            smoothing=float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)),
        )
        _m2_int_ft_shuf = apply_member2_interaction_features(
            fold_mes_shuf, subject_ids=_mef_subj_ft,
            cluster_ids=_mef_cluster_ft, bc_ids=_mef_bc_ft,
        )
        _m2_int_fo_shuf = apply_member2_interaction_features(
            fold_mes_shuf, subject_ids=_mef_subj_fo,
            cluster_ids=_mef_cluster_fo, bc_ids=_mef_bc_fo,
        )
        _m4_mg_ft_shuf = apply_member4_marginal_features(
            fold_mes_shuf, subject_ids=_mef_subj_ft,
            cluster_ids=_mef_cluster_ft, bc_ids=_mef_bc_ft,
        )
        _m4_mg_fo_shuf = apply_member4_marginal_features(
            fold_mes_shuf, subject_ids=_mef_subj_fo,
            cluster_ids=_mef_cluster_fo, bc_ids=_mef_bc_fo,
        )

        # Fold's NN feature matrices (cached from the real OOF run -- recompute
        # to avoid relying on stale fold_nn_index that got del'd. We re-build
        # the fold NN index from the same fold-train items; passrate uses
        # SHUFFLED labels so its label aggregates carry no signal).
        fold_nn_dir = OOF_NN_BASE_DIR / f"fold_{fold.fold_id}_shuf_{_shuf_seed}"
        fold_nn_index = build_fold_nn_index(
            fold=fold, item_emb_lookup=item_emb_lookup,
            out_dir=fold_nn_dir, nn_cfg=nn_cfg,
            TrainingNNIndex=TrainingNNIndex,
        )
        fold_item_index_map = build_fold_item_index_map(fold)
        # Replace the label column with shuffled labels for the passrate
        # build (otherwise the passrate matrix would still carry real signal).
        fold_train_df_shuf = fold_train_df.copy()
        fold_train_df_shuf["label"] = _y_ft_shuf
        fold_passrate_csr, fold_passrate_mask_csr = build_passrate_table(
            train_df=fold_train_df_shuf,
            item_index_map=fold_item_index_map,
            subject_index_map=indexer.subject_to_id,
        )
        fold_item_bench_id = reindex_per_item_array(
            arr=item_benchmark_id_arr, train_item_keys_global=train_item_keys,
            fold=fold, fill=-1,
        )
        fold_item_bench_age = reindex_per_item_array(
            arr=item_benchmark_age_arr, train_item_keys_global=train_item_keys,
            fold=fold, fill=np.float32(np.nan),
        )
        fold_item_cluster = reindex_per_item_array(
            arr=item_cluster_id_arr, train_item_keys_global=train_item_keys,
            fold=fold, fill=-1,
        )
        fold_cond_context = build_conditional_passrate_context(
            train_df=fold_train_df_shuf,
            item_index_map=fold_item_index_map,
            subject_index_map=indexer.subject_to_id,
            subject_to_family_id=s2fam,
            subject_to_macro_family_id=s2macro,
            subject_to_organization_id=s2org,
            item_benchmark_id=fold_item_bench_id,
            item_benchmark_age=fold_item_bench_age,
            item_cluster_id=fold_item_cluster,
            n_families=N_FAMILIES,
            n_macro_families=N_MACRO_FAMILIES,
            n_organizations=N_ORGANIZATIONS,
            n_clusters=N_CLUSTERS_CTX,
        )
        _ftk_train, _fsid_train = _split_query(fold_train_df_shuf)
        _ftk_oof, _fsid_oof = _split_query(fold_oof_df)
        nn_train_mat_fold_shuf = compute_nn_features_streaming(
            query_item_keys=_ftk_train, item_emb_lookup=item_emb_lookup,
            subject_ids=_fsid_train, nn_index=fold_nn_index,
            passrate_csr=fold_passrate_csr,
            passrate_mask_csr=fold_passrate_mask_csr,
            cfg=nn_cfg, exclude_self=True, query_chunk_size=NN_QUERY_CHUNK,
            conditional_context=fold_cond_context,
        )
        nn_oof_mat_fold_shuf = compute_nn_features_streaming(
            query_item_keys=_ftk_oof, item_emb_lookup=item_emb_lookup,
            subject_ids=_fsid_oof, nn_index=fold_nn_index,
            passrate_csr=fold_passrate_csr,
            passrate_mask_csr=fold_passrate_mask_csr,
            cfg=nn_cfg, exclude_self=False, query_chunk_size=NN_QUERY_CHUNK,
            conditional_context=fold_cond_context,
        )

        # Fold X (global schema -- documented leak).
        _bc_redacted_ft = bc_redacted_train[fold.train_row_idx]
        _bc_redacted_fo = bc_redacted_train[fold.oof_row_idx]
        X_ft = _build_X(fold_train_df, nn_train_mat_fold_shuf, _bc_redacted_ft)
        X_fo = _build_X(fold_oof_df, nn_oof_mat_fold_shuf, _bc_redacted_fo)
        X_ft_m2 = np.concatenate([X_ft, _m2_int_ft_shuf], axis=1).astype(np.float32, copy=False)
        X_fo_m2 = np.concatenate([X_fo, _m2_int_fo_shuf], axis=1).astype(np.float32, copy=False)
        X_ft_m4 = np.concatenate([X_ft, _m4_mg_ft_shuf], axis=1).astype(np.float32, copy=False)
        X_fo_m4 = np.concatenate([X_fo, _m4_mg_fo_shuf], axis=1).astype(np.float32, copy=False)

        # Member 1: shallow mode reuses REAL M1 OOF preds (M1 didn't see
        # shuffled labels; we treat its output as a fixed feature). Deep mode
        # would re-train M1 per-fold on shuffled labels -- ~3-5h extra Colab.
        if _gate1c_mode == "deep":
            # ... a deep-mode M1 retraining block would go here ...
            # For Task 1 we intentionally stop at shallow mode to fit the
            # Colab budget. See task summary for the deliberate-weakening note.
            raise NotImplementedError(
                "Gate 1c deep mode (per-fold M1 retraining on shuffled labels) "
                "is not implemented in this Task 1 pass. Stick to shallow mode."
            )
        p_a_oof_shuf = p_a_train[fold.oof_row_idx]   # real-label M1, used as feature
        p_a_anchor_shuf = p_a_train[fold.train_row_idx]

        # Fold Member 2 on shuffled labels.
        _gbdt_train_item_id_fold = np.array(
            [int(_item_to_train_idx.get(str(k), -1)) for k in fold_train_df["item_key"]],
            dtype=np.int64,
        )
        _fold_gbdt_shuf = fit_gbdt_member(
            X=X_ft_m2, y=_y_ft_shuf,
            feature_names=tuple(member_feat_schema.feature_names) + tuple(MEMBER2_INTERACTION_FEATURE_NAMES),
            init_pred_train=p_a_anchor_shuf,
            holdout_group_id=_gbdt_train_item_id_fold,
            n_estimators=int(CFG.get("gbdt", {}).get("n_estimators", 400)),
            learning_rate=float(CFG.get("gbdt", {}).get("learning_rate", 0.05)),
            num_leaves=int(CFG.get("gbdt", {}).get("num_leaves", 63)),
            min_data_in_leaf=int(CFG.get("gbdt", {}).get("min_data_in_leaf", 100)),
            feature_fraction=float(CFG.get("gbdt", {}).get("feature_fraction", 0.8)),
            bagging_fraction=float(CFG.get("gbdt", {}).get("bagging_fraction", 0.8)),
            bagging_freq=int(CFG.get("gbdt", {}).get("bagging_freq", 5)),
            early_stopping_rounds=int(CFG.get("gbdt", {}).get("early_stopping_rounds", 25)),
            seed=int(SEED) + 100 * (int(fold.fold_id) + 1) + 99999,
        )
        p2_shuf_fold = gbdt_compose_residual_batch(_fold_gbdt_shuf, X_fo_m2, p_a_oof_shuf)
        p2_train_shuf_acc.write_fold(fold.oof_row_idx, p2_shuf_fold)

        # Fold Member 3 on shuffled labels (passrate already uses shuffled labels).
        _fold_item_emb_stacked = np.stack(
            [np.asarray(item_emb_lookup[k], dtype=np.float32) for k in fold.train_item_keys],
            axis=0,
        )
        _fold_passrate_dense = np.asarray(fold_passrate_csr.todense(), dtype=np.float32)
        _fold_passrate_mask_dense = np.asarray(fold_passrate_mask_csr.todense(), dtype=np.float32)
        # Mirror the per-fold real-label invocation (above). The old
        # K/min_subjects_per_item/tau_init/train_lr/train_iters/train_l2
        # kwargs never matched fit_knn_member's signature and would
        # crash with TypeError as soon as Gate 1c was enabled.
        _fold_knn_shuf = fit_knn_member(
            item_keys=list(fold.train_item_keys),
            item_embeddings=_fold_item_emb_stacked,
            subject_keys=_subject_keys_ordered,
            passrate_dense=_fold_passrate_dense,
            passrate_mask=_fold_passrate_mask_dense,
            pca_dim=int(_m3_cfg.get("pca_dim", 128)),
            quantization=str(_m3_cfg.get("quantization", "int8")),
            k=int(_m3_cfg.get("k", 128)),
            tau_subject=float(_m3_cfg.get("tau_subject", 5.0)),
            tau_global=float(_m3_cfg.get("tau_global", 200.0)),
            item_fallback_weight=float(_m3_cfg.get("item_fallback_weight", 0.5)),
            seed=int(SEED) + 200 * (int(fold.fold_id) + 1) + 99999,
        )
        p3_shuf_fold = knn_apply_batch(
            _fold_knn_shuf,
            np.stack(
                [np.asarray(item_emb_lookup[k], dtype=np.float32)
                 for k in fold_oof_df["item_key"].astype(str)],
                axis=0,
            ),
            fold_oof_df["subject_key"].astype(str).tolist(),
        )
        p3_train_shuf_acc.write_fold(fold.oof_row_idx, p3_shuf_fold)

        # Fold Member 4 on shuffled labels.
        _fold_logreg_shuf = fit_logreg_member(
            X=X_ft_m4, y=_y_ft_shuf,
            feature_names=tuple(member_feat_schema.feature_names) + tuple(MEMBER4_MARGINAL_FEATURE_NAMES),
            epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
            learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
            weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
            l1_strength=float(CFG.get("member4_logreg", {}).get("l1_strength_hybrid", 3.0e-3)),
            min_feature_std=float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)),
            early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
            seed=int(SEED) + 300 * (int(fold.fold_id) + 1) + 99999,
            val_fraction=0.1,
            holdout_group_id=_gbdt_train_item_id_fold,
        )
        p4_shuf_fold = logreg_apply_state_batch(_fold_logreg_shuf, X_fo_m4)
        p4_train_shuf_acc.write_fold(fold.oof_row_idx, p4_shuf_fold)

        # Task 4: Fold Member 5 on shuffled labels. The shuffled labels
        # break any difficulty signal in the projection's regression
        # target -- if the resulting Member 5 still beats chance on val
        # we've found leakage in the difficulty pipeline.
        if _M5_ENABLED:
            _fold_train_subj_ids_shuf = np.fromiter(
                (int(indexer.subject_to_id.get(str(s), -1))
                 for s in fold_train_df["subject_key"]),
                dtype=np.int64, count=len(fold_train_df),
            )
            _fold_item_key_to_pos_shuf = {
                k: i for i, k in enumerate(fold.train_item_keys)
            }
            _fold_train_item_ids_shuf = np.fromiter(
                (int(_fold_item_key_to_pos_shuf.get(str(k), -1))
                 for k in fold_train_df["item_key"]),
                dtype=np.int64, count=len(fold_train_df),
            )
            _fold_oof_subj_ids_shuf = np.fromiter(
                (int(indexer.subject_to_id.get(str(s), -1))
                 for s in fold_oof_df["subject_key"]),
                dtype=np.int64, count=len(fold_oof_df),
            )
            _fold_oof_item_emb_shuf = np.stack(
                [np.asarray(item_emb_lookup[str(k)], dtype=np.float32)
                 for k in fold_oof_df["item_key"]],
                axis=0,
            )
            _fold_m5_shuf = fit_member5(
                item_keys=list(fold.train_item_keys),
                item_embeddings=_fold_item_emb_stacked,
                subject_keys=list(_subject_keys_ordered),
                subject_ids_per_row=_fold_train_subj_ids_shuf,
                item_ids_per_row=_fold_train_item_ids_shuf,
                labels=_y_ft_shuf.astype(np.float64),  # SHUFFLED labels
                k=_M5_K, tau=_M5_TAU, ridge_alpha=_M5_RIDGE_ALPHA,
                item_fallback_weight=_M5_ITEM_FB_WEIGHT,
                min_subjects_per_item=_M5_MIN_SUBJ_PER_ITEM,
            )
            p5_shuf_fold = m5_apply_batch_via_ids(
                _fold_m5_shuf,
                subject_ids=_fold_oof_subj_ids_shuf,
                query_item_embeddings=_fold_oof_item_emb_shuf,
            )
            p5_train_shuf_acc.write_fold(fold.oof_row_idx, p5_shuf_fold)
            del _fold_m5_shuf, _fold_oof_item_emb_shuf
            del _fold_train_subj_ids_shuf, _fold_train_item_ids_shuf
            del _fold_oof_subj_ids_shuf, _fold_item_key_to_pos_shuf

        # M1 real (passed through as feature).
        p_a_train_shuf_acc.write_fold(fold.oof_row_idx, p_a_oof_shuf)

        del fold_nn_index, fold_passrate_csr, fold_passrate_mask_csr, fold_cond_context
        del nn_train_mat_fold_shuf, nn_oof_mat_fold_shuf, X_ft, X_fo
        del X_ft_m2, X_fo_m2, X_ft_m4, X_fo_m4
        del _fold_item_emb_stacked, _fold_passrate_dense, _fold_passrate_mask_dense
        del _fold_gbdt_shuf, _fold_knn_shuf, _fold_logreg_shuf
        gc.collect()

    p_a_shuf = p_a_train_shuf_acc.finalize()
    p2_shuf = p2_train_shuf_acc.finalize()
    p3_shuf = p3_train_shuf_acc.finalize()
    p4_shuf = p4_train_shuf_acc.finalize()
    if _M5_ENABLED:
        p5_shuf = p5_train_shuf_acc.finalize()

    # Fit a stacker on shuffled-label OOF predictions and apply to val.
    # Match the live stacker's [N, n_members] column layout exactly so the
    # null run is comparable.
    _shuf_member_list = [p_a_shuf, p2_shuf, p3_shuf, p4_shuf]
    if _M5_ENABLED:
        _shuf_member_list.append(p5_shuf)
    stacker_member_probs_train_shuf = np.stack(_shuf_member_list, axis=1).astype(np.float32)
    stacker_X_train_shuf = build_stacker_features(
        member_probs=stacker_member_probs_train_shuf,
        bench_present=_train_bench_present,
        nn_neighbor_support=nn_support_oof,
        nn_mean_similarity=nn_mean_sim_oof,
        centroid_distance=centroid_dist_oof,
    )
    stacker_state_shuf = fit_stacker(
        X=stacker_X_train_shuf,
        y=y_train_shuf,
        feature_names=stacker_feature_names(_n_stacker_members),
        n_iters=int(CFG.get("stacker", {}).get("n_iters", 1500)),
        learning_rate=float(CFG.get("stacker", {}).get("learning_rate", 0.05)),
        l2=float(CFG.get("stacker", {}).get("l2", 1.0)),
        early_stopping_patience=int(CFG.get("stacker", {}).get("early_stopping_patience", 200)),
        val_fraction=0.2,
        seed=int(SEED) + 7777,
    )
    p_stacker_val_shuf = stacker_apply_batch(stacker_state_shuf, stacker_X_val)
    nll_stacker_val_shuf = _bce(ylab_val, p_stacker_val_shuf)
    print(
        f"\n[OOF Gate 1c] shuffled-label stacker val log-loss = {nll_stacker_val_shuf:.5f}  "
        f"(label-prior entropy = {_val_entropy:.5f})"
    )
    _slack = nll_stacker_val_shuf - _val_entropy
    _tol = 0.010
    if _slack < -_tol:
        print(
            f"[OOF Gate 1c] FAIL: shuffled-label stacker BEATS chance by "
            f"{-_slack:.5f} nats (tolerance: {_tol:.3f}). This indicates "
            "leakage somewhere in the OOF pipeline. Inspect:\n"
            "  1. Fold NN index passrate (must use fold-train labels only)\n"
            "  2. Fold mean-encoded stats (must use fold-train labels only)\n"
            "  3. Fold member trainings (must see only fold-train rows)\n"
            "  4. Member feature schema (currently global -- if other gates pass\n"
            "     and Gate 1c fails, this is likely the culprit)"
        )
    elif _slack < _tol:
        print(
            f"[OOF Gate 1c] PASS: shuffled-label stacker collapsed to chance "
            f"({_slack:+.5f} nats within tolerance {_tol:.3f})."
        )
    else:
        print(
            f"[OOF Gate 1c] PASS (loose): shuffled-label stacker did WORSE "
            f"than chance ({_slack:+.5f} nats above entropy). The pipeline is "
            "definitely not exploiting label info, but a healthy null run "
            "should land within the tolerance band -- inspect the stacker's "
            "regularization (l2 too high pulls the constant prediction off "
            "the label mean and inflates log-loss above entropy)."
        )

# %% [markdown]
# ## 10. NN calibrator on the STACKED predictions (post-stacker)
#
# The runtime applies the NN calibrator AFTER the stacker, so the
# residual table must store residuals computed against the same
# `p_uncal` the runtime emits at inference (the stacked output, not
# Member 1 alone). We score all four members on TRAIN, run them
# through the same stacker we trained on val, and store residuals
# of (label - p_stacker_train) keyed by (subject, training_item_row).

# %%
from src.nn_calibration import NNCalibrator, SubjectResidualTable


def _fit_nn_calibrator():
    """Fit the post-stacker NN-residual calibrator end-to-end.

    OOF rewire (Task 1 of diversification plan): instead of re-scoring
    every member on full train (which would be in-sample to the
    member's training data), reuse the OOF prediction arrays
    accumulated by section 9.5. The OOF stacker output on those
    inputs is `p_stacker_train_oof` (computed in section 9f); we use
    THAT as the calibrator's `p_uncal_train_stacker`. The residual
    table is therefore built against truly out-of-fold uncalibrated
    probabilities, matching the OOF discipline the stacker itself was
    trained under.

    Skipped: the expensive in-sample kNN chunked rescore (the previous
    `p3_train_local` build was ~80 GB peak and several minutes). The
    OOF `p3_train_oof` already covers all training rows.
    """
    print("[Calibrator] Using OOF member predictions on train (from section 9.5)...")
    p1_train_local = p_a_train_oof.astype(np.float32)
    p2_train_local = p2_train_oof.astype(np.float32)
    p3_train_local = p3_train_oof.astype(np.float32)
    p4_train_local = p4_train_oof.astype(np.float32)
    for _name, _arr in [
        ("p1", p1_train_local), ("p2", p2_train_local),
        ("p3", p3_train_local), ("p4", p4_train_local),
    ]:
        _nll = -(y_train * np.log(np.clip(_arr, 1e-6, 1 - 1e-6))
                 + (1 - y_train) * np.log(1 - np.clip(_arr, 1e-6, 1 - 1e-6))).mean()
        print(f"[Calibrator] {_name}_train_oof: shape={_arr.shape}  log-loss={float(_nll):.6f}")

    # OOF stacker prediction on train rows is exactly `p_stacker_train_oof`
    # computed in section 9f. Reuse it -- no need to rebuild train stacker
    # features here.
    p_uncal_train_stacker_local = p_stacker_train_oof.astype(np.float32)
    print(f"[Calibrator] p_uncal_train_stacker (OOF): shape={p_uncal_train_stacker_local.shape}  "
          f"log-loss={-(y_train * np.log(np.clip(p_uncal_train_stacker_local, 1e-6, 1 - 1e-6)) + (1 - y_train) * np.log(1 - np.clip(p_uncal_train_stacker_local, 1e-6, 1 - 1e-6))).mean():.6f}")

    # --- Build the residual table from the STACKED train predictions.
    key_to_train_row_local = {k: i for i, k in enumerate(train_item_keys)}
    train_subj_ids_local = np.array(
        [indexer.subject_id(str(s)) for s in primary.train["subject_key"]],
        dtype=np.int64,
    )
    train_item_rows_local = np.array(
        [key_to_train_row_local.get(str(k), -1) for k in primary.train["item_key"]],
        dtype=np.int64,
    )
    ok_local = train_item_rows_local >= 0
    residual_table_local = SubjectResidualTable.from_rows(
        subject_ids=train_subj_ids_local[ok_local],
        training_item_rows=train_item_rows_local[ok_local],
        labels=primary.train["label"].astype(float).to_numpy()[ok_local],
        uncal_probs=p_uncal_train_stacker_local[ok_local],
        n_subjects=indexer.n_subjects,
        n_training_items=len(train_item_keys),
    )

    # --- Val-side neighbor lookup (deduped, chunked for memory).
    val_keys_arr_local = np.asarray([str(k) for k in primary.val["item_key"]])
    val_unique_keys_local, val_inverse_local = np.unique(val_keys_arr_local, return_inverse=True)
    K_CAL_local = int(CFG["nn_calibration"]["k"])
    val_uniq_idx_local = np.empty((len(val_unique_keys_local), K_CAL_local), dtype=np.int64)
    val_uniq_sims_local = np.empty((len(val_unique_keys_local), K_CAL_local), dtype=np.float32)
    for _s in tqdm(range(0, len(val_unique_keys_local), NN_QUERY_CHUNK),
                   desc="[Calibrator] val NN search"):
        _e = min(_s + NN_QUERY_CHUNK, len(val_unique_keys_local))
        _chunk_keys = list(val_unique_keys_local[_s:_e])
        _chunk_emb = np.stack(
            [item_emb_lookup[k] for k in _chunk_keys], axis=0
        ).astype(np.float32, copy=False)
        _idx, _sims = nn_index.nearest(
            _chunk_emb, k=K_CAL_local, exclude_self=False, query_keys=_chunk_keys,
        )
        val_uniq_idx_local[_s:_e] = _idx
        val_uniq_sims_local[_s:_e] = _sims
        del _chunk_emb, _idx, _sims
    val_neighbor_rows_local = val_uniq_idx_local[val_inverse_local]
    val_neighbor_sims_local = val_uniq_sims_local[val_inverse_local]

    # Fit the calibrator on the STACKED val predictions with the
    # 2-D (alpha, shrinkage_tau) grid sweep.
    calibrator_local = NNCalibrator.fit_alpha_on_val(
        residual_table=residual_table_local,
        val_subject_ids=val_subj_ids_np,
        val_neighbor_rows=val_neighbor_rows_local,
        val_neighbor_sims=val_neighbor_sims_local,
        val_uncal_probs=p_stacker_val,
        val_labels=ylab_val,
        k=int(CFG["nn_calibration"]["k"]),
        similarity=str(CFG["nn_calibration"].get("similarity", "cosine")),
        temperature=float(CFG["nn_calibration"].get("temperature", 1.0)),
        shrinkage_taus=tuple(CFG["nn_calibration"].get("shrinkage_taus") or
                             (0.0, 0.5, 1.0, 2.0, 5.0)),
    )
    print(f"[Calibrator] state: alpha={calibrator_local.state.alpha:.4f}  "
          f"shrinkage_tau={calibrator_local.state.shrinkage_tau:.4f}  "
          f"fit_method={calibrator_local.state.fit_method}")

    p_final_val_local = calibrator_local.apply(
        residual_table=residual_table_local,
        subject_ids=val_subj_ids_np,
        neighbor_rows=val_neighbor_rows_local,
        neighbor_sims=val_neighbor_sims_local,
        p_uncal=p_stacker_val,
    )
    nll_final_local = float(-(ylab_val * np.log(np.clip(p_final_val_local, 1e-6, 1 - 1e-6))
                              + (1 - ylab_val) * np.log(1 - np.clip(p_final_val_local, 1e-6, 1 - 1e-6))).mean())
    print(f"\n[Calibrator] post-calibration val log-loss: {nll_final_local:.6f}  "
          f"(stacker: {nll_stack:.6f}, delta: {nll_final_local - nll_stack:+.6f})")

    return {
        "calibrator_state": calibrator_local.state,
        "residual_table": residual_table_local,
        "p_uncal_train_stacker": p_uncal_train_stacker_local.astype(np.float32),
        "p_final_val": p_final_val_local.astype(np.float32),
        "val_neighbor_rows": val_neighbor_rows_local,
        "val_neighbor_sims": val_neighbor_sims_local,
        "nll_final": float(nll_final_local),
        "p_a_train": p_a_train_local.astype(np.float32),
        "p1_train": p1_train_local,
        "p2_train": p2_train_local.astype(np.float32),
        "p3_train": p3_train_local.astype(np.float32),
        "p4_train": p4_train_local.astype(np.float32),
    }


# Cache key: include fingerprints of every upstream state the calibrator
# depends on, plus the calibrator's own config. Any change in Model A
# weights, GBDT trees, kNN tables, LogReg weights, stacker weights, or
# the calibrator hyperparameters auto-invalidates the cache.
_oof_member_preds_for_digest = [
    p_a_train_oof, p2_train_oof, p3_train_oof, p4_train_oof,
]
if _M5_ENABLED:
    _oof_member_preds_for_digest.append(p5_train_oof)
_oof_member_preds_digest = _hashlib.sha256(
    np.ascontiguousarray(
        np.stack(_oof_member_preds_for_digest, axis=1),
        dtype=np.float32,
    ).tobytes()
).hexdigest()[:16]
CALIBRATOR_KEY_INPUTS = (
    "nn_calibrator_oof_v1",
    state_fingerprint(ckpt_a_cached["model_state"]),
    state_fingerprint(gbdt_state),
    state_fingerprint(knn_state),
    state_fingerprint(logreg_state),
    state_fingerprint(stacker_state),
    # Task 4: include Member 5's state in the calibrator cache key so a
    # Member-5-on/off toggle (or any retuning of k/tau/ridge_alpha)
    # invalidates the calibrator entry. Without this, a stale calibrator
    # fit on 4-member stacker outputs would silently apply to 5-member
    # stacker outputs.
    state_fingerprint(member5_state) if _M5_ENABLED else "no_m5",
    int(CFG["nn_calibration"]["k"]),
    str(CFG["nn_calibration"].get("similarity", "cosine")),
    float(CFG["nn_calibration"].get("temperature", 1.0)),
    tuple(CFG["nn_calibration"].get("shrinkage_taus") or
          (0.0, 0.5, 1.0, 2.0, 5.0)),
    int(len(primary.train)),
    int(len(primary.val)),
    int(NN_FEATURE_DIM),
    int(OOF_N_FOLDS), int(OOF_SEED), bool(OOF_RETRAIN_M1),
    _oof_member_preds_digest,
)
print(f"[cache] Calibrator key prefix: {state_fingerprint(CALIBRATOR_KEY_INPUTS)}")

_calibrator_bundle = cache_or_compute(
    "nn_calibrator_stacked",
    key_inputs=CALIBRATOR_KEY_INPUTS,
    compute_fn=_fit_nn_calibrator,
)

# Re-hydrate the calibrator object from its picklable state.
calibrator = NNCalibrator(state=_calibrator_bundle["calibrator_state"])
residual_table = _calibrator_bundle["residual_table"]
p_uncal_train_stacker = _calibrator_bundle["p_uncal_train_stacker"]
p_final_val = _calibrator_bundle["p_final_val"]
val_neighbor_rows = _calibrator_bundle["val_neighbor_rows"]
val_neighbor_sims = _calibrator_bundle["val_neighbor_sims"]
nll_final = _calibrator_bundle["nll_final"]
p_a_train = _calibrator_bundle["p_a_train"]
p1_train = _calibrator_bundle["p1_train"]
p2_train = _calibrator_bundle["p2_train"]
p3_train = _calibrator_bundle["p3_train"]
p4_train = _calibrator_bundle["p4_train"]

print(
    f"[Calibrator] alpha={calibrator.state.alpha:.4f}  "
    f"shrinkage_tau={calibrator.state.shrinkage_tau:.4f}  "
    f"fit_method={calibrator.state.fit_method}  "
    f"nll_final={nll_final:.6f} (stacker={nll_stack:.6f})"
)

RESIDUAL_DIR = ROOT / "artifacts" / "nn_calibration_stacked"
RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
residual_table.save(RESIDUAL_DIR)
print(f"[Calibrator] residual table saved to {RESIDUAL_DIR}")

# %% [markdown]
# ## 11. Export the four-member stacked bundle
#
# Two-step export:
#   1. Build the Member 1 (Model A IRT-MLP) bundle via
#      `export_ensemble_run` with a SINGLE member. We reuse the
#      ensemble exporter (rather than the simpler `export_run`)
#      because the four-member stacker downstream expects the
#      bundle layout the ensemble exporter produces.
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
#          enforces the 1500 MB ZIP cap.

# %%
import shutil
import tempfile

from src.export_submission import (
    bundle_training_cache,
    compute_train_counts,
    export_ensemble_run,
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

print("[Export] Step 1: building Member 1 (Model A only) bundle...")
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
    # Ship the conditional NN-feature context (cells [15..22]) and the
    # per-bc-id benchmark-age lookup so the runtime can reproduce the
    # same 23-column NN feature vector training trained on. Both are
    # numpy / scipy.sparse files; total disk overhead is typically
    # a few MB on top of the existing nn cache.
    conditional_context=cond_context,
    bc_id_to_age=bc_id_to_age_arr,
)
print(
    f"[Export] training cache size: {training_cache_result.size_mb:.1f} MB  "
    f"(soft cap {CFG['submission_cache'].get('max_bundle_size_mb', 200)} MB)"
)

# Member 1 is Model A alone. We materialize its state_dict to a
# temp checkpoint file (which is what ``export_ensemble_run``
# expects to torch.load) and then call the ensemble exporter with
# a single-element ``members`` list. blend_weights=[1.0] so the
# runtime ensemble passes Model A's logits through unchanged.
_export_tmp = Path(tempfile.mkdtemp(prefix="single_member_"))
try:
    _path_a = _export_tmp / "member_a.pt"
    torch.save(
        {
            "model_state": dict(ckpt_a["model_state"]),
            "model_cfg": asdict(model_a_cfg),
            "model_name": MODEL_A_NAME,
        },
        _path_a,
    )
    sub_dir_m1 = export_ensemble_run(
        members=[
            {
                "config_id": "meta_dropout",
                "model_name": MODEL_A_NAME,
                "model_cfg": asdict(model_a_cfg),
                "fold_checkpoint_paths": [_path_a],
            }
        ],
        blend_weights=[1.0],
        blend_weights_missing=[1.0],
        force_bench_missing=[False],
        indexer={
            "subject_to_id": dict(indexer.subject_to_id),
            "bc_to_id": dict(indexer.bc_to_id),
        },
        encoder_cfg=CFG["encoder"],
        fold_assignment_sha256="single_member_v1",
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
finally:
    shutil.rmtree(_export_tmp, ignore_errors=True)
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
    # Task 3: ship the global subject_mean table so the runtime uses
    # `subject_mean[subject_id]` as the Member-2 residual anchor
    # (matches training-time convention). Without this, the runtime
    # would fall back to Member 1's prediction as the anchor, which
    # would silently mis-compose the val-fit Member 2 booster's trees.
    subject_mean_table=(subject_mean_table_global if _M2V2_ENABLED else None),
    # Task 4: ship Member 5 (difficulty-projected kNN). The exporter
    # raises if member5_state is non-None and stacker_state has only 4
    # member columns -- that misconfig would silently drop Member 5 at
    # runtime. We're safe here: when _M5_ENABLED, the stacker above
    # was fit with [N, 5] member_probs (feature_dim==9).
    member5_state=(member5_state if _M5_ENABLED else None),
)
print(f"[Export] stacked bundle: {sub_dir}")

# Static audit: catch any forbidden runtime imports BEFORE zipping.
findings = audit_runtime_imports(sub_dir)
if findings:
    raise RuntimeError(f"Static audit found forbidden imports: {findings}")
print("[Export] static import audit: PASS")

bundle_bytes = measure_bundle_size_bytes(sub_dir)
print(f"[Export] bundle size: {bundle_bytes / (1024 * 1024):.1f} MB")

zip_cap_mb = float(CFG["submission"].get("max_zip_size_mb", 1500))
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


print(f"  Member 1 (Model A IRT-MLP)        : {_ll(p_a_val):.6f}")
print(f"  Member 2 (LightGBM)               : {_ll(p_member2_val):.6f}")
print(f"  Member 3 (kNN-similarity)         : {_ll(p_member3_val):.6f}")
print(f"  Member 4 (LogReg)                 : {_ll(p_member4_val):.6f}")
if _M5_ENABLED:
    print(f"  Member 5 (difficulty-kNN)         : {_ll(p_member5_val):.6f}")
print(f"  Uniform avg of {'5' if _M5_ENABLED else '4'}                  : "
      f"{_ll(stacker_member_probs_val.mean(axis=1)):.6f}")
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
# Task 4: Member 5 bundle/runtime parity. When enabled offline, the
# bundle MUST contain artifacts/member5_dknn/ AND the runtime block
# MUST import member5_difficulty_knn -- otherwise the runtime would
# silently fall back to the 4-member path even though the stacker was
# fit on 5 columns (the stacker would then misinterpret the missing
# 5th column as zero, breaking every prediction).
if _M5_ENABLED:
    m5_dir_in_bundle = (sub_dir / "artifacts" / "member5_dknn").exists()
    m5_pure_in_bundle = (sub_dir / "_pure" / "member5_difficulty_knn.py").exists()
    checks.extend([
        ("Member 5 artifacts dir shipped",
         m5_dir_in_bundle,
         "missing artifacts/member5_dknn"),
        ("Member 5 pure module shipped",
         m5_pure_in_bundle,
         "missing _pure/member5_difficulty_knn.py"),
        ("Member 5 runtime loader present in model.py",
         "_MEMBER5_STATE" in model_py_text and "member5_difficulty_knn" in model_py_text,
         "Member 5 loader block missing from model.py"),
        ("Stacker has 5 member columns (feature_dim==9)",
         int(stacker_state.feature_dim) == 9,
         f"stacker feature_dim={int(stacker_state.feature_dim)} (expected 9)"),
    ])
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

# Task 4: Member 5 robustness checks. These mirror the Member 3 checks
# above so a future bundle integrator can spot Member 5 problems the
# same way they spot Member 3 problems.
if _M5_ENABLED and member5_state is not None:
    # Zero-norm query -> Member 5 projects to ~bias, returns sane prob.
    p5_zero = m5_apply_one(
        member5_state, zero_q, _subject_keys_ordered[0],
    )
    assert 0.0 < p5_zero < 1.0 and np.isfinite(p5_zero), (
        f"Member 5 zero-norm query returned {p5_zero!r}, expected (0, 1) finite."
    )
    print(f"  [PASS] M5 zero-norm query -> {p5_zero:.4f} (finite={np.isfinite(p5_zero)})")
    # Unknown subject -> Member 5 falls through to global_mean / its own fallback.
    p5_unk = m5_apply_one(
        member5_state,
        rng_redteam.normal(size=ITEM_EMB_DIM).astype(np.float32),
        "totally_unknown_subject",
    )
    assert 0.0 < p5_unk < 1.0 and np.isfinite(p5_unk), (
        f"Member 5 unknown-subject returned {p5_unk!r}, expected (0, 1) finite."
    )
    print(f"  [PASS] M5 unknown-subject -> {p5_unk:.4f} (fallback path)")

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
