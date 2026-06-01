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
# # SOLVER-PROXY PROBE (early-exit copy of the stacked notebook)
#
# This is a **copy** of `qwen8b_four_member_stacked.py` whose only purpose is
# to answer, as cheaply and as early as possible: *does a solver-based
# item-side proxy (self-consistency / answer-entropy / P(True) /
# cross-model disagreement) carry signal that is **orthogonal** to the NN
# pass-rate signal the production stack already extracts?*
#
# It reuses the data load + embeddings + NN features + item-grouped OOF folds
# (everything the probe needs), then runs the probe in a new section
# (**"6.7 SOLVER-PROXY PROBE"**) inserted right after the OOF fold setup
# (section 6.5) and **before** any of the heavy member training. By default
# `CFG["proxy_probe"]["stop_after"] = True` raises a clean `StopExecution`
# once the verdict is printed, so you get a real incremental-NLL number
# (with an item-clustered bootstrap CI, sliced by NN-support) without paying
# for the full 9-member pipeline. Set `stop_after = False` to fall through
# into the normal notebook below.
#
# Solver models (see `CFG["proxy_probe"]["models"]`): a diverse trio chosen
# for family diversity (cross-model disagreement only estimates item
# *discrimination* if the models fail differently) and -- critically --
# **ungated** HF repos, so we don't repeat the gated-repo 403 that killed
# `google/gemma-2-9b-it`. Each entry may be a plain id or a dict carrying
# per-model overrides (`trust_remote_code`, `system_prompt`, `enable_thinking`):
#   * `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (NVIDIA) -- Llama-3.1-8B reasoning
#     derivative; ungated. Reasoning is toggled by the SYSTEM prompt, so we
#     pass `"detailed thinking off"` to keep answers short/parseable in the
#     512-token budget (otherwise it emits a truncated reasoning trace).
#   * `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` (LG AI) -- different lineage
#     (bilingual KO/EN); ungated but ships custom modeling code, so it needs
#     `trust_remote_code=True`.
#   * a Mistral model (the third slot) -- distinct error modes again.
# All three run by default (`multi_model = True`, `n_samples = 3`); set
# `multi_model = False` to fall back to the single-model (first) probe.
#
# ---
# Original header follows.
#
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
#   * **Member 2 (metadata MLP)**: GLU MLP on subject / benchmark /
#     cluster IDs plus 14 mean-encoded marginals (no item embeddings).
#     Pure-NumPy inference -- **no `torch` import in `model.py`**.
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
#   - `artifacts/member2_metadata_mlp/` -- Member 2 GLU MLP weights + marginal scaler.
#   - `artifacts/member3_knn/` -- Member 3 PCA basis, quantized embeddings,
#     pass-rate table.
#   - `artifacts/member4_logreg/` -- Member 4 weights + bias.
#   - `artifacts/stacker/` -- 8-dim ridge weights + bias.
#   - `artifacts/nn_calibrator_stacked/` + `artifacts/residual_table/`
#     -- post-stacker calibrator state + per-(subject, item) residual
#     table.
#   - `_pure/{member2_metadata_mlp,knn,logreg,stacker,nn_calibration,member_features}.py`
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
import math
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

# Member 2: metadata-only GLU MLP (subject / bc / cluster embeddings +
# 14 mean-encoded marginals). No item embeddings -- structurally orthogonal
# to M1/M3/M4/M6.
CFG.setdefault("member2_mlp", {})
# Feature-composition mode.
#
# "metadata_only" (default after the May 2026 diagnostic):
#   M2 sees ONLY subject/benchmark metadata: family, macro_family,
#   organization, bench_topic (categorical) plus subject_numerical
#   (log_params, release_date + missing flags) and bench_numerical
#   (benchmark_age + missing flag). Subject_id, bc_id, cluster_id,
#   bc_redacted_flag, and the 14 mean-encoded marginals are all
#   DROPPED. This forces M2's signal to come strictly from group-
#   level metadata, which is structurally disjoint from M1
#   (subject/bc embeddings) and M4/M6 (which use marginals).
#
# "full" (legacy, pre-2026-05-30):
#   Includes subject_id, bc_id, cluster_id embeddings + bc_redacted
#   + marginals on top of the metadata block. This was the v2_dcnv2
#   architecture; it left M2 with err-corr 0.87 against M1 and only
#   +0.012 stacker weight on the last full run, so we restrict
#   composition rather than throw more capacity at it.
# Diversification pass (2026-05-30): M2 now also gets the item-cluster
# embedding on top of the metadata block ("metadata_cluster"). Subject/
# bc id embeddings and the marginals stay OFF (those live in M4/M6/M7).
CFG["member2_mlp"].setdefault("feature_mode", "metadata_cluster")
# Categorical-embedding widths. In metadata_only mode the subject /
# bc / cluster widths are forced to 0 below (those embedding tables
# vanish from the model entirely; the saved state still serialises
# their [n+1, 0] empty slabs for runtime API compatibility).
CFG["member2_mlp"].setdefault("d_subj", 32)
CFG["member2_mlp"].setdefault("d_bc", 32)
CFG["member2_mlp"].setdefault("d_cluster", 16)
CFG["member2_mlp"].setdefault("d_family", 16)
CFG["member2_mlp"].setdefault("d_macro", 8)
CFG["member2_mlp"].setdefault("d_org", 16)
CFG["member2_mlp"].setdefault("d_topic", 16)
# Deep tower (two GLU blocks).
CFG["member2_mlp"].setdefault("hid1", 256)
CFG["member2_mlp"].setdefault("hid2", 128)
# DCN-v2 cross tower (parallel with the deep tower).
CFG["member2_mlp"].setdefault("n_cross_layers", 2)
CFG["member2_mlp"].setdefault("cross_rank", 64)
# Optimisation.
CFG["member2_mlp"].setdefault("learning_rate", 1.0e-3)
CFG["member2_mlp"].setdefault("weight_decay", 1.0e-5)
CFG["member2_mlp"].setdefault("epochs", 60)
CFG["member2_mlp"].setdefault("batch_size", 16384)
CFG["member2_mlp"].setdefault("early_stopping_patience", 8)
CFG["member2_mlp"].setdefault("val_fraction", 0.1)
# LR schedule.
CFG["member2_mlp"].setdefault("warmup_epochs", 2)
CFG["member2_mlp"].setdefault("use_cosine_schedule", True)
# EMA + snapshot ensembling.
CFG["member2_mlp"].setdefault("ema_decay", 0.999)
CFG["member2_mlp"].setdefault("snapshot_ensemble_k", 3)
# Categorical dropout (per-field independent UNK replacement).
CFG["member2_mlp"].setdefault("cat_dropout_subject", 0.05)
CFG["member2_mlp"].setdefault("cat_dropout_bc", 0.10)
CFG["member2_mlp"].setdefault("cat_dropout_cluster", 0.10)
CFG["member2_mlp"].setdefault("cat_dropout_family", 0.05)
CFG["member2_mlp"].setdefault("cat_dropout_macro", 0.05)
CFG["member2_mlp"].setdefault("cat_dropout_org", 0.05)
CFG["member2_mlp"].setdefault("cat_dropout_topic", 0.10)
CFG["member2_mlp"].setdefault("feat_dropout", 0.10)
# Label smoothing + Mixup (Mixup on numerical channel only).
CFG["member2_mlp"].setdefault("label_smoothing", 0.005)
CFG["member2_mlp"].setdefault("mixup_alpha", 0.0)
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
# v2: pick the Member 5 *variant*. The legacy "difficulty_knn" path is
# kept for A/B comparison; "residual_cluster" is the new default and
# uses src/member5_subject_cluster_residual.py -- a pure-lookup
# subject x item-cluster residual passrate predictor with zero learned
# weights at inference, structurally orthogonal to the embedding-based
# Members 1/3 and the additive-linear Member 4.
CFG["member5"].setdefault("variant", "residual_cluster")
# Legacy (difficulty_knn) hyperparameters -- still consumed when
# variant == "difficulty_knn".
CFG["member5"].setdefault("k", 32)
CFG["member5"].setdefault("tau", 0.05)
CFG["member5"].setdefault("ridge_alpha", 10.0)
CFG["member5"].setdefault("item_fallback_weight", 0.3)
CFG["member5"].setdefault("min_subjects_per_item", 3)
# Sample size for Gate 4d's apply round-trip probe.
CFG["member5"].setdefault("gate4d_sample_size", 64)
# New (residual_cluster) hyperparameters. ``smoothing_cell`` controls
# how strongly the per-(subject, cluster) cell mean shrinks toward the
# additive (subject + cluster - global) baseline -- larger = more
# shrinkage = residual closer to 0. ``smoothing_marginal`` shrinks the
# 1-D marginals (subject mean, cluster mean) toward the global mean.
# ``residual_scale`` is a final scalar multiplier on the residual term
# at fit time (baked into the saved table); 1.0 means use the full
# residual, smaller values further damp it relative to the marginals.
CFG["member5"].setdefault("smoothing_cell", 30.0)
CFG["member5"].setdefault("smoothing_marginal", 10.0)
CFG["member5"].setdefault("residual_scale", 1.0)
# --- Member 6: Field-weighted Factorization Machine on the M4 dense
# matrix (the same hybrid features Members 2/4 use). FwFM provides a
# bilinear-interaction inductive bias that none of M1/M3/M4 can
# express: M1 is non-linear-on-embeddings, M3 is locally-constant kNN,
# M4 is strictly additive linear. The same dense matrix M4 already
# consumes is fed to M6 unchanged; the *only* difference is the
# pairwise <v_i, v_j> x_i x_j term FwFM adds on top of the linear part.
#
# field_ids defaults to all-zero (single field -> classic FM with a
# single scalar r[0, 0]). To enable true field weighting, the notebook
# cell that fits M6 can pass a 2-field assignment splitting the
# embedding-derived columns from the mean-encoded marginal columns.
#
# Set enabled=False to skip Member 6 entirely (revert to 5-member
# stacker). All downstream cells (stacker, calibrator, heatmaps,
# bundle) detect this flag and act accordingly.
CFG.setdefault("member6", {})
CFG["member6"].setdefault("enabled", True)
# k bumped 8 -> 16. The post-run diagnostic (Gate 6 PASS,
# stacker weight +0.627, second-best individual NLL) shows FwFM is
# the highest-leverage member in the ensemble and Gate 6 explicitly
# suggested raising k. Doubling k doubles the V matrix [F, k] from
# ~10k to ~20k floats -- still trivial in absolute terms but gives
# the bilinear term enough capacity to keep pulling its weight as the
# rest of the ensemble gets refined.
CFG["member6"].setdefault("k", 16)
CFG["member6"].setdefault("learning_rate", 1.0e-3)
CFG["member6"].setdefault("weight_decay_w", 1.0e-5)
# weight_decay_V bumped 1e-4 -> 1.5e-4 because V has 2x params now;
# matching the regularizer scale keeps the effective shrinkage per
# parameter constant.
CFG["member6"].setdefault("weight_decay_V", 1.5e-4)
CFG["member6"].setdefault("weight_decay_r", 1.0e-4)
# epochs bumped 100 -> 140 and patience 10 -> 15 because the larger
# k=16 model has more parameters to fit and the last run was still
# improving past epoch 40. ``early_stopping_patience`` keeps us
# honest: we'll exit early when val-loss plateaus rather than burning
# the full budget every time.
CFG["member6"].setdefault("epochs", 140)
CFG["member6"].setdefault("batch_size", 16384)
CFG["member6"].setdefault("early_stopping_patience", 15)
CFG["member6"].setdefault("val_fraction", 0.1)
# field_split_mode: "single" (classic FM, all cols in field 0) or
# "embedding_vs_marginal" (2 fields: 0=embedding-derived M2 columns,
# 1=mean-encoded marginal columns from MEMBER4_MARGINAL_FEATURE_NAMES).
CFG["member6"].setdefault("field_split_mode", "embedding_vs_marginal")

# --- PCA tail subspace (shared by Members 4 & 5) -------------------------
# Diversification pass (2026-05-30). We fit ONE unsupervised randomized
# PCA on the unique train item embeddings, drop the top `head_drop`
# components (the coarse semantic axis M1/M3 already key on) and keep
# the next `tail_take` components as a "tail" subspace. Members 4 & 5
# operate ONLY on this residual geometry, which carries a different
# error structure than the variance-dominant head. The fit is label-
# free, so reusing one global basis across OOF folds leaks nothing.
CFG.setdefault("pca_tail", {})
CFG["pca_tail"].setdefault("n_components", 256)
CFG["pca_tail"].setdefault("head_drop", 32)
CFG["pca_tail"].setdefault("tail_take", 128)
CFG["pca_tail"].setdefault("seed_offset", 7)

# --- Member 5 variant override: tail-subspace kNN ------------------------
# The diversification pass repurposes Member 5 from the residual-cluster
# lookup to a kNN in the PCA tail subspace (subject-aware via the same
# subject x item passrate matrix Member 3 uses). Set variant back to
# "residual_cluster"/"difficulty_knn" for the legacy behaviour.
CFG["member5"]["variant"] = "tail_knn"
# kNN hyperparameters for the tail-subspace variant (reuses
# fit_knn_member, so the knobs mirror Member 3's).
CFG["member5"].setdefault("tail_k", 128)
CFG["member5"].setdefault("tail_tau_subject", 5.0)
CFG["member5"].setdefault("tail_tau_global", 200.0)
CFG["member5"].setdefault("tail_item_fallback_weight", 0.5)
CFG["member5"].setdefault("tail_quantization", "int8")

# --- Member 7: pure-marginal GLU-MLP -------------------------------------
# A small GLU-MLP on the 14 mean-encoded marginals ONLY (no embeddings,
# no raw metadata). Captures non-linear interactions among the marginals
# that the additive-linear M4 and field-bilinear M6 cannot express.
CFG.setdefault("member7", {})
CFG["member7"].setdefault("enabled", True)
CFG["member7"].setdefault("hid1", 64)
CFG["member7"].setdefault("hid2", 32)
CFG["member7"].setdefault("learning_rate", 1.0e-3)
CFG["member7"].setdefault("weight_decay", 1.0e-5)
CFG["member7"].setdefault("epochs", 40)
CFG["member7"].setdefault("batch_size", 16384)
CFG["member7"].setdefault("early_stopping_patience", 6)
CFG["member7"].setdefault("feat_dropout", 0.10)

# --- Member 8: embeddings GLU-MLP ----------------------------------------
# A learned subject-id embedding concatenated with the full item
# embedding, fed to a GLU-MLP. Collaborative + content; structurally
# distinct from M1's IRT-MLP head. Memory-safe: item embeddings are
# gathered from unique-item storage, never materialised as [N, D].
CFG.setdefault("member8", {})
CFG["member8"].setdefault("enabled", True)
CFG["member8"].setdefault("subj_emb_dim", 32)
CFG["member8"].setdefault("hid1", 256)
CFG["member8"].setdefault("hid2", 128)
CFG["member8"].setdefault("learning_rate", 1.0e-3)
CFG["member8"].setdefault("weight_decay", 1.0e-5)
CFG["member8"].setdefault("epochs", 30)
CFG["member8"].setdefault("batch_size", 16384)
CFG["member8"].setdefault("early_stopping_patience", 5)
CFG["member8"].setdefault("feat_dropout", 0.10)

# --- Item-form features + CoT/item-type interactions for dense members ---
# The hand-engineered item-form features (token_len, has_code, has_latex,
# is_multiple_choice, ...) already feed Member 1. The diversification pass
# (2026-05-31) also broadcasts a z-scored item-form block + condition x item
# interactions (cot_x_*) + item_type one-hots to the SECONDARY dense members
# M2/M6/M8. M4 (tail specialist) and M7 (pure marginals) stay narrow on
# purpose -- feeding them the same block would just raise their correlation
# with M2/M6/M8, working against NCL.
CFG.setdefault("dense_item_features", {})
CFG["dense_item_features"].setdefault("enabled", True)
CFG["dense_item_features"].setdefault("members", ("member2", "member6", "member8"))
CFG["dense_item_features"].setdefault("add_item_type", True)
CFG["dense_item_features"].setdefault("add_cot_interactions", True)

# --- Negative-correlation learning (NCL) ---------------------------------
# Adds a per-batch decorrelation penalty
#     lambda * mean((p - y) * (p_anchor - y))
# to the penalized members so they learn to be right where the strong
# anchors are wrong. The anchor target is the mean of the GLOBAL anchors'
# train-row predictions -- a regulariser on TRAIN rows only, so the OOF
# predictions the stacker consumes stay honest. Master switch: set
# enabled=False to revert to the plain (non-NCL) members instantly.
CFG.setdefault("ncl", {})
CFG["ncl"].setdefault("enabled", True)
CFG["ncl"].setdefault("lambda_", 0.30)
CFG["ncl"].setdefault("anchors", ("member1", "member6", "member8"))
CFG["ncl"].setdefault("penalized", ("member2", "member4", "member7"))
# Member 9 (FwFM clone) decorrelates from this (smaller) anchor set so it
# diverges from M1/M8 while its twin M6 stays a pure anchor.
CFG["ncl"].setdefault("m9_anchors", ("member1", "member8"))

# --- Member 9: FwFM clone trained with NCL -------------------------------
# Same architecture / dense block as Member 6 (FwFM on the embedding-vs-
# marginal block) but trained with the NCL penalty against {M1, M8}. Gives
# the ensemble a "diversified" factorization machine alongside the pure M6.
CFG.setdefault("member9", {})
CFG["member9"].setdefault("enabled", True)
CFG["member9"].setdefault("ncl_lambda", 0.50)

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
# ## 6-pre. Shared item-form / interaction block + NCL config
#
# The diversification+NCL pass (2026-05-31) broadcasts a small dense block
# of item-form features (z-scored text-pool features) + item-type one-hots +
# condition x item interactions (cot_x_*) to the secondary dense members
# M2/M6/M8. M6 already has the raw pool features via ``member_feat_schema``,
# so it only takes the NEW columns (item_type + cot_x_*); M2/M8 take the
# full block. Everything here is item-or-row level and leakage-free.

# %%
from src.item_features import (
    COT_INTERACTION_BASE as _COT_BASE,
    ITEM_TYPE_NAMES as _ITEM_TYPE_NAMES,
    build_cot_interactions as _build_cot_interactions,
    cot_interaction_names as _cot_interaction_names,
    is_cot_from_condition as _is_cot_from_condition,
    item_type_onehot as _item_type_onehot,
)

# Text-form (non-centroid) pool columns, z-scored, keyed by item_key.
_FORM_COLS = [c for c in combined_cols if not c.startswith("centroid_dist_")]
_pool_z_by_item = pool_features_z.set_index("item_key")
_pool_raw_by_item = pool_df.set_index("item_key")

# Per-item item-type one-hot (depends only on the raw form features).
_item_type_by_item: dict[str, dict[str, float]] = {
    str(ik): _item_type_onehot(row.to_dict())
    for ik, row in _pool_raw_by_item.iterrows()
}

DENSE_FORM_FULL_NAMES = (
    tuple(_FORM_COLS) + tuple(_ITEM_TYPE_NAMES) + tuple(_cot_interaction_names())
)
DENSE_FORM_EXTRA_NAMES = tuple(_ITEM_TYPE_NAMES) + tuple(_cot_interaction_names())


def build_item_form_block(item_keys, conditions, *, full: bool):
    """Return ``(matrix [N, F], names)`` for the item-form interaction block.

    ``full=True``  -> [pool_z form | item_type one-hot | cot_x_* interactions]
    ``full=False`` -> [item_type one-hot | cot_x_* interactions]  (M6: pool
                      features already present via member_feat_schema).
    """
    ik = [str(k) for k in item_keys]
    n = len(ik)
    pz = _pool_z_by_item.reindex(ik)[_FORM_COLS].to_numpy(dtype=np.float32)
    pz = np.nan_to_num(pz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tnames = list(_ITEM_TYPE_NAMES)
    tmat = np.zeros((n, len(tnames)), dtype=np.float32)
    for i, k in enumerate(ik):
        t = _item_type_by_item.get(k)
        if t is not None:
            for j, tn in enumerate(tnames):
                tmat[i, j] = t[tn]
    base = np.concatenate([pz, tmat], axis=1).astype(np.float32)
    base_names = list(_FORM_COLS) + tnames
    is_cot = np.fromiter(
        (_is_cot_from_condition(c) for c in conditions),
        count=n, dtype=np.float32,
    )
    inter, inter_names = _build_cot_interactions(base, base_names, is_cot)
    if full:
        mat = np.concatenate([pz, tmat, inter], axis=1).astype(np.float32)
        names = tuple(_FORM_COLS) + tuple(tnames) + tuple(inter_names)
    else:
        mat = np.concatenate([tmat, inter], axis=1).astype(np.float32)
        names = tuple(tnames) + tuple(inter_names)
    return mat, names


# Per-row item_key / condition arrays for the global train & val splits.
_train_row_item_keys_g = primary.train["item_key"].astype(str).to_numpy()
_val_row_item_keys_g = primary.val["item_key"].astype(str).to_numpy()
_train_row_conditions_g = primary.train["condition"].astype(str).to_numpy()
_val_row_conditions_g = primary.val["condition"].astype(str).to_numpy()

_DENSE_ITEM_CFG = CFG.get("dense_item_features", {})
_DENSE_ITEM_ENABLED = bool(_DENSE_ITEM_CFG.get("enabled", False))
_DENSE_ITEM_MEMBERS = tuple(_DENSE_ITEM_CFG.get("members", ()))


def _dense_form_on(member_key: str) -> bool:
    return _DENSE_ITEM_ENABLED and (member_key in _DENSE_ITEM_MEMBERS)


# --- NCL config (master switch + per-member lambda resolver) ---
_NCL_CFG = CFG.get("ncl", {})
_NCL_ENABLED = bool(_NCL_CFG.get("enabled", False))
_NCL_LAMBDA = float(_NCL_CFG.get("lambda_", 0.0))
_NCL_ANCHORS = tuple(_NCL_CFG.get("anchors", ()))
_NCL_PENALIZED = tuple(_NCL_CFG.get("penalized", ()))
_NCL_M9_ANCHORS = tuple(_NCL_CFG.get("m9_anchors", ()))


def _ncl_lambda_for(member_key: str) -> float:
    """Penalty strength for a member (0.0 disables the NCL term)."""
    if _NCL_ENABLED and member_key in _NCL_PENALIZED:
        return _NCL_LAMBDA
    return 0.0


print(
    f"[dense item features] enabled={_DENSE_ITEM_ENABLED} members={_DENSE_ITEM_MEMBERS} "
    f"full_block={len(DENSE_FORM_FULL_NAMES)} extra_block={len(DENSE_FORM_EXTRA_NAMES)}"
)
print(
    f"[NCL] enabled={_NCL_ENABLED} lambda={_NCL_LAMBDA} "
    f"anchors={_NCL_ANCHORS} penalized={_NCL_PENALIZED} m9_anchors={_NCL_M9_ANCHORS}"
)

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
# ## 6.7. SOLVER-PROXY PROBE (orthogonal-signal go/no-go)
#
# Inserted by the probe copy. Everything the probe needs is already built:
# `primary.train` (item text + labels), `nn_train_mat` (the NN pass-rate
# features = our cheap "current signal" baseline), and `folds` (honest
# item-cold OOF). We:
#
#   1. fit an out-of-fold logistic on `nn_train_mat` -> `p_base` (the signal
#      we must beat),
#   2. sample a few hundred TRAIN items, stratified by NN-support,
#   3. run ALL solver models (multi-model by default), querying each model
#      **3 times** per item, and distil two kinds of item-side scalars:
#        - **per-model self-consistency** (`selfcons::<model>`, plus the
#          `selfcons::mean` across models): does each model agree *with
#          itself* across its 3 samples -- a within-model difficulty signal;
#        - **cross-model disagreement** (`disagreement`): do *different*
#          models give different answers -- an estimate of item
#          *discrimination* a_i;
#        - plus reference single-model columns (`answer_entropy`, `p_true`,
#          `mean_trace_len`) from the primary model,
#   4. measure held-out **ΔNLL** of EACH proxy on top of `p_base` with an
#      item-clustered bootstrap CI, sliced by NN-support quartile -- i.e.
#      whether self-consistency and/or disagreement generate real signal.
#
# A proxy "passes" only if its bootstrap CI for ΔNLL is fully below 0 in at
# least one slice -- ideally the low-support (cold) quartile. By default we
# `raise SystemExit` after printing the verdict so the expensive 9-member
# pipeline below never runs; set `CFG["proxy_probe"]["stop_after"]=False` to
# fall through.

# %%
import sys as _sys

import numpy as _np_probe

from src.proxy_eval import (
    fit_nn_baseline_oof,
    format_probe_report,
    run_proxy_probe,
)
from src.solver_proxy import (
    SolverProxy,
    SolverProxyConfig,
    build_proxy_row_vector,
    cross_model_disagreement,
)

# --- Probe config (self-contained; override in CFG if you like) -------------
_PP = CFG.setdefault("proxy_probe", {})
_PP.setdefault("enabled", True)
_PP.setdefault("stop_after", True)          # early-exit before heavy training
_PP.setdefault("n_probe_items", 400)        # distinct TRAIN items to solve
_PP.setdefault("n_samples", 3)              # CoT samples per item PER model
_PP.setdefault("max_new_tokens", 512)
_PP.setdefault("temperature", 0.8)
_PP.setdefault("batch_size", 16)
_PP.setdefault("compute_p_true", True)
_PP.setdefault("use_chat_template", True)   # instruct models -> emit EOS, finish early (fast)
_PP.setdefault("enable_thinking", False)    # Qwen3 etc.: skip long <think> traces
_PP.setdefault("chunk_items", 32)           # items per cache flush (visible/resumable progress)
_PP.setdefault("multi_model", True)         # run ALL models for disagreement + per-model consistency
_PP.setdefault("n_boot", 500)
_PP.setdefault("support_col", 7)            # nn_train_mat col 7 = n_labeled_neighbors_log1p
# Extra single-model columns (from the primary model) to also evaluate.
_PP.setdefault("proxy_cols", ["answer_entropy", "p_true", "mean_trace_len"])
# Each entry is either a plain HF id (str) or a dict with the id plus
# per-model overrides: trust_remote_code / system_prompt / enable_thinking.
_PP.setdefault("models", [
    # NVIDIA Llama-Nemotron: ungated; reasoning toggled via system prompt.
    {
        "model_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "system_prompt": "detailed thinking off",
    },
    # LG AI EXAONE 3.5: ungated but needs trust_remote_code for custom code.
    {
        "model_id": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "trust_remote_code": True,
    },
    # Mistral (third slot): ungated Apache-2.0 7B instruct (Ministral-8B and
    # Mistral-Small/Magistral are gated and/or too large for the probe GPU).
    "mistralai/Mistral-7B-Instruct-v0.3",
])

if not _PP["enabled"]:
    print("[proxy probe] disabled via CFG['proxy_probe']['enabled']=False; skipping.")
else:
    # 1) Honest item-cold NN baseline = the "current signal" we must beat.
    _y_train = primary.train["label"].astype(float).to_numpy()
    _item_key_rows = primary.train["item_key"].astype(str).to_numpy()
    _tr_idx = [f.train_row_idx for f in folds]
    _oof_idx = [f.oof_row_idx for f in folds]
    print(f"[proxy probe] fitting OOF NN baseline on {nn_train_mat.shape} ...")
    _p_base = fit_nn_baseline_oof(nn_train_mat, _y_train, _tr_idx, _oof_idx, l2=5.0)
    from src.proxy_eval import bce as _bce_probe
    print(
        f"[proxy probe] baseline OOF NLL={_bce_probe(_y_train, _p_base):.5f}  "
        f"(const={_bce_probe(_y_train, _np_probe.full_like(_y_train, _y_train.mean())):.5f})"
    )

    # Row-level NN support (lower == colder); aggregate to item for sampling.
    _support_row = nn_train_mat[:, int(_PP["support_col"])].astype(_np_probe.float64)
    _df_supp = (
        primary.train[["item_key"]].assign(_s=_support_row, _ik=_item_key_rows)
        .groupby("_ik")["_s"].mean()
    )
    _all_items = _df_supp.index.to_numpy()
    # Stratified sample by support quartile so the cold slice is represented.
    _rng = _np_probe.random.default_rng(int(CFG.get("seed", 0)))
    _n_take = min(int(_PP["n_probe_items"]), len(_all_items))
    _q = _np_probe.quantile(_df_supp.to_numpy(), [0.25, 0.5, 0.75])
    _bucket = _np_probe.digitize(_df_supp.to_numpy(), _q)
    _per_bucket = max(1, _n_take // 4)
    _picked = []
    for _b in range(4):
        _cand = _all_items[_bucket == _b]
        if len(_cand):
            _picked.append(_rng.choice(_cand, size=min(_per_bucket, len(_cand)), replace=False))
    _sample_items = set(_np_probe.concatenate(_picked).tolist()) if _picked else set()
    print(f"[proxy probe] sampled {len(_sample_items)} TRAIN items (stratified by NN support)")

    # 2) Build the per-item solve rows (first row per sampled item).
    _sample_rows = (
        primary.train[primary.train["item_key"].astype(str).isin(_sample_items)]
        [["item_key", "benchmark", "condition", "item_content"]]
        .drop_duplicates(subset=["item_key"]).reset_index(drop=True)
    )

    # 3) Run EVERY solver model (multi-model by default). For each model we
    #    keep (a) its per-item self_consistency -- "is this model stable on
    #    this item by itself" -- and (b) its modal answer, which feeds the
    #    cross-model disagreement -- "do DIFFERENT models agree on the answer".
    #    (a) is a within-model difficulty signal; (b) estimates discrimination.
    _models = list(_PP["models"]) if bool(_PP["multi_model"]) else list(_PP["models"])[:1]

    def _normalize_model_entry(_e):
        """Accept a plain id or a dict with per-model overrides."""
        if isinstance(_e, str):
            return {"model_id": _e}
        _d = dict(_e)
        if "model_id" not in _d:
            raise ValueError(f"model entry missing 'model_id': {_e!r}")
        return _d

    _model_dfs: dict[str, "pd.DataFrame"] = {}
    _modal_by_model: dict[str, dict[str, str]] = {}
    for _mi, _entry in enumerate(_models):
        _spec = _normalize_model_entry(_entry)
        _mid = _spec["model_id"]
        _cfg_m = SolverProxyConfig(
            model_id=_mid,
            n_samples=int(_PP["n_samples"]),
            max_new_tokens=int(_PP["max_new_tokens"]),
            temperature=float(_PP["temperature"]),
            batch_size=int(_PP["batch_size"]),
            # P(True) only on the primary model (cheap extra; others skip it).
            compute_p_true=bool(_PP["compute_p_true"]) and (_mi == 0),
            use_chat_template=bool(_PP["use_chat_template"]),
            # Per-model overrides (fall back to the global probe defaults).
            enable_thinking=bool(_spec.get("enable_thinking", _PP["enable_thinking"])),
            trust_remote_code=bool(_spec.get("trust_remote_code", False)),
            system_prompt=str(_spec.get("system_prompt", "")),
            chunk_items=int(_PP["chunk_items"]),
            cache_dir=str(ROOT / "artifacts" / "solver_proxy"),
        )
        print(
            f"[proxy probe] solving {len(_sample_rows)} items with "
            f"{_mid} (n_samples={_cfg_m.n_samples}) [{_mi + 1}/{len(_models)}] ..."
        )
        with SolverProxy(_cfg_m) as _spm:
            _df_m = _spm.score_items(_sample_rows, progress=True)
        _model_dfs[_mid] = _df_m
        _modal_by_model[_mid] = dict(
            zip(_df_m["item_key"].astype(str), _df_m["modal_answer"].astype(str))
        )

    _primary_id = _normalize_model_entry(_models[0])["model_id"]
    _proxy_df = _model_dfs[_primary_id]

    # ---- Assemble the candidate proxies -----------------------------------
    _proxy_per_item: dict[str, dict[str, float]] = {}

    # (a) Each model's OWN self-consistency (modal-vote share across its 3 samples).
    _selfcons_frames = []
    for _mid, _df_m in _model_dfs.items():
        _short = _mid.split("/")[-1]
        _sc = dict(zip(_df_m["item_key"].astype(str), _df_m["self_consistency"].astype(float)))
        _proxy_per_item[f"selfcons::{_short}"] = _sc
        _selfcons_frames.append(
            _df_m[["item_key", "self_consistency"]].rename(
                columns={"self_consistency": _short}
            )
        )
    # Mean self-consistency across models (robust within-model difficulty).
    _sc_merged = _selfcons_frames[0]
    for _f in _selfcons_frames[1:]:
        _sc_merged = _sc_merged.merge(_f, on="item_key", how="outer")
    _sc_cols = [c for c in _sc_merged.columns if c != "item_key"]
    _sc_mean = _sc_merged.set_index("item_key")[_sc_cols].mean(axis=1)
    _proxy_per_item["selfcons::mean"] = _sc_mean.to_dict()

    # (b) Cross-model DISAGREEMENT (estimates item discrimination a_i).
    if len(_models) > 1:
        _proxy_per_item["disagreement"] = cross_model_disagreement(_modal_by_model)

    # (c) Extra single-model columns from the primary model, for reference.
    for _col in list(_PP["proxy_cols"]):
        if _col in _proxy_df.columns:
            _proxy_per_item[f"{_col}::{_primary_id.split('/')[-1]}"] = dict(
                zip(_proxy_df["item_key"].astype(str), _proxy_df[_col].astype(float))
            )

    # ---- Cross-model descriptive summary (before the signal test) ---------
    print("\n[proxy probe] cross-model summary on sampled items:")
    for _mid, _df_m in _model_dfs.items():
        _scv = _df_m["self_consistency"].astype(float).to_numpy()
        _rev = _df_m["refusal_rate"].astype(float).to_numpy()
        print(
            f"  {_mid.split('/')[-1]:<22} self_consistency mean={_np_probe.nanmean(_scv):.3f} "
            f"(frac fully-consistent={_np_probe.mean(_scv >= 0.999):.3f})  "
            f"refusal mean={_np_probe.nanmean(_rev):.3f}"
        )
    if "disagreement" in _proxy_per_item:
        _dis_vals = _np_probe.array(
            [v for v in _proxy_per_item["disagreement"].values() if _np_probe.isfinite(v)]
        )
        print(
            f"  cross-model DISAGREEMENT mean={_dis_vals.mean():.3f}  "
            f"(frac all-agree={_np_probe.mean(_dis_vals <= 1e-9):.3f}, "
            f"frac all-differ={_np_probe.mean(_dis_vals >= 0.66):.3f}, n={_dis_vals.size})"
        )
        # Sanity: disagreement should anti-correlate with mean self-consistency.
        _keys = [k for k in _proxy_per_item["disagreement"]
                 if _np_probe.isfinite(_proxy_per_item["disagreement"][k])
                 and k in _proxy_per_item["selfcons::mean"]]
        if len(_keys) > 10:
            _da = _np_probe.array([_proxy_per_item["disagreement"][k] for k in _keys])
            _sc = _np_probe.array([_proxy_per_item["selfcons::mean"][k] for k in _keys])
            if _da.std() > 1e-9 and _sc.std() > 1e-9:
                print(
                    f"  corr(disagreement, mean self_consistency) = "
                    f"{_np_probe.corrcoef(_da, _sc)[0, 1]:+.3f}  (expect negative)"
                )

    # 4) Restrict rows to sampled items; run the incremental-NLL probe per proxy.
    _row_in_sample = _np_probe.fromiter(
        (k in _sample_items for k in _item_key_rows),
        count=len(_item_key_rows), dtype=bool,
    )
    _ri = _np_probe.where(_row_in_sample)[0]
    print(
        f"[proxy probe] evaluating on {_ri.size:,} rows over {len(_sample_items)} items\n"
        + "=" * 78
    )
    _probe_summary: dict[str, dict] = {}
    for _col, _per_item in _proxy_per_item.items():
        # Mean-fill missing items so a few solver failures don't drop rows.
        _vals = _np_probe.array([v for v in _per_item.values() if _np_probe.isfinite(v)])
        _fill = float(_vals.mean()) if _vals.size else 0.0
        _z_rows = build_proxy_row_vector(_item_key_rows[_ri], _per_item, fill=_fill)
        _results = run_proxy_probe(
            p_base=_p_base[_ri],
            z_per_row=_z_rows,
            labels=_y_train[_ri],
            item_key_per_row=_item_key_rows[_ri],
            support_per_row=_support_row[_ri],
            n_boot=int(_PP["n_boot"]),
            seed=int(CFG.get("seed", 0)),
        )
        print(format_probe_report(_results, title=f"proxy={_col}"))
        print("-" * 78)
        _probe_summary[_col] = {k: v.delta_nll for k, v in _results.items()}

    print("=" * 78)
    print("[proxy probe] DONE. Per-proxy ΔNLL (all-slice):")
    for _col, _d in _probe_summary.items():
        print(f"  {_col:<16} ΔNLL_all={_d.get('all', float('nan')):+.5f}")

    if bool(_PP["stop_after"]):
        print(
            "\n[proxy probe] stop_after=True -> halting before the 9-member "
            "pipeline. Set CFG['proxy_probe']['stop_after']=False to continue."
        )
        raise SystemExit(0)

# %% [markdown]
# ## 7. Build training tensors + metadata artifacts

# %%
from src.data import prepare_metadata_artifacts
from src.embeddings import index_embeddings
from src.item_features import build_feature_matrix
from src.metadata_features import MetadataSchema
from src.models import IndexedEmbeddingView, LookupDataset


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
    """Build a ``LookupDataset`` for ``split_part`` using an
    :class:`IndexedEmbeddingView` for the item embeddings.

    The previous implementation called ``stack_lookup`` to
    materialise a per-row ``[N, 4096]`` float32 tensor. On the
    full M-train split (~5,095,197 rows, 4,096-dim Qwen3 vectors,
    over ~295,671 unique items) that single tensor is ~80 GB --
    enough to OOM the box on its own, and it makes the
    downstream Model A training + member fits (which themselves
    each want ~15-25 GB of CPU RAM) effectively impossible
    without first freeing the dataset. The conditional NN
    feature stage upstream already pushes RSS to ~43 GB; adding
    80 GB on top reliably crashes Colab.

    Replacing the stacked tensor with an ``IndexedEmbeddingView``
    over the unique-item embedding stack drops the per-split
    cost to ``U*D*4 + N*8`` bytes -- roughly 4.85 GB instead of
    80 GB on the train split, 4.85 GB instead of 4.4 GB on val.
    Downstream consumers (``LookupDataset.__getitem__`` for the
    DataLoader and the chunked ``_score_dataset``) only need
    ``view[i]`` / ``view[a:b]``, both of which the view supports
    with at most one ``index_select`` per batch -- negligible
    relative to a model forward pass.
    """
    s = np.array([indexer.subject_id(k) for k in split_part["subject_key"]], dtype=np.int64)
    bc = np.array(
        [indexer.bc_id(k) for k in split_part["benchmark_condition_key"]],
        dtype=np.int64,
    )
    _uniq_ie_np, _row_to_uniq_np = index_embeddings(
        split_part["item_key"].astype(str).tolist(), item_emb_lookup,
    )
    # IndexedEmbeddingView accepts numpy arrays directly and wraps
    # them as zero-copy torch tensors internally, so we don't need
    # torch in scope at the call site (torch is first imported much
    # further down, in cell 7c right before Model A training).
    ie = IndexedEmbeddingView(_uniq_ie_np, _row_to_uniq_np)
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
# ``nn_val_mat`` with the conditional context, and although ``_build``
# now uses :class:`IndexedEmbeddingView` (which keeps only the ~296k
# unique item vectors plus a per-row pointer, ~5 GB per split), building
# both here and in cell 7b would still double-allocate the pool / NN /
# label tensors. The downstream cells (training, scoring) all bind to
# the cell-7b versions, so we skip the throwaway build here.
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
# Why this matters: ``_build`` builds the per-split LookupDataset.
# The item-embedding channel is now wrapped as
# :class:`IndexedEmbeddingView` over the ~296k unique-item Qwen3
# vectors plus a per-row pointer (~5 GB / split, down from the
# ~80 GB per split of the legacy stacked dense ``[N, 4096]``
# tensor), but the pool / NN / label / id tensors still total a
# few hundred MB per split, and the per-split query-metadata
# dicts (``_*_qmeta``) carry ~80 MB on the train split each.
# Dropping the previous ``train_ds`` / ``val_ds`` before
# allocating the new ones still avoids transient peaks on cell
# re-runs.
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
# (metadata MLP, kNN-similarity, logistic regression on top of Model A),
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

# Pre-score Model A on train ONCE here (cached). Even though the
# item-embedding channel is now an :class:`IndexedEmbeddingView`
# (~5 GB per split rather than the legacy ~80 GB stacked tensor),
# the dataset still holds pool / NN / id / label tensors that
# add up to a few GB per split, and the X_train_dense / X_val_dense
# build below wants ~25 GB of fresh CPU RAM. Caching ``p_a_train``
# here lets us drop ``train_ds`` / ``val_ds`` immediately after the
# Model A scoring pass; the downstream calibrator consumes
# ``p_a_train`` directly instead of re-scoring, so this isn't a
# duplicate cost.
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

# Free the train + val LookupDatasets now that Model A has produced
# both train + val predictions. With ``IndexedEmbeddingView`` the
# item-embedding channel is no longer the dominant cost (~5 GB per
# split, down from ~80 GB on the legacy stacked path), but the
# pool / NN / id / label tensors plus the unique-embedding stack
# still add up to a few GB per split. Dropping them now leaves
# headroom for the X_train_dense / X_val_dense build below and
# the downstream member fits. The trained model itself stays
# bound (cheap) in case the calibrator wants to score on
# additional rows; per the cached ``p_a_train`` / ``p_a_val`` we
# don't actually need it any more.
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
# ## 9b-bis. Mean-encoded features for Member 4 (also reused by Member 2)
#
# These cells fit per-cell pass-rate statistics on TRAIN rows and emit
# a compact dense feature matrix:
#
#   * `member4_marginal_train/val` -- 14 mean-encoded marginal columns
#     (subject mean, bc mean, cluster mean, plus a handful of two-way
#     interactions and a constant). These are appended to Member 4's
#     dense feature view, and are also fed directly as the only
#     continuous inputs to the Member 2 metadata MLP (which sees only
#     subject / bc / cluster IDs plus these 14 marginals).
#
# Cells with no training observations fall back through:
# subj_cluster -> cluster_mean -> global_mean, etc. (Bayesian
# shrinkage with `smoothing=30`). Val rows look up the precomputed
# train-time aggregates -- the val labels themselves are NEVER
# consulted at fit time.

# %%
from src.mean_encoded_features import (
    MEMBER4_MARGINAL_FEATURE_DIM,
    MEMBER4_MARGINAL_FEATURE_NAMES,
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
    f"[mean-enc] Member 4 marginal features:    train {member4_marginal_train.shape}  "
    f"val {member4_marginal_val.shape}  cols={MEMBER4_MARGINAL_FEATURE_DIM}"
)

# Augmented dense matrices for Member 2 (interaction features appended
# to the existing 1202-feature schema). Member 2 sees BOTH the original
# embedding-derived features AND the new mean-encoded interactions.
#
# Member 2 uses metadata IDs + M4 marginals only (no dense embedding matrix).
X_train_dense_m2 = None
X_val_dense_m2 = None
print("[mean-enc] Member 2: metadata MLP path (no dense_m2 matrices).")

# %% [markdown]
# ## 9c. Train Member 2 (dense metadata MLP with DCN-v2)
#
# Dense metadata-only MLP. Inputs:
#   * **Categorical** (7 fields, all embedded): subject_id, bc_id,
#     cluster_id, family_id, macro_family_id, organization_id,
#     bench_topic_id.
#   * **Numerical**: subject CSV numerics (log_params + missing-flag,
#     release_date + missing-flag), benchmark numerics (benchmark_age
#     + missing-flag), per-row bc-redacted flag, plus the 14 mean-encoded
#     marginals Member 4 also consumes.
#
# Architecture: parallel DCN-v2 cross tower + GLU deep tower, fused at
# the head. Training tricks: linear-warmup + cosine LR, weight EMA,
# snapshot-ensemble over the top-K best val checkpoints, optional
# numerical-channel Mixup, optional label smoothing. See
# ``src/member2_metadata_mlp.py`` for the full design.
#
# Why no item embeddings: keeps the member structurally orthogonal to
# M1 (deep on embeddings), M3 (kNN on embeddings), M4 (linear on
# embeddings + marginals), and M6 (bilinear on embeddings + marginals).

# %%
from src.member2_metadata_mlp import (
    apply_state_batch as m2_apply_state_batch,
    assemble_numerical as m2_assemble_numerical,
    fit_member2_metadata_mlp,
    numerical_feature_names as m2_numerical_feature_names,
)

# ---- Per-row metadata arrays (shared by global / OOF / shuffled fits) ----
# Build once on the full train/val partitions; per-fold sections just
# slice these by ``fold.train_row_idx`` / ``fold.oof_row_idx`` (or
# rebuild from ``_compute_id_arrays(fold_*_df)`` for the family/macro/
# org/topic/subject_num/bench_num columns since those are pure lookups
# from the global subject_tables / meta_id_tables).

_M2_SUBJECT_CAT_FIELDS = list(_meta_schema.subject_categorical)
_M2_BENCH_CAT_FIELDS = list(_meta_schema.benchmark_categorical)
_M2_SUBJECT_NUM_FIELDS = list(_meta_schema.subject_numeric)
_M2_BENCH_NUM_FIELDS = list(_meta_schema.benchmark_numeric)

_M2_SUBJ_NUM_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for col in _M2_SUBJECT_NUM_FIELDS
    for name in (f"{col}_value", f"{col}_missing")
)
_M2_BENCH_NUM_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for col in _M2_BENCH_NUM_FIELDS
    for name in (f"{col}_value", f"{col}_missing")
)
_M2_N_SUBJ_NUM: int = int(len(_M2_SUBJ_NUM_FEATURE_NAMES))
_M2_N_BENCH_NUM: int = int(len(_M2_BENCH_NUM_FEATURE_NAMES))

# Per-subject_id metadata lookup tables (numpy copies; static across
# the run). For unknown subjects we route to UNK at gather time via
# ``np.where(subj >= 0, ...)``.
_M2_SUBJ_CAT_TABLE = (
    meta_id_tables.subject_cat_ids.cpu().numpy().astype(np.int64)
)              # [n_subjects, n_subject_cat_fields]
_M2_SUBJ_NUM_TABLE = (
    meta_id_tables.subject_num.cpu().numpy().astype(np.float32)
)              # [n_subjects, 2 * n_subject_numeric]

# Per-bc_id metadata lookup tables.
_M2_BC_CAT_TABLE = (
    meta_id_tables.bc_cat_ids.cpu().numpy().astype(np.int64)
)              # [n_bc, n_benchmark_cat_fields]
_M2_BC_NUM_TABLE = (
    meta_id_tables.bc_num.cpu().numpy().astype(np.float32)
)              # [n_bc, 2 * n_benchmark_numeric]

# Index of each cat trait inside the per-subject / per-bc cat tables.
_M2_FAMILY_COL = (
    _M2_SUBJECT_CAT_FIELDS.index("family")
    if "family" in _M2_SUBJECT_CAT_FIELDS else -1
)
_M2_MACRO_COL = (
    _M2_SUBJECT_CAT_FIELDS.index("macro_family")
    if "macro_family" in _M2_SUBJECT_CAT_FIELDS else -1
)
_M2_ORG_COL = (
    _M2_SUBJECT_CAT_FIELDS.index("organization")
    if "organization" in _M2_SUBJECT_CAT_FIELDS else -1
)
_M2_TOPIC_COL = (
    _M2_BENCH_CAT_FIELDS.index("topic")
    if "topic" in _M2_BENCH_CAT_FIELDS else -1
)

# Cardinalities -- include the UNK/MISSING row each vocab reserves.
_M2_N_FAMILIES = int(N_FAMILIES) if _M2_FAMILY_COL >= 0 else 1
_M2_N_MACRO_FAMILIES = int(N_MACRO_FAMILIES) if _M2_MACRO_COL >= 0 else 1
_M2_N_ORGANIZATIONS = int(N_ORGANIZATIONS) if _M2_ORG_COL >= 0 else 1
_M2_N_BENCH_TOPICS = (
    int(meta_id_tables.benchmark_cat_cardinalities[_M2_TOPIC_COL])
    if (_M2_TOPIC_COL >= 0 and len(meta_id_tables.benchmark_cat_cardinalities) > _M2_TOPIC_COL)
    else 1
)


def _m2_gather_metadata(subj_ids: np.ndarray, bc_ids: np.ndarray) -> dict:
    """Gather per-row M2 metadata: family/macro/org/topic ids + subj/bench numerics.

    Out-of-range ids (-1 or beyond cardinality) are routed to UNK on
    the categorical side (-1 stays -1; the apply / fit functions
    route -1 to the UNK slot). For numerics we zero out the value and
    set the missing-flag to 1 for cold rows, mirroring how the
    NumericScaler encodes a missing source value.
    """
    s = np.asarray(subj_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    n = int(s.shape[0])
    n_subj = int(_M2_SUBJ_CAT_TABLE.shape[0])
    n_bc = int(_M2_BC_CAT_TABLE.shape[0])
    s_clamped = np.clip(s, 0, max(n_subj - 1, 0)).astype(np.int64)
    b_clamped = np.clip(b, 0, max(n_bc - 1, 0)).astype(np.int64)
    s_valid = (s >= 0) & (s < n_subj)
    b_valid = (b >= 0) & (b < n_bc)

    # Subject categoricals.
    if _M2_FAMILY_COL >= 0:
        family = np.where(
            s_valid, _M2_SUBJ_CAT_TABLE[s_clamped, _M2_FAMILY_COL], -1
        ).astype(np.int64)
    else:
        family = np.full(n, -1, dtype=np.int64)
    if _M2_MACRO_COL >= 0:
        macro = np.where(
            s_valid, _M2_SUBJ_CAT_TABLE[s_clamped, _M2_MACRO_COL], -1
        ).astype(np.int64)
    else:
        macro = np.full(n, -1, dtype=np.int64)
    if _M2_ORG_COL >= 0:
        organization = np.where(
            s_valid, _M2_SUBJ_CAT_TABLE[s_clamped, _M2_ORG_COL], -1
        ).astype(np.int64)
    else:
        organization = np.full(n, -1, dtype=np.int64)
    # Benchmark categorical.
    if _M2_TOPIC_COL >= 0:
        topic = np.where(
            b_valid, _M2_BC_CAT_TABLE[b_clamped, _M2_TOPIC_COL], -1
        ).astype(np.int64)
    else:
        topic = np.full(n, -1, dtype=np.int64)

    # Subject numerics (and missing-flag for cold subjects).
    if _M2_N_SUBJ_NUM > 0:
        subj_num = _M2_SUBJ_NUM_TABLE[s_clamped].astype(np.float32, copy=True)
        cold = ~s_valid
        if cold.any():
            for j in range(int(len(_M2_SUBJECT_NUM_FIELDS))):
                subj_num[cold, 2 * j] = 0.0
                subj_num[cold, 2 * j + 1] = 1.0
    else:
        subj_num = np.zeros((n, 0), dtype=np.float32)

    # Benchmark numerics.
    if _M2_N_BENCH_NUM > 0:
        bench_num = _M2_BC_NUM_TABLE[b_clamped].astype(np.float32, copy=True)
        cold = ~b_valid
        if cold.any():
            for j in range(int(len(_M2_BENCH_NUM_FIELDS))):
                bench_num[cold, 2 * j] = 0.0
                bench_num[cold, 2 * j + 1] = 1.0
    else:
        bench_num = np.zeros((n, 0), dtype=np.float32)

    return {
        "family_ids": family,
        "macro_family_ids": macro,
        "organization_ids": organization,
        "bench_topic_ids": topic,
        "subject_numerical": subj_num,
        "bench_numerical": bench_num,
    }


# ---- Resolve metadata-only vs full feature composition --------------------
# CFG["member2_mlp"]["feature_mode"] controls whether M2 sees only the
# pure metadata block (default after Gate-3e diagnostic) or the full
# legacy v2_dcnv2 block (subject/bc/cluster embeddings + marginals +
# bc_redacted_flag). The same toggle has to apply to both the global
# fit and every OOF fold; we resolve it once here so the two callers
# can't drift apart.
_M2_FEATURE_MODE = str(CFG.get("member2_mlp", {}).get("feature_mode", "metadata_cluster"))
if _M2_FEATURE_MODE not in {"metadata_only", "metadata_cluster", "full"}:
    raise ValueError(
        f"CFG['member2_mlp']['feature_mode'] must be 'metadata_only', "
        f"'metadata_cluster' or 'full', got {_M2_FEATURE_MODE!r}"
    )
# Three composition modes, expressed as three orthogonal channel gates
# so the global / OOF / shuffle-null fits can't drift:
#   metadata_only    -> pure subject/bench metadata block (no ids, no
#                       cluster, no marginals).
#   metadata_cluster -> metadata block + item-cluster embedding (still
#                       no subject/bc id embeddings, no marginals).
#   full             -> everything (subject/bc/cluster ids + marginals).
_M2_USE_SUBJ_BC = (_M2_FEATURE_MODE == "full")
_M2_USE_CLUSTER = (_M2_FEATURE_MODE in {"metadata_cluster", "full"})
_M2_USE_MARGINALS = (_M2_FEATURE_MODE == "full")
# Back-compat alias still consulted by the marginal/redact-zeroing
# branches below (now keyed on "are marginals active?", which is the
# only thing those branches actually care about).
_M2_METADATA_ONLY = not _M2_USE_MARGINALS
# When marginals are inactive the numerical block excludes them so
# _M2_NUM_FEATURE_NAMES has to match (its length is part of the state
# contract via num_feature_names).
if not _M2_USE_MARGINALS:
    _M2_NUM_FEATURE_NAMES = m2_numerical_feature_names(
        subj_num_names=_M2_SUBJ_NUM_FEATURE_NAMES,
        bench_num_names=_M2_BENCH_NUM_FEATURE_NAMES,
        marginal_names=(),
    )
else:
    _M2_NUM_FEATURE_NAMES = m2_numerical_feature_names(
        subj_num_names=_M2_SUBJ_NUM_FEATURE_NAMES,
        bench_num_names=_M2_BENCH_NUM_FEATURE_NAMES,
        marginal_names=MEMBER4_MARGINAL_FEATURE_NAMES,
    )

# Per-row train / val metadata (built once at module scope).
_m2_meta_train = _m2_gather_metadata(_mef_train_subj, _mef_train_bc)
_m2_meta_val = _m2_gather_metadata(_mef_val_subj, _mef_val_bc)
# In metadata_only mode every per-row marginal is replaced with an
# empty [N, 0] slab so the assembled numerical channel becomes
# subj_num | bench_num | bc_redacted(=0) | <no marginals>.  The
# bc_redacted column itself is still emitted (the state's numerical
# block-sum hard-codes a "+1 redact" slot, and runtime always ships
# it as 0.0); we just clamp it to zero on every row so the model
# can't anchor to a synthetic train-time mask.
if _M2_METADATA_ONLY:
    _N_train_rows = int(_m2_meta_train["subject_numerical"].shape[0])
    _N_val_rows = int(_m2_meta_val["subject_numerical"].shape[0])
    _m2_marginal_train_active = np.zeros((_N_train_rows, 0), dtype=np.float32)
    _m2_marginal_val_active = np.zeros((_N_val_rows, 0), dtype=np.float32)
    _m2_bc_redact_train_active = np.zeros(_N_train_rows, dtype=np.float32)
    _m2_bc_redact_val_active = np.zeros(_N_val_rows, dtype=np.float32)
    _M2_N_MARGINALS_ACTIVE = 0
else:
    _m2_marginal_train_active = member4_marginal_train
    _m2_marginal_val_active = member4_marginal_val
    _m2_bc_redact_train_active = bc_redacted_train
    _m2_bc_redact_val_active = bc_redacted_val
    _M2_N_MARGINALS_ACTIVE = int(MEMBER4_MARGINAL_FEATURE_DIM)

# Broadcast the item-form + CoT/item-type interaction block into M2's
# numerical channel. It rides in the "marginals" extension slot (the
# trainer's block_sum contract treats anything past subj/bench/redact as
# marginals), so no schema special-casing is required.
_M2_FORM_NAMES: tuple[str, ...] = ()
if _dense_form_on("member2"):
    _m2_form_train, _M2_FORM_NAMES = build_item_form_block(
        _train_row_item_keys_g, _train_row_conditions_g, full=True,
    )
    _m2_form_val, _ = build_item_form_block(
        _val_row_item_keys_g, _val_row_conditions_g, full=True,
    )
    _m2_marginal_train_active = np.concatenate(
        [_m2_marginal_train_active, _m2_form_train], axis=1,
    ).astype(np.float32, copy=False)
    _m2_marginal_val_active = np.concatenate(
        [_m2_marginal_val_active, _m2_form_val], axis=1,
    ).astype(np.float32, copy=False)
    _m2_active_marginal_names = (
        (tuple(MEMBER4_MARGINAL_FEATURE_NAMES) if _M2_USE_MARGINALS else ())
        + tuple(_M2_FORM_NAMES)
    )
    _M2_N_MARGINALS_ACTIVE = int(_m2_marginal_train_active.shape[1])
    _M2_NUM_FEATURE_NAMES = m2_numerical_feature_names(
        subj_num_names=_M2_SUBJ_NUM_FEATURE_NAMES,
        bench_num_names=_M2_BENCH_NUM_FEATURE_NAMES,
        marginal_names=_m2_active_marginal_names,
    )
    print(
        f"[Member 2 MLP] + item-form block: {len(_M2_FORM_NAMES)} cols "
        f"-> n_marginals_active={_M2_N_MARGINALS_ACTIVE}"
    )

m2_numerical_train = m2_assemble_numerical(
    subject_numerical=_m2_meta_train["subject_numerical"],
    bench_numerical=_m2_meta_train["bench_numerical"],
    bc_redacted_flag=_m2_bc_redact_train_active,
    marginals=_m2_marginal_train_active,
)
m2_numerical_val = m2_assemble_numerical(
    subject_numerical=_m2_meta_val["subject_numerical"],
    bench_numerical=_m2_meta_val["bench_numerical"],
    bc_redacted_flag=_m2_bc_redact_val_active,
    marginals=_m2_marginal_val_active,
)
print(
    f"[Member 2 MLP] feature_mode={_M2_FEATURE_MODE!r}; "
    f"dense metadata channel built: "
    f"n_num={m2_numerical_train.shape[1]} "
    f"(subj={_M2_N_SUBJ_NUM}, bench={_M2_N_BENCH_NUM}, "
    f"redact=1, marg={_M2_N_MARGINALS_ACTIVE})"
)
print(
    f"[Member 2 MLP] cat cardinalities: "
    f"subjects={int(indexer.n_subjects)}, bcs={int(indexer.n_bc)}, "
    f"clusters=?, families={_M2_N_FAMILIES}, "
    f"macro_families={_M2_N_MACRO_FAMILIES}, "
    f"organizations={_M2_N_ORGANIZATIONS}, "
    f"bench_topics={_M2_N_BENCH_TOPICS}"
)

_m2_cfg = CFG.get("member2_mlp", {})
_bc_keys_ordered = tuple(f"bc_{i}" for i in range(int(indexer.n_bc)))
# Provenance keys in indexer id order (Member 3 kNN reuses this list).
_subject_keys_ordered = [
    k for k, _ in sorted(indexer.subject_to_id.items(), key=lambda kv: kv[1])
]
assert len(_subject_keys_ordered) == indexer.n_subjects

_item_to_train_idx = {str(k): i for i, k in enumerate(train_item_keys)}
m2_holdout_item_id = np.fromiter(
    (
        _item_to_train_idx.get(str(k), -1)
        for k in primary.train["item_key"].astype(str).tolist()
    ),
    count=len(primary.train),
    dtype=np.int64,
)
print(
    f"[Member 2 MLP] item-cold holdout groups: "
    f"{int(np.unique(m2_holdout_item_id).size):,} unique items"
)
# Shared item-cold holdout id (Members 4/6 early stopping use the same split).
holdout_item_id = m2_holdout_item_id

_m2_n_clusters = max(
    int(_N_CLUSTERS_ME),
    int(_mef_train_cluster.max()) + 1 if _mef_train_cluster.size else 0,
    int(_mef_val_cluster.max()) + 1 if _mef_val_cluster.size else 0,
)


def _fit_member2_mlp_global():
    print(
        "[Member 2 MLP] training DCN-v2 + GLU MLP on full train "
        f"(N={len(primary.train):,}, n_num={m2_numerical_train.shape[1]})..."
    )
    return fit_member2_metadata_mlp(
        subject_ids=_mef_train_subj,
        bc_ids=_mef_train_bc,
        cluster_ids=_mef_train_cluster,
        family_ids=_m2_meta_train["family_ids"],
        macro_family_ids=_m2_meta_train["macro_family_ids"],
        organization_ids=_m2_meta_train["organization_ids"],
        bench_topic_ids=_m2_meta_train["bench_topic_ids"],
        numerical=m2_numerical_train,
        y=y_train,
        subject_keys=_subject_keys_ordered,
        bc_keys=_bc_keys_ordered,
        num_feature_names=_M2_NUM_FEATURE_NAMES,
        n_subjects=int(indexer.n_subjects),
        n_bcs=int(indexer.n_bc),
        n_clusters=int(_m2_n_clusters),
        n_families=int(_M2_N_FAMILIES),
        n_macro_families=int(_M2_N_MACRO_FAMILIES),
        n_organizations=int(_M2_N_ORGANIZATIONS),
        n_bench_topics=int(_M2_N_BENCH_TOPICS),
        n_subj_num=int(_M2_N_SUBJ_NUM),
        n_bench_num=int(_M2_N_BENCH_NUM),
        n_marginals=int(_M2_N_MARGINALS_ACTIVE),
        d_subj=(int(_m2_cfg.get("d_subj", 32)) if _M2_USE_SUBJ_BC else 0),
        d_bc=(int(_m2_cfg.get("d_bc", 32)) if _M2_USE_SUBJ_BC else 0),
        d_cluster=(int(_m2_cfg.get("d_cluster", 16)) if _M2_USE_CLUSTER else 0),
        d_family=int(_m2_cfg.get("d_family", 16)),
        d_macro=int(_m2_cfg.get("d_macro", 8)),
        d_org=int(_m2_cfg.get("d_org", 16)),
        d_topic=int(_m2_cfg.get("d_topic", 16)),
        hid1=int(_m2_cfg.get("hid1", 256)),
        hid2=int(_m2_cfg.get("hid2", 128)),
        n_cross_layers=int(_m2_cfg.get("n_cross_layers", 2)),
        cross_rank=int(_m2_cfg.get("cross_rank", 64)),
        learning_rate=float(_m2_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(_m2_cfg.get("weight_decay", 1.0e-5)),
        epochs=int(_m2_cfg.get("epochs", 40)),
        batch_size=int(_m2_cfg.get("batch_size", 16384)),
        val_fraction=float(_m2_cfg.get("val_fraction", 0.1)),
        early_stopping_patience=int(_m2_cfg.get("early_stopping_patience", 5)),
        # In metadata_only mode the subject/bc/cluster embeddings have
        # width 0 so their cat-dropout knobs are no-ops; we still pass
        # them through so the state object's saved meta is identical
        # in shape to the "full" mode and the runtime loader doesn't
        # need to special-case the metadata_only bundle.
        cat_dropout_subject=float(_m2_cfg.get("cat_dropout_subject", 0.05)),
        cat_dropout_bc=float(_m2_cfg.get("cat_dropout_bc", 0.10)),
        cat_dropout_cluster=float(_m2_cfg.get("cat_dropout_cluster", 0.10)),
        cat_dropout_family=float(_m2_cfg.get("cat_dropout_family", 0.05)),
        cat_dropout_macro=float(_m2_cfg.get("cat_dropout_macro", 0.05)),
        cat_dropout_org=float(_m2_cfg.get("cat_dropout_org", 0.05)),
        cat_dropout_topic=float(_m2_cfg.get("cat_dropout_topic", 0.10)),
        feat_dropout=float(_m2_cfg.get("feat_dropout", 0.10)),
        warmup_epochs=int(_m2_cfg.get("warmup_epochs", 2)),
        use_cosine_schedule=bool(_m2_cfg.get("use_cosine_schedule", True)),
        ema_decay=float(_m2_cfg.get("ema_decay", 0.999)),
        snapshot_ensemble_k=int(_m2_cfg.get("snapshot_ensemble_k", 3)),
        label_smoothing=float(_m2_cfg.get("label_smoothing", 0.005)),
        mixup_alpha=float(_m2_cfg.get("mixup_alpha", 0.0)),
        seed=int(SEED),
        holdout_group_id=m2_holdout_item_id,
        show_progress=True,
        # Global M2 decorrelates from M1 (p_a_train), the dominant anchor and
        # exactly what the Gate-3e val diagnostic measures. The OOF path uses
        # the full {M1,M6,M8} anchor set for the stacker.
        ncl_anchor_preds=(p_a_train if _ncl_lambda_for("member2") > 0 else None),
        ncl_lambda=_ncl_lambda_for("member2"),
    )


import hashlib as _m2_hashlib


def _content_digest(*arrays, k_rows: int = 4096) -> str:
    """Stable short content digest of one or more numpy arrays.

    Used to make ``cache_or_compute`` keys depend on the actual
    contents (not just shapes) of the inputs to a fit.

    Two complementary fingerprints are mixed in:

    1. A stride-sampled byte hash (~``k_rows`` rows + the head/tail
       64 rows) that catches any change to the "geometric layout"
       of the array even when it spans 100s of MB.

    2. A whole-array reduction (sum, sum-of-squares, abs-sum, min,
       max) cast to float64 / int64. This is what guarantees that
       **any** single-cell modification anywhere in the array
       changes the digest -- the stride sample alone has misses
       for cells that fall between strides. Both pieces together
       give effective collision resistance against any practical
       upstream feature-pipeline drift.

    Cost on the working dataset (5M rows x ~30 cols float32) is
    ~3-4 seconds per array on a single CPU core, dominated by the
    reduction. That is negligible compared to a 4-8 minute M2/M6
    fit, and it is paid once per ``cache_or_compute`` call.
    """
    h = _m2_hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"|None|")
            continue
        ac = np.ascontiguousarray(a)
        h.update(b"|dtype=")
        h.update(str(ac.dtype).encode("ascii"))
        h.update(b"|shape=")
        h.update(str(ac.shape).encode("ascii"))
        if ac.ndim == 0 or ac.size == 0:
            continue
        if ac.shape[0] <= int(k_rows):
            h.update(ac.tobytes())
        else:
            stride = max(int(ac.shape[0]) // int(k_rows), 1)
            h.update(ac[::stride].tobytes())
            h.update(ac[:64].tobytes())
            h.update(ac[-64:].tobytes())
        # Chunked aggregates: peak-memory-safe for huge arrays.
        #
        # The previous implementation did
        # ``ac.astype(np.float64) ** 2`` (and a second
        # ``ac.astype(np.float64)`` for ``abs``), which
        # materialized TWO full-array float64 copies. On
        # ``X_train_dense_m4`` (~24 GB float32) that is ~48 GB
        # *each* in transient -- big enough to host-OOM Colab
        # before ``fit_fwfm_member`` ever started.
        #
        # The chunked path here holds at most
        # ``CHUNK_ROWS * F * 8 * 2`` bytes transient (~1.3 GB
        # at chunk=65536, F=1216) while producing the same five
        # aggregates: sum, sum-of-squares, abs-sum, min, max.
        _CD_CHUNK = 65_536
        n_rows = int(ac.shape[0])
        if ac.dtype.kind == "f":
            s_sum = 0.0
            s_sq = 0.0
            s_abs = 0.0
            mn = float("inf")
            mx = float("-inf")
            for _s in range(0, n_rows, _CD_CHUNK):
                _e = min(_s + _CD_CHUNK, n_rows)
                _chunk = ac[_s:_e].astype(np.float64, copy=False)
                s_sum += float(_chunk.sum())
                s_sq += float((_chunk * _chunk).sum())
                s_abs += float(np.abs(_chunk).sum())
                mn = min(mn, float(_chunk.min()))
                mx = max(mx, float(_chunk.max()))
                _chunk = None  # noqa: F841
            agg = np.asarray([s_sum, s_sq, s_abs, mn, mx], dtype=np.float64)
        else:
            i_sum = 0
            i_sq = 0
            i_abs = 0
            i_mn = int(ac.ravel()[0])
            i_mx = int(ac.ravel()[0])
            for _s in range(0, n_rows, _CD_CHUNK):
                _e = min(_s + _CD_CHUNK, n_rows)
                _chunk = ac[_s:_e]
                _c64 = _chunk.astype(np.int64, copy=False)
                i_sum += int(_c64.sum())
                i_sq += int((_c64 * _c64).sum())
                i_abs += int(np.abs(_c64).sum())
                i_mn = min(i_mn, int(_chunk.min()))
                i_mx = max(i_mx, int(_chunk.max()))
                _chunk = None  # noqa: F841
                _c64 = None  # noqa: F841
            agg = np.asarray([i_sum, i_sq, i_abs, i_mn, i_mx], dtype=np.int64)
        h.update(agg.tobytes())
    return h.hexdigest()


_m2_num_digest = _content_digest(m2_numerical_train, k_rows=8192)
_m2_cat_digest = _content_digest(
    _mef_train_subj.astype(np.int64, copy=False),
    _mef_train_bc.astype(np.int64, copy=False),
    _mef_train_cluster.astype(np.int64, copy=False),
    _m2_meta_train["family_ids"].astype(np.int64, copy=False),
    _m2_meta_train["macro_family_ids"].astype(np.int64, copy=False),
    _m2_meta_train["organization_ids"].astype(np.int64, copy=False),
    _m2_meta_train["bench_topic_ids"].astype(np.int64, copy=False),
    y_train.astype(np.float32, copy=False),
    m2_holdout_item_id.astype(np.int64, copy=False),
    k_rows=8192,
)

member2_mlp_state = cache_or_compute(
    "member2_mlp_state",
    key_inputs=(
        # v3_metadata_mode: the v2_dcnv2_audit1 tag covered the
        # cross-tower + audit. v3 explicitly encodes the new
        # feature_mode toggle ("metadata_only" vs "full") so a
        # mode flip can never silently reuse the other mode's
        # cached weights. The (num, cat, y, holdout) content
        # digests already invalidate on any data drift; the
        # mode tag adds belt-and-braces invalidation on the
        # composition change itself.
        "member2_metadata_mlp_v3_metadata_mode",
        _M2_FEATURE_MODE,
        int(len(primary.train)), int(SEED),
        int(indexer.n_subjects), int(indexer.n_bc), int(_m2_n_clusters),
        int(_M2_N_FAMILIES), int(_M2_N_MACRO_FAMILIES),
        int(_M2_N_ORGANIZATIONS), int(_M2_N_BENCH_TOPICS),
        int(_M2_N_SUBJ_NUM), int(_M2_N_BENCH_NUM),
        int(_M2_N_MARGINALS_ACTIVE),
        int(m2_numerical_train.shape[1]),
        round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
        _m2_num_digest,
        _m2_cat_digest,
        tuple(sorted(_m2_cfg.items())),
        # NCL invalidation: lambda + anchor digest so a toggle/relam can't
        # silently reuse a non-NCL (or differently-anchored) checkpoint.
        ("ncl", round(_ncl_lambda_for("member2"), 5)),
        (_content_digest(p_a_train, k_rows=4096) if _ncl_lambda_for("member2") > 0 else "noncl"),
    ),
    compute_fn=_fit_member2_mlp_global,
)

p_member2_val = m2_apply_state_batch(
    member2_mlp_state,
    subject_ids=_mef_val_subj,
    bc_ids=_mef_val_bc,
    cluster_ids=_mef_val_cluster,
    family_ids=_m2_meta_val["family_ids"],
    macro_family_ids=_m2_meta_val["macro_family_ids"],
    organization_ids=_m2_meta_val["organization_ids"],
    bench_topic_ids=_m2_meta_val["bench_topic_ids"],
    numerical=m2_numerical_val,
)
nll_m2 = float(
    -(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
      + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean()
)
print(f"[Member 2 MLP] val log-loss: {nll_m2:.6f}")
print(
    f"[Member 2 MLP] train/val NLL in state: "
    f"{member2_mlp_state.train_loss:.6f} / {member2_mlp_state.val_loss:.6f}"
)

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


# _subject_keys_ordered: built in section 9c (indexer id order).


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
# ## 9d-tail. PCA tail subspace (shared by Members 4 & 5)
#
# Diversification pass (2026-05-30). Fit ONE unsupervised randomized PCA
# on the unique train item embeddings, drop the top `head_drop` PCs (the
# coarse semantic axis M1/M3/M8 key on), keep the next `tail_take` PCs as
# the residual "tail" subspace. Members 4 (logreg) and 5 (kNN) operate
# ONLY on this geometry so their errors decorrelate from the head-driven
# members. The fit is label-free, so one global basis reused across OOF
# folds leaks nothing.

# %%
from src.pca_tail import PcaTailBasis, fit_pca_tail

_PT_CFG = CFG["pca_tail"]
_PT_NCOMP = int(_PT_CFG.get("n_components", 256))
_PT_HEAD = int(_PT_CFG.get("head_drop", 32))
_PT_TAIL = int(_PT_CFG.get("tail_take", 128))
_PT_SEED = int(SEED) + int(_PT_CFG.get("seed_offset", 7))
_PT_D_EMB = int(item_emb_lookup[train_item_keys[0]].shape[0])


def _fit_pca_tail_global():
    print(
        f"[PCA-tail] stacking {len(train_item_keys):,} unique train item "
        "embeddings (inside cache compute_fn so a HIT skips this)..."
    )
    _emb = np.empty((len(train_item_keys), _PT_D_EMB), dtype=np.float32)
    for _i, _k in enumerate(train_item_keys):
        _emb[_i] = item_emb_lookup[_k]
    _basis = fit_pca_tail(
        _emb, n_components=_PT_NCOMP, head_drop=_PT_HEAD,
        tail_take=_PT_TAIL, seed=_PT_SEED,
    )
    _tail_unique = _basis.project(_emb).astype(np.float32)
    del _emb
    gc.collect()
    return {"basis": _basis, "tail_unique": _tail_unique}


_pca_tail_bundle = cache_or_compute(
    "pca_tail_subspace",
    key_inputs=(
        "pca_tail_v1", int(len(train_item_keys)), int(_PT_D_EMB),
        _PT_NCOMP, _PT_HEAD, _PT_TAIL, _PT_SEED,
    ),
    compute_fn=_fit_pca_tail_global,
)
pca_tail: PcaTailBasis = _pca_tail_bundle["basis"]
tail_proj_unique_train = _pca_tail_bundle["tail_unique"]   # [n_train_items, tail_dim]
_TAIL_DIM = int(pca_tail.tail_dim)
print(
    f"[PCA-tail] basis: D={pca_tail.d_emb} -> tail_dim={_TAIL_DIM} "
    f"(dropped top {pca_tail.head_drop}); "
    f"mean tail explained-var={float(pca_tail.explained_variance.mean()):.4g}"
)

# Per-row tail projection for TRAIN rows via the unique-item gather.
# ``m2_holdout_item_id`` is the train-item index per row (all train rows
# resolve to a valid train item, so the gather is total).
assert int(m2_holdout_item_id.min()) >= 0, (
    "every train row must map to a train item for the tail gather"
)
tail_proj_train_rows = tail_proj_unique_train[m2_holdout_item_id]   # [N_train, tail_dim]

# Per-row tail projection for VAL rows. Val items are cold (not in
# train_item_keys), so project their embeddings directly.
_val_emb_for_tail = np.empty((len(primary.val), int(pca_tail.d_emb)), dtype=np.float32)
for _i, _k in enumerate(primary.val["item_key"]):
    _val_emb_for_tail[_i] = item_emb_lookup[str(_k)]
tail_proj_val_rows = pca_tail.project(_val_emb_for_tail)            # [N_val, tail_dim]
del _val_emb_for_tail
gc.collect()
print(
    f"[PCA-tail] per-row projections: train {tail_proj_train_rows.shape}  "
    f"val {tail_proj_val_rows.shape}"
)

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

    _m5_cfg = CFG["member5"]
    _M5_VARIANT = str(_m5_cfg.get("variant", "residual_cluster")).lower()
    if _M5_VARIANT not in {"difficulty_knn", "residual_cluster", "tail_knn"}:
        raise ValueError(
            f"CFG['member5']['variant']={_M5_VARIANT!r} unrecognized; "
            "expected 'difficulty_knn', 'residual_cluster' or 'tail_knn'."
        )

    _M5_ENABLED = True
    # Hyperparams for the legacy difficulty_knn path are kept around so
    # the cache keys / per-fold OOF block can still consult them when
    # _M5_VARIANT == 'difficulty_knn'. Unused under residual_cluster.
    _M5_K = int(_m5_cfg.get("k", 32))
    _M5_TAU = float(_m5_cfg.get("tau", 0.05))
    _M5_RIDGE_ALPHA = float(_m5_cfg.get("ridge_alpha", 10.0))
    _M5_ITEM_FB_WEIGHT = float(_m5_cfg.get("item_fallback_weight", 0.3))
    _M5_MIN_SUBJ_PER_ITEM = int(_m5_cfg.get("min_subjects_per_item", 3))
    _M5_GATE4D_SAMPLE_SIZE = int(_m5_cfg.get("gate4d_sample_size", 64))
    # Hyperparams for the new residual_cluster path.
    _M5_SMOOTH_CELL = float(_m5_cfg.get("smoothing_cell", 30.0))
    _M5_SMOOTH_MARG = float(_m5_cfg.get("smoothing_marginal", 10.0))
    _M5_RES_SCALE = float(_m5_cfg.get("residual_scale", 1.0))

    print(
        f"[Member 5] enabled. variant={_M5_VARIANT}  "
        f"legacy(knn)[k={_M5_K} tau={_M5_TAU} ridge={_M5_RIDGE_ALPHA}]  "
        f"residual[smooth_cell={_M5_SMOOTH_CELL} smooth_marg={_M5_SMOOTH_MARG} "
        f"scale={_M5_RES_SCALE}]"
    )

# ---- Branch: tail_knn variant (diversification pass 2026-05-30) ----------
# Member 5 = kNN in the PCA tail subspace. We reuse fit_knn_member (the
# exact machinery Member 3 uses) but feed it TAIL-projected item
# embeddings instead of raw embeddings, so neighbours are found by
# fine-grained residual similarity -- a different neighbour set than M3's
# head-dominated cosine kNN. Subject-awareness comes from the same
# subject x item passrate matrix M3 uses (so this honours "subject-aware"
# without folding subject features into the ill-defined kNN distance).
if CFG.get("member5", {}).get("enabled", False) and str(
    CFG["member5"].get("variant", "residual_cluster")
).lower() == "tail_knn":
    _M5_KIND = "tail_knn"
    _M5_TAIL_K = int(_m5_cfg.get("tail_k", 128))
    _M5_TAIL_TAU_S = float(_m5_cfg.get("tail_tau_subject", 5.0))
    _M5_TAIL_TAU_G = float(_m5_cfg.get("tail_tau_global", 200.0))
    _M5_TAIL_FB = float(_m5_cfg.get("tail_item_fallback_weight", 0.5))
    _M5_TAIL_QUANT = str(_m5_cfg.get("tail_quantization", "int8"))

    def _fit_member5_tail():
        print(
            f"[Member 5/tail] fitting kNN on tail subspace "
            f"(tail_dim={_TAIL_DIM}, k={_M5_TAIL_K}) over "
            f"{len(train_item_keys):,} train items..."
        )
        return fit_knn_member(
            item_keys=list(train_item_keys),
            item_embeddings=tail_proj_unique_train,
            subject_keys=list(_subject_keys_ordered),
            passrate_dense=m3_passrate_dense,
            passrate_mask=m3_passrate_mask,
            pca_dim=int(_TAIL_DIM),
            quantization=_M5_TAIL_QUANT,
            k=_M5_TAIL_K,
            tau_subject=_M5_TAIL_TAU_S,
            tau_global=_M5_TAIL_TAU_G,
            item_fallback_weight=_M5_TAIL_FB,
            seed=int(SEED) + 777,
        )

    member5_state = cache_or_compute(
        "member5_tail_knn_state",
        key_inputs=(
            "m5_tail_knn_v1", int(len(train_item_keys)),
            int(indexer.n_subjects), int(_TAIL_DIM),
            _M5_TAIL_K, round(_M5_TAIL_TAU_S, 6), round(_M5_TAIL_TAU_G, 6),
            round(_M5_TAIL_FB, 6), _M5_TAIL_QUANT,
            int(_PT_HEAD), int(_PT_TAIL), int(_PT_NCOMP), int(SEED),
        ),
        compute_fn=_fit_member5_tail,
    )
    print(
        f"[Member 5/tail] state: pca_dim={member5_state.pca_dim}  "
        f"k={member5_state.k}  n_items={len(train_item_keys):,}"
    )
    p_member5_val = knn_apply_batch(
        member5_state, tail_proj_val_rows,
        [str(s) for s in primary.val["subject_key"]],
        chunk_size=_M3_CHUNK, use_gpu=_M3_USE_GPU, progress=True,
    )
    nll_m5 = float(-(
        ylab_val * np.log(np.clip(p_member5_val, 1e-6, 1 - 1e-6))
        + (1 - ylab_val) * np.log(1 - np.clip(p_member5_val, 1e-6, 1 - 1e-6))
    ).mean())
    print(
        f"[Member 5/tail] val log-loss: {nll_m5:.6f}  "
        f"p stats: min={float(p_member5_val.min()):.4f} "
        f"mean={float(p_member5_val.mean()):.4f} "
        f"max={float(p_member5_val.max()):.4f}"
    )

# ---- Branch: residual_cluster variant (new default) ----------------------
if CFG.get("member5", {}).get("enabled", False) and str(
    CFG["member5"].get("variant", "residual_cluster")
).lower() == "residual_cluster":
    from src.member5_subject_cluster_residual import (
        Member5ResidualState,
        apply_state_batch as m5res_apply_state_batch,
        fit_member5_residual,
    )

    _M5_KIND = "residual_cluster"
    # We need per-row (subject_id, cluster_id, label) for train and val.
    # These were already computed by section 9b-bis (the mean-encoded
    # interaction features module), which builds _mef_train_cluster /
    # _mef_val_cluster from `query_cluster_ids` AND the train/val
    # subject_id arrays from indexer.subject_to_id.
    _m5res_train_subj = np.fromiter(
        (
            int(indexer.subject_to_id.get(str(s), -1))
            for s in primary.train["subject_key"]
        ),
        dtype=np.int64, count=len(primary.train),
    )
    _m5res_val_subj = np.fromiter(
        (
            int(indexer.subject_to_id.get(str(s), -1))
            for s in primary.val["subject_key"]
        ),
        dtype=np.int64, count=len(primary.val),
    )
    _m5res_train_cluster = np.asarray(_mef_train_cluster, dtype=np.int64)
    _m5res_val_cluster = np.asarray(_mef_val_cluster, dtype=np.int64)
    _m5res_train_y = primary.train["label"].astype(float).to_numpy().astype(np.float64)

    def _fit_member5_residual_global():
        return fit_member5_residual(
            subject_ids=_m5res_train_subj,
            cluster_ids=_m5res_train_cluster,
            labels=_m5res_train_y,
            subject_keys=list(_subject_keys_ordered),
            n_clusters=int(N_CLUSTERS_CTX),
            smoothing_cell=_M5_SMOOTH_CELL,
            smoothing_marginal=_M5_SMOOTH_MARG,
            residual_scale=_M5_RES_SCALE,
            seed=int(SEED),
        )

    member5_state = cache_or_compute(
        "member5_residual_state",
        key_inputs=(
            "m5_residual_v1",
            int(len(primary.train)),
            int(indexer.n_subjects),
            int(N_CLUSTERS_CTX),
            round(_M5_SMOOTH_CELL, 4),
            round(_M5_SMOOTH_MARG, 4),
            round(_M5_RES_SCALE, 4),
            int(SEED),
        ),
        compute_fn=_fit_member5_residual_global,
    )
    print(
        f"[Member 5/residual] state: n_subjects={member5_state.n_subjects:,}  "
        f"n_clusters={member5_state.n_clusters:,}  "
        f"smoothing_cell={member5_state.smoothing_cell:.1f}  "
        f"global_mean_logit={member5_state.global_mean_logit:+.4f}  "
        f"train_loss(sample)={member5_state.train_loss:.5f}  "
        f"max|residual_logit|={float(np.max(np.abs(member5_state.residual_logit))):.3f}  "
        f"mean|residual_logit|={float(np.mean(np.abs(member5_state.residual_logit))):.4f}"
    )

    # Val scoring -- pure lookup, fast.
    p_member5_val = m5res_apply_state_batch(
        member5_state,
        subject_ids=_m5res_val_subj,
        cluster_ids=_m5res_val_cluster,
    )
    nll_m5 = float(-(
        ylab_val * np.log(np.clip(p_member5_val, 1e-6, 1 - 1e-6))
        + (1 - ylab_val) * np.log(1 - np.clip(p_member5_val, 1e-6, 1 - 1e-6))
    ).mean())
    print(
        f"[Member 5/residual] val log-loss: {nll_m5:.6f}  "
        f"p stats: min={float(p_member5_val.min()):.4f} "
        f"mean={float(p_member5_val.mean()):.4f} "
        f"max={float(p_member5_val.max()):.4f}"
    )
    # Sanity: if val NLL is worse than the prior, the residual is wrong.
    _p_prior_m5 = float(np.clip(ylab_val.mean(), 1e-6, 1 - 1e-6))
    _nll_prior_m5 = -(
        _p_prior_m5 * math.log(_p_prior_m5)
        + (1 - _p_prior_m5) * math.log(1 - _p_prior_m5)
    )
    if nll_m5 > _nll_prior_m5 + 0.01:
        print(
            f"  >>> WARNING: Member 5/residual val NLL {nll_m5:.4f} is "
            f"worse than prior {_nll_prior_m5:.4f}; check cluster/subject "
            "id alignment."
        )
    else:
        print(
            f"  [OK] beats prior NLL {_nll_prior_m5:.4f} by "
            f"{_nll_prior_m5 - nll_m5:+.4f} nats"
        )

# ---- Branch: legacy difficulty_knn variant -------------------------------
_M5_RUN_DIFFICULTY = (
    CFG.get("member5", {}).get("enabled", False)
    and str(CFG["member5"].get("variant", "residual_cluster")).lower()
        == "difficulty_knn"
)
if _M5_RUN_DIFFICULTY:
    from src.member5_difficulty_knn import (
        Member5State,
        assert_projection_disjoint_from_val,
        apply_batch_via_ids as m5_apply_batch_via_ids,
        apply_batch as m5_apply_batch,
        apply_one as m5_apply_one,
        fit_member5,
    )

    _M5_KIND = "difficulty_knn"

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

# ---- Disabled branch: bind all the names so downstream cells don't NameError.
if not CFG.get("member5", {}).get("enabled", False):
    _M5_ENABLED = False
    _M5_VARIANT = "disabled"
    _M5_KIND = "disabled"
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


# Diversification pass (2026-05-30): Member 4 is now a logistic
# regression on the PCA TAIL subspace + two subject features
# (IRT theta, subject mean-passrate). It is decorrelated-by-construction
# from M1/M3/M8 (which key on the embedding HEAD) and from M6 (full dense
# matrix). The subject features make it a genuine (subject, item) model
# rather than a per-item difficulty constant.
def _build_m4_tail_matrix(tail_rows, subj_ids, marginal_block):
    _theta = theta_s_per_subject[
        np.clip(subj_ids, 0, len(theta_s_per_subject) - 1)
    ].astype(np.float32)
    # subject mean-passrate is marginal column 0 (mg__subj_mean).
    _submean = np.asarray(marginal_block[:, 0], dtype=np.float32)
    return np.concatenate(
        [tail_rows, _theta[:, None], _submean[:, None]], axis=1
    ).astype(np.float32, copy=False)


X_train_tail_m4 = _build_m4_tail_matrix(
    tail_proj_train_rows, _mef_train_subj, member4_marginal_train,
)
X_val_tail_m4 = _build_m4_tail_matrix(
    tail_proj_val_rows, _mef_val_subj, member4_marginal_val,
)
member4_feature_names = tuple(f"tail_pc_{i}" for i in range(_TAIL_DIM)) + (
    "subj_theta", "subj_mean_passrate",
)
print(
    f"[Member 4] tail feature matrix: train {X_train_tail_m4.shape}  "
    f"val {X_val_tail_m4.shape}  (tail_dim={_TAIL_DIM} + theta + subj_mean)"
)


def _fit_member4():
    print(
        f"[Member 4] training torch logistic regression on TAIL subspace "
        f"(X cols={X_train_tail_m4.shape[1]}, tail+subject features)..."
    )
    return fit_logreg_member(
        X=X_train_tail_m4,
        y=y_train,
        feature_names=member4_feature_names,
        epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
        learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
        weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
        l1_strength=float(CFG.get("member4_logreg", {}).get("l1_strength_tail", 1.0e-4)),
        min_feature_std=float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)),
        early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
        seed=SEED,
        val_fraction=0.1,
        holdout_group_id=holdout_item_id,
        # Global M4 decorrelates from M1 (dominant anchor / Gate diagnostic).
        ncl_anchor_preds=(p_a_train if _ncl_lambda_for("member4") > 0 else None),
        ncl_lambda=_ncl_lambda_for("member4"),
    )


# Cache key tag ``tail_v1`` ties this entry to the tail-subspace feature
# matrix; old (hybrid / marginal-only / embedding-only) entries have a
# different feature dimensionality and must not be reused.
logreg_state = cache_or_compute(
    "logreg_state",
    key_inputs=(
        int(X_train_tail_m4.shape[1]), len(primary.train), SEED,
        "std_v1", "l1minfreq_v1", "coldsplit_v1", "tail_v1",
        int(_TAIL_DIM), int(_PT_HEAD), int(_PT_TAIL), int(_PT_NCOMP),
        round(float(CFG.get("member4_logreg", {}).get("l1_strength_tail", 1.0e-4)), 6),
        round(float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)), 6),
        round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
        ("ncl", round(_ncl_lambda_for("member4"), 5)),
    ),
    compute_fn=_fit_member4,
)
p_member4_val = logreg_apply_state_batch(logreg_state, X_val_tail_m4)
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
# ## 9e-ter. Train Member 6 (Field-weighted Factorization Machine)
#
# FwFM consumes the SAME hybrid feature matrix Member 4 uses
# (X_train_dense_m4, shape ``[N_train, 1216]``); the inductive
# difference vs M4 is the bilinear ``sum_{i<j} r[F(i), F(j)]
# <v_i, v_j> x_i x_j`` interaction term. None of M1 / M3 / M4 can
# express that on its own (M1 only via deep nonlinearity of the
# embedding; M3 via cosine kNN which is locally-constant; M4 strictly
# additive linear). The fit shares the M4 dense matrices so we do not
# allocate a second 16 GB / 3 GB copy.
#
# field_split_mode="single" treats every column as one field -> classic
# FM with a single scalar r[0, 0]. field_split_mode="embedding_vs_marginal"
# splits at the boundary between the M2-style embedding columns and the
# 14 mean-encoded marginal columns -> a 2-field FwFM that can up- or
# down-weight cross-bucket interactions vs within-bucket.

# %%
_M6_ENABLED = bool(CFG.get("member6", {}).get("enabled", False))
if _M6_ENABLED:
    from src.fwfm_member import (
        FwFMState,
        apply_state_batch as fwfm_apply_state_batch,
        fit_fwfm_member,
    )

    _m6_cfg = CFG["member6"]
    _M6_K = int(_m6_cfg.get("k", 8))
    _M6_LR = float(_m6_cfg.get("learning_rate", 1.0e-3))
    _M6_WD_W = float(_m6_cfg.get("weight_decay_w", 1.0e-5))
    _M6_WD_V = float(_m6_cfg.get("weight_decay_V", 1.0e-4))
    _M6_WD_R = float(_m6_cfg.get("weight_decay_r", 1.0e-4))
    _M6_EPOCHS = int(_m6_cfg.get("epochs", 40))
    _M6_BS = int(_m6_cfg.get("batch_size", 16384))
    _M6_PATIENCE = int(_m6_cfg.get("early_stopping_patience", 5))
    _M6_VAL_FRAC = float(_m6_cfg.get("val_fraction", 0.1))
    _M6_FIELD_MODE = str(_m6_cfg.get("field_split_mode", "embedding_vs_marginal")).lower()

    # Member 4 moved to the PCA tail subspace, so it no longer builds the
    # dense hybrid matrix. Member 6 still needs it, so rebuild it here
    # (embedding-derived dense features + 14 mean-encoded marginals).
    X_train_dense_m4 = np.concatenate(
        [X_train_dense, member4_marginal_train], axis=1,
    ).astype(np.float32, copy=False)
    X_val_dense_m4 = np.concatenate(
        [X_val_dense, member4_marginal_val], axis=1,
    ).astype(np.float32, copy=False)
    _m6_dense_feature_names = tuple(member_feat_schema.feature_names) + tuple(
        MEMBER4_MARGINAL_FEATURE_NAMES
    )

    # Diversification: append the item_type one-hots + CoT interaction
    # columns (M6 already carries the raw pool features via
    # member_feat_schema, so only the NEW "extra" block is added). These
    # land past _F_M4_BASE so the embedding_vs_marginal field split assigns
    # them to the marginal/non-embedding field automatically.
    _M6_FORM_NAMES: tuple[str, ...] = ()
    if _dense_form_on("member6"):
        _m6_form_train, _M6_FORM_NAMES = build_item_form_block(
            _train_row_item_keys_g, _train_row_conditions_g, full=False,
        )
        _m6_form_val, _ = build_item_form_block(
            _val_row_item_keys_g, _val_row_conditions_g, full=False,
        )
        X_train_dense_m4 = np.concatenate(
            [X_train_dense_m4, _m6_form_train], axis=1,
        ).astype(np.float32, copy=False)
        X_val_dense_m4 = np.concatenate(
            [X_val_dense_m4, _m6_form_val], axis=1,
        ).astype(np.float32, copy=False)
        _m6_dense_feature_names = _m6_dense_feature_names + tuple(_M6_FORM_NAMES)
        _m6_form_train = None
        _m6_form_val = None
        print(f"[Member 6] + item-form interaction block: {len(_M6_FORM_NAMES)} cols")

    _F_M4_TOTAL = int(X_train_dense_m4.shape[1])
    _F_M4_BASE = int(member_feat_schema.feature_dim)
    if _M6_FIELD_MODE == "single":
        _m6_field_ids = np.zeros(_F_M4_TOTAL, dtype=np.int32)
        _m6_n_fields = 1
    elif _M6_FIELD_MODE == "embedding_vs_marginal":
        _m6_field_ids = np.zeros(_F_M4_TOTAL, dtype=np.int32)
        _m6_field_ids[_F_M4_BASE:] = 1
        _m6_n_fields = 2
    else:
        raise ValueError(
            f"CFG['member6']['field_split_mode']={_M6_FIELD_MODE!r} not in "
            "{'single', 'embedding_vs_marginal'}"
        )
    print(
        f"[Member 6] FwFM enabled. X cols={_F_M4_TOTAL}  k={_M6_K}  "
        f"field_split={_M6_FIELD_MODE} -> n_fields={_m6_n_fields}  "
        f"lr={_M6_LR}  wd_w={_M6_WD_W} wd_V={_M6_WD_V} wd_r={_M6_WD_R}  "
        f"epochs={_M6_EPOCHS} batch={_M6_BS} patience={_M6_PATIENCE}"
    )

    def _fit_member6():
        return fit_fwfm_member(
            X=X_train_dense_m4,
            y=y_train,
            feature_names=_m6_dense_feature_names,
            field_ids=_m6_field_ids,
            k=_M6_K,
            learning_rate=_M6_LR,
            weight_decay_w=_M6_WD_W,
            weight_decay_V=_M6_WD_V,
            weight_decay_r=_M6_WD_R,
            epochs=_M6_EPOCHS,
            batch_size=_M6_BS,
            val_fraction=_M6_VAL_FRAC,
            early_stopping_patience=_M6_PATIENCE,
            seed=int(SEED),
            standardize=True,
            holdout_group_id=holdout_item_id,
        )

    # Content digests over the actual fit inputs. Previously the
    # FwFM cache only saw shape + hyperparams, so a content shift in
    # X_train_dense_m4 (e.g. the mean-encoded smoothing changes, the
    # member_feat_schema is rebuilt, or the redaction seed flips)
    # could silently HIT a stale FwFM. We sample-hash to keep this
    # cheap on the ~16 GB matrix.
    _m6_x_digest_train = _content_digest(X_train_dense_m4, k_rows=8192)
    _m6_y_digest_train = _content_digest(
        y_train.astype(np.float32, copy=False), k_rows=8192,
    )
    _m6_holdout_digest_train = _content_digest(
        holdout_item_id.astype(np.int64, copy=False), k_rows=8192,
    )
    fwfm_state = cache_or_compute(
        "fwfm_state",
        key_inputs=(
            # v2_streaming: bumps over v2_audit1 because the
            # data path inside fit_fwfm_member was rewritten
            # from "load full [N,F] standardized matrix into
            # GPU memory" to a per-batch streaming gather +
            # standardize. The bilinear math is unchanged, but
            # the val + final-train losses now aggregate
            # per-chunk float32 partial sums (vs one big
            # tensor-wide sum); that's a different float
            # reduction order, which can pick a different
            # ``best_state`` for early stopping by 1-2 epochs
            # on the same data. Bumping defensively so the
            # first run after this notebook update trains from
            # scratch.
            "m6_fwfm_v3_scaled_k16",
            int(X_train_dense_m4.shape[1]), len(primary.train), int(SEED),
            int(_M6_K), str(_M6_FIELD_MODE), int(_m6_n_fields),
            round(_M6_LR, 6), round(_M6_WD_W, 7), round(_M6_WD_V, 7), round(_M6_WD_R, 7),
            int(_M6_EPOCHS), int(_M6_BS), int(_M6_PATIENCE),
            round(_M6_VAL_FRAC, 4),
            round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
            _m6_x_digest_train,
            _m6_y_digest_train,
            _m6_holdout_digest_train,
        ),
        compute_fn=_fit_member6,
    )
    # Belt-and-braces FwFM teardown: the trainer already releases its
    # internal GPU tensors + caching-allocator pool, but on a cache
    # HIT the trainer never ran. Either way, force one more GC + VRAM
    # flush so the next member fit doesn't inherit any pinned blocks.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    p_member6_val = fwfm_apply_state_batch(fwfm_state, X_val_dense_m4)

    # Global train-row preds: M6 is an NCL anchor, so its train predictions
    # feed the {M1,M6,M8} decorrelation target used by the OOF penalized
    # members. Chunked to avoid a transient [N,F] standardized copy (~24 GB).
    def _fwfm_apply_chunked(_state, _X, _chunk=262_144):
        _out = np.empty(int(_X.shape[0]), dtype=np.float32)
        for _s in range(0, int(_X.shape[0]), int(_chunk)):
            _e = min(_s + int(_chunk), int(_X.shape[0]))
            _out[_s:_e] = fwfm_apply_state_batch(_state, _X[_s:_e])
        return _out

    p6_train_global = _fwfm_apply_chunked(fwfm_state, X_train_dense_m4)
    nll_m6 = float(-(
        ylab_val * np.log(np.clip(p_member6_val, 1e-6, 1 - 1e-6))
        + (1 - ylab_val) * np.log(1 - np.clip(p_member6_val, 1e-6, 1 - 1e-6))
    ).mean())
    print(
        f"[Member 6] val log-loss: {nll_m6:.6f}  "
        f"||w||={float(np.linalg.norm(fwfm_state.w)):.3f}  "
        f"||V||={float(np.linalg.norm(fwfm_state.V)):.3f}  "
        f"r={np.round(fwfm_state.r, 3).tolist()}  w0={fwfm_state.w0:+.3f}  "
        f"fit_method={fwfm_state.fit_method}"
    )
    if nll_m6 > _nll_prior + 0.01:
        print(
            f"  >>> WARNING: Member 6 val NLL {nll_m6:.4f} is worse than "
            f"prior {_nll_prior:.4f}. The bilinear regularization may be "
            "too weak / strong, or k is wrong for this scale."
        )
    else:
        print(
            f"  [OK] beats prior NLL {_nll_prior:.4f} by "
            f"{_nll_prior - nll_m6:+.4f} nats"
        )
else:
    fwfm_state = None
    p_member6_val = None
    p6_train_global = None
    nll_m6 = float("nan")
    print("[Member 6] DISABLED via CFG['member6']['enabled']=False")

# %% [markdown]
# ## 9e-quater. Train Member 7 (pure-marginal GLU-MLP) and Member 8
# (embeddings GLU-MLP)
#
# Diversification pass (2026-05-30). Two new GLU-MLP members:
#   * M7 -- non-linear interactions among the 14 mean-encoded marginals
#     ONLY (no embeddings, no raw metadata). Complements the additive M4
#     and bilinear M6 on the marginal channel.
#   * M8 -- learned subject-id embedding + full item embedding -> MLP. A
#     collaborative+content model; the item embeddings are gathered from
#     unique-item storage (never materialised as [N, D]).
# Both run BEFORE the pre-OOF cleanup because M7 consumes
# ``member4_marginal_train`` (freed there).

# %%
from src.mlp_member import (
    MlpMemberState as _MlpMemberState,
    apply_state_batch as mlp_apply_state_batch,
    fit_mlp_member,
)


def _nll_vec(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(ylab_val * np.log(p) + (1 - ylab_val) * np.log(1 - p)).mean())


_M7_ENABLED = bool(CFG.get("member7", {}).get("enabled", False))
if _M7_ENABLED:
    _m7_cfg = CFG["member7"]

    def _fit_member7():
        print(
            f"[Member 7] training GLU-MLP on {int(MEMBER4_MARGINAL_FEATURE_DIM)} "
            "marginals (full train)..."
        )
        return fit_mlp_member(
            labels=y_train,
            dense_X=member4_marginal_train,
            dense_feature_names=MEMBER4_MARGINAL_FEATURE_NAMES,
            hid1=int(_m7_cfg.get("hid1", 64)),
            hid2=int(_m7_cfg.get("hid2", 32)),
            learning_rate=float(_m7_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(_m7_cfg.get("weight_decay", 1.0e-5)),
            epochs=int(_m7_cfg.get("epochs", 40)),
            batch_size=int(_m7_cfg.get("batch_size", 16384)),
            early_stopping_patience=int(_m7_cfg.get("early_stopping_patience", 6)),
            feat_dropout=float(_m7_cfg.get("feat_dropout", 0.10)),
            seed=int(SEED) + 701,
            holdout_group_id=holdout_item_id,
            show_progress=True,
            ncl_anchor_preds=(p_a_train if _ncl_lambda_for("member7") > 0 else None),
            ncl_lambda=_ncl_lambda_for("member7"),
        )

    member7_state = cache_or_compute(
        "member7_marginal_mlp_state",
        key_inputs=(
            "m7_marg_mlp_v1", int(MEMBER4_MARGINAL_FEATURE_DIM), len(primary.train),
            int(_m7_cfg.get("hid1", 64)), int(_m7_cfg.get("hid2", 32)),
            round(float(_m7_cfg.get("learning_rate", 1.0e-3)), 6),
            round(float(_m7_cfg.get("weight_decay", 1.0e-5)), 7),
            int(_m7_cfg.get("epochs", 40)),
            int(_m7_cfg.get("early_stopping_patience", 6)),
            round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
            int(SEED),
            ("ncl", round(_ncl_lambda_for("member7"), 5)),
        ),
        compute_fn=_fit_member7,
    )
    p_member7_val = mlp_apply_state_batch(member7_state, dense_X=member4_marginal_val)
    nll_m7 = _nll_vec(p_member7_val)
    print(
        f"[Member 7] val log-loss: {nll_m7:.6f}  "
        f"p[min/mean/max]={float(p_member7_val.min()):.4f}/"
        f"{float(p_member7_val.mean()):.4f}/{float(p_member7_val.max()):.4f}"
    )
else:
    member7_state = None
    p_member7_val = None
    nll_m7 = float("nan")
    print("[Member 7] DISABLED via CFG['member7']['enabled']=False")


_M8_ENABLED = bool(CFG.get("member8", {}).get("enabled", False))
if _M8_ENABLED:
    _m8_cfg = CFG["member8"]

    # Item-form dense block for M8 (full block: pool_z + item_type + cot_x_*).
    _M8_FORM_NAMES: tuple[str, ...] = ()
    _m8_form_train = None
    _m8_form_val = None
    if _dense_form_on("member8"):
        _m8_form_train, _M8_FORM_NAMES = build_item_form_block(
            _train_row_item_keys_g, _train_row_conditions_g, full=True,
        )
        _m8_form_val, _ = build_item_form_block(
            _val_row_item_keys_g, _val_row_conditions_g, full=True,
        )
        print(f"[Member 8] + item-form block: {len(_M8_FORM_NAMES)} dense cols")

    def _fit_member8():
        print(
            f"[Member 8] stacking {len(train_item_keys):,} unique item "
            "embeddings (inside cache compute_fn so a HIT skips this)..."
        )
        _emb = np.empty((len(train_item_keys), _PT_D_EMB), dtype=np.float32)
        for _i, _k in enumerate(train_item_keys):
            _emb[_i] = item_emb_lookup[_k]
        _st = fit_mlp_member(
            labels=y_train,
            subject_ids=_mef_train_subj,
            n_subjects=int(indexer.n_subjects),
            subj_emb_dim=int(_m8_cfg.get("subj_emb_dim", 32)),
            item_emb_unique=_emb,
            row_to_uniq=m2_holdout_item_id,
            dense_X=_m8_form_train,
            dense_feature_names=_M8_FORM_NAMES,
            hid1=int(_m8_cfg.get("hid1", 256)),
            hid2=int(_m8_cfg.get("hid2", 128)),
            learning_rate=float(_m8_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(_m8_cfg.get("weight_decay", 1.0e-5)),
            epochs=int(_m8_cfg.get("epochs", 30)),
            batch_size=int(_m8_cfg.get("batch_size", 16384)),
            early_stopping_patience=int(_m8_cfg.get("early_stopping_patience", 5)),
            feat_dropout=float(_m8_cfg.get("feat_dropout", 0.10)),
            seed=int(SEED) + 801,
            holdout_group_id=m2_holdout_item_id,
            show_progress=True,
        )
        del _emb
        gc.collect()
        return _st

    member8_state = cache_or_compute(
        "member8_embedding_mlp_state",
        key_inputs=(
            "m8_emb_mlp_v1", len(train_item_keys), int(_PT_D_EMB),
            int(indexer.n_subjects), int(_m8_cfg.get("subj_emb_dim", 32)),
            int(_m8_cfg.get("hid1", 256)), int(_m8_cfg.get("hid2", 128)),
            round(float(_m8_cfg.get("learning_rate", 1.0e-3)), 6),
            round(float(_m8_cfg.get("weight_decay", 1.0e-5)), 7),
            int(_m8_cfg.get("epochs", 30)),
            int(_m8_cfg.get("early_stopping_patience", 5)),
            int(SEED),
            ("form", len(_M8_FORM_NAMES)),
        ),
        compute_fn=_fit_member8,
    )

    # Chunked per-row apply (item embeddings gathered per chunk so we never
    # materialise [N, D]). Shared by val scoring and the global train-row
    # anchor preds (M8 is an NCL anchor).
    def _m8_apply_chunked(_state, _item_keys, _subj_ids, _form, _chunk=131_072):
        _keys = np.asarray(_item_keys).astype(str)
        _out = np.empty(int(_keys.shape[0]), dtype=np.float32)
        for _s in range(0, int(_keys.shape[0]), int(_chunk)):
            _e = min(_s + int(_chunk), int(_keys.shape[0]))
            _emb = np.empty((_e - _s, _PT_D_EMB), dtype=np.float32)
            for _j, _k in enumerate(_keys[_s:_e]):
                _emb[_j] = item_emb_lookup[_k]
            _dz = None if _form is None else _form[_s:_e]
            _out[_s:_e] = mlp_apply_state_batch(
                _state, subject_ids=_subj_ids[_s:_e], item_emb=_emb, dense_X=_dz,
            )
            _emb = None
        return _out

    p_member8_val = _m8_apply_chunked(
        member8_state, _val_row_item_keys_g, _mef_val_subj, _m8_form_val,
    )
    # Global train-row preds (NCL anchor for the OOF penalized members + M9).
    p8_train_global = _m8_apply_chunked(
        member8_state, _train_row_item_keys_g, _mef_train_subj, _m8_form_train,
    )
    gc.collect()
    nll_m8 = _nll_vec(p_member8_val)
    print(
        f"[Member 8] val log-loss: {nll_m8:.6f}  "
        f"p[min/mean/max]={float(p_member8_val.min()):.4f}/"
        f"{float(p_member8_val.mean()):.4f}/{float(p_member8_val.max()):.4f}"
    )
else:
    member8_state = None
    p_member8_val = None
    p8_train_global = None
    nll_m8 = float("nan")
    print("[Member 8] DISABLED via CFG['member8']['enabled']=False")


# %% [markdown]
# ## 9e-quinquies. Build the global NCL anchor targets + Member 9 (FwFM+NCL)
#
# The OOF penalized members (M2/M4/M7) decorrelate from the mean of the
# GLOBAL anchors' train-row predictions {M1, M6, M8}; Member 9 (the FwFM
# clone) decorrelates from {M1, M8}. We build those targets here (all
# anchors are now fit) and immediately train the global Member 9 while
# Member 6's dense matrix (which M9 reuses) is still resident -- before the
# pre-OOF cleanup frees it.

# %%
def _mean_of_anchor_preds(anchor_keys):
    """Row-wise mean of the available global anchor train-preds."""
    _parts = []
    for _k in anchor_keys:
        if _k == "member1":
            _parts.append(p_a_train)
        elif _k == "member6" and ("p6_train_global" in globals()) and (p6_train_global is not None):
            _parts.append(p6_train_global)
        elif _k == "member8" and ("p8_train_global" in globals()) and (p8_train_global is not None):
            _parts.append(p8_train_global)
    if not _parts:
        return None
    return np.mean(np.stack(_parts, axis=0), axis=0).astype(np.float32)


_ncl_anchor_oof = (
    _mean_of_anchor_preds(_NCL_ANCHORS) if _NCL_ENABLED else None
)
_ncl_anchor_oof_m9 = (
    _mean_of_anchor_preds(_NCL_M9_ANCHORS) if _NCL_ENABLED else None
)
if _ncl_anchor_oof is not None:
    print(
        f"[NCL] OOF anchor target built from {_NCL_ANCHORS} "
        f"(mean={float(np.mean(_ncl_anchor_oof)):.4f})"
    )

_M9_ENABLED = bool(CFG.get("member9", {}).get("enabled", False)) and _M6_ENABLED
if _M9_ENABLED:
    _m9_cfg = CFG["member9"]
    _M9_NCL_LAMBDA = float(_m9_cfg.get("ncl_lambda", 0.0)) if _NCL_ENABLED else 0.0
    # M9 = clone of M6 (same dense matrix / fields / hyperparams) trained
    # with the NCL penalty against {M1, M8} so it diverges from its twin.
    _m9_anchor_global = (
        _mean_of_anchor_preds(_NCL_M9_ANCHORS) if _M9_NCL_LAMBDA > 0 else None
    )

    def _fit_member9():
        print(
            f"[Member 9] training FwFM clone with NCL (lambda={_M9_NCL_LAMBDA}, "
            f"anchors={_NCL_M9_ANCHORS}) on X cols={int(X_train_dense_m4.shape[1])}..."
        )
        return fit_fwfm_member(
            X=X_train_dense_m4,
            y=y_train,
            feature_names=_m6_dense_feature_names,
            field_ids=_m6_field_ids,
            k=_M6_K,
            learning_rate=_M6_LR,
            weight_decay_w=_M6_WD_W,
            weight_decay_V=_M6_WD_V,
            weight_decay_r=_M6_WD_R,
            epochs=_M6_EPOCHS,
            batch_size=_M6_BS,
            val_fraction=_M6_VAL_FRAC,
            early_stopping_patience=_M6_PATIENCE,
            seed=int(SEED) + 901,
            standardize=True,
            holdout_group_id=holdout_item_id,
            ncl_anchor_preds=_m9_anchor_global,
            ncl_lambda=_M9_NCL_LAMBDA,
        )

    member9_state = cache_or_compute(
        "member9_fwfm_ncl_state",
        key_inputs=(
            "m9_fwfm_ncl_v1",
            int(X_train_dense_m4.shape[1]), len(primary.train), int(SEED),
            int(_M6_K), str(_M6_FIELD_MODE), int(_m6_n_fields),
            round(_M6_LR, 6), round(_M6_WD_W, 7), round(_M6_WD_V, 7), round(_M6_WD_R, 7),
            int(_M6_EPOCHS), int(_M6_BS), int(_M6_PATIENCE), round(_M6_VAL_FRAC, 4),
            _m6_x_digest_train, _m6_y_digest_train, _m6_holdout_digest_train,
            ("ncl", round(_M9_NCL_LAMBDA, 5), tuple(_NCL_M9_ANCHORS)),
        ),
        compute_fn=_fit_member9,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    p_member9_val = fwfm_apply_state_batch(member9_state, X_val_dense_m4)
    nll_m9 = float(-(
        ylab_val * np.log(np.clip(p_member9_val, 1e-6, 1 - 1e-6))
        + (1 - ylab_val) * np.log(1 - np.clip(p_member9_val, 1e-6, 1 - 1e-6))
    ).mean())
    print(
        f"[Member 9] val log-loss: {nll_m9:.6f}  "
        f"(M6 twin val={nll_m6:.6f})  fit_method={member9_state.fit_method}"
    )
else:
    member9_state = None
    p_member9_val = None
    nll_m9 = float("nan")
    print("[Member 9] DISABLED (needs member9.enabled and member6.enabled)")

# %% [markdown]
# ## 9e-bis. Pre-OOF memory reclamation
#
# AGGRESSIVE MEMORY DISCIPLINE: every dense matrix in this notebook is
# float32 with ~1200 columns and ~5M train rows (~16 GB) or ~880k val
# rows (~3 GB). At this point in the run, all of the GLOBAL members are
# fit and val-scored, and the matrices below are no longer needed by
# any downstream code:
#
#   * ``X_train_dense`` (~16 GB)  -- only used to concat into m4
#   * ``X_val_dense``   (~3 GB)   -- only used to concat into m4
#   * ``X_train_dense_m2`` / ``X_val_dense_m2`` -- always ``None`` since
#     the metadata MLP M2 doesn't build dense matrices. Kept in the
#     free-list for forward-compat with experimental M2 variants.
#   * ``X_train_dense_m4`` (~16 GB) -- consumed by global Member 4 fit
#   * ``X_val_dense_m4``   (~3 GB)  -- consumed by Member 4 val scoring
#   * ``member4_marginal_train/val`` (~few hundred MB, absorbed in m4
#     and consumed by the global Member 2 MLP fit above)
#
# The OOF loop in section 9.5 builds its own fold-scoped matrices on
# the fly. Holding the global matrices through that loop is what made
# the OOF fold-0 OOM happen at "Building fold X_*_dense" -- the loop
# never even got a chance to allocate its 16 GB X_fold_train_dense
# because ~50 GB of now-obsolete globals were already pinned.

# %%
print("[Pre-OOF cleanup] Freeing global dense matrices no longer needed...")
_pre_oof_to_free = [
    ("X_train_dense", X_train_dense),
    ("X_val_dense", X_val_dense),
    ("X_train_dense_m4", X_train_dense_m4),
    ("X_val_dense_m4", X_val_dense_m4),
    ("member4_marginal_train", member4_marginal_train),
    ("member4_marginal_val", member4_marginal_val),
    ("tail_proj_train_rows", tail_proj_train_rows),
    ("tail_proj_unique_train", tail_proj_unique_train),
    ("X_train_tail_m4", X_train_tail_m4),
    ("X_val_tail_m4", X_val_tail_m4),
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
# Tail projections / matrices used only by the GLOBAL M4/M5 fits.
# Per-fold OOF builds its own tail projections, so these can go.
tail_proj_train_rows = None
tail_proj_unique_train = None
X_train_tail_m4 = None
X_val_tail_m4 = None
gc.collect()
# Flush PyTorch's CUDA caching allocator: at this point Members 1, 2,
# and 6 have all just been trained on GPU. The trainer functions
# release their per-fit pools internally, but the cache HIT path skips
# those teardowns and the global allocator can still be holding a
# multi-GB block from the last fit. We do a final flush here so the
# OOF loop's per-fold M1 retraining inherits a clean VRAM state.
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
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
    # Only the difficulty_knn variant populates this; the residual_cluster
    # variant uses categorical ids only and has no projection-leakage
    # surface, so its per-fold dict stays empty.
    _gate4c_per_fold: dict[int, dict] = {}
else:
    p5_train_oof_acc = None
    _gate4c_per_fold = {}

# Member 6 (FwFM) OOF accumulator -- created only when M6 is enabled.
_M6_ENABLED = bool(CFG.get("member6", {}).get("enabled", False))
if _M6_ENABLED:
    p6_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p6_train_oof")
else:
    p6_train_oof_acc = None

# Members 7 (marginal MLP) & 8 (embedding MLP) OOF accumulators.
if _M7_ENABLED:
    p7_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p7_train_oof")
else:
    p7_train_oof_acc = None
if _M8_ENABLED:
    p8_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p8_train_oof")
else:
    p8_train_oof_acc = None

# Member 9 (FwFM + NCL clone) OOF accumulator.
if _M9_ENABLED:
    p9_train_oof_acc = OofPredictionAccumulator(_N_TRAIN, name="p9_train_oof")
else:
    p9_train_oof_acc = None


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
    # fold_mean_encoded_stats has been consumed by the two apply_*
    # calls above; nothing else references it for this fold. Free now.
    del fold_mean_encoded_stats
    gc.collect()

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
    print(
        f"[OOF f{fold.fold_id}] DEFERRING X_fold_*_dense build "
        "(Member 4 consumes them; built JIT below)."
    )
    X_fold_train_dense = None
    X_fold_oof_dense = None
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
        # Even with the new :class:`IndexedEmbeddingView` wrapper (which
        # cuts the per-split item-embedding tensor from the legacy ~80 GB
        # stacked form down to ~5 GB of unique-vector storage + ~40 MB of
        # per-row pointers), three concurrent fold datasets still hold
        # several GB each of pool / NN / id / label / unique-embedding
        # tensors. On cache HIT that's pure waste; on cache MISS it
        # stacks on top of the Model A training peak. We defer the
        # train/val build into the compute_fn (so cache HIT skips it
        # entirely; cache MISS GCs them at function return), and build
        # _oof_ds_fold only after the cache call (it's only needed for
        # scoring, never for training).

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
        #   - cache HIT path: train/val ds were never built
        #   - cache MISS path: train/val ds were built inside compute_fn
        #     and freed before this point
        # With IndexedEmbeddingView the item-embedding portion of each
        # dataset is ~5 GB (unique items, dedup'd) rather than ~80 GB
        # (per-row stacked). _oof_ds_fold is the only live dataset
        # during scoring; the savings dominate the steady-state RSS
        # of the OOF inner loop.
        _oof_ds_fold = _build(fold_oof_df, nn_oof_mat_fold)
        # Load fold M1 weights into a fresh model + score on OOF rows.
        _fold_model = _build_model_for_inf(MODEL_A_NAME, model_a_cfg)
        _fold_model.attach_metadata_tables(meta_id_tables)
        _fold_model.load_state_dict(_fold_ckpt["model_state"])
        _fold_model = _fold_model.to(device).eval()
        p_a_oof_fold = _score_dataset(_oof_ds_fold, _fold_model)
        p_a_anchor_fold_train = None
        _fold_model = _fold_model.to("cpu")
        del _fold_model, _oof_ds_fold
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Use global p_a_train (saw all train items -- documented small leak).
        p_a_oof_fold = p_a_train[fold.oof_row_idx]
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

    # ----- Fold Member 2 (dense metadata MLP) -----
    print(f"[OOF f{fold.fold_id}] Training fold Member 2 (dense metadata MLP)...")
    _m2_holdout_fold = np.array(
        [int(_item_to_train_idx.get(str(k), -1)) for k in fold_train_df["item_key"]],
        dtype=np.int64,
    )
    _m2_fold_n_clusters = max(
        int(_N_CLUSTERS_ME),
        int(_mef_cluster_fold_train.max()) + 1 if _mef_cluster_fold_train.size else 0,
        int(_mef_cluster_fold_oof.max()) + 1 if _mef_cluster_fold_oof.size else 0,
    )

    _m2_meta_fold_train = _m2_gather_metadata(
        _mef_subj_fold_train, _mef_bc_fold_train,
    )
    _m2_meta_fold_oof = _m2_gather_metadata(
        _mef_subj_fold_oof, _mef_bc_fold_oof,
    )
    # Mirror the global path's metadata_only handling: zero out the
    # marginals and bc_redacted flag so the fold fit sees the exact
    # same composition as the global fit. Any drift here would silently
    # invalidate the OOF -> stacker pipeline for M2.
    if _M2_METADATA_ONLY:
        _n_fold_train_rows = int(_m2_meta_fold_train["subject_numerical"].shape[0])
        _n_fold_oof_rows = int(_m2_meta_fold_oof["subject_numerical"].shape[0])
        _m2_fold_marg_train = np.zeros((_n_fold_train_rows, 0), dtype=np.float32)
        _m2_fold_marg_oof = np.zeros((_n_fold_oof_rows, 0), dtype=np.float32)
        _m2_fold_redact_train = np.zeros(_n_fold_train_rows, dtype=np.float32)
        _m2_fold_redact_oof = np.zeros(_n_fold_oof_rows, dtype=np.float32)
    else:
        _m2_fold_marg_train = fold_member4_marginal_train
        _m2_fold_marg_oof = fold_member4_marginal_oof
        _m2_fold_redact_train = bc_redacted_train[fold.train_row_idx]
        _m2_fold_redact_oof = bc_redacted_train[fold.oof_row_idx]

    # Item-form block for this fold (must match the global M2 composition).
    if _dense_form_on("member2"):
        _m2_form_fold_train, _ = build_item_form_block(
            _train_row_item_keys_g[fold.train_row_idx],
            _train_row_conditions_g[fold.train_row_idx],
            full=True,
        )
        _m2_form_fold_oof, _ = build_item_form_block(
            _train_row_item_keys_g[fold.oof_row_idx],
            _train_row_conditions_g[fold.oof_row_idx],
            full=True,
        )
        _m2_fold_marg_train = np.concatenate(
            [_m2_fold_marg_train, _m2_form_fold_train], axis=1,
        ).astype(np.float32, copy=False)
        _m2_fold_marg_oof = np.concatenate(
            [_m2_fold_marg_oof, _m2_form_fold_oof], axis=1,
        ).astype(np.float32, copy=False)

    _m2_num_fold_train = m2_assemble_numerical(
        subject_numerical=_m2_meta_fold_train["subject_numerical"],
        bench_numerical=_m2_meta_fold_train["bench_numerical"],
        bc_redacted_flag=_m2_fold_redact_train,
        marginals=_m2_fold_marg_train,
    )
    _m2_num_fold_oof = m2_assemble_numerical(
        subject_numerical=_m2_meta_fold_oof["subject_numerical"],
        bench_numerical=_m2_meta_fold_oof["bench_numerical"],
        bc_redacted_flag=_m2_fold_redact_oof,
        marginals=_m2_fold_marg_oof,
    )

    def _fit_fold_m2_mlp(_ff=fold):
        return fit_member2_metadata_mlp(
            subject_ids=_mef_subj_fold_train,
            bc_ids=_mef_bc_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            family_ids=_m2_meta_fold_train["family_ids"],
            macro_family_ids=_m2_meta_fold_train["macro_family_ids"],
            organization_ids=_m2_meta_fold_train["organization_ids"],
            bench_topic_ids=_m2_meta_fold_train["bench_topic_ids"],
            numerical=_m2_num_fold_train,
            y=_y_fold_train,
            subject_keys=_subject_keys_ordered,
            bc_keys=_bc_keys_ordered,
            num_feature_names=_M2_NUM_FEATURE_NAMES,
            n_subjects=int(indexer.n_subjects),
            n_bcs=int(indexer.n_bc),
            n_clusters=int(_m2_fold_n_clusters),
            n_families=int(_M2_N_FAMILIES),
            n_macro_families=int(_M2_N_MACRO_FAMILIES),
            n_organizations=int(_M2_N_ORGANIZATIONS),
            n_bench_topics=int(_M2_N_BENCH_TOPICS),
            n_subj_num=int(_M2_N_SUBJ_NUM),
            n_bench_num=int(_M2_N_BENCH_NUM),
            n_marginals=int(_M2_N_MARGINALS_ACTIVE),
            d_subj=(int(_m2_cfg.get("d_subj", 32)) if _M2_USE_SUBJ_BC else 0),
            d_bc=(int(_m2_cfg.get("d_bc", 32)) if _M2_USE_SUBJ_BC else 0),
            d_cluster=(int(_m2_cfg.get("d_cluster", 16)) if _M2_USE_CLUSTER else 0),
            d_family=int(_m2_cfg.get("d_family", 16)),
            d_macro=int(_m2_cfg.get("d_macro", 8)),
            d_org=int(_m2_cfg.get("d_org", 16)),
            d_topic=int(_m2_cfg.get("d_topic", 16)),
            hid1=int(_m2_cfg.get("hid1", 256)),
            hid2=int(_m2_cfg.get("hid2", 128)),
            n_cross_layers=int(_m2_cfg.get("n_cross_layers", 2)),
            cross_rank=int(_m2_cfg.get("cross_rank", 64)),
            learning_rate=float(_m2_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(_m2_cfg.get("weight_decay", 1.0e-5)),
            epochs=int(_m2_cfg.get("epochs", 40)),
            batch_size=int(_m2_cfg.get("batch_size", 16384)),
            val_fraction=float(_m2_cfg.get("val_fraction", 0.1)),
            early_stopping_patience=int(_m2_cfg.get("early_stopping_patience", 5)),
            cat_dropout_subject=float(_m2_cfg.get("cat_dropout_subject", 0.05)),
            cat_dropout_bc=float(_m2_cfg.get("cat_dropout_bc", 0.10)),
            cat_dropout_cluster=float(_m2_cfg.get("cat_dropout_cluster", 0.10)),
            cat_dropout_family=float(_m2_cfg.get("cat_dropout_family", 0.05)),
            cat_dropout_macro=float(_m2_cfg.get("cat_dropout_macro", 0.05)),
            cat_dropout_org=float(_m2_cfg.get("cat_dropout_org", 0.05)),
            cat_dropout_topic=float(_m2_cfg.get("cat_dropout_topic", 0.10)),
            feat_dropout=float(_m2_cfg.get("feat_dropout", 0.10)),
            warmup_epochs=int(_m2_cfg.get("warmup_epochs", 2)),
            use_cosine_schedule=bool(_m2_cfg.get("use_cosine_schedule", True)),
            ema_decay=float(_m2_cfg.get("ema_decay", 0.999)),
            snapshot_ensemble_k=int(_m2_cfg.get("snapshot_ensemble_k", 3)),
            label_smoothing=float(_m2_cfg.get("label_smoothing", 0.005)),
            mixup_alpha=float(_m2_cfg.get("mixup_alpha", 0.0)),
            seed=int(SEED) + 100 * (int(_ff.fold_id) + 1),
            holdout_group_id=_m2_holdout_fold,
            show_progress=False,
            # OOF NCL: decorrelate from the {M1,M6,M8} global anchor target
            # restricted to this fold's train rows (regulariser only).
            ncl_anchor_preds=(
                _ncl_anchor_oof[_ff.train_row_idx]
                if (_ncl_lambda_for("member2") > 0 and _ncl_anchor_oof is not None)
                else None
            ),
            ncl_lambda=_ncl_lambda_for("member2"),
        )

    # Per-fold content digests over the actual fit inputs. The
    # previous OOF cache key only saw cardinalities + column count,
    # so any future change to ``_m2_gather_metadata`` semantics, the
    # subject/benchmark numerical column ordering, or the per-fold
    # marginal computation would silently HIT a stale OOF state.
    _m2_num_digest_fold = _content_digest(_m2_num_fold_train, k_rows=8192)
    _m2_cat_digest_fold = _content_digest(
        _mef_subj_fold_train.astype(np.int64, copy=False),
        _mef_bc_fold_train.astype(np.int64, copy=False),
        _mef_cluster_fold_train.astype(np.int64, copy=False),
        _m2_meta_fold_train["family_ids"].astype(np.int64, copy=False),
        _m2_meta_fold_train["macro_family_ids"].astype(np.int64, copy=False),
        _m2_meta_fold_train["organization_ids"].astype(np.int64, copy=False),
        _m2_meta_fold_train["bench_topic_ids"].astype(np.int64, copy=False),
        _y_fold_train.astype(np.float32, copy=False),
        _m2_holdout_fold.astype(np.int64, copy=False),
        k_rows=8192,
    )
    _fold_m2_state = cache_or_compute(
        "member2_mlp_oof_fold",
        key_inputs=(
            # v3_metadata_mode: matches the global key bump; the
            # _M2_FEATURE_MODE token explicitly forbids cross-mode
            # cache reuse even if every other key matched.
            "member2_metadata_mlp_oof_v3_metadata_mode",
            _M2_FEATURE_MODE,
            fold.fold_id, fold_suffix,
            int(len(fold.train_row_idx)), int(len(fold.oof_row_idx)),
            int(indexer.n_subjects), int(indexer.n_bc), int(_m2_fold_n_clusters),
            int(_M2_N_FAMILIES), int(_M2_N_MACRO_FAMILIES),
            int(_M2_N_ORGANIZATIONS), int(_M2_N_BENCH_TOPICS),
            int(_M2_N_SUBJ_NUM), int(_M2_N_BENCH_NUM),
            int(_M2_N_MARGINALS_ACTIVE),
            int(_m2_num_fold_train.shape[1]),
            round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
            _m2_num_digest_fold,
            _m2_cat_digest_fold,
            tuple(sorted(_m2_cfg.items())),
            int(SEED),
            ("ncl", round(_ncl_lambda_for("member2"), 5)),
        ),
        compute_fn=_fit_fold_m2_mlp,
    )
    p2_oof_fold = m2_apply_state_batch(
        _fold_m2_state,
        subject_ids=_mef_subj_fold_oof,
        bc_ids=_mef_bc_fold_oof,
        cluster_ids=_mef_cluster_fold_oof,
        family_ids=_m2_meta_fold_oof["family_ids"],
        macro_family_ids=_m2_meta_fold_oof["macro_family_ids"],
        organization_ids=_m2_meta_fold_oof["organization_ids"],
        bench_topic_ids=_m2_meta_fold_oof["bench_topic_ids"],
        numerical=_m2_num_fold_oof,
    )
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

    # ----- Fold Member 5 (variant-dependent) -----
    # One mutually-exclusive if/elif/elif/else chain. Every branch
    # produces a [N_oof_fold] float32 ``p5_oof_fold`` written into the
    # global accumulator, AND frees the M3-built fold arrays
    # (_fold_item_emb_stacked / _fold_passrate_dense /
    # _fold_passrate_mask_dense) exactly once before Member 4's training
    # tensors land. The chain MUST stay an if/elif/else so the arrays are
    # never double-freed (the crash that motivated this structure):
    #   - "tail_knn" (current default): kNN in the PCA tail subspace,
    #     subject-aware via the M3 passrate matrix.
    #   - "residual_cluster": subject x cluster residual passrate lookup.
    #   - "difficulty_knn" (legacy): the original 1-D difficulty kNN.
    #   - else (disabled / unknown variant): p5=None, free the arrays.
    if _M5_ENABLED and _M5_VARIANT == "tail_knn":
        print(f"[OOF f{fold.fold_id}] Fitting fold Member 5 (tail_knn)...")
        # kNN in the PCA tail subspace. Reuses the M3-built fold passrate
        # (subject-awareness) but projects items into the tail basis so
        # neighbours differ from M3's head-dominated cosine kNN.
        _fold_tail_items = pca_tail.project(_fold_item_emb_stacked)  # [n_fit, tail_dim]
        _fold_m5_state = cache_or_compute(
            "member5_tail_knn_state_oof_fold",
            key_inputs=(
                "m5_tail_knn_oof_v1", fold.fold_id, fold_suffix,
                int(len(fold.train_item_keys)), int(indexer.n_subjects),
                int(_TAIL_DIM), int(_M5_TAIL_K),
                round(_M5_TAIL_TAU_S, 6), round(_M5_TAIL_TAU_G, 6),
                round(_M5_TAIL_FB, 6), _M5_TAIL_QUANT,
                int(_PT_HEAD), int(_PT_TAIL), int(SEED),
            ),
            compute_fn=lambda _ff=fold: fit_knn_member(
                item_keys=list(_ff.train_item_keys),
                item_embeddings=_fold_tail_items,
                subject_keys=list(_subject_keys_ordered),
                passrate_dense=_fold_passrate_dense,
                passrate_mask=_fold_passrate_mask_dense,
                pca_dim=int(_TAIL_DIM),
                quantization=_M5_TAIL_QUANT,
                k=_M5_TAIL_K,
                tau_subject=_M5_TAIL_TAU_S,
                tau_global=_M5_TAIL_TAU_G,
                item_fallback_weight=_M5_TAIL_FB,
                seed=int(SEED) + 777 * (int(_ff.fold_id) + 1),
            ),
        )
        del _fold_item_emb_stacked, _fold_passrate_dense
        del _fold_passrate_mask_dense, _fold_tail_items
        gc.collect()
        # Score OOF rows: project their (cold) item embeddings to tail.
        _m5_oof_emb = np.empty((len(fold_oof_df), _PT_D_EMB), dtype=np.float32)
        for _i, _k in enumerate(fold_oof_df["item_key"].astype(str)):
            _m5_oof_emb[_i] = item_emb_lookup[_k]
        _m5_oof_tail = pca_tail.project(_m5_oof_emb)
        del _m5_oof_emb
        p5_oof_fold = knn_apply_batch(
            _fold_m5_state, _m5_oof_tail,
            fold_oof_df["subject_key"].astype(str).tolist(),
        )
        p5_train_oof_acc.write_fold(fold.oof_row_idx, p5_oof_fold)
        del _fold_m5_state, _m5_oof_tail
        gc.collect()

    elif _M5_ENABLED and _M5_VARIANT == "residual_cluster":
        print(f"[OOF f{fold.fold_id}] Fitting fold Member 5 (residual_cluster)...")
        # Free the M3-built fold passrate/embeddings; the residual M5
        # only needs row-level (subject_id, cluster_id, label) arrays.
        del _fold_item_emb_stacked, _fold_passrate_dense, _fold_passrate_mask_dense
        gc.collect()

        _fold_m5res_train_subj = np.fromiter(
            (
                int(indexer.subject_to_id.get(str(s), -1))
                for s in fold_train_df["subject_key"]
            ),
            dtype=np.int64, count=len(fold_train_df),
        )
        _fold_m5res_train_cluster = np.asarray(
            _mef_cluster_fold_train, dtype=np.int64,
        )
        _fold_m5res_train_y = fold_train_df["label"].astype(float).to_numpy().astype(np.float64)
        _fold_m5res_state = cache_or_compute(
            "member5_residual_state_oof_fold",
            key_inputs=(
                "m5res_oof_v1", fold.fold_id, fold_suffix,
                int(len(fold_train_df)), int(indexer.n_subjects),
                int(N_CLUSTERS_CTX),
                round(_M5_SMOOTH_CELL, 4),
                round(_M5_SMOOTH_MARG, 4),
                round(_M5_RES_SCALE, 4),
                int(SEED),
            ),
            compute_fn=lambda: fit_member5_residual(
                subject_ids=_fold_m5res_train_subj,
                cluster_ids=_fold_m5res_train_cluster,
                labels=_fold_m5res_train_y,
                subject_keys=list(_subject_keys_ordered),
                n_clusters=int(N_CLUSTERS_CTX),
                smoothing_cell=_M5_SMOOTH_CELL,
                smoothing_marginal=_M5_SMOOTH_MARG,
                residual_scale=_M5_RES_SCALE,
                seed=int(SEED) + 400 * (int(fold.fold_id) + 1),
            ),
        )
        del _fold_m5res_train_subj, _fold_m5res_train_cluster, _fold_m5res_train_y
        gc.collect()

        # Score fold Member 5 on OOF rows -- pure lookup, fast.
        _fold_oof_subj_ids_m5 = np.fromiter(
            (
                int(indexer.subject_to_id.get(str(s), -1))
                for s in fold_oof_df["subject_key"]
            ),
            dtype=np.int64, count=len(fold_oof_df),
        )
        _fold_oof_cluster_ids_m5 = np.asarray(
            _mef_cluster_fold_oof, dtype=np.int64,
        )
        p5_oof_fold = m5res_apply_state_batch(
            _fold_m5res_state,
            subject_ids=_fold_oof_subj_ids_m5,
            cluster_ids=_fold_oof_cluster_ids_m5,
        )
        p5_train_oof_acc.write_fold(fold.oof_row_idx, p5_oof_fold)
        del (
            _fold_oof_subj_ids_m5, _fold_oof_cluster_ids_m5,
            _fold_m5res_state,
        )
        gc.collect()

    elif _M5_ENABLED and _M5_VARIANT == "difficulty_knn":
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

    # ===== Diversification members: M4 (tail logreg), M7 (marginal MLP),
    #       M8 (embedding MLP). Fit here -- BEFORE the dense-matrix build
    #       below frees the fold marginals that M7/M4 consume. M5 (tail
    #       kNN) was already fit + written in its branch above. =====
    print(f"[OOF f{fold.fold_id}] Fitting diversification members (M4-tail / M7 / M8)...")
    # Fold-train + fold-oof UNIQUE item embeddings (gather once; reused by
    # M4's tail features and M8's item channel). Unique-item counts are
    # small (tens of thousands), so these stacks are a few GB at most and
    # are freed at the end of this block, before the dense build lands.
    _div_tr_item_keys = [str(k) for k in fold.train_item_keys]
    _div_tr_item_idx = {k: i for i, k in enumerate(_div_tr_item_keys)}
    _div_of_item_keys = [str(k) for k in fold.oof_item_keys]
    _div_of_item_idx = {k: i for i, k in enumerate(_div_of_item_keys)}
    _div_tr_uniq_emb = np.empty((len(_div_tr_item_keys), _PT_D_EMB), dtype=np.float32)
    for _i, _k in enumerate(_div_tr_item_keys):
        _div_tr_uniq_emb[_i] = item_emb_lookup[_k]
    _div_of_uniq_emb = np.empty((len(_div_of_item_keys), _PT_D_EMB), dtype=np.float32)
    for _i, _k in enumerate(_div_of_item_keys):
        _div_of_uniq_emb[_i] = item_emb_lookup[_k]
    _div_tr_r2u = np.fromiter(
        (_div_tr_item_idx[str(k)] for k in fold_train_df["item_key"]),
        dtype=np.int64, count=len(fold_train_df),
    )
    _div_of_r2u = np.fromiter(
        (_div_of_item_idx[str(k)] for k in fold_oof_df["item_key"]),
        dtype=np.int64, count=len(fold_oof_df),
    )

    # ---- M4 (tail logreg): [tail-PCA + theta + subject mean-passrate] ----
    _div_tr_uniq_tail = pca_tail.project(_div_tr_uniq_emb)
    _div_of_uniq_tail = pca_tail.project(_div_of_uniq_emb)
    _m4_theta_tr = theta_s_per_subject[
        np.clip(_mef_subj_fold_train, 0, len(theta_s_per_subject) - 1)
    ].astype(np.float32)
    _m4_theta_of = theta_s_per_subject[
        np.clip(_mef_subj_fold_oof, 0, len(theta_s_per_subject) - 1)
    ].astype(np.float32)
    _m4_submean_tr = fold_member4_marginal_train[:, 0].astype(np.float32)
    _m4_submean_of = fold_member4_marginal_oof[:, 0].astype(np.float32)
    _X_m4_tr = np.concatenate(
        [_div_tr_uniq_tail[_div_tr_r2u], _m4_theta_tr[:, None], _m4_submean_tr[:, None]],
        axis=1,
    ).astype(np.float32)
    _X_m4_of = np.concatenate(
        [_div_of_uniq_tail[_div_of_r2u], _m4_theta_of[:, None], _m4_submean_of[:, None]],
        axis=1,
    ).astype(np.float32)
    _m4_tail_names = tuple(f"tail_pc_{i}" for i in range(_TAIL_DIM)) + (
        "subj_theta", "subj_mean_passrate",
    )
    _fold_logreg_state = cache_or_compute(
        "logreg_state_oof_fold",
        key_inputs=(
            "logreg_tail_oof_v1", fold.fold_id, fold_suffix,
            int(_X_m4_tr.shape[1]), int(_X_m4_tr.shape[0]),
            int(_TAIL_DIM), int(_PT_HEAD), int(_PT_TAIL), int(SEED),
            ("ncl", round(_ncl_lambda_for("member4"), 5)),
        ),
        compute_fn=lambda _ff=fold: fit_logreg_member(
            X=_X_m4_tr,
            y=_y_fold_train,
            feature_names=_m4_tail_names,
            epochs=int(CFG.get("member4_logreg", {}).get("epochs", 200)),
            learning_rate=float(CFG.get("member4_logreg", {}).get("learning_rate", 0.05)),
            weight_decay=float(CFG.get("member4_logreg", {}).get("weight_decay", 1.0e-3)),
            l1_strength=float(CFG.get("member4_logreg", {}).get("l1_strength_tail", 1.0e-4)),
            min_feature_std=float(CFG.get("member4_logreg", {}).get("min_feature_std", 1.0e-2)),
            early_stopping_patience=int(CFG.get("member4_logreg", {}).get("early_stopping_patience", 20)),
            seed=int(SEED) + 300 * (int(_ff.fold_id) + 1),
            val_fraction=0.1,
            holdout_group_id=_m2_holdout_fold,
            ncl_anchor_preds=(
                _ncl_anchor_oof[_ff.train_row_idx]
                if (_ncl_lambda_for("member4") > 0 and _ncl_anchor_oof is not None)
                else None
            ),
            ncl_lambda=_ncl_lambda_for("member4"),
        ),
    )
    p4_oof_fold = logreg_apply_state_batch(_fold_logreg_state, _X_m4_of)
    p4_train_oof_acc.write_fold(fold.oof_row_idx, p4_oof_fold)
    del _X_m4_tr, _X_m4_of, _fold_logreg_state
    del _div_tr_uniq_tail, _div_of_uniq_tail
    gc.collect()

    # ---- M7 (marginal MLP) ----
    if _M7_ENABLED:
        _fold_m7_state = cache_or_compute(
            "member7_marginal_mlp_state_oof_fold",
            key_inputs=(
                "m7_marg_mlp_oof_v1", fold.fold_id, fold_suffix,
                int(MEMBER4_MARGINAL_FEATURE_DIM), int(len(fold_train_df)),
                int(_m7_cfg.get("hid1", 64)), int(_m7_cfg.get("hid2", 32)),
                int(_m7_cfg.get("epochs", 40)), int(SEED),
                ("ncl", round(_ncl_lambda_for("member7"), 5)),
            ),
            compute_fn=lambda _ff=fold: fit_mlp_member(
                labels=_y_fold_train,
                dense_X=fold_member4_marginal_train,
                dense_feature_names=MEMBER4_MARGINAL_FEATURE_NAMES,
                hid1=int(_m7_cfg.get("hid1", 64)), hid2=int(_m7_cfg.get("hid2", 32)),
                learning_rate=float(_m7_cfg.get("learning_rate", 1.0e-3)),
                weight_decay=float(_m7_cfg.get("weight_decay", 1.0e-5)),
                epochs=int(_m7_cfg.get("epochs", 40)),
                batch_size=int(_m7_cfg.get("batch_size", 16384)),
                early_stopping_patience=int(_m7_cfg.get("early_stopping_patience", 6)),
                feat_dropout=float(_m7_cfg.get("feat_dropout", 0.10)),
                seed=int(SEED) + 701 * (int(_ff.fold_id) + 1),
                holdout_group_id=_m2_holdout_fold,
                show_progress=False,
                ncl_anchor_preds=(
                    _ncl_anchor_oof[_ff.train_row_idx]
                    if (_ncl_lambda_for("member7") > 0 and _ncl_anchor_oof is not None)
                    else None
                ),
                ncl_lambda=_ncl_lambda_for("member7"),
            ),
        )
        p7_oof_fold = mlp_apply_state_batch(_fold_m7_state, dense_X=fold_member4_marginal_oof)
        p7_train_oof_acc.write_fold(fold.oof_row_idx, p7_oof_fold)
        del _fold_m7_state
        gc.collect()
    else:
        p7_oof_fold = None

    # ---- M8 (embedding MLP): subject-id emb + item emb (+ item-form block) ----
    if _M8_ENABLED:
        if _dense_form_on("member8"):
            _m8_form_fold_train, _ = build_item_form_block(
                _train_row_item_keys_g[fold.train_row_idx],
                _train_row_conditions_g[fold.train_row_idx],
                full=True,
            )
            _m8_form_fold_oof, _ = build_item_form_block(
                _train_row_item_keys_g[fold.oof_row_idx],
                _train_row_conditions_g[fold.oof_row_idx],
                full=True,
            )
        else:
            _m8_form_fold_train = None
            _m8_form_fold_oof = None
        _fold_m8_state = cache_or_compute(
            "member8_embedding_mlp_state_oof_fold",
            key_inputs=(
                "m8_emb_mlp_oof_v1", fold.fold_id, fold_suffix,
                int(len(_div_tr_item_keys)), int(_PT_D_EMB),
                int(indexer.n_subjects), int(_m8_cfg.get("subj_emb_dim", 32)),
                int(_m8_cfg.get("hid1", 256)), int(_m8_cfg.get("hid2", 128)),
                int(_m8_cfg.get("epochs", 30)), int(SEED),
                ("form", int(len(DENSE_FORM_FULL_NAMES)) if _dense_form_on("member8") else 0),
            ),
            compute_fn=lambda _ff=fold: fit_mlp_member(
                labels=_y_fold_train,
                subject_ids=_mef_subj_fold_train,
                n_subjects=int(indexer.n_subjects),
                subj_emb_dim=int(_m8_cfg.get("subj_emb_dim", 32)),
                item_emb_unique=_div_tr_uniq_emb,
                row_to_uniq=_div_tr_r2u,
                dense_X=_m8_form_fold_train,
                dense_feature_names=(DENSE_FORM_FULL_NAMES if _dense_form_on("member8") else ()),
                hid1=int(_m8_cfg.get("hid1", 256)), hid2=int(_m8_cfg.get("hid2", 128)),
                learning_rate=float(_m8_cfg.get("learning_rate", 1.0e-3)),
                weight_decay=float(_m8_cfg.get("weight_decay", 1.0e-5)),
                epochs=int(_m8_cfg.get("epochs", 30)),
                batch_size=int(_m8_cfg.get("batch_size", 16384)),
                early_stopping_patience=int(_m8_cfg.get("early_stopping_patience", 5)),
                feat_dropout=float(_m8_cfg.get("feat_dropout", 0.10)),
                seed=int(SEED) + 801 * (int(_ff.fold_id) + 1),
                holdout_group_id=_m2_holdout_fold,
                show_progress=False,
            ),
        )
        # Score OOF rows in chunks (avoid materialising [N_oof, D]).
        _N_OOF8 = int(len(fold_oof_df))
        p8_oof_fold = np.empty(_N_OOF8, dtype=np.float32)
        _CH8 = 200_000
        for _cs in range(0, _N_OOF8, _CH8):
            _ce = min(_cs + _CH8, _N_OOF8)
            _emb_ch = _div_of_uniq_emb[_div_of_r2u[_cs:_ce]]
            p8_oof_fold[_cs:_ce] = mlp_apply_state_batch(
                _fold_m8_state,
                subject_ids=_mef_subj_fold_oof[_cs:_ce],
                item_emb=_emb_ch,
                dense_X=(None if _m8_form_fold_oof is None else _m8_form_fold_oof[_cs:_ce]),
            )
            del _emb_ch
        p8_train_oof_acc.write_fold(fold.oof_row_idx, p8_oof_fold)
        del _fold_m8_state
        _m8_form_fold_train = None
        _m8_form_fold_oof = None
        gc.collect()
    else:
        p8_oof_fold = None

    del _div_tr_uniq_emb, _div_of_uniq_emb, _div_tr_r2u, _div_of_r2u
    del _div_tr_item_keys, _div_tr_item_idx, _div_of_item_keys, _div_of_item_idx
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

    # NOTE: Member 4 is now a TAIL-subspace logreg, fit + written in the
    # diversification block above. The dense matrix here is built solely
    # for Member 6 (FwFM). ``_fold_m4_feature_names`` is the FwFM feature
    # name vector (kept under its historical name to minimise churn).
    _fold_m4_feature_names = (
        tuple(member_feat_schema.feature_names)
        + tuple(MEMBER4_MARGINAL_FEATURE_NAMES)
    )
    # Append the M6 item-form interaction block (item_type + cot_x_*) so the
    # fold matrix matches the GLOBAL M6 width / field_ids / feature names.
    if _dense_form_on("member6"):
        _m6_form_fold_train, _ = build_item_form_block(
            _train_row_item_keys_g[fold.train_row_idx],
            _train_row_conditions_g[fold.train_row_idx],
            full=False,
        )
        X_fold_train_dense_m4 = np.concatenate(
            [X_fold_train_dense_m4, _m6_form_fold_train], axis=1,
        ).astype(np.float32, copy=False)
        _fold_m4_feature_names = _fold_m4_feature_names + tuple(DENSE_FORM_EXTRA_NAMES)
        _m6_form_fold_train = None
    # ----- Fold Member 6 (FwFM) -- on the dense hybrid matrix -----
    # FwFM fits on the [N_train, F+14(+form)] dense matrix built just above.
    if _M6_ENABLED:
        print(f"[OOF f{fold.fold_id}] Training fold Member 6 (FwFM)...")
        _m6_x_digest_fold = _content_digest(X_fold_train_dense_m4, k_rows=8192)
        _m6_y_digest_fold = _content_digest(
            _y_fold_train.astype(np.float32, copy=False), k_rows=8192,
        )
        _m6_holdout_digest_fold = _content_digest(
            _m2_holdout_fold.astype(np.int64, copy=False), k_rows=8192,
        )
        _fold_fwfm_state = cache_or_compute(
            "fwfm_state_oof_fold",
            key_inputs=(
                # v2_streaming: bumps over v2_audit1 because
                # ``fit_fwfm_member`` was rewritten to a
                # streaming data path (see the global FwFM cache
                # key for the full reasoning). Same content
                # digests, just a fresh version tag so any
                # pre-rewrite per-fold state is rebuilt.
                "m6_fwfm_oof_v3_scaled_k16", fold.fold_id, fold_suffix,
                int(X_fold_train_dense_m4.shape[1]),
                int(X_fold_train_dense_m4.shape[0]),
                int(_M6_K), str(_M6_FIELD_MODE), int(_m6_n_fields),
                round(_M6_LR, 6),
                round(_M6_WD_W, 7), round(_M6_WD_V, 7), round(_M6_WD_R, 7),
                int(_M6_EPOCHS), int(_M6_BS), int(_M6_PATIENCE),
                round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
                _m6_x_digest_fold,
                _m6_y_digest_fold,
                _m6_holdout_digest_fold,
                int(SEED),
            ),
            compute_fn=lambda _ff=fold: fit_fwfm_member(
                X=X_fold_train_dense_m4,
                y=_y_fold_train,
                feature_names=_fold_m4_feature_names,
                field_ids=_m6_field_ids,
                k=_M6_K,
                learning_rate=_M6_LR,
                weight_decay_w=_M6_WD_W,
                weight_decay_V=_M6_WD_V,
                weight_decay_r=_M6_WD_R,
                epochs=_M6_EPOCHS,
                batch_size=_M6_BS,
                val_fraction=_M6_VAL_FRAC,
                early_stopping_patience=_M6_PATIENCE,
                seed=int(SEED) + 500 * (int(_ff.fold_id) + 1),
                standardize=True,
                holdout_group_id=_m2_holdout_fold,
            ),
        )
        # Belt-and-braces FwFM teardown after every fold: the trainer
        # already releases its GPU pool, but cache HITs skip the
        # trainer entirely. Either way we force one more GC + VRAM
        # flush so the NEXT fold's M1 retraining (~3-4 GB GPU model
        # + train/val activations) doesn't OOM on a still-held FwFM
        # block from the previous fold.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ----- Fold Member 9 (FwFM clone + NCL) -- reuses M6's train matrix -----
    if _M9_ENABLED:
        print(f"[OOF f{fold.fold_id}] Training fold Member 9 (FwFM + NCL)...")
        _m9_anchor_fold = (
            _ncl_anchor_oof_m9[fold.train_row_idx]
            if (_M9_NCL_LAMBDA > 0 and _ncl_anchor_oof_m9 is not None)
            else None
        )
        _fold_m9_state = cache_or_compute(
            "member9_fwfm_ncl_state_oof_fold",
            key_inputs=(
                "m9_fwfm_ncl_oof_v1", fold.fold_id, fold_suffix,
                int(X_fold_train_dense_m4.shape[1]),
                int(X_fold_train_dense_m4.shape[0]),
                int(_M6_K), str(_M6_FIELD_MODE), int(_m6_n_fields),
                round(_M6_LR, 6),
                round(_M6_WD_W, 7), round(_M6_WD_V, 7), round(_M6_WD_R, 7),
                int(_M6_EPOCHS), int(_M6_BS), int(_M6_PATIENCE),
                _m6_x_digest_fold, _m6_y_digest_fold, _m6_holdout_digest_fold,
                int(SEED),
                ("ncl", round(_M9_NCL_LAMBDA, 5), tuple(_NCL_M9_ANCHORS)),
            ),
            compute_fn=lambda _ff=fold, _anc=_m9_anchor_fold: fit_fwfm_member(
                X=X_fold_train_dense_m4,
                y=_y_fold_train,
                feature_names=_fold_m4_feature_names,
                field_ids=_m6_field_ids,
                k=_M6_K,
                learning_rate=_M6_LR,
                weight_decay_w=_M6_WD_W,
                weight_decay_V=_M6_WD_V,
                weight_decay_r=_M6_WD_R,
                epochs=_M6_EPOCHS,
                batch_size=_M6_BS,
                val_fraction=_M6_VAL_FRAC,
                early_stopping_patience=_M6_PATIENCE,
                seed=int(SEED) + 900 * (int(_ff.fold_id) + 1),
                standardize=True,
                holdout_group_id=_m2_holdout_fold,
                ncl_anchor_preds=_anc,
                ncl_lambda=_M9_NCL_LAMBDA,
            ),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    else:
        _fold_m9_state = None

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
    # Append the M6 item-form interaction block to the OOF matrix too.
    if _dense_form_on("member6"):
        _m6_form_fold_oof, _ = build_item_form_block(
            _train_row_item_keys_g[fold.oof_row_idx],
            _train_row_conditions_g[fold.oof_row_idx],
            full=False,
        )
        X_fold_oof_dense_m4 = np.concatenate(
            [X_fold_oof_dense_m4, _m6_form_fold_oof], axis=1,
        ).astype(np.float32, copy=False)
        _m6_form_fold_oof = None
    # Member 4 (p4) was already scored + written in the diversification
    # block above (tail-subspace logreg). The dense OOF matrix here is
    # consumed by Members 6 and 9.

    # M6 (FwFM) OOF score on the dense OOF matrix.
    if _M6_ENABLED:
        p6_oof_fold = fwfm_apply_state_batch(_fold_fwfm_state, X_fold_oof_dense_m4)
        p6_train_oof_acc.write_fold(fold.oof_row_idx, p6_oof_fold)
        del _fold_fwfm_state
    else:
        p6_oof_fold = None

    # M9 (FwFM + NCL) OOF score on the same dense OOF matrix.
    if _M9_ENABLED and (_fold_m9_state is not None):
        p9_oof_fold = fwfm_apply_state_batch(_fold_m9_state, X_fold_oof_dense_m4)
        p9_train_oof_acc.write_fold(fold.oof_row_idx, p9_oof_fold)
        del _fold_m9_state
    else:
        p9_oof_fold = None

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
        + (f"  M6={_nll(p6_oof_fold):.5f}" if _M6_ENABLED else "")
        + (f"  M7={_nll(p7_oof_fold):.5f}" if (_M7_ENABLED and p7_oof_fold is not None) else "")
        + (f"  M8={_nll(p8_oof_fold):.5f}" if (_M8_ENABLED and p8_oof_fold is not None) else "")
        + (f"  M9={_nll(p9_oof_fold):.5f}" if (_M9_ENABLED and p9_oof_fold is not None) else "")
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
if _M6_ENABLED:
    p6_train_oof = p6_train_oof_acc.finalize()
else:
    p6_train_oof = None
if _M7_ENABLED:
    p7_train_oof = p7_train_oof_acc.finalize()
else:
    p7_train_oof = None
if _M8_ENABLED:
    p8_train_oof = p8_train_oof_acc.finalize()
else:
    p8_train_oof = None
if _M9_ENABLED:
    p9_train_oof = p9_train_oof_acc.finalize()
else:
    p9_train_oof = None
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
if _M6_ENABLED:
    print(f"  M6 OOF: {_nll_full(p6_train_oof):.5f}")
if _M7_ENABLED:
    print(f"  M7 OOF: {_nll_full(p7_train_oof):.5f}")
if _M8_ENABLED:
    print(f"  M8 OOF: {_nll_full(p8_train_oof):.5f}")
if _M9_ENABLED:
    print(f"  M9 OOF: {_nll_full(p9_train_oof):.5f}")
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
# ## 9f. Train the stacker (ridge logistic regression on OOF train predictions)
#
# Replaces the previous val-trained stacker. Now the stacker learns on
# OOF train predictions (each row's prediction comes from a member that
# never saw that row's item) + the per-row labels. Val is reported but
# NOT fit on -- it is now a pure holdout for the meta-learner.

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    apply_bucketed_batch as stacker_apply_bucketed_batch,
    build_stacker_features,
    fit_bucketed_stacker,
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

# Build the per-row [N, n_members] member-probability stack. Column order
# is LOCKED across val and OOF-train so the runtime's stacker.apply_one
# sees a consistent ordering. The stacker column order is always:
# M1, M2, M3, M4, [M5], [M6]. Disabled members are silently skipped.
_stacker_member_list_val = [p_a_val, p_member2_val, p_member3_val, p_member4_val]
_stacker_member_names_val = ["M1 (p_a_val)", "M2 (p_member2_val)", "M3 (p_member3_val)", "M4 (p_member4_val)"]
if _M5_ENABLED:
    _stacker_member_list_val.append(p_member5_val)
    _stacker_member_names_val.append("M5 (p_member5_val)")
if _M6_ENABLED:
    _stacker_member_list_val.append(p_member6_val)
    _stacker_member_names_val.append("M6 (p_member6_val)")
if _M7_ENABLED:
    _stacker_member_list_val.append(p_member7_val)
    _stacker_member_names_val.append("M7 (p_member7_val)")
if _M8_ENABLED:
    _stacker_member_list_val.append(p_member8_val)
    _stacker_member_names_val.append("M8 (p_member8_val)")
if _M9_ENABLED and (p_member9_val is not None):
    _stacker_member_list_val.append(p_member9_val)
    _stacker_member_names_val.append("M9 (p_member9_val)")

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
_n_members_dyn = 4 + int(_M5_ENABLED) + int(_M6_ENABLED) + int(_M7_ENABLED) + int(_M8_ENABLED) + int(_M9_ENABLED)
print(f"[Stacker] X_val (final-reporting view): {stacker_X_val.shape} "
      f"(n_members={_n_members_dyn})")

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
if _M6_ENABLED:
    _stacker_member_list_train_oof.append(p6_train_oof.astype(np.float32))
    _stacker_oof_names.append("M6 (p6_train_oof)")
if _M7_ENABLED:
    _stacker_member_list_train_oof.append(p7_train_oof.astype(np.float32))
    _stacker_oof_names.append("M7 (p7_train_oof)")
if _M8_ENABLED:
    _stacker_member_list_train_oof.append(p8_train_oof.astype(np.float32))
    _stacker_oof_names.append("M8 (p8_train_oof)")
if _M9_ENABLED and (p9_train_oof is not None):
    _stacker_member_list_train_oof.append(p9_train_oof.astype(np.float32))
    _stacker_oof_names.append("M9 (p9_train_oof)")

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


_n_stacker_members = 4 + int(_M5_ENABLED) + int(_M6_ENABLED) + int(_M7_ENABLED) + int(_M8_ENABLED) + int(_M9_ENABLED)
_stacker_feature_names = stacker_feature_names(_n_stacker_members)
assert int(stacker_X_train_oof.shape[1]) == len(_stacker_feature_names), (
    f"stacker_X_train_oof has {stacker_X_train_oof.shape[1]} cols but "
    f"expected {len(_stacker_feature_names)} for n_members={_n_stacker_members}"
)


# ---------- Bucketed stacker gate ----------
#
# The user spec asks for "two coarse buckets" -- a separate
# weight vector for well-known vs unknown benchmarks. The
# training-time analog of "unknown benchmark" is the per-row
# bc_redaction mask (``bc_redacted_train == 1`` simulates the
# cold-bc condition by zeroing M2's bc input). At runtime the
# equivalent signal is ``bench_present`` (1.0 == benchmark id
# was in the trained indexer's vocab).
#
# We deliberately bucket on ``bc_redacted`` at training time
# (not ``bench_present``, which is uniformly 1.0 on both
# train and val for item-cold-start splits), because that is
# the only per-row signal that actually has variance on the
# offline dataset.
_train_bench_known = (bc_redacted_train.astype(np.int8) == 0)
_val_bench_known = (bc_redacted_val.astype(np.int8) == 0)
print(
    f"[Stacker] bucketed gate: train known={int(_train_bench_known.sum()):,} / "
    f"unknown={int((~_train_bench_known).sum()):,} ; "
    f"val known={int(_val_bench_known.sum()):,} / "
    f"unknown={int((~_val_bench_known).sum()):,}"
)


def _fit_stacker_oof():
    return fit_bucketed_stacker(
        X=stacker_X_train_oof,
        y=ylab_train_np,
        bench_known=_train_bench_known,
        feature_names=_stacker_feature_names,
        n_iters=int(CFG.get("stacker", {}).get("n_iters", 1500)),
        learning_rate=float(CFG.get("stacker", {}).get("learning_rate", 0.05)),
        l2=float(CFG.get("stacker", {}).get("l2", 1.0)),
        early_stopping_patience=int(CFG.get("stacker", {}).get("early_stopping_patience", 200)),
        # Internal 80/20 split inside fit_stacker for early stopping;
        # fit_bucketed_stacker forwards this per-bucket.
        val_fraction=0.2,
        seed=SEED,
        min_rows_per_bucket=int(
            CFG.get("stacker", {}).get("min_rows_per_bucket", 1024)
        ),
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
# Bucket gate is part of the cache key so any change to the gate
# definition (e.g. swapping the bc_redaction signal for a count-
# based proxy in the future) invalidates the stacker cache.
_stacker_bucket_digest = _hashlib.sha256(
    np.ascontiguousarray(_train_bench_known.astype(np.uint8)).tobytes()
).hexdigest()[:16]
stacker_state = cache_or_compute(
    "stacker_state",
    key_inputs=(
        stacker_X_train_oof.shape[1], len(ylab_train_np), SEED,
        # Cache invalidators: OOF (Task 1), Member 5 (Task 4), Member 6
        # (FwFM), and the chosen M5 variant. The m5_variant tag bumps
        # the key whenever the residual_cluster vs difficulty_knn
        # selection flips so a stale stacker can never be reused on
        # numerically-different M5 OOF preds. The m6 tag does the same
        # for Member 6 add/remove. The "bucketed_v1" tag makes any
        # legacy single-stacker cache miss explicitly (BucketedStackerState
        # and StackerState are not interchangeable on the wire).
        "oof_v1",
        "bucketed_v1",
        f"m5_{_M5_VARIANT}" if _M5_ENABLED else "no_m5",
        "m6_v1" if _M6_ENABLED else "no_m6",
        int(_n_stacker_members),
        int(OOF_N_FOLDS), int(OOF_SEED),
        bool(OOF_RETRAIN_M1),
        _stacker_Xtrain_digest, _stacker_ytrain_digest,
        _stacker_bucket_digest,
        int(CFG.get("stacker", {}).get("min_rows_per_bucket", 1024)),
    ),
    compute_fn=_fit_stacker_oof,
)
print(
    "[Stacker] bucketed stacker fit OK: "
    f"known(n_train={stacker_state.n_train_known:,}, "
    f"val_loss={stacker_state.known.val_loss:.5f}, "
    f"bias={stacker_state.known.bias:+.4f}) "
    f"unknown(n_train={stacker_state.n_train_unknown:,}, "
    f"val_loss={stacker_state.unknown.val_loss:.5f}, "
    f"bias={stacker_state.unknown.bias:+.4f})"
)
print(f"[Stacker] known weights:   {stacker_state.known.weights}")
print(f"[Stacker] unknown weights: {stacker_state.unknown.weights}")

# Apply the OOF-fit bucketed stacker on (a) OOF-train inputs -- for
# Gate 1d -- and (b) val inputs -- for final reporting (val never
# touched the fit). The gate uses the same bc_redacted-derived
# bench_known signal we trained on so val rows route to the bucket
# they were learned for.
p_stacker_train_oof = stacker_apply_bucketed_batch(
    stacker_state, stacker_X_train_oof, _train_bench_known,
)
p_stacker_val = stacker_apply_bucketed_batch(
    stacker_state, stacker_X_val, _val_bench_known,
)

def _bce(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

nll_stack_train_oof = _bce(ylab_train_np, p_stacker_train_oof)
nll_stack_val = _bce(ylab_val, p_stacker_val)
nll_uniform_val = _bce(ylab_val, stacker_member_probs_val.mean(axis=1))

print(f"\n[Stacker] val log-loss summary (OOF-fit stacker, val is TRUE holdout):")
print(f"  Member 1 (Model A IRT-MLP, val):    {_bce(ylab_val, p_a_val):.6f}")
print(f"  Member 2 (metadata MLP, val):       {nll_m2:.6f}")
print(f"  Member 3 (kNN-similarity, val):     {nll_m3:.6f}")
print(f"  Member 4 (LogReg, val):             {nll_m4:.6f}")
if _M5_ENABLED:
    _m5_label = (
        "residual subj-cluster" if _M5_VARIANT == "residual_cluster"
        else "difficulty-kNN"
    )
    print(f"  Member 5 ({_m5_label}, val):  {nll_m5:.6f}")
if _M6_ENABLED:
    print(f"  Member 6 (FwFM, val):               {nll_m6:.6f}")
if _M7_ENABLED:
    print(f"  Member 7 (marginal MLP, val):       {nll_m7:.6f}")
if _M8_ENABLED:
    print(f"  Member 8 (embedding MLP, val):      {nll_m8:.6f}")
if _M9_ENABLED:
    print(f"  Member 9 (FwFM+NCL clone, val):     {nll_m9:.6f}")
print(f"  Uniform avg of {_n_stacker_members} members (val):     {nll_uniform_val:.6f}")
print(f"  STACKER (val, OOF-fit):             {nll_stack_val:.6f}")
print(f"  STACKER (OOF-train, in-sample):     {nll_stack_train_oof:.6f}")
if nll_stack_val > nll_uniform_val + 1e-3:
    print(
        "WARNING: Stacker did not beat uniform average. Consider increasing "
        "stacker.l2 or stacker.n_iters in CFG, or check that members are "
        "diverse enough."
    )

# Gate 3e (RED-TEAM): error-correlation between Member 2 and Member 1 on val.
# If the metadata MLP is still highly correlated with Member 1's errors,
# it's contributing redundant signal to the stacker and Task 3 didn't
# achieve its decorrelation goal.
_y64 = ylab_val.astype(np.float64)
_err_m1 = p_a_val.astype(np.float64) - _y64
_err_m2 = p_member2_val.astype(np.float64) - _y64
_corr_m2_m1 = float(np.corrcoef(_err_m2, _err_m1)[0, 1])
print(
    f"\n[Gate 3e] Member 2 MLP vs Member 1 error correlation (val): "
    f"corr(err_m2, err_m1) = {_corr_m2_m1:+.4f}"
)
if abs(_corr_m2_m1) > 0.85:
    print(
        f"[Gate 3e] FLAG: |corr|={abs(_corr_m2_m1):.4f} > 0.85 -- Member 2 is still "
        "strongly correlated with Member 1."
    )
else:
    print(
        f"[Gate 3e] PASS: |corr|={abs(_corr_m2_m1):.4f} <= 0.85; Member 2 errors are "
        "sufficiently decorrelated from Member 1."
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
        if _M5_VARIANT == "difficulty_knn":
            print(
                f"[Gate 4b] FLAG: |corr(err_m5, err_m3)|={abs(_corr_m5_m3):.4f} "
                ">= 0.90 -- Member 5 (difficulty-kNN) is essentially redundant "
                "with Member 3. Possible fixes:\n"
                "  1. Lower CFG['member5']['tau'] so the kernel is sharper and\n"
                "     the difficulty distance distinguishes neighbors more.\n"
                "  2. Increase CFG['member5']['ridge_alpha'] to suppress the\n"
                "     low-signal embedding dimensions in the projection.\n"
                "  3. Switch to CFG['member5']['variant']='residual_cluster'\n"
                "     to use the categorical-only subject x cluster predictor."
            )
        else:
            print(
                f"[Gate 4b] FLAG: |corr(err_m5, err_m3)|={abs(_corr_m5_m3):.4f} "
                ">= 0.90 -- Member 5 (residual subj-cluster) is unexpectedly "
                "correlated with the embedding kNN. Possible causes:\n"
                "  1. The cluster ids are tracking embedding geometry too\n"
                "     closely (try a larger n_clusters or different basis).\n"
                "  2. The residual_scale is too high; lower\n"
                "     CFG['member5']['residual_scale'] toward 0.5.\n"
                "  3. Drop Member 5 if the diversity gain doesn't justify it."
            )
    else:
        print(
            f"[Gate 4b] PASS: |corr(err_m5, err_m3)|={abs(_corr_m5_m3):.4f} "
            "< 0.90; Member 5's signal is sufficiently distinct from "
            "Member 3's."
        )
    # Report Member 5's stacker weight per bucket as a final diversity
    # diagnostic: a weight near zero (in BOTH buckets) means the
    # stacker found nothing to add.
    _w5_known = float(stacker_state.known.weights[4])  # logit_member5 is column 4
    _w5_unknown = float(stacker_state.unknown.weights[4])
    _w5_max = max(abs(_w5_known), abs(_w5_unknown))
    print(
        f"[Task 4] Member 5 stacker weight: "
        f"known={_w5_known:+.4f}  unknown={_w5_unknown:+.4f} "
        + ("(close to zero in both buckets -- stacker found little to "
           "add; consider dropping or retuning Member 5)"
           if _w5_max < 0.05
           else "(non-trivial in at least one bucket -- Member 5 is contributing)")
    )

# Gate 6 (RED-TEAM, FwFM): error-correlation between Member 6 (FwFM)
# and Member 4 (LogReg). M6 and M4 consume the SAME feature matrix;
# the inductive difference is purely the bilinear interaction term.
# If their val errors are >= 0.95 correlated, FwFM's bilinear term is
# contributing only noise relative to the linear baseline -- raise k,
# loosen weight_decay_V, or drop M6.
if _M6_ENABLED and p_member6_val is not None:
    _y64_g6 = ylab_val.astype(np.float64)
    _err_m4 = p_member4_val.astype(np.float64) - _y64_g6
    _err_m6 = p_member6_val.astype(np.float64) - _y64_g6
    _err_m1 = p_a_val.astype(np.float64) - _y64_g6
    _err_m2 = p_member2_val.astype(np.float64) - _y64_g6
    _corr_m6_m4 = float(np.corrcoef(_err_m6, _err_m4)[0, 1])
    _corr_m6_m2 = float(np.corrcoef(_err_m6, _err_m2)[0, 1])
    _corr_m6_m1 = float(np.corrcoef(_err_m6, _err_m1)[0, 1])
    print(
        f"\n[Gate 6] Member 6 (FwFM) error correlations on val:\n"
        f"  corr(err_m6, err_m4) = {_corr_m6_m4:+.4f}  "
        f"corr(err_m6, err_m2) = {_corr_m6_m2:+.4f}  "
        f"corr(err_m6, err_m1) = {_corr_m6_m1:+.4f}"
    )
    if abs(_corr_m6_m4) >= 0.95:
        print(
            f"[Gate 6] FLAG: |corr(err_m6, err_m4)|={abs(_corr_m6_m4):.4f} "
            ">= 0.95 -- FwFM's bilinear term is barely adding any signal "
            "beyond the linear LogReg baseline. Try:\n"
            "  1. Raise CFG['member6']['k'] (8 -> 16) to give V more capacity\n"
            "  2. Lower CFG['member6']['weight_decay_V'] to let V grow\n"
            "  3. Switch CFG['member6']['field_split_mode'] = 'single' or\n"
            "     a richer field schema (multiple buckets)\n"
            "  4. Drop Member 6 if the cost outweighs the gain."
        )
    else:
        print(
            f"[Gate 6] PASS: |corr(err_m6, err_m4)|={abs(_corr_m6_m4):.4f} "
            "< 0.95; FwFM's bilinear term provides signal independent of M4."
        )
    # M6 stacker weight (column index = 4 + int(_M5_ENABLED)),
    # reported per bucket.
    _w6_col = 4 + int(_M5_ENABLED)
    _w6_known = float(stacker_state.known.weights[_w6_col])
    _w6_unknown = float(stacker_state.unknown.weights[_w6_col])
    _w6_max = max(abs(_w6_known), abs(_w6_unknown))
    print(
        f"[Member 6] stacker weight (col {_w6_col}): "
        f"known={_w6_known:+.4f}  unknown={_w6_unknown:+.4f} "
        + ("(close to zero in both buckets -- the stacker found little "
           "to add from FwFM; consider tuning weight_decay_V or dropping M6)"
           if _w6_max < 0.05
           else "(non-trivial in at least one bucket -- FwFM is contributing)")
    )

# Gate 9 (NCL): does the FwFM+NCL clone (M9) actually decorrelate from its
# anchors {M1, M8} and from its twin M6? Lower |corr| vs M6's own anchor
# correlations means the NCL penalty bought genuine diversity.
if _M9_ENABLED and p_member9_val is not None:
    _y64_g9 = ylab_val.astype(np.float64)
    _err_m9 = p_member9_val.astype(np.float64) - _y64_g9
    _err_m1_g9 = p_a_val.astype(np.float64) - _y64_g9
    _corr_m9_m1 = float(np.corrcoef(_err_m9, _err_m1_g9)[0, 1])
    _corr_m9_m6 = (
        float(np.corrcoef(_err_m9, p_member6_val.astype(np.float64) - _y64_g9)[0, 1])
        if (_M6_ENABLED and p_member6_val is not None) else float("nan")
    )
    _corr_m9_m8 = (
        float(np.corrcoef(_err_m9, p_member8_val.astype(np.float64) - _y64_g9)[0, 1])
        if (_M8_ENABLED and p_member8_val is not None) else float("nan")
    )
    print(
        f"\n[Gate 9] Member 9 (FwFM+NCL) error correlations on val:\n"
        f"  corr(err_m9, err_m1) = {_corr_m9_m1:+.4f}  "
        f"corr(err_m9, err_m8) = {_corr_m9_m8:+.4f}  "
        f"corr(err_m9, err_m6) = {_corr_m9_m6:+.4f}"
    )
    _w9_col = 4 + int(_M5_ENABLED) + int(_M6_ENABLED) + int(_M7_ENABLED) + int(_M8_ENABLED)
    _w9_known = float(stacker_state.known.weights[_w9_col])
    _w9_unknown = float(stacker_state.unknown.weights[_w9_col])
    print(
        f"[Member 9] stacker weight (col {_w9_col}): "
        f"known={_w9_known:+.4f}  unknown={_w9_unknown:+.4f}"
    )

# %% [markdown]
# ## 9f-quad. Member-correlation heatmaps (val)
#
# Two heatmaps to make ensemble redundancy visible at a glance:
#
# 1. **Prediction correlation**: Pearson correlation of each pair of
#    members' val probabilities. High values mean the members produce
#    similar score *patterns* -- which is not the same as "correlated
#    errors" but is a useful directional read.
# 2. **Joint-error correlation**: Pearson correlation of each pair of
#    members' *signed errors* (``p - y``). This is the metric that
#    actually drives stacker diversity. We also print the *joint-wrong
#    rate*: for each pair, what fraction of val rows do both members
#    misclassify (using p > 0.5 as the decision boundary). High
#    joint-wrong rate means the pair's mistakes coincide, which the
#    stacker cannot fix.
#
# Both plots are saved to ``CACHE_DIR / "diagnostics" / "*.png"`` and
# returned inline. Skip silently when matplotlib is unavailable.

# %%
try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    _HEATMAP_OK = True
except Exception as _hm_exc:
    _HEATMAP_OK = False
    print(f"[Heatmaps] SKIP: matplotlib unavailable ({_hm_exc!r})")

if _HEATMAP_OK:
    _hm_names = ["M1", "M2", "M3", "M4"]
    _hm_preds = [
        p_a_val.astype(np.float64),
        p_member2_val.astype(np.float64),
        p_member3_val.astype(np.float64),
        p_member4_val.astype(np.float64),
    ]
    if _M5_ENABLED and p_member5_val is not None:
        _hm_names.append("M5")
        _hm_preds.append(p_member5_val.astype(np.float64))
    if _M6_ENABLED and p_member6_val is not None:
        _hm_names.append("M6")
        _hm_preds.append(p_member6_val.astype(np.float64))
    if _M7_ENABLED and p_member7_val is not None:
        _hm_names.append("M7")
        _hm_preds.append(p_member7_val.astype(np.float64))
    if _M8_ENABLED and p_member8_val is not None:
        _hm_names.append("M8")
        _hm_preds.append(p_member8_val.astype(np.float64))
    if _M9_ENABLED and p_member9_val is not None:
        _hm_names.append("M9")
        _hm_preds.append(p_member9_val.astype(np.float64))
    # Validate alignment defensively.
    _hm_n = int(ylab_val.shape[0])
    _hm_clean_names, _hm_clean_preds = [], []
    for _nm, _arr in zip(_hm_names, _hm_preds):
        _flat = np.asarray(_arr).reshape(-1)
        if int(_flat.shape[0]) != _hm_n:
            print(
                f"[Heatmaps] WARN: {_nm} length {_flat.shape[0]} != "
                f"ylab_val {_hm_n}; skipping this member."
            )
            continue
        _hm_clean_names.append(_nm)
        _hm_clean_preds.append(_flat.astype(np.float64))
    if len(_hm_clean_preds) < 2:
        print("[Heatmaps] SKIP: need at least 2 members with matching length.")
    else:
        _hm_stack = np.stack(_hm_clean_preds, axis=0)  # [M, N]
        _hm_M = int(_hm_stack.shape[0])
        _y_hm = ylab_val.astype(np.float64).reshape(-1)

        # Per-pair: Pearson correlation of predictions.
        _pred_corr = np.corrcoef(_hm_stack)            # [M, M]

        # Per-pair: Pearson correlation of signed errors (p - y).
        _err_stack = _hm_stack - _y_hm[None, :]
        _err_corr = np.corrcoef(_err_stack)            # [M, M]

        # Per-pair: joint-wrong rate. ``wrong[m] = (p_m > 0.5) != y``.
        # ``joint[m1, m2] = mean(wrong[m1] AND wrong[m2])``.
        _wrong = (_hm_stack > 0.5) != _y_hm[None, :].astype(bool)
        _joint_wrong = (_wrong[:, None, :] & _wrong[None, :, :]).mean(axis=2)

        # Per-member solo error rate (the diagonal of joint_wrong).
        _solo_wrong_rate = _wrong.mean(axis=1)
        print(
            "[Heatmaps] per-member val error rate (p>0.5 vs y): "
            + ", ".join(
                f"{nm}={r:.3f}" for nm, r in zip(_hm_clean_names, _solo_wrong_rate)
            )
        )

        # --- Plot helper ---
        def _plot_heatmap(M_, title_, ax_, fmt_=".2f", cmap_="RdBu_r", vmin_=None, vmax_=None):
            im = ax_.imshow(M_, cmap=cmap_, vmin=vmin_, vmax=vmax_, aspect="equal")
            ax_.set_xticks(range(len(_hm_clean_names)))
            ax_.set_yticks(range(len(_hm_clean_names)))
            ax_.set_xticklabels(_hm_clean_names)
            ax_.set_yticklabels(_hm_clean_names)
            ax_.set_title(title_, fontsize=10)
            for ii in range(M_.shape[0]):
                for jj in range(M_.shape[1]):
                    ax_.text(jj, ii, format(float(M_[ii, jj]), fmt_),
                             ha="center", va="center",
                             color="black", fontsize=8)
            return im

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        _plot_heatmap(_pred_corr, "Pearson corr of val predictions",
                      axes[0], cmap_="RdBu_r", vmin_=-1.0, vmax_=1.0)
        _plot_heatmap(_err_corr, "Pearson corr of val signed errors (p - y)",
                      axes[1], cmap_="RdBu_r", vmin_=-1.0, vmax_=1.0)
        # Joint-wrong rate is bounded in [0, 1] but typical values are
        # ~0.1-0.3; use a sequential cmap and let imshow auto-scale so
        # the contrast is informative.
        _plot_heatmap(_joint_wrong, "Joint wrong-rate (both members wrong, p>0.5)",
                      axes[2], cmap_="Oranges")
        fig.suptitle(
            f"Member correlation diagnostics on val "
            f"(N={_hm_n:,}, members={_hm_clean_names})",
            fontsize=11,
        )
        fig.tight_layout()
        try:
            _diag_dir = ROOT / "artifacts" / "diagnostics"
            _diag_dir.mkdir(parents=True, exist_ok=True)
            _out_png = _diag_dir / "member_correlation_heatmaps.png"
            fig.savefig(_out_png, dpi=120, bbox_inches="tight")
            print(f"[Heatmaps] saved to {_out_png}")
        except Exception as _save_exc:
            print(f"[Heatmaps] WARN: failed to save figure ({_save_exc!r})")
        plt.show()
        plt.close(fig)

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
    if _M6_ENABLED:
        p6_train_shuf_acc = OofPredictionAccumulator(_N_TRAIN, name="p6_train_shuf")
    else:
        p6_train_shuf_acc = None

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

        # Fold Member 2 (dense metadata MLP) on shuffled labels.
        _m2_holdout_shuf = np.array(
            [int(_item_to_train_idx.get(str(k), -1)) for k in fold_train_df["item_key"]],
            dtype=np.int64,
        )
        _m2_shuf_n_clusters = max(
            int(_N_CLUSTERS_ME),
            int(_mef_cluster_ft.max()) + 1 if _mef_cluster_ft.size else 0,
            int(_mef_cluster_fo.max()) + 1 if _mef_cluster_fo.size else 0,
        )
        _m2_meta_ft_shuf = _m2_gather_metadata(_mef_subj_ft, _mef_bc_ft)
        _m2_meta_fo_shuf = _m2_gather_metadata(_mef_subj_fo, _mef_bc_fo)
        # Mirror the metadata_only composition the real M2 fit uses,
        # so the shuffle-null sanity check tests the same architecture.
        if _M2_METADATA_ONLY:
            _n_ft_shuf = int(_m2_meta_ft_shuf["subject_numerical"].shape[0])
            _n_fo_shuf = int(_m2_meta_fo_shuf["subject_numerical"].shape[0])
            _m2_shuf_marg_ft = np.zeros((_n_ft_shuf, 0), dtype=np.float32)
            _m2_shuf_marg_fo = np.zeros((_n_fo_shuf, 0), dtype=np.float32)
            _m2_shuf_redact_ft = np.zeros(_n_ft_shuf, dtype=np.float32)
            _m2_shuf_redact_fo = np.zeros(_n_fo_shuf, dtype=np.float32)
        else:
            _m2_shuf_marg_ft = _m4_mg_ft_shuf
            _m2_shuf_marg_fo = _m4_mg_fo_shuf
            _m2_shuf_redact_ft = _bc_redacted_ft
            _m2_shuf_redact_fo = _bc_redacted_fo
        _m2_num_ft_shuf = m2_assemble_numerical(
            subject_numerical=_m2_meta_ft_shuf["subject_numerical"],
            bench_numerical=_m2_meta_ft_shuf["bench_numerical"],
            bc_redacted_flag=_m2_shuf_redact_ft,
            marginals=_m2_shuf_marg_ft,
        )
        _m2_num_fo_shuf = m2_assemble_numerical(
            subject_numerical=_m2_meta_fo_shuf["subject_numerical"],
            bench_numerical=_m2_meta_fo_shuf["bench_numerical"],
            bc_redacted_flag=_m2_shuf_redact_fo,
            marginals=_m2_shuf_marg_fo,
        )
        _fold_m2_shuf = fit_member2_metadata_mlp(
            subject_ids=_mef_subj_ft,
            bc_ids=_mef_bc_ft,
            cluster_ids=_mef_cluster_ft,
            family_ids=_m2_meta_ft_shuf["family_ids"],
            macro_family_ids=_m2_meta_ft_shuf["macro_family_ids"],
            organization_ids=_m2_meta_ft_shuf["organization_ids"],
            bench_topic_ids=_m2_meta_ft_shuf["bench_topic_ids"],
            numerical=_m2_num_ft_shuf,
            y=_y_ft_shuf,
            subject_keys=_subject_keys_ordered,
            bc_keys=_bc_keys_ordered,
            num_feature_names=_M2_NUM_FEATURE_NAMES,
            n_subjects=int(indexer.n_subjects),
            n_bcs=int(indexer.n_bc),
            n_clusters=int(_m2_shuf_n_clusters),
            n_families=int(_M2_N_FAMILIES),
            n_macro_families=int(_M2_N_MACRO_FAMILIES),
            n_organizations=int(_M2_N_ORGANIZATIONS),
            n_bench_topics=int(_M2_N_BENCH_TOPICS),
            n_subj_num=int(_M2_N_SUBJ_NUM),
            n_bench_num=int(_M2_N_BENCH_NUM),
            n_marginals=int(_M2_N_MARGINALS_ACTIVE),
            d_subj=(int(_m2_cfg.get("d_subj", 32)) if _M2_USE_SUBJ_BC else 0),
            d_bc=(int(_m2_cfg.get("d_bc", 32)) if _M2_USE_SUBJ_BC else 0),
            d_cluster=(int(_m2_cfg.get("d_cluster", 16)) if _M2_USE_CLUSTER else 0),
            d_family=int(_m2_cfg.get("d_family", 16)),
            d_macro=int(_m2_cfg.get("d_macro", 8)),
            d_org=int(_m2_cfg.get("d_org", 16)),
            d_topic=int(_m2_cfg.get("d_topic", 16)),
            hid1=int(_m2_cfg.get("hid1", 256)),
            hid2=int(_m2_cfg.get("hid2", 128)),
            n_cross_layers=int(_m2_cfg.get("n_cross_layers", 2)),
            cross_rank=int(_m2_cfg.get("cross_rank", 64)),
            learning_rate=float(_m2_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(_m2_cfg.get("weight_decay", 1.0e-5)),
            epochs=int(_m2_cfg.get("epochs", 40)),
            batch_size=int(_m2_cfg.get("batch_size", 16384)),
            val_fraction=float(_m2_cfg.get("val_fraction", 0.1)),
            early_stopping_patience=int(_m2_cfg.get("early_stopping_patience", 5)),
            cat_dropout_subject=float(_m2_cfg.get("cat_dropout_subject", 0.05)),
            cat_dropout_bc=float(_m2_cfg.get("cat_dropout_bc", 0.10)),
            cat_dropout_cluster=float(_m2_cfg.get("cat_dropout_cluster", 0.10)),
            cat_dropout_family=float(_m2_cfg.get("cat_dropout_family", 0.05)),
            cat_dropout_macro=float(_m2_cfg.get("cat_dropout_macro", 0.05)),
            cat_dropout_org=float(_m2_cfg.get("cat_dropout_org", 0.05)),
            cat_dropout_topic=float(_m2_cfg.get("cat_dropout_topic", 0.10)),
            feat_dropout=float(_m2_cfg.get("feat_dropout", 0.10)),
            warmup_epochs=int(_m2_cfg.get("warmup_epochs", 2)),
            use_cosine_schedule=bool(_m2_cfg.get("use_cosine_schedule", True)),
            ema_decay=float(_m2_cfg.get("ema_decay", 0.999)),
            snapshot_ensemble_k=int(_m2_cfg.get("snapshot_ensemble_k", 3)),
            label_smoothing=float(_m2_cfg.get("label_smoothing", 0.005)),
            mixup_alpha=float(_m2_cfg.get("mixup_alpha", 0.0)),
            seed=int(SEED) + 100 * (int(fold.fold_id) + 1) + 99999,
            holdout_group_id=_m2_holdout_shuf,
            show_progress=False,
        )
        p2_shuf_fold = m2_apply_state_batch(
            _fold_m2_shuf,
            subject_ids=_mef_subj_fo,
            bc_ids=_mef_bc_fo,
            cluster_ids=_mef_cluster_fo,
            family_ids=_m2_meta_fo_shuf["family_ids"],
            macro_family_ids=_m2_meta_fo_shuf["macro_family_ids"],
            organization_ids=_m2_meta_fo_shuf["organization_ids"],
            bench_topic_ids=_m2_meta_fo_shuf["bench_topic_ids"],
            numerical=_m2_num_fo_shuf,
        )
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
            holdout_group_id=_m2_holdout_shuf,
        )
        p4_shuf_fold = logreg_apply_state_batch(_fold_logreg_shuf, X_fo_m4)
        p4_train_shuf_acc.write_fold(fold.oof_row_idx, p4_shuf_fold)

        # Task 4: Fold Member 5 on shuffled labels. Two variants:
        #   - difficulty_knn: refit the difficulty-projected kNN on
        #     shuffled labels. If it beats chance on val we've found
        #     leakage in the difficulty pipeline.
        #   - residual_cluster: refit the subject x cluster lookup on
        #     shuffled labels; same null-detection rationale.
        if _M5_ENABLED and _M5_VARIANT == "difficulty_knn":
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
        elif _M5_ENABLED and _M5_VARIANT == "residual_cluster":
            _fold_train_subj_ids_shuf = np.fromiter(
                (int(indexer.subject_to_id.get(str(s), -1))
                 for s in fold_train_df["subject_key"]),
                dtype=np.int64, count=len(fold_train_df),
            )
            _fold_train_cluster_shuf = np.asarray(
                _mef_cluster_ft, dtype=np.int64,
            )
            _fold_oof_subj_ids_shuf = np.fromiter(
                (int(indexer.subject_to_id.get(str(s), -1))
                 for s in fold_oof_df["subject_key"]),
                dtype=np.int64, count=len(fold_oof_df),
            )
            _fold_oof_cluster_shuf = np.asarray(
                _mef_cluster_fo, dtype=np.int64,
            )
            _fold_m5res_shuf = fit_member5_residual(
                subject_ids=_fold_train_subj_ids_shuf,
                cluster_ids=_fold_train_cluster_shuf,
                labels=_y_ft_shuf.astype(np.float64),  # SHUFFLED labels
                subject_keys=list(_subject_keys_ordered),
                n_clusters=int(N_CLUSTERS_CTX),
                smoothing_cell=_M5_SMOOTH_CELL,
                smoothing_marginal=_M5_SMOOTH_MARG,
                residual_scale=_M5_RES_SCALE,
                seed=int(SEED) + 400 * (int(fold.fold_id) + 1) + 99999,
            )
            p5_shuf_fold = m5res_apply_state_batch(
                _fold_m5res_shuf,
                subject_ids=_fold_oof_subj_ids_shuf,
                cluster_ids=_fold_oof_cluster_shuf,
            )
            p5_train_shuf_acc.write_fold(fold.oof_row_idx, p5_shuf_fold)
            del _fold_m5res_shuf, _fold_train_subj_ids_shuf
            del _fold_train_cluster_shuf, _fold_oof_subj_ids_shuf
            del _fold_oof_cluster_shuf

        # Member 6 (FwFM) on shuffled labels.
        if _M6_ENABLED:
            _fold_fwfm_shuf = fit_fwfm_member(
                X=X_ft_m4, y=_y_ft_shuf,
                feature_names=(
                    tuple(member_feat_schema.feature_names)
                    + tuple(MEMBER4_MARGINAL_FEATURE_NAMES)
                ),
                field_ids=_m6_field_ids,
                k=_M6_K, learning_rate=_M6_LR,
                weight_decay_w=_M6_WD_W, weight_decay_V=_M6_WD_V,
                weight_decay_r=_M6_WD_R,
                epochs=_M6_EPOCHS, batch_size=_M6_BS,
                early_stopping_patience=_M6_PATIENCE,
                val_fraction=_M6_VAL_FRAC,
                seed=int(SEED) + 500 * (int(fold.fold_id) + 1) + 99999,
                standardize=True,
                holdout_group_id=_m2_holdout_shuf,
            )
            p6_shuf_fold = fwfm_apply_state_batch(_fold_fwfm_shuf, X_fo_m4)
            p6_train_shuf_acc.write_fold(fold.oof_row_idx, p6_shuf_fold)
            del _fold_fwfm_shuf

        # M1 real (passed through as feature).
        p_a_train_shuf_acc.write_fold(fold.oof_row_idx, p_a_oof_shuf)

        del fold_nn_index, fold_passrate_csr, fold_passrate_mask_csr, fold_cond_context
        del nn_train_mat_fold_shuf, nn_oof_mat_fold_shuf, X_ft, X_fo
        del X_ft_m4, X_fo_m4
        del _fold_item_emb_stacked, _fold_passrate_dense, _fold_passrate_mask_dense
        del _fold_m2_shuf, _fold_knn_shuf, _fold_logreg_shuf
        gc.collect()

    p_a_shuf = p_a_train_shuf_acc.finalize()
    p2_shuf = p2_train_shuf_acc.finalize()
    p3_shuf = p3_train_shuf_acc.finalize()
    p4_shuf = p4_train_shuf_acc.finalize()
    if _M5_ENABLED:
        p5_shuf = p5_train_shuf_acc.finalize()
    if _M6_ENABLED:
        p6_shuf = p6_train_shuf_acc.finalize()

    # Fit a stacker on shuffled-label OOF predictions and apply to val.
    # Match the live stacker's [N, n_members] column layout exactly so the
    # null run is comparable.
    _shuf_member_list = [p_a_shuf, p2_shuf, p3_shuf, p4_shuf]
    if _M5_ENABLED:
        _shuf_member_list.append(p5_shuf)
    if _M6_ENABLED:
        _shuf_member_list.append(p6_shuf)
    stacker_member_probs_train_shuf = np.stack(_shuf_member_list, axis=1).astype(np.float32)
    # The shuffled-label null is a leakage probe only; it intentionally
    # uses the subset of members that have a cheap shuffle-path fit (M1-M6,
    # no M7/M8). Derive the feature-name width from the ACTUAL shuf stack
    # so it stays consistent regardless of how many members were stacked.
    _n_shuf_members = int(stacker_member_probs_train_shuf.shape[1])
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
        feature_names=stacker_feature_names(_n_shuf_members),
        n_iters=int(CFG.get("stacker", {}).get("n_iters", 1500)),
        learning_rate=float(CFG.get("stacker", {}).get("learning_rate", 0.05)),
        l2=float(CFG.get("stacker", {}).get("l2", 1.0)),
        early_stopping_patience=int(CFG.get("stacker", {}).get("early_stopping_patience", 200)),
        val_fraction=0.2,
        seed=int(SEED) + 7777,
    )
    # Build a matching val feature matrix from the SAME member subset the
    # shuf stacker was fit on (the first _n_shuf_members columns of the
    # locked val stack are M1..M6 in order). Avoids a dim mismatch now
    # that the live stacker carries 8 members but the null carries <=6.
    _shuf_val_member_probs = stacker_member_probs_val[:, :_n_shuf_members]
    stacker_X_val_shuf = build_stacker_features(
        member_probs=_shuf_val_member_probs,
        bench_present=val_bench_present,
        nn_neighbor_support=val_nn_support,
        nn_mean_similarity=val_nn_mean_sim,
        centroid_distance=val_centroid_dist,
    )
    p_stacker_val_shuf = stacker_apply_batch(stacker_state_shuf, stacker_X_val_shuf)
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
          f"(stacker: {nll_stack_val:.6f}, delta: {nll_final_local - nll_stack_val:+.6f})")

    return {
        "calibrator_state": calibrator_local.state,
        "residual_table": residual_table_local,
        "p_uncal_train_stacker": p_uncal_train_stacker_local.astype(np.float32),
        "p_final_val": p_final_val_local.astype(np.float32),
        "val_neighbor_rows": val_neighbor_rows_local,
        "val_neighbor_sims": val_neighbor_sims_local,
        "nll_final": float(nll_final_local),
        "p_a_train": p1_train_local.astype(np.float32),
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
if _M6_ENABLED:
    _oof_member_preds_for_digest.append(p6_train_oof)
if _M7_ENABLED:
    _oof_member_preds_for_digest.append(p7_train_oof)
if _M8_ENABLED:
    _oof_member_preds_for_digest.append(p8_train_oof)
if _M9_ENABLED:
    _oof_member_preds_for_digest.append(p9_train_oof)
_oof_member_preds_digest = _hashlib.sha256(
    np.ascontiguousarray(
        np.stack(_oof_member_preds_for_digest, axis=1),
        dtype=np.float32,
    ).tobytes()
).hexdigest()[:16]
CALIBRATOR_KEY_INPUTS = (
    "nn_calibrator_oof_v1",
    state_fingerprint(ckpt_a_cached["model_state"]),
    state_fingerprint(member2_mlp_state),
    state_fingerprint(knn_state),
    state_fingerprint(logreg_state),
    state_fingerprint(stacker_state),
    # Task 4: include Member 5's state in the calibrator cache key so a
    # Member-5-on/off toggle (or any retuning of k/tau/ridge_alpha)
    # invalidates the calibrator entry. Without this, a stale calibrator
    # fit on 4-member stacker outputs would silently apply to 5-member
    # stacker outputs. The variant tag distinguishes residual_cluster
    # vs difficulty_knn so flipping between them busts the cache too.
    state_fingerprint(member5_state) if _M5_ENABLED else "no_m5",
    f"m5_variant_{_M5_VARIANT}",
    # Member 6 (FwFM) cache invalidator -- same discipline.
    state_fingerprint(fwfm_state) if _M6_ENABLED else "no_m6",
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
    f"nll_final={nll_final:.6f} (stacker={nll_stack_val:.6f})"
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
#          ``artifacts/{member2_metadata_mlp,member3_knn,member4_logreg,
#          stacker,nn_calibrator_stacked,residual_table}/``,
#        - copies the pure-NumPy runtime modules into ``_pure/``,
#        - patches ``model.py`` to strip ``import faiss`` and append
#          a stacker postprocessing block that reassigns ``predict``
#          to the four-member orchestration,
#        - audits the resulting bundle for forbidden imports and
#          enforces the 1500 MB ZIP cap.
#
# The exporter currently bundles Members 1-5 (legacy difficulty_knn
# variant of M5) only. Member 6 (FwFM) and the residual_cluster variant
# of M5 do not yet have runtime templates in src/_pure/, so we gate
# the export below when either is active. The training pipeline keeps
# producing val/OOF predictions and the stacker / heatmaps still work
# end-to-end -- only the Codabench bundle build is skipped. To ship,
# set CFG['member5']['variant']='difficulty_knn' and
# CFG['member6']['enabled']=False, re-run, and rerun this section.

# %%
# Diversification pass (2026-05-30): the exporter only has runtime
# templates for the legacy member set (M1-M4 hybrid logreg, M5
# difficulty_knn). The new diversification members -- M4 as a TAIL-
# subspace logreg, M5 as a tail_knn, M6 (FwFM), M7 (marginal MLP) and
# M8 (embedding MLP) -- have no src/_pure/ template yet, so the bundle
# is skipped while we evaluate offline. Training / stacker / heatmaps
# still run end-to-end.
_EXPORT_BUNDLE_OK = (
    not _M6_ENABLED
    and not _M7_ENABLED
    and not _M8_ENABLED
    and not _M9_ENABLED
    and ((not _M5_ENABLED) or _M5_VARIANT == "difficulty_knn")
)
if not _EXPORT_BUNDLE_OK:
    import warnings as _exp_warn
    _exp_warn.warn(
        "[Export] SKIPPING bundle build: the active configuration includes "
        f"diversification members (M6={_M6_ENABLED}, M7={_M7_ENABLED}, "
        f"M8={_M8_ENABLED}) or a non-difficulty_knn Member 5 "
        f"({_M5_VARIANT=}), and src/_pure/ has no runtime template for these "
        "yet. Stacker quality / heatmap diagnostics above remain valid; only "
        "the Codabench bundle is skipped."
    )
    print(
        "[Export] SKIPPED. _M5_ENABLED="
        f"{_M5_ENABLED} _M5_VARIANT={_M5_VARIANT!r} _M6_ENABLED={_M6_ENABLED}"
    )

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

# Hard gate: when M6 (FwFM) is enabled or M5 is using the new
# residual_cluster variant, the runtime in src/_pure/ does NOT yet
# ship a matching template, so the produced bundle would crash on
# Codabench. We raise SystemExit here so that running as a script
# stops cleanly and Jupyter Run All halts at this cell.
if not _EXPORT_BUNDLE_OK:
    print(
        "[Export] SKIPPED bundle build. _M5_ENABLED="
        f"{_M5_ENABLED} _M5_VARIANT={_M5_VARIANT!r} _M6_ENABLED={_M6_ENABLED}."
    )
    print(
        "[Export] To ship, set CFG['member5']['variant']='difficulty_knn' and "
        "CFG['member6']['enabled']=False, then rerun this section."
    )
    raise SystemExit(0)

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
    member2_mlp_state=member2_mlp_state,
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
    mean_encoded_stats=mean_encoded_stats,
    # Task 4: ship Member 5 (difficulty-projected kNN). The exporter
    # raises if member5_state is non-None and stacker_state has only 4
    # member columns -- that misconfig would silently drop Member 5 at
    # runtime. We're safe here: when _M5_ENABLED, the stacker above
    # was fit with [N, 5] member_probs (feature_dim==9).
    member5_state=(member5_state if _M5_ENABLED else None),
    # Ship Member 2 dense-metadata lookup tables so the runtime can
    # gather subject + benchmark categoricals + numerics per row.
    # Without these, M2 would fall through to its cold-start path on
    # every prediction (categoricals -> UNK, numerics -> 0 with
    # missing-flag=1) and lose its conditional signal.
    member2_metadata_tables={
        "subject_cat_ids": _M2_SUBJ_CAT_TABLE,
        "subject_num": _M2_SUBJ_NUM_TABLE,
        "bc_cat_ids": _M2_BC_CAT_TABLE,
        "bc_num": _M2_BC_NUM_TABLE,
        "family_col": _M2_FAMILY_COL,
        "macro_family_col": _M2_MACRO_COL,
        "organization_col": _M2_ORG_COL,
        "topic_col": _M2_TOPIC_COL,
    },
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
print(f"  Member 2 (metadata MLP)           : {_ll(p_member2_val):.6f}")
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
p_stack_a = stacker_apply_bucketed_batch(stacker_state, stacker_X_val, _val_bench_known)
p_stack_b = stacker_apply_bucketed_batch(stacker_state, stacker_X_val, _val_bench_known)
assert np.array_equal(p_stack_a, p_stack_b)
print(f"  [PASS] stacker_apply_bucketed_batch determinism (max delta: {float(np.abs(p_stack_a - p_stack_b).max())})")
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
# Member 2 MLP: cold-start UNK ids + zero numerical channel -> finite probability.
from src.member2_metadata_mlp import apply_state_one as _m2_apply_one
p2_cold = _m2_apply_one(
    member2_mlp_state,
    subject_id=-1,
    bc_id=-1,
    cluster_id=-1,
    family_id=-1,
    macro_family_id=-1,
    organization_id=-1,
    bench_topic_id=-1,
    numerical=np.zeros(int(member2_mlp_state.n_num), dtype=np.float32),
)
print(f"  [PASS] M2 cold-start UNK -> {p2_cold:.4f} (finite={np.isfinite(p2_cold)})")

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
