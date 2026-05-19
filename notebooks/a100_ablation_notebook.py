# %% [markdown]
# # A100 ablation notebook -- Predictive AI Evaluation Challenge
#
# Compares three models under item cold-start validation:
# 1. k-factor / neural-IRT
# 2. k-factor + ordinary MLP residual
# 3. k-factor + gated MLP / SwiGLU residual
#
# Heavy lifting:
# - Downloads `aims-foundations/measurement-db` from Hugging Face.
# - Embeds every unique item / subject text once with a configurable HF
#   encoder (default: `Qwen/Qwen3-Embedding-4B`) on the A100.
# - Trains all three model variants across multiple seeds.
# - Reports primary metric = item-cold-start val log-loss (lower is better).
# - Lets you pick which run to export to a Codabench-compatible submission.
#
# IMPORTANT
# ---------
# Primary selection MUST happen on the item-cold-start split, NOT random-row.
# Random-row validation is reported only as a sanity comparator; if a model
# only improves random-row but not item-cold-start, the notebook flags it.

# %% [markdown]
# ## 0. Clone the project repo (Colab / fresh Vertex AI instances only)
#
# When running on Colab or a fresh Vertex AI Workbench instance, this cell
# clones the project repo so that ``src/``, ``configs/`` and ``scripts/`` are
# available. If those folders already exist next to the notebook (e.g. you
# launched Jupyter from inside the repo), it does nothing.
#
# To use a private fork, change ``REPO_URL`` and optionally set the
# ``GIT_AUTH_TOKEN`` env var (or paste a token into ``REPO_URL`` directly).

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/bwathomas/prediction-competition-321M.git"
REPO_NAME = "prediction-competition-321M"
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")


def _bootstrap_repo() -> Path:
    """Return the absolute path to the repo root, cloning if necessary.

    Search order:
    1. parent of this notebook (when run from the repo directly).
    2. ``./{REPO_NAME}`` under the current working directory (Colab convention).
    3. clone fresh into the cwd.
    """
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(globals()["__file__"]).resolve().parent.parent)
    candidates.append(Path.cwd() / REPO_NAME)
    candidates.append(Path.cwd())

    for cand in candidates:
        if (cand / "src" / "data.py").is_file():
            print(f"[bootstrap] using existing repo at {cand}")
            return cand

    target = Path.cwd() / REPO_NAME
    if target.exists():
        print(f"[bootstrap] {target} exists but is incomplete; pulling latest")
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", REPO_BRANCH],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "reset", "--hard", f"origin/{REPO_BRANCH}"],
            check=True,
        )
    else:
        clone_url = REPO_URL
        token = os.environ.get("GIT_AUTH_TOKEN", "").strip()
        if token and clone_url.startswith("https://github.com/"):
            # Inject token without ever printing it.
            clone_url = clone_url.replace(
                "https://github.com/", f"https://{token}@github.com/"
            )
        print(f"[bootstrap] cloning into {target}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, clone_url, str(target)],
            check=True,
        )
    return target


ROOT = _bootstrap_repo()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
print(f"ROOT (cwd)    : {ROOT}")

# %% [markdown]
# ## 1. Install pinned requirements and import the stack

# %%
import importlib
import subprocess
import sys
import time

REQUIREMENTS_PATH = ROOT / "requirements.txt"
INSTALL_REQUIREMENTS = bool(int(os.environ.get("INSTALL_REQUIREMENTS", "1")))
if INSTALL_REQUIREMENTS and REQUIREMENTS_PATH.exists():
    print(f"[bootstrap] pip install -r {REQUIREMENTS_PATH}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS_PATH)],
        check=False,
    )


def _install_flash_attention() -> bool:
    """Install Flash Attention 2 from the prebuilt wheel matching this runtime.

    The PyPI ``flash-attn`` package defaults to building from source via
    nvcc, which takes 20-40 minutes on Colab and frequently OOMs. We avoid
    that entirely by constructing the exact prebuilt wheel URL from torch's
    CUDA version, the C++ ABI flag, and the Python version, then
    pip-installing that URL directly. If that 404s (or the constructed
    triple has no wheel), we fall back to ``pip install flash-attn
    --no-build-isolation`` which lets pip's resolver find any matching
    prebuilt wheel before resorting to source. The embedder's SDPA path
    keeps the pipeline alive if both routes fail.
    """
    try:
        import flash_attn  # type: ignore  # noqa: F401

        print(f"[flash-attn] already installed: {flash_attn.__version__}")
        return True
    except Exception:
        pass

    try:
        import torch as _torch  # local import: torch was just pip-installed
    except Exception:
        print("[flash-attn] torch unavailable; skipping install")
        return False
    if not _torch.cuda.is_available():
        print("[flash-attn] no CUDA device; skipping install (SDPA fallback)")
        return False

    # Pin to a known-good release. Bump this when you upgrade torch.
    FA_VERSION = "2.7.4.post1"

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_major_minor = ".".join(_torch.__version__.split(".")[:2])  # e.g. "2.4"
    cuda_major = _torch.version.cuda.split(".")[0]                   # e.g. "12"
    # Flash-attn wheels are built with the pre-C++11 ABI for torch <2.5
    # and with cxx11abiTRUE for torch >=2.5 nightly builds. Stable torch
    # wheels still use FALSE; if you upgrade torch and hit "no matching
    # wheel," flip this to TRUE.
    cxx11_abi = "FALSE"

    wheel_name = (
        f"flash_attn-{FA_VERSION}+cu{cuda_major}torch{torch_major_minor}"
        f"cxx11abi{cxx11_abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
    )
    wheel_url = (
        f"https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v{FA_VERSION}/{wheel_name}"
    )

    print(f"[flash-attn] target: py={py_tag} torch={torch_major_minor} cuda={cuda_major}")
    print(f"[flash-attn] url   : {wheel_url}")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", wheel_url],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        try:
            import flash_attn  # type: ignore  # noqa: F401

            print(f"[flash-attn] installed: {flash_attn.__version__} in {time.time()-t0:.1f}s")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[flash-attn] post-install import failed: {exc}")
            return False

    print(
        f"[flash-attn] direct wheel install failed in {time.time()-t0:.1f}s, "
        "retrying via pip resolver"
    )
    tail = (proc.stderr or proc.stdout or "")[-400:].strip()
    if tail:
        print(f"[flash-attn] last lines:\n{tail}")

    # Fallback: let pip find any matching prebuilt. If it falls through to a
    # source build it can hang for ages -- kill the cell if that happens.
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "flash-attn"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        try:
            import flash_attn  # type: ignore  # noqa: F401

            print(f"[flash-attn] installed (fallback): {flash_attn.__version__}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[flash-attn] post-install import failed: {exc}")
            return False

    print("[flash-attn] install failed; falling back to SDPA")
    tail = (proc.stderr or proc.stdout or "")[-400:].strip()
    if tail:
        print(f"[flash-attn] last lines:\n{tail}")
    return False


INSTALL_FLASH_ATTN = bool(int(os.environ.get("INSTALL_FLASH_ATTN", "1")))
if INSTALL_REQUIREMENTS and INSTALL_FLASH_ATTN:
    _install_flash_attention()


def _install_faiss_gpu() -> bool:
    """Upgrade ``faiss-cpu`` -> ``faiss-gpu-cu12`` when running on CUDA 12.

    The training-time k-means path in ``src.clustering`` automatically uses
    FAISS GPU when ``faiss.get_num_gpus() > 0``; that requires the
    ``faiss-gpu-cu12`` wheel (the CPU build reports zero GPUs). On
    non-GPU / non-CUDA-12 environments we leave the existing ``faiss-cpu``
    install alone and the clustering step transparently falls back to
    sklearn CPU.
    """
    try:
        import torch as _torch
    except Exception:
        print("[faiss-gpu] torch unavailable; keeping faiss-cpu")
        return False
    if not _torch.cuda.is_available():
        print("[faiss-gpu] no CUDA device; keeping faiss-cpu")
        return False
    cuda_major = _torch.version.cuda.split(".")[0] if _torch.version.cuda else "0"
    if cuda_major != "12":
        print(f"[faiss-gpu] CUDA {cuda_major}.x: only cu12 wheel is published; skipping")
        return False
    try:
        import faiss  # type: ignore

        if int(getattr(faiss, "get_num_gpus", lambda: 0)()) > 0:
            print("[faiss-gpu] already installed with GPU support")
            return True
    except Exception:
        pass
    print("[faiss-gpu] installing faiss-gpu-cu12 ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "faiss-gpu-cu12"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-400:].strip()
        print(f"[faiss-gpu] install failed; clustering will use sklearn CPU\n{tail}")
        return False
    try:
        import faiss  # type: ignore  # noqa: F401

        n = int(faiss.get_num_gpus())
        print(f"[faiss-gpu] installed; visible GPUs: {n}")
        return n > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[faiss-gpu] post-install import failed: {exc}")
        return False


INSTALL_FAISS_GPU = bool(int(os.environ.get("INSTALL_FAISS_GPU", "1")))
if INSTALL_REQUIREMENTS and INSTALL_FAISS_GPU:
    _install_faiss_gpu()


import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("notebook")

print(f"Python        : {sys.version.split()[0]}")
print(f"Torch         : {torch.__version__}")

# %% [markdown]
# ## 2. Environment / GPU banner
# Fails loudly if no CUDA device is present (set ``ALLOW_CPU=1`` to override).

# %%
from src.embeddings import print_gpu_banner

print_gpu_banner(allow_cpu=bool(int(os.environ.get("ALLOW_CPU", "0"))))

# %% [markdown]
# ## 3. Load configuration
#
# Override anything you like by editing the ``CFG`` dict before running the
# downstream cells. The encoder model id is the most common knob.

# %%
with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

# Quick overrides for fast iteration. Tweak these in-place.
# CFG["data"]["max_rows_per_benchmark"] = 5000
# CFG["encoder"]["model_id"] = "sentence-transformers/all-mpnet-base-v2"
# CFG["train"]["epochs"] = 5
# CFG["train"]["seeds"] = [0]

print(json.dumps(CFG, indent=2))

# %% [markdown]
# ## 4. Resolve HF_TOKEN and log in to Hugging Face
#
# Order:
# 1. ``HF_TOKEN`` environment variable
# 2. Google Colab ``userdata.get('HF_TOKEN')`` secret (auto on Colab; create
#    it once via the Secrets panel and grant this notebook access)
# 3. Google Secret Manager secret named ``HF_TOKEN`` (if running on GCP and
#    google-cloud-secret-manager is installed)
# 4. Interactive ``getpass`` prompt
#
# The token is **never** logged or written to disk.

# %%
from src.embeddings import login_huggingface, resolve_hf_token

HF_TOKEN = resolve_hf_token(interactive=True)
login_huggingface(HF_TOKEN)

# %% [markdown]
# ## 5. Download + load + key the dataset
#
# Downloads the per-benchmark parquet files into ``artifacts/data/`` (idempotent),
# joins them with the registry tables, normalizes ``condition``, builds
# stable ``subject_key`` / ``item_key`` / ``benchmark_condition_key`` columns,
# and reports descriptive statistics.

# %%
from src.data import (
    DatasetStats,
    compute_dataset_stats,
    prepare_dataset,
    print_dataset_stats,
)

df = prepare_dataset(CFG["data"], token=HF_TOKEN, download=True)
print(f"Final dataset rows: {len(df):,}")
print(df.head(3).to_dict(orient="records"))

stats = compute_dataset_stats(df)
print_dataset_stats(stats)

# %% [markdown]
# ## 6. Build the validation splits
#
# - **item_cold_start** (PRIMARY): val item_keys disjoint from train item_keys.
# - **benchmark_heldout** (optional): hold out whole benchmarks.
# - **random_row_debug** (LEAKY): shuffle-by-row. ONLY for sanity comparison.

# %%
from src.data import (
    make_benchmark_heldout_split,
    make_item_cold_start_split,
    make_random_row_split,
)
from src.sanity_checks import (
    print_results,
    run_data_checks,
    to_dataframe,
)

SEED = int(CFG["seed"])
split_cfg = CFG["splits"]

splits = {}

splits["item_cold_start"] = make_item_cold_start_split(
    df,
    val_fraction=float(split_cfg["val_fraction"]),
    seed=SEED,
    holdout_benchmarks=split_cfg.get("holdout_benchmarks") or None,
)

if split_cfg.get("holdout_benchmarks"):
    splits["benchmark_heldout"] = make_benchmark_heldout_split(
        df,
        holdout_benchmarks=split_cfg["holdout_benchmarks"],
        seed=SEED,
    )

if split_cfg.get("enable_random_row_debug", False):
    splits["random_row_debug"] = make_random_row_split(
        df, val_fraction=float(split_cfg["val_fraction"]), seed=SEED
    )

for name, art in splits.items():
    print(
        f"[{name}] train={len(art.train):>9,}  val={len(art.val):>7,}  "
        f"val_unseen_subject={len(art.val_unseen_subject):>5,}  notes={art.notes}"
    )

# %% [markdown]
# ## 7. Data sanity checks
#
# Required columns, label range, leakage, key stability, duplicate /
# inconsistent rows.

# %%
data_checks = run_data_checks(
    df,
    train=splits["item_cold_start"].train,
    val=splits["item_cold_start"].val,
)
print_results(data_checks)

# %% [markdown]
# ## 8. Build the encoder and embed unique items / subjects
#
# We embed each unique ``item_key`` and ``subject_key`` once and persist the
# result to ``artifacts/embeddings/{encoder_slug}/`` as ``items.parquet`` /
# ``subjects.parquet`` + ``meta.json`` + ``encoding_log.json``.
#
# When ``drive_cache.enabled`` is true (the Colab default) we first try to
# pull a previously-encoded cache from Google Drive. On a content-hash hit
# the encoder is never even loaded.

# %%
import time
import warnings

from tqdm.auto import tqdm

from src.embeddings import (
    EncoderConfig,
    TransformerEmbedder,
    assert_deduplicated,
    build_unique_items,
    build_unique_subjects,
    content_hash_for_items,
    encoder_slug as _encoder_slug,
    verify_flash_attention,
)
from src import drive_cache as drive_cache_mod


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# Verify Flash Attention 2 is actually usable before constructing the
# embedder. The Python `flash_attn` shim can import even when the CUDA
# kernel binding is missing or ABI-mismatched against the running torch;
# verifying both eagerly here lets us downgrade to SDPA in the config
# instead of crashing at the first forward pass.
requested_fa = bool(CFG["encoder"].get("use_flash_attention", False))
fa_active, fa_msg = verify_flash_attention(requested_fa)
print(f"Flash Attention 2   : {'ACTIVE' if fa_active else 'OFF'} -- {fa_msg}")
if requested_fa and not fa_active:
    warnings.warn(
        "use_flash_attention=True in config but flash_attn is not importable. "
        "Downgrading to SDPA for this run."
    )
    CFG["encoder"]["use_flash_attention"] = False

# Encoder defaults are loaded from configs/default.yaml. Override CFG["encoder"]
# here if you want to A/B different encoders without editing the yaml.
enc_cfg = EncoderConfig(**CFG["encoder"])
embedder = TransformerEmbedder(enc_cfg)
slug = _encoder_slug(enc_cfg.model_id)

print(f"Encoder             : {enc_cfg.model_id}")
print(f"Embedding cache dir : {embedder.base}")
print(f"Batch size (config) : {enc_cfg.batch_size} (fallback {enc_cfg.batch_size_fallback})")
print(f"max_length          : {enc_cfg.max_length or 'auto (99th pct, /64)'}")
print(f"Use Flash Attn 2    : {enc_cfg.use_flash_attention}")
print(f"Pooling             : {enc_cfg.pooling}")
print(f"Contextual items    : {enc_cfg.use_contextual_item_text}")

# Build the dedup'd item / subject lists. Dedup is verified loudly at the
# top of every encode call -- duplicate forwards are typically the largest
# source of wasted encoder time.
required_cols = {"item_key", "benchmark", "condition", "item_content"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"df is missing required item columns: {sorted(missing)}")
required_cols = {"subject_key", "subject_content"}
missing = required_cols - set(df.columns)
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

# Content hash detects whether the underlying texts changed since the last
# encoder run; persisted to meta.json so future runs (including the Drive
# cache) can do a strict "skip encoding" check.
item_pairs = list(zip(item_keys_list, item_texts_list))
subject_pairs = list(zip(subject_keys_list, subject_texts_list))
CONTENT_HASH = content_hash_for_items(item_pairs + subject_pairs)
print(f"Content hash        : {CONTENT_HASH[:16]}...  (over {len(item_pairs):,} items + {len(subject_pairs):,} subjects)")

# Phase-level progress bar so a long encoding pass shows where we are at a
# glance. The per-batch progress comes from the embedder's built-in tqdm.
phases = [
    "drive-resolve",
    "warm-cache",
    "encode-items",
    "encode-subjects",
    "finalize",
    "drive-upload",
]
phase_pbar = tqdm(phases, desc="Embedding pipeline", leave=True)
phase_times: dict[str, float] = {}


def _next_phase(name: str) -> float:
    phase_pbar.set_postfix_str(name)
    phase_pbar.update(1)
    return time.time()


# 1. Drive cache lookup (no-op outside Colab / when disabled).
t0 = _next_phase("drive-resolve")
cache_root = ROOT / CFG["encoder"]["cache_dir"]
drive_status = drive_cache_mod.resolve_cache(
    cfg=CFG,
    encoder_slug=slug,
    local_cache_root=cache_root,
    expected_hash=CONTENT_HASH,
)
phase_times["drive_resolve"] = float(time.time() - t0)
print(f"\nDrive cache decision: {drive_status.reason}")
print(json.dumps(drive_status.as_dict(), indent=2))

# 2. Warm in-memory caches from disk (drive sync writes parquet files there).
t0 = _next_phase("warm-cache")
embedder.warm_caches_from_disk()
phase_times["warm_cache"] = float(time.time() - t0)

# 3. Encode whatever is still missing. On a strict drive hit every key is
#    already in the parquet cache and the encoder is never loaded.
t0 = _next_phase("encode-items")
print(f"\nEncoding unique items: {len(item_keys_list):,}")
item_emb_lookup, item_log = embedder.embed_unique(
    kind="item",
    keys=item_keys_list,
    texts=item_texts_list,
    benchmarks=item_benches_list,
)
phase_times["items"] = float(time.time() - t0)
LOG.info(
    "items: total=%d cached=%d encoded=%d elapsed=%s",
    item_log["n_total"],
    item_log["n_cache_hits"],
    item_log["n_encoded"],
    _fmt_time(phase_times["items"]),
)

t0 = _next_phase("encode-subjects")
print(f"\nEncoding unique subjects: {len(subject_keys_list):,}")
subject_emb_lookup, subject_log = embedder.embed_unique(
    kind="subject",
    keys=subject_keys_list,
    texts=subject_texts_list,
)
phase_times["subjects"] = float(time.time() - t0)
LOG.info(
    "subjects: total=%d cached=%d encoded=%d elapsed=%s",
    subject_log["n_total"],
    subject_log["n_cache_hits"],
    subject_log["n_encoded"],
    _fmt_time(phase_times["subjects"]),
)

# 4. Persist parquet caches + meta + encoding_log.json.
t0 = _next_phase("finalize")
embedder.finalize(
    content_hash=CONTENT_HASH,
    n_items=len(item_keys_list),
    n_subjects=len(subject_keys_list),
    extra_log={
        "items": item_log,
        "subjects": subject_log,
        "drive_cache": drive_status.as_dict(),
        "phase_seconds": phase_times,
        "flash_attention_active": fa_active,
    },
)
phase_times["finalize"] = float(time.time() - t0)

# 5. Upload to Drive if anything new was encoded (or if the on-disk hash
#    differed from Drive's).
t0 = _next_phase("drive-upload")
drive_cfg = (CFG.get("drive_cache") or {})
if (
    drive_cfg.get("enabled")
    and drive_cfg.get("upload_on_completion", True)
    and drive_status.mounted
):
    if (
        drive_status.cache_hit
        and item_log["n_encoded"] == 0
        and subject_log["n_encoded"] == 0
    ):
        print("Drive cache up to date; skipping upload.")
    else:
        drive_folder = Path(drive_cfg["folder"]) / slug
        upload_summary = drive_cache_mod.upload_from_local(
            local_folder=embedder.base,
            drive_folder=drive_folder,
        )
        print(f"Drive upload: {json.dumps(upload_summary, indent=2)}")
elif drive_cfg.get("enabled") and not drive_status.mounted:
    print("Drive cache enabled but mount unavailable -- skipping upload.")
phase_times["drive_upload"] = float(time.time() - t0)
phase_pbar.close()

emb_stats = embedder.stats.report()
print("\nPhase timings:")
for k, v in phase_times.items():
    print(f"  {k:<16s} {_fmt_time(v)}")
print("\nEncoder diagnostics:")
print(json.dumps(emb_stats, indent=2))

# %% [markdown]
# ## 8b. Pool features and k-means clustering on items
#
# Two cheap item-side channels that the new Item-IRT variants compose with
# the embedding:
#
# - **Pool features**: hand-engineered scalars (token / char length, has-latex,
#   has-code, question count, MC indicators, language). Computed once per
#   unique item, cached to ``artifacts/item_features/pool_features.parquet``,
#   z-scored using *train-only* stats persisted alongside.
# - **Clusters**: k-means on the cached item embeddings. Centroids go to
#   ``artifacts/cluster_centroids.npy`` and per-item assignments to
#   ``artifacts/item_clusters.parquet``. Cluster id 0 is reserved for UNK.

# %%
from src.clustering import fit_and_assign, load_centroids
from src.item_features import (
    POOL_FEATURE_NAMES,
    apply_zscore,
    compute_features_for_items,
    fit_zscore_stats,
    load_pool_features,
    load_zscore_stats,
    save_pool_features,
    save_zscore_stats,
)

ITEM_FEATURES_CFG = CFG.get("item_features", {}) or {}
USE_POOL_FEATURES = bool(ITEM_FEATURES_CFG.get("use_pool", True))
USE_CLUSTER_FEATURES = bool(ITEM_FEATURES_CFG.get("use_clusters", True))
POOL_FEATURE_DIM = int(ITEM_FEATURES_CFG.get("pool_feature_dim", len(POOL_FEATURE_NAMES)))
CLUSTER_EMBED_DIM = int(ITEM_FEATURES_CFG.get("cluster_embed_dim", 16))
POOL_FEATURES_DIR = ROOT / ITEM_FEATURES_CFG.get("cache_dir", "artifacts/item_features")
POOL_FEATURES_PATH = POOL_FEATURES_DIR / "pool_features.parquet"
POOL_STATS_PATH = POOL_FEATURES_DIR / "pool_features_stats.json"

CLUSTERING_CFG = CFG.get("clustering", {}) or {}
N_CLUSTERS = int(CLUSTERING_CFG.get("k", 64))
CLUSTERING_SEED = int(CLUSTERING_CFG.get("seed", 0))
CENTROIDS_PATH = ROOT / CLUSTERING_CFG.get(
    "centroids_path", "artifacts/cluster_centroids.npy"
)
ASSIGNMENTS_PATH = ROOT / CLUSTERING_CFG.get(
    "assignments_path", "artifacts/item_clusters.parquet"
)
# FAISS GPU k-means controls. For closest-to-old sklearn behavior, use
# niter=100, nredo=4. For faster feature engineering, niter=30-50 and
# nredo=1 is usually enough on A100s.
FAISS_NITER = int(CLUSTERING_CFG.get("faiss_niter", 50))
FAISS_NREDO = int(CLUSTERING_CFG.get("faiss_nredo", 1))
FAISS_ASSIGN_BATCH = int(CLUSTERING_CFG.get("faiss_assign_batch", 65536))
FAISS_GPU_ID = int(CLUSTERING_CFG.get("gpu_id", 0))
CLUSTERING_BACKEND = str(CLUSTERING_CFG.get("backend", "auto"))
OVERWRITE_CLUSTERS = bool(CLUSTERING_CFG.get("overwrite", False))

# 1) Pool features ----------------------------------------------------------
if USE_POOL_FEATURES:
    pool_df = load_pool_features(POOL_FEATURES_PATH)
    if pool_df is None or set(POOL_FEATURE_NAMES).difference(pool_df.columns):
        print(f"Computing pool features for {len(item_df):,} unique items ...")
        pool_df = compute_features_for_items(item_df, progress=True)
        save_pool_features(pool_df, POOL_FEATURES_PATH)
        print(f"Cached pool features -> {POOL_FEATURES_PATH.relative_to(ROOT)}")
    else:
        print(f"Loaded cached pool features ({len(pool_df):,} rows) from {POOL_FEATURES_PATH.relative_to(ROOT)}")
else:
    pool_df = None
    print("Pool features disabled (CFG.item_features.use_pool = false).")

print(f"Pool feature columns: {list(POOL_FEATURE_NAMES)}")

# 2) k-means on item embeddings -------------------------------------------
# Uses FAISS GPU k-means when faiss-gpu is available (~50-100x faster than
# sklearn on full-corpus A100 runs); falls back to sklearn CPU otherwise.
# Both paths produce identical on-disk artifacts.
if USE_CLUSTER_FEATURES:
    from tqdm.auto import tqdm as _tqdm

    item_keys_for_clusters = item_df["item_key"].astype(str).tolist()
    missing_keys = [k for k in item_keys_for_clusters if k not in item_emb_lookup]
    if missing_keys:
        raise KeyError(
            f"Missing {len(missing_keys):,} item embeddings before clustering. "
            f"First missing keys: {missing_keys[:10]}"
        )
    first_vec = np.asarray(item_emb_lookup[item_keys_for_clusters[0]], dtype=np.float32)
    emb_dim = int(first_vec.shape[-1])
    item_emb_matrix = np.empty((len(item_keys_for_clusters), emb_dim), dtype=np.float32)
    for i, k in _tqdm(
        enumerate(item_keys_for_clusters),
        total=len(item_keys_for_clusters),
        desc="Building item embedding matrix",
        unit="item",
        leave=False,
    ):
        vec = np.asarray(item_emb_lookup[k], dtype=np.float32)
        if vec.shape != (emb_dim,):
            raise ValueError(
                f"Embedding for item_key={k!r} has shape {vec.shape}; expected {(emb_dim,)}"
            )
        item_emb_matrix[i] = vec

    centroids, cluster_assignments = fit_and_assign(
        item_keys_for_clusters,
        item_emb_matrix,
        k=N_CLUSTERS,
        seed=CLUSTERING_SEED,
        centroids_path=CENTROIDS_PATH,
        assignments_path=ASSIGNMENTS_PATH,
        overwrite=OVERWRITE_CLUSTERS,
        niter=FAISS_NITER,
        nredo=FAISS_NREDO,
        gpu_id=FAISS_GPU_ID,
        assign_batch_size=FAISS_ASSIGN_BATCH,
        backend=CLUSTERING_BACKEND,
    )
    print(
        f"Clusters: k={N_CLUSTERS} centroids={CENTROIDS_PATH.relative_to(ROOT)} "
        f"assignments={ASSIGNMENTS_PATH.relative_to(ROOT)}"
    )
else:
    centroids = None
    cluster_assignments = None
    print("Cluster features disabled (CFG.item_features.use_clusters = false).")

# %% [markdown]
# ## 9. Embedding sanity checks

# %%
from src.sanity_checks import (
    check_embedding_determinism,
    check_embedding_nan_inf,
    check_embedding_shape,
    check_embedding_truncation,
)

all_item_emb = np.stack(list(item_emb_lookup.values()), axis=0)
all_subject_emb = np.stack(list(subject_emb_lookup.values()), axis=0)
embed_checks = [
    check_embedding_shape(embedder),
    check_embedding_nan_inf(all_item_emb),
    check_embedding_nan_inf(all_subject_emb),
    check_embedding_truncation(emb_stats),
    check_embedding_determinism(embedder),
]
print_results(embed_checks)

# %% [markdown]
# ## 10. Build the indexer and the training matrices
#
# Subject + benchmark-condition keys -> integer ids. Index 0 is UNK in both
# spaces; test-time predict() will route unseen keys there.

# %%
from src.models import Indexer, LookupDataset, ModelConfig
from src.embeddings import stack_lookup
from src.item_features import build_feature_matrix


def _pool_matrix(keys, features_df):
    """Build a [N, pool_feature_dim] z-scored matrix for the given item keys.

    Returns ``None`` if pool features are disabled / missing.
    """
    if features_df is None:
        return None
    return build_feature_matrix(
        [str(k) for k in keys],
        features_df,
        feature_cols=list(POOL_FEATURE_NAMES),
        key_col="item_key",
    )


def _cluster_vector(keys, assignments):
    """Build a [N] int64 cluster-id vector. Unknown keys map to 0 (UNK)."""
    if assignments is None:
        return None
    return np.array(
        [int(assignments.get(str(k), 0)) for k in keys], dtype=np.int64
    )


def _build_arrays(
    split_art,
    indexer,
    item_lookup,
    subject_lookup,
    *,
    use_subject_emb: bool = False,
    pool_features_z=None,
    cluster_assignments=None,
):
    train = split_art.train
    val = split_art.val
    s_train = np.array([indexer.subject_id(k) for k in train["subject_key"]])
    s_val = np.array([indexer.subject_id(k) for k in val["subject_key"]])
    bc_train = np.array([indexer.bc_id(k) for k in train["benchmark_condition_key"]])
    bc_val = np.array([indexer.bc_id(k) for k in val["benchmark_condition_key"]])
    ie_train = stack_lookup(train["item_key"], item_lookup)
    ie_val = stack_lookup(val["item_key"], item_lookup)
    se_train = None
    se_val = None
    if use_subject_emb:
        se_train = stack_lookup(train["subject_key"], subject_lookup)
        se_val = stack_lookup(val["subject_key"], subject_lookup)
    pf_train = _pool_matrix(train["item_key"], pool_features_z)
    pf_val = _pool_matrix(val["item_key"], pool_features_z)
    ci_train = _cluster_vector(train["item_key"], cluster_assignments)
    ci_val = _cluster_vector(val["item_key"], cluster_assignments)
    y_train = train["label"].astype(float).to_numpy()
    y_val = val["label"].astype(float).to_numpy()
    return (
        LookupDataset(
            s_train, bc_train, ie_train, y_train, se_train, pf_train, ci_train
        ),
        LookupDataset(
            s_val, bc_val, ie_val, y_val, se_val, pf_val, ci_val
        ),
    )


primary = splits["item_cold_start"]
indexer = Indexer.fit(
    subject_keys=primary.train["subject_key"].tolist(),
    bc_keys=primary.train["benchmark_condition_key"].tolist(),
)
print(f"Indexer: n_subjects={indexer.n_subjects}  n_bc={indexer.n_bc}")

USE_SUBJECT_EMB = False  # set True to feed subject text embeddings to the residual
SUBJECT_EMB_DIM = embedder.embedding_dim if USE_SUBJECT_EMB else 0
ITEM_EMB_DIM = embedder.embedding_dim

# Fit z-score stats on the TRAIN items only, then apply to all items.
if USE_POOL_FEATURES and pool_df is not None:
    train_item_keys = set(primary.train["item_key"].astype(str).tolist())
    train_features = pool_df[pool_df["item_key"].astype(str).isin(train_item_keys)]
    pool_stats = load_zscore_stats(POOL_STATS_PATH)
    if pool_stats is None:
        pool_stats = fit_zscore_stats(train_features, feature_cols=list(POOL_FEATURE_NAMES))
        save_zscore_stats(pool_stats, POOL_STATS_PATH)
        print(f"Fit pool-feature z-score stats on {len(train_features):,} train items -> {POOL_STATS_PATH.relative_to(ROOT)}")
    else:
        print(f"Loaded pool-feature z-score stats from {POOL_STATS_PATH.relative_to(ROOT)}")
    pool_features_z = apply_zscore(pool_df, pool_stats)
else:
    pool_stats = None
    pool_features_z = None

train_ds, val_ds = _build_arrays(
    primary,
    indexer,
    item_emb_lookup,
    subject_emb_lookup,
    use_subject_emb=USE_SUBJECT_EMB,
    pool_features_z=pool_features_z,
    cluster_assignments=cluster_assignments,
)
print(
    f"train rows: {len(train_ds)} | val rows: {len(val_ds)} | "
    f"pool_feats: {'on' if USE_POOL_FEATURES else 'off'} | "
    f"clusters: {'on' if USE_CLUSTER_FEATURES else 'off'}"
)

# %% [markdown]
# ## 11. Model sanity checks: forward pass, tiny-batch overfit, random labels

# %%
from src.models import build_model
from src.sanity_checks import (
    check_forward_pass,
    check_overfit_tiny_batch,
    check_random_labels_sanity,
)


def _model_cfg(
    k: int,
    model_name: str,
    *,
    use_pool: bool | None = None,
    use_clusters: bool | None = None,
) -> ModelConfig:
    """Construct a ModelConfig honoring the pool / cluster flags.

    ``use_pool`` and ``use_clusters`` default to the global notebook flags;
    pass explicit booleans in the ablation grid to toggle them per run.
    """
    irt_reg = (CFG["train"].get("irt_reg") or {})
    up = USE_POOL_FEATURES if use_pool is None else bool(use_pool)
    uc = USE_CLUSTER_FEATURES if use_clusters is None else bool(use_clusters)
    return ModelConfig(
        k=k,
        item_embed_dim=ITEM_EMB_DIM,
        item_map_hidden_dim=int(CFG["train"]["item_map_hidden_dim"]),
        residual_hidden_dim=int(CFG["train"]["residual_hidden_dim"]),
        dropout=float(CFG["train"]["dropout"]),
        n_subjects=indexer.n_subjects,
        n_benchmark_conditions=indexer.n_bc,
        use_subject_text_embedding=USE_SUBJECT_EMB,
        subject_embed_dim=SUBJECT_EMB_DIM,
        lambda_resid_init=float(CFG["train"]["lambda_resid_init"]),
        lambda_resid_trainable=bool(CFG["train"]["lambda_resid_trainable"]),
        use_pool_features=bool(up and pool_features_z is not None),
        pool_feature_dim=POOL_FEATURE_DIM if up else 0,
        use_cluster_features=bool(uc and cluster_assignments is not None),
        n_clusters=N_CLUSTERS if uc else 0,
        cluster_embed_dim=CLUSTER_EMBED_DIM if uc else 0,
        irt_lambda_beta=float(irt_reg.get("lambda_beta", 1.0e-4)),
        irt_lambda_alpha=float(irt_reg.get("lambda_alpha", 1.0e-4)),
    )


SMOKE_K = int(CFG["train"]["k_factors"][0])
smoke_model_cfg = _model_cfg(SMOKE_K, "kfactor")
forward_result = check_forward_pass(
    build_model("kfactor", smoke_model_cfg),
    item_emb_dim=ITEM_EMB_DIM,
    n_subjects=indexer.n_subjects,
    n_bc=indexer.n_bc,
    subject_emb_dim=SUBJECT_EMB_DIM,
)
overfit_result = check_overfit_tiny_batch(
    build_model("kfactor", smoke_model_cfg),
    item_emb_dim=ITEM_EMB_DIM,
    n_subjects=indexer.n_subjects,
    n_bc=indexer.n_bc,
    subject_emb_dim=SUBJECT_EMB_DIM,
)
random_label_result = check_random_labels_sanity(
    lambda: build_model("kfactor", smoke_model_cfg),
    item_emb_dim=ITEM_EMB_DIM,
    n_subjects=indexer.n_subjects,
    n_bc=indexer.n_bc,
)
print_results([forward_result, overfit_result, random_label_result])

# %% [markdown]
# ## 12. Baselines (computed before training the heavy models)
#
# Global mean / subject-shrinkage / benchmark-condition shrinkage / logistic
# on raw embeddings. Saved into the per-split results table.

# %%
from src.eval import (
    bc_mean_with_shrinkage,
    compute_metrics,
    global_mean_baseline,
    logistic_baseline_on_embeddings_streaming,
    subject_mean_with_shrinkage,
)

per_run_predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
all_results: list[dict] = []


def _add_baseline(name: str, split_name: str, y_val: np.ndarray, p_val: np.ndarray):
    m = compute_metrics(y_val, p_val, n_bins=int(CFG["eval"]["ece_bins"]))
    row = {
        "model_name": name,
        "split": split_name,
        "k": -1,
        "seed": 0,
        "val_log_loss": m.log_loss,
        "val_brier": m.brier,
        "val_auc": m.auc,
        "val_accuracy": m.accuracy,
        "val_ece": m.ece,
        "n_val": m.n,
    }
    all_results.append(row)
    per_run_predictions.setdefault(split_name, {})[name] = (y_val, p_val)


# Cheap baselines (every split).
for split_name, art in splits.items():
    y_val = art.val["label"].to_numpy().astype(float)
    y_train = art.train["label"].to_numpy().astype(float)

    p_gm = global_mean_baseline(y_train, y_val)
    _add_baseline("baseline_global_mean", split_name, y_val, p_gm)

    p_sub = subject_mean_with_shrinkage(art.train, art.val, alpha=20.0)
    _add_baseline("baseline_subject_shrinkage", split_name, y_val, p_sub)

    p_bc = bc_mean_with_shrinkage(art.train, art.val, alpha=20.0)
    _add_baseline("baseline_bc_shrinkage", split_name, y_val, p_bc)

# Memory-safe streaming logistic baseline on item embeddings (primary split).
# Avoids materializing [n_rows, embedding_dim] -- important for high-dim
# encoders like Qwen3-Embedding-4B (d=2560) on multi-million-row datasets.
y_val = primary.val["label"].to_numpy().astype(float)
p_log = logistic_baseline_on_embeddings_streaming(
    primary.train,
    primary.val,
    item_emb_lookup,
    batch_size=16384,        # bump to 65536 if GPU is underutilized
    epochs=3,                # baseline only; do not over-invest
    lr=1e-3,
    weight_decay=1e-4,
    bf16=bool(CFG["encoder"]["bf16"]),
)
_add_baseline("baseline_logistic_items_streaming", "item_cold_start", y_val, p_log)

baseline_df = pd.DataFrame(all_results).sort_values(["split", "val_log_loss"])
print(baseline_df.to_string(index=False))

# %% [markdown]
# ## 13. Extended ablation grid
#
# Trains every configured model variant across the configured ``k_factors``
# and ``seeds``. The three new variants (``kfactor_irt_item``,
# ``kfactor_irt_item_mlp``, ``kfactor_irt_item_gated_mlp``) add a parallel
# Item-IRT channel: ``logit = alpha(item) * (theta_subj - beta(item))`` plus
# the existing offsets and (optionally) a residual MLP that can also see
# pool features and cluster embeddings.
#
# When ``RUN_FEATURE_TOGGLE_GRID`` is true we also re-run the variants that
# *can* consume pool / cluster features with those channels off, so the
# diagnostic in cell 14b can attribute gains cleanly. Set it to ``False``
# for fast iteration.
#
# Saves best checkpoint per run by item-cold-start val log-loss. The trainer
# streams JSONL progress events to ``PROGRESS_FILE`` so you can tail it from
# another shell.

# %%
import subprocess

from src.train import TrainConfig, train_one

# Make sure the `train` logger inherits INFO from the root logger configured
# in section 1 -- without this, info-level progress lines are swallowed in
# Colab when other libraries reset the root logger.
logging.getLogger("train").setLevel(logging.INFO)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    torch.set_float32_matmul_precision("high")
else:
    print("WARNING: CUDA not available. Training will be slow.")


def fmt_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        util, mem_used, mem_total, power = [x.strip() for x in out.split(",")[:4]]
        return f"GPU util={util}% mem={mem_used}/{mem_total}MB power={power}W"
    except Exception:
        return "GPU status unavailable"


CKPT_DIR = ROOT / CFG["eval"]["checkpoints_dir"]
CKPT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = ROOT / "artifacts" / "training_progress.jsonl"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

train_cfg = TrainConfig(
    learning_rate=float(CFG["train"]["learning_rate"]),
    weight_decay=float(CFG["train"]["weight_decay"]),
    batch_size=int(CFG["train"]["batch_size"]),
    epochs=int(CFG["train"]["epochs"]),
    warmup_steps=int(CFG["train"]["warmup_steps"]),
    scheduler=str(CFG["train"]["scheduler"]),
    grad_clip=float(CFG["train"]["grad_clip"]),
    early_stopping_patience=int(CFG["train"]["early_stopping_patience"]),
    bf16=bool(CFG["encoder"]["bf16"]),
    num_workers=int(CFG["train"].get("num_workers", 0)),
    progress=True,
    log_every_batches=1,
    progress_file=str(PROGRESS_FILE),
)

print("Train config:")
print(asdict(train_cfg))
print(gpu_status())
print("Progress file:", PROGRESS_FILE)

# ---------------------------------------------------------------------------
# Extended ablation grid
#
# Trains every configured model variant. When RUN_FEATURE_TOGGLE_GRID is
# True, models that can consume pool / cluster features are also trained
# with those channels disabled so the diagnostic in cell 14b can attribute
# gains cleanly.
# ---------------------------------------------------------------------------

models_to_train = list(CFG["train"]["models"])
ks = [int(k) for k in CFG["train"]["k_factors"]]
seeds = [int(s) for s in CFG["train"]["seeds"]]

# Set to False for fast iteration; the toggle grid adds ~2x runs.
RUN_FEATURE_TOGGLE_GRID = True

# Variants that have a residual MLP can actually consume pool / cluster
# features; toggling them on the pure kfactor / pure IRT variants is a no-op
# so we skip those rows in the toggle grid.
_MLP_VARIANTS = {
    "kfactor_mlp",
    "kfactor_gated_mlp",
    "kfactor_irt_item_mlp",
    "kfactor_irt_item_gated_mlp",
}


def _feature_toggles_for(model_name: str) -> list[tuple[str, bool, bool]]:
    """Return a list of (tag, use_pool, use_clusters) to train for this model."""
    if not RUN_FEATURE_TOGGLE_GRID or model_name not in _MLP_VARIANTS:
        return [("full", USE_POOL_FEATURES, USE_CLUSTER_FEATURES)]
    return [
        ("full", USE_POOL_FEATURES, USE_CLUSTER_FEATURES),
        ("nofeat", False, False),
    ]


jobs = [
    (model_name, k, seed, tag, up, uc)
    for model_name in models_to_train
    for k in ks
    for seed in seeds
    for (tag, up, uc) in _feature_toggles_for(model_name)
]

print(f"\nTotal jobs: {len(jobs)}")
print("Models:", models_to_train)
print("k values:", ks)
print("Seeds:", seeds)
print(f"Feature-toggle grid: {'on' if RUN_FEATURE_TOGGLE_GRID else 'off'}")

ALL_RUNS: list[dict] = []
completed_times: list[float] = []
global_t0 = time.time()

from tqdm.auto import tqdm

pbar = tqdm(jobs, desc="Extended ablation grid", unit="run", dynamic_ncols=True)

for job_idx, (model_name, k, seed, tag, up, uc) in enumerate(pbar, start=1):
    run_id = f"{model_name}_k{k}_seed{seed}_{tag}"
    model_cfg = _model_cfg(k, model_name, use_pool=up, use_clusters=uc)

    pbar.set_description(f"{model_name} k={k} seed={seed} {tag}")

    print("\n" + "=" * 100)
    print(f"Starting {job_idx}/{len(jobs)}: {run_id}")
    print("Model config:")
    print(asdict(model_cfg))
    print(gpu_status())

    job_t0 = time.time()
    r = train_one(
        model_name=model_name,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        indexer=indexer,
        seed=seed,
        run_id=run_id,
        checkpoint_dir=CKPT_DIR,
        extra_metadata={
            "encoder_model_id": CFG["encoder"]["model_id"],
            "use_subject_text_embedding": USE_SUBJECT_EMB,
            "use_pool_features": bool(model_cfg.use_pool_features),
            "use_cluster_features": bool(model_cfg.use_cluster_features),
            "feature_tag": tag,
        },
    )

    job_elapsed = time.time() - job_t0
    completed_times.append(job_elapsed)

    ALL_RUNS.append(
        {
            "run_id": r.run_id,
            "model_name": r.model_name,
            "k": r.k,
            "seed": r.seed,
            "feature_tag": tag,
            "use_pool_features": bool(model_cfg.use_pool_features),
            "use_cluster_features": bool(model_cfg.use_cluster_features),
            "epoch_best": r.epoch_best,
            "best_val_log_loss": r.best_val_log_loss,
            "best_val_brier": r.best_val_brier,
            "best_val_auc": r.best_val_auc,
            "checkpoint_path": r.checkpoint_path,
            "metadata_path": r.metadata_path,
            "elapsed_seconds": r.elapsed_seconds,
        }
    )

    avg_time = sum(completed_times) / len(completed_times)
    remaining_runs = len(jobs) - job_idx
    eta = avg_time * remaining_runs
    total_elapsed = time.time() - global_t0

    pbar.set_postfix(
        {
            "last": fmt_seconds(job_elapsed),
            "avg": fmt_seconds(avg_time),
            "ETA": fmt_seconds(eta),
            "best_ll": f"{r.best_val_log_loss:.5f}",
        }
    )

    runs_df_live = pd.DataFrame(ALL_RUNS).sort_values(
        "best_val_log_loss", ascending=True
    )
    print(f"\nFinished {job_idx}/{len(jobs)}: {run_id}")
    print(f"Run time: {fmt_seconds(job_elapsed)}")
    print(f"Total elapsed: {fmt_seconds(total_elapsed)}")
    print(f"Estimated remaining: {fmt_seconds(eta)}")
    print(f"Best epoch: {r.epoch_best}")
    print(
        f"Best val log-loss: {r.best_val_log_loss:.6f} | "
        f"Brier: {r.best_val_brier:.6f} | "
        f"AUC: {r.best_val_auc if r.best_val_auc is not None else 'n/a'}"
    )
    print(gpu_status())
    print("\nCurrent top runs:")
    print(runs_df_live.head(10).to_string(index=False))

runs_df = pd.DataFrame(ALL_RUNS).sort_values("best_val_log_loss", ascending=True)
print("\n=== Extended ablation grid sorted by item-cold-start val log-loss ===")
print(runs_df.to_string(index=False))
print(f"\nTotal grid time: {fmt_seconds(time.time() - global_t0)}")

# %% [markdown]
# ## 14. Evaluate trained checkpoints on every split + slicewise metrics
#
# Builds the canonical results table and writes ``artifacts/results/results.csv``.

# %%
from src.eval import (
    attach_subject_family,
    build_results_dataframe,
    metrics_by_group,
    metrics_by_token_length,
)
from src.train import evaluate_model

RESULTS_PATH = ROOT / CFG["eval"]["results_path"]
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_for_inference(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = dict(ck["model_cfg"])
    model_cfg_obj = ModelConfig(**cfg_dict)
    return ck, model_cfg_obj


per_run_split_predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}


def _predict_with_checkpoint(model_name: str, ckpt_path: Path, split_art) -> np.ndarray:
    ck, mcfg = _load_for_inference(ckpt_path)
    mdl = build_model(model_name, mcfg)
    mdl.load_state_dict(ck["model_state"], strict=False)
    # Mirror the checkpoint's feature-channel config when building the val
    # dataset so pool / cluster channels line up with the trained model.
    pf_z = pool_features_z if getattr(mcfg, "use_pool_features", False) else None
    ca = cluster_assignments if getattr(mcfg, "use_cluster_features", False) else None
    val_ds_split = _build_arrays(
        split_art,
        indexer,
        item_emb_lookup,
        subject_emb_lookup,
        use_subject_emb=USE_SUBJECT_EMB,
        pool_features_z=pf_z,
        cluster_assignments=ca,
    )[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_bs = max(int(CFG["train"]["batch_size"]), 4096)
    ll, brier_val, auc_val, p_val, y_val = evaluate_model(
        mdl, val_ds_split, device=device, batch_size=eval_bs, bf16=bool(CFG["encoder"]["bf16"])
    )
    return p_val, y_val


for r in ALL_RUNS:
    ckpt = Path(r["checkpoint_path"])
    if not ckpt.exists():
        continue
    for split_name, art in splits.items():
        # Build per-split dataset using THIS split's train/val
        if split_name != "item_cold_start":
            split_indexer = Indexer.fit(
                subject_keys=art.train["subject_key"].tolist(),
                bc_keys=art.train["benchmark_condition_key"].tolist(),
            )
            # NOTE: we only EVAL the existing checkpoint, which was trained on
            # the item_cold_start split. Indices may not align if subjects /
            # bc don't overlap; we therefore evaluate using the ORIGINAL
            # indexer (mapping new keys to UNK).
            pass
        p_val, y_val = _predict_with_checkpoint(r["model_name"], ckpt, art)
        m = compute_metrics(y_val, p_val, n_bins=int(CFG["eval"]["ece_bins"]))
        row = {
            "model_name": r["model_name"],
            "run_id": r["run_id"],
            "split": split_name,
            "k": r["k"],
            "seed": r["seed"],
            "val_log_loss": m.log_loss,
            "val_brier": m.brier,
            "val_auc": m.auc,
            "val_accuracy": m.accuracy,
            "val_ece": m.ece,
            "n_val": m.n,
        }
        all_results.append(row)
        per_run_split_predictions.setdefault(split_name, {})[r["run_id"]] = (y_val, p_val)

results_df = build_results_dataframe(
    all_results, primary_split="item_cold_start", primary_metric="val_log_loss"
)
results_df.to_csv(RESULTS_PATH, index=False)
print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
print(results_df.to_string(index=False))

# %% [markdown]
# ## 14b. Feature contribution and component decomposition (diagnostic)
#
# Two analyses on the *item-cold-start val split* for the best-performing
# trained model:
#
# **Analysis A -- Leave-one-out feature ablation.** For each channel in the
# model (pool features, cluster embedding, IRT alpha, IRT beta, residual
# MLP, and each individual pool feature), we zero / clamp that channel at
# inference and record val NLL / AUC vs. the unablated baseline. The
# inference-only ablation is fast and directionally honest: it shows how
# much *the trained model relies on* each channel today. To upgrade to a
# "retrain only the residual head" estimate (more accurate when channels
# strongly interact), retrain the residual MLP with the same masks applied
# during training -- the helpers below take a model factory so wrapping
# that loop is straightforward.
#
# **Analysis B -- Logit-component decomposition.** For each val example we
# decompose the final logit into its additive components (IRT, offsets,
# MLP) and report Var, Pearson(c_i, y), Solo NLL (a 2-param logistic on c_i
# alone, so the metric reflects information not scale), and Solo AUC.
#
# Save CSVs to ``artifacts/results/`` and plots to ``artifacts/plots/``.

# %%
import torch
import torch.nn.functional as F

from src.eval import (
    component_decomposition_table,
    feature_ablation_table,
    plot_component_variance,
    plot_feature_ablation,
)
from src.item_features import POOL_FEATURE_NAMES
from src.models import build_model as _build_model_for_diag
from src.train import evaluate_model as _eval_for_diag


def _predict_with_ablation(
    model: torch.nn.Module,
    val_ds_split,
    *,
    device: str,
    batch_size: int,
    bf16: bool,
    pool_mask: np.ndarray | None = None,
    zero_cluster: bool = False,
    force_alpha_one: bool = False,
    force_beta_zero: bool = False,
    zero_mlp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run val inference with a specific channel zeroed / forced.

    Returns ``(probs, y_true)``. Designed to call the same forward path as
    training: pool features are multiplied element-wise by ``pool_mask``
    (default all-ones) so individual features can be zeroed surgically.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        val_ds_split,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast("cpu", enabled=False)
    )
    preds, targets = [], []
    with torch.inference_mode():
        with autocast:
            for batch in loader:
                s, bc, ie, se, pf, ci, y = [b.to(device, non_blocking=True) for b in batch]
                se_use = se if se.shape[-1] > 0 else None
                pf_use = pf if pf.shape[-1] > 0 else None
                ci_use = ci if ci.numel() > 0 else None
                if pf_use is not None and pool_mask is not None:
                    mask = torch.from_numpy(np.asarray(pool_mask, dtype=np.float32)).to(
                        pf_use.device
                    )
                    pf_use = pf_use * mask
                if zero_cluster and ci_use is not None:
                    ci_use = torch.zeros_like(ci_use)
                kwargs: dict = {}
                if force_alpha_one and getattr(model, "has_irt_heads", False):
                    kwargs["override_alpha"] = torch.ones(
                        s.shape[0], device=device, dtype=torch.float32
                    )
                if force_beta_zero and getattr(model, "has_irt_heads", False):
                    kwargs["override_beta"] = torch.zeros(
                        s.shape[0], device=device, dtype=torch.float32
                    )
                if zero_mlp and getattr(model, "has_residual", False) and hasattr(
                    model, "lambda_resid"
                ):
                    kwargs["override_mlp_zero"] = True
                try:
                    logits = model(s, bc, ie, se_use, pf_use, ci_use, **kwargs)
                except TypeError:
                    # The model doesn't expose override kwargs (e.g. pure
                    # kfactor); fall back to the plain forward.
                    logits = model(s, bc, ie, se_use, pf_use, ci_use)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                preds.append(probs)
                targets.append(y.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _decompose_on_val(
    model: torch.nn.Module,
    val_ds_split,
    *,
    device: str,
    batch_size: int,
    bf16: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run ``model.decompose`` across val_ds_split and stack components."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        val_ds_split,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast("cpu", enabled=False)
    )
    parts: dict[str, list[np.ndarray]] = {}
    ys: list[np.ndarray] = []
    with torch.inference_mode():
        with autocast:
            for batch in loader:
                s, bc, ie, se, pf, ci, y = [b.to(device, non_blocking=True) for b in batch]
                se_use = se if se.shape[-1] > 0 else None
                pf_use = pf if pf.shape[-1] > 0 else None
                ci_use = ci if ci.numel() > 0 else None
                d = model.decompose(s, bc, ie, se_use, pf_use, ci_use)
                for name, tensor in d.items():
                    # We only care about per-row scalar components for the
                    # table (irt / offset / mlp / factor). Skip multi-dim
                    # auxiliary entries like theta/beta_i/alpha_i for the
                    # table but expose them via the dict if useful later.
                    if tensor.dim() == 1:
                        parts.setdefault(name, []).append(
                            tensor.float().cpu().numpy()
                        )
                ys.append(y.float().cpu().numpy())
    stacked = {k: np.concatenate(v) for k, v in parts.items()}
    return stacked, np.concatenate(ys)


PLOTS_DIR = ROOT / CFG["eval"]["plots_dir"]
RESULTS_DIR = (ROOT / CFG["eval"]["results_path"]).parent
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

best_row = (
    runs_df.sort_values("best_val_log_loss", ascending=True).iloc[0]
    if len(runs_df)
    else None
)
if best_row is None:
    print("No trained runs found; skipping diagnostic.")
else:
    BEST_RUN_ID = str(best_row["run_id"])
    BEST_MODEL_NAME = str(best_row["model_name"])
    BEST_CKPT = Path(str(best_row["checkpoint_path"]))
    print(f"Diagnostic on best run: {BEST_RUN_ID}  (model={BEST_MODEL_NAME})")

    ck_diag = torch.load(BEST_CKPT, map_location="cpu")
    mcfg_diag = ModelConfig(**dict(ck_diag["model_cfg"]))
    diag_device = "cuda" if torch.cuda.is_available() else "cpu"
    diag_bs = max(int(CFG["train"]["batch_size"]), 4096)
    diag_bf16 = bool(CFG["encoder"]["bf16"])

    mdl_diag = _build_model_for_diag(BEST_MODEL_NAME, mcfg_diag).to(diag_device)
    mdl_diag.load_state_dict(ck_diag["model_state"], strict=False)

    diag_pf_z = pool_features_z if getattr(mcfg_diag, "use_pool_features", False) else None
    diag_ca = cluster_assignments if getattr(mcfg_diag, "use_cluster_features", False) else None
    _, val_ds_diag = _build_arrays(
        primary,
        indexer,
        item_emb_lookup,
        subject_emb_lookup,
        use_subject_emb=USE_SUBJECT_EMB,
        pool_features_z=diag_pf_z,
        cluster_assignments=diag_ca,
    )

    # ----- Analysis A: feature ablation ------------------------------------
    ablations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    p_full, y_full = _predict_with_ablation(
        mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs, bf16=diag_bf16
    )
    ablations["full"] = (p_full, y_full)

    if getattr(mcfg_diag, "use_pool_features", False):
        mask_zero = np.zeros(int(mcfg_diag.pool_feature_dim), dtype=np.float32)
        ablations["without_pool"] = _predict_with_ablation(
            mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
            bf16=diag_bf16, pool_mask=mask_zero,
        )
        # individual pool features
        names = list(POOL_FEATURE_NAMES)
        for i, name in enumerate(names[: int(mcfg_diag.pool_feature_dim)]):
            mask_one = np.ones(int(mcfg_diag.pool_feature_dim), dtype=np.float32)
            mask_one[i] = 0.0
            ablations[f"pool_feature[{name}]"] = _predict_with_ablation(
                mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
                bf16=diag_bf16, pool_mask=mask_one,
            )

    if getattr(mcfg_diag, "use_cluster_features", False):
        ablations["without_cluster"] = _predict_with_ablation(
            mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
            bf16=diag_bf16, zero_cluster=True,
        )

    if getattr(mdl_diag, "has_irt_heads", False):
        ablations["without_alpha"] = _predict_with_ablation(
            mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
            bf16=diag_bf16, force_alpha_one=True,
        )
        ablations["without_beta"] = _predict_with_ablation(
            mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
            bf16=diag_bf16, force_beta_zero=True,
        )

    if getattr(mdl_diag, "has_residual", False):
        ablations["without_mlp"] = _predict_with_ablation(
            mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs,
            bf16=diag_bf16, zero_mlp=True,
        )

    ablation_df = feature_ablation_table(ablations, full_key="full")
    ablation_csv = RESULTS_DIR / f"feature_ablation_{BEST_RUN_ID}.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    ablation_plot = PLOTS_DIR / f"feature_ablation_{BEST_RUN_ID}.png"
    plot_feature_ablation(
        ablation_df, ablation_plot, title=f"Feature ablation: {BEST_RUN_ID}"
    )
    print("\n=== Analysis A: feature ablation ===")
    print(ablation_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Wrote {ablation_csv.relative_to(ROOT)}")
    print(f"Wrote {ablation_plot.relative_to(ROOT)}")

    # ----- Analysis B: logit-component decomposition ----------------------
    components, y_dec = _decompose_on_val(
        mdl_diag, val_ds_diag, device=diag_device, batch_size=diag_bs, bf16=diag_bf16
    )
    decomp_df = component_decomposition_table(components, y_dec)
    decomp_csv = RESULTS_DIR / f"component_decomp_{BEST_RUN_ID}.csv"
    decomp_df.to_csv(decomp_csv, index=False)
    decomp_plot = PLOTS_DIR / f"component_decomp_{BEST_RUN_ID}.png"
    plot_component_variance(
        components, decomp_plot, title=f"Component variance: {BEST_RUN_ID}"
    )
    print("\n=== Analysis B: logit-component decomposition ===")
    print(decomp_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Wrote {decomp_csv.relative_to(ROOT)}")
    print(f"Wrote {decomp_plot.relative_to(ROOT)}")

    # One-line summaries
    non_full = ablation_df[ablation_df["channel_removed"] != "full"]
    if not non_full.empty:
        top = non_full.sort_values("delta_nll", ascending=False).iloc[0]
        print(
            f"\nLargest single contributor: {top['channel_removed']} "
            f"(Δ NLL = {float(top['delta_nll']):.5f})"
        )
    if not decomp_df.empty:
        best_comp = decomp_df.sort_values("solo_nll", ascending=True).iloc[0]
        print(
            f"Best solo component: {best_comp['component']} "
            f"(Solo NLL = {float(best_comp['solo_nll']):.5f})"
        )

# %% [markdown]
# ## 15. Random-row overfit flag
#
# If a residual model only improves random-row val but not item-cold-start
# val, the residual is overfitting. We print a warning when this happens.

# %%
def _mean_metric(df, model, split, metric="val_log_loss"):
    sub = df[(df["model_name"] == model) & (df["split"] == split)]
    return float(sub[metric].mean()) if len(sub) else float("nan")


for residual_name in (
    "kfactor_mlp",
    "kfactor_gated_mlp",
    "kfactor_irt_item",
    "kfactor_irt_item_mlp",
    "kfactor_irt_item_gated_mlp",
):
    base = "kfactor"
    if "random_row_debug" not in {row["split"] for row in all_results}:
        continue
    base_ic = _mean_metric(results_df, base, "item_cold_start")
    res_ic = _mean_metric(results_df, residual_name, "item_cold_start")
    base_rr = _mean_metric(results_df, base, "random_row_debug")
    res_rr = _mean_metric(results_df, residual_name, "random_row_debug")
    improves_random = res_rr < base_rr - 1e-4
    improves_cold = res_ic < base_ic - 1e-4
    if improves_random and not improves_cold:
        print(
            f"WARN: {residual_name} improves random-row "
            f"({base_rr:.5f} -> {res_rr:.5f}) but NOT item-cold-start "
            f"({base_ic:.5f} -> {res_ic:.5f}). Likely overfitting."
        )
    else:
        print(
            f"OK  : {residual_name} item-cold-start {base_ic:.5f} -> {res_ic:.5f} "
            f"(random-row {base_rr:.5f} -> {res_rr:.5f})"
        )

# %% [markdown]
# ## 16. Slicewise metrics: by benchmark / condition / subject family / token length

# %%
attached = attach_subject_family(primary.val.copy())

# Pick the BEST run for slicewise plots
best_run = (
    runs_df.sort_values("best_val_log_loss", ascending=True).iloc[0]
    if len(runs_df)
    else None
)
if best_run is not None:
    p_best, y_best = _predict_with_checkpoint(
        best_run["model_name"], Path(best_run["checkpoint_path"]), primary
    )
    attached["_pred"] = p_best
    by_bench = metrics_by_group(attached, group_col="benchmark", min_n=20)
    by_cond = metrics_by_group(attached, group_col="condition", min_n=20)
    by_fam = metrics_by_group(attached, group_col="subject_family", min_n=50)
    print("=== by benchmark ===")
    print(by_bench.to_string(index=False))
    print("=== by condition ===")
    print(by_cond.to_string(index=False))
    print("=== by subject family ===")
    print(by_fam.to_string(index=False))

# %% [markdown]
# ## 17. Plots: log-loss by model, calibration curves, by benchmark, residual delta

# %%
from src.eval import (
    plot_calibration_curves,
    plot_logloss_by_benchmark,
    plot_residual_improvement_by_benchmark,
    plot_val_logloss_by_model,
)

PLOTS = ROOT / CFG["eval"]["plots_dir"]
PLOTS.mkdir(parents=True, exist_ok=True)

primary_results = results_df[results_df["split"] == "item_cold_start"]
plot_val_logloss_by_model(primary_results, PLOTS / "val_logloss_by_model.png")

per_run_cal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for r in ALL_RUNS:
    rid = r["run_id"]
    if rid in per_run_split_predictions.get("item_cold_start", {}):
        per_run_cal[rid] = per_run_split_predictions["item_cold_start"][rid]
if per_run_cal:
    plot_calibration_curves(per_run_cal, PLOTS / "calibration_curves.png")

per_run_per_bench = {}
for r in ALL_RUNS:
    p_val, y_val = per_run_split_predictions.get("item_cold_start", {}).get(
        r["run_id"], (None, None)
    )
    if p_val is None:
        continue
    val = primary.val.copy()
    val["_pred"] = p_val
    per_run_per_bench[r["run_id"]] = metrics_by_group(val, group_col="benchmark", min_n=20)
plot_logloss_by_benchmark(per_run_per_bench, PLOTS / "logloss_by_benchmark.png")

# Residual deltas: kfactor vs kfactor_mlp / kfactor_gated_mlp (averaged over seeds)
def _avg_per_bench(model_name: str):
    parts = []
    for r in ALL_RUNS:
        if r["model_name"] != model_name:
            continue
        p_val, y_val = per_run_split_predictions.get("item_cold_start", {}).get(
            r["run_id"], (None, None)
        )
        if p_val is None:
            continue
        v = primary.val.copy()
        v["_pred"] = p_val
        parts.append(metrics_by_group(v, group_col="benchmark", min_n=20))
    if not parts:
        return pd.DataFrame(columns=["benchmark", "log_loss"])
    return (
        pd.concat(parts)
        .groupby("benchmark")["log_loss"]
        .mean()
        .reset_index()
    )


base_pb = _avg_per_bench("kfactor")
for chal in (
    "kfactor_mlp",
    "kfactor_gated_mlp",
    "kfactor_irt_item",
    "kfactor_irt_item_mlp",
    "kfactor_irt_item_gated_mlp",
):
    chal_pb = _avg_per_bench(chal)
    if not base_pb.empty and not chal_pb.empty:
        plot_residual_improvement_by_benchmark(
            base_pb,
            chal_pb,
            PLOTS / f"residual_delta_{chal}_vs_kfactor.png",
            base_label="kfactor",
            challenger_label=chal,
        )

# Performance vs item token length
import collections

token_len_map: dict[str, int] = {}
if not bool(CFG["encoder"].get("use_random_embeddings", False)):
    # The encoder's stats record per-batch lengths; we don't have a
    # per-item map directly. Fall back to character length / 4 as a cheap
    # proxy for the plot.
    pass
primary_val = primary.val.copy()
primary_val["item_token_len"] = (
    primary_val["item_content"].astype(str).str.len() // 4 + 1
)
per_run_by_len = {}
for r in ALL_RUNS:
    p_val, y_val = per_run_split_predictions.get("item_cold_start", {}).get(
        r["run_id"], (None, None)
    )
    if p_val is None:
        continue
    v = primary_val.copy()
    v["_pred"] = p_val
    per_run_by_len[r["run_id"]] = metrics_by_token_length(v)
from src.eval import plot_perf_vs_token_length
plot_perf_vs_token_length(per_run_by_len, PLOTS / "perf_vs_item_length.png")

print(f"Plots written to {PLOTS.resolve()}")

# %% [markdown]
# ## 18. Choose a trained run to export
#
# Default: best item-cold-start val log-loss. Override by setting
# ``SELECTED_RUN_ID`` below. If ``ipywidgets`` is installed, a dropdown lets
# you pick interactively.

# %%
SELECTED_RUN_ID: str | None = None  # set manually to override

try:
    import ipywidgets as widgets  # type: ignore
    from IPython.display import display  # type: ignore

    options = [
        (
            f"{r['run_id']}  ll={r['best_val_log_loss']:.5f}",
            r["run_id"],
        )
        for _, r in runs_df.iterrows()
    ]
    dropdown = widgets.Dropdown(options=options, description="Run:")
    display(dropdown)
    selected_widget = dropdown
except Exception:
    selected_widget = None

if SELECTED_RUN_ID is None:
    if selected_widget is not None and selected_widget.value:
        SELECTED_RUN_ID = selected_widget.value
    elif len(runs_df) > 0:
        SELECTED_RUN_ID = runs_df.iloc[0]["run_id"]
print(f"SELECTED_RUN_ID = {SELECTED_RUN_ID}")

# %% [markdown]
# ## 19. Export the selected run as a submission folder + zip
#
# Produces:
# - ``submission/model.py`` (self-contained runtime)
# - ``submission/labeling.py`` (uncertainty acquisition)
# - ``submission/models.txt``
# - ``submission/requirements.txt``
# - ``submission/artifacts/checkpoint.pt`` + ``runtime_meta.json``
# - ``submission.zip``

# %%
from src.export_submission import bundle_training_cache, export_run, make_submission_zip
from src.train import TrainResult

selected = runs_df[runs_df["run_id"] == SELECTED_RUN_ID].iloc[0]
selected_dict = selected.to_dict()
selected_meta = json.loads(Path(selected_dict["metadata_path"]).read_text())
sel_result = TrainResult(
    run_id=selected_dict["run_id"],
    model_name=selected_dict["model_name"],
    seed=int(selected_dict["seed"]),
    k=int(selected_dict["k"]),
    epoch_best=int(selected_meta["result"]["epoch_best"]),
    best_val_log_loss=float(selected_dict["best_val_log_loss"]),
    best_val_brier=float(selected_dict["best_val_brier"]),
    best_val_auc=(
        float(selected_dict["best_val_auc"])
        if selected_dict["best_val_auc"] is not None
        else None
    ),
    history=list(selected_meta["result"]["history"]),
    checkpoint_path=str(selected_dict["checkpoint_path"]),
    metadata_path=str(selected_dict["metadata_path"]),
    n_train=int(selected_meta["result"].get("n_train", 0)),
    n_val=int(selected_meta["result"].get("n_val", 0)),
    elapsed_seconds=float(selected_dict.get("elapsed_seconds", 0.0)),
)

# 19a. Build the quantized training-item cache (int8 + optional PCA + FAISS).
# This is the artifact shipped inside submission/cache/ for runtime nearest-
# neighbor lookup. Fails loudly if max_bundle_size_mb is exceeded.
training_cache_dir = ROOT / "artifacts" / "submission_cache"
submission_cache_cfg = CFG.get("submission_cache", {}) or {}
training_cache_result = None
if bool(submission_cache_cfg.get("enabled", True)):
    cluster_assign_map = (
        dict(cluster_assignments) if cluster_assignments is not None else None
    )
    training_cache_result = bundle_training_cache(
        items_parquet_path=embedder.items_path,
        out_dir=training_cache_dir,
        submission_cache_cfg=submission_cache_cfg,
        encoder_cfg=CFG["encoder"],
        items_meta_df=item_df,
        cluster_assignments=cluster_assign_map,
        n_clusters=N_CLUSTERS if USE_CLUSTER_FEATURES else 0,
        train_df=primary.train,
    )
    print(
        f"Training cache: {training_cache_result.total_mb:.2f} MB at "
        f"{training_cache_dir.relative_to(ROOT)}"
    )
    for fname, mb in training_cache_result.sizes_mb.items():
        print(f"  {fname:32s} {mb:7.2f} MB")
else:
    print("submission_cache.enabled = false; not shipping training-item cache")

sub_dir = export_run(
    result=sel_result,
    encoder_cfg=CFG["encoder"],
    submission_dir=ROOT / CFG["submission"]["dir"],
    include_labeling=True,
    extra_models_txt=None,
    pool_stats_path=POOL_STATS_PATH if USE_POOL_FEATURES else None,
    cluster_centroids_path=CENTROIDS_PATH if USE_CLUSTER_FEATURES else None,
    pool_feature_names=list(POOL_FEATURE_NAMES),
    training_cache_dir=training_cache_dir if training_cache_result is not None else None,
)
zip_path = make_submission_zip(
    submission_dir=sub_dir,
    zip_path=ROOT / CFG["submission"]["zip_path"],
)
sub_bundle_mb = sum(
    p.stat().st_size for p in sub_dir.rglob("*") if p.is_file()
) / (1024 * 1024)
print(f"Submission ready: {sub_dir}")
print(f"Submission size : {sub_bundle_mb:.2f} MB")
print(f"Zip             : {zip_path}")

# %% [markdown]
# ## 20. Submission smoke test (notebook variant)
#
# Imports ``submission/model.py``, calls predict() on 20 held-out rows,
# checks output types, range, finiteness, and timing.

# %%
import importlib
import io

sub_dir_str = str(sub_dir.resolve())
if sub_dir_str not in sys.path:
    sys.path.insert(0, sub_dir_str)
if "model" in sys.modules:
    del sys.modules["model"]
sub_model = importlib.import_module("model")

smoke_rows = primary.val.sample(n=min(20, len(primary.val)), random_state=0)
ok = True
t0 = time.time()
for _, row in smoke_rows.iterrows():
    inp = {
        "benchmark": str(row["benchmark"]),
        "condition": str(row["condition"]),
        "subject_content": str(row["subject_content"]),
        "item_content": str(row["item_content"]),
    }
    p = sub_model.predict(inp, None)
    if not (isinstance(p, float) and np.isfinite(p) and 0.0 <= p <= 1.0):
        print(f"FAIL: {p!r} for inp keys {list(inp)}")
        ok = False
print(f"Smoke test (predict): {'OK' if ok else 'FAIL'} -- elapsed {time.time() - t0:.2f}s")

# 20b. Nearest-neighbor lookup smoke test. Pick 3 val items, fetch the
# encoder embedding from the model module, and verify TRAINING_CACHE
# returns well-formed (indices, scores) of the right shape and value range.
training_cache = getattr(sub_model, "TRAINING_CACHE", None)
nn_ok = True
if training_cache is None:
    print("NN smoke test: SKIP (TRAINING_CACHE not loaded)")
else:
    K_NN = 10
    nn_rows = primary.val.sample(n=min(3, len(primary.val)), random_state=1)
    n_total = int(training_cache.embeddings_q.shape[0])
    for _, row in nn_rows.iterrows():
        item_emb = sub_model._get_item_embedding(
            str(row["benchmark"]),
            str(row["condition"]),
            str(row["item_content"]),
        )
        idx, scores = training_cache.nearest(item_emb, k=K_NN)
        if idx.shape != (K_NN,) or scores.shape != (K_NN,):
            print(f"FAIL: bad NN shapes idx={idx.shape} scores={scores.shape}")
            nn_ok = False
            continue
        if not np.all(np.isfinite(scores)):
            print(f"FAIL: non-finite NN scores {scores}")
            nn_ok = False
            continue
        if int(idx.min()) < 0 or int(idx.max()) >= n_total:
            print(f"FAIL: NN indices out of range: min={idx.min()} max={idx.max()} n={n_total}")
            nn_ok = False
            continue
        print(
            f"NN OK item_key={str(row['item_key'])[:8]}... "
            f"top-{K_NN} score range [{float(scores.min()):.3f}, {float(scores.max()):.3f}]"
        )
    print(f"Smoke test (NN lookup): {'OK' if nn_ok else 'FAIL'}")

# %% [markdown]
# ## 21. Optional GCS sync
#
# If you set ``CFG['gcs']['bucket']`` to a `gs://...` prefix, this cell copies
# the ``artifacts/`` and ``submission/`` directories there. Never syncs
# environment variables, tokens, or .ipynb_checkpoints.

# %%
gcs_bucket = (CFG.get("gcs") or {}).get("bucket")
if gcs_bucket and CFG.get("gcs", {}).get("sync_artifacts", False):
    from google.cloud import storage  # type: ignore

    client = storage.Client()
    bucket_name = gcs_bucket.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_bucket.replace("gs://", "").split("/")[1:]).strip("/")
    bucket = client.bucket(bucket_name)
    for root_dir in ("artifacts", "submission"):
        rd = ROOT / root_dir
        if not rd.exists():
            continue
        for path in rd.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            key = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
            blob = bucket.blob(key)
            blob.upload_from_filename(str(path))
    print(f"Synced artifacts/ and submission/ to {gcs_bucket}")
else:
    print("GCS sync skipped (CFG['gcs']['bucket'] not set or sync_artifacts=False)")
