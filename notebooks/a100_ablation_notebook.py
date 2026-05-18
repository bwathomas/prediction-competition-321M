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

REQUIREMENTS_PATH = ROOT / "requirements.txt"
INSTALL_REQUIREMENTS = bool(int(os.environ.get("INSTALL_REQUIREMENTS", "1")))
if INSTALL_REQUIREMENTS and REQUIREMENTS_PATH.exists():
    print(f"[bootstrap] pip install -r {REQUIREMENTS_PATH}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS_PATH)],
        check=False,
    )

import json
import logging
import time
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
# We embed each unique ``item_key`` and ``subject_key`` once. Caches live in
# ``artifacts/embeddings/{encoder_slug}/``. Truncation rates and token-length
# quantiles are reported. Encoder defaults to a heavy 7B model: swap for a
# lighter one in CFG if GPU memory is tight.

# %%
from tqdm.auto import tqdm

from src.embeddings import (
    EncoderConfig,
    TransformerEmbedder,
    item_contextual_text,
    item_only_text,
    subject_text,
)


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _embed_batches_with_progress(
    *,
    texts: list[str],
    keys: list[str],
    embedder: TransformerEmbedder,
    folder,
    batch_size: int,
    desc: str,
    benchmarks: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Progress-wrapped wrapper around ``embedder.embed_texts``.

    Returns ``{key: embedding}``. Reports rate, ETA, and cache hit / miss
    counts so the user can tell whether the run is hitting cold encoder
    forward passes or just walking the on-disk cache.
    """
    assert len(texts) == len(keys), "texts and keys must have same length"
    if benchmarks is not None:
        assert len(benchmarks) == len(texts), (
            "benchmarks and texts must have same length"
        )

    n = len(texts)
    lookup: dict[str, np.ndarray] = {}

    t0 = time.time()
    last_cache_hits = embedder.stats.cache_hits
    last_cache_misses = embedder.stats.cache_misses

    pbar = tqdm(total=n, desc=desc, unit="text", dynamic_ncols=True)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_keys = keys[start:end]
        batch_benchmarks = (
            benchmarks[start:end] if benchmarks is not None else None
        )
        batch_t0 = time.time()

        vecs = embedder.embed_texts(
            batch_texts,
            folder=folder,
            keys=batch_keys,
            benchmarks=batch_benchmarks,
        )

        if not np.isfinite(vecs).all():
            raise ValueError(
                f"NaN/inf detected in embeddings for batch {start}:{end}"
            )

        for k, v in zip(batch_keys, vecs):
            lookup[k] = v

        elapsed = time.time() - t0
        done = end
        rate = done / max(elapsed, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        new_hits = embedder.stats.cache_hits - last_cache_hits
        new_misses = embedder.stats.cache_misses - last_cache_misses

        pbar.update(end - start)
        pbar.set_postfix(
            {
                "rate": f"{rate:.2f}/s",
                "elapsed": _fmt_time(elapsed),
                "ETA": _fmt_time(eta),
                "cache_hit": new_hits,
                "cache_miss": new_misses,
                "batch": _fmt_time(time.time() - batch_t0),
            }
        )

    pbar.close()

    total = time.time() - t0
    print(
        f"{desc} complete: {n:,} texts in {_fmt_time(total)} "
        f"({n / max(total, 1e-9):.2f} texts/sec)"
    )
    return lookup


# Encoder defaults are loaded from configs/default.yaml. Override CFG["encoder"]
# here if you want to A/B different encoders without editing the yaml.
enc_cfg = EncoderConfig(**CFG["encoder"])
embedder = TransformerEmbedder(enc_cfg)

print(f"Encoder             : {enc_cfg.model_id}")
print(f"Embedding dim       : {embedder.embedding_dim}")
print(f"Embedding batch size: {enc_cfg.batch_size}")
print(f"Max length          : {enc_cfg.max_length}")
print(f"Pooling             : {enc_cfg.pooling}")
print(f"Contextual items    : {enc_cfg.use_contextual_item_text}")

# Prepare unique items.
required_cols = {"item_key", "benchmark", "condition", "item_content"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"df is missing required item columns: {sorted(missing)}")

item_df = (
    df[["item_key", "benchmark", "condition", "item_content"]]
    .drop_duplicates(subset=["item_key"])
    .reset_index(drop=True)
)

if enc_cfg.use_contextual_item_text:
    item_texts = [
        item_contextual_text(b, c, t, prefix=enc_cfg.passage_prefix)
        for b, c, t in zip(
            item_df["benchmark"].astype(str),
            item_df["condition"].astype(str),
            item_df["item_content"].astype(str),
        )
    ]
    item_folder = embedder.dir_ctx
else:
    item_texts = [
        item_only_text(t, prefix=enc_cfg.passage_prefix)
        for t in item_df["item_content"].astype(str)
    ]
    item_folder = embedder.dir_item

item_keys = item_df["item_key"].astype(str).tolist()
item_benchmarks = item_df["benchmark"].astype(str).tolist()
print(f"\nUnique items to embed: {len(item_texts):,}")

# Quick timing estimate on a small sample so the user knows what to expect.
estimate_n = min(len(item_texts), max(enc_cfg.batch_size * 3, 32))
if estimate_n > 0:
    print(f"Running quick timing estimate on {estimate_n:,} items...")
    t_est = time.time()
    _ = _embed_batches_with_progress(
        texts=item_texts[:estimate_n],
        keys=item_keys[:estimate_n],
        embedder=embedder,
        folder=item_folder,
        batch_size=enc_cfg.batch_size,
        desc="Timing sample",
        benchmarks=item_benchmarks[:estimate_n],
    )
    sample_dt = time.time() - t_est
    sample_rate = estimate_n / max(sample_dt, 1e-9)
    est_full = len(item_texts) / max(sample_rate, 1e-9)
    print(
        f"Estimated full item embedding time: {_fmt_time(est_full)} "
        f"at {sample_rate:.2f} items/sec"
    )
    print(
        "Note: this estimate includes cache behavior. If the sample was "
        "already cached the estimate will be too optimistic."
    )

# Embed unique items.
t0 = time.time()
item_emb_lookup = _embed_batches_with_progress(
    texts=item_texts,
    keys=item_keys,
    embedder=embedder,
    folder=item_folder,
    batch_size=enc_cfg.batch_size,
    desc="Embedding unique items",
    benchmarks=item_benchmarks,
)
LOG.info("Embedded %d unique items in %.1fs", len(item_emb_lookup), time.time() - t0)

# Prepare unique subjects.
required_cols = {"subject_key", "subject_content"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"df is missing required subject columns: {sorted(missing)}")

subject_df = (
    df[["subject_key", "subject_content"]]
    .drop_duplicates(subset=["subject_key"])
    .reset_index(drop=True)
)
subject_keys_list = subject_df["subject_key"].astype(str).tolist()
subject_texts = [
    subject_text(t, prefix=enc_cfg.query_prefix)
    for t in subject_df["subject_content"].astype(str)
]
print(f"\nUnique subjects to embed: {len(subject_texts):,}")

t0 = time.time()
subject_emb_lookup = _embed_batches_with_progress(
    texts=subject_texts,
    keys=subject_keys_list,
    embedder=embedder,
    folder=embedder.dir_subject,
    batch_size=enc_cfg.batch_size,
    desc="Embedding unique subjects",
    benchmarks=None,
)
LOG.info(
    "Embedded %d unique subjects in %.1fs",
    len(subject_emb_lookup),
    time.time() - t0,
)

emb_stats = embedder.stats.report()
print("\nEncoder diagnostics:")
print(json.dumps(emb_stats, indent=2))

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


def _build_arrays(split_art, indexer, item_lookup, subject_lookup, use_subject_emb=False):
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
    y_train = train["label"].astype(float).to_numpy()
    y_val = val["label"].astype(float).to_numpy()
    return (
        LookupDataset(s_train, bc_train, ie_train, y_train, se_train),
        LookupDataset(s_val, bc_val, ie_val, y_val, se_val),
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

train_ds, val_ds = _build_arrays(
    primary, indexer, item_emb_lookup, subject_emb_lookup, USE_SUBJECT_EMB
)
print(f"train rows: {len(train_ds)} | val rows: {len(val_ds)}")

# %% [markdown]
# ## 11. Model sanity checks: forward pass, tiny-batch overfit, random labels

# %%
from src.models import build_model
from src.sanity_checks import (
    check_forward_pass,
    check_overfit_tiny_batch,
    check_random_labels_sanity,
)


def _model_cfg(k: int, model_name: str) -> ModelConfig:
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
# ## 13. Gated-MLP residual-strength sweep
#
# Sweeps the residual-gate initial strength on `kfactor_gated_mlp` across
# the configured ``k_factors`` and ``seeds``. Saves best checkpoint per
# run by item-cold-start val log-loss. The trainer streams JSONL progress
# events to ``PROGRESS_FILE`` so you can tail it from another shell.

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
# Gated-MLP lambda sweep
#
# We sweep the residual gate strength on `kfactor_gated_mlp` across a few
# initial values, with the gate trainable. Item-cold-start val log-loss is
# the primary metric.
# ---------------------------------------------------------------------------

model_name = "kfactor_gated_mlp"
ks = [int(k) for k in CFG["train"]["k_factors"]]
seeds = [1]

lambda_jobs = [
    {"lambda_resid_init": 0.50, "lambda_resid_trainable": True, "lambda_tag": "learn_lam050"},
    {"lambda_resid_init": 0.05, "lambda_resid_trainable": True, "lambda_tag": "learn_lam005"},
    {"lambda_resid_init": 0.10, "lambda_resid_trainable": True, "lambda_tag": "learn_lam010"},
    {"lambda_resid_init": 0.20, "lambda_resid_trainable": True, "lambda_tag": "learn_lam020"},
]

jobs = [
    (model_name, k, seed, lj)
    for k in ks
    for seed in seeds
    for lj in lambda_jobs
]

print(f"\nTotal jobs: {len(jobs)}")
print("Model:", model_name)
print("k values:", ks)
print("Seeds:", seeds)
print("Lambda configs:")
for lj in lambda_jobs:
    print(lj)

ALL_RUNS: list[dict] = []
completed_times: list[float] = []
global_t0 = time.time()

from tqdm.auto import tqdm

pbar = tqdm(jobs, desc="Gated MLP lambda sweep", unit="run", dynamic_ncols=True)

for job_idx, (model_name, k, seed, lambda_cfg) in enumerate(pbar, start=1):
    lambda_init = float(lambda_cfg["lambda_resid_init"])
    lambda_trainable = bool(lambda_cfg["lambda_resid_trainable"])
    lambda_tag = str(lambda_cfg["lambda_tag"])

    run_id = f"{model_name}_k{k}_seed{seed}_{lambda_tag}"
    model_cfg = _model_cfg(k, model_name)
    model_cfg.lambda_resid_init = lambda_init
    model_cfg.lambda_resid_trainable = lambda_trainable

    pbar.set_description(f"gated k={k} seed={seed} {lambda_tag}")

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
            "lambda_resid_init": lambda_init,
            "lambda_resid_trainable": lambda_trainable,
            "lambda_tag": lambda_tag,
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
            "lambda_resid_init": lambda_init,
            "lambda_resid_trainable": lambda_trainable,
            "lambda_tag": lambda_tag,
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
print("\n=== Gated MLP lambda sweep sorted by item-cold-start val log-loss ===")
print(runs_df.to_string(index=False))
print(f"\nTotal sweep time: {fmt_seconds(time.time() - global_t0)}")

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
    val_ds_split = _build_arrays(
        split_art, indexer, item_emb_lookup, subject_emb_lookup, USE_SUBJECT_EMB
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
# ## 15. Random-row overfit flag
#
# If a residual model only improves random-row val but not item-cold-start
# val, the residual is overfitting. We print a warning when this happens.

# %%
def _mean_metric(df, model, split, metric="val_log_loss"):
    sub = df[(df["model_name"] == model) & (df["split"] == split)]
    return float(sub[metric].mean()) if len(sub) else float("nan")


for residual_name in ("kfactor_mlp", "kfactor_gated_mlp"):
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
for chal in ("kfactor_mlp", "kfactor_gated_mlp"):
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
from src.export_submission import export_run, make_submission_zip
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

sub_dir = export_run(
    result=sel_result,
    encoder_cfg=CFG["encoder"],
    submission_dir=ROOT / CFG["submission"]["dir"],
    include_labeling=True,
    extra_models_txt=None,
)
zip_path = make_submission_zip(
    submission_dir=sub_dir,
    zip_path=ROOT / CFG["submission"]["zip_path"],
)
print(f"Submission ready: {sub_dir}")
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
print(f"Smoke test: {'OK' if ok else 'FAIL'} -- elapsed {time.time() - t0:.2f}s")

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
