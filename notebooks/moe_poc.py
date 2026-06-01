# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # MoE proof-of-concept on the M8 architecture
#
# **Question.** Does training one Member-8 instance per bucket (upweighted
# on its own bucket, full data otherwise) and hard-routing val rows to
# the matching expert beat a single M8 trained on all data -- and is the
# win actually attributable to **routing-specific specialization** rather
# than a free sample-weight training trick?
#
# **Hypothesis.** Different input regions live on qualitatively different
# feature manifolds, so a single global MLP is forced to compromise
# across them. Per-region experts free up capacity for within-region
# structure and produce errors that disagree on an axis the global model
# cannot exploit.
#
# **Partition axes (EXP['partition_axis']):**
# * ``item_type``  -- K=4 form buckets (math/code/mcq/prose). The v1 POC.
#   Coarse + heavily imbalanced (~75% prose). Showed -0.011 nat total lift
#   but controls revealed ~70% of it was the global focal-style trick.
# * ``nn_support`` -- **default**. K=N octiles on item-embedding NN
#   density (mean cosine sim to top-5 nearest TRAIN items, leave-one-out
#   for train items). Balanced by construction; directly aligned to the
#   cold-start axis the test set is heavy on; expected to give wider
#   per-bucket NLL spread => more routing-specific lift potential.
#
# **What this notebook reports** (mandatory controls baked in):
# 1. Baseline (uniform M8) -- the apples-to-apples reference.
# 2. K experts (M8 per bucket, 5x upweight on own bucket, full data).
# 3. **Hard-routed MoE** -- the "real" MoE prediction.
# 4. **Scrambled routing** -- route to a RANDOM expert. Isolates the
#    global lift (focal trick) from routing-specific lift.
# 5. **Uniform avg of experts** -- diversity sanity check (does naive
#    averaging beat the best single expert?).
# 6. **Best single expert** -- floor for any routing scheme.
# 7. **Per-bucket diagonal** -- does each expert actually win on its own
#    bucket?
# 8. **Residual-vs-baseline correlation heatmap** -- the diversity number
#    a stacker / FWLS merge actually sees (NOT signed-error corr, which
#    is misleading for strong models).
#
# **Decision rule:**
# * routing_lift (routed - scrambled) <= -0.002 AND >= K/2 diagonal wins
#   -> escalate this partition (soft routing, weight sweep, more buckets).
# * routing_lift ~0 BUT global_lift (scrambled - baseline) <= -0.003
#   -> upweighting is a free training trick; apply to vanilla M8.
# * neither -> partition doesn't pay. Try a different axis.
#
# **All shared infrastructure (data, embeddings, OOF folds, dense block,
# M8 trainer) is reused from the production stack and the loss-diversity
# probe. This notebook adds ONLY the MoE-specific cells.**

# %% [markdown]
# ## 0. Colab bootstrap (clone repo, mount Drive, set cwd)
#
# Mirrors the production notebook so the data + embedding caches resolve
# identically. Safe to run locally too (the Colab branch is skipped).

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/bwathomas/prediction-competition-321M.git"
REPO_DIR = Path("/content/Prediction-Competition-321M")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    elif not REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True
        )
    os.chdir(REPO_DIR)
    subprocess.run(
        ["pip", "install", "-q", "faiss-gpu-cu12", "sentence-transformers", "tqdm"],
        check=False,
    )

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
print(f"Working directory: {ROOT}")
print(f"In Colab: {IN_COLAB}")

# %% [markdown]
# ## 1. Config + experiment knobs

# %%
import numpy as np
import yaml

with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

CFG["encoder"].update({
    "model_id": "Qwen/Qwen3-Embedding-8B",
    "max_length": 512,
    "batch_size": 8,
    "use_flash_attention": True,
    "trust_remote_code": False,
    "pooling": "last_token",
    "use_contextual_item_text": True,
})

SEED = int(CFG["seed"])

# ---- Experiment configuration -------------------------------------------
# Same compute budget as the loss-diversity probe so the comparison is
# apples-to-apples. K experts + 1 baseline = K+1 OOF passes over the M8
# architecture.
EXP = {
    # Data / compute budget
    "max_train_rows": 600_000,
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,
    "emb_device": None,
    "predict_chunk": 131_072,

    # Feature channels (held FIXED across baseline + experts).
    "use_metadata": True,

    # Shared M8 architecture (held FIXED across baseline + experts).
    "subj_emb_dim": 32,
    "hid1": 256,
    "hid2": 128,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "feat_dropout": 0.10,

    # ---- MoE-specific ---------------------------------------------------
    # Sample-weight multiplier on rows whose bucket matches the expert.
    # 5x is the recommended starting point: enough to bias the gradient
    # toward the region without near-zero gradient on out-of-region rows
    # (the failure mode of hard partitioning).
    "expert_weight_multiplier": 5.0,

    # Partition axis for the K experts. Two implementations:
    #   "item_type"  -> K=4 mutually-exclusive form buckets from
    #                   src.item_features.item_type_onehot (priority
    #                   code > mcq > math > prose). Coarse + heavily
    #                   imbalanced (~75% prose) -- the v1 POC.
    #   "nn_support" -> K=n_buckets octiles on item-embedding neighborhood
    #                   density (mean cosine sim to top-k nearest TRAIN
    #                   items, leave-one-out for train items so an item
    #                   is not its own neighbor). Bucket boundaries are
    #                   fit on the TRAIN distribution; val items
    #                   distribute by how isolated they are (bucket 0 =
    #                   coldest support). Balanced by construction +
    #                   directly aligned to the cold-start axis the
    #                   test set is heavy on.
    "partition_axis": "nn_support",
    "n_buckets": 8,
    "type_buckets": ("type_code", "type_mcq", "type_math", "type_prose"),
    "support_k_neighbors": 5,

    "routing": "hard",
    "seed": SEED,
}

print("Experiment config:")
for k, v in EXP.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Load data + item-cold split (item-based sub-sample)
#
# Identical to the loss-diversity probe: full val, item-based sub-sample
# of train rows so cold-start guarantees and rows-per-item distributions
# are preserved.

# %%
import pandas as pd

from src.data import (
    compute_dataset_stats,
    make_item_cold_start_split,
    prepare_dataset,
    print_dataset_stats,
)
from src.embeddings import login_huggingface, resolve_hf_token

HF_TOKEN = resolve_hf_token()
if HF_TOKEN:
    login_huggingface(HF_TOKEN)

df = prepare_dataset(CFG["data"], token=HF_TOKEN, download=True)
print(f"Dataset rows: {len(df):,}")
print_dataset_stats(compute_dataset_stats(df))

primary = make_item_cold_start_split(
    df, val_fraction=float(CFG["splits"]["val_fraction"]), seed=SEED,
)
print(f"train rows: {len(primary.train):,}  val rows: {len(primary.val):,}")

_rng = np.random.default_rng(SEED)
_full_train = primary.train.reset_index(drop=True)
_N_full = len(_full_train)
_max_rows = EXP["max_train_rows"]
if _max_rows is not None and _N_full > int(_max_rows):
    _full_keys = _full_train["item_key"].astype(str).to_numpy()
    _uniq_keys, _rows_per_item = np.unique(_full_keys, return_counts=True)
    _perm = _rng.permutation(len(_uniq_keys))
    _shuffled_counts = _rows_per_item[_perm]
    _cumrows = np.cumsum(_shuffled_counts)
    _cutoff = int(np.searchsorted(_cumrows, int(_max_rows), side="left")) + 1
    _cutoff = min(_cutoff, len(_uniq_keys))
    _kept_items = set(_uniq_keys[_perm[:_cutoff]].tolist())
    _keep_mask = np.fromiter((k in _kept_items for k in _full_keys),
                             count=_N_full, dtype=bool)
    train_df = _full_train.iloc[_keep_mask].reset_index(drop=True)
    print(f"[subsample] item-based: kept {len(_kept_items):,} of "
          f"{len(_uniq_keys):,} unique items -> {len(train_df):,} of "
          f"{_N_full:,} rows (target ~{_max_rows:,})")
else:
    train_df = _full_train
    print(f"[subsample] using ALL {len(train_df):,} train rows "
          f"({train_df['item_key'].nunique():,} unique items)")
val_df = primary.val.reset_index(drop=True)

y_train = train_df["label"].astype(float).to_numpy().astype(np.float32)
y_val = val_df["label"].astype(float).to_numpy().astype(np.float32)
N_TRAIN = int(len(train_df))
N_VAL = int(len(val_df))

# %% [markdown]
# ## 3. Item embeddings (cache-aware; no re-encoding)

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

_enc_known = {f.name for f in _dc_fields(EncoderConfig)}
_enc_kwargs = {k: v for k, v in CFG["encoder"].items() if k in _enc_known}
enc_cfg = EncoderConfig(**_enc_kwargs)
embedder = TransformerEmbedder(enc_cfg)
slug = encoder_slug(enc_cfg.model_id)
print(f"Encoder: {enc_cfg.model_id}  cache: {embedder.base}")

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

# Hydrate the local encoder cache from Drive BEFORE warming the in-memory
# index. On a fresh Colab kernel the local artifacts dir is empty, so
# without this step `warm_caches_from_disk` finds nothing and `embed_unique`
# re-encodes every item. Mirrors the production notebook exactly so the
# probe hits the same Drive cache the main pipeline populates.
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
    kind="item", keys=item_keys_list, texts=item_texts_list,
    benchmarks=item_benches_list,
)
print(f"  cached={item_log['n_cache_hits']}  encoded={item_log['n_encoded']}")

# If the probe encoded anything new (shouldn't happen if the production
# cache is up to date, but is the safety net), upload back to Drive so
# future runs hit cache too.
embedder.finalize(
    content_hash=CONTENT_HASH,
    n_items=len(item_keys_list),
    n_subjects=0,
    extra_log={"items": item_log, "drive_cache": drive_status.as_dict()},
)
_drive_cfg = CFG.get("drive_cache") or {}
if (
    _drive_cfg.get("enabled")
    and _drive_cfg.get("upload_on_completion", True)
    and getattr(drive_status, "mounted", False)
    and item_log["n_encoded"]
):
    _drive_folder = Path(_drive_cfg["folder"]) / slug
    print(drive_cache_mod.upload_from_local(
        local_folder=embedder.base, drive_folder=_drive_folder,
    ))

ITEM_EMB_DIM = int(embedder.embedding_dim)
print(f"Item embedding dim D = {ITEM_EMB_DIM}")

# %% [markdown]
# ## 4. Subject indexer + unique-embedding stack + per-row pointers

# %%
from src.models import Indexer

indexer = Indexer.fit(
    subject_keys=primary.train["subject_key"].tolist(),
    bc_keys=primary.train["benchmark_condition_key"].tolist(),
)
print(f"Indexer: n_subjects={indexer.n_subjects}  n_bc={indexer.n_bc}")


def _subject_ids(rows_df) -> np.ndarray:
    return np.fromiter(
        (indexer.subject_id(str(s)) for s in rows_df["subject_key"]),
        count=int(len(rows_df)), dtype=np.int64,
    )


subj_train = _subject_ids(train_df)
subj_val = _subject_ids(val_df)

_train_keys = train_df["item_key"].astype(str).to_numpy()
_val_keys = val_df["item_key"].astype(str).to_numpy()

_all_keys = sorted(set(_train_keys.tolist()) | set(_val_keys.tolist()))
_all_keys = [k for k in _all_keys if k in item_emb_lookup]
_key_to_idx = {k: i for i, k in enumerate(_all_keys)}
_U = len(_all_keys)
_ZERO_IDX = _U

ALL_UNIQ = np.zeros((_U + 1, ITEM_EMB_DIM), dtype=np.float32)
for _i, _k in enumerate(_all_keys):
    ALL_UNIQ[_i] = item_emb_lookup[_k]

r2u_train = np.fromiter(
    (_key_to_idx.get(str(k), _ZERO_IDX) for k in _train_keys),
    count=N_TRAIN, dtype=np.int64,
)
r2u_val = np.fromiter(
    (_key_to_idx.get(str(k), _ZERO_IDX) for k in _val_keys),
    count=N_VAL, dtype=np.int64,
)
print(f"ALL_UNIQ: {ALL_UNIQ.shape}  (~{ALL_UNIQ.nbytes / 1e9:.2f} GB)")

# %% [markdown]
# ## 5. Item-cold OOF folds (same machinery as production)

# %%
from src.oof_folds import make_item_grouped_folds
from src.oof_pipeline import OofPredictionAccumulator

folds = make_item_grouped_folds(
    item_keys_per_row=_train_keys,
    n_folds=int(EXP["n_folds"]),
    seed=int(CFG.get("oof", {}).get("seed", 7)),
)
print(f"Built {len(folds)} item-cold folds:")
for f in folds:
    print(f"  fold {f.fold_id}: train_rows={len(f.train_row_idx):,}  "
          f"oof_rows={len(f.oof_row_idx):,}  oof_items={len(f.oof_item_keys):,}")

# Hard invariants -- same checks the loss-diversity probe runs.
for f in folds:
    _tr_items = set(_train_keys[f.train_row_idx].tolist())
    _oo_items = set(_train_keys[f.oof_row_idx].tolist())
    if _tr_items & _oo_items:
        raise RuntimeError(f"fold {f.fold_id} has items on BOTH train and OOF sides")
_val_item_set = set(val_df["item_key"].astype(str).tolist())
for f in folds:
    _oo_items = set(_train_keys[f.oof_row_idx].tolist())
    if _oo_items & _val_item_set:
        raise RuntimeError(f"fold {f.fold_id} OOF items overlap val")
print("[oof check] item-cold invariants OK.")

# %% [markdown]
# ## 6. Dense item-form / metadata block (matches production Member 8)
#
# Identical block to the loss-diversity probe: z-scored pool features +
# item-type one-hots + CoT interactions. Train stats only (no leakage).
# This also gives us the per-row item-type bucket we need for routing
# below -- the one-hots in the dense block come from the same
# `item_type_onehot` function we use to assign experts.

# %%
from src.item_features import (
    POOL_FEATURE_NAMES as _POOL_FEATURE_NAMES,
    ITEM_TYPE_NAMES,
    apply_zscore,
    build_cot_interactions as _build_cot_interactions,
    compute_features_for_items,
    cot_interaction_names as _cot_interaction_names,
    fit_zscore_stats,
    is_cot_from_condition as _is_cot_from_condition,
    item_type_onehot,
    load_pool_features,
    save_pool_features,
)

_ART_FEATURES = ROOT / "artifacts" / "item_features"
_POOL_PATH = _ART_FEATURES / "pool_features.parquet"
_pool_df = load_pool_features(_POOL_PATH)
if _pool_df is None:
    _pool_df = compute_features_for_items(item_df, progress=True)
    save_pool_features(_pool_df, _POOL_PATH)
    print(f"[form] computed pool features for {len(_pool_df):,} items")
else:
    print(f"[form] loaded pool features for {len(_pool_df):,} items")

_POOL_COLS = list(_POOL_FEATURE_NAMES)
_train_item_set = set(train_df["item_key"].astype(str).tolist())
_pool_train = _pool_df[_pool_df["item_key"].astype(str).isin(_train_item_set)]
_pool_stats = fit_zscore_stats(_pool_train, feature_cols=_POOL_COLS)
_pool_z = apply_zscore(_pool_df, _pool_stats)
_pool_z_by_item = _pool_z.set_index("item_key")
_pool_raw_by_item = _pool_df.set_index("item_key")
_item_type_by_item = {
    str(ik): item_type_onehot(row.to_dict())
    for ik, row in _pool_raw_by_item.iterrows()
}
_FORM_TYPE_NAMES = list(ITEM_TYPE_NAMES)
FORM_BLOCK_NAMES = (
    tuple(_POOL_COLS) + tuple(_FORM_TYPE_NAMES) + tuple(_cot_interaction_names())
)
print(f"[form] dense block width = {len(FORM_BLOCK_NAMES)}")


def _build_form_block(item_keys, conditions):
    ik = [str(k) for k in item_keys]
    n = len(ik)
    pz = _pool_z_by_item.reindex(ik)[_POOL_COLS].to_numpy(dtype=np.float32)
    pz = np.nan_to_num(pz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tmat = np.zeros((n, len(_FORM_TYPE_NAMES)), dtype=np.float32)
    for i, k in enumerate(ik):
        t = _item_type_by_item.get(k)
        if t is not None:
            for j, tn in enumerate(_FORM_TYPE_NAMES):
                tmat[i, j] = t[tn]
    base = np.concatenate([pz, tmat], axis=1).astype(np.float32)
    base_names = list(_POOL_COLS) + list(_FORM_TYPE_NAMES)
    is_cot = np.fromiter((_is_cot_from_condition(c) for c in conditions),
                         count=n, dtype=np.float32)
    inter, _ = _build_cot_interactions(base, base_names, is_cot)
    return np.concatenate([pz, tmat, inter], axis=1).astype(np.float32)


_dense_train_raw = _build_form_block(
    _train_keys, train_df["condition"].astype(str).to_numpy()
)
_dense_val_raw = _build_form_block(
    _val_keys, val_df["condition"].astype(str).to_numpy()
)
_dmean = _dense_train_raw.mean(axis=0).astype(np.float32)
_dstd = _dense_train_raw.std(axis=0).astype(np.float32)
_dstd = np.where(_dstd < 1e-6, 1.0, _dstd).astype(np.float32)
dense_train = ((_dense_train_raw - _dmean) / _dstd).astype(np.float32) if EXP["use_metadata"] else None
dense_val = ((_dense_val_raw - _dmean) / _dstd).astype(np.float32) if EXP["use_metadata"] else None
DENSE_DIM = 0 if dense_train is None else int(dense_train.shape[1])
print(f"[form] dense_train width = {DENSE_DIM}  (use_metadata={EXP['use_metadata']})")

# %% [markdown]
# ## 7. Per-row expert bucket (the partition axis + the router)
#
# This is the MoE-specific bit. We partition train + val rows into K
# buckets along a single axis, then train one expert per bucket
# (upweighted on its bucket, full data otherwise). The bucket id is BOTH
# the source of the sample-weight upweight per expert (training) AND the
# deterministic router at inference. No learned router; the partition is
# the entire moving part.
#
# Two partition axes are supported (see EXP['partition_axis']):
#
# * ``item_type``  -- K=4 form buckets from src.item_features.item_type_onehot
#   (priority code > mcq > math > prose). Mutually exclusive but heavily
#   imbalanced on this dataset (~75% prose, so expert_type_prose ends up
#   training on almost all rows with a small tilt -- the v1 POC showed
#   exactly this).
#
# * ``nn_support`` -- K=n_buckets octiles by item-embedding neighborhood
#   density. For each item we measure the mean cosine similarity to its
#   top-k nearest TRAIN items (leave-one-out for train items so an item
#   is not its own neighbor). Bucket boundaries are fit on the TRAIN item
#   distribution. Val items distribute naturally by how isolated they
#   are: bucket 0 = coldest support (smallest mean similarity), bucket
#   K-1 = warmest. Balanced by construction AND aligned to the cold-start
#   axis the test set is heavy on -- the right next-step axis after the
#   v1 (item_type) controls revealed that ~70% of the v1 win was actually
#   focal-style global lift, not routing-specific specialization.

# %%
_PARTITION_AXIS = str(EXP.get("partition_axis", "nn_support"))
print(f"[partition] axis = {_PARTITION_AXIS}")

if _PARTITION_AXIS == "item_type":
    _BUCKET_NAMES = list(EXP["type_buckets"])
    K_EXPERTS = len(_BUCKET_NAMES)
    _bucket_to_idx = {b: i for i, b in enumerate(_BUCKET_NAMES)}

    def _row_bucket_ids(item_keys) -> np.ndarray:
        out = np.zeros(int(len(item_keys)), dtype=np.int64)
        fallback = _bucket_to_idx.get("type_prose", 0)
        for i, k in enumerate(item_keys):
            t = _item_type_by_item.get(str(k))
            if t is None:
                out[i] = fallback
                continue
            for j, name in enumerate(_BUCKET_NAMES):
                if float(t.get(name, 0.0)) >= 0.5:
                    out[i] = j
                    break
        return out

elif _PARTITION_AXIS == "nn_support":
    import faiss

    K_EXPERTS = int(EXP["n_buckets"])
    K_NN = int(EXP["support_k_neighbors"])

    # Unique TRAIN item embeddings, L2-normalized so FAISS inner-product
    # == cosine similarity. ~50k items @ 4096 D -> ~800 MB resident; well
    # within budget. Keep a separate L2-normalized matrix because the
    # global ALL_UNIQ table is RAW (the GLU-MLP expects raw embeddings).
    _train_uniq_keys = sorted(
        k for k in set(_train_keys.tolist()) if k in item_emb_lookup
    )
    _train_emb_uniq = np.stack(
        [item_emb_lookup[k] for k in _train_uniq_keys]
    ).astype(np.float32)
    _train_norms = np.linalg.norm(_train_emb_uniq, axis=1, keepdims=True)
    _train_emb_uniq = (
        _train_emb_uniq / np.clip(_train_norms, 1e-12, None)
    ).astype(np.float32)

    print(
        f"[support] building FAISS IP index on {len(_train_uniq_keys):,} "
        f"unique train items (D={ITEM_EMB_DIM})..."
    )
    _faiss_idx = faiss.IndexFlatIP(ITEM_EMB_DIM)
    _faiss_idx.add(_train_emb_uniq)

    # K+1 neighbors so we can drop rank-0 (self) for train items. Mean
    # of the next K is the density score -- higher means denser local
    # neighborhood (warmer support).
    _sims_train, _ = _faiss_idx.search(_train_emb_uniq, K_NN + 1)
    _score_train_uniq = _sims_train[:, 1:].mean(axis=1).astype(np.float32)
    print(
        f"[support] train item density: min={_score_train_uniq.min():.4f} "
        f"mean={_score_train_uniq.mean():.4f} "
        f"max={_score_train_uniq.max():.4f}"
    )

    # Val items are NOT in the index (item-cold guarantee), so all K
    # neighbors are different items -- no self-exclusion needed.
    _val_uniq_keys = sorted(
        k for k in set(_val_keys.tolist()) if k in item_emb_lookup
    )
    _val_emb_uniq = np.stack(
        [item_emb_lookup[k] for k in _val_uniq_keys]
    ).astype(np.float32)
    _val_norms = np.linalg.norm(_val_emb_uniq, axis=1, keepdims=True)
    _val_emb_uniq = (
        _val_emb_uniq / np.clip(_val_norms, 1e-12, None)
    ).astype(np.float32)
    _sims_val, _ = _faiss_idx.search(_val_emb_uniq, K_NN)
    _score_val_uniq = _sims_val.mean(axis=1).astype(np.float32)
    print(
        f"[support] val item density:   min={_score_val_uniq.min():.4f} "
        f"mean={_score_val_uniq.mean():.4f} "
        f"max={_score_val_uniq.max():.4f}"
    )

    # Bucket boundaries from TRAIN distribution. K-1 cut points produce
    # K buckets (octiles for K=8). Items below the first cut go in
    # bucket 0; items above the last cut go in bucket K-1.
    _boundaries = np.quantile(
        _score_train_uniq, np.linspace(0.0, 1.0, K_EXPERTS + 1)
    )[1:-1].astype(np.float32)
    print(
        f"[support] {K_EXPERTS}-bucket cut points: "
        + ", ".join(f"{b:.4f}" for b in _boundaries)
    )

    _support_by_item: dict[str, float] = dict(
        zip(_train_uniq_keys, _score_train_uniq.tolist())
    )
    for k, s in zip(_val_uniq_keys, _score_val_uniq.tolist()):
        _support_by_item[k] = float(s)
    _median_score = float(np.median(_score_train_uniq))

    def _row_bucket_ids(item_keys) -> np.ndarray:
        scores = np.fromiter(
            (
                float(_support_by_item.get(str(k), _median_score))
                for k in item_keys
            ),
            count=int(len(item_keys)),
            dtype=np.float32,
        )
        return np.searchsorted(
            _boundaries, scores, side="right"
        ).astype(np.int64)

    # q1 = coldest (lowest mean-sim quantile); qK = warmest.
    _BUCKET_NAMES = [f"support_q{j + 1}" for j in range(K_EXPERTS)]

    # Free the L2-normalized matrices + the index now that we have the
    # bucket assignments. The GLU-MLP downstream uses ALL_UNIQ (raw
    # embeddings), not these.
    del _train_emb_uniq, _val_emb_uniq, _faiss_idx, _sims_train, _sims_val
else:
    raise ValueError(
        f"Unknown EXP['partition_axis']={_PARTITION_AXIS!r}; "
        "expected one of {'item_type', 'nn_support'}"
    )

bucket_train = _row_bucket_ids(_train_keys)
bucket_val = _row_bucket_ids(_val_keys)

print(f"\nK_EXPERTS = {K_EXPERTS}")
print("Bucket distribution (TRAIN rows):")
for j, name in enumerate(_BUCKET_NAMES):
    n = int((bucket_train == j).sum())
    print(f"  {name:14s} train_rows={n:>9,} ({n / N_TRAIN:>5.1%})")
print("Bucket distribution (VAL rows):")
for j, name in enumerate(_BUCKET_NAMES):
    n = int((bucket_val == j).sum())
    print(f"  {name:14s} val_rows={n:>9,} ({n / N_VAL:>5.1%})")

# %% [markdown]
# ## 8. Train baseline + K experts (item-cold OOF, shared trainer)
#
# **Baseline**: identical M8 trained on full data with uniform sample
# weights (= 1 everywhere). The fair control.
#
# **Expert k**: same M8, trained on full data with
# `sample_weight = expert_weight_multiplier` on rows whose bucket equals
# k, and `1.0` elsewhere (then mean-normalized to 1.0 so total gradient
# magnitude per batch matches the baseline). Every expert still sees the
# full data every epoch -- failure here cannot be blamed on
# undertraining.
#
# Same OOF folds, same epochs, same patience, same seed offsetting as
# the loss-diversity probe -- so we can read the comparison as a clean
# axis-swap (loss-variation -> region-variation).

# %%
import torch

from src.mlp_variant import predict_probs, train_m8_variant

_dev = EXP["emb_device"] or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_dev}")
EMB_T = torch.from_numpy(ALL_UNIQ).to(_dev)


def _make_weights(in_bucket_mask, mult):
    """Sample-weight vector that's `mult` on in-bucket rows, 1.0 elsewhere."""
    w = np.where(in_bucket_mask, float(mult), 1.0).astype(np.float32)
    return (w / w.mean()).astype(np.float32)


def _run_one_oof(name, sample_weights):
    """OOF-train one M8 variant; return (oof_train[N], val[N_val])."""
    print(f"\n=== {name} ===")
    acc = OofPredictionAccumulator(N_TRAIN, name=f"oof_{name}")
    val_stack = []
    for fold in folds:
        tr, oof = fold.train_row_idx, fold.oof_row_idx
        print(f"  [fold {fold.fold_id}] train={len(tr):,} oof={len(oof):,}")
        net = train_m8_variant(
            y=y_train[tr], subj_ids=subj_train[tr], r2u=r2u_train[tr],
            emb_tensor=EMB_T, n_subjects=int(indexer.n_subjects), device=_dev,
            dense_X=(dense_train[tr] if dense_train is not None else None),
            objective="bce",
            sample_weights=(sample_weights[tr] if sample_weights is not None else None),
            subj_emb_dim=int(EXP["subj_emb_dim"]), hid1=int(EXP["hid1"]),
            hid2=int(EXP["hid2"]), lr=float(EXP["learning_rate"]),
            wd=float(EXP["weight_decay"]), epochs=int(EXP["epochs"]),
            batch_size=int(EXP["batch_size"]), val_fraction=float(EXP["val_fraction"]),
            patience=int(EXP["patience"]), feat_dropout=float(EXP["feat_dropout"]),
            seed=SEED + fold.fold_id, show_progress=False,
        )
        acc.write_fold(oof, predict_probs(
            net, subj_train[oof], r2u_train[oof], EMB_T, _dev,
            int(EXP["predict_chunk"]),
            dense_X=(dense_train[oof] if dense_train is not None else None),
        ))
        val_stack.append(predict_probs(
            net, subj_val, r2u_val, EMB_T, _dev, int(EXP["predict_chunk"]),
            dense_X=dense_val,
        ))
        del net
        if _dev == "cuda":
            torch.cuda.empty_cache()
    return (
        acc.finalize().astype(np.float32),
        np.mean(val_stack, axis=0).astype(np.float32),
    )


oof_preds = {}
val_preds = {}

# Baseline first -- uniform weights, no upweighting. The fair control.
oof_preds["baseline"], val_preds["baseline"] = _run_one_oof(
    "baseline (uniform weights)", sample_weights=None,
)

# K experts, one per bucket.
for j, name in enumerate(_BUCKET_NAMES):
    in_bucket = (bucket_train == j)
    w = _make_weights(in_bucket, EXP["expert_weight_multiplier"])
    n_in = int(in_bucket.sum())
    print(
        f"\n[expert {name}] upweight={EXP['expert_weight_multiplier']}x on "
        f"{n_in:,} rows ({n_in / N_TRAIN:.1%} of train); "
        f"mean weight after normalize={w.mean():.4f}"
    )
    oof_preds[f"expert_{name}"], val_preds[f"expert_{name}"] = _run_one_oof(
        f"expert {name}", sample_weights=w,
    )

print("\nAll experts trained.")

# %% [markdown]
# ## 9. Per-model Platt calibration (apples-to-apples NLL)
#
# Each variant gets a 1-D Platt fit on its own OOF preds before solo
# scoring -- same recipe as the loss-diversity probe so the numbers are
# directly comparable.

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    build_stacker_features,
    fit_stacker,
    stacker_feature_names,
)

_zeros_tr = np.zeros(N_TRAIN, dtype=np.float32)
_zeros_va = np.zeros(N_VAL, dtype=np.float32)
_half_tr = np.full(N_TRAIN, 0.5, dtype=np.float32)
_half_va = np.full(N_VAL, 0.5, dtype=np.float32)


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _platt(p_oof, p_val):
    Xo = build_stacker_features(
        member_probs=p_oof[:, None], bench_present=_zeros_tr,
        nn_neighbor_support=_zeros_tr, nn_mean_similarity=_zeros_tr,
        centroid_distance=_half_tr,
    )
    st = fit_stacker(X=Xo, y=y_train, feature_names=stacker_feature_names(1),
                     seed=SEED, n_iters=1500)
    Xv = build_stacker_features(
        member_probs=p_val[:, None], bench_present=_zeros_va,
        nn_neighbor_support=_zeros_va, nn_mean_similarity=_zeros_va,
        centroid_distance=_half_va,
    )
    return stacker_apply_batch(st, Xv).astype(np.float32)


_names = ["baseline"] + [f"expert_{b}" for b in _BUCKET_NAMES]
cal_val = {n: _platt(oof_preds[n], val_preds[n]) for n in _names}
print("\nCalibrated solo val log-loss (lower is better):")
_solo = {n: _nll(cal_val[n], y_val) for n in _names}
_base_nll = _solo["baseline"]
for n in _names:
    d = _solo[n] - _base_nll
    print(f"  {n:24s}: {_solo[n]:.6f}   (vs baseline: {d:+.6f})")

# %% [markdown]
# ## 10. Per-bucket specialization diagnostic
#
# **The single most important table in this notebook.** For each
# (expert k, bucket j) we report calibrated val NLL on the SLICE of val
# rows in bucket j. The diagonal entries (expert k on its own bucket)
# are where routing-specific MoE has to pay off. Compare to the baseline
# row to see whether the upweighting bought anything that scrambled
# routing wouldn't also capture.

# %%
print("\n" + "=" * 96)
print(
    f"PER-BUCKET SPECIALIZATION DIAGNOSTIC "
    f"(calibrated val NLL on each bucket slice; axis={_PARTITION_AXIS})"
)
print("=" * 96)
header = (
    f"{'model':<28s}"
    + "".join(f"{b:>14s}" for b in _BUCKET_NAMES)
    + f"{'overall':>10s}"
)
print(header)
print("-" * len(header))


def _slice_nll(p, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return _nll(p[mask], y_val[mask])


_slice_table = {}
for n in _names:
    row = [f"{n:<28s}"]
    slice_vals = []
    for j, _ in enumerate(_BUCKET_NAMES):
        mask = (bucket_val == j)
        v = _slice_nll(cal_val[n], mask)
        slice_vals.append(v)
        row.append(f"{v:>14.6f}")
    row.append(f"{_solo[n]:>10.6f}")
    _slice_table[n] = slice_vals
    print("".join(row))

print("\nDIAGONAL CHECK -- each expert vs the baseline ON ITS OWN BUCKET:")
print(f"  {'bucket':<14s}{'baseline':>12s}{'expert':>12s}{'delta':>12s}")
for j, b in enumerate(_BUCKET_NAMES):
    base_v = _slice_table["baseline"][j]
    exp_v = _slice_table[f"expert_{b}"][j]
    d = exp_v - base_v
    flag = "  <-- specialization works" if d < -0.0005 else ""
    print(f"  {b:<14s}{base_v:>12.6f}{exp_v:>12.6f}{d:>+12.6f}{flag}")

# %% [markdown]
# ## 11. Hard-routed ensemble + decomposition controls
#
# Three reported numbers:
#
# * **routed** -- hard-route each val row to the expert for its bucket.
#   The actual MoE prediction.
# * **scrambled** -- route each val row to a RANDOM expert. Isolates the
#   "experts are systematically better than baseline" lift (focal-style
#   sample-weight effect) from the routing-specific lift.
# * **uniform avg** -- mean of all expert predictions per row. Tells us
#   whether the experts are diverse enough that naive averaging beats
#   the best single one.
# * **best single expert** -- the expert with lowest overall NLL on its
#   own. Floor for any routing/merging scheme; if "routed" doesn't beat
#   this, routing is adding no value beyond pick-the-best-expert.
#
# The decomposition is:
#     routed_lift  =  (scrambled - baseline)  +  (routed - scrambled)
#                  =       global lift         +     routing lift
# If most of `routed_lift` is global, the win is a free training trick
# you can apply to vanilla M8 without any MoE architecture.

# %%
def _route_hard(per_expert_val, bucket_per_row):
    """Hard-route: row i takes prediction from expert_<bucket_per_row[i]>."""
    out = np.empty(int(len(bucket_per_row)), dtype=np.float32)
    for j, b in enumerate(_BUCKET_NAMES):
        mask = (bucket_per_row == j)
        out[mask] = per_expert_val[f"expert_{b}"][mask]
    return out


# Calibrated per-expert val preds so every comparison is apples-to-apples.
p_routed = _route_hard(cal_val, bucket_val)
nll_routed = _nll(p_routed, y_val)

# Scrambled routing: route to a random expert (not the row's bucket).
_rng_route = np.random.default_rng(SEED + 12345)
_scrambled_bucket = _rng_route.integers(0, K_EXPERTS, size=N_VAL).astype(np.int64)
p_scrambled = _route_hard(cal_val, _scrambled_bucket)
nll_scrambled = _nll(p_scrambled, y_val)

# Uniform average of all expert predictions.
p_uniform = np.mean(
    [cal_val[f"expert_{b}"] for b in _BUCKET_NAMES], axis=0
).astype(np.float32)
nll_uniform = _nll(p_uniform, y_val)

# Best single expert (smallest overall calibrated NLL among experts).
_expert_solo = {
    n: _solo[n] for n in _names if n != "baseline"
}
_best_single_name = min(_expert_solo, key=_expert_solo.get)
nll_best_single = _expert_solo[_best_single_name]

# Decomposition: split routed_lift into global vs routing components.
_global_lift = nll_scrambled - _base_nll
_routing_lift = nll_routed - nll_scrambled
_total_lift = nll_routed - _base_nll

print("\n" + "=" * 72)
print("ROUTED ENSEMBLE vs BASELINE (val log-loss; lower is better)")
print("=" * 72)
print(f"  Baseline (uniform M8, calibrated)        : {_base_nll:.6f}")
print(f"  Scrambled routing (random expert per row): {nll_scrambled:.6f}  "
      f"({nll_scrambled - _base_nll:+.6f})")
print(f"  Uniform avg of {K_EXPERTS} experts                : "
      f"{nll_uniform:.6f}  ({nll_uniform - _base_nll:+.6f})")
print(f"  Best single expert ({_best_single_name:<22s}): "
      f"{nll_best_single:.6f}  ({nll_best_single - _base_nll:+.6f})")
print(f"  Hard-routed MoE (K={K_EXPERTS}, upweight"
      f"={EXP['expert_weight_multiplier']}x)     : "
      f"{nll_routed:.6f}  ({_total_lift:+.6f})")

print("\nLIFT DECOMPOSITION:")
print(f"  Global (scrambled - baseline)  : {_global_lift:+.6f}  "
      f"(any-expert vs baseline; pure focal effect, no routing needed)")
print(f"  Routing (routed   - scrambled) : {_routing_lift:+.6f}  "
      f"(routed - scrambled; pure routing-specific gain)")
print(f"  Total  (routed    - baseline)  : {_total_lift:+.6f}")
if abs(_total_lift) > 1e-9:
    _routing_share = max(0.0, -_routing_lift) / max(1e-9, -_total_lift) * 100.0
    print(f"  Routing share of total lift    : {_routing_share:.1f}%")

# Routed-ensemble per-bucket slice.
print("\nPer-bucket slice for the routed ensemble:")
print(f"  {'bucket':<14s}{'baseline':>12s}{'routed':>12s}{'delta':>12s}")
for j, b in enumerate(_BUCKET_NAMES):
    mask = (bucket_val == j)
    base_v = _slice_nll(cal_val["baseline"], mask)
    routed_v = _slice_nll(p_routed, mask)
    d = routed_v - base_v
    print(f"  {b:<14s}{base_v:>12.6f}{routed_v:>12.6f}{d:>+12.6f}")

# %% [markdown]
# ## 12. Diversity heatmaps
#
# Two complementary correlation matrices, both on val:
#
# * **Signed-error correlation** `corr(p_i - y, p_j - y)`. The classic
#   ensemble-diversity number. Misleading when models are all reasonably
#   strong: most of the variance in `p_i - y` is explained by the binary
#   label, so any two competent models score ~0.95+ even when they
#   disagree meaningfully on row-level probabilities.
# * **Residual-vs-baseline correlation** `corr(p_i - p_base, p_j - p_base)`.
#   The right diversity metric for ensembling around an existing strong
#   model: subtract the baseline level, then ask whether the experts
#   move in the same direction or different directions. This is what
#   stacker weights actually see and what governs whether a per-region
#   FWLS merge can recover signal.

# %%
_corr_names = _names + ["routed"]
_p_stack = {n: cal_val[n] for n in _names}
_p_stack["routed"] = p_routed

# Signed-error correlation (classic, often misleading).
_err_stack = np.stack([_p_stack[n] - y_val for n in _corr_names], axis=1)
_corr_err = np.corrcoef(_err_stack.T)

# Residual-vs-baseline (drop baseline column since it would be 0).
_resid_names = [n for n in _corr_names if n != "baseline"]
_resid_stack = np.stack(
    [_p_stack[n] - cal_val["baseline"] for n in _resid_names], axis=1
)
_corr_resid = np.corrcoef(_resid_stack.T)

_K = _corr_err.shape[0]
_mask = ~np.eye(_K, dtype=bool)
print(
    f"\nMean off-diagonal |error correlation|       (all)     = "
    f"{np.abs(_corr_err[_mask]).mean():.4f}"
)

_exp_idx = [i for i, n in enumerate(_corr_names) if n.startswith("expert_")]
_exp_err = _corr_err[np.ix_(_exp_idx, _exp_idx)]
_exp_mask = ~np.eye(len(_exp_idx), dtype=bool)
print(
    f"Mean off-diagonal |error correlation|       (experts) = "
    f"{np.abs(_exp_err[_exp_mask]).mean():.4f}"
)

_Kr = _corr_resid.shape[0]
_mask_r = ~np.eye(_Kr, dtype=bool)
print(
    f"Mean off-diagonal |residual-vs-base corr|   (experts) = "
    f"{np.abs(_corr_resid[_mask_r]).mean():.4f}  "
    f"<-- DIVERSITY signal (lower = more orthogonal)"
)


def _render(corr, names, title, fname, cmap_high=0.7, fig_w=9.0, fig_h=7.5):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > cmap_high else "black",
                    fontsize=7,
                )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _outdir = ROOT / "artifacts" / "diagnostics"
        _outdir.mkdir(parents=True, exist_ok=True)
        _outpath = _outdir / fname
        fig.savefig(_outpath, dpi=120, bbox_inches="tight")
        print(f"[heatmap] saved -> {_outpath}")
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"[heatmap {fname}] skipped ({exc!r})")
        print(np.round(corr, 3))


_render(
    _corr_err,
    _corr_names,
    title=(
        f"MoE signed-error correlation (val)\n"
        f"axis={_PARTITION_AXIS}, K={K_EXPERTS}, "
        f"upweight={EXP['expert_weight_multiplier']}x, hard routing"
    ),
    fname=f"moe_poc_{_PARTITION_AXIS}_K{K_EXPERTS}_err_corr.png",
)
_render(
    _corr_resid,
    _resid_names,
    title=(
        f"MoE residual-vs-baseline correlation (val)\n"
        f"axis={_PARTITION_AXIS}, K={K_EXPERTS}, "
        f"upweight={EXP['expert_weight_multiplier']}x  "
        f"(lower off-diag = more useful to a stacker)"
    ),
    fname=f"moe_poc_{_PARTITION_AXIS}_K{K_EXPERTS}_resid_corr.png",
)

# %% [markdown]
# ## 13. Verdict
#
# We score the partition on **three** signals, not just `routed - baseline`:
#
# 1. **routing-specific lift**  = `routed - scrambled` -- the only
#    component that's actually attributable to MoE-style specialization.
#    If this is ~0, the win (if any) is just the focal-style sample-weight
#    trick and you should pursue that directly on vanilla M8, not MoE.
# 2. **per-bucket diagonal wins** = number of buckets where the matching
#    expert beats baseline on its own slice. Tells us whether each expert
#    is actually specializing.
# 3. **residual-vs-baseline orthogonality** = mean off-diagonal of
#    `corr(p_expert - p_base, p_other_expert - p_base)`. The number a
#    stacker (or FWLS merge) actually sees. Lower = more useful.
#
# Decision tree:
# * routing_lift <= -0.002 AND >= K/2 diagonal wins
#     -> escalate this partition (soft routing, weight sweep, more buckets).
# * routing_lift >= -0.001 BUT global_lift <= -0.003
#     -> the upweighting is a free training trick. Apply it to vanilla M8
#        (no architecture change) and move on.
# * neither holds
#     -> partition does nothing. Try a different axis.

# %%
_diag_wins = 0
_diag_total = 0
for j, b in enumerate(_BUCKET_NAMES):
    base_v = _slice_table["baseline"][j]
    exp_v = _slice_table[f"expert_{b}"][j]
    _diag_total += 1
    if exp_v < base_v - 0.0005:
        _diag_wins += 1
_mean_resid_corr = float(np.abs(_corr_resid[_mask_r]).mean())

print("=" * 72)
print(f"VERDICT (partition_axis={_PARTITION_AXIS}, K={K_EXPERTS}, "
      f"upweight={EXP['expert_weight_multiplier']}x)")
print("=" * 72)
print(f"  Baseline (uniform M8)                : {_base_nll:.6f}")
print(f"  Scrambled routing                    : {nll_scrambled:.6f}  "
      f"({_global_lift:+.6f}  global lift)")
print(f"  Hard-routed MoE                      : {nll_routed:.6f}  "
      f"({_total_lift:+.6f}  total)")
print(f"  Routing-specific lift (routed-scram) : {_routing_lift:+.6f}")
print(f"  Best single expert ({_best_single_name:<22s}) : {nll_best_single:.6f}")
print(f"  Uniform avg of {K_EXPERTS} experts            : {nll_uniform:.6f}")
print(f"  Experts winning on own bucket        : {_diag_wins}/{_diag_total}")
print(f"  Mean residual-vs-base |corr| (exp)   : {_mean_resid_corr:.4f}  "
      f"(lower = more useful to stacker)")

if _routing_lift <= -0.002 and _diag_wins >= max(1, _diag_total // 2):
    print(
        f"  -> ROUTING-SPECIFIC WIN on axis={_PARTITION_AXIS}. Escalate: "
        "sweep weight multiplier, try soft routing (item-conditional "
        "blend), add this as a 5th member to the ensemble."
    )
elif _routing_lift >= -0.0010 and _global_lift <= -0.003:
    print(
        "  -> NO routing-specific lift, but the upweighting is a "
        "free training trick. Apply non-uniform sample weights to "
        "vanilla M8 (no MoE architecture) and move on."
    )
elif _mean_resid_corr < 0.5 and _total_lift <= -0.001:
    print(
        "  -> Mixed: routing doesn't beat scrambled by much, BUT the "
        "experts are diverse enough that a stacker / FWLS merge "
        "might recover the lift. Worth feeding into the L1 stacker "
        "as separate members."
    )
else:
    print(
        f"  -> Axis '{_PARTITION_AXIS}' does not pay here. Try a "
        "different partition (NN support if you ran item_type, or "
        "vice versa; subject cluster; embedding k-means)."
    )
