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
# **Question.** Does training one Member-8 instance per item-type bucket
# (upweighted on its own type, full data otherwise) and hard-routing val
# rows to the matching expert beat a single M8 trained on all data?
#
# **Hypothesis.** Different item types (math / code / MCQ / prose) live
# on qualitatively different feature manifolds, so a single global MLP is
# forced to compromise across them. Per-type experts free up capacity for
# within-type structure and should produce errors that disagree on a
# different axis than a fixed-architecture global model.
#
# **Why this is a CLEAN test of the MoE hypothesis** (vs. loss-diversity):
# 1. The training-signal change is along a different axis -- **input-space
#    specialization**, not loss-shape variation. The loss-diversity probe
#    falsified loss-shape diversity; we are testing whether the OTHER
#    axis (input partitioning) carries diversity the global model misses.
# 2. **Soft training (upweighting) instead of hard partitioning.** Every
#    expert sees the full data; it just gets a 5x gradient bump on its
#    own type's rows. Failure here means "MoE doesn't help on this axis"
#    rather than "the expert undertrained on 1/K of the data" -- removes
#    the most common false-negative for stratified ensembles.
# 3. **Hard routing at inference.** Routing is the deterministic item-type
#    indicator from `src/item_features.py` -- no learned router to debug.
#    Decouples "do experts specialize?" from "is the router any good?".
#
# **Decision rule from this run:**
# * Routed ensemble beats baseline by >= 0.003 nats AND the per-type
#   diagnostic shows each expert beats baseline on its own region:
#   -> MoE direction is real, escalate to (a) tune weight multiplier,
#      (b) try soft routing, (c) try joint type x cold-warm partition,
#      (d) eventually proper end-to-end MoE with a learned router.
# * Routed beats by < 0.003 but per-type diagnostic shows specialization:
#   -> routing is wrong; try soft routing or per-row blending.
# * Per-type diagnostic shows experts barely beating baseline in their
#   region: -> MoE doesn't work on this axis at this scale. Don't escalate.
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
# apples-to-apples. K=4 experts (item type) + 1 baseline = 5 OOF passes
# over the M8 architecture, identical to the 5 quickest objectives there.
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
    # Sample-weight multiplier on rows whose item type matches the expert.
    # 5x is the recommended starting point: enough to bias the gradient
    # toward the region without near-zero gradient on out-of-region rows
    # (the failure mode of hard partitioning). Sweep this if the POC
    # passes -- (3x, 5x, 10x, inf [= hard partition]) is the natural arc.
    "expert_weight_multiplier": 5.0,
    # Item-type buckets (mutually exclusive, defined in src/item_features.py
    # via item_type_onehot: priority code > mcq > math > prose, so every
    # item lands in exactly one bucket).
    "type_buckets": ("type_code", "type_mcq", "type_math", "type_prose"),
    # Hard-route val rows to the expert matching the item's type. Soft
    # routing (blend by type one-hot probabilities) is the obvious next
    # extension if hard routing under-performs; left as a follow-up so
    # this POC has exactly one moving part vs. the baseline.
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
# ## 7. Per-row item type (the partition axis + the router)
#
# This is the MoE-specific bit. For every train/val row we look up the
# unique item's coarse type via `item_type_onehot` (priority code > mcq >
# math > prose, so every item is in exactly one bucket). The resulting
# per-row int vector is BOTH:
#   * the source of the sample-weight upweight per expert (training), and
#   * the deterministic router at inference (hard-route to expert k).
# No learned router; no extra moving parts beyond the experts themselves.

# %%
_BUCKETS = list(EXP["type_buckets"])
_BUCKET_TO_IDX = {b: i for i, b in enumerate(_BUCKETS)}
K_EXPERTS = len(_BUCKETS)


def _row_type_ids(item_keys) -> np.ndarray:
    """Per-row int in [0, K-1] giving the item-type bucket index.

    Items not present in `_item_type_by_item` (shouldn't happen for any
    train/val row) fall back to `type_prose` -- the largest residual
    bucket -- so the router never produces an out-of-range index.
    """
    out = np.zeros(int(len(item_keys)), dtype=np.int64)
    fallback = _BUCKET_TO_IDX.get("type_prose", 0)
    for i, k in enumerate(item_keys):
        t = _item_type_by_item.get(str(k))
        if t is None:
            out[i] = fallback
            continue
        # Each item is in exactly one bucket by construction; argmax over
        # the one-hot gives the bucket index.
        for j, name in enumerate(_BUCKETS):
            if float(t.get(name, 0.0)) >= 0.5:
                out[i] = j
                break
    return out


type_train = _row_type_ids(_train_keys)
type_val = _row_type_ids(_val_keys)

print("Type-bucket distribution (TRAIN rows):")
for j, name in enumerate(_BUCKETS):
    n = int((type_train == j).sum())
    print(f"  {name:12s} train_rows={n:>9,} ({n / N_TRAIN:>5.1%})")
print("Type-bucket distribution (VAL rows):")
for j, name in enumerate(_BUCKETS):
    n = int((type_val == j).sum())
    print(f"  {name:12s} val_rows={n:>9,} ({n / N_VAL:>5.1%})")

# %% [markdown]
# ## 8. Train baseline + K experts (item-cold OOF, shared trainer)
#
# **Baseline**: identical M8 trained on full data with uniform sample
# weights (= 1 everywhere). The fair control.
#
# **Expert k**: same M8, trained on full data with
# `sample_weight = expert_weight_multiplier` on rows whose item type is
# bucket k, and `1.0` elsewhere. Every expert still sees the full data
# every epoch -- failure here cannot be blamed on undertraining.
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

# K experts, one per item-type bucket.
for j, name in enumerate(_BUCKETS):
    in_bucket = (type_train == j)
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


_names = ["baseline"] + [f"expert_{b}" for b in _BUCKETS]
cal_val = {n: _platt(oof_preds[n], val_preds[n]) for n in _names}
print("\nCalibrated solo val log-loss (lower is better):")
_solo = {n: _nll(cal_val[n], y_val) for n in _names}
_base_nll = _solo["baseline"]
for n in _names:
    d = _solo[n] - _base_nll
    print(f"  {n:24s}: {_solo[n]:.6f}   (vs baseline: {d:+.6f})")

# %% [markdown]
# ## 10. Per-type specialization diagnostic
#
# **The single most important table in this notebook.** For each
# (expert k, type bucket j) we report calibrated val NLL on the SLICE of
# val rows whose item type is j. The diagonal entries (expert k on its
# own type) are where MoE has to pay off. Compare to the baseline row.

# %%
print("\n" + "=" * 88)
print("PER-TYPE SPECIALIZATION DIAGNOSTIC (calibrated val NLL on each type slice)")
print("=" * 88)
header = f"{'model':<24s}" + "".join(f"{b:>16s}" for b in _BUCKETS) + f"{'overall':>10s}"
print(header)
print("-" * len(header))


def _slice_nll(p, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return _nll(p[mask], y_val[mask])


_slice_table = {}
for n in _names:
    row = [f"{n:<24s}"]
    slice_vals = []
    for j, b in enumerate(_BUCKETS):
        mask = (type_val == j)
        v = _slice_nll(cal_val[n], mask)
        slice_vals.append(v)
        row.append(f"{v:>16.6f}")
    row.append(f"{_solo[n]:>10.6f}")
    _slice_table[n] = slice_vals
    print("".join(row))

print("\nDIAGONAL CHECK -- each expert vs the baseline ON ITS OWN TYPE:")
print(f"  {'type':<14s}{'baseline':>12s}{'expert':>12s}{'delta':>12s}")
for j, b in enumerate(_BUCKETS):
    base_v = _slice_table["baseline"][j]
    exp_v = _slice_table[f"expert_{b}"][j]
    d = exp_v - base_v
    flag = "  <-- specialization works" if d < -0.0005 else ""
    print(f"  {b:<14s}{base_v:>12.6f}{exp_v:>12.6f}{d:>+12.6f}{flag}")

# %% [markdown]
# ## 11. Hard-routed ensemble: route val row to expert matching its type
#
# This is the actual MoE prediction. No learned router -- each row goes
# to the expert assigned to its item-type bucket.

# %%
def _route_hard(per_expert_val, type_per_row):
    """Hard-route: per_expert_val is dict[expert_name] -> [N_val] probs.

    Returns [N_val] where row i takes the prediction from
    `expert_<bucket_for_row_i>`.
    """
    out = np.empty(int(len(type_per_row)), dtype=np.float32)
    for j, b in enumerate(_BUCKETS):
        mask = (type_per_row == j)
        out[mask] = per_expert_val[f"expert_{b}"][mask]
    return out


# Use the calibrated per-expert val preds for the routed ensemble, so
# the comparison to the calibrated baseline is apples-to-apples (each
# expert contributes a Platt-calibrated probability to its routed slice).
p_routed = _route_hard(cal_val, type_val)
nll_routed = _nll(p_routed, y_val)

print("\n" + "=" * 64)
print("ROUTED ENSEMBLE vs BASELINE (val log-loss; lower is better)")
print("=" * 64)
print(f"  Baseline (uniform M8, calibrated)   : {_base_nll:.6f}")
print(f"  Hard-routed MoE (K={K_EXPERTS} experts, "
      f"upweight={EXP['expert_weight_multiplier']}x)  : "
      f"{nll_routed:.6f}  ({nll_routed - _base_nll:+.6f})")

# Routed-ensemble per-type slice diagnostic.
print("\nPer-type slice for the routed ensemble:")
print(f"  {'type':<14s}{'baseline':>12s}{'routed':>12s}{'delta':>12s}")
for j, b in enumerate(_BUCKETS):
    mask = (type_val == j)
    base_v = _slice_nll(cal_val["baseline"], mask)
    routed_v = _slice_nll(p_routed, mask)
    d = routed_v - base_v
    print(f"  {b:<14s}{base_v:>12.6f}{routed_v:>12.6f}{d:>+12.6f}")

# %% [markdown]
# ## 12. Error-correlation heatmap
#
# Signed-error correlation `(p_cal - y)` on val across:
# baseline + K experts + the hard-routed ensemble. We want EXPERTS to
# de-correlate FROM EACH OTHER (off-diagonal much below 1.0); routed
# vs baseline tells us how much the routing axis adds beyond a single
# global model.

# %%
_corr_names = _names + ["routed"]
_err_stack = np.stack(
    [cal_val[n] - y_val for n in _names] + [p_routed - y_val], axis=1
)
_corr = np.corrcoef(_err_stack.T)

# Per-pair max-off-diagonal summary (the headline diversity number).
_K = _corr.shape[0]
_mask = ~np.eye(_K, dtype=bool)
print(f"\nMean off-diagonal |error correlation| (all-pairs) = "
      f"{np.abs(_corr[_mask]).mean():.4f}")

# Just the experts vs experts (drop baseline/routed) -- this is the
# "do the experts disagree with each other?" diagnostic.
_exp_idx = [i for i, n in enumerate(_corr_names) if n.startswith("expert_")]
_exp_corr = _corr[np.ix_(_exp_idx, _exp_idx)]
_exp_mask = ~np.eye(len(_exp_idx), dtype=bool)
print(f"Mean off-diagonal |error correlation| (experts only) = "
      f"{np.abs(_exp_corr[_exp_mask]).mean():.4f}")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(_corr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(_K))
    ax.set_yticks(range(_K))
    ax.set_xticklabels(_corr_names, rotation=45, ha="right")
    ax.set_yticklabels(_corr_names)
    for i in range(_K):
        for j in range(_K):
            ax.text(j, i, f"{_corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if _corr[i, j] < 0.7 else "black", fontsize=8)
    ax.set_title(
        f"MoE error correlation (val)\n"
        f"K={K_EXPERTS} item-type experts, upweight"
        f"={EXP['expert_weight_multiplier']}x, hard routing"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _outdir = ROOT / "artifacts" / "diagnostics"
    _outdir.mkdir(parents=True, exist_ok=True)
    _outpath = _outdir / "moe_poc_corr.png"
    fig.savefig(_outpath, dpi=120, bbox_inches="tight")
    print(f"[heatmap] saved -> {_outpath}")
    plt.show()
except Exception as exc:  # pragma: no cover - plotting optional
    print(f"[heatmap] skipped ({exc!r})")
    print(np.round(_corr, 3))

# %% [markdown]
# ## 13. Verdict
#
# Apply the decision rule from the header:
# * Routed beats baseline by >= 0.003 nats AND per-type diagonal shows
#   each expert beating baseline on its own type
#   -> MoE direction is real; escalate (weight sweep, soft routing, MoE
#      proper with learned router).
# * Routed beats by < 0.003 but per-type shows specialization
#   -> routing is wrong; try soft routing or per-row blending.
# * Per-type diagonal shows experts barely beating baseline in own
#   region -> MoE doesn't work on this axis at this scale; don't escalate.

# %%
_diag_wins = 0
_diag_total = 0
for j, b in enumerate(_BUCKETS):
    base_v = _slice_table["baseline"][j]
    exp_v = _slice_table[f"expert_{b}"][j]
    _diag_total += 1
    if exp_v < base_v - 0.0005:
        _diag_wins += 1
_routed_delta = nll_routed - _base_nll
print("=" * 64)
print("VERDICT")
print("=" * 64)
print(f"  Baseline (uniform M8)        : {_base_nll:.6f}")
print(f"  Hard-routed MoE              : {nll_routed:.6f}  ({_routed_delta:+.6f})")
print(f"  Experts winning on own type  : {_diag_wins}/{_diag_total}")
print(f"  Mean experts-only |err corr| : "
      f"{np.abs(_exp_corr[_exp_mask]).mean():.4f}")
if _routed_delta < -0.003 and _diag_wins == _diag_total:
    print("  -> MoE WORKS on item-type axis. Escalate: sweep weight "
          "multiplier, try soft routing, try joint partition.")
elif _routed_delta < -0.0005 and _diag_wins >= max(1, _diag_total // 2):
    print("  -> Marginal MoE win. Likely the routing is leaving lift on "
          "the table; try soft routing next.")
elif _diag_wins == 0:
    print("  -> No specialization on the per-type diagonal. MoE on "
          "item-type does not pay here. Try a different partition axis "
          "(subject family, embedding cluster) or accept the conclusion.")
else:
    print("  -> Mixed: some experts specialize but routing doesn't recover "
          "it. Soft routing is the next experiment.")
