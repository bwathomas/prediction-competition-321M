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
# # Feature-dropout × NN-support MoE × per-row gated combiner
#
# **The question this notebook answers.** The
# `feature_dropout_ensemble.py` probe showed something we hadn't seen
# before: **truly orthogonal / anti-correlated residuals** between
# rich-M8 variants where one feature class is masked at train AND val.
# The logit-linear Platt stacker only extracted -0.0010 nat from those
# residuals because two of the most orthogonal variants
# (`no_subj_channel`, `no_nn_block`) are also the weakest in absolute
# terms, and a single-weight-per-member linear combiner has no way to
# leverage their orthogonality without importing their absolute error.
#
# This notebook tests **whether a per-row gated combiner can extract
# the rest** by:
#
# 1. Training the NN-support MoE (8 buckets, soft-routed at
#    `tau = 1.0 * std`) **for each of the K+1 = 6 feature-dropout
#    variants**. Each variant gets its own routed prediction, so we
#    have 6 routed-MoE outputs instead of 1.
# 2. Combining those 6 routed predictions with three different
#    blenders, sharing the same OOF features and val targets so the
#    comparison is paired:
#    * **Uniform avg** -- naive baseline.
#    * **Logit-linear stacker** (current production style) -- one
#      global weight per member; auxiliary features enter as bias.
#    * **FWLS-style stacker** -- explicit (member x aux) cross
#      terms so per-row gating is possible. Built by hand because
#      `fit_stacker` only adds aux features as standalone columns.
#    * **Small MLP combiner** -- a tiny 2-layer net on the 6 logits +
#      4 aux features. Non-linear, so it can express interactions the
#      stacker can't.
# 3. Reporting the per-combiner val NLL versus the relevant
#    references (rich_baseline alone, rich_baseline + MoE, feature-
#    dropout uniform_avg, feature-dropout Platt stacker) so we know
#    whether the per-row gating actually pays.
#
# **Order-of-operations note.** This is the
# *feature-dropout-outer / MoE-inner* arrangement that the prior
# notebook's header argued for: each feature-dropout variant gets its
# own complete soft-routed MoE on top, then the combiner does a per-
# row weighted blend across the K+1 routed predictions. The (variant,
# bucket) grid has `(K + 1) * M = 6 * 8 = 48` trained nets per fold;
# at `n_folds = 3` that's 144 trainings before the combiners run.
# Knob it down with `EXP["variants"]` and `EXP["support_k_buckets"]`
# if compute is tight.
#
# **What this notebook shares with the prior two notebooks.** Same
# data, same item-cold OOF folds, same per-unit UNK redaction at
# train AND val time, same Qwen3 item embedding cache, same metadata
# preprocessor, same rich net architecture + per-variant feature
# masking, same NN-support partition geometry. Only the per-variant
# MoE + the FWLS / MLP combiners are new.

# %% [markdown]
# ## 0. Colab bootstrap

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

EXP = {
    # ---- Shared budget --------------------------------------------------
    "max_train_rows": 600_000,
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,
    "emb_device": None,
    "predict_chunk": 131_072,
    "use_metadata": True,

    # ---- Rich-M8 arch knobs (matches the prior two probes) --------------
    "rich_subj_emb_dim": 32,
    "rich_bc_emb_dim": 16,
    "rich_cluster_emb_dim": 16,
    "rich_family_emb_dim": 16,
    "rich_macro_emb_dim": 8,
    "rich_org_emb_dim": 16,
    "rich_topic_emb_dim": 16,
    "rich_hid1": 256,
    "rich_hid2": 128,
    "rich_lr": 1.0e-3,
    "rich_wd": 1.0e-5,
    "rich_feat_dropout": 0.10,
    "rich_n_cross_layers": 2,
    "rich_cross_rank": 64,
    "rich_cat_dropout_subject": 0.05,
    "rich_cat_dropout_bc": 0.20,
    "rich_cat_dropout_cluster": 0.10,
    "rich_cat_dropout_family": 0.05,
    "rich_cat_dropout_macro": 0.05,
    "rich_cat_dropout_org": 0.05,
    "rich_cat_dropout_topic": 0.10,

    # ---- Feature blocks ------------------------------------------------
    "use_nn_block": True,
    "use_subject_numeric": True,
    "use_bench_numeric": True,
    "use_cluster_block": True,
    "n_clusters": 8,
    "kmeans_iters": 50,
    "kmeans_seed": 23,

    # ---- Feature-dropout variants -- which to run ----------------------
    # The full (K + 1) = 6 grid. Shrink this list (e.g. drop the
    # catastrophic 'no_nn_block') to save compute; the combiner cells
    # below auto-adapt to whatever variants are populated.
    "variants": (
        "full",
        "no_subj_channel",
        "no_bench_channel",
        "no_form_block",
        "no_nn_block",
        "no_cluster_channel",
    ),

    # ---- NN-support MoE knobs ------------------------------------------
    "support_k_buckets": 8,
    "support_k_neighbors": 5,
    "expert_weight_multiplier": 5.0,
    # Single tau picked from the prior probe sweep (1.0xstd was the
    # winner at -0.0038 nat over rich_baseline). Sweep would 5x the
    # combiner-fit time without changing the per-variant routed
    # prediction substantially.
    "support_kernel_tau_mult": 1.0,

    # ---- Combiner knobs ------------------------------------------------
    "combiner_uniform_avg": True,
    "combiner_logit_linear_stacker": True,
    "combiner_fwls_stacker": True,
    "combiner_small_mlp": True,
    # Small MLP combiner -- intentionally tiny so it doesn't overfit
    # to 600K OOF rows with ~10 input features.
    "mlp_hidden": 16,
    "mlp_dropout": 0.2,
    "mlp_lr": 5.0e-3,
    "mlp_wd": 1.0e-4,
    "mlp_epochs": 60,
    "mlp_patience": 8,
    "mlp_val_fraction": 0.10,

    "seed": SEED,
}

print("Experiment config (feature-dropout x NN-support MoE x combiner):")
for k, v in EXP.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Load data + item-cold split (item-based sub-sample)

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

cache_root = ROOT / CFG["encoder"]["cache_dir"]
drive_status = drive_cache_mod.resolve_cache(
    cfg=CFG, encoder_slug=slug,
    local_cache_root=cache_root, expected_hash=CONTENT_HASH,
)
print("Cache decision:", drive_status.reason, "(hit:", bool(drive_status.cache_hit), ")")

embedder.warm_caches_from_disk()
print("Encoding items (cache-aware)...")
item_emb_lookup, item_log = embedder.embed_unique(
    kind="item", keys=item_keys_list, texts=item_texts_list,
    benchmarks=item_benches_list,
)
print(f"  cached={item_log['n_cache_hits']}  encoded={item_log['n_encoded']}")
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
# ## 4. Subject indexer + per-row pointers + r2u

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


def _bc_ids(rows_df) -> np.ndarray:
    return np.fromiter(
        (indexer.bc_id(str(s)) for s in rows_df["benchmark_condition_key"]),
        count=int(len(rows_df)), dtype=np.int64,
    )


subj_train = _subject_ids(train_df)
subj_val = _subject_ids(val_df)
bc_train = _bc_ids(train_df)
bc_val = _bc_ids(val_df)

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
# ## 5. Item-cold OOF folds

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

# %% [markdown]
# ## 6. Dense item-form block

# %%
from src.item_features import (
    POOL_FEATURE_NAMES as _POOL_FEATURE_NAMES,
    ITEM_TYPE_NAMES,
    apply_zscore,
    build_cot_interactions as _build_cot_interactions,
    compute_features_for_items,
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


_form_train = _build_form_block(
    _train_keys, train_df["condition"].astype(str).to_numpy()
)
_form_val = _build_form_block(
    _val_keys, val_df["condition"].astype(str).to_numpy()
)
FORM_DIM = int(_form_train.shape[1])
print(f"[form] dense form block width = {FORM_DIM}")

# %% [markdown]
# ## 7. Subject + benchmark metadata id arrays + numeric arrays

# %%
from src.data import prepare_metadata_artifacts
from src.metadata_features import MetadataSchema

META_SECTION = CFG["metadata"]
_meta_schema = MetadataSchema(
    subject_categorical=tuple(META_SECTION.get("subject_categorical", ()) or ()),
    subject_numeric=tuple(META_SECTION.get("subject_numeric", ()) or ()),
    benchmark_categorical=tuple(META_SECTION.get("benchmark_categorical", ()) or ()),
    benchmark_numeric=tuple(META_SECTION.get("benchmark_numeric", ()) or ()),
)
meta_preprocessor, meta_id_tables = prepare_metadata_artifacts(
    primary.train, indexer, schema=_meta_schema,
)

_SUBJ_CAT_FIELDS = list(_meta_schema.subject_categorical)
_BENCH_CAT_FIELDS = list(_meta_schema.benchmark_categorical)
_SUBJ_NUM_FIELDS = list(_meta_schema.subject_numeric)
_BENCH_NUM_FIELDS = list(_meta_schema.benchmark_numeric)

_SUBJ_CAT_TBL = meta_id_tables.subject_cat_ids.cpu().numpy().astype(np.int64)
_SUBJ_NUM_TBL = meta_id_tables.subject_num.cpu().numpy().astype(np.float32)
_BC_CAT_TBL = meta_id_tables.bc_cat_ids.cpu().numpy().astype(np.int64)
_BC_NUM_TBL = meta_id_tables.bc_num.cpu().numpy().astype(np.float32)

_FAMILY_COL = (_SUBJ_CAT_FIELDS.index("family") if "family" in _SUBJ_CAT_FIELDS else -1)
_MACRO_COL = (_SUBJ_CAT_FIELDS.index("macro_family") if "macro_family" in _SUBJ_CAT_FIELDS else -1)
_ORG_COL = (_SUBJ_CAT_FIELDS.index("organization") if "organization" in _SUBJ_CAT_FIELDS else -1)
_TOPIC_COL = (_BENCH_CAT_FIELDS.index("topic") if "topic" in _BENCH_CAT_FIELDS else -1)


def _vocab_n(name: str, side: str) -> int:
    vocabs = (
        meta_preprocessor.subject_cat_vocabs if side == "subject"
        else meta_preprocessor.benchmark_cat_vocabs
    )
    v = vocabs.get(name)
    return int(v.n_tokens) if v is not None else 1


N_FAMILIES = _vocab_n("family", "subject") if _FAMILY_COL >= 0 else 1
N_MACROS = _vocab_n("macro_family", "subject") if _MACRO_COL >= 0 else 1
N_ORGS = _vocab_n("organization", "subject") if _ORG_COL >= 0 else 1
N_TOPICS = _vocab_n("topic", "benchmark") if _TOPIC_COL >= 0 else 1
print(
    f"[meta] cardinalities: families={N_FAMILIES}  macros={N_MACROS}  "
    f"orgs={N_ORGS}  topics={N_TOPICS}"
)


def _gather_cats(subject_ids, bc_ids):
    s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    n = int(s.shape[0])
    n_subj = int(_SUBJ_CAT_TBL.shape[0])
    n_bc = int(_BC_CAT_TBL.shape[0])
    s_clamped = np.clip(s, 0, max(n_subj - 1, 0)).astype(np.int64)
    b_clamped = np.clip(b, 0, max(n_bc - 1, 0)).astype(np.int64)
    s_valid = (s >= 0) & (s < n_subj)
    b_valid = (b >= 0) & (b < n_bc)
    family = (
        np.where(s_valid, _SUBJ_CAT_TBL[s_clamped, _FAMILY_COL], 0)
        if _FAMILY_COL >= 0 else np.zeros(n, dtype=np.int64)
    )
    macro = (
        np.where(s_valid, _SUBJ_CAT_TBL[s_clamped, _MACRO_COL], 0)
        if _MACRO_COL >= 0 else np.zeros(n, dtype=np.int64)
    )
    org = (
        np.where(s_valid, _SUBJ_CAT_TBL[s_clamped, _ORG_COL], 0)
        if _ORG_COL >= 0 else np.zeros(n, dtype=np.int64)
    )
    topic = (
        np.where(b_valid, _BC_CAT_TBL[b_clamped, _TOPIC_COL], 0)
        if _TOPIC_COL >= 0 else np.zeros(n, dtype=np.int64)
    )
    return family.astype(np.int64), macro.astype(np.int64), \
           org.astype(np.int64), topic.astype(np.int64)


def _gather_nums(subject_ids, bc_ids):
    s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    n = int(s.shape[0])
    n_subj = int(_SUBJ_NUM_TBL.shape[0])
    n_bc = int(_BC_NUM_TBL.shape[0])
    s_clamped = np.clip(s, 0, max(n_subj - 1, 0)).astype(np.int64)
    b_clamped = np.clip(b, 0, max(n_bc - 1, 0)).astype(np.int64)
    s_valid = (s >= 0) & (s < n_subj)
    b_valid = (b >= 0) & (b < n_bc)
    if _SUBJ_NUM_TBL.shape[1] > 0:
        subj_num = _SUBJ_NUM_TBL[s_clamped].astype(np.float32, copy=True)
        cold = ~s_valid
        if cold.any():
            for j in range(int(len(_SUBJ_NUM_FIELDS))):
                subj_num[cold, 2 * j] = 0.0
                subj_num[cold, 2 * j + 1] = 1.0
    else:
        subj_num = np.zeros((n, 0), dtype=np.float32)
    if _BC_NUM_TBL.shape[1] > 0:
        bench_num = _BC_NUM_TBL[b_clamped].astype(np.float32, copy=True)
        cold = ~b_valid
        if cold.any():
            for j in range(int(len(_BENCH_NUM_FIELDS))):
                bench_num[cold, 2 * j] = 0.0
                bench_num[cold, 2 * j + 1] = 1.0
    else:
        bench_num = np.zeros((n, 0), dtype=np.float32)
    return subj_num, bench_num


family_train, macro_train, org_train, topic_train = _gather_cats(subj_train, bc_train)
family_val,   macro_val,   org_val,   topic_val   = _gather_cats(subj_val,   bc_val)
_subj_num_train, _bench_num_train = _gather_nums(subj_train, bc_train)
_subj_num_val,   _bench_num_val   = _gather_nums(subj_val,   bc_val)

SUBJ_NUM_DIM = int(_subj_num_train.shape[1]) if EXP["use_subject_numeric"] else 0
BENCH_NUM_DIM = int(_bench_num_train.shape[1]) if EXP["use_bench_numeric"] else 0
print(f"[meta] subject_num_dim={SUBJ_NUM_DIM}  bench_num_dim={BENCH_NUM_DIM}")

# %% [markdown]
# ## 8. NN block (15 cells; global passrate; no per-fold rebuild)

# %%
import gc

from src.nn_features import (
    NNFeaturesConfig,
    TrainingNNIndex,
    build_passrate_table,
    compute_nn_features_streaming,
)

if EXP["use_nn_block"]:
    nn_cfg = NNFeaturesConfig.from_dict(CFG["nn_features"])
    NN_DIR = ROOT / nn_cfg.cache_dir / "training_combiner"
    NN_DIR.mkdir(parents=True, exist_ok=True)

    train_item_keys = sorted(set(primary.train["item_key"].astype(str)))
    train_item_keys = [k for k in train_item_keys if k in item_emb_lookup]
    print(f"[nn] building NN index on {len(train_item_keys):,} unique train items...")
    nn_index = TrainingNNIndex.build_from_lookup(
        item_emb_lookup=item_emb_lookup, out_dir=NN_DIR, cfg=nn_cfg,
        item_keys=train_item_keys,
    )
    nn_item_index_map = {k: i for i, k in enumerate(train_item_keys)}
    nn_passrate_csr, nn_passrate_mask_csr = build_passrate_table(
        train_df=primary.train,
        item_index_map=nn_item_index_map,
        subject_index_map=indexer.subject_to_id,
    )
    _NN_CHUNK = int(CFG["nn_features"].get("query_chunk_size", 4096))
    print("[nn] computing NN features for train...")
    nn_train_mat = compute_nn_features_streaming(
        query_item_keys=train_df["item_key"].astype(str).tolist(),
        item_emb_lookup=item_emb_lookup,
        subject_ids=subj_train, nn_index=nn_index,
        passrate_csr=nn_passrate_csr, passrate_mask_csr=nn_passrate_mask_csr,
        cfg=nn_cfg, exclude_self=True, query_chunk_size=_NN_CHUNK,
    )
    gc.collect()
    print("[nn] computing NN features for val...")
    nn_val_mat = compute_nn_features_streaming(
        query_item_keys=val_df["item_key"].astype(str).tolist(),
        item_emb_lookup=item_emb_lookup,
        subject_ids=subj_val, nn_index=nn_index,
        passrate_csr=nn_passrate_csr, passrate_mask_csr=nn_passrate_mask_csr,
        cfg=nn_cfg, exclude_self=False, query_chunk_size=_NN_CHUNK,
    )
    NN_DIM = int(nn_train_mat.shape[1])
    print(f"[nn] nn_train_mat={nn_train_mat.shape}  nn_val_mat={nn_val_mat.shape}")
else:
    nn_train_mat = np.zeros((N_TRAIN, 0), dtype=np.float32)
    nn_val_mat = np.zeros((N_VAL, 0), dtype=np.float32)
    NN_DIM = 0

# %% [markdown]
# ## 9. Cluster block: k-means on item embeddings -> cluster_id + 8 centroid distances

# %%
if EXP["use_cluster_block"]:
    from sklearn.cluster import KMeans

    K_CLUSTERS = int(EXP["n_clusters"])
    _uniq_train_item_keys = sorted(
        k for k in set(_train_keys.tolist()) if k in item_emb_lookup
    )
    _uniq_train_emb = np.stack(
        [item_emb_lookup[k] for k in _uniq_train_item_keys]
    ).astype(np.float32)
    _norms = np.linalg.norm(_uniq_train_emb, axis=1, keepdims=True)
    _uniq_train_emb_n = (_uniq_train_emb / np.clip(_norms, 1e-12, None)).astype(np.float32)
    print(
        f"[cluster] fitting k-means K={K_CLUSTERS} on "
        f"{len(_uniq_train_item_keys):,} unique train items..."
    )
    km = KMeans(
        n_clusters=K_CLUSTERS, n_init=4,
        max_iter=int(EXP["kmeans_iters"]),
        random_state=int(EXP["kmeans_seed"]),
    )
    km.fit(_uniq_train_emb_n)
    _centroids_n = km.cluster_centers_.astype(np.float32)
    _centroid_norms = np.linalg.norm(_centroids_n, axis=1, keepdims=True)
    _centroids_n = (_centroids_n / np.clip(_centroid_norms, 1e-12, None)).astype(np.float32)
    _cluster_id_by_item: dict[str, int] = dict(
        zip(_uniq_train_item_keys, km.labels_.astype(np.int64).tolist())
    )
    _uniq_val_keys = sorted(
        k for k in set(_val_keys.tolist())
        if k in item_emb_lookup and k not in _cluster_id_by_item
    )
    if _uniq_val_keys:
        _uniq_val_emb = np.stack(
            [item_emb_lookup[k] for k in _uniq_val_keys]
        ).astype(np.float32)
        _uniq_val_emb_n = _uniq_val_emb / np.clip(
            np.linalg.norm(_uniq_val_emb, axis=1, keepdims=True), 1e-12, None,
        )
        _val_cluster_ids = km.predict(_uniq_val_emb_n.astype(np.float32))
        for k, c in zip(_uniq_val_keys, _val_cluster_ids.tolist()):
            _cluster_id_by_item[k] = int(c)
        del _uniq_val_emb, _uniq_val_emb_n

    def _cluster_ids(item_keys) -> np.ndarray:
        return np.fromiter(
            (int(_cluster_id_by_item.get(str(k), K_CLUSTERS)) for k in item_keys),
            count=int(len(item_keys)), dtype=np.int64,
        )

    def _centroid_distances(item_keys) -> np.ndarray:
        ik = [str(k) for k in item_keys]
        n = len(ik)
        out = np.zeros((n, K_CLUSTERS), dtype=np.float32)
        for i, k in enumerate(ik):
            if k in item_emb_lookup:
                e = item_emb_lookup[k]
                en = e / max(np.linalg.norm(e), 1e-12)
                out[i] = (1.0 - _centroids_n @ en.astype(np.float32)).astype(np.float32)
            else:
                out[i] = 1.0
        return out

    cluster_train = _cluster_ids(_train_keys)
    cluster_val = _cluster_ids(_val_keys)
    _centroid_dist_train = _centroid_distances(_train_keys)
    _centroid_dist_val = _centroid_distances(_val_keys)
    CENTROID_DIST_DIM = K_CLUSTERS
else:
    K_CLUSTERS = 1
    cluster_train = np.zeros(N_TRAIN, dtype=np.int64)
    cluster_val = np.zeros(N_VAL, dtype=np.int64)
    _centroid_dist_train = np.zeros((N_TRAIN, 0), dtype=np.float32)
    _centroid_dist_val = np.zeros((N_VAL, 0), dtype=np.float32)
    CENTROID_DIST_DIM = 0

# %% [markdown]
# ## 9b. Val-side metadata redaction (item-grouped + subject-grouped)

# %%
EXP.setdefault("redact_val", True)
EXP.setdefault("val_dropout_bc", EXP["rich_cat_dropout_bc"])
EXP.setdefault("val_dropout_topic", EXP["rich_cat_dropout_topic"])
EXP.setdefault("val_dropout_cluster", EXP["rich_cat_dropout_cluster"])
EXP.setdefault("val_dropout_subject", EXP["rich_cat_dropout_subject"])
EXP.setdefault("val_dropout_family", EXP["rich_cat_dropout_family"])
EXP.setdefault("val_dropout_macro", EXP["rich_cat_dropout_macro"])
EXP.setdefault("val_dropout_org", EXP["rich_cat_dropout_org"])

_UNK_BC = int(indexer.n_bc)
_UNK_TOPIC = int(N_TOPICS)
_UNK_CLUSTER = int(K_CLUSTERS) if EXP["use_cluster_block"] else 0
_UNK_SUBJ = int(indexer.n_subjects)
_UNK_FAM = int(N_FAMILIES)
_UNK_MAC = int(N_MACROS)
_UNK_ORG = int(N_ORGS)


def _build_val_unit_masks(seed: int):
    rng = np.random.default_rng(int(seed))
    n_items_total = int(ALL_UNIQ.shape[0])
    n_subj_total = int(indexer.n_subjects)
    item_bc = rng.uniform(size=n_items_total) < float(EXP["val_dropout_bc"])
    item_topic = rng.uniform(size=n_items_total) < float(EXP["val_dropout_topic"])
    item_cluster = rng.uniform(size=n_items_total) < float(EXP["val_dropout_cluster"])
    subj_subj = rng.uniform(size=n_subj_total) < float(EXP["val_dropout_subject"])
    subj_fam = rng.uniform(size=n_subj_total) < float(EXP["val_dropout_family"])
    subj_mac = rng.uniform(size=n_subj_total) < float(EXP["val_dropout_macro"])
    subj_org = rng.uniform(size=n_subj_total) < float(EXP["val_dropout_org"])
    return (item_bc, item_topic, item_cluster, subj_subj, subj_fam, subj_mac, subj_org)


def _apply_val_redaction(
    bc, topic, cluster, subj, family, macro, org,
    *, item_per_row, subj_per_row, masks,
):
    item_bc, item_topic, item_cluster, subj_subj, subj_fam, subj_mac, subj_org = masks
    redact_bc = item_bc[np.clip(item_per_row, 0, len(item_bc) - 1)]
    redact_topic = item_topic[np.clip(item_per_row, 0, len(item_topic) - 1)]
    redact_cluster = item_cluster[np.clip(item_per_row, 0, len(item_cluster) - 1)]
    bc_v = np.where(redact_bc, _UNK_BC, bc).astype(np.int64)
    topic_v = np.where(redact_topic, _UNK_TOPIC, topic).astype(np.int64)
    cluster_v = np.where(redact_cluster, _UNK_CLUSTER, cluster).astype(np.int64)
    safe_subj = np.clip(subj_per_row, 0, len(subj_subj) - 1)
    redact_subj = subj_subj[safe_subj]
    redact_fam = subj_fam[safe_subj]
    redact_mac = subj_mac[safe_subj]
    redact_org = subj_org[safe_subj]
    subj_v = np.where(redact_subj, _UNK_SUBJ, subj).astype(np.int64)
    family_v = np.where(redact_fam, _UNK_FAM, family).astype(np.int64)
    macro_v = np.where(redact_mac, _UNK_MAC, macro).astype(np.int64)
    org_v = np.where(redact_org, _UNK_ORG, org).astype(np.int64)
    return bc_v, topic_v, cluster_v, subj_v, family_v, macro_v, org_v


if EXP["redact_val"]:
    _val_masks = _build_val_unit_masks(seed=SEED + 2026)
    (
        bc_val_red, topic_val_red, cluster_val_red,
        subj_val_red, family_val_red, macro_val_red, org_val_red,
    ) = _apply_val_redaction(
        bc_val, topic_val, cluster_val, subj_val,
        family_val, macro_val, org_val,
        item_per_row=r2u_val, subj_per_row=subj_val, masks=_val_masks,
    )
else:
    bc_val_red = bc_val.copy()
    topic_val_red = topic_val.copy()
    cluster_val_red = cluster_val.copy()
    subj_val_red = subj_val.copy()
    family_val_red = family_val.copy()
    macro_val_red = macro_val.copy()
    org_val_red = org_val.copy()

# %% [markdown]
# ## 10. Rich dense block + column-offset tracking

# %%
_rich_parts_train = [_form_train]
_rich_parts_val = [_form_val]
_blocks: list[tuple[str, int, int]] = []
_off = 0
_blocks.append(("form", _off, FORM_DIM)); _off += FORM_DIM
if SUBJ_NUM_DIM > 0:
    _rich_parts_train.append(_subj_num_train)
    _rich_parts_val.append(_subj_num_val)
    _blocks.append(("subj_num", _off, SUBJ_NUM_DIM)); _off += SUBJ_NUM_DIM
if BENCH_NUM_DIM > 0:
    _rich_parts_train.append(_bench_num_train)
    _rich_parts_val.append(_bench_num_val)
    _blocks.append(("bench_num", _off, BENCH_NUM_DIM)); _off += BENCH_NUM_DIM
if NN_DIM > 0:
    _rich_parts_train.append(nn_train_mat.astype(np.float32))
    _rich_parts_val.append(nn_val_mat.astype(np.float32))
    _blocks.append(("nn", _off, NN_DIM)); _off += NN_DIM
if CENTROID_DIST_DIM > 0:
    _rich_parts_train.append(_centroid_dist_train.astype(np.float32))
    _rich_parts_val.append(_centroid_dist_val.astype(np.float32))
    _blocks.append(("centroid_dist", _off, CENTROID_DIST_DIM)); _off += CENTROID_DIST_DIM
_rich_train_raw = np.concatenate(_rich_parts_train, axis=1).astype(np.float32)
_rich_val_raw = np.concatenate(_rich_parts_val, axis=1).astype(np.float32)
_rmean = _rich_train_raw.mean(axis=0).astype(np.float32)
_rstd = _rich_train_raw.std(axis=0).astype(np.float32)
_rstd = np.where(_rstd < 1e-6, 1.0, _rstd).astype(np.float32)
rich_dense_train = ((_rich_train_raw - _rmean) / _rstd).astype(np.float32)
rich_dense_val = ((_rich_val_raw - _rmean) / _rstd).astype(np.float32)
RICH_DENSE_DIM = int(rich_dense_train.shape[1])
DENSE_BLOCK_OFFSETS = {name: (start, width) for name, start, width in _blocks}
print(
    f"[dense] rich dense width = {RICH_DENSE_DIM};  blocks: "
    + ", ".join(f"{n}:[{s}:{s + w})" for (n, s, w) in _blocks)
)
del _rich_train_raw, _rich_val_raw

# %% [markdown]
# ## 11. NN-support partition (FAISS index + per-row support score + 8 buckets)

# %%
import faiss

_SUP_K = int(EXP["support_k_buckets"])
K_NN = int(EXP["support_k_neighbors"])

# Reuse the unique train embedding stack from the k-means cell if it ran.
if EXP["use_cluster_block"]:
    _support_emb_uniq = _uniq_train_emb_n
    _support_uniq_keys = _uniq_train_item_keys
else:
    _support_uniq_keys = sorted(
        k for k in set(_train_keys.tolist()) if k in item_emb_lookup
    )
    _support_emb_raw = np.stack(
        [item_emb_lookup[k] for k in _support_uniq_keys]
    ).astype(np.float32)
    _support_emb_uniq = (
        _support_emb_raw / np.clip(
            np.linalg.norm(_support_emb_raw, axis=1, keepdims=True), 1e-12, None,
        )
    ).astype(np.float32)
    del _support_emb_raw

print(
    f"[partition support] FAISS IP index on {len(_support_uniq_keys):,} unique "
    f"train items (D={ITEM_EMB_DIM})..."
)
_sup_idx = faiss.IndexFlatIP(ITEM_EMB_DIM)
_sup_idx.add(_support_emb_uniq)
_sims_train, _ = _sup_idx.search(_support_emb_uniq, K_NN + 1)
_score_train_uniq = _sims_train[:, 1:].mean(axis=1).astype(np.float32)

_uniq_val_keys_sup = sorted(
    k for k in set(_val_keys.tolist()) if k in item_emb_lookup
)
_uniq_val_emb_sup_raw = np.stack(
    [item_emb_lookup[k] for k in _uniq_val_keys_sup]
).astype(np.float32)
_uniq_val_emb_sup = (
    _uniq_val_emb_sup_raw / np.clip(
        np.linalg.norm(_uniq_val_emb_sup_raw, axis=1, keepdims=True), 1e-12, None,
    )
).astype(np.float32)
_sims_val, _ = _sup_idx.search(_uniq_val_emb_sup, K_NN)
_score_val_uniq = _sims_val.mean(axis=1).astype(np.float32)

_sup_boundaries = np.quantile(
    _score_train_uniq, np.linspace(0.0, 1.0, _SUP_K + 1)
)[1:-1].astype(np.float32)
print(
    f"[partition support] K={_SUP_K}  cut points: "
    + ", ".join(f"{b:.4f}" for b in _sup_boundaries)
)

_support_by_item: dict[str, float] = dict(
    zip(_support_uniq_keys, _score_train_uniq.tolist())
)
for k, s in zip(_uniq_val_keys_sup, _score_val_uniq.tolist()):
    _support_by_item[k] = float(s)
_median_sup = float(np.median(_score_train_uniq))


def _support_score(item_keys) -> np.ndarray:
    return np.fromiter(
        (float(_support_by_item.get(str(k), _median_sup)) for k in item_keys),
        count=int(len(item_keys)), dtype=np.float32,
    )


def _support_bucket(scores: np.ndarray) -> np.ndarray:
    return np.searchsorted(_sup_boundaries, scores, side="right").astype(np.int64)


support_score_train = _support_score(_train_keys)
support_score_val = _support_score(_val_keys)
bucket_support_train = _support_bucket(support_score_train)
bucket_support_val = _support_bucket(support_score_val)
_SUP_BUCKET_NAMES = [f"support_q{j + 1}" for j in range(_SUP_K)]
print("  train bucket sizes:")
for j, name in enumerate(_SUP_BUCKET_NAMES):
    n = int((bucket_support_train == j).sum())
    print(f"    {name:<14s} train_rows={n:>9,} ({n / N_TRAIN:>5.1%})")

# Bucket centroids on TRAIN scores for the Gaussian-kernel soft router.
_sup_centroids = np.array([
    float(support_score_train[bucket_support_train == j].mean())
    if int((bucket_support_train == j).sum()) > 0
    else float(np.mean(_sup_boundaries))
    for j in range(_SUP_K)
], dtype=np.float32)
print(
    f"[partition support] bucket centroids on train: "
    + ", ".join(f"{c:.4f}" for c in _sup_centroids)
)

del _sup_idx, _sims_train, _sims_val
gc.collect()

# %% [markdown]
# ## 12. Feature-class specs + per-variant masking helpers

# %%
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    drop_subject_channel: bool = False
    drop_bench_channel: bool = False
    drop_form_block: bool = False
    drop_nn_block: bool = False
    drop_cluster_channel: bool = False
    drop_item_embedding: bool = False


VARIANT_SPECS: dict[str, FeatureSpec] = {
    "full":               FeatureSpec(),
    "no_subj_channel":    FeatureSpec(drop_subject_channel=True),
    "no_bench_channel":   FeatureSpec(drop_bench_channel=True),
    "no_form_block":      FeatureSpec(drop_form_block=True),
    "no_nn_block":        FeatureSpec(drop_nn_block=True),
    "no_cluster_channel": FeatureSpec(drop_cluster_channel=True),
}
VARIANT_KEYS = [v for v in EXP["variants"] if v in VARIANT_SPECS]
assert "full" in VARIANT_KEYS, "the 'full' variant is the comparison anchor"
print(f"[variants] active ({len(VARIANT_KEYS)}): {VARIANT_KEYS}")


def _apply_dense_mask(dense_mat: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    any_drop = (
        spec.drop_form_block or spec.drop_subject_channel or
        spec.drop_bench_channel or spec.drop_nn_block or
        spec.drop_cluster_channel
    )
    if not any_drop:
        return dense_mat
    out = dense_mat.copy()
    if spec.drop_form_block and "form" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["form"]; out[:, s:s + w] = 0.0
    if spec.drop_subject_channel and "subj_num" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["subj_num"]; out[:, s:s + w] = 0.0
    if spec.drop_bench_channel and "bench_num" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["bench_num"]; out[:, s:s + w] = 0.0
    if spec.drop_nn_block and "nn" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["nn"]; out[:, s:s + w] = 0.0
    if spec.drop_cluster_channel and "centroid_dist" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["centroid_dist"]; out[:, s:s + w] = 0.0
    return out


def _apply_cat_mask(spec: FeatureSpec, *, side: str):
    if side == "train":
        s, b, c, f, mf, o, t = (
            subj_train, bc_train, cluster_train,
            family_train, macro_train, org_train, topic_train,
        )
    elif side == "val":
        s, b, c, f, mf, o, t = (
            subj_val_red, bc_val_red, cluster_val_red,
            family_val_red, macro_val_red, org_val_red, topic_val_red,
        )
    else:
        raise ValueError(side)
    if spec.drop_subject_channel:
        s = np.full_like(s, _UNK_SUBJ)
        f = np.full_like(f, _UNK_FAM)
        mf = np.full_like(mf, _UNK_MAC)
        o = np.full_like(o, _UNK_ORG)
    if spec.drop_bench_channel:
        b = np.full_like(b, _UNK_BC)
        t = np.full_like(t, _UNK_TOPIC)
    if spec.drop_cluster_channel:
        c = np.full_like(c, _UNK_CLUSTER)
    return s, b, c, f, mf, o, t


# Pre-compute per-variant masked arrays ONCE outside the per-fold loop.
# Saves re-copying ~600K x 200-column matrices on every fold / bucket.
VARIANT_ARRAYS: dict[str, dict[str, np.ndarray]] = {}
for vname in VARIANT_KEYS:
    spec = VARIANT_SPECS[vname]
    s_tr, b_tr, c_tr, f_tr, mf_tr, o_tr, t_tr = _apply_cat_mask(spec, side="train")
    s_va, b_va, c_va, f_va, mf_va, o_va, t_va = _apply_cat_mask(spec, side="val")
    VARIANT_ARRAYS[vname] = dict(
        subj_train=s_tr, bc_train=b_tr, cluster_train=c_tr,
        family_train=f_tr, macro_train=mf_tr, org_train=o_tr, topic_train=t_tr,
        subj_val=s_va, bc_val=b_va, cluster_val=c_va,
        family_val=f_va, macro_val=mf_va, org_val=o_va, topic_val=t_va,
        dense_train=_apply_dense_mask(rich_dense_train, spec),
        dense_val=_apply_dense_mask(rich_dense_val, spec),
    )
print(
    f"[variants] pre-built masked id+dense arrays for {len(VARIANT_KEYS)} variants "
    f"(~{sum(a['dense_train'].nbytes for a in VARIANT_ARRAYS.values()) / 1e9:.2f} GB "
    "of cached dense)"
)

# %% [markdown]
# ## 13. Trainer + per-(variant, bucket, fold) OOF runner
#
# Pattern: for each (variant, bucket), train ONE rich net per fold with
# the variant's feature mask applied AND the bucket's rows upweighted
# 5x. Store both OOF and val predictions; per-fold val preds are
# averaged across folds (same as the prior probes). The seed varies
# by (variant_idx, fold_id) so different variants and folds use
# different init, but all buckets WITHIN a (variant, fold) share a
# seed so the per-bucket lift is purely from the upweighting, not
# from seed diversity.

# %%
import torch

from src.rich_mlp_variant import (
    RichMLPConfig, predict_rich_mlp, train_rich_mlp,
    soft_routing_weights_kernel, apply_soft_routing,
)

_dev = EXP["emb_device"] or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_dev}")
EMB_T = torch.from_numpy(ALL_UNIQ).to(_dev)
_EMB_T_ZERO = torch.zeros_like(EMB_T)


def _rich_cfg(seed_):
    return RichMLPConfig(
        subj_emb_dim=int(EXP["rich_subj_emb_dim"]),
        bc_emb_dim=int(EXP["rich_bc_emb_dim"]),
        cluster_emb_dim=int(EXP["rich_cluster_emb_dim"]) if EXP["use_cluster_block"] else 0,
        family_emb_dim=int(EXP["rich_family_emb_dim"]),
        macro_emb_dim=int(EXP["rich_macro_emb_dim"]),
        org_emb_dim=int(EXP["rich_org_emb_dim"]),
        topic_emb_dim=int(EXP["rich_topic_emb_dim"]),
        hid1=int(EXP["rich_hid1"]), hid2=int(EXP["rich_hid2"]),
        feat_dropout=float(EXP["rich_feat_dropout"]),
        n_cross_layers=int(EXP["rich_n_cross_layers"]),
        cross_rank=int(EXP["rich_cross_rank"]),
        cat_dropout_subject=float(EXP["rich_cat_dropout_subject"]),
        cat_dropout_bc=float(EXP["rich_cat_dropout_bc"]),
        cat_dropout_cluster=float(EXP["rich_cat_dropout_cluster"]) if EXP["use_cluster_block"] else 0.0,
        cat_dropout_family=float(EXP["rich_cat_dropout_family"]),
        cat_dropout_macro=float(EXP["rich_cat_dropout_macro"]),
        cat_dropout_org=float(EXP["rich_cat_dropout_org"]),
        cat_dropout_topic=float(EXP["rich_cat_dropout_topic"]),
        lr=float(EXP["rich_lr"]), wd=float(EXP["rich_wd"]),
        epochs=int(EXP["epochs"]), batch_size=int(EXP["batch_size"]),
        val_fraction=float(EXP["val_fraction"]), patience=int(EXP["patience"]),
        seed=int(seed_),
    )


_RICH_NS = int(indexer.n_subjects)
_RICH_NB = int(indexer.n_bc)
_RICH_NC = int(K_CLUSTERS) if EXP["use_cluster_block"] else 0
_RICH_NF = int(N_FAMILIES)
_RICH_NMF = int(N_MACROS)
_RICH_NO = int(N_ORGS)
_RICH_NT = int(N_TOPICS)


def _make_weights(in_bucket_mask, mult):
    w = np.where(in_bucket_mask, float(mult), 1.0).astype(np.float32)
    return (w / w.mean()).astype(np.float32)


def _run_variant_bucket_oof(variant_name: str, bucket_j: int,
                             variant_idx: int):
    """Train one rich net per fold for (variant, bucket); return (oof, val).

    The variant's pre-masked id+dense arrays are pulled from
    ``VARIANT_ARRAYS[variant_name]``. Bucket upweighting is applied
    via ``sample_weights`` (5x on rows in ``bucket_j``).
    """
    arr = VARIANT_ARRAYS[variant_name]
    spec = VARIANT_SPECS[variant_name]
    emb_for_variant = _EMB_T_ZERO if spec.drop_item_embedding else EMB_T
    in_bucket = (bucket_support_train == int(bucket_j))
    sample_weights = _make_weights(in_bucket, EXP["expert_weight_multiplier"])

    acc = OofPredictionAccumulator(
        N_TRAIN, name=f"oof_{variant_name}_q{bucket_j + 1}",
    )
    val_stack = []
    for fold in folds:
        tr, oof = fold.train_row_idx, fold.oof_row_idx
        cfg = _rich_cfg(SEED + 1000 * int(variant_idx) + int(fold.fold_id))
        net = train_rich_mlp(
            y=y_train[tr],
            subject_ids=arr["subj_train"][tr], bc_ids=arr["bc_train"][tr],
            cluster_ids=arr["cluster_train"][tr], family_ids=arr["family_train"][tr],
            macro_ids=arr["macro_train"][tr], org_ids=arr["org_train"][tr],
            topic_ids=arr["topic_train"][tr],
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_train[tr],
            dense_X=(arr["dense_train"][tr] if EXP["use_metadata"] else None),
            n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
            n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
            n_topics=_RICH_NT, cfg=cfg, device=_dev,
            sample_weights=sample_weights[tr], show_progress=False,
        )
        oof_p = predict_rich_mlp(
            net,
            subject_ids=arr["subj_train"][oof], bc_ids=arr["bc_train"][oof],
            cluster_ids=arr["cluster_train"][oof], family_ids=arr["family_train"][oof],
            macro_ids=arr["macro_train"][oof], org_ids=arr["org_train"][oof],
            topic_ids=arr["topic_train"][oof],
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_train[oof],
            dense_X=(arr["dense_train"][oof] if EXP["use_metadata"] else None),
            n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
            n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
            n_topics=_RICH_NT, device=_dev, chunk=int(EXP["predict_chunk"]),
        )
        val_p = predict_rich_mlp(
            net,
            subject_ids=arr["subj_val"], bc_ids=arr["bc_val"],
            cluster_ids=arr["cluster_val"], family_ids=arr["family_val"],
            macro_ids=arr["macro_val"], org_ids=arr["org_val"],
            topic_ids=arr["topic_val"],
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_val,
            dense_X=(arr["dense_val"] if EXP["use_metadata"] else None),
            n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
            n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
            n_topics=_RICH_NT, device=_dev, chunk=int(EXP["predict_chunk"]),
        )
        acc.write_fold(oof, oof_p)
        val_stack.append(val_p)
        del net
        if _dev == "cuda":
            torch.cuda.empty_cache()
    return (
        acc.finalize().astype(np.float32),
        np.mean(val_stack, axis=0).astype(np.float32),
    )

# %% [markdown]
# ## 14. Run the (variant x bucket) grid
#
# Default size: ``len(variants) * support_k_buckets * n_folds`` = 6 * 8 * 3
# = 144 trainings. Set EXP['variants'] to a shorter list (or shrink
# ``support_k_buckets``) to cut compute. Progress is printed every
# bucket; a per-(variant, bucket) heading lets you tell how far in
# you are by reading the last log line.

# %%
import time

# oof_grid[variant][bucket_name] = [N_TRAIN] OOF predictions
# val_grid[variant][bucket_name] = [N_VAL] val predictions (fold-averaged)
oof_grid: dict[str, dict[str, np.ndarray]] = {v: {} for v in VARIANT_KEYS}
val_grid: dict[str, dict[str, np.ndarray]] = {v: {} for v in VARIANT_KEYS}

_t0 = time.time()
_N_combos = len(VARIANT_KEYS) * _SUP_K
_done = 0
for vi, vname in enumerate(VARIANT_KEYS):
    for j, bname in enumerate(_SUP_BUCKET_NAMES):
        _done += 1
        n_in = int((bucket_support_train == j).sum())
        print(
            f"\n[{_done:>3d}/{_N_combos}] variant={vname:<22s} bucket={bname}  "
            f"({n_in:,} bucket rows; elapsed {(time.time() - _t0) / 60:.1f} min)"
        )
        oof_p, val_p = _run_variant_bucket_oof(vname, j, variant_idx=vi)
        oof_grid[vname][bname] = oof_p
        val_grid[vname][bname] = val_p
print(
    f"\n[grid] done. Trained {len(VARIANT_KEYS) * _SUP_K * len(folds)} "
    f"rich nets in {(time.time() - _t0) / 60:.1f} min"
)

# %% [markdown]
# ## 15. Per-variant soft-routed prediction (NN-support kernel, tau = 1.0xstd)
#
# Within each variant, combine its 8 bucket experts via the Gaussian
# kernel router from :func:`src.rich_mlp_variant.soft_routing_weights_kernel`.
# Uses ``tau = 1.0 * std(support_score_*)``, the winner from the prior
# probe's tau sweep. Output: one routed prediction per variant on OOF
# (used by the combiner) and on val (used for solo NLL + combiner).

# %%
_std_score_tr = float(np.std(support_score_train).clip(1e-6, None))
_std_score_va = float(np.std(support_score_val).clip(1e-6, None))
_TAU_TR = float(EXP["support_kernel_tau_mult"]) * _std_score_tr
_TAU_VA = float(EXP["support_kernel_tau_mult"]) * _std_score_va
print(
    f"[soft route] tau_train={_TAU_TR:.4f}  tau_val={_TAU_VA:.4f}  "
    f"(tau_mult={EXP['support_kernel_tau_mult']}xstd)"
)

_w_tr = soft_routing_weights_kernel(
    support_score_train, bucket_centroids=_sup_centroids, tau=_TAU_TR,
)  # [N_TRAIN, K]
_w_va = soft_routing_weights_kernel(
    support_score_val, bucket_centroids=_sup_centroids, tau=_TAU_VA,
)  # [N_VAL, K]

oof_routed: dict[str, np.ndarray] = {}
val_routed: dict[str, np.ndarray] = {}
for vname in VARIANT_KEYS:
    expert_keys = _SUP_BUCKET_NAMES  # keys used inside the variant's grid
    oof_routed[vname] = apply_soft_routing(
        oof_grid[vname], expert_names=expert_keys, weights=_w_tr,
    ).astype(np.float32)
    val_routed[vname] = apply_soft_routing(
        val_grid[vname], expert_names=expert_keys, weights=_w_va,
    ).astype(np.float32)
print(f"[soft route] built routed preds for {len(VARIANT_KEYS)} variants")

# %% [markdown]
# ## 16. Per-variant Platt calibration

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    build_stacker_features,
    fit_stacker,
    logit_clipped,
    stacker_feature_names,
)


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _aux_zeros(n):
    return np.zeros(int(n), dtype=np.float32)


def _aux_half(n):
    return np.full(int(n), 0.5, dtype=np.float32)


def _platt(p_oof, p_val):
    n_o, n_v = int(len(p_oof)), int(len(p_val))
    Xo = build_stacker_features(
        member_probs=p_oof[:, None], bench_present=_aux_zeros(n_o),
        nn_neighbor_support=_aux_zeros(n_o), nn_mean_similarity=_aux_zeros(n_o),
        centroid_distance=_aux_half(n_o),
    )
    st = fit_stacker(X=Xo, y=y_train, feature_names=stacker_feature_names(1),
                     seed=SEED, n_iters=1500)
    Xv = build_stacker_features(
        member_probs=p_val[:, None], bench_present=_aux_zeros(n_v),
        nn_neighbor_support=_aux_zeros(n_v), nn_mean_similarity=_aux_zeros(n_v),
        centroid_distance=_aux_half(n_v),
    )
    return stacker_apply_batch(st, Xv).astype(np.float32)


cal_val: dict[str, np.ndarray] = {
    v: _platt(oof_routed[v], val_routed[v]) for v in VARIANT_KEYS
}
cal_oof: dict[str, np.ndarray] = {
    v: _platt(oof_routed[v], oof_routed[v]) for v in VARIANT_KEYS
}

print("\nCalibrated solo val log-loss per variant (each variant = its own "
      "NN-support soft-routed MoE):")
_solo = {v: _nll(cal_val[v], y_val) for v in VARIANT_KEYS}
_full_routed_nll = _solo["full"]
print(f"  {'full + MoE (anchor)':<32s}: {_full_routed_nll:.6f}   (reference)")
for v in VARIANT_KEYS:
    if v == "full":
        continue
    d = _solo[v] - _full_routed_nll
    print(f"  {v + ' + MoE':<32s}: {_solo[v]:.6f}   ({d:+.6f})")

# %% [markdown]
# ## 17. Auxiliary features for the gated combiners
#
# Four numeric per-row gates used by the FWLS stacker and the MLP
# combiner. All are honest features available at inference time:
#
# * ``bench_present`` -- 1 if bc is visible (not redacted to UNK), 0
#   otherwise. Lets the combiner trust the ``no_bench_channel``
#   variant when bench is hidden in production.
# * ``nn_neighbor_support`` -- the NN-support score per row (mean
#   cosine sim to top-K train neighbours). Already used by the
#   per-variant soft router; also fed to the combiner so it can
#   gate cross-variant blending on the same axis.
# * ``nn_mean_similarity`` -- alias of the support score in this
#   implementation; kept as a separate column so the production
#   stacker schema matches (its build_stacker_features expects 4 aux).
# * ``centroid_distance`` -- min centroid distance per row from the
#   k-means atlas. A small distance means "this item lies in a dense
#   cluster the model has seen many neighbours of"; a large distance
#   means it's near the boundary of the embedding atlas (effectively
#   cold from the cluster's POV).

# %%
def _bench_present_for(bc_arr):
    return (np.asarray(bc_arr, dtype=np.int64) != int(_UNK_BC)).astype(np.float32)


def _min_centroid_dist(row_centroid_dist_mat):
    return row_centroid_dist_mat.min(axis=1).astype(np.float32)


# Train-side aux uses the ORIGINAL (un-redacted) bc_train -- the train
# OOF rows aren't part of the val redaction contract.
aux_train = dict(
    bench_present=_bench_present_for(bc_train),
    nn_neighbor_support=support_score_train.astype(np.float32),
    nn_mean_similarity=support_score_train.astype(np.float32),
    centroid_distance=(
        _min_centroid_dist(_centroid_dist_train)
        if CENTROID_DIST_DIM > 0 else _aux_half(N_TRAIN)
    ),
)
# Val-side aux uses the REDACTED bc_val_red, matching what the val
# rows actually see at predict time.
aux_val = dict(
    bench_present=_bench_present_for(bc_val_red),
    nn_neighbor_support=support_score_val.astype(np.float32),
    nn_mean_similarity=support_score_val.astype(np.float32),
    centroid_distance=(
        _min_centroid_dist(_centroid_dist_val)
        if CENTROID_DIST_DIM > 0 else _aux_half(N_VAL)
    ),
)
_AUX_NAMES = ["bench_present", "nn_neighbor_support",
              "nn_mean_similarity", "centroid_distance"]
print(
    f"[aux] built {len(_AUX_NAMES)} per-row features for combiner "
    f"({N_TRAIN:,} train; {N_VAL:,} val)"
)
print(
    "  bench_present rates: "
    f"train={aux_train['bench_present'].mean():.3f}  "
    f"val={aux_val['bench_present'].mean():.3f}  "
    f"(val < train by ~{aux_train['bench_present'].mean() - aux_val['bench_present'].mean():.3f} "
    "due to per-unit redaction)"
)

# Z-score the non-binary aux features on train stats so the FWLS /
# MLP combiners get inputs at the same scale. ``bench_present`` is
# already in {0, 1} so we leave it alone.
_aux_means = np.array([
    0.0,
    aux_train["nn_neighbor_support"].mean(),
    aux_train["nn_mean_similarity"].mean(),
    aux_train["centroid_distance"].mean(),
], dtype=np.float32)
_aux_stds = np.array([
    1.0,
    max(float(aux_train["nn_neighbor_support"].std()), 1e-6),
    max(float(aux_train["nn_mean_similarity"].std()), 1e-6),
    max(float(aux_train["centroid_distance"].std()), 1e-6),
], dtype=np.float32)


def _aux_to_matrix(d: dict, *, normalize: bool) -> np.ndarray:
    A = np.stack([d[k] for k in _AUX_NAMES], axis=1).astype(np.float32)
    if normalize:
        A = ((A - _aux_means[None]) / _aux_stds[None]).astype(np.float32)
    return A


aux_train_mat_norm = _aux_to_matrix(aux_train, normalize=True)
aux_val_mat_norm = _aux_to_matrix(aux_val, normalize=True)
aux_train_mat_raw = _aux_to_matrix(aux_train, normalize=False)
aux_val_mat_raw = _aux_to_matrix(aux_val, normalize=False)
print(f"[aux] aux_train_mat shape={aux_train_mat_norm.shape}")

# %% [markdown]
# ## 18. Combiners
#
# 18a. **Uniform avg** -- baseline ensemble.
# 18b. **Logit-linear stacker** -- production-style; one global weight
#      per member, aux features as standalone bias columns. This is
#      what `feature_dropout_ensemble.py`'s Platt stacker did, applied
#      here to the routed-MoE predictions instead of the raw
#      per-variant predictions.
# 18c. **FWLS stacker** -- explicit (member x aux) cross terms. Built
#      by hand because `fit_stacker` only adds aux features as
#      standalone columns. Gives per-row gated linear blending.
# 18d. **Small MLP combiner** -- a tiny 2-layer net on the K+1 logits
#      + 4 aux features. Non-linear; can express interactions the
#      stacker can't.

# %%
print("\n" + "=" * 80)
print("COMBINERS -- val NLL")
print("=" * 80)

_M = len(VARIANT_KEYS)
_G = len(_AUX_NAMES)
_stack_oof = np.stack([cal_oof[v] for v in VARIANT_KEYS], axis=1).astype(np.float32)
_stack_val = np.stack([cal_val[v] for v in VARIANT_KEYS], axis=1).astype(np.float32)

# 18a. Uniform avg
if EXP["combiner_uniform_avg"]:
    _uniform_val = _stack_val.mean(axis=1).astype(np.float32)
    nll_uniform = _nll(_uniform_val, y_val)
    print(f"  uniform_avg                       : {nll_uniform:.6f}   "
          f"(vs full+MoE: {nll_uniform - _full_routed_nll:+.6f})")

# 18b. Logit-linear stacker (member probs + aux as standalone columns).
#      This is what build_stacker_features produces; reusing it keeps
#      apples-to-apples with the prior probe's stacker.
if EXP["combiner_logit_linear_stacker"]:
    Xo_ll = build_stacker_features(
        member_probs=_stack_oof,
        bench_present=aux_train["bench_present"],
        nn_neighbor_support=aux_train["nn_neighbor_support"],
        nn_mean_similarity=aux_train["nn_mean_similarity"],
        centroid_distance=aux_train["centroid_distance"],
    )
    Xv_ll = build_stacker_features(
        member_probs=_stack_val,
        bench_present=aux_val["bench_present"],
        nn_neighbor_support=aux_val["nn_neighbor_support"],
        nn_mean_similarity=aux_val["nn_mean_similarity"],
        centroid_distance=aux_val["centroid_distance"],
    )
    stk_ll = fit_stacker(
        X=Xo_ll, y=y_train, feature_names=stacker_feature_names(_M),
        seed=SEED + 11, n_iters=2000,
    )
    _val_ll = stacker_apply_batch(stk_ll, Xv_ll).astype(np.float32)
    nll_ll = _nll(_val_ll, y_val)
    print(f"  logit_linear_stacker              : {nll_ll:.6f}   "
          f"(vs full+MoE: {nll_ll - _full_routed_nll:+.6f})")
    _w_ll = np.asarray(stk_ll.weights, dtype=np.float32).reshape(-1)
    print("  weights (member channels):")
    for i, v in enumerate(VARIANT_KEYS):
        print(f"    {v:<28s} weight={_w_ll[i]:+.4f}")

# 18c. FWLS stacker -- true per-row gated linear blend.
#
# Feature layout (column order):
#   [0 .. M)                : logit(member_i)               (the M base members)
#   [M .. M + M*G)          : logit(member_i) * aux_g       (M*G cross terms)
#   [M + M*G .. M + M*G+G)  : aux_g                          (G standalone aux for bias)
#
# Standalone aux columns are kept normalized; cross terms use
# RAW aux so the per-row gating has interpretable units (e.g. a
# weight on "logit(no_bench) * (1 - bench_present)" reads as
# "trust no_bench more by W per unit of bench-redaction").
if EXP["combiner_fwls_stacker"]:
    def _build_fwls_features(member_probs, aux_mat_raw, aux_mat_norm):
        N, M = member_probs.shape
        G = aux_mat_raw.shape[1]
        lp = np.asarray(logit_clipped(np.asarray(member_probs, dtype=np.float64))).astype(np.float32)
        out = np.zeros((N, M + M * G + G), dtype=np.float32)
        out[:, :M] = lp
        for g in range(G):
            out[:, M + g * M:M + (g + 1) * M] = (
                lp * aux_mat_raw[:, g:g + 1].astype(np.float32)
            )
        out[:, M + M * G:M + M * G + G] = aux_mat_norm
        out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
        return out

    _fwls_feat_names = (
        [f"m_{v}" for v in VARIANT_KEYS]
        + [f"m_{v}_x_{g}" for g in _AUX_NAMES for v in VARIANT_KEYS]
        + list(_AUX_NAMES)
    )
    Xo_fwls = _build_fwls_features(
        _stack_oof, aux_train_mat_raw, aux_train_mat_norm,
    )
    Xv_fwls = _build_fwls_features(
        _stack_val, aux_val_mat_raw, aux_val_mat_norm,
    )
    print(f"\n  [fwls] feature dim = {Xo_fwls.shape[1]}  "
          f"(M={_M}, G={_G}, M*G={_M * _G})")
    stk_fwls = fit_stacker(
        X=Xo_fwls, y=y_train, feature_names=_fwls_feat_names,
        seed=SEED + 22, n_iters=3000, l2=2.0, early_stopping_patience=300,
    )
    _val_fwls = stacker_apply_batch(stk_fwls, Xv_fwls).astype(np.float32)
    nll_fwls = _nll(_val_fwls, y_val)
    print(f"  fwls_stacker                      : {nll_fwls:.6f}   "
          f"(vs full+MoE: {nll_fwls - _full_routed_nll:+.6f})")
    _w_fwls = np.asarray(stk_fwls.weights, dtype=np.float32).reshape(-1)
    print("  FWLS main-effect weights (per member, no aux gate):")
    for i, v in enumerate(VARIANT_KEYS):
        print(f"    {v:<28s} main={_w_fwls[i]:+.4f}")
    print("  FWLS gated-weight effects (member * aux):")
    for gi, g in enumerate(_AUX_NAMES):
        for vi, v in enumerate(VARIANT_KEYS):
            idx = _M + gi * _M + vi
            print(f"    {v:<28s} x {g:<24s} gate={_w_fwls[idx]:+.4f}")

# 18d. Small MLP combiner (PyTorch).
#
# Tiny 2-layer net: input [logits | aux_normalized] -> hidden ->
# sigmoid. Trains on OOF predictions with an internal val split for
# early stopping. Kept small (hidden=16) so 600K rows x 10 inputs
# can't overfit it.
if EXP["combiner_small_mlp"]:
    import torch.nn as nn

    class TinyCombiner(nn.Module):
        def __init__(self, n_in, hidden, p_drop):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.GELU(),
                nn.Dropout(float(p_drop)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(p_drop)),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def _mlp_train_predict():
        # Build [N, M + G] feature matrices.
        Xo = np.concatenate([
            np.asarray(logit_clipped(np.asarray(_stack_oof, dtype=np.float64))).astype(np.float32),
            aux_train_mat_norm,
        ], axis=1)
        Xv = np.concatenate([
            np.asarray(logit_clipped(np.asarray(_stack_val, dtype=np.float64))).astype(np.float32),
            aux_val_mat_norm,
        ], axis=1)
        Xo = np.where(np.isfinite(Xo), Xo, 0.0).astype(np.float32)
        Xv = np.where(np.isfinite(Xv), Xv, 0.0).astype(np.float32)

        dev = _dev
        rng = np.random.default_rng(SEED + 33)
        N = Xo.shape[0]
        n_val = max(2000, int(round(float(EXP["mlp_val_fraction"]) * N)))
        perm = rng.permutation(N)
        vidx = perm[:n_val]
        tidx = perm[n_val:]

        net = TinyCombiner(
            n_in=Xo.shape[1], hidden=int(EXP["mlp_hidden"]),
            p_drop=float(EXP["mlp_dropout"]),
        ).to(dev)
        opt = torch.optim.AdamW(
            net.parameters(), lr=float(EXP["mlp_lr"]),
            weight_decay=float(EXP["mlp_wd"]),
        )
        bce = nn.BCEWithLogitsLoss()
        Xo_t = torch.from_numpy(Xo).to(dev)
        y_t = torch.from_numpy(y_train).to(dev)

        bs = 65536
        best_val = float("inf")
        best_state = None
        no_imp = 0
        for epoch in range(int(EXP["mlp_epochs"])):
            net.train()
            perm_tr = rng.permutation(tidx)
            for i in range(0, len(perm_tr), bs):
                idx = perm_tr[i:i + bs]
                logits = net(Xo_t[idx])
                loss = bce(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v_logits = net(Xo_t[vidx])
                vl = bce(v_logits, y_t[vidx]).item()
            if vl + 1e-5 < best_val:
                best_val = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= int(EXP["mlp_patience"]):
                    print(f"    [mlp] early stop at epoch {epoch + 1}; "
                          f"best val BCE={best_val:.5f}")
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

        # Predict val.
        net.eval()
        Xv_t = torch.from_numpy(Xv).to(dev)
        with torch.no_grad():
            p_val_logits = torch.zeros(Xv.shape[0], dtype=torch.float32, device=dev)
            for i in range(0, Xv.shape[0], bs):
                end = min(i + bs, Xv.shape[0])
                p_val_logits[i:end] = net(Xv_t[i:end])
            p_val = torch.sigmoid(p_val_logits).cpu().numpy().astype(np.float32)
        return p_val

    print("\n  [mlp] training tiny combiner...")
    _val_mlp = _mlp_train_predict()
    nll_mlp = _nll(_val_mlp, y_val)
    print(f"  small_mlp_combiner                : {nll_mlp:.6f}   "
          f"(vs full+MoE: {nll_mlp - _full_routed_nll:+.6f})")

# %% [markdown]
# ## 19. Comparison table -- reference NLLs from the prior probes
#
# We don't re-train the plain-vs-rich or the single-variant
# rich+MoE baselines here (they're already in the prior probe
# logs). Embedded as constants below for the apples-to-apples table.
# The numbers come from this same dataset / fold split / val
# redaction sample, so the comparison is honest.

# %%
# Reference NLLs (from prior runs of the same dataset / val redaction):
REF_PLAIN = 0.509200            # plain-M8 baseline
REF_RICH_NO_MOE = 0.462682       # rich_baseline (no MoE) from rich_mlp_moe_probe
REF_RICH_NN_MOE = 0.458928       # rich + NN-support soft-MoE tau=1.0xstd (best from prior probe)
REF_FD_UNIFORM = 0.463049        # feature-dropout uniform_avg from feature_dropout_ensemble
REF_FD_PLATT = 0.461719          # feature-dropout platt_stacker from feature_dropout_ensemble
REF_FD_BEAT4 = 0.461018          # feature-dropout uniform_avg over 4 'beat-full' variants

print("\n" + "=" * 80)
print("COMBINED COMPARISON -- val NLL (lower is better)")
print("=" * 80)


def _row(name, nll, ref_name, ref_nll):
    delta = nll - ref_nll
    return f"  {name:<48s}: {nll:.6f}   (vs {ref_name}: {delta:+.6f})"


print(_row("plain_baseline (PRIOR)", REF_PLAIN, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("rich_baseline NO MoE (PRIOR ref)", REF_RICH_NO_MOE, "self", REF_RICH_NO_MOE))
print(_row("rich + NN-support soft-MoE (PRIOR)", REF_RICH_NN_MOE, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("feature-dropout uniform_avg (PRIOR)", REF_FD_UNIFORM, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("feature-dropout platt_stacker (PRIOR)", REF_FD_PLATT, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("feature-dropout 4-beat-full uniform (PRIOR)", REF_FD_BEAT4, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("full + MoE (THIS RUN; per-variant anchor)", _full_routed_nll, "rich_no_moe", REF_RICH_NO_MOE))
print()
if EXP["combiner_uniform_avg"]:
    print(_row("THIS: variant-MoE uniform_avg", nll_uniform, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_logit_linear_stacker"]:
    print(_row("THIS: variant-MoE logit_linear_stacker", nll_ll, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_fwls_stacker"]:
    print(_row("THIS: variant-MoE fwls_stacker", nll_fwls, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_small_mlp"]:
    print(_row("THIS: variant-MoE small_mlp", nll_mlp, "rich_no_moe", REF_RICH_NO_MOE))

# %% [markdown]
# ## 20. Residual correlation heatmap (across variant-MoE preds)

# %%
import matplotlib.pyplot as plt


def _render_heatmap(corr, names, title, fname,
                    cmap_high=0.6, fig_w=10.0, fig_h=8.0):
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


# Residual vs 'full + MoE' anchor. The 'full' column carries the
# raw (prob - y) anchor (otherwise it would be all zeros and break
# corrcoef on that column).
_resid_names = list(VARIANT_KEYS)
_p_full = cal_val["full"]
_resid_stack = np.stack(
    [(cal_val[v] - _p_full) for v in _resid_names], axis=1,
)
_resid_stack[:, _resid_names.index("full")] = (
    _p_full - y_val
).astype(np.float32)
corr = np.corrcoef(_resid_stack.T)
_K = corr.shape[0]
_off = ~np.eye(_K, dtype=bool)
_mean_abs_off = float(np.abs(corr[_off]).mean())
print(
    f"\n[diversity] mean |residual-vs-full corr| across variant-MoE pairs = "
    f"{_mean_abs_off:.4f} (lower => more independent errors)"
)
_render_heatmap(
    corr, _resid_names,
    title=(
        "Variant-MoE residual correlation (val) -- after NN-support soft-MoE\n"
        f"K+1={len(VARIANT_KEYS)} variants; 'full' column = raw (prob - y) anchor"
    ),
    fname=f"feature_dropout_moe_combiner_K{len(VARIANT_KEYS)}_resid_corr.png",
)

# %% [markdown]
# ## 21. Final verdict
#
# Decides which combiner is the right one given the lifts vs the
# reference baselines. Thresholds match the prior probes:
#
# * ``>= 0.003 nat`` over the best prior reference: clear win,
#   promote.
# * ``>= 0.001 nat``: marginal, worth one more validation run with
#   a different seed before promoting.
# * ``< 0.001 nat``: not worth the extra compute / inference
#   complexity.

# %%
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Best prior reference is "rich + NN-support soft-MoE" at 0.458928.
# Anything better than that justifies the (K + 1) x M trainings.
_BEST_PRIOR = REF_RICH_NN_MOE
print(f"  Best prior reference: rich + NN-support soft-MoE = {_BEST_PRIOR:.6f}")
print(f"  Anchor in this run  : full + NN-support soft-MoE = {_full_routed_nll:.6f}")
print(
    f"  Drift between runs  : {_full_routed_nll - _BEST_PRIOR:+.6f} "
    "(should be ~0 if seeds + data match)"
)

_candidates = []
if EXP["combiner_uniform_avg"]:
    _candidates.append(("uniform_avg", nll_uniform))
if EXP["combiner_logit_linear_stacker"]:
    _candidates.append(("logit_linear_stacker", nll_ll))
if EXP["combiner_fwls_stacker"]:
    _candidates.append(("fwls_stacker", nll_fwls))
if EXP["combiner_small_mlp"]:
    _candidates.append(("small_mlp", nll_mlp))

print("\n  Combiner deltas vs BEST PRIOR reference (rich + NN-support soft-MoE):")
for name, nll in _candidates:
    d = nll - _BEST_PRIOR
    if d <= -0.003:
        verdict = "CLEAR WIN -- promote this combiner over single-variant MoE"
    elif d <= -0.001:
        verdict = "marginal win -- re-validate at another seed before promoting"
    elif d <= 0.001:
        verdict = "tie within noise -- the gating couldn't extract extra signal"
    else:
        verdict = "LOSS -- combiner adds overhead without lift"
    print(f"    {name:<24s}: {nll:.6f}   ({d:+.6f})   -> {verdict}")

print(
    f"\n  Cross-variant residual diversity (mean |corr|): {_mean_abs_off:.4f}"
)
if _mean_abs_off < 0.3:
    print("    -> Variant-MoE preds remain strongly decorrelated even after "
          "individual MoE routing. Per-row gating SHOULD pay if a combiner can "
          "capture the orthogonality.")
elif _mean_abs_off < 0.6:
    print("    -> Moderate diversity. Gating helps but lift is bounded by "
          "the residual correlation floor.")
else:
    print("    -> High residual correlation. The MoE step homogenised the "
          "variants; the feature-dropout diversity has largely been "
          "consumed by the routing.")

print(
    "\n  Reading guide:\n"
    "  * If FWLS or MLP beats logit_linear_stacker by >= 0.001 nat,\n"
    "    per-row gating is the lever and the orthogonal-residuals\n"
    "    story is real. Promote that combiner as the L1 stack.\n"
    "  * If logit_linear ties FWLS / MLP, a single-weight-per-member\n"
    "    blend is enough; the residual orthogonality didn't survive\n"
    "    the per-variant MoE step.\n"
    "  * If no combiner beats `full + NN-support soft-MoE`, the\n"
    "    per-variant MoE trainings are sunk cost and we should fall\n"
    "    back to (a) single rich-M8 + NN-support MoE, or (b) a\n"
    "    `lean_rich_M8` (no form / bench / cluster) + NN-support MoE\n"
    "    informed by the prior probe's solo-NLL surprise."
)
