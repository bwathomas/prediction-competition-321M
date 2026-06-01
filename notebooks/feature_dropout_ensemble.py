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
# # Feature-dropout ensemble of Rich-M8
#
# **The question this notebook answers.** We can already train one
# rich-M8 ("the full model") that consumes every available feature
# class. If we instead train one *additional* rich-M8 per "strictly
# orthogonal" feature class -- each missing one of those classes
# (cat ids set to UNK, dense subblock zeroed) -- does the
# **K + 1 model ensemble** (full + K leave-one-out variants) beat
# the full model alone?
#
# **Why we expect this might work.** Forcing each variant to discard
# one orthogonal channel pushes that variant's residual error pattern
# in a different direction than the full model. The residuals
# decorrelate, and a stacker / uniform average extracts a bit of free
# variance reduction. The previous MoE probe showed that ~60% of the
# MoE lift was *just diversity* (scrambled routing beat the
# baseline). Feature-dropout is a more interpretable way to inject
# that diversity.
#
# **Strictly orthogonal feature classes** (5; the rich net has more
# inputs, but these are the disjoint families):
#
# 1. **subject_channel** -- ``subject_id`` + ``family`` + ``macro`` +
#    ``organization`` + subject numerics (``log_params``,
#    ``release_date``). All five derive from subject identity.
# 2. **bench_channel** -- ``bc_id`` + ``topic`` + ``benchmark_age``.
#    All three derive from benchmark identity.
# 3. **form_block** -- the POOL + ITEM_TYPE + cot-interaction dense
#    features (the "what does this prompt look like" channel).
# 4. **nn_block** -- the 15-cell NN-features dense block (the "what
#    do similar items look like" channel).
# 5. **cluster_channel** -- ``cluster_id`` (cat) + the 8 centroid-
#    distance dense columns (the "where in the embedding atlas does
#    this item live" channel).
#
# Item embedding is **always on**; it's the substrate every variant
# rests on, not an auxiliary feature class. Drop it and the model
# collapses to "rich-features-only" which isn't comparable.
#
# **Order of operations -- IF you want to combine this with NN-support MoE.**
#
# These two diversity schemes are mathematically equivalent in their
# final model count: ``(K + 1) × M`` (where ``K`` = feature classes,
# ``M`` = MoE buckets) either way. So order doesn't change *what* the
# final ensemble computes if both stages use uniform averaging (and
# soft routing is just a per-row weighted average, which commutes
# with averaging over variants).
#
# What order DOES affect:
#
# 1. **Operational cost** -- doing feature-dropout *first*, as a
#    standalone experiment, lets you stop early if it doesn't pay.
#    The MoE probe already showed ~60% of MoE lift is diversity;
#    feature-dropout MIGHT capture that diversity for less compute.
#    So test feature-dropout first as a cheap proxy.
# 2. **Routing geometry** -- if you do MoE *first*, each bucket
#    expert has a known specialization (high vs low support). If
#    you then drop a feature class from each bucket expert, the
#    bucket's specialty might collapse (e.g. dropping NN from the
#    "high NN-support" expert is self-defeating). Doing feature-
#    dropout *first* and routing *second* keeps the per-variant
#    feature regime stable across buckets.
# 3. **Inference ergonomics** -- feature-dropout-outer / MoE-inner
#    means production prediction is "for each variant, route to its
#    bucket, average the K+1 routed preds". MoE-outer / dropout-
#    inner means "route once, then average K+1 variants in that
#    bucket". The first is more interpretable; the second has
#    slightly cheaper inference (one routing decision).
#
# **Recommendation:** run feature-dropout FIRST as a standalone
# probe. If it gives ≥0.002 nat lift over the full rich_baseline,
# consider adding MoE on top (feature-dropout-outer, MoE-inner).
# This notebook executes the standalone test.
#
# **What this notebook shares with `rich_mlp_moe_probe.py`.** Same
# data, same item-cold OOF folds, same per-unit UNK redaction at
# train AND val time, same item embedding cache, same metadata
# preprocessor, same rich net architecture. The only divergence is
# the per-variant feature masking + the ensemble combination.

# %% [markdown]
# ## 0. Colab bootstrap (clone repo, mount Drive, set cwd)

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
    # ---- Shared data / compute budget (same as rich_mlp_moe_probe) --------
    "max_train_rows": 600_000,
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,
    "emb_device": None,
    "predict_chunk": 131_072,
    "use_metadata": True,

    # ---- Rich-M8 arch knobs (same as rich_mlp_moe_probe) ------------------
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
    # Per-unit UNK dropout at training time. Matches the production
    # ~20% bc-redact rate and small subject-side dropout.
    "rich_cat_dropout_subject": 0.05,
    "rich_cat_dropout_bc": 0.20,
    "rich_cat_dropout_cluster": 0.10,
    "rich_cat_dropout_family": 0.05,
    "rich_cat_dropout_macro": 0.05,
    "rich_cat_dropout_org": 0.05,
    "rich_cat_dropout_topic": 0.10,

    # ---- Feature blocks ----------------------------------------------------
    "use_nn_block": True,
    "use_subject_numeric": True,
    "use_bench_numeric": True,
    "use_cluster_block": True,
    "n_clusters": 8,
    "kmeans_iters": 50,
    "kmeans_seed": 23,

    # ---- Feature-dropout ensemble knobs ------------------------------------
    # Each variant trains one rich-M8 with the named feature class
    # MASKED (cat ids -> UNK, dense subblock -> 0). The "full" variant
    # has nothing masked.
    "variants": (
        "full",
        "no_subj_channel",
        "no_bench_channel",
        "no_form_block",
        "no_nn_block",
        "no_cluster_channel",
    ),
    # Ensemble combination methods to evaluate.
    "ensemble_uniform_avg": True,
    "ensemble_platt_stacker": True,

    "seed": SEED,
}

print("Experiment config (feature-dropout ensemble):")
for k, v in EXP.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Load data + item-cold split (item-based sub-sample)
#
# Identical preparation to `rich_mlp_moe_probe.py` so the comparison is
# apples-to-apples.

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
print("Cache decision:", drive_status.reason,
      "(hit:", bool(drive_status.cache_hit), ")")

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
# ## 6. Dense item-form block (POOL + ITEM_TYPE + cot interactions)

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
print(
    f"[meta] schema: subject_cat={list(_meta_schema.subject_categorical)}  "
    f"subject_num={list(_meta_schema.subject_numeric)}  "
    f"bench_cat={list(_meta_schema.benchmark_categorical)}  "
    f"bench_num={list(_meta_schema.benchmark_numeric)}"
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

_FAMILY_COL = (
    _SUBJ_CAT_FIELDS.index("family") if "family" in _SUBJ_CAT_FIELDS else -1
)
_MACRO_COL = (
    _SUBJ_CAT_FIELDS.index("macro_family") if "macro_family" in _SUBJ_CAT_FIELDS else -1
)
_ORG_COL = (
    _SUBJ_CAT_FIELDS.index("organization") if "organization" in _SUBJ_CAT_FIELDS else -1
)
_TOPIC_COL = (
    _BENCH_CAT_FIELDS.index("topic") if "topic" in _BENCH_CAT_FIELDS else -1
)


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
    NN_DIR = ROOT / nn_cfg.cache_dir / "training_featdrop"
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
    print(
        f"[cluster] cluster id range: train={cluster_train.min()}..{cluster_train.max()}, "
        f"val={cluster_val.min()}..{cluster_val.max()}"
    )
else:
    K_CLUSTERS = 1
    cluster_train = np.zeros(N_TRAIN, dtype=np.int64)
    cluster_val = np.zeros(N_VAL, dtype=np.int64)
    _centroid_dist_train = np.zeros((N_TRAIN, 0), dtype=np.float32)
    _centroid_dist_val = np.zeros((N_VAL, 0), dtype=np.float32)
    CENTROID_DIST_DIM = 0

# %% [markdown]
# ## 9b. Val-side metadata redaction (item-grouped + subject-grouped)
#
# Same per-unit redaction sample as `rich_mlp_moe_probe.py` so the
# reported NLL reflects the production test-time redaction distribution
# rather than full-metadata best-case. The redaction is FIXED (one
# draw at this cell's seed) so every variant scores the same val
# distribution and ensemble comparisons stay paired.

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
print(f"[val redact] applied (item-grouped + subject-grouped sample at seed={SEED + 2026})")

# %% [markdown]
# ## 10. Rich dense block + COLUMN OFFSET tracking
#
# Same rich dense matrix as the probe (form + subj_num + bench_num + nn
# + centroid_dist), but here we ALSO remember the column-offset of each
# subblock so the feature-dropout masking can zero only the right slice
# while leaving the other dense channels intact.

# %%
_rich_parts_train = [_form_train]
_rich_parts_val = [_form_val]
_blocks: list[tuple[str, int, int]] = []  # (name, start_col, width)
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
    f"[dense] rich dense width = {RICH_DENSE_DIM};  block offsets: "
    + ", ".join(f"{n}:[{s}:{s + w})" for (n, s, w) in _blocks)
)
del _rich_train_raw, _rich_val_raw

# %% [markdown]
# ## 11. Feature-class specs and per-variant masking
#
# Each variant has a ``FeatureSpec`` listing which (cat field, dense
# block) pieces to mask. The trainer/predictor still receives all
# input slots -- masked cat ids go to UNK (= cardinality), masked
# dense columns go to 0 (the post-z-score mean). The "full" variant
# masks nothing.
#
# **Crucially**, the masking is applied to BOTH train AND val arrays
# so the model never sees the feature class it's been asked to ignore.

# %%
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature classes are MASKED in this variant."""
    drop_subject_channel: bool = False
    drop_bench_channel: bool = False
    drop_form_block: bool = False
    drop_nn_block: bool = False
    drop_cluster_channel: bool = False
    # If True, the always-on item embedding lookup is replaced by the
    # all-zero UNK row. Off by default -- dropping the item emb makes
    # the variant a wildly different model.
    drop_item_embedding: bool = False


VARIANT_SPECS: dict[str, FeatureSpec] = {
    "full":                FeatureSpec(),
    "no_subj_channel":     FeatureSpec(drop_subject_channel=True),
    "no_bench_channel":    FeatureSpec(drop_bench_channel=True),
    "no_form_block":       FeatureSpec(drop_form_block=True),
    "no_nn_block":         FeatureSpec(drop_nn_block=True),
    "no_cluster_channel":  FeatureSpec(drop_cluster_channel=True),
}
VARIANT_KEYS = [v for v in EXP["variants"] if v in VARIANT_SPECS]
assert "full" in VARIANT_KEYS, "the 'full' variant is the comparison anchor"
print(f"[variants] active: {VARIANT_KEYS}")


def _apply_dense_mask(dense_mat: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Return a copy of ``dense_mat`` with the masked block columns zeroed."""
    out = dense_mat.copy() if (
        spec.drop_form_block or spec.drop_subject_channel or
        spec.drop_bench_channel or spec.drop_nn_block or
        spec.drop_cluster_channel
    ) else dense_mat
    if out is dense_mat:
        return out
    if spec.drop_form_block and "form" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["form"]
        out[:, s:s + w] = 0.0
    if spec.drop_subject_channel and "subj_num" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["subj_num"]
        out[:, s:s + w] = 0.0
    if spec.drop_bench_channel and "bench_num" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["bench_num"]
        out[:, s:s + w] = 0.0
    if spec.drop_nn_block and "nn" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["nn"]
        out[:, s:s + w] = 0.0
    if spec.drop_cluster_channel and "centroid_dist" in DENSE_BLOCK_OFFSETS:
        s, w = DENSE_BLOCK_OFFSETS["centroid_dist"]
        out[:, s:s + w] = 0.0
    return out


def _apply_cat_mask(spec: FeatureSpec, *, side: str):
    """Return tuple (subj, bc, cluster, family, macro, org, topic) for either
    side='train' or side='val'. For val we mask on top of the REDACTED val
    arrays so the "full" variant ALSO sees realistic redaction."""
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

# %% [markdown]
# ## 12. Trainer + OOF runner (rich-only, per-variant feature masking)
#
# Same `train_rich_mlp` / `predict_rich_mlp` as the probe. The per-
# variant feature masking is applied to the id arrays + dense matrix
# BEFORE the trainer sees them, so the trainer's UNK-dropout layer
# doesn't double-mask -- it just learns on a model whose feature
# subblock is already zeroed.

# %%
import torch

from src.rich_mlp_variant import (
    RichMLPConfig, predict_rich_mlp, train_rich_mlp,
)

_dev = EXP["emb_device"] or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_dev}")
EMB_T = torch.from_numpy(ALL_UNIQ).to(_dev)
_EMB_T_ZERO = torch.zeros_like(EMB_T)  # for drop_item_embedding variants


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


def _run_variant_oof(variant_name: str, spec: FeatureSpec):
    """OOF-train the named feature-dropout variant; return (oof, val) preds."""
    print(f"\n=== variant: {variant_name} ===  spec={spec}")
    # Build the masked id arrays + dense matrices ONCE per variant
    # (cheap relative to training; saves per-fold rebuild).
    s_tr, b_tr, c_tr, f_tr, mf_tr, o_tr, t_tr = _apply_cat_mask(spec, side="train")
    s_va, b_va, c_va, f_va, mf_va, o_va, t_va = _apply_cat_mask(spec, side="val")
    dense_tr = _apply_dense_mask(rich_dense_train, spec)
    dense_va = _apply_dense_mask(rich_dense_val, spec)
    emb_for_variant = _EMB_T_ZERO if spec.drop_item_embedding else EMB_T

    acc = OofPredictionAccumulator(N_TRAIN, name=f"oof_{variant_name}")
    val_stack = []
    for fold in folds:
        tr, oof = fold.train_row_idx, fold.oof_row_idx
        print(f"  [fold {fold.fold_id}] train={len(tr):,} oof={len(oof):,}")
        cfg = _rich_cfg(SEED + int(fold.fold_id))
        net = train_rich_mlp(
            y=y_train[tr],
            subject_ids=s_tr[tr], bc_ids=b_tr[tr],
            cluster_ids=c_tr[tr], family_ids=f_tr[tr],
            macro_ids=mf_tr[tr], org_ids=o_tr[tr], topic_ids=t_tr[tr],
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_train[tr],
            dense_X=(dense_tr[tr] if EXP["use_metadata"] else None),
            n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
            n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
            n_topics=_RICH_NT,
            cfg=cfg, device=_dev, show_progress=False,
        )
        # OOF predictions use ORIGINAL (un-redacted) train-side ids
        # after applying the per-variant feature-class mask. The
        # variant masking is what we want present; we do NOT layer
        # the val-style redaction sample on train OOF rows.
        s_tr_oof, b_tr_oof, c_tr_oof = s_tr[oof], b_tr[oof], c_tr[oof]
        f_tr_oof, mf_tr_oof, o_tr_oof, t_tr_oof = (
            f_tr[oof], mf_tr[oof], o_tr[oof], t_tr[oof],
        )
        oof_p = predict_rich_mlp(
            net,
            subject_ids=s_tr_oof, bc_ids=b_tr_oof,
            cluster_ids=c_tr_oof, family_ids=f_tr_oof,
            macro_ids=mf_tr_oof, org_ids=o_tr_oof, topic_ids=t_tr_oof,
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_train[oof],
            dense_X=(dense_tr[oof] if EXP["use_metadata"] else None),
            n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
            n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
            n_topics=_RICH_NT, device=_dev, chunk=int(EXP["predict_chunk"]),
        )
        # Val predictions use the per-unit redacted val ids THEN the
        # variant mask (already composed above into s_va/...).
        val_p = predict_rich_mlp(
            net,
            subject_ids=s_va, bc_ids=b_va, cluster_ids=c_va,
            family_ids=f_va, macro_ids=mf_va, org_ids=o_va, topic_ids=t_va,
            item_emb_tensor=emb_for_variant, row_to_uniq=r2u_val,
            dense_X=(dense_va if EXP["use_metadata"] else None),
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
# ## 13. Train all variants

# %%
oof_preds: dict[str, np.ndarray] = {}
val_preds: dict[str, np.ndarray] = {}
for vname in VARIANT_KEYS:
    oof_preds[vname], val_preds[vname] = _run_variant_oof(
        vname, VARIANT_SPECS[vname],
    )

# %% [markdown]
# ## 14. Per-variant Platt calibration (apples-to-apples NLL)

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


cal_val: dict[str, np.ndarray] = {
    n: _platt(oof_preds[n], val_preds[n]) for n in oof_preds
}
cal_oof: dict[str, np.ndarray] = {
    n: _platt(oof_preds[n], oof_preds[n]) for n in oof_preds
}

print("\nCalibrated solo val log-loss (lower is better):")
_solo = {n: _nll(cal_val[n], y_val) for n in cal_val}
_full_nll = _solo["full"]
print(f"  {'full (anchor)':<28s}: {_full_nll:.6f}   (reference; rich-baseline with everything on)")
for n in VARIANT_KEYS:
    if n == "full":
        continue
    d = _solo[n] - _full_nll
    print(f"  {n:<28s}: {_solo[n]:.6f}   (vs full: {d:+.6f})")

# %% [markdown]
# ## 15. Ensemble combinations
#
# Two combinations:
#
# 1. **Uniform average** -- simplest possible ensemble; averages the
#    calibrated val probabilities across all K+1 variants. Equivalent
#    to a model-averaging Bayesian-style combine where every variant
#    is weighted equally.
# 2. **Platt-stacked logistic blend** -- fit a logistic regression on
#    the K+1 calibrated OOF probabilities to learn variant weights.
#    Reuses `src.stacker.fit_stacker` so the blend uses the same
#    L-BFGS / early-stop machinery as production.

# %%
print("\n" + "=" * 80)
print("ENSEMBLE COMBINATIONS")
print("=" * 80)

# 1. Uniform average over all variants.
if EXP["ensemble_uniform_avg"]:
    _uniform_val = np.mean(
        [cal_val[n] for n in VARIANT_KEYS], axis=0
    ).astype(np.float32)
    nll_uniform = _nll(_uniform_val, y_val)
    print(
        f"  uniform_avg over {len(VARIANT_KEYS)} variants : "
        f"{nll_uniform:.6f}   (vs full: {nll_uniform - _full_nll:+.6f})"
    )

# 2. Platt-stacked logistic blend.
if EXP["ensemble_platt_stacker"]:
    _M = len(VARIANT_KEYS)
    _stack_oof = np.stack([cal_oof[n] for n in VARIANT_KEYS], axis=1).astype(np.float32)
    _stack_val = np.stack([cal_val[n] for n in VARIANT_KEYS], axis=1).astype(np.float32)
    Xo_stack = build_stacker_features(
        member_probs=_stack_oof, bench_present=_zeros_tr,
        nn_neighbor_support=_zeros_tr, nn_mean_similarity=_zeros_tr,
        centroid_distance=_half_tr,
    )
    Xv_stack = build_stacker_features(
        member_probs=_stack_val, bench_present=_zeros_va,
        nn_neighbor_support=_zeros_va, nn_mean_similarity=_zeros_va,
        centroid_distance=_half_va,
    )
    stk = fit_stacker(
        X=Xo_stack, y=y_train, feature_names=stacker_feature_names(_M),
        seed=SEED + 7, n_iters=1500,
    )
    _stacked_val = stacker_apply_batch(stk, Xv_stack).astype(np.float32)
    nll_stacked = _nll(_stacked_val, y_val)
    print(
        f"  platt_stacker  over {len(VARIANT_KEYS)} variants : "
        f"{nll_stacked:.6f}   (vs full: {nll_stacked - _full_nll:+.6f})"
    )
    # Print learned weights so we can see if the stacker is using
    # the dropout variants or just collapsing to the full model.
    _w = np.asarray(stk.weights, dtype=np.float32).reshape(-1)
    print(f"  stacker weights (member channels first):")
    for i, n in enumerate(VARIANT_KEYS):
        print(f"    {n:<28s} weight={_w[i]:+.4f}")

# 3. Best-N ensemble: average only the variants that beat the full
#    model on solo NLL (a cheap "skip the bad variants" heuristic).
_better_than_full = [
    n for n in VARIANT_KEYS if n != "full" and _solo[n] < _full_nll
]
if _better_than_full:
    _members = ["full"] + _better_than_full
    _best_avg = np.mean([cal_val[n] for n in _members], axis=0).astype(np.float32)
    nll_best_avg = _nll(_best_avg, y_val)
    print(
        f"  uniform_avg over {len(_members)} 'beat-full' variants "
        f"({_members}): {nll_best_avg:.6f}   "
        f"(vs full: {nll_best_avg - _full_nll:+.6f})"
    )

# %% [markdown]
# ## 16. Residual-vs-full diversity heatmap
#
# Off-diagonal correlation in residual space tells us how much each
# variant's errors *decorrelate* from the full model and from each
# other. Lower mean |corr| = more independent error signal =
# friendlier soil for any blender / stacker.

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


# 1. Residual-from-full correlation across the K leave-one-out variants
#    plus the "full" diagonal-of-self (always +1). Excludes the
#    ensemble outputs so the matrix is just per-variant.
_resid_names = list(VARIANT_KEYS)
_p_full = cal_val["full"]
_resid_stack = np.stack(
    [(cal_val[n] - _p_full) for n in _resid_names], axis=1,
)
# Replace the "full" residual column (which is all zeros) with the
# raw (prob - y) of the full model so the heatmap still has a useful
# anchor row. Otherwise corrcoef hits a divide-by-zero on that column.
_resid_stack[:, _resid_names.index("full")] = (_p_full - y_val).astype(np.float32)
corr = np.corrcoef(_resid_stack.T)
_K = corr.shape[0]
_off = ~np.eye(_K, dtype=bool)
_mean_abs_off = float(np.abs(corr[_off]).mean())
print(
    f"\n[diversity] mean |residual-vs-full corr| across variant pairs = "
    f"{_mean_abs_off:.4f} (lower => more independent errors)"
)
_render_heatmap(
    corr, _resid_names,
    title=(
        "Feature-dropout ensemble residual correlation (val)\n"
        f"K+1={len(VARIANT_KEYS)} variants; "
        f"'full' column = raw (prob - y) anchor"
    ),
    fname=f"feature_dropout_K{len(VARIANT_KEYS)}_resid_corr.png",
)

# 2. Raw-probability correlation matrix (for cross-checking; less
#    informative than residuals but easy to interpret).
_prob_stack = np.stack([cal_val[n] for n in _resid_names], axis=1)
prob_corr = np.corrcoef(_prob_stack.T)
print("\n[diversity] mean prob-corr off-diagonal = "
      f"{float(prob_corr[_off].mean()):.4f}")

# %% [markdown]
# ## 17. Final verdict
#
# We answer **three** questions:
#
# 1. Did the leave-one-out variants individually beat / match the full
#    model? (Solo NLL deltas above.)
# 2. Did **uniform averaging** of all K+1 variants beat the full
#    model? (If yes, the diversity is non-trivial.)
# 3. Did the **Platt stacker** beat both? (If yes, weighted blending
#    extracts more than uniform; if no, the stacker collapses to the
#    full model and the dropout variants add noise to the blend.)

# %%
print("\n" + "=" * 80)
print("FINAL VERDICT -- feature-dropout ensemble")
print("=" * 80)
print(f"  full (anchor) NLL                : {_full_nll:.6f}")
_best_solo = min(VARIANT_KEYS, key=lambda n: _solo[n])
print(
    f"  best SOLO variant: {_best_solo:<20s}: {_solo[_best_solo]:.6f}   "
    f"({_solo[_best_solo] - _full_nll:+.6f})"
)
if _best_solo != "full" and _solo[_best_solo] < _full_nll - 1e-4:
    print("    -> The best single variant beats 'full' on solo NLL. "
          "Investigate whether the dropped feature class is hurting "
          "the full model (overfit / collinearity)?")

if EXP["ensemble_uniform_avg"]:
    _uni_d = nll_uniform - _full_nll
    print(f"  uniform_avg (K+1)                : {nll_uniform:.6f}   ({_uni_d:+.6f})")
    if _uni_d <= -0.002:
        print("    -> Uniform average pays. The variants are decorrelated "
              "enough that simple averaging extracts >=0.002 nat.")
    elif _uni_d <= -0.0005:
        print("    -> Uniform average gives a marginal lift (<0.002 nat). "
              "May not be worth the K+1x training cost.")
    else:
        print("    -> Uniform average DOES NOT help. The leave-one-out "
              "variants are not diverse enough OR are too weak.")

if EXP["ensemble_platt_stacker"]:
    _stk_d = nll_stacked - _full_nll
    print(f"  platt_stacker (K+1)              : {nll_stacked:.6f}   ({_stk_d:+.6f})")
    if _stk_d <= -0.003 and _stk_d < (_uni_d if EXP["ensemble_uniform_avg"] else 0):
        print("    -> Stacker WINS over both 'full' and 'uniform_avg'. "
              "Weighted blending is extracting real per-row signal.")
    elif _stk_d <= -0.0005:
        print("    -> Stacker beats 'full' but the gap over uniform_avg "
              "is small; uniform_avg may be more robust.")
    else:
        print("    -> Stacker collapses to / underperforms 'full'. The "
              "leave-one-out variants add noise to the blend.")

print(
    f"\n  residual diversity (mean |corr|): {_mean_abs_off:.4f}"
)
if _mean_abs_off < 0.3:
    print("    -> Residuals are strongly decorrelated. Good substrate "
          "for ensembling.")
elif _mean_abs_off < 0.6:
    print("    -> Residuals are moderately decorrelated. Ensemble "
          "lift expected to be modest.")
else:
    print("    -> Residuals are highly correlated. Variants are doing "
          "essentially the same thing; ensemble unlikely to help.")

print(
    "\nNext-step recommendation:\n"
    "  - If uniform_avg / stacker beat 'full' by >=0.002 nat AND the\n"
    "    diversity is <0.5, promote feature-dropout ensemble as the\n"
    "    rich-M8 member; consider MoE on top only if there's still\n"
    "    visible per-bucket structure.\n"
    "  - If uniform_avg lifts <0.001 nat, the feature classes are\n"
    "    redundant with the item embedding -- skip dropout ensembling\n"
    "    and try MoE (per the prior probe) for the diversity bump.\n"
    "  - If the stacker zeroes out leave-one-out weights and keeps\n"
    "    'full' at ~1.0, the dropped channels are net-negative to\n"
    "    diversify on; consider dropping a DIFFERENT axis (item\n"
    "    embedding itself, with set drop_item_embedding=True)."
)
