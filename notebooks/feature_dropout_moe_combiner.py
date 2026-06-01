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
# # Feature-dropout x NN-support MoE x linear stacker -- SWEEP edition
#
# **The question this notebook answers.** Prior run of this notebook
# (single MoE config: upweight=5x, K=8 NN-support quartiles) showed
# the **logit-linear stacker on 6 variant-MoE preds wins at 0.45568
# val NLL** -- the first stacker in the project that actually uses
# multiple members with meaningful weights (no_subj +0.13, no_form
# +0.36, no_nn -0.09 corrective, etc.). FWLS overfit; MLP added
# nothing.
#
# Now we want to know: **was (upweight=5x, K=8) the right MoE
# hyperparameter choice?** This notebook does a 2-stage sweep:
#
# **Stage A -- cheap single-variant MoE grid** over
# (`expert_weight_multiplier`, `support_k_buckets`). Uses ONLY the
# `full` variant (no feature dropout). Reports per-cell solo NLL
# after Platt calibration of the soft-routed prediction. ~10 hours
# at the default grid (3 upweights x 3 K-buckets x 1 variant x
# avg(K)=9.3 buckets x 3 folds = 252 trainings).
#
# **Stage B -- full pipeline at winner.** Take the best (upweight*,
# K*) from Stage A, then re-run the full 6-variant feature-dropout
# pipeline at that config and report the logit-linear stacker NLL.
# ~6 hours at K=8 (6 variants x 8 buckets x 3 folds = 144
# trainings); scales linearly with K_winner.
#
# **What's the same as the prior run.** Same data, same item-cold
# OOF folds, same per-unit UNK redaction at train AND val time,
# same Qwen3 item embedding cache, same metadata preprocessor, same
# rich net architecture + per-variant feature masking, same FAISS
# per-row NN-support SCORE (only the bucket *partition* depends on
# K). Only Stage A and Stage B's outer sweep loop are new.
#
# **Combiner choice.** Per the prior result, only the logit-linear
# stacker is enabled by default (FWLS overfit; small MLP added no
# signal). Flip the knobs in EXP if you want them back.
#
# **Caveat on the sweep design.** Stage A picks the winner based on
# single-variant (`full`) solo NLL, not on the full feature-dropout
# stacker NLL. The two orderings could differ -- the stacker might
# prefer a slightly different (upweight, K). To be sure, ship the
# Stage B result only after re-running it at the runner-up (upweight,
# K) and confirming it's worse. We skip that here to keep compute
# bounded; the verdict cell flags the residual uncertainty.

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
# ## 1. Config + experiment knobs (NEW: sweep grid)

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
    # ---- Shared budget ----------------------------------------------
    "max_train_rows": 600_000,
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,
    "emb_device": None,
    "predict_chunk": 131_072,
    "use_metadata": True,

    # ---- Rich-M8 arch (matches prior two probes) ---------------------
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

    # ---- Feature blocks ---------------------------------------------
    "use_nn_block": True,
    "use_subject_numeric": True,
    "use_bench_numeric": True,
    "use_cluster_block": True,
    "n_clusters": 8,
    "kmeans_iters": 50,
    "kmeans_seed": 23,

    # ---- Feature-dropout variants -- used in Stage B (winner) -------
    # 'full' is also the sole Stage A sweep variant.
    "variants": (
        "full",
        "no_subj_channel",
        "no_bench_channel",
        "no_form_block",
        "no_nn_block",
        "no_cluster_channel",
    ),

    # ---- NN-support MoE knobs ---------------------------------------
    "support_k_neighbors": 5,
    # Single tau picked from the prior probe sweep (1.0xstd was the
    # winner). We sweep K + upweight only; if you also want to sweep
    # tau, do it inside _eval_config_at_winner after Stage B.
    "support_kernel_tau_mult": 1.0,

    # ---- SWEEP GRID (Stage A) ---------------------------------------
    # All upweights are >= the prior run's 5x (per user spec).
    # All K-bucket values are >= 4 (per user spec). The default
    # 3 x 3 = 9-cell grid trains the 'full' variant at each cell;
    # at avg(K)=9.3 buckets x 3 folds that's 252 single-variant
    # rich trainings (~10 h on A100). Shrink the tuples if compute
    # is tight; the rest of the notebook auto-adapts.
    "sweep_upweights": (5.0, 10.0, 20.0),
    "sweep_k_buckets": (4, 8, 16),
    # Which variant to sweep on in Stage A. 'full' is the canonical
    # anchor; using anything else is fine but the comparison to
    # the prior single-config result becomes harder to read.
    "sweep_stage_a_variant": "full",

    # ---- Combiner selection (Stage B) -------------------------------
    # Per the prior run: linear wins, FWLS overfit, MLP no signal.
    # Re-enable the others below if you want to retest at the
    # winning (upweight, K).
    "combiner_uniform_avg": True,
    "combiner_logit_linear_stacker": True,
    "combiner_fwls_stacker": False,
    "combiner_small_mlp": False,
    "mlp_hidden": 16,
    "mlp_dropout": 0.2,
    "mlp_lr": 5.0e-3,
    "mlp_wd": 1.0e-4,
    "mlp_epochs": 60,
    "mlp_patience": 8,
    "mlp_val_fraction": 0.10,

    "seed": SEED,
}

print("Experiment config (feature-dropout x MoE x linear stacker SWEEP):")
for k, v in EXP.items():
    print(f"  {k}: {v}")

# Quick compute estimate so the user knows what they signed up for.
_avg_k_a = float(np.mean(EXP["sweep_k_buckets"]))
_stage_a_trainings = (
    len(EXP["sweep_upweights"]) * len(EXP["sweep_k_buckets"]) * 1
    * _avg_k_a * int(EXP["n_folds"])
)
_stage_b_trainings_max = (
    len(EXP["variants"]) * max(EXP["sweep_k_buckets"]) * int(EXP["n_folds"])
)
print(
    f"\n[budget] Stage A: ~{int(_stage_a_trainings)} trainings "
    f"({len(EXP['sweep_upweights'])} upweights x {len(EXP['sweep_k_buckets'])} K-buckets "
    f"x 1 variant x avg-K={_avg_k_a:.1f} x {EXP['n_folds']} folds)"
)
print(
    f"[budget] Stage B (upper bound, K=K_max): "
    f"~{int(_stage_b_trainings_max)} trainings "
    f"({len(EXP['variants'])} variants x {max(EXP['sweep_k_buckets'])} buckets x {EXP['n_folds']} folds)"
)

# %% [markdown]
# ## 2. Load data + item-cold split

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
# ## 3. Item embeddings (cache-aware)

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
print(f"Built {len(folds)} item-cold folds")
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

# %% [markdown]
# ## 8. NN block

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
    NN_DIR = ROOT / nn_cfg.cache_dir / "training_combiner_sweep"
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
else:
    nn_train_mat = np.zeros((N_TRAIN, 0), dtype=np.float32)
    nn_val_mat = np.zeros((N_VAL, 0), dtype=np.float32)
    NN_DIM = 0

# %% [markdown]
# ## 9. Cluster block

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
# ## 11. NN-support per-row SCORE + parameterized partition factory
#
# The FAISS index + per-row support score is computed ONCE here (it's
# the same for every K). Only the bucket partition (boundaries +
# bucket assignments + Gaussian-kernel centroids) depends on K, so
# we factor that into ``_make_support_partition(k)`` and call it
# at each sweep cell.

# %%
import faiss

K_NN = int(EXP["support_k_neighbors"])

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


support_score_train = _support_score(_train_keys)
support_score_val = _support_score(_val_keys)
del _sup_idx, _sims_train, _sims_val
gc.collect()


def _make_support_partition(k: int):
    """Build the K-bucket NN-support partition.

    Returns
    -------
    boundaries : [k - 1] cut points
    bucket_train, bucket_val : [N_*] int64 bucket id per row
    centroids : [k] mean support score per bucket on TRAIN rows
    """
    boundaries = np.quantile(
        _score_train_uniq, np.linspace(0.0, 1.0, k + 1)
    )[1:-1].astype(np.float32)
    bucket_train = np.searchsorted(
        boundaries, support_score_train, side="right"
    ).astype(np.int64)
    bucket_val = np.searchsorted(
        boundaries, support_score_val, side="right"
    ).astype(np.int64)
    centroids = np.array([
        float(support_score_train[bucket_train == j].mean())
        if int((bucket_train == j).sum()) > 0
        else float(np.mean(boundaries)) if len(boundaries) > 0 else 0.0
        for j in range(k)
    ], dtype=np.float32)
    return boundaries, bucket_train, bucket_val, centroids


# Sanity print for one default-K partition so we have a reference
# bucket-size table at the start.
_b_, _bt_, _bv_, _bc_ = _make_support_partition(8)
print(
    f"[partition support k=8] cut points: "
    + ", ".join(f"{b:.4f}" for b in _b_)
)
print("  train bucket sizes (k=8):")
for j in range(8):
    n = int((_bt_ == j).sum())
    print(f"    support_q{j + 1}    train_rows={n:>9,} ({n / N_TRAIN:>5.1%})")
del _b_, _bt_, _bv_, _bc_

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
SWEEP_VARIANT = EXP["sweep_stage_a_variant"]
assert SWEEP_VARIANT in VARIANT_SPECS, (
    f"sweep_stage_a_variant={SWEEP_VARIANT} not in VARIANT_SPECS"
)


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


# Pre-build per-variant masked arrays (cheap; same across all
# (upweight, K) sweep cells). Only the variants we will actually
# train in Stage A or Stage B get cached; the others are skipped.
_needed_variants = set(VARIANT_KEYS) | {SWEEP_VARIANT}
VARIANT_ARRAYS: dict[str, dict[str, np.ndarray]] = {}
for vname in _needed_variants:
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
    f"[variants] pre-built masked id+dense arrays for {len(_needed_variants)} variants"
)

# %% [markdown]
# ## 13. Parameterized per-(variant, bucket, fold) OOF runner
#
# Takes (variant_name, bucket_id_in_partition, upweight, bucket_assignment_train,
# variant_idx_seed) and returns per-fold-averaged (oof, val) predictions
# for that single (variant, bucket) cell. The seed scheme keeps
# different (variant, fold) pairs independent while bucket-mates within
# the same (variant, fold) share a seed so per-bucket lift is purely
# from the upweighting, not from re-initialization.

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


def _run_variant_bucket_oof(
    variant_name: str, bucket_j: int, *,
    upweight: float, bucket_train: np.ndarray, variant_idx: int,
):
    """Train one rich net per fold for (variant, bucket); return (oof, val)."""
    arr = VARIANT_ARRAYS[variant_name]
    spec = VARIANT_SPECS[variant_name]
    emb_for_variant = _EMB_T_ZERO if spec.drop_item_embedding else EMB_T
    in_bucket = (bucket_train == int(bucket_j))
    sample_weights = _make_weights(in_bucket, float(upweight))

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
# ## 14. ``_run_one_moe_config(upweight, k, variants)`` -- one sweep cell
#
# Trains the full (variants x K buckets x folds) grid at a single
# (upweight, K) configuration, soft-routes per variant, Platt-
# calibrates the routed prediction, and returns:
#
# * ``cal_val[variant]``: [N_VAL] calibrated val probability
# * ``cal_oof[variant]``: [N_TRAIN] calibrated OOF probability (used
#   downstream by the Stage B stacker)
# * ``solo_nll[variant]``: float, calibrated solo val NLL
# * ``meta``: dict with bucket sizes, tau, and partition info
#
# This is the single function the sweep + the winner step both call.

# %%
from src.stacker import (
    apply_batch as stacker_apply_batch,
    build_stacker_features,
    fit_stacker,
    logit_clipped,
    stacker_feature_names,
)
import time


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


def _run_one_moe_config(*, upweight: float, k_buckets: int,
                        variants: list[str]) -> dict:
    """Train + soft-route + Platt-calibrate for one MoE configuration.

    Parameters
    ----------
    upweight : float
        Sample-weight multiplier on rows in the bucket being trained
        (the remaining rows get weight 1.0; weights are mean-normalised).
    k_buckets : int
        Number of NN-support buckets in this MoE.
    variants : list[str]
        Which variants to actually train (each variant gets ``k_buckets``
        experts trained on ``n_folds`` folds each).

    Returns
    -------
    dict with keys: 'cal_val', 'cal_oof', 'solo_nll', 'oof_grid',
    'val_grid', 'meta'.
    """
    _t0 = time.time()
    print(f"\n  ----- config(upweight={upweight}, K={k_buckets}, "
          f"variants={variants}) -----")

    # Partition (cheap; only the bucket assignments change with K).
    _bds, _bt, _bv, _cents = _make_support_partition(int(k_buckets))
    bucket_names = [f"support_q{j + 1}" for j in range(int(k_buckets))]
    print("    train bucket sizes: " + ", ".join(
        f"{name}={int((_bt == j).sum()):,}"
        for j, name in enumerate(bucket_names)
    ))

    # (variants x buckets x folds) grid.
    oof_grid: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
    val_grid: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
    n_done = 0
    n_total = len(variants) * int(k_buckets)
    for vi, vname in enumerate(variants):
        for j, bname in enumerate(bucket_names):
            n_done += 1
            n_in = int((_bt == j).sum())
            elapsed = (time.time() - _t0) / 60.0
            print(f"    [{n_done:>3d}/{n_total}] variant={vname:<22s} "
                  f"bucket={bname} ({n_in:,} bucket rows; elapsed {elapsed:.1f}m)")
            oof_p, val_p = _run_variant_bucket_oof(
                vname, j,
                upweight=float(upweight), bucket_train=_bt,
                # Stable per-variant seed offset so the same variant
                # in different sweep cells reuses the same init seed.
                variant_idx=vi,
            )
            oof_grid[vname][bname] = oof_p
            val_grid[vname][bname] = val_p

    # Per-variant soft routing.
    std_score_tr = float(np.std(support_score_train).clip(1e-6, None))
    std_score_va = float(np.std(support_score_val).clip(1e-6, None))
    tau_tr = float(EXP["support_kernel_tau_mult"]) * std_score_tr
    tau_va = float(EXP["support_kernel_tau_mult"]) * std_score_va
    w_tr = soft_routing_weights_kernel(
        support_score_train, bucket_centroids=_cents, tau=tau_tr,
    )
    w_va = soft_routing_weights_kernel(
        support_score_val, bucket_centroids=_cents, tau=tau_va,
    )

    oof_routed: dict[str, np.ndarray] = {}
    val_routed: dict[str, np.ndarray] = {}
    for vname in variants:
        oof_routed[vname] = apply_soft_routing(
            oof_grid[vname], expert_names=bucket_names, weights=w_tr,
        ).astype(np.float32)
        val_routed[vname] = apply_soft_routing(
            val_grid[vname], expert_names=bucket_names, weights=w_va,
        ).astype(np.float32)

    # Platt calibration per variant.
    cal_val = {v: _platt(oof_routed[v], val_routed[v]) for v in variants}
    cal_oof = {v: _platt(oof_routed[v], oof_routed[v]) for v in variants}
    solo_nll = {v: _nll(cal_val[v], y_val) for v in variants}

    elapsed_total = (time.time() - _t0) / 60.0
    print(f"    config done in {elapsed_total:.1f}m  solo NLLs:")
    for v in variants:
        print(f"      {v:<22s} : {solo_nll[v]:.6f}")
    return dict(
        cal_val=cal_val, cal_oof=cal_oof, solo_nll=solo_nll,
        oof_grid=oof_grid, val_grid=val_grid,
        meta=dict(
            upweight=float(upweight), k_buckets=int(k_buckets),
            bucket_names=bucket_names, bucket_centroids=_cents,
            tau_tr=tau_tr, tau_va=tau_va,
            elapsed_min=elapsed_total,
        ),
    )

# %% [markdown]
# ## 15. STAGE A -- 2D sweep over (upweight, K) with the 'full' variant only
#
# Cheap MoE-only sweep. For each cell in the
# ``EXP["sweep_upweights"]`` x ``EXP["sweep_k_buckets"]`` grid, we
# train just the ``sweep_stage_a_variant`` (default 'full') x K
# buckets x n_folds, then report the Platt-calibrated solo val NLL.
# The winner of this grid becomes the (upweight, K) used for Stage B.
#
# **Why this is the right cheap proxy.** The combiner lift in the
# prior run was -0.0021 nat on top of the single-variant MoE
# (full + MoE 0.4578 -> stacker 0.4557). A 0.001 swing in the
# single-variant NLL across sweep cells dominates that combiner lift,
# so the sweep winner on solo NLL is almost certainly also the
# winner on stacker NLL. If two cells tie within 0.001 nat on Stage A,
# re-run Stage B at the runner-up before committing.

# %%
print("\n" + "=" * 80)
print(f"STAGE A -- sweep ({SWEEP_VARIANT} only) over "
      f"upweights={EXP['sweep_upweights']}  K_buckets={EXP['sweep_k_buckets']}")
print("=" * 80)

# stage_a_grid[(upweight, k)] = solo NLL for SWEEP_VARIANT
stage_a_grid: dict[tuple[float, int], float] = {}
stage_a_meta: dict[tuple[float, int], dict] = {}
stage_a_t0 = time.time()
_cell_idx = 0
_cell_total = len(EXP["sweep_upweights"]) * len(EXP["sweep_k_buckets"])
for up in EXP["sweep_upweights"]:
    for k in EXP["sweep_k_buckets"]:
        _cell_idx += 1
        print(f"\n[STAGE A {_cell_idx}/{_cell_total}] "
              f"upweight={up}  K={k}  variant={SWEEP_VARIANT}  "
              f"total elapsed {(time.time() - stage_a_t0) / 60:.1f}m")
        res = _run_one_moe_config(
            upweight=float(up), k_buckets=int(k),
            variants=[SWEEP_VARIANT],
        )
        stage_a_grid[(float(up), int(k))] = float(res["solo_nll"][SWEEP_VARIANT])
        stage_a_meta[(float(up), int(k))] = res["meta"]

print(f"\n[STAGE A] done in {(time.time() - stage_a_t0) / 60:.1f}m")

# %% [markdown]
# ## 15b. Stage A summary -- table + heatmap

# %%
print("\n" + "-" * 80)
print(f"STAGE A grid -- solo Platt-calibrated val NLL ({SWEEP_VARIANT} variant)")
print("-" * 80)

_ups = list(EXP["sweep_upweights"])
_ks = list(EXP["sweep_k_buckets"])

# Print as a text table.
header = f"  {'upweight \\ K':<14s}" + "".join(f"{k:>14d}" for k in _ks)
print(header)
print("  " + "-" * (len(header) - 2))
_grid_arr = np.zeros((len(_ups), len(_ks)), dtype=np.float64)
for i, up in enumerate(_ups):
    row = [f"  {up:<14g}"]
    for j, k in enumerate(_ks):
        v = stage_a_grid.get((float(up), int(k)), float("nan"))
        _grid_arr[i, j] = v
        row.append(f"{v:>14.6f}")
    print("".join(row))

# Mark the best cell.
_flat_idx = int(np.nanargmin(_grid_arr))
_best_i, _best_j = np.unravel_index(_flat_idx, _grid_arr.shape)
_best_up = _ups[_best_i]
_best_k = _ks[_best_j]
_best_nll = float(_grid_arr[_best_i, _best_j])
print(f"\n[STAGE A winner] (upweight={_best_up}, K={_best_k}) "
      f"with solo NLL = {_best_nll:.6f}")

# Heatmap so the surface shape is easy to read at a glance.
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.9 * len(_ks) + 3, 0.7 * len(_ups) + 2))
    im = ax.imshow(_grid_arr, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(_ks)))
    ax.set_yticks(range(len(_ups)))
    ax.set_xticklabels([str(k) for k in _ks])
    ax.set_yticklabels([str(u) for u in _ups])
    ax.set_xlabel("K (support buckets)")
    ax.set_ylabel("upweight (multiplier)")
    for i in range(len(_ups)):
        for j in range(len(_ks)):
            v = _grid_arr[i, j]
            text = f"{v:.5f}"
            # Highlight the winner in red.
            color = "red" if (i, j) == (_best_i, _best_j) else "white"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=9,
                    fontweight=("bold" if (i, j) == (_best_i, _best_j) else "normal"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="solo NLL (lower=better)")
    ax.set_title(
        f"Stage A: solo Platt-calibrated val NLL\n"
        f"({SWEEP_VARIANT} variant; winner highlighted)"
    )
    fig.tight_layout()
    _outdir = ROOT / "artifacts" / "diagnostics"
    _outdir.mkdir(parents=True, exist_ok=True)
    _outpath = _outdir / "feature_dropout_moe_combiner_sweep_stage_a.png"
    fig.savefig(_outpath, dpi=120, bbox_inches="tight")
    print(f"[heatmap] saved -> {_outpath}")
    plt.show()
except Exception as exc:  # pragma: no cover - plotting optional
    print(f"[stage A heatmap] skipped ({exc!r})")

# Runner-up callout -- if the best cell is within 0.001 nat of the
# 2nd-best, flag it so the user knows the stacker pick is uncertain.
_sorted = sorted(stage_a_grid.items(), key=lambda kv: kv[1])
if len(_sorted) >= 2:
    _runner_up_key, _runner_up_nll = _sorted[1]
    _gap = _runner_up_nll - _best_nll
    print(
        f"\n[STAGE A runner-up] {_runner_up_key} with NLL = {_runner_up_nll:.6f}  "
        f"(+{_gap:.6f} vs winner)"
    )
    if _gap < 0.001:
        print(
            "  ! WARNING: winner / runner-up gap < 0.001 nat. The Stage B "
            "stacker NLL could prefer the runner-up. Consider re-running "
            "Stage B at the runner-up before committing to the winner."
        )

# %% [markdown]
# ## 16. STAGE B -- full feature-dropout pipeline at the (upweight, K) winner
#
# Now train all K+1 = 6 feature-dropout variants at (upweight*, K*)
# from Stage A, soft-route per variant, Platt-calibrate, then fit the
# logit-linear stacker on the 6 calibrated routed predictions + 4
# auxiliary features.
#
# Per the prior run the logit-linear stacker beats uniform_avg, FWLS,
# and a small MLP; only ``combiner_logit_linear_stacker`` is enabled
# by default. Re-enable the others in EXP if you want to re-test
# them at the new (upweight*, K*).

# %%
print("\n" + "=" * 80)
print(f"STAGE B -- full feature-dropout pipeline at WINNER "
      f"(upweight={_best_up}, K={_best_k})")
print("=" * 80)

stage_b_t0 = time.time()
stage_b_res = _run_one_moe_config(
    upweight=float(_best_up), k_buckets=int(_best_k),
    variants=list(VARIANT_KEYS),
)
print(f"\n[STAGE B] done in {(time.time() - stage_b_t0) / 60:.1f}m")

cal_val = stage_b_res["cal_val"]
cal_oof = stage_b_res["cal_oof"]
_solo = stage_b_res["solo_nll"]
_full_routed_nll = _solo["full"]

print("\nStage B per-variant solo val log-loss (each variant = its own routed MoE):")
print(f"  {'full + MoE (anchor)':<32s}: {_full_routed_nll:.6f}")
for v in VARIANT_KEYS:
    if v == "full":
        continue
    d = _solo[v] - _full_routed_nll
    print(f"  {v + ' + MoE':<32s}: {_solo[v]:.6f}   ({d:+.6f})")

# %% [markdown]
# ## 17. Auxiliary features + combiners on the Stage B routed predictions
#
# Identical aux-feature construction to the prior run so the stacker
# weights are directly comparable.

# %%
def _bench_present_for(bc_arr):
    return (np.asarray(bc_arr, dtype=np.int64) != int(_UNK_BC)).astype(np.float32)


def _min_centroid_dist(row_centroid_dist_mat):
    return row_centroid_dist_mat.min(axis=1).astype(np.float32)


aux_train = dict(
    bench_present=_bench_present_for(bc_train),
    nn_neighbor_support=support_score_train.astype(np.float32),
    nn_mean_similarity=support_score_train.astype(np.float32),
    centroid_distance=(
        _min_centroid_dist(_centroid_dist_train)
        if CENTROID_DIST_DIM > 0 else _aux_half(N_TRAIN)
    ),
)
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

# %% [markdown]
# ## 18. Combiners (linear only by default; flip EXP knobs to re-test)

# %%
print("\n" + "=" * 80)
print("STAGE B COMBINERS -- val NLL")
print("=" * 80)

_M = len(VARIANT_KEYS)
_G = len(_AUX_NAMES)
_stack_oof = np.stack([cal_oof[v] for v in VARIANT_KEYS], axis=1).astype(np.float32)
_stack_val = np.stack([cal_val[v] for v in VARIANT_KEYS], axis=1).astype(np.float32)

nll_uniform = None
nll_ll = None
nll_fwls = None
nll_mlp = None
_w_ll = None

if EXP["combiner_uniform_avg"]:
    _uniform_val = _stack_val.mean(axis=1).astype(np.float32)
    nll_uniform = _nll(_uniform_val, y_val)
    print(f"  uniform_avg                       : {nll_uniform:.6f}   "
          f"(vs full+MoE: {nll_uniform - _full_routed_nll:+.6f})")

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
    print("  weights (aux channels):")
    for i, g in enumerate(_AUX_NAMES):
        print(f"    {g:<28s} weight={_w_ll[_M + i]:+.4f}")
    print(f"  bias = {stk_ll.bias:+.4f}")

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
    Xo_fwls = _build_fwls_features(_stack_oof, aux_train_mat_raw, aux_train_mat_norm)
    Xv_fwls = _build_fwls_features(_stack_val, aux_val_mat_raw, aux_val_mat_norm)
    stk_fwls = fit_stacker(
        X=Xo_fwls, y=y_train, feature_names=_fwls_feat_names,
        seed=SEED + 22, n_iters=3000, l2=2.0, early_stopping_patience=300,
    )
    _val_fwls = stacker_apply_batch(stk_fwls, Xv_fwls).astype(np.float32)
    nll_fwls = _nll(_val_fwls, y_val)
    print(f"  fwls_stacker                      : {nll_fwls:.6f}   "
          f"(vs full+MoE: {nll_fwls - _full_routed_nll:+.6f})")

if EXP["combiner_small_mlp"]:
    import torch.nn as nn

    class TinyCombiner(nn.Module):
        def __init__(self, n_in, hidden, p_drop):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(float(p_drop)),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(float(p_drop)),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

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

    rng = np.random.default_rng(SEED + 33)
    N = Xo.shape[0]
    n_val = max(2000, int(round(float(EXP["mlp_val_fraction"]) * N)))
    perm = rng.permutation(N)
    vidx = perm[:n_val]; tidx = perm[n_val:]
    net = TinyCombiner(Xo.shape[1], int(EXP["mlp_hidden"]), float(EXP["mlp_dropout"])).to(_dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(EXP["mlp_lr"]),
                            weight_decay=float(EXP["mlp_wd"]))
    bce = nn.BCEWithLogitsLoss()
    Xo_t = torch.from_numpy(Xo).to(_dev)
    y_t = torch.from_numpy(y_train).to(_dev)
    bs = 65536
    best_val = float("inf"); best_state = None; no_imp = 0
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
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    Xv_t = torch.from_numpy(Xv).to(_dev)
    with torch.no_grad():
        p_val_logits = torch.zeros(Xv.shape[0], dtype=torch.float32, device=_dev)
        for i in range(0, Xv.shape[0], bs):
            end = min(i + bs, Xv.shape[0])
            p_val_logits[i:end] = net(Xv_t[i:end])
        _val_mlp = torch.sigmoid(p_val_logits).cpu().numpy().astype(np.float32)
    nll_mlp = _nll(_val_mlp, y_val)
    print(f"  small_mlp_combiner                : {nll_mlp:.6f}   "
          f"(vs full+MoE: {nll_mlp - _full_routed_nll:+.6f})")

# %% [markdown]
# ## 19. Combined comparison table -- prior refs + Stage A grid + Stage B winner

# %%
# References from earlier runs of the same dataset / val redaction.
REF_PLAIN = 0.509200
REF_RICH_NO_MOE = 0.462682
REF_RICH_NN_MOE = 0.458928
REF_FD_UNIFORM = 0.463049
REF_FD_PLATT = 0.461719
REF_FD_BEAT4 = 0.461018
REF_PRIOR_STACKER_at_5_8 = 0.455683  # the prior run's logit_linear winner

print("\n" + "=" * 80)
print("COMBINED COMPARISON -- val NLL (lower is better)")
print("=" * 80)


def _row(name, nll, ref_name, ref_nll):
    delta = nll - ref_nll
    return f"  {name:<48s}: {nll:.6f}   (vs {ref_name}: {delta:+.6f})"


print(_row("plain_baseline (PRIOR)", REF_PLAIN, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("rich_baseline NO MoE (PRIOR ref)", REF_RICH_NO_MOE, "self", REF_RICH_NO_MOE))
print(_row("rich + NN-support MoE @ (5, 8) (PRIOR)", REF_RICH_NN_MOE, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("feature-dropout uniform_avg (PRIOR)", REF_FD_UNIFORM, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("feature-dropout platt_stacker (PRIOR)", REF_FD_PLATT, "rich_no_moe", REF_RICH_NO_MOE))
print(_row("PRIOR variant-MoE logit_linear @ (5, 8)", REF_PRIOR_STACKER_at_5_8,
           "rich_no_moe", REF_RICH_NO_MOE))
print()
print(f"  STAGE A grid solo NLL ({SWEEP_VARIANT} only):")
for (up, k), v in sorted(stage_a_grid.items()):
    print(f"    upweight={up:<6g} K={k:<4d} : {v:.6f}")
print()
print(_row(f"STAGE B full+MoE anchor @ ({_best_up}, {_best_k})",
           _full_routed_nll, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_uniform_avg"]:
    print(_row(f"STAGE B variant-MoE uniform_avg @ ({_best_up}, {_best_k})",
               nll_uniform, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_logit_linear_stacker"]:
    print(_row(f"STAGE B variant-MoE logit_linear @ ({_best_up}, {_best_k})",
               nll_ll, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_fwls_stacker"]:
    print(_row(f"STAGE B variant-MoE fwls @ ({_best_up}, {_best_k})",
               nll_fwls, "rich_no_moe", REF_RICH_NO_MOE))
if EXP["combiner_small_mlp"]:
    print(_row(f"STAGE B variant-MoE small_mlp @ ({_best_up}, {_best_k})",
               nll_mlp, "rich_no_moe", REF_RICH_NO_MOE))

# %% [markdown]
# ## 20. Residual correlation heatmap at the Stage B winner

# %%
def _render_heatmap(corr, names, title, fname,
                    cmap_high=0.6, fig_w=10.0, fig_h=8.0):
    try:
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
    except Exception as exc:  # pragma: no cover
        print(f"[heatmap {fname}] skipped ({exc!r})")
        print(np.round(corr, 3))


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
    f"\n[diversity] STAGE B mean |residual-vs-full corr| = {_mean_abs_off:.4f}"
)
_render_heatmap(
    corr, _resid_names,
    title=(
        f"Stage B variant-MoE residual correlation (val) -- "
        f"WINNER (upweight={_best_up}, K={_best_k})\n"
        f"K+1={len(VARIANT_KEYS)} variants; 'full' column = raw (prob - y) anchor"
    ),
    fname=(f"feature_dropout_moe_combiner_sweep_K{len(VARIANT_KEYS)}"
           f"_up{_best_up}_k{_best_k}_resid_corr.png"),
)

# %% [markdown]
# ## 21. Final verdict
#
# Decides whether the (upweight*, K*) winner from Stage A + the
# logit-linear stacker beats the prior (5, 8) baseline by enough to
# promote, and surfaces any caveats (close runner-up, large drift
# between prior and current single-config anchors, etc.).

# %%
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print(f"  Stage A winner: (upweight={_best_up}, K={_best_k})  "
      f"solo NLL = {_best_nll:.6f}")
print(f"  Stage B at winner:")
print(f"    full + MoE anchor : {_full_routed_nll:.6f}")
if EXP["combiner_logit_linear_stacker"]:
    print(f"    logit_linear stk  : {nll_ll:.6f}   "
          f"(vs full+MoE: {nll_ll - _full_routed_nll:+.6f})")

print()
print(f"  Prior reference (5x, K=8) logit_linear stacker  : {REF_PRIOR_STACKER_at_5_8:.6f}")
if EXP["combiner_logit_linear_stacker"]:
    _delta_vs_prior = nll_ll - REF_PRIOR_STACKER_at_5_8
    print(f"  Stage B winner stacker - prior @ (5, 8)         : {_delta_vs_prior:+.6f}")
    if _delta_vs_prior <= -0.003:
        verdict = ("CLEAR WIN -- (upweight={uw}, K={k}) + linear stacker beats "
                   "the prior (5, 8) baseline by >= 0.003 nat. Promote this "
                   "MoE configuration.").format(uw=_best_up, k=_best_k)
    elif _delta_vs_prior <= -0.001:
        verdict = ("Marginal win -- {delta:+.6f} nat is real but small. "
                   "Re-validate with a different SEED before promoting.").format(
                   delta=_delta_vs_prior)
    elif _delta_vs_prior <= 0.001:
        verdict = ("Tie within noise -- the prior (5, 8) config was already "
                   "near-optimal. Stick with (5, 8) unless the sweep grid "
                   "is extended.")
    else:
        verdict = ("LOSS -- the prior (5, 8) baseline beats this sweep's "
                   "winner. Either the sweep grid missed the true optimum "
                   "(extend it) or run-to-run drift exceeded the lift.")
    print(f"  -> {verdict}")

print(
    f"\n  Cross-variant residual diversity (mean |corr|): {_mean_abs_off:.4f}"
)
print(
    "\nReading guide:\n"
    "  * If Stage B winner stacker beats prior @ (5, 8) by >= 0.003 nat,\n"
    "    promote the new (upweight*, K*) for the production NN-support MoE.\n"
    "  * If it ties prior @ (5, 8), 5x upweight + K=8 was the right pick.\n"
    "    Save compute and skip future sweeps in this region.\n"
    "  * If the Stage A surface is monotonically improving toward an edge\n"
    "    of the grid (e.g. K=16 is best), extend the grid in that direction\n"
    "    and re-sweep.\n"
    "  * If the Stage A surface is U-shaped (interior optimum), you've\n"
    "    found the right (upweight, K) regime.\n"
    "  * If the diversity dropped sharply from the prior 0.22 -> >0.4, the\n"
    "    new MoE config homogenised the variants and the stacker lift will\n"
    "    erode.\n"
    "  * If you want a third-party check on the Stage A winner, re-run\n"
    "    Stage B at the Stage A runner-up; if its stacker NLL is worse,\n"
    "    the winner is robust."
)
