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
# # Rich-MLP + MoE (soft routing) probe
#
# **Three nested questions** -- answered in one notebook so the
# comparisons share a fixed compute budget, the same data, and the same
# OOF folds:
#
# 1. **Does a richer M8 beat the plain M8?**
#    Plain M8 = [subject_emb | item_emb | form-dense] -> GLU MLP.
#    Rich  M8 = [subject_emb | bc_emb | cluster_emb | family_emb |
#                macro_emb | org_emb | bench_topic_emb | item_emb |
#                form-dense + subject_numeric + bench_numeric + NN block +
#                centroid-distance block] -> **DCN-V2 cross tower** +
#    GLU deep tower (fused at the head).
#    The cross tower is the **only** place a single-MLP can learn
#    non-additive interactions between the new cat embeddings and the
#    item embedding -- a plain MLP can't recover (family * cold-NN)
#    or (benchmark_topic * difficulty) crosses no matter how wide we
#    make the deep tower.
#
# 2. **With the rich M8 as the substrate, does MoE soft-routing pay --
#    on NN-support OR on family?**
#    The previous MoE POC ran hard routing on NN support quartiles with
#    plain M8. ~94% of the lift was global (focal-style sample-weight
#    trick); only ~6% was routing-specific. The diagnosed cause was a
#    fatal mismatch: NN support is a property of the *neighborhood*,
#    but plain M8 doesn't consume NN features, so the partition axis
#    was invisible to the model. The rich M8 fixes that: it sees the
#    NN block (so support is a feature), the benchmark identity, and
#    the subject metadata (so family is a feature). The MoE re-runs are
#    fair tests of whether the model can now exploit those axes.
#
# 3. **Does the rich M8 generalize when metadata is dropped?**
#    Test-time has ~20% benchmark redaction and a non-trivial fraction
#    of cold subjects with no model-info match. We bake per-field UNK-
#    dropout into training (~20% bc, ~5-10% subject / family / etc.)
#    so the model learns to degrade gracefully. The verdict cell
#    reports both the train-time (full metadata) val loss AND a
#    "metadata-redacted" val loss where we zero the bc + subject
#    embedding rows at predict time. If the gap is small we've
#    achieved generalization; if it's large we've memorized the
#    metadata channel.
#
# **Soft routing** is provided by :mod:`src.rich_mlp_variant`:
# * **Categorical (family)**: smoothed one-hot,
#   ``weight = (1 - eps)`` on own bucket, ``eps / (K - 1)`` on others.
#   ``eps=0`` -> hard routing; ``eps=(K-1)/K`` -> uniform avg.
# * **Continuous (NN support)**: Gaussian kernel,
#   ``weight_k ∝ exp(-(score_i - mu_k)^2 / tau^2)``. Adjacent buckets
#   blend, distant buckets don't. ``tau`` is auto-set to a small
#   multiple of ``std(score)``.
#
# **What this notebook is NOT.** It is a *probe*, not a production
# member: the trained nets stay in torch (no NumPy inference path
# export), per-fold checkpoints are discarded, the NN block uses
# the first 15 cells (global passrate; no per-fold leak-protection
# on the conditional cells). Use it to decide whether the rich-MLP +
# soft-MoE direction is worth promoting to the production pipeline,
# not as the final pipeline itself.

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
#
# Mirrors :mod:`notebooks.moe_poc` so the comparison versus the previous
# probe is apples-to-apples on data, embeddings, folds, and budget. The
# *new* knobs are the rich-MLP arch + the two MoE-axis blocks.

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
    # ---- Shared data / compute budget (matches moe_poc / loss probe) -------
    "max_train_rows": 600_000,
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,
    "emb_device": None,
    "predict_chunk": 131_072,
    "use_metadata": True,

    # ---- Plain-M8 (baseline) arch knobs -- identical to moe_poc -----------
    "plain_subj_emb_dim": 32,
    "plain_hid1": 256,
    "plain_hid2": 128,
    "plain_lr": 1.0e-3,
    "plain_wd": 1.0e-5,
    "plain_feat_dropout": 0.10,

    # ---- Rich-M8 arch knobs ------------------------------------------------
    # Cat embedding dims are scaled by cardinality and what the audit
    # canvas said was needed: subject (~thousands) needs 32; bc
    # (~33) needs 16; cluster (~8-16) needs 16; family (~10) gets 16;
    # macro_family (~5) gets 8; org (~10) gets 16; topic (~10) gets 16.
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
    # Per-field UNK-dropout at training time. The 20% bc rate matches
    # the production redact rate the test set ships with.
    "rich_cat_dropout_subject": 0.05,
    "rich_cat_dropout_bc": 0.20,
    "rich_cat_dropout_cluster": 0.10,
    "rich_cat_dropout_family": 0.05,
    "rich_cat_dropout_macro": 0.05,
    "rich_cat_dropout_org": 0.05,
    "rich_cat_dropout_topic": 0.10,

    # ---- Feature blocks ----------------------------------------------------
    "use_nn_block": True,           # 15 NN cells from compute_nn_features_streaming
    "use_subject_numeric": True,    # log_params + release_date (+ missing flags)
    "use_bench_numeric": True,      # benchmark_age (+ missing flag)
    "use_cluster_block": True,      # cluster_id + 8 centroid distances
    "n_clusters": 8,                # k-means on item embeddings
    "kmeans_iters": 50,
    "kmeans_seed": 23,

    # ---- MoE (per axis: family OR nn_support) ------------------------------
    # Default: run BOTH axes. Set "run_moe_*" to False to skip.
    "run_moe_family": True,
    "run_moe_nn_support": True,
    "expert_weight_multiplier": 5.0,    # same as previous probes

    # Family bucketing: top-K-1 families + "other/UNK" bucket = K_FAMILY.
    "family_k": 8,
    # NN-support bucketing: K_SUPPORT octiles on top-K mean cosine sim.
    "support_k_buckets": 8,
    "support_k_neighbors": 5,

    # Soft routing sweep. eps=0 collapses to hard routing; eps=(K-1)/K
    # collapses to uniform avg. The sweep tells us which side of that
    # spectrum the partition wants to live on.
    "soft_routing_eps_grid_family": (0.0, 0.05, 0.15, 0.35, 0.7, 0.875),
    # Tau is a multiplier on std(support_score); 0.25 = sharp routing,
    # 1.0 = soft blending, 4.0 ~= uniform.
    "soft_routing_tau_grid_support_mult": (0.25, 0.5, 1.0, 2.0, 4.0),

    "seed": SEED,
}

print("Experiment config (rich-MLP MoE probe):")
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

# Hydrate the local encoder cache from Drive BEFORE warming the in-memory
# index. Same pattern moe_poc / loss_diversity_probe use.
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
for f in folds:
    _tr_items = set(_train_keys[f.train_row_idx].tolist())
    _oo_items = set(_train_keys[f.oof_row_idx].tolist())
    if _tr_items & _oo_items:
        raise RuntimeError(f"fold {f.fold_id} has items on BOTH train and OOF sides")
print("[oof check] item-cold invariants OK.")

# %% [markdown]
# ## 6. Dense item-form block (same as moe_poc; the bottom layer of the dense channel)

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
#
# Calls :func:`src.data.prepare_metadata_artifacts` to fit the
# production metadata preprocessor on TRAIN, then gathers per-row id
# arrays (family / macro_family / org / topic) and per-row numeric
# arrays (subject log_params + release_date, benchmark age -- each
# with a missingness flag). The same plumbing M2 uses.

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

# Each preprocessor vocab includes the reserved MISSING/UNK row at idx 0.
# Cardinalities come straight from the vocab so the rich net's embedding
# tables size match the id range exactly (we then add +1 for our own
# "out-of-range" UNK row inside the rich MLP).
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
print(
    f"[meta] subject_numeric_dim={SUBJ_NUM_DIM}  "
    f"bench_numeric_dim={BENCH_NUM_DIM}"
)
print(
    f"[meta] family unique: train={len(np.unique(family_train))}  "
    f"val={len(np.unique(family_val))}"
)

# %% [markdown]
# ## 8. NN block (first 15 cells -- global passrate; no per-fold rebuild)
#
# Uses the same :mod:`src.nn_features` plumbing the production member
# graph uses, but with one global passrate table (= fold leakage on
# the subject-passrate channel is bounded; the cells the model sees
# for an OOF row are still computed by querying that row's embedding
# against the train index). The richer 23-cell variant pulls in
# benchmark-conditional and cluster-conditional cells that need the
# meta_id_tables + a refit per fold -- we leave that for the
# production pipeline.

# %%
import gc

from src.nn_features import (
    NNFeaturesConfig,
    TrainingNNIndex,
    build_passrate_table,
    compute_nn_features_streaming,
)

if EXP["use_nn_block"]:
    # Use the dataclass's `from_dict` helper because `CFG["nn_features"]`
    # also carries `query_chunk_size` (a kwarg of compute_nn_features_streaming,
    # NOT a field on the config dataclass) -- spreading the dict directly
    # into the dataclass constructor would TypeError on that key.
    nn_cfg = NNFeaturesConfig.from_dict(CFG["nn_features"])
    NN_DIR = ROOT / nn_cfg.cache_dir / "training_richprobe"
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
    print(f"[nn] passrate matrix nnz={nn_passrate_csr.nnz:,}")

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
    nn_index = None
    print("[nn] skipped (EXP['use_nn_block']=False)")

# %% [markdown]
# ## 9. Cluster block: k-means on item embeddings -> cluster_id + 8 centroid distances
#
# Fit k-means once on unique TRAIN item embeddings (so val items
# distribute by their natural distance to train clusters). Each row
# gets:
# * ``cluster_id``: integer in [0, K_CLUSTERS) for the cat embedding table.
# * 8 ``centroid_distance_k`` floats: cosine distance to each centroid.
#
# This is independent of the production cluster pipeline -- the probe
# only needs *something* to test the "cluster id + distances" channel.
# Production M3/M4 use a separate cluster system trained on a different
# embedding pool; revisit if we promote this path.

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
    # L2-normalize for stable k-means on inner-product (cosine) geometry.
    _norms = np.linalg.norm(_uniq_train_emb, axis=1, keepdims=True)
    _uniq_train_emb_n = (_uniq_train_emb / np.clip(_norms, 1e-12, None)).astype(np.float32)
    print(
        f"[cluster] fitting k-means K={K_CLUSTERS} on "
        f"{len(_uniq_train_item_keys):,} unique train items (D={ITEM_EMB_DIM})..."
    )
    km = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=4,
        max_iter=int(EXP["kmeans_iters"]),
        random_state=int(EXP["kmeans_seed"]),
    )
    km.fit(_uniq_train_emb_n)
    _centroids = km.cluster_centers_.astype(np.float32)
    _centroid_norms = np.linalg.norm(_centroids, axis=1, keepdims=True)
    _centroids_n = (_centroids / np.clip(_centroid_norms, 1e-12, None)).astype(np.float32)
    print(
        f"[cluster] inertia={km.inertia_:.4f}  "
        f"cluster sizes (train items): "
        + ", ".join(
            str(int((km.labels_ == j).sum())) for j in range(K_CLUSTERS)
        )
    )

    # Build per-item cluster_id + distance-vector lookups.
    _cluster_id_by_item: dict[str, int] = dict(
        zip(_uniq_train_item_keys, km.labels_.astype(np.int64).tolist())
    )
    # For val items we predict cluster from the trained centroids.
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
    print(f"[cluster] assigned cluster id for {len(_cluster_id_by_item):,} items")

    def _cluster_ids(item_keys) -> np.ndarray:
        # Items absent from item_emb_lookup get UNK = K_CLUSTERS.
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
                # cosine DISTANCE = 1 - cosine similarity, in [0, 2].
                out[i] = (1.0 - _centroids_n @ en.astype(np.float32)).astype(np.float32)
            else:
                out[i] = 1.0  # UNK -> midpoint of cosine distance range
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
    print("[cluster] skipped (EXP['use_cluster_block']=False)")

# %% [markdown]
# ## 9b. Val-side metadata redaction (item-grouped + subject-grouped)
#
# **Why redact val.** Test rows arrive with ~20% benchmark redaction
# and a non-trivial fraction of cold subjects (no model_info match).
# If we evaluate every fold on full metadata, the reported val NLL is
# optimistic by exactly the amount the cat embeddings memorize. The
# realistic NLL is the one where val mimics the test distribution.
#
# **Why item-grouped / subject-grouped.** Per-row redaction leaks: if
# item X has bc visible in one val row and bc masked in another, the
# model can infer bc from the item embedding pattern on the visible
# row and apply it on the masked one. We redact per UNIT (per item
# for ``bc`` / ``topic`` / ``cluster``; per subject for
# ``subject`` / ``family`` / ``macro`` / ``organization``) so every
# row of a masked item / subject is masked together. Matches the
# train-time per-unit dropout in :func:`src.rich_mlp_variant.train_rich_mlp`.
#
# The redaction sample is FIXED (one draw at this cell's seed) so
# every baseline / expert / soft-router scores the same val
# distribution and the comparisons stay paired.

# %%
# Default: val redaction probabilities mirror the training-time
# rich_cat_dropout_* knobs so the reported NLL reflects "model trained
# with this redaction rate, evaluated on the same redaction rate".
EXP.setdefault("redact_val", True)
EXP.setdefault("val_dropout_bc", EXP["rich_cat_dropout_bc"])
EXP.setdefault("val_dropout_topic", EXP["rich_cat_dropout_topic"])
EXP.setdefault("val_dropout_cluster", EXP["rich_cat_dropout_cluster"])
EXP.setdefault("val_dropout_subject", EXP["rich_cat_dropout_subject"])
EXP.setdefault("val_dropout_family", EXP["rich_cat_dropout_family"])
EXP.setdefault("val_dropout_macro", EXP["rich_cat_dropout_macro"])
EXP.setdefault("val_dropout_org", EXP["rich_cat_dropout_org"])

# UNK row ids (matching what the rich net's embedding tables expect
# from src.rich_mlp_variant; predict_rich_mlp also internally clamps
# any id >= cardinality to the UNK row, so passing these is safe).
_UNK_BC = int(indexer.n_bc)
_UNK_TOPIC = int(N_TOPICS)
_UNK_CLUSTER = int(K_CLUSTERS) if EXP["use_cluster_block"] else 0
_UNK_SUBJ = int(indexer.n_subjects)
_UNK_FAM = int(N_FAMILIES)
_UNK_MAC = int(N_MACROS)
_UNK_ORG = int(N_ORGS)


def _build_val_unit_masks(seed: int):
    """Sample per-item and per-subject Bernoulli masks for val.

    Returns ``(item_mask_*, subj_mask_*)`` boolean arrays indexed by
    item-embedding-index and subject-id respectively. Same units appear
    across many val rows; this lookup table guarantees consistency.
    """
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
    # Item-grouped redaction.
    redact_bc = item_bc[np.clip(item_per_row, 0, len(item_bc) - 1)]
    redact_topic = item_topic[np.clip(item_per_row, 0, len(item_topic) - 1)]
    redact_cluster = item_cluster[np.clip(item_per_row, 0, len(item_cluster) - 1)]
    bc_v = np.where(redact_bc, _UNK_BC, bc).astype(np.int64)
    topic_v = np.where(redact_topic, _UNK_TOPIC, topic).astype(np.int64)
    cluster_v = np.where(redact_cluster, _UNK_CLUSTER, cluster).astype(np.int64)
    # Subject-grouped redaction.
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

print(
    f"[val redact] redact_val={EXP['redact_val']}  "
    f"bc={EXP['val_dropout_bc']:.2f}  topic={EXP['val_dropout_topic']:.2f}  "
    f"cluster={EXP['val_dropout_cluster']:.2f}  subj={EXP['val_dropout_subject']:.2f}  "
    f"family={EXP['val_dropout_family']:.2f}  macro={EXP['val_dropout_macro']:.2f}  "
    f"org={EXP['val_dropout_org']:.2f}"
)
print(
    f"[val redact] redaction rates actually applied (item-grouped, subject-grouped):"
)
print(
    f"  bc:      {(bc_val_red == _UNK_BC).mean() - (bc_val == _UNK_BC).mean():.3%} of val rows masked"
)
print(
    f"  topic:   {(topic_val_red == _UNK_TOPIC).mean() - (topic_val == _UNK_TOPIC).mean():.3%} of val rows masked"
)
print(
    f"  cluster: {(cluster_val_red == _UNK_CLUSTER).mean() - (cluster_val == _UNK_CLUSTER).mean():.3%} of val rows masked"
)
print(
    f"  subj:    {(subj_val_red == _UNK_SUBJ).mean() - (subj_val == _UNK_SUBJ).mean():.3%} of val rows masked"
)
print(
    f"  family:  {(family_val_red == _UNK_FAM).mean() - (family_val == _UNK_FAM).mean():.3%} of val rows masked"
)

# %% [markdown]
# ## 10. Assemble dense block (PLAIN vs RICH) + z-score
#
# **Plain dense** (for the M8 baseline): form block only -- matches
# :mod:`notebooks.moe_poc` so the plain-vs-rich comparison is honest.
#
# **Rich dense**: form block + subject numerics + bench numerics + NN
# block + centroid distances (cat ids flow through the rich MLP's
# embedding tables, NOT through dense_X). Z-scored on TRAIN stats
# (train rows only) so val sees calibrated features.

# %%
# Plain dense (form-block only)
_plain_train_raw = _form_train.astype(np.float32)
_plain_val_raw = _form_val.astype(np.float32)
_pmean = _plain_train_raw.mean(axis=0).astype(np.float32)
_pstd = _plain_train_raw.std(axis=0).astype(np.float32)
_pstd = np.where(_pstd < 1e-6, 1.0, _pstd).astype(np.float32)
plain_dense_train = ((_plain_train_raw - _pmean) / _pstd).astype(np.float32)
plain_dense_val = ((_plain_val_raw - _pmean) / _pstd).astype(np.float32)
PLAIN_DENSE_DIM = int(plain_dense_train.shape[1])
print(f"[dense] plain dense block width = {PLAIN_DENSE_DIM}")

# Rich dense: form + subject_num + bench_num + nn + centroid_dist.
# subject_num and bench_num are already "scaled values + missing flags"
# from the preprocessor; we still z-score them on train so the rich
# scale matches everything else in the dense block.
_rich_parts_train = [_form_train]
_rich_parts_val = [_form_val]
if SUBJ_NUM_DIM > 0:
    _rich_parts_train.append(_subj_num_train)
    _rich_parts_val.append(_subj_num_val)
if BENCH_NUM_DIM > 0:
    _rich_parts_train.append(_bench_num_train)
    _rich_parts_val.append(_bench_num_val)
if NN_DIM > 0:
    _rich_parts_train.append(nn_train_mat.astype(np.float32))
    _rich_parts_val.append(nn_val_mat.astype(np.float32))
if CENTROID_DIST_DIM > 0:
    _rich_parts_train.append(_centroid_dist_train.astype(np.float32))
    _rich_parts_val.append(_centroid_dist_val.astype(np.float32))
_rich_train_raw = np.concatenate(_rich_parts_train, axis=1).astype(np.float32)
_rich_val_raw = np.concatenate(_rich_parts_val, axis=1).astype(np.float32)
_rmean = _rich_train_raw.mean(axis=0).astype(np.float32)
_rstd = _rich_train_raw.std(axis=0).astype(np.float32)
_rstd = np.where(_rstd < 1e-6, 1.0, _rstd).astype(np.float32)
rich_dense_train = ((_rich_train_raw - _rmean) / _rstd).astype(np.float32)
rich_dense_val = ((_rich_val_raw - _rmean) / _rstd).astype(np.float32)
RICH_DENSE_DIM = int(rich_dense_train.shape[1])
print(
    f"[dense] rich dense block width = {RICH_DENSE_DIM} "
    f"(form={FORM_DIM}  subj_num={SUBJ_NUM_DIM}  bench_num={BENCH_NUM_DIM}  "
    f"nn={NN_DIM}  centroid_dist={CENTROID_DIST_DIM})"
)
del _rich_train_raw, _rich_val_raw

# %% [markdown]
# ## 11. Routing partitions
#
# Both partitions are computed up front so the bucket vectors are
# available to every train / eval call below. Bucket assignments are
# fixed (train-distribution-based) so train + val see consistent
# partition geometry.

# %%
# ---- Family partition (categorical; "unknown" goes in the last bucket) -----
_FAM_K = int(EXP["family_k"])
_family_counts = pd.Series(family_train).value_counts()
# Family id 0 in the preprocessor is the reserved MISSING token -- we
# always group those into the "other/UNK" bucket, never spend a
# dedicated expert on them.
_top_family_ids = [
    int(f) for f in _family_counts.index.tolist()
    if int(f) != 0
][: _FAM_K - 1]
_family_to_bucket = {int(f): i for i, f in enumerate(_top_family_ids)}
_FAM_UNK_BUCKET = _FAM_K - 1
_FAM_BUCKET_NAMES = [
    f"fam_id{f}" for f in _top_family_ids
] + ["fam_other"]


def _family_bucket(fam_ids: np.ndarray) -> np.ndarray:
    out = np.full(int(len(fam_ids)), int(_FAM_UNK_BUCKET), dtype=np.int64)
    for f, b in _family_to_bucket.items():
        out[fam_ids == int(f)] = int(b)
    return out


bucket_family_train = _family_bucket(family_train)
# Routing decisions for val use the REDACTED family ids: a row whose
# subject is masked has family=UNK and must route to the "other/UNK"
# bucket (= what the production scenario would see). The bucket
# assignment is therefore consistent with what the rich net actually
# receives on its input.
bucket_family_val = _family_bucket(family_val_red)
_bucket_family_val_oracle = _family_bucket(family_val)  # for ablation
print(
    f"[partition family] K={_FAM_K}  "
    f"(top {_FAM_K - 1} families + 'other/UNK')"
)
print(
    f"  routing uses REDACTED family_val ({(family_val_red == _UNK_FAM).sum():,} of "
    f"{N_VAL:,} rows route to 'other/UNK' purely from redaction; "
    f"{((family_val == 0)).sum():,} were already UNK pre-redaction)."
)
print("  train bucket sizes:")
for j, name in enumerate(_FAM_BUCKET_NAMES):
    n = int((bucket_family_train == j).sum())
    print(f"    {name:<14s} train_rows={n:>9,} ({n / N_TRAIN:>5.1%})")
print("  val bucket sizes:")
for j, name in enumerate(_FAM_BUCKET_NAMES):
    n = int((bucket_family_val == j).sum())
    print(f"    {name:<14s} val_rows={n:>9,} ({n / N_VAL:>5.1%})")

# ---- NN-support partition (continuous; same FAISS code as moe_poc) --------
import faiss

_SUP_K = int(EXP["support_k_buckets"])
K_NN = int(EXP["support_k_neighbors"])

# Reuse the unique train embedding stack for the FAISS support search.
# We built _uniq_train_emb_n above (k-means normalized); reuse it if
# the cluster block was on, else recompute.
if EXP["use_cluster_block"]:
    _support_emb_uniq = _uniq_train_emb_n  # already L2-normalized
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
print("  val bucket sizes:")
for j, name in enumerate(_SUP_BUCKET_NAMES):
    n = int((bucket_support_val == j).sum())
    print(f"    {name:<14s} val_rows={n:>9,} ({n / N_VAL:>5.1%})")

# Bucket centroids on TRAIN scores -- used by the Gaussian kernel
# soft router so blending mirrors the partition geometry.
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

del _sup_idx, _support_emb_uniq, _sims_train, _sims_val
gc.collect()

# %% [markdown]
# ## 12. Trainers (plain + rich) and OOF runner
#
# Two trainers wrapped behind one ``_run_one_oof(name, kind, weights)``
# call so the baseline + experts share a single bookkeeping path.

# %%
import torch

from src.mlp_variant import predict_probs, train_m8_variant
from src.rich_mlp_variant import (
    RichMLPConfig, predict_rich_mlp, train_rich_mlp,
    soft_routing_weights_categorical, soft_routing_weights_kernel,
    apply_soft_routing,
)

_dev = EXP["emb_device"] or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_dev}")
EMB_T = torch.from_numpy(ALL_UNIQ).to(_dev)


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


# Cardinalities for the rich nets (UNK row appended automatically).
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


def _run_one_oof(name, kind, sample_weights):
    """OOF-train one variant (kind in {'plain', 'rich'}).

    Returns (oof_preds[N_TRAIN], val_preds[N_VAL]). Predictions are
    raw (uncalibrated); Platt fit happens later.
    """
    if kind not in {"plain", "rich"}:
        raise ValueError(kind)
    print(f"\n=== {kind}: {name} ===")
    acc = OofPredictionAccumulator(N_TRAIN, name=f"oof_{kind}_{name}")
    val_stack = []
    for fold in folds:
        tr, oof = fold.train_row_idx, fold.oof_row_idx
        print(f"  [fold {fold.fold_id}] train={len(tr):,} oof={len(oof):,}")
        if kind == "plain":
            net = train_m8_variant(
                y=y_train[tr], subj_ids=subj_train[tr], r2u=r2u_train[tr],
                emb_tensor=EMB_T, n_subjects=int(indexer.n_subjects), device=_dev,
                dense_X=(plain_dense_train[tr] if EXP["use_metadata"] else None),
                objective="bce",
                sample_weights=(sample_weights[tr] if sample_weights is not None else None),
                subj_emb_dim=int(EXP["plain_subj_emb_dim"]),
                hid1=int(EXP["plain_hid1"]), hid2=int(EXP["plain_hid2"]),
                lr=float(EXP["plain_lr"]), wd=float(EXP["plain_wd"]),
                epochs=int(EXP["epochs"]), batch_size=int(EXP["batch_size"]),
                val_fraction=float(EXP["val_fraction"]),
                patience=int(EXP["patience"]),
                feat_dropout=float(EXP["plain_feat_dropout"]),
                seed=SEED + int(fold.fold_id), show_progress=False,
            )
            oof_p = predict_probs(
                net, subj_train[oof], r2u_train[oof], EMB_T, _dev,
                int(EXP["predict_chunk"]),
                dense_X=(plain_dense_train[oof] if EXP["use_metadata"] else None),
            )
            val_p = predict_probs(
                net, subj_val, r2u_val, EMB_T, _dev, int(EXP["predict_chunk"]),
                dense_X=(plain_dense_val if EXP["use_metadata"] else None),
            )
        else:  # rich
            cfg = _rich_cfg(SEED + int(fold.fold_id))
            net = train_rich_mlp(
                y=y_train[tr], subject_ids=subj_train[tr], bc_ids=bc_train[tr],
                cluster_ids=cluster_train[tr], family_ids=family_train[tr],
                macro_ids=macro_train[tr], org_ids=org_train[tr],
                topic_ids=topic_train[tr],
                item_emb_tensor=EMB_T, row_to_uniq=r2u_train[tr],
                dense_X=(rich_dense_train[tr] if EXP["use_metadata"] else None),
                n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
                n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
                n_topics=_RICH_NT,
                cfg=cfg, device=_dev,
                sample_weights=(sample_weights[tr] if sample_weights is not None else None),
                show_progress=False,
            )
            # OOF predictions use the ORIGINAL (un-redacted) train-side
            # ids -- those rows are the model's own "test set" for the
            # stacker downstream and aren't part of the val redaction
            # contract.
            oof_p = predict_rich_mlp(
                net,
                subject_ids=subj_train[oof], bc_ids=bc_train[oof],
                cluster_ids=cluster_train[oof], family_ids=family_train[oof],
                macro_ids=macro_train[oof], org_ids=org_train[oof],
                topic_ids=topic_train[oof],
                item_emb_tensor=EMB_T, row_to_uniq=r2u_train[oof],
                dense_X=(rich_dense_train[oof] if EXP["use_metadata"] else None),
                n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
                n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
                n_topics=_RICH_NT, device=_dev, chunk=int(EXP["predict_chunk"]),
            )
            # Val predictions use the REDACTED val ids so the reported
            # NLL reflects "trained with per-unit dropout, evaluated on
            # the same per-unit dropout sample" -- the realistic
            # generalization scenario the user asked for. Item embedding
            # and dense_X are NEVER redacted (they're the always-on
            # signal); only the cat-id channels go through UNK.
            val_p = predict_rich_mlp(
                net,
                subject_ids=subj_val_red, bc_ids=bc_val_red,
                cluster_ids=cluster_val_red, family_ids=family_val_red,
                macro_ids=macro_val_red, org_ids=org_val_red,
                topic_ids=topic_val_red,
                item_emb_tensor=EMB_T, row_to_uniq=r2u_val,
                dense_X=(rich_dense_val if EXP["use_metadata"] else None),
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
# ## 13. Train the two baselines (plain M8, rich M8)

# %%
oof_preds: dict[str, np.ndarray] = {}
val_preds: dict[str, np.ndarray] = {}

oof_preds["plain_baseline"], val_preds["plain_baseline"] = _run_one_oof(
    "plain (M8 baseline, no rich features, no MoE)", "plain", None,
)
oof_preds["rich_baseline"], val_preds["rich_baseline"] = _run_one_oof(
    "rich (DCN-V2 + cats + NN + cluster, no MoE)", "rich", None,
)

# %% [markdown]
# ## 14. Train experts (rich) per partition axis -- ONLY for axes enabled in EXP.

# %%
if EXP["run_moe_family"]:
    for j, name in enumerate(_FAM_BUCKET_NAMES):
        in_bucket = (bucket_family_train == j)
        n_in = int(in_bucket.sum())
        w = _make_weights(in_bucket, EXP["expert_weight_multiplier"])
        print(
            f"\n[expert family/{name}] upweight={EXP['expert_weight_multiplier']}x on "
            f"{n_in:,} rows ({n_in / N_TRAIN:.1%}); mean-weight after normalize={w.mean():.4f}"
        )
        ok = f"expert_family_{name}"
        oof_preds[ok], val_preds[ok] = _run_one_oof(name, "rich", w)
else:
    print("\n[moe family] skipped (EXP['run_moe_family']=False)")

if EXP["run_moe_nn_support"]:
    for j, name in enumerate(_SUP_BUCKET_NAMES):
        in_bucket = (bucket_support_train == j)
        n_in = int(in_bucket.sum())
        w = _make_weights(in_bucket, EXP["expert_weight_multiplier"])
        print(
            f"\n[expert support/{name}] upweight={EXP['expert_weight_multiplier']}x on "
            f"{n_in:,} rows ({n_in / N_TRAIN:.1%}); mean-weight after normalize={w.mean():.4f}"
        )
        ok = f"expert_support_{name}"
        oof_preds[ok], val_preds[ok] = _run_one_oof(name, "rich", w)
else:
    print("\n[moe support] skipped (EXP['run_moe_nn_support']=False)")

print(f"\nAll variants trained: {list(oof_preds.keys())}")

# %% [markdown]
# ## 15. Per-model Platt calibration (apples-to-apples NLL)

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

print("\nCalibrated solo val log-loss (lower is better):")
_solo = {n: _nll(cal_val[n], y_val) for n in cal_val}
_plain_nll = _solo["plain_baseline"]
_rich_nll = _solo["rich_baseline"]
print(f"  {'plain_baseline':<32s}: {_plain_nll:.6f}   (reference)")
print(
    f"  {'rich_baseline':<32s}: {_rich_nll:.6f}   "
    f"(vs plain: {_rich_nll - _plain_nll:+.6f}  <-- RICH ARCH/FEATURES LIFT)"
)
for n in cal_val:
    if n in {"plain_baseline", "rich_baseline"}:
        continue
    d = _solo[n] - _rich_nll
    print(f"  {n:<32s}: {_solo[n]:.6f}   (vs rich:  {d:+.6f})")

# %% [markdown]
# ## 16. Routing helpers (hard + soft) + controls

# %%
def _route_hard(per_expert_val, bucket_per_row, expert_names_prefix):
    """Hard route: row i takes prediction from expert_<bucket_per_row[i]>."""
    out = np.empty(int(len(bucket_per_row)), dtype=np.float32)
    for j, b in enumerate(expert_names_prefix):
        mask = (bucket_per_row == j)
        out[mask] = per_expert_val[b][mask]
    return out


def _soft_route_categorical(per_expert_val, bucket_per_row, expert_keys, eps):
    K = len(expert_keys)
    w = soft_routing_weights_categorical(
        bucket_per_row, n_buckets=K, epsilon=float(eps),
    )
    return apply_soft_routing(per_expert_val, expert_names=expert_keys, weights=w)


def _soft_route_kernel(per_expert_val, score_per_row, expert_keys,
                       bucket_centroids, tau):
    w = soft_routing_weights_kernel(
        score_per_row, bucket_centroids=bucket_centroids, tau=float(tau),
    )
    return apply_soft_routing(per_expert_val, expert_names=expert_keys, weights=w)


def _summarize_axis(axis_name, expert_prefix, bucket_per_row_train,
                    bucket_per_row_val, expert_keys, bucket_names,
                    bucket_centroids=None, score_per_row_val=None,
                    soft_param_grid=None, soft_kind="categorical"):
    """Run the full hard + scrambled + uniform + best-single + soft sweep
    table for one partition axis. Returns a dict of named val-pred arrays
    for the heatmap section."""
    print("\n" + "=" * 80)
    print(f"AXIS: {axis_name}  (K_EXPERTS={len(expert_keys)}, "
          f"baseline=rich_baseline)")
    print("=" * 80)
    K = len(expert_keys)

    # Per-bucket slice diagnostic.
    print("\nPer-bucket NLL on each bucket slice (calibrated val):")
    header = (
        f"  {'model':<28s}"
        + "".join(f"{n:>13s}" for n in bucket_names) + f"{'overall':>10s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    _slice_table = {}
    for n in ["rich_baseline"] + expert_keys:
        row = [f"  {n:<28s}"]
        slice_vals = []
        for j in range(K):
            mask = (bucket_per_row_val == j)
            v = (
                _nll(cal_val[n][mask], y_val[mask])
                if int(mask.sum()) > 0 else float("nan")
            )
            slice_vals.append(v)
            row.append(f"{v:>13.6f}")
        row.append(f"{_solo[n]:>10.6f}")
        _slice_table[n] = slice_vals
        print("".join(row))

    print("\nDIAGONAL CHECK -- each expert vs rich_baseline on its own bucket:")
    print(f"  {'bucket':<14s}{'rich_base':>12s}{'expert':>12s}{'delta':>12s}")
    _diag_wins = 0
    for j, name in enumerate(bucket_names):
        base_v = _slice_table["rich_baseline"][j]
        ek = expert_keys[j]
        exp_v = _slice_table[ek][j]
        d = exp_v - base_v
        flag = "  <-- specialization works" if d < -0.0005 else ""
        if d < -0.0005:
            _diag_wins += 1
        print(f"  {name:<14s}{base_v:>12.6f}{exp_v:>12.6f}{d:>+12.6f}{flag}")

    # Routing schemes.
    p_routed_hard = _route_hard(cal_val, bucket_per_row_val, expert_keys)
    nll_routed = _nll(p_routed_hard, y_val)

    _rng_route = np.random.default_rng(SEED + 99)
    _scrambled_bucket = _rng_route.integers(0, K, size=N_VAL).astype(np.int64)
    p_scrambled = _route_hard(cal_val, _scrambled_bucket, expert_keys)
    nll_scrambled = _nll(p_scrambled, y_val)

    p_uniform = np.mean(
        [cal_val[k] for k in expert_keys], axis=0
    ).astype(np.float32)
    nll_uniform = _nll(p_uniform, y_val)

    _expert_solo = {k: _solo[k] for k in expert_keys}
    _best_single_name = min(_expert_solo, key=_expert_solo.get)
    nll_best_single = _expert_solo[_best_single_name]

    # Soft routing sweep.
    soft_results: list[tuple[str, float, np.ndarray]] = []
    if soft_kind == "categorical" and soft_param_grid is not None:
        for eps in soft_param_grid:
            p = _soft_route_categorical(cal_val, bucket_per_row_val,
                                        expert_keys, float(eps))
            soft_results.append((f"eps={eps:.3f}", _nll(p, y_val), p))
    elif soft_kind == "kernel" and soft_param_grid is not None:
        std_score = float(np.std(score_per_row_val).clip(1e-6, None))
        for tau_mult in soft_param_grid:
            tau = float(tau_mult) * std_score
            p = _soft_route_kernel(cal_val, score_per_row_val, expert_keys,
                                   bucket_centroids, tau)
            soft_results.append(
                (f"tau={tau:.4f} ({tau_mult}xstd)", _nll(p, y_val), p),
            )

    print("\n" + "-" * 80)
    print("ROUTING SCHEMES vs BASELINES (lower is better):")
    print(f"  {'rich_baseline':<48s}: {_rich_nll:.6f}   (reference)")
    print(
        f"  {'plain_baseline':<48s}: {_plain_nll:.6f}   "
        f"(vs rich: {_plain_nll - _rich_nll:+.6f})"
    )
    print(
        f"  {'scrambled routing (random expert per row)':<48s}: "
        f"{nll_scrambled:.6f}   (vs rich: {nll_scrambled - _rich_nll:+.6f})"
    )
    print(
        f"  {'uniform avg of ' + str(K) + ' experts':<48s}: "
        f"{nll_uniform:.6f}   (vs rich: {nll_uniform - _rich_nll:+.6f})"
    )
    print(
        f"  {'best single expert (' + _best_single_name + ')':<48s}: "
        f"{nll_best_single:.6f}   (vs rich: {nll_best_single - _rich_nll:+.6f})"
    )
    print(
        f"  {'hard-routed MoE':<48s}: "
        f"{nll_routed:.6f}   (vs rich: {nll_routed - _rich_nll:+.6f})"
    )
    if soft_results:
        print(f"  soft-routed MoE sweep ({soft_kind}):")
        for tag, nll, _ in soft_results:
            print(
                f"    {tag:<46s}: {nll:.6f}   (vs rich: {nll - _rich_nll:+.6f})"
            )

    best_soft = (
        min(soft_results, key=lambda r: r[1]) if soft_results else None
    )
    if best_soft is not None:
        tag, nll, p_best_soft = best_soft
        print(
            f"  -> BEST SOFT ROUTING ({tag}) NLL = {nll:.6f}  "
            f"(vs hard {nll_routed:+.6f}; vs rich {nll - _rich_nll:+.6f})"
        )

    # Lift decomposition (vs rich baseline).
    g = nll_scrambled - _rich_nll
    r = nll_routed - nll_scrambled
    t = nll_routed - _rich_nll
    print(f"\nLIFT DECOMPOSITION (vs rich_baseline):")
    print(f"  Global (scrambled - rich)        : {g:+.6f}")
    print(f"  Routing-specific (routed - scrambled): {r:+.6f}")
    print(f"  Total (routed - rich)            : {t:+.6f}")
    if abs(t) > 1e-9:
        share = max(0.0, -r) / max(1e-9, -t) * 100.0
        print(f"  Routing share of total           : {share:.1f}%")

    return {
        "K": K,
        "expert_keys": expert_keys,
        "bucket_names": bucket_names,
        "slice_table": _slice_table,
        "diag_wins": _diag_wins,
        "diag_total": K,
        "p_routed_hard": p_routed_hard,
        "p_scrambled": p_scrambled,
        "p_uniform": p_uniform,
        "nll_routed": nll_routed,
        "nll_scrambled": nll_scrambled,
        "nll_uniform": nll_uniform,
        "nll_best_single": nll_best_single,
        "best_single_name": _best_single_name,
        "soft_results": soft_results,
        "best_soft": best_soft,
    }


# %% [markdown]
# ## 17. Run the per-axis summaries

# %%
results = {}
if EXP["run_moe_family"]:
    _fam_expert_keys = [f"expert_family_{name}" for name in _FAM_BUCKET_NAMES]
    results["family"] = _summarize_axis(
        axis_name="FAMILY (subject family + 'other/UNK')",
        expert_prefix="expert_family_",
        bucket_per_row_train=bucket_family_train,
        bucket_per_row_val=bucket_family_val,
        expert_keys=_fam_expert_keys,
        bucket_names=_FAM_BUCKET_NAMES,
        soft_param_grid=EXP["soft_routing_eps_grid_family"],
        soft_kind="categorical",
    )

if EXP["run_moe_nn_support"]:
    _sup_expert_keys = [f"expert_support_{name}" for name in _SUP_BUCKET_NAMES]
    results["nn_support"] = _summarize_axis(
        axis_name="NN-SUPPORT (mean cosine-sim to top-K train neighbors)",
        expert_prefix="expert_support_",
        bucket_per_row_train=bucket_support_train,
        bucket_per_row_val=bucket_support_val,
        expert_keys=_sup_expert_keys,
        bucket_names=_SUP_BUCKET_NAMES,
        bucket_centroids=_sup_centroids,
        score_per_row_val=support_score_val,
        soft_param_grid=EXP["soft_routing_tau_grid_support_mult"],
        soft_kind="kernel",
    )

# %% [markdown]
# ## 18. Generalization check: metadata-redacted val prediction
#
# Take the trained rich_baseline (the LAST fold's val preds approximate
# this since we average across folds; for a clean gen check we re-run
# the FIRST fold's rich net on val with bc / subject cat ids zeroed).
# Compares "full metadata at predict" vs "redacted" so we know how much
# of the rich lift came from memorizing those channels vs how much
# generalizes when they're hidden.

# %%
print("\n" + "=" * 80)
print("GENERALIZATION CHECK -- rich_baseline with metadata zeroed at predict time")
print("=" * 80)

# Re-train one rich net on fold-0 train rows for the gen probe (fast:
# one extra fit). We don't average across folds for this probe -- it's
# a sanity check on the dropout, not a primary metric.
_fold0 = folds[0]
_tr0 = _fold0.train_row_idx
_cfg0 = _rich_cfg(SEED + 999)
print(f"  fitting rich net on fold-0 train ({len(_tr0):,} rows) for the gen probe...")
_gen_net = train_rich_mlp(
    y=y_train[_tr0], subject_ids=subj_train[_tr0], bc_ids=bc_train[_tr0],
    cluster_ids=cluster_train[_tr0], family_ids=family_train[_tr0],
    macro_ids=macro_train[_tr0], org_ids=org_train[_tr0],
    topic_ids=topic_train[_tr0],
    item_emb_tensor=EMB_T, row_to_uniq=r2u_train[_tr0],
    dense_X=(rich_dense_train[_tr0] if EXP["use_metadata"] else None),
    n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
    n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
    n_topics=_RICH_NT, cfg=_cfg0, device=_dev, show_progress=False,
)


def _predict_with_override(*, ids_override: dict | None = None) -> np.ndarray:
    """Predict on val with the listed fields overridden.

    ``ids_override`` is a dict whose keys are field names
    (``bc``/``topic``/``cluster``/``subject``/``family``/``macro``/``org``)
    and values are full-length per-row id arrays. Any field not in
    ``ids_override`` uses the REDACTED val arrays (the realistic
    baseline) so the bracket cases below isolate the marginal effect
    of swapping ONE field's redaction state without rebuilding the
    rest of the row.
    """
    o = ids_override or {}
    return predict_rich_mlp(
        _gen_net,
        subject_ids=o.get("subject", subj_val_red),
        bc_ids=o.get("bc", bc_val_red),
        cluster_ids=o.get("cluster", cluster_val_red),
        family_ids=o.get("family", family_val_red),
        macro_ids=o.get("macro", macro_val_red),
        org_ids=o.get("org", org_val_red),
        topic_ids=o.get("topic", topic_val_red),
        item_emb_tensor=EMB_T, row_to_uniq=r2u_val,
        dense_X=(rich_dense_val if EXP["use_metadata"] else None),
        n_subjects=_RICH_NS, n_bcs=_RICH_NB, n_clusters=_RICH_NC,
        n_families=_RICH_NF, n_macros=_RICH_NMF, n_orgs=_RICH_NO,
        n_topics=_RICH_NT, device=_dev, chunk=int(EXP["predict_chunk"]),
    )


def _all_unk(unk_id):
    return np.full(N_VAL, int(unk_id), dtype=np.int64)


# Bracket scenarios. The "realistic" row (default redaction sample) is
# the headline number that matches the rich_baseline NLL above.
# "oracle" gives the floor (if metadata were never redacted), and
# "all-UNK" the ceiling (if every cat field were missing).
_gen_realistic = _nll(_predict_with_override(), y_val)
_gen_oracle = _nll(_predict_with_override(ids_override={
    "subject": subj_val, "bc": bc_val, "cluster": cluster_val,
    "family": family_val, "macro": macro_val, "org": org_val,
    "topic": topic_val,
}), y_val)
_gen_bc_unk = _nll(_predict_with_override(ids_override={
    "bc": _all_unk(_RICH_NB), "topic": _all_unk(_RICH_NT),
}), y_val)
_gen_subj_unk = _nll(_predict_with_override(ids_override={
    "subject": _all_unk(_RICH_NS), "family": _all_unk(_RICH_NF),
    "macro": _all_unk(_RICH_NMF), "org": _all_unk(_RICH_NO),
}), y_val)
_gen_all_unk = _nll(_predict_with_override(ids_override={
    "subject": _all_unk(_RICH_NS), "bc": _all_unk(_RICH_NB),
    "cluster": _all_unk(max(_RICH_NC, 1)),
    "family": _all_unk(_RICH_NF), "macro": _all_unk(_RICH_NMF),
    "org": _all_unk(_RICH_NO), "topic": _all_unk(_RICH_NT),
}), y_val)
print(
    f"  oracle (no redaction, all metadata visible) : {_gen_oracle:.6f}   "
    f"(floor; this is the 'if metadata were perfect' lower bound)"
)
print(
    f"  realistic (item-grouped redaction sample)   : {_gen_realistic:.6f}   "
    f"(headline; matches the rich_baseline NLL above)"
)
print(
    f"  stress: bc + topic fully UNK                : {_gen_bc_unk:.6f}   "
    f"({_gen_bc_unk - _gen_realistic:+.6f} vs realistic)"
)
print(
    f"  stress: subject + family/macro/org fully UNK: {_gen_subj_unk:.6f}   "
    f"({_gen_subj_unk - _gen_realistic:+.6f} vs realistic)"
)
print(
    f"  stress: ALL cat fields fully UNK            : {_gen_all_unk:.6f}   "
    f"({_gen_all_unk - _gen_realistic:+.6f} vs realistic)"
)
print(
    f"  oracle - all-UNK spread                     : "
    f"{_gen_all_unk - _gen_oracle:+.6f}   "
    f"(total cat-channel contribution; smaller = more of the lift is "
    f"in item-emb / dense / NN, less is in cat-id memorization)"
)
print(
    "\nInterpretation: realistic should be within ~0.005 nat of oracle "
    "if the model generalizes well to the redaction distribution. "
    "If oracle - realistic > 0.02 nat the cat channels memorized "
    "specific items / subjects; raise the val (and train) "
    "rich_cat_dropout_* rates and re-run."
)
del _gen_net
if _dev == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## 19. Residual-vs-baseline diversity heatmaps
#
# One heatmap per active axis. Both use the RICH baseline as the
# subtraction anchor so the diversity number reflects what a stacker
# sitting on top of rich_baseline + experts would actually see.

# %%
def _render_heatmap(corr, names, title, fname,
                    cmap_high=0.6, fig_w=10.0, fig_h=8.0):
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


for axis, res in results.items():
    expert_keys = res["expert_keys"]
    p_rich_base = cal_val["rich_baseline"]
    _resid_names = expert_keys + ["routed", "uniform"]
    _resid_stack = np.stack(
        [cal_val[k] - p_rich_base for k in expert_keys]
        + [res["p_routed_hard"] - p_rich_base, res["p_uniform"] - p_rich_base],
        axis=1,
    )
    corr = np.corrcoef(_resid_stack.T)
    _Kr = corr.shape[0]
    mask_off = ~np.eye(_Kr, dtype=bool)
    expert_only_corr = np.corrcoef(_resid_stack[:, : len(expert_keys)].T)
    expert_only_mask = ~np.eye(len(expert_keys), dtype=bool)
    print(
        f"\n[axis={axis}] mean |residual-vs-rich corr| (experts only) = "
        f"{np.abs(expert_only_corr[expert_only_mask]).mean():.4f} "
        "(lower = more orthogonal => more useful to a stacker)"
    )
    _render_heatmap(
        corr, _resid_names,
        title=(
            f"Rich-MLP MoE residual-vs-rich correlation (val)\n"
            f"axis={axis}, K={res['K']}, "
            f"upweight={EXP['expert_weight_multiplier']}x"
        ),
        fname=f"rich_moe_{axis}_K{res['K']}_resid_corr.png",
    )

# %% [markdown]
# ## 20. Final verdict (per-axis decision)

# %%
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print(f"  plain_baseline NLL: {_plain_nll:.6f}")
print(f"  rich_baseline  NLL: {_rich_nll:.6f}   ({_rich_nll - _plain_nll:+.6f})")
if _rich_nll - _plain_nll <= -0.003:
    print("  -> RICH ARCHITECTURE/FEATURES PAY. Promote DCN + cat embeddings "
          "+ NN + centroid + meta numerics path to the next round of M8 "
          "experiments.")
elif _rich_nll - _plain_nll >= 0.0:
    print("  -> Rich does not beat plain. Either the cross tower is "
          "underfit, the new channels add noise, or the cat-dropout "
          "is too aggressive. Try (a) lower cat dropout, (b) drop the "
          "subject/bc embedding from rich, or (c) keep rich but trim "
          "to NN+centroid only.")
else:
    print("  -> Rich beats plain by < 0.003 nat. Marginal. Check the "
          "per-axis MoE tables: if any soft routing recovers >0.005 "
          "nat the combined rich+MoE path is worth promoting; else "
          "fall back to plain M8 and try a different diversity axis.")

for axis, res in results.items():
    g = res["nll_scrambled"] - _rich_nll
    r = res["nll_routed"] - res["nll_scrambled"]
    t = res["nll_routed"] - _rich_nll
    soft_t = (
        res["best_soft"][1] - _rich_nll
        if res["best_soft"] is not None else float("nan")
    )
    print(f"\n  [axis={axis}, K={res['K']}]")
    print(f"    diag_wins={res['diag_wins']}/{res['diag_total']}  "
          f"best_single={res['best_single_name']}  "
          f"({res['nll_best_single']:.6f})")
    print(f"    hard routing total lift (vs rich): {t:+.6f}  "
          f"(routing-specific {r:+.6f}, global {g:+.6f})")
    if res["best_soft"] is not None:
        tag, nll, _ = res["best_soft"]
        print(f"    best soft routing ({tag}) lift (vs rich): {soft_t:+.6f}")
    if r <= -0.002 and res["diag_wins"] >= max(1, res["diag_total"] // 2):
        print(f"    -> ROUTING-SPECIFIC WIN on axis={axis}. Add this as "
              "a separate L0 member (FWLS-stack on top of rich_baseline).")
    elif soft_t == soft_t and soft_t <= -0.003:
        print(f"    -> SOFT ROUTING WIN. Use the rich + soft router "
              f"({res['best_soft'][0]}) as an additional member.")
    elif g <= -0.003 and r >= -0.001:
        print(f"    -> NO routing-specific lift; the upweighting is a "
              "free training trick. Apply non-uniform sample weights to "
              "vanilla rich-M8 and skip the MoE architecture for this axis.")
    else:
        print(f"    -> Axis '{axis}' does not pay here under the rich MLP "
              "either. Try item-cluster k-means / a different cardinality / "
              "skip MoE on this dataset.")
