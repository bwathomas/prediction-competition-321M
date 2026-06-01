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
# # Loss-diversity probe: does training the SAME MLP on different error
# # scores and stacking them beat the NLL model alone?
#
# **Question.** We keep one architecture + one feature view fixed (the
# strong Member 8 = embedding + metadata GLU-MLP -- learned subject
# embedding + the 4096-D Qwen3-Embedding item vector + a dense item-form /
# metadata block) and only change the **training objective**. We then
# OOF-stack the variants and ask two things:
#
# 1. Does the stacked ensemble of loss-variants beat the plain NLL model?
# 2. How correlated are the variants' errors? (the diversity that a
#    stacker can actually exploit)
#
# **Why hold features fixed.** This isolates the *loss* as the only source
# of diversity. If different objectives don't decorrelate a fixed model,
# the multi-loss idea is dead regardless of features.
#
# **Objectives tested** (your table + the cold-start idea):
#
# | key            | objective                         | mechanism                                   |
# |----------------|-----------------------------------|---------------------------------------------|
# | `nll`          | BCE / log-loss (BASELINE)         | the reference model                         |
# | `brier`        | Brier / probability MSE           | smoother probs, gentler tails               |
# | `label_smooth` | label-smoothed CE                 | reduces overconfidence                      |
# | `focal`        | focal loss (gamma)                | gradient emphasis on hard rows              |
# | `class_weighted` | class-weighted CE               | up-weights the minority class               |
# | `ranking`      | pairwise logistic (AUC surrogate) | ordering signal; calibrated post-hoc        |
# | `distill`      | distillation from NLL teacher (T) | soft targets from the NLL OOF preds         |
# | `specialist`   | residual/specialist (OOF-weighted)| up-weights rows the NLL model gets wrong    |
# | `cold`         | cold-start up-weighting           | up-weights rare (item-cold) rows            |
#
# **OOF discipline.** Every variant is trained with the *same* item-cold
# K-fold OOF procedure as the production notebook. The teacher signal for
# `distill`/`specialist` comes from the NLL model's **OOF** predictions
# (held-out, so no leakage). Each variant's val prediction is the mean of
# its per-fold models (val items are item-cold to all folds).
#
# **Calibration.** `ranking` and `class_weighted` produce NLL-unsafe raw
# outputs, so every variant's *solo* NLL is reported AFTER a 1-D Platt fit
# on its OOF preds. The ensemble stacker works in logit space and
# re-calibrates each column internally, so it is fed raw probabilities.
#
# **Runtime.** This is a targeted concept test, not a production run. It
# trains `len(objectives) * n_folds` MLPs, so by default it sub-samples
# the train rows and uses few epochs. Scale `EXP` up once the concept is
# validated. All embeddings are read from the same cache the main
# notebook populated -- no re-encoding.

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

# Same encoder the production notebook uses (so the embedding cache hits).
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
# This is a CONCEPT TEST. Defaults keep the run to ~30-60 min on one GPU.
# Bump MAX_TRAIN_ROWS -> None and EPOCHS up once the concept is validated;
# trim OBJECTIVES to shorten the run.
EXP = {
    # Data / compute budget
    "max_train_rows": 600_000,   # None = use all train rows
    "n_folds": 3,
    "epochs": 20,
    "batch_size": 16384,
    "patience": 4,
    "val_fraction": 0.10,        # internal early-stopping split (item-grouped)
    "emb_device": None,          # None -> cuda if available else cpu
    "predict_chunk": 131_072,

    # Feature channels (held FIXED across all objectives). The strong
    # production Member 8 is embedding + metadata: learned subject embedding
    # + 4096-D item embedding + a dense item-form/metadata block. Set
    # use_metadata=False to ablate back to embedding-only.
    "use_metadata": True,

    # Shared M8 architecture (held FIXED across all objectives)
    "subj_emb_dim": 32,
    "hid1": 256,
    "hid2": 128,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "feat_dropout": 0.10,

    # Objective hyperparameters
    "label_smooth_eps": 0.05,
    "focal_gamma": 2.0,
    "distill_temperature": 2.0,
    "cold_count_floor": 5.0,     # weight ~ 1 / (item_train_count + floor)
    "specialist_weight_floor": 0.25,  # floor on hardness weight (mean-normalized)

    # Which objectives to run. `nll` is always run first (it is the
    # baseline AND the teacher for distill/specialist).
    "objectives": [
        "nll", "brier", "label_smooth", "focal",
        "class_weighted", "ranking", "distill", "specialist", "cold",
    ],
    "seed": SEED,
}

print("Experiment config:")
for k, v in EXP.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Load data + item-cold split (optionally sub-sampled)

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

# ---- optional sub-sample of TRAIN, ITEM-BASED (val kept full) ----------
# Sampling is done on UNIQUE ITEMS, not rows. We pick whole items at random
# until the cumulative row count first exceeds `max_train_rows`, then keep
# every row belonging to those items. This preserves:
#   * Item-cold guarantee at every downstream level (sub-sample boundary is
#     also an item boundary, so the OOF folds + internal val + cold-start
#     weights cannot see partial-item leakage).
#   * The natural rows-per-item distribution (rare items stay rare, hot
#     items stay hot), which the cold-start weighting in section 6 needs to
#     be meaningful.
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

# Build the unique-item table over the FULL df (train + val) so every
# row we ever score has an embedding available.
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
embedder.warm_caches_from_disk()
print("Encoding items (cache-aware)...")
item_emb_lookup, item_log = embedder.embed_unique(
    kind="item", keys=item_keys_list, texts=item_texts_list,
    benchmarks=item_benches_list,
)
print(f"  cached={item_log['n_cache_hits']}  encoded={item_log['n_encoded']}")
ITEM_EMB_DIM = int(embedder.embedding_dim)
print(f"Item embedding dim D = {ITEM_EMB_DIM}")

# %% [markdown]
# ## 4. Subject indexer + unique-embedding stack + per-row pointers
#
# We build ONE unique-embedding matrix `ALL_UNIQ` over every item key the
# experiment touches (train-subsample + val), plus per-row int pointers
# (`r2u_train`, `r2u_val`). A trailing zero row absorbs any (shouldn't
# happen) missing key so indexing never crashes.

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
_ZERO_IDX = _U  # trailing zero row

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
print(f"missing train keys: {(r2u_train == _ZERO_IDX).sum()}  "
      f"missing val keys: {(r2u_val == _ZERO_IDX).sum()}")

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

# Invariant: every fold's train-side item set must be DISJOINT from its
# OOF-side item set (item-cold guarantee). Hard-fail if violated -- this
# would silently leak through if `make_item_grouped_folds` were ever
# changed or fed row-grouped keys by mistake.
for f in folds:
    _train_items = set(_train_keys[f.train_row_idx].tolist())
    _oof_items = set(_train_keys[f.oof_row_idx].tolist())
    _overlap = _train_items & _oof_items
    if _overlap:
        raise RuntimeError(
            f"[oof check] fold {f.fold_id} has {len(_overlap)} items on "
            f"BOTH train and OOF sides -- item-cold guarantee violated."
        )
# Same invariant between every fold's OOF items and the final val set.
_val_item_set = set(val_df["item_key"].astype(str).tolist())
for f in folds:
    _oof_items = set(_train_keys[f.oof_row_idx].tolist())
    if _oof_items & _val_item_set:
        raise RuntimeError(
            f"[oof check] fold {f.fold_id} OOF items overlap the final val "
            "set -- this would leak val items into the per-fold training."
        )
print("[oof check] item-cold invariants OK: train/oof disjoint per fold + "
      "OOF disjoint from val.")

# %% [markdown]
# ## 6. Cold-start weights (item rarity in train)

# %%
# How many TRAIN rows each item appears in. Rare items (low count) are the
# cold-start regime the held-out eval is heavy on. Weight ~ 1/(count + c).
_item_counts = pd.Series(_train_keys).value_counts()
_count_per_row = _item_counts.reindex(_train_keys).to_numpy().astype(np.float64)
_cold_floor = float(EXP["cold_count_floor"])
cold_weights = (1.0 / (_count_per_row + _cold_floor)).astype(np.float32)
cold_weights = (cold_weights / cold_weights.mean()).astype(np.float32)  # mean 1
print(f"cold weights: min={cold_weights.min():.3f} mean={cold_weights.mean():.3f} "
      f"max={cold_weights.max():.3f}  (rare-item rows up-weighted)")

# %% [markdown]
# ## 6b. Dense item-form / metadata block (matches production Member 8)
#
# The strong Member 8 is NOT embedding-only: it also feeds a dense block of
# item-form metadata -- z-scored text-pool features (token/char length,
# has_latex, has_code, n_questions, n_numbers, is_multiple_choice, n_choices,
# lang_en), item-type one-hots, and condition x form (CoT) interactions.
# We replicate that exact block here so the probe MLP is "embedding +
# metadata", not embedding-only. Pool features are z-scored on TRAIN items
# only (val reuses train stats -> no leakage).

# %%
if EXP["use_metadata"]:
    from src.item_features import (
        POOL_FEATURE_NAMES as _POOL_FEATURE_NAMES,
        apply_zscore,
        build_cot_interactions as _build_cot_interactions,
        compute_features_for_items,
        cot_interaction_names as _cot_interaction_names,
        fit_zscore_stats,
        is_cot_from_condition as _is_cot_from_condition,
        item_type_onehot as _item_type_onehot,
        load_pool_features,
        save_pool_features,
        ITEM_TYPE_NAMES as _ITEM_TYPE_NAMES,
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
        str(ik): _item_type_onehot(row.to_dict())
        for ik, row in _pool_raw_by_item.iterrows()
    }
    _FORM_TYPE_NAMES = list(_ITEM_TYPE_NAMES)
    FORM_BLOCK_NAMES = (
        tuple(_POOL_COLS) + tuple(_FORM_TYPE_NAMES) + tuple(_cot_interaction_names())
    )
    print(f"[form] dense block width = {len(FORM_BLOCK_NAMES)} "
          f"({len(_POOL_COLS)} pool + {len(_FORM_TYPE_NAMES)} type + "
          f"{len(_cot_interaction_names())} cot-interactions)")

    def build_form_block(item_keys, conditions):
        """Replicates production build_item_form_block(full=True): z-scored
        pool features + item-type one-hots + CoT interactions -> [N, F]."""
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

    _dense_train_raw = build_form_block(
        _train_keys, train_df["condition"].astype(str).to_numpy()
    )
    _dense_val_raw = build_form_block(
        _val_keys, val_df["condition"].astype(str).to_numpy()
    )
    # Standardize the whole block on TRAIN (pool cols are already train
    # z-scored; this also scales the one-hot / interaction columns). Val
    # reuses the train stats, so there is no leakage.
    _dmean = _dense_train_raw.mean(axis=0).astype(np.float32)
    _dstd = _dense_train_raw.std(axis=0).astype(np.float32)
    _dstd = np.where(_dstd < 1e-6, 1.0, _dstd).astype(np.float32)
    dense_train = ((_dense_train_raw - _dmean) / _dstd).astype(np.float32)
    dense_val = ((_dense_val_raw - _dmean) / _dstd).astype(np.float32)
    DENSE_DIM = int(dense_train.shape[1])
    print(f"[form] dense_train={dense_train.shape}  dense_val={dense_val.shape}")
else:
    dense_train = None
    dense_val = None
    DENSE_DIM = 0
    print("[form] use_metadata=False -> embedding-only MLP")

# %% [markdown]
# ## 7. The shared GLU-MLP trainer (objective-parameterized)
#
# Architecture is byte-for-byte the Member 8 design from
# `src/mlp_member.py` (subject embedding | item embedding | dense metadata
# -> two GLU layers -> linear head). Only the **loss** changes per
# objective. Kept self-contained here so the production module is untouched.

# %%
import math


def _build_net(n_subjects, subj_emb_dim, item_dim, dense_dim, hid1, hid2,
               feat_dropout, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(int(seed))
    in_dim = int(subj_emb_dim) + int(item_dim) + int(dense_dim)
    use_dense = int(dense_dim) > 0

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.subj = nn.Embedding(int(n_subjects) + 1, int(subj_emb_dim))
            nn.init.normal_(self.subj.weight, std=0.05)
            self.l1v = nn.Linear(in_dim, hid1)
            self.l1g = nn.Linear(in_dim, hid1)
            self.l2v = nn.Linear(hid1, hid2)
            self.l2g = nn.Linear(hid1, hid2)
            self.head = nn.Linear(hid2, 1)
            self.drop = nn.Dropout(float(feat_dropout))

        def forward(self, sid, iemb, dz):
            parts = [self.subj(sid), iemb]
            if use_dense and dz is not None:
                parts.append(dz)
            x = torch.cat(parts, dim=1)
            x = self.drop(x)
            h1 = self.l1v(x) * torch.sigmoid(self.l1g(x))
            h2 = self.l2v(h1) * torch.sigmoid(self.l2g(h1))
            return self.head(h2).reshape(-1)

    return _Net().to(device)


def _objective_loss(logits, target, weights, *, objective, focal_gamma,
                    label_smooth_eps):
    """Per-objective scalar loss. `target` may be soft (distillation)."""
    import torch
    import torch.nn.functional as F

    if objective in ("bce", "class_weighted", "cold", "specialist", "distill"):
        eps = float(label_smooth_eps) if objective == "label_smooth" else 0.0
        t = target * (1.0 - eps) + 0.5 * eps
        per = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    elif objective == "label_smooth":
        eps = float(label_smooth_eps)
        t = target * (1.0 - eps) + 0.5 * eps
        per = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    elif objective == "brier":
        p = torch.sigmoid(logits)
        per = (p - target) ** 2
    elif objective == "focal":
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = target * p + (1.0 - target) * (1.0 - p)
        per = (1.0 - pt).clamp(min=0.0, max=1.0) ** float(focal_gamma) * ce
    else:
        raise ValueError(f"unknown objective {objective!r}")

    if weights is not None:
        return (per * weights).sum() / weights.sum().clamp(min=1e-8)
    return per.mean()


def train_m8_variant(*, y, subj_ids, r2u, emb_tensor, n_subjects, device,
                     dense_X=None,
                     objective="bce", focal_gamma=2.0, label_smooth_eps=0.0,
                     sample_weights=None, soft_targets=None, ranking=False,
                     subj_emb_dim=32, hid1=256, hid2=128, lr=1e-3, wd=1e-5,
                     epochs=20, batch_size=16384, val_fraction=0.10,
                     patience=4, feat_dropout=0.10, seed=0, show_progress=True):
    """Train one objective variant of the embedding+metadata GLU-MLP.
    Returns the trained torch net (eval mode, on `device`)."""
    import torch

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    N = int(y.shape[0])
    dense_dim = 0 if dense_X is None else int(np.asarray(dense_X).shape[1])
    rng = np.random.default_rng(int(seed))

    # Item-grouped internal val split (early stopping on held-out items).
    groups = np.asarray(r2u, dtype=np.int64).reshape(-1)
    uniq_g = np.unique(groups)
    n_val_g = max(1, int(round(float(val_fraction) * uniq_g.size)))
    val_g = set(uniq_g[rng.permutation(uniq_g.size)[:n_val_g]].tolist())
    val_mask = np.fromiter((g in val_g for g in groups), count=N, dtype=bool)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    if tr_idx.size == 0 or va_idx.size == 0:
        tr_idx = np.arange(N)
        va_idx = np.arange(N)

    net = _build_net(n_subjects, subj_emb_dim, int(emb_tensor.shape[1]), dense_dim,
                     hid1, hid2, feat_dropout, seed, device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    sid_all = torch.from_numpy(np.asarray(subj_ids, dtype=np.int64).reshape(-1)).to(device)
    r2u_all = torch.from_numpy(groups).to(device)
    y_all = torch.from_numpy(y).to(device)
    dz_all = (
        torch.from_numpy(np.asarray(dense_X, dtype=np.float32)).to(device)
        if dense_dim > 0 else None
    )
    tgt_all = (
        torch.from_numpy(np.asarray(soft_targets, dtype=np.float32).reshape(-1)).to(device)
        if soft_targets is not None else y_all
    )
    w_all = (
        torch.from_numpy(np.asarray(sample_weights, dtype=np.float32).reshape(-1)).to(device)
        if sample_weights is not None else None
    )

    def _logits(idx_t):
        dz = dz_all[idx_t] if dz_all is not None else None
        return net(sid_all[idx_t], emb_tensor[r2u_all[idx_t]], dz)

    tr_idx_t = torch.from_numpy(tr_idx.astype(np.int64)).to(device)
    va_idx_t = torch.from_numpy(va_idx.astype(np.int64)).to(device)

    n_steps = max(1, int(math.ceil(tr_idx.size / batch_size)))
    total_steps = n_steps * int(epochs)
    warmup = n_steps * 2

    def _lr_at(step):
        if step < warmup and warmup > 0:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _val_metric():
        net.eval()
        with torch.no_grad():
            if ranking:
                lg = []
                for bs in range(0, va_idx.size, batch_size):
                    b = va_idx_t[bs:bs + batch_size]
                    lg.append(_logits(b))
                lg = torch.cat(lg)
                yv = y_all[va_idx_t]
                pos = lg[yv >= 0.5]
                neg = lg[yv < 0.5]
                m = int(min(pos.numel(), neg.numel()))
                if m < 8:
                    return float("inf")
                gp = torch.Generator(device=device).manual_seed(0)
                pi = torch.randperm(pos.numel(), generator=gp, device=device)[:m]
                ni = torch.randperm(neg.numel(), generator=gp, device=device)[:m]
                return float(torch.nn.functional.softplus(-(pos[pi] - neg[ni])).mean())
            vs, vn = 0.0, 0
            for bs in range(0, va_idx.size, batch_size):
                b = va_idx_t[bs:bs + batch_size]
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    _logits(b), y_all[b], reduction="sum"
                )
                vs += float(per)
                vn += int(b.shape[0])
            return vs / max(1, vn)

    best_val = float("inf")
    best_state = None
    bad = 0
    step = 0
    for ep in range(int(epochs)):
        net.train()
        perm = torch.randperm(tr_idx.size, device=device)
        tr_shuf = tr_idx_t[perm]
        for bs in range(0, tr_idx.size, batch_size):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step)
            b = tr_shuf[bs:bs + batch_size]
            logits = _logits(b)
            if ranking:
                yb = y_all[b]
                pos = logits[yb >= 0.5]
                neg = logits[yb < 0.5]
                m = int(min(pos.numel(), neg.numel()))
                if m >= 1:
                    pi = torch.randperm(pos.numel(), device=device)[:m]
                    ni = torch.randperm(neg.numel(), device=device)[:m]
                    loss = torch.nn.functional.softplus(-(pos[pi] - neg[ni])).mean()
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, yb
                    )
            else:
                loss = _objective_loss(
                    logits, tgt_all[b],
                    (w_all[b] if w_all is not None else None),
                    objective=objective, focal_gamma=focal_gamma,
                    label_smooth_eps=label_smooth_eps,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        vloss = _val_metric()
        if show_progress:
            print(f"    epoch {ep + 1}/{epochs}  val={vloss:.5f}")
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_probs(net, subj_ids, r2u, emb_tensor, device, chunk=131_072,
                  dense_X=None):
    """Chunked forward -> [N] float32 probabilities (sigmoid of logit)."""
    import torch
    sid = np.asarray(subj_ids, dtype=np.int64).reshape(-1)
    r2 = np.asarray(r2u, dtype=np.int64).reshape(-1)
    dz = None if dense_X is None else np.asarray(dense_X, dtype=np.float32)
    n = int(sid.shape[0])
    out = np.empty(n, dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sid_t = torch.from_numpy(sid[s:e]).to(device)
            r2_t = torch.from_numpy(r2[s:e]).to(device)
            dz_t = torch.from_numpy(dz[s:e]).to(device) if dz is not None else None
            logits = net(sid_t, emb_tensor[r2_t], dz_t)
            out[s:e] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return np.clip(out, 1e-6, 1.0 - 1e-6)

# %% [markdown]
# ## 8. Run every objective with OOF discipline
#
# `nll` runs first (baseline + teacher). `distill`/`specialist` then reuse
# the NLL **OOF** train preds as an honest (held-out) teacher signal.

# %%
import torch

_dev = EXP["emb_device"] or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_dev}")
EMB_T = torch.from_numpy(ALL_UNIQ).to(_dev)


def _spec_for(name, *, p_nll_oof=None):
    """Return kwargs (sample_weights, soft_targets, ranking, objective...)
    for a given objective name. p_nll_oof = baseline OOF train preds."""
    base = dict(objective="bce", ranking=False, sample_weights=None,
                soft_targets=None, label_smooth_eps=0.0,
                focal_gamma=float(EXP["focal_gamma"]))
    if name == "nll":
        return base
    if name == "brier":
        return {**base, "objective": "brier"}
    if name == "label_smooth":
        return {**base, "objective": "label_smooth",
                "label_smooth_eps": float(EXP["label_smooth_eps"])}
    if name == "focal":
        return {**base, "objective": "focal"}
    if name == "class_weighted":
        pos = float((y_train >= 0.5).mean())
        pos = min(max(pos, 1e-3), 1 - 1e-3)
        w = np.where(y_train >= 0.5, (1 - pos) / pos, 1.0).astype(np.float32)
        w = (w / w.mean()).astype(np.float32)
        return {**base, "sample_weights": w}
    if name == "cold":
        return {**base, "sample_weights": cold_weights}
    if name == "ranking":
        return {**base, "ranking": True}
    if name == "distill":
        if p_nll_oof is None:
            raise ValueError("distill needs the NLL OOF preds")
        T = float(EXP["distill_temperature"])
        logit = np.log(np.clip(p_nll_oof, 1e-6, 1 - 1e-6) /
                       (1 - np.clip(p_nll_oof, 1e-6, 1 - 1e-6)))
        soft = 1.0 / (1.0 + np.exp(-(logit / T)))
        return {**base, "soft_targets": soft.astype(np.float32)}
    if name == "specialist":
        if p_nll_oof is None:
            raise ValueError("specialist needs the NLL OOF preds")
        hard = np.abs(y_train - p_nll_oof).astype(np.float32)
        w = hard / max(hard.mean(), 1e-6)
        w = np.maximum(w, float(EXP["specialist_weight_floor"])).astype(np.float32)
        w = (w / w.mean()).astype(np.float32)
        return {**base, "sample_weights": w}
    raise ValueError(f"unknown objective {name!r}")


def run_objective_oof(name, spec):
    """Full item-cold OOF for one objective. Returns (oof_train[N], val[N_val])."""
    print(f"\n=== Objective: {name} ===")
    acc = OofPredictionAccumulator(N_TRAIN, name=f"oof_{name}")
    val_stack = []
    for fold in folds:
        tr, oof = fold.train_row_idx, fold.oof_row_idx
        print(f"  [fold {fold.fold_id}] train={len(tr):,} oof={len(oof):,}")
        sw = spec["sample_weights"]
        st = spec["soft_targets"]
        net = train_m8_variant(
            y=y_train[tr], subj_ids=subj_train[tr], r2u=r2u_train[tr],
            emb_tensor=EMB_T, n_subjects=int(indexer.n_subjects), device=_dev,
            dense_X=(dense_train[tr] if dense_train is not None else None),
            objective=spec["objective"], focal_gamma=spec["focal_gamma"],
            label_smooth_eps=spec["label_smooth_eps"],
            sample_weights=(sw[tr] if sw is not None else None),
            soft_targets=(st[tr] if st is not None else None),
            ranking=spec["ranking"],
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
    return acc.finalize().astype(np.float32), np.mean(val_stack, axis=0).astype(np.float32)


# Run baseline first, then everything else.
oof_preds = {}
val_preds = {}
_objs = list(EXP["objectives"])
if "nll" in _objs:
    _objs = ["nll"] + [o for o in _objs if o != "nll"]

for name in _objs:
    spec = _spec_for(name, p_nll_oof=oof_preds.get("nll"))
    oof_preds[name], val_preds[name] = run_objective_oof(name, spec)

print("\nAll objectives trained.")

# %% [markdown]
# ## 9. Per-model calibration (Platt) + solo val NLL
#
# Every variant is Platt-calibrated on its own OOF preds before solo
# scoring, so `ranking`/`class_weighted` are NLL-safe and the comparison
# is apples-to-apples.

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
    """1-D logistic calibration fit on OOF, applied to val (reuses the
    stacker as a single-feature Platt scaler)."""
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


cal_val = {name: _platt(oof_preds[name], val_preds[name]) for name in _objs}

print("\nSolo val log-loss (Platt-calibrated):")
_solo = {name: _nll(cal_val[name], y_val) for name in _objs}
_nll_base = _solo["nll"]
for name in _objs:
    d = _solo[name] - _nll_base
    print(f"  {name:14s}: {_solo[name]:.6f}   (vs NLL: {d:+.6f})")

# %% [markdown]
# ## 10. Stacked ensembles vs NLL alone
#
# The stacker is fit OOF (logit space, single bucket) and applied to val.
# We compare: full ensemble, the {NLL, cold} pair, the uniform average,
# and NLL alone.

# %%
def _stack(names):
    M = len(names)
    Xo = build_stacker_features(
        member_probs=np.stack([oof_preds[n] for n in names], axis=1),
        bench_present=_zeros_tr, nn_neighbor_support=_zeros_tr,
        nn_mean_similarity=_zeros_tr, centroid_distance=_half_tr,
    )
    st = fit_stacker(X=Xo, y=y_train, feature_names=stacker_feature_names(M),
                     seed=SEED, n_iters=3000)
    Xv = build_stacker_features(
        member_probs=np.stack([val_preds[n] for n in names], axis=1),
        bench_present=_zeros_va, nn_neighbor_support=_zeros_va,
        nn_mean_similarity=_zeros_va, centroid_distance=_half_va,
    )
    pv = stacker_apply_batch(st, Xv).astype(np.float32)
    return pv, st


print("=" * 64)
print("ENSEMBLE COMPARISON (val log-loss; lower is better)")
print("=" * 64)
print(f"  NLL model alone (calibrated)     : {_nll_base:.6f}")

# Uniform average of all calibrated variants.
_uni = np.mean([cal_val[n] for n in _objs], axis=0).astype(np.float32)
print(f"  Uniform avg of {len(_objs)} variants       : {_nll(_uni, y_val):.6f}  "
      f"({_nll(_uni, y_val) - _nll_base:+.6f})")

# {NLL, cold} pair -- isolates the cold-start idea.
if "cold" in _objs:
    _p_cold, _ = _stack(["nll", "cold"])
    print(f"  Stacker {{NLL, cold}}             : {_nll(_p_cold, y_val):.6f}  "
          f"({_nll(_p_cold, y_val) - _nll_base:+.6f})")

# Full ensemble.
_p_full, _st_full = _stack(_objs)
print(f"  Stacker ALL {len(_objs)} variants        : {_nll(_p_full, y_val):.6f}  "
      f"({_nll(_p_full, y_val) - _nll_base:+.6f})")

print("\nFull-ensemble stacker weights (member columns, logit space):")
_w = _st_full.known.weights if hasattr(_st_full, "known") else _st_full.weights
for i, name in enumerate(_objs):
    print(f"  {name:14s}: {float(_w[i]):+.4f}")

# %% [markdown]
# ## 11. Error-correlation heatmap
#
# Correlation of signed errors `(p_cal - y)` on val across variants. High
# off-diagonal values (~0.9+) mean the loss change did NOT produce
# exploitable diversity.

# %%
_errs = np.stack([cal_val[n] - y_val for n in _objs], axis=1)  # [N_val, K]
_corr = np.corrcoef(_errs.T)
print("Mean off-diagonal |error correlation| = "
      f"{np.abs(_corr[~np.eye(len(_objs), dtype=bool)]).mean():.4f}")

try:
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(_corr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(_objs)))
    ax.set_yticks(range(len(_objs)))
    ax.set_xticklabels(_objs, rotation=45, ha="right")
    ax.set_yticklabels(_objs)
    for i in range(len(_objs)):
        for j in range(len(_objs)):
            ax.text(j, i, f"{_corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if _corr[i, j] < 0.7 else "black", fontsize=8)
    ax.set_title("Loss-variant error correlation (val)\nsame model + features, different objective")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _outdir = ROOT / "artifacts" / "diagnostics"
    _outdir.mkdir(parents=True, exist_ok=True)
    _outpath = _outdir / "loss_diversity_corr.png"
    fig.savefig(_outpath, dpi=120, bbox_inches="tight")
    print(f"[heatmap] saved -> {_outpath}")
    plt.show()
except Exception as exc:  # pragma: no cover - plotting optional
    print(f"[heatmap] skipped ({exc!r})")
    print(np.round(_corr, 3))

# %% [markdown]
# ## 12. Verdict
#
# Read it like this:
# * If **Stacker ALL** beats **NLL alone** by more than run-to-run noise
#   (rule of thumb on a sub-sample: > ~0.002 val NLL) AND several variants
#   carry non-trivial stacker weight, loss-diversity adds something.
# * If the gain is ~0 (or negative) and the heatmap is uniformly ~0.9+,
#   the objectives are near-duplicates and multi-loss is not worth it --
#   diversity must come from features/architecture instead.
# * Watch `cold` specifically: if {NLL, cold} helps but the others don't,
#   the win is cold-start re-weighting, not loss-shape variety.

# %%
_best = min(_objs, key=lambda n: _nll(cal_val[n], y_val))
_full = _nll(_p_full, y_val)
print("=" * 64)
print("VERDICT")
print("=" * 64)
print(f"  NLL alone            : {_nll_base:.6f}")
print(f"  Best single variant  : {_best} ({_solo[_best]:.6f})")
print(f"  Full stacked ensemble: {_full:.6f}  ({_full - _nll_base:+.6f} vs NLL)")
print(f"  Mean |off-diag corr| : "
      f"{np.abs(_corr[~np.eye(len(_objs), dtype=bool)]).mean():.4f}")
if _full < _nll_base - 0.002:
    print("  -> Loss-diversity ensemble HELPS on this run. Scale up & confirm.")
elif _full < _nll_base - 0.0005:
    print("  -> Marginal gain; likely variance reduction, not new signal.")
else:
    print("  -> No meaningful gain. Objectives are near-duplicates; spend the "
          "diversity budget on features/architecture, not losses.")
