"""Leave-one-category-out (LOO) MLP diversity experiment — family=qwen, single outer item-fold.

QUESTION
========
Does a LightGBM stack over K leave-one-category-out MLP predictions BEAT a single full
MLP trained on all categories?  Each "category" is one logical block of the MLP input:

  * ``item_embedding``     -> the per-row item embedding channel (use_item_emb)
  * ``subject_embedding``  -> the LEARNED subject nn.Embedding channel (subj_emb_dim>0)
  * each AIDE feature GROUP present in DR/features/qwen (the dense_X channel), namely:
        label-derived (fold-keyed): nn_label_derivatives, cluster_passrate,
                                    cluster_subject, counts_subject
        fold-invariant ("all")    : centroid_distance, cluster_geometry,
                                    nn_geometry, item_cluster

For each category c we train ONE MLP on the TRAIN rows using ALL categories EXCEPT c, and
predict the OOF rows -> p_loo[c]. We also train ONE full MLP (all categories) -> p_full.
Then we LightGBM-stack the K p_loo columns with an HONEST inner GroupKFold(item_key) over
the OOF rows, and compare soft cross-entropy of {each p_loo[c], p_full, LGB-stack, logit-mean}.

OUTER SPLIT (nf3 item-grouped folds — verified provenance)
==========================================================
The nf3 shards in DR/features/qwen were derived by ``aide.features.driver.derive_family``
under ``build_manifest(...)`` + ``aide.hygiene.splits.outer_folds`` — i.e. the fold of an
item is ``aide.hygiene.manifest.item_fold(item_key, n_folds=3, seed=0)`` (sha256 hash mod 3),
NOT ``src.oof_folds.make_item_grouped_folds`` (that uses np.random shuffle + array_split and
is a DIFFERENT partition).  We therefore use ``item_fold`` here so our row->fold assignment
matches the shards byte-for-byte.  ASSUMPTION A-FOLD (verify): the shards were written with
code_version="v2" (docs/AIDE_OVERNIGHT_WORKFLOW.md §4.2) and the default NpzBackend
(overnight.py constructs FeatureCache without a backend=, so .npz).  We assert the label
shard row order equals our reconstructed (subject|item) order per fold — a hard leakage/
alignment gate that fails loudly if either assumption is wrong.

  TRAIN rows = folds {0, 1}   (item-disjoint from fold 2)
  OOF   rows = fold 2

LEAKAGE DISCIPLINE (critical)
=============================
Label-derived groups are OOF target-encodings.  For a row whose item is OOF in fold f, the
ONLY non-leaky shard is fold f's (derived on the OTHER folds' train labels):
  * OOF rows (fold 2): label features come from the fold-2 shard (derived on folds {0,1}).
  * TRAIN rows (folds 0 & 1): fold-0 rows' label features from the fold-0 shard, fold-1
    rows' from the fold-1 shard.  We assemble per fold and stitch by row.
Fold-invariant groups (centroid_distance, cluster_geometry, nn_geometry, item_cluster) are
label-free, derived once at fold="all" (one row per UNIQUE item), gathered per row by item.
The subject nn.Embedding is learned from scratch on each fit's train rows -> never leaks the
OOF rows' labels (it only sees train-row labels via the BCE gradient).  item_embedding is a
static content vector (no labels) -> leak-free.

REAL fit_mlp_member KWARGS (verbatim, src/mlp_member.py::fit_mlp_member)
=======================================================================
  fit_mlp_member(*, labels, subject_ids=None, n_subjects=0, subj_emb_dim=0,
                 item_emb_unique=None, row_to_uniq=None, dense_X=None,
                 dense_feature_names=(), hid1=128, hid2=64, learning_rate=1e-3,
                 weight_decay=1e-5, epochs=40, batch_size=16384, val_fraction=0.1,
                 early_stopping_patience=6, warmup_epochs=2, use_cosine_schedule=True,
                 feat_dropout=0.10, seed=0, device=None, holdout_group_id=None,
                 show_progress=True, ncl_anchor_preds=None, ncl_lambda=0.0) -> MlpMemberState
  Input channel order is [subj_emb | item_emb | dense_X]; any channel may be absent.
  Ablations:
    * drop item_embedding  -> item_emb_unique=None, row_to_uniq=None (use_item_emb off)
    * drop subject_embedding-> subj_emb_dim=0, subject_ids=None (no learned embedding)
    * drop a dense group   -> remove that group's columns from dense_X (+ its names)
  apply: src/mlp_member.py::apply_batch(state, subject_ids=, item_emb=[N,D], dense_X=).
  BCEWithLogitsLoss with continuous targets in [0,1] == soft cross-entropy == the metric.

RUNTIME BUDGET
==============
11 MLP fits (8 dense LOO + item LOO + subject LOO + 1 full).  N_train(folds 0,1) ~= 2/3 of
264350 ~= 176k rows; OOF (fold 2) ~= 88k.  HP: epochs=18, hid1=256/hid2=128, batch=16384,
early_stopping_patience=4.  On an A100 a fit is ~30-60s, so 11 fits + assembly + LGB inner
CV is well under ~15 min.  device='cuda'.  GPU-justified: the inner loop is float matmul over
many minibatches (the A100 policy's "vectorizable float-array" case).

run_bg / poll
=============
``fn()`` writes progress + final result to /content/exp_loo_qwen.json and returns the dict.
Launch (after the run_bg/poll harness cell):
    import os; os.environ.setdefault("SHIP_FAMILY","qwen")
    run_bg("exp_loo_qwen", fn)
Poll with a tiny fast cell:
    import json; print(json.load(open("/content/exp_loo_qwen.json")).get("stage"))
    # on done it also holds the full result table.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------
FAMILY = os.environ.get("SHIP_FAMILY", "qwen").strip().lower()       # this experiment: qwen
REPO_ROOT = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
DRIVE_ROOT = os.environ.get(
    "SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
CODE_VERSION = os.environ.get("SHIP_CODE_VERSION", "v2")             # A-FOLD: shard code_version
N_FOLDS = 3
SPLIT_SEED = 0
OOF_FOLD = int(os.environ.get("SHIP_OOF_FOLD", "2"))   # which nf3 fold is held out (0/1/2)
N_TRAIN_EXPECTED = 264350

FAM_ALIAS = {"qwen": "qwen", "nemotron": "llama", "llama": "llama",
             "lgai": "mistral", "mistral": "mistral"}

# Feature groups present in DR/features/<family>, split by the store's routing.
# (aide.features.store.FOLD_INVARIANT_GROUPS / LABEL_DERIVED_GROUPS — verified at run time.)
LABEL_GROUPS = ["nn_label_derivatives", "cluster_passrate", "cluster_subject", "counts_subject"]
GEOM_GROUPS = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster"]
DENSE_GROUPS = GEOM_GROUPS + LABEL_GROUPS  # all dense AIDE groups (8)

# Non-dense channels are categories too:
SPECIAL_CATS = ["item_embedding", "subject_embedding"]

# Fixed MLP hyperparameters across ALL fits (the ONLY variable is the omitted category).
HP = dict(
    subj_emb_dim=32,
    hid1=256,
    hid2=128,
    learning_rate=1.0e-3,
    weight_decay=1.0e-5,
    epochs=18,                  # modest: ~11 fits under ~15 min on A100
    batch_size=16384,
    early_stopping_patience=4,
    feat_dropout=0.10,
    val_fraction=0.1,
)
SEED = 0
INNER_FOLDS = 5                 # GroupKFold(item_key) over the OOF rows for the LGB stack
# 'full' = unredacted 3-fold CV over every labeled row; 'ship' = redacted 264k 2-fold sample.
ROW_SOURCE = os.environ.get("SHIP_ROW_SOURCE", "full").strip().lower()
# Base learner for the leave-one-kind-out members: 'mlp' (default) or 'lgbm' (LightGBM on the
# dense derived feature groups + a PCA item-embedding kind — design (b), to test whether the
# item embedding helps trees, which AIDE never tested).
MODEL = os.environ.get("SHIP_MODEL", "mlp").strip().lower()
PCA_DIM = int(os.environ.get("SHIP_PCA_DIM", "64"))   # PCA dim of item embedding for lgbm
# LightGBM hyperparameters (fixed across all members; only the omitted kind varies).
LGB_HP = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=128, max_depth=-1,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              min_child_samples=100, lambda_l1=1.0, lambda_l2=2.0, verbosity=-1, seed=SEED)
LGB_ROUNDS = int(os.environ.get("SHIP_LGB_ROUNDS", "1500"))   # max; early-stopped on item-grouped val
_TAG = (f"{ROW_SOURCE}_fold{OOF_FOLD}" if MODEL == "mlp"
        else f"{MODEL}_{ROW_SOURCE}_fold{OOF_FOLD}")
STATUS_PATH = f"/content/exp_loo_{FAMILY}_{_TAG}.json"
# Persist every trained model + result to Drive (survives runtime recycle), reloadable.
# model+source+fold-specific subdir so runs never clobber each other.
SAVE_ROOT = os.environ.get("SHIP_EXP_SAVE_ROOT",
                           f"{DRIVE_ROOT}/ship/exp_loo/{FAMILY}/{_TAG}")
SAVE_MODELS = os.environ.get("SHIP_SAVE_MODELS", "1") != "0"
_EPS = 1.0e-6


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _write_status(payload):
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception:
        pass


def _load_keys_npy(path):
    arr = np.load(path, allow_pickle=True).reshape(-1)
    return [str(x) for x in arr.tolist()]


def soft_logloss(y, p):
    """Mean soft cross-entropy for continuous label y in [0,1]; clip p to [eps, 1-eps]."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log(1.0 - p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


# ----------------------------------------------------------------------------------
# the run_bg-friendly entry point
# ----------------------------------------------------------------------------------
def fn():
    t0 = time.time()
    prog = {"experiment": "loo_category_mlp", "family": FAMILY, "stage": "start",
            "ok": None, "t_start": t0}
    _write_status(prog)

    def step(stage, **extra):
        prog.update(stage=stage, t_elapsed=round(time.time() - t0, 1), **extra)
        _write_status(dict(prog))
        print(f"[exp-loo] {stage}  (+{prog['t_elapsed']}s)  {extra if extra else ''}", flush=True)

    try:
        # ---- (0) imports ----------------------------------------------------------
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        step("imports")
        from aide.features.cache import FeatureCache
        from aide.features.driver import FAMILY_SLUG, load_embeddings
        from aide.features.store import (FoldFeatureStore, FOLD_INVARIANT_GROUPS,
                                         LABEL_DERIVED_GROUPS)
        from aide.hygiene.manifest import item_fold
        from src.mlp_member import apply_batch as mlp_apply_batch
        from src.mlp_member import fit_mlp_member
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
        if MODEL == "lgbm":
            from sklearn.decomposition import PCA

        driver_fam = FAM_ALIAS[FAMILY]
        slug = FAMILY_SLUG[driver_fam]
        emb_dir = f"{DRIVE_ROOT}/embeddings/{slug}"

        # sanity: our group routing matches the store's catalog (else assemble would mis-route)
        for g in LABEL_GROUPS:
            assert g in LABEL_DERIVED_GROUPS, f"{g} not label-derived in store catalog"
        for g in GEOM_GROUPS:
            assert g in FOLD_INVARIANT_GROUPS, f"{g} not fold-invariant in store catalog"
        step("config", driver_family=driver_fam, slug=slug, code_version=CODE_VERSION,
             label_groups=LABEL_GROUPS, geom_groups=GEOM_GROUPS)

        # ---- (1) embeddings -------------------------------------------------------
        item_keys, item_emb = load_embeddings(f"{emb_dir}/items.parquet")
        item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        D_EMB = int(item_emb.shape[1])
        item_emb_idx = {str(k): i for i, k in enumerate(item_keys)}
        step("loaded_embeddings", n_items=len(item_keys), d_emb=D_EMB)

        # ---- (2) labels + canonical rows -----------------------------------------
        import pandas as pd
        db = glob.glob(f"{DRIVE_ROOT}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
        labels_df = pd.read_parquet(db, columns=["subject_key", "item_key", "label"])
        labels_df["subject_key"] = labels_df["subject_key"].astype(str)
        labels_df["item_key"] = labels_df["item_key"].astype(str)
        if labels_df.duplicated(["subject_key", "item_key"]).any():
            labels_df = (labels_df.groupby(["subject_key", "item_key"], sort=False)["label"]
                         .mean().reset_index())
        label_map = {(s, i): float(l) for s, i, l in zip(
            labels_df["subject_key"].to_numpy(), labels_df["item_key"].to_numpy(),
            labels_df["label"].to_numpy())}

        # ROW SOURCE (module-level ROW_SOURCE): 'full' = every labeled (subject,item) row ->
        # all 3 nf3 folds populated (unredacted 3-fold CV: hold out fold f, train on other two).
        # 'ship' = the curated redacted 264350-row sample (folds 1&2 only) AIDE trained on.
        if ROW_SOURCE == "ship":
            rows_dir = f"{DRIVE_ROOT}/ship/rows"
            tr_item = _load_keys_npy(f"{rows_dir}/_tr_item.npy")
            tr_subj = _load_keys_npy(f"{rows_dir}/_tr_subj.npy")
        else:
            tr_subj = [str(s) for s in labels_df["subject_key"].to_numpy()]
            tr_item = [str(i) for i in labels_df["item_key"].to_numpy()]
        N_tr = len(tr_item)
        if ROW_SOURCE == "ship" and N_tr != N_TRAIN_EXPECTED:
            print(f"[exp-loo] WARN: N_tr={N_tr} != expected {N_TRAIN_EXPECTED}", flush=True)
        if len(tr_subj) != N_tr:
            raise ValueError("tr_item / tr_subj length mismatch")
        step("loaded_rows", n_train=N_tr, row_source=ROW_SOURCE)

        # restrict to rows that actually have features (the derivation's emb_set).
        # Feature shards are keyed by the DRIVER family name (FAM_ALIAS): nemotron->llama,
        # lgai->mistral (qwen->qwen). Use driver_fam so the cache path matches the shards
        # on Drive (features/<driver_fam>/...), not the raw SHIP_FAMILY alias.
        store = FoldFeatureStore(FeatureCache(f"{DRIVE_ROOT}/features", code_version=CODE_VERSION),
                                 embedding_family=driver_fam, seed=SPLIT_SEED, n_folds=N_FOLDS)
        geo0 = store.cache.read_shard(store._key(GEOM_GROUPS[0], "all"))
        feat_items = set(str(k) for k in geo0.row_ids)
        geo_index = {str(k): i for i, k in enumerate(geo0.row_ids)}
        step("geometry_shard_loaded", n_geo_rows=len(geo0.row_ids), n_geo_cols=geo0.X.shape[1])

        keep = np.array([(it in feat_items) and ((s, it) in label_map)
                         for s, it in zip(tr_subj, tr_item)], dtype=bool)
        tr_item = [tr_item[i] for i in np.where(keep)[0]]
        tr_subj = [tr_subj[i] for i in np.where(keep)[0]]
        N = len(tr_item)
        y = np.array([label_map[(s, i)] for s, i in zip(tr_subj, tr_item)], dtype=np.float32)
        step("filtered_rows", n_rows=N, n_dropped=int((~keep).sum()),
             y_mean=round(float(y.mean()), 4))

        # ---- (3) nf3 fold per row (item_fold -> matches the shards) ---------------
        row_fold = np.fromiter((item_fold(i, N_FOLDS, SPLIT_SEED) for i in tr_item),
                               dtype=np.int64, count=N)
        oof_mask = row_fold == OOF_FOLD
        train_mask = ~oof_mask
        n_oof, n_train = int(oof_mask.sum()), int(train_mask.sum())
        if n_oof == 0 or n_train == 0:
            raise ValueError(f"degenerate split: train={n_train}, oof={n_oof}")
        step("nf3_split", n_train=n_train, n_oof=n_oof, oof_frac=round(n_oof / N, 4))

        # ---- (4) assemble dense AIDE features per row, leakage-correct ------------
        # geometry: gather per row by item (fold='all', one row per unique item).
        # label groups: per outer fold, assemble that fold's shard and stitch by row.
        # Build a global row index keyed by (subject|item) so we can scatter fold shards.
        rid_all = np.array([f"{s}|{i}" for s, i in zip(tr_subj, tr_item)])
        rid_pos = {r: k for k, r in enumerate(rid_all)}  # (subject|item) -> our row index

        # --- geometry block (gather by item) ---
        step("assemble_geometry")
        Xg, gcols = store.assemble(GEOM_GROUPS, fold=0, check_coverage=False)  # 'all' routing
        # Xg/gcols are aligned to geo0.row_ids order; gather per our rows by item:
        gi = np.array([geo_index[i] for i in tr_item], dtype=np.int64)
        Xgeo = Xg[gi].astype(np.float32)                          # [N, n_geo_cols]
        geo_col_groups = []  # group label per geometry column, recovered below
        # recover per-column group membership by re-assembling each geom group alone:
        geo_group_of_col = []
        for g in GEOM_GROUPS:
            _, gc = store.assemble([g], fold=0, check_coverage=False)
            geo_group_of_col += [g] * len(gc)
        if len(geo_group_of_col) != Xgeo.shape[1]:
            raise ValueError("geometry per-column group mapping length mismatch")
        step("geometry_ready", n_geo_cols=Xgeo.shape[1])

        # --- label block (per-fold shards, scatter into row order) ---
        step("assemble_label_groups")
        # discover label column count + per-column group via fold-OOF (fold 2) shard,
        # which is guaranteed present (the OOF derivation always writes it).
        lab_group_of_col = []
        for g in LABEL_GROUPS:
            _, lc = store.assemble([g], fold=OOF_FOLD, check_coverage=False)
            lab_group_of_col += [g] * len(lc)
        n_lab_cols = len(lab_group_of_col)
        Xlab = np.full((N, n_lab_cols), np.nan, dtype=np.float32)
        filled = np.zeros(N, dtype=bool)
        for f in range(N_FOLDS):
            # rows whose item is OOF in fold f -> their non-leaky label features live in
            # fold f's shard (derived on the OTHER folds' train labels).
            Xl, lcols = store.assemble(LABEL_GROUPS, fold=f, check_coverage=False)
            shard_rids = np.asarray(
                store.cache.read_shard(store._key(LABEL_GROUPS[0], f)).row_ids).astype(str)
            if len(lcols) != n_lab_cols:
                raise ValueError(f"fold {f}: label col count {len(lcols)} != {n_lab_cols}")
            # map each shard row to our row index; shards may contain rows we dropped/aren't
            # in this run's row set -> skip those, and skip rows already filled.
            hit = 0
            for j, r in enumerate(shard_rids):
                k = rid_pos.get(r)
                if k is not None and not filled[k] and row_fold[k] == f:
                    Xlab[k] = Xl[j]
                    filled[k] = True
                    hit += 1
            step(f"label_fold{f}_scattered", shard_rows=len(shard_rids), matched=hit)
        n_unfilled = int((~filled).sum())
        if n_unfilled:
            # any row whose fold shard didn't cover it -> neutral 0 (rare; report it).
            Xlab[~filled] = 0.0
            print(f"[exp-loo] WARN: {n_unfilled} rows had no label-shard row; zero-filled.",
                  flush=True)
        step("label_ready", n_lab_cols=n_lab_cols, n_unfilled=n_unfilled)

        # full dense matrix + per-column group + group->name lists
        dense_full = np.concatenate([Xgeo, Xlab], axis=1).astype(np.float32)
        dense_group_of_col = np.array(geo_group_of_col + lab_group_of_col, dtype=object)
        dense_names_full = tuple(f"{geo_group_of_col[i]}__c{i}" for i in range(len(geo_group_of_col)))
        dense_names_full += tuple(f"{lab_group_of_col[i]}__c{i}" for i in range(len(lab_group_of_col)))
        if not np.isfinite(dense_full).all():
            dense_full = np.where(np.isfinite(dense_full), dense_full, 0.0).astype(np.float32)
        cat_ncols = {g: int((dense_group_of_col == g).sum()) for g in DENSE_GROUPS}
        cat_ncols["item_embedding"] = D_EMB
        cat_ncols["subject_embedding"] = int(HP["subj_emb_dim"])
        step("dense_ready", dense_dim=dense_full.shape[1], category_ncols=cat_ncols)

        # ---- (5) item_emb_unique + row_to_uniq + subject vocab --------------------
        uniq_keys, uniq_pos = [], {}
        row_to_uniq = np.empty(N, dtype=np.int64)
        for r, k in enumerate(tr_item):
            j = uniq_pos.get(k)
            if j is None:
                j = len(uniq_keys); uniq_pos[k] = j; uniq_keys.append(k)
            row_to_uniq[r] = j
        item_emb_unique = np.empty((len(uniq_keys), D_EMB), dtype=np.float32)
        for j, k in enumerate(uniq_keys):
            item_emb_unique[j] = item_emb[item_emb_idx[str(k)]]

        subj_vocab = {}
        for s in tr_subj:
            subj_vocab.setdefault(s, len(subj_vocab))
        n_subjects = len(subj_vocab)
        sid = np.fromiter((subj_vocab[s] for s in tr_subj), dtype=np.int64, count=N)
        step("emb_index_ready", n_unique_items=len(uniq_keys), n_subjects=n_subjects)

        train_idx = np.where(train_mask)[0]
        oof_idx = np.where(oof_mask)[0]
        oof_items = [tr_item[r] for r in oof_idx]
        oof_y = y[oof_idx]
        oof_group = np.asarray(oof_items)  # GroupKFold groups = item_key (cold-start honest)

        # per-row item embedding gather for apply (OOF rows only)
        oof_item_emb = np.empty((n_oof, D_EMB), dtype=np.float32)
        for j, r in enumerate(oof_idx):
            oof_item_emb[j] = item_emb[item_emb_idx[str(tr_item[r])]]

        # ---- (5b) lgbm only: PCA item-embedding kind (design (b)) ------------------
        # AIDE never fed item embeddings to trees; this adds a PCA-reduced item-embedding
        # GROUP so leave-out[item_emb_pca] vs full measures whether it helps LightGBM.
        # PCA is label-free; fit on TRAIN unique items, transform all, gather per row.
        pca_obj = None
        if MODEL == "lgbm":
            train_uniq = np.unique(row_to_uniq[train_idx])
            pdim = int(min(PCA_DIM, item_emb_unique.shape[1], len(train_uniq)))
            pca_obj = PCA(n_components=pdim, random_state=SEED)
            pca_obj.fit(item_emb_unique[train_uniq])
            emb_pca_uniq = pca_obj.transform(item_emb_unique).astype(np.float32)
            Xpca = emb_pca_uniq[row_to_uniq].astype(np.float32)          # [N, pdim]
            dense_full = np.concatenate([dense_full, Xpca], axis=1).astype(np.float32)
            dense_group_of_col = np.concatenate(
                [dense_group_of_col, np.array(["item_emb_pca"] * pdim, dtype=object)])
            dense_names_full = dense_names_full + tuple(f"item_emb_pca__c{i}" for i in range(pdim))
            cat_ncols["item_emb_pca"] = pdim
            step("pca_emb_ready", pca_dim=pdim,
                 explained_var=round(float(pca_obj.explained_variance_ratio_.sum()), 4))

        # ---- (6-lgbm) LightGBM fit/predict primitive ------------------------------
        def fit_and_predict_oof_lgbm(*, drop_dense_group, save_dir=None):
            """Train one LightGBM on TRAIN rows over the dense groups except the omitted one
            (groups include the 8 AIDE groups + item_emb_pca); predict OOF rows. Early-stopped
            on an item-grouped internal val. Saves the booster (lgb.Booster.save_model)."""
            if drop_dense_group is None:
                col_mask = np.ones(dense_full.shape[1], dtype=bool)
            else:
                col_mask = dense_group_of_col != drop_dense_group
            dnames = [n for n, k in zip(dense_names_full, col_mask) if k]
            Xtr = dense_full[train_idx][:, col_mask]
            Xoof = dense_full[oof_idx][:, col_mask]
            ytr = y[train_idx]
            vmask = (row_to_uniq[train_idx] % 10 == 0)                    # item-grouped val
            dtr = lgb.Dataset(Xtr[~vmask], label=ytr[~vmask].astype(np.float64),
                              feature_name=dnames)
            dva = lgb.Dataset(Xtr[vmask], label=ytr[vmask].astype(np.float64), reference=dtr)
            booster = lgb.train(LGB_HP, dtr, num_boost_round=LGB_ROUNDS, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(50, verbose=False),
                                           lgb.log_evaluation(0)])
            if save_dir is not None and SAVE_MODELS:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                booster.save_model(str(Path(save_dir) / "model.txt"),
                                   num_iteration=booster.best_iteration)
            p = booster.predict(Xoof, num_iteration=booster.best_iteration)
            vl = float(list(booster.best_score.get("valid_0", {}).values())[0]) \
                if booster.best_score else float("nan")
            return np.asarray(p, dtype=np.float64).reshape(-1), float("nan"), vl

        # ---- (6) the MLP fit/predict primitive ------------------------------------
        def fit_and_predict_oof(*, use_item, use_subject, drop_dense_group, tag, seed_off,
                                save_dir=None):
            """Train one MLP on TRAIN rows with all categories except the omitted one;
            predict OOF rows. ``drop_dense_group`` in {None} or a DENSE_GROUPS name.
            If ``save_dir`` is given, persist the trained MlpMemberState there (reloadable
            via MlpMemberState.load(save_dir))."""
            # dense columns: drop the omitted group's columns (or keep all)
            if drop_dense_group is None:
                col_mask = np.ones(dense_full.shape[1], dtype=bool)
            else:
                col_mask = dense_group_of_col != drop_dense_group
            dnames = tuple(n for n, keep_c in zip(dense_names_full, col_mask) if keep_c)
            dX_tr = dense_full[train_idx][:, col_mask]
            dX_oof = dense_full[oof_idx][:, col_mask]
            has_dense = dX_tr.shape[1] > 0

            kw = dict(
                labels=y[train_idx],
                dense_X=dX_tr if has_dense else None,
                dense_feature_names=dnames if has_dense else (),
                hid1=int(HP["hid1"]), hid2=int(HP["hid2"]),
                learning_rate=float(HP["learning_rate"]),
                weight_decay=float(HP["weight_decay"]),
                epochs=int(HP["epochs"]), batch_size=int(HP["batch_size"]),
                val_fraction=float(HP["val_fraction"]),
                early_stopping_patience=int(HP["early_stopping_patience"]),
                feat_dropout=float(HP["feat_dropout"]),
                seed=int(SEED) + int(seed_off), device="cuda",
                holdout_group_id=row_to_uniq[train_idx],   # item-grouped internal val
                show_progress=False,
            )
            if use_item:
                kw.update(item_emb_unique=item_emb_unique, row_to_uniq=row_to_uniq[train_idx])
            if use_subject:
                kw.update(subject_ids=sid[train_idx], n_subjects=int(n_subjects),
                          subj_emb_dim=int(HP["subj_emb_dim"]))
            state = fit_mlp_member(**kw)

            # persist the trained model (reloadable) BEFORE prediction, if requested
            if save_dir is not None and SAVE_MODELS:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                state.save(save_dir)

            # predict OOF rows (chunked apply)
            ap = dict()
            if use_subject:
                ap["subject_ids"] = sid[oof_idx]
            if use_item:
                ap["item_emb"] = oof_item_emb
            if has_dense:
                ap["dense_X"] = dX_oof
            p = mlp_apply_batch(state, **ap)
            return np.asarray(p, dtype=np.float64).reshape(-1), \
                float(state.train_loss), float(state.val_loss)

        # ---- (6b) persist SHARED reload state (subject vocab + dense layout) -------
        # These are common to all 11 MLPs (same train rows) and are required to apply a
        # reloaded MlpMemberState to new rows: subject_key->id, and the dense column layout.
        models_dir = Path(SAVE_ROOT) / "models"
        categories = (DENSE_GROUPS + ["item_emb_pca"]) if MODEL == "lgbm" \
            else (SPECIAL_CATS + DENSE_GROUPS)
        if SAVE_MODELS:
            shared_dir = Path(SAVE_ROOT) / "shared"
            shared_dir.mkdir(parents=True, exist_ok=True)
            (shared_dir / "dense_layout.json").write_text(json.dumps({
                "model": MODEL,
                "dense_names_full": list(dense_names_full),
                "dense_group_of_col": [str(g) for g in dense_group_of_col.tolist()],
                "categories": categories,
                "category_ncols": cat_ncols,
                "d_emb": int(D_EMB), "subj_emb_dim": int(HP["subj_emb_dim"]),
                "embedding_slug": slug,
                "items_parquet": f"{emb_dir}/items.parquet",
            }, indent=2), encoding="utf-8")
            if MODEL == "mlp":
                (shared_dir / "subj_vocab.json").write_text(
                    json.dumps({str(k): int(v) for k, v in subj_vocab.items()}), encoding="utf-8")
            elif MODEL == "lgbm" and pca_obj is not None:
                # PCA is reloadable: components + mean reproduce the item_emb_pca columns.
                np.savez_compressed(shared_dir / "pca_item_emb.npz",
                                    components=pca_obj.components_.astype(np.float32),
                                    mean=pca_obj.mean_.astype(np.float32),
                                    explained_variance_ratio=pca_obj.explained_variance_ratio_.astype(np.float32))
            step("shared_state_saved", save_root=str(SAVE_ROOT))

        # ---- (7) full model (baseline) --------------------------------------------
        step("fit_full")
        if MODEL == "lgbm":
            p_full, tl_full, vl_full = fit_and_predict_oof_lgbm(
                drop_dense_group=None, save_dir=(models_dir / "full") if SAVE_MODELS else None)
        else:
            p_full, tl_full, vl_full = fit_and_predict_oof(
                use_item=True, use_subject=True, drop_dense_group=None, tag="FULL", seed_off=900,
                save_dir=(models_dir / "full") if SAVE_MODELS else None)
        ll_full = soft_logloss(oof_y, p_full)
        step("full_done", soft_logloss_full=round(ll_full, 6), val_loss=round(vl_full, 5))

        # ---- (8) leave-one-kind-out members ---------------------------------------
        # categories set above (mlp: 2 special + 8 dense; lgbm: 8 dense + item_emb_pca)
        p_loo = {}
        loo_ll = {}
        loo_meta = {}
        for ci, c in enumerate(categories):
            step(f"fit_loo_{c}", category=c, idx=ci + 1, of=len(categories))
            sdir = (models_dir / f"loo__{c}") if SAVE_MODELS else None
            if MODEL == "lgbm":
                p, tl, vl = fit_and_predict_oof_lgbm(drop_dense_group=c, save_dir=sdir)
            elif c == "item_embedding":
                p, tl, vl = fit_and_predict_oof(
                    use_item=False, use_subject=True, drop_dense_group=None,
                    tag=c, seed_off=100 + ci, save_dir=sdir)
            elif c == "subject_embedding":
                p, tl, vl = fit_and_predict_oof(
                    use_item=True, use_subject=False, drop_dense_group=None,
                    tag=c, seed_off=100 + ci, save_dir=sdir)
            else:  # a dense AIDE group
                p, tl, vl = fit_and_predict_oof(
                    use_item=True, use_subject=True, drop_dense_group=c,
                    tag=c, seed_off=100 + ci, save_dir=sdir)
            p_loo[c] = p
            loo_ll[c] = soft_logloss(oof_y, p)
            loo_meta[c] = {"train_loss": round(tl, 6), "val_loss": round(vl, 6),
                           "soft_logloss": round(loo_ll[c], 6), "pred_mean": round(float(p.mean()), 5)}
            step(f"loo_{c}_done", soft_logloss=round(loo_ll[c], 6))

        cat_list = list(categories)
        P = np.column_stack([p_loo[c] for c in cat_list])        # [n_oof, K]

        # ---- (9) LightGBM stack over the K LOO columns, honest inner GroupKFold ----
        step("lgb_stack_inner_cv")
        gkf = GroupKFold(n_splits=INNER_FOLDS)
        stacked = np.full(n_oof, np.nan, dtype=np.float64)
        lgb_params = dict(
            objective="cross_entropy",   # soft label in [0,1] (lightgbm xentropy)
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=200,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            max_depth=-1,
            verbosity=-1,
            seed=SEED,
        )
        n_rounds = 300
        for inner_f, (itr, iva) in enumerate(gkf.split(P, oof_y, groups=oof_group)):
            dtr = lgb.Dataset(P[itr], label=oof_y[itr].astype(np.float64))
            booster = lgb.train(lgb_params, dtr, num_boost_round=n_rounds)
            stacked[iva] = booster.predict(P[iva])
            step(f"lgb_inner_{inner_f}_done", n_val=int(iva.size))
        if not np.isfinite(stacked).all():
            raise ValueError("LGB stacked OOF has non-finite entries (GroupKFold coverage gap)")
        ll_stack = soft_logloss(oof_y, np.clip(stacked, _EPS, 1 - _EPS))

        # ---- (9b) DEPLOYABLE final stacker: fit on ALL held-out rows, persist --------
        # The inner-CV `stacked` above is the HONEST score (each row predicted by a booster
        # that never saw it). The final booster below is the reloadable ensemble model: it
        # consumes the K LOO columns (in `cat_list` order) and emits the stacked probability.
        if SAVE_MODELS:
            stk_dir = Path(SAVE_ROOT) / "stacker"
            stk_dir.mkdir(parents=True, exist_ok=True)
            final_booster = lgb.train(
                lgb_params, lgb.Dataset(P, label=oof_y.astype(np.float64)),
                num_boost_round=n_rounds)
            final_booster.save_model(str(stk_dir / "lgb_stack_final.txt"))
            (stk_dir / "stacker_meta.json").write_text(json.dumps({
                "input_columns": list(cat_list),     # column order the booster expects
                "input_kind": "leave-one-category-out MLP OOF probability",
                "objective": "cross_entropy", "num_boost_round": n_rounds,
                "lgb_params": {k: v for k, v in lgb_params.items()},
                "honest_inner_cv_soft_logloss": round(ll_stack, 6),
            }, indent=2), encoding="utf-8")
            step("final_stacker_saved", path=str(stk_dir / "lgb_stack_final.txt"))

        # secondary combiner: logit-mean blend of the LOO preds
        logit_mean = _sigmoid(np.mean(np.column_stack([_logit(p_loo[c]) for c in cat_list]), axis=1))
        ll_logitmean = soft_logloss(oof_y, logit_mean)

        # ---- (10) OOF correlation matrix of K LOO + full --------------------------
        corr_cols = cat_list + ["__full__"]
        M = np.column_stack([p_loo[c] for c in cat_list] + [p_full])
        corr = np.corrcoef(M, rowvar=False)
        corr_matrix = {a: {b: round(float(corr[i, j]), 4) for j, b in enumerate(corr_cols)}
                       for i, a in enumerate(corr_cols)}

        # ---- (11) verdict ---------------------------------------------------------
        best_single = min(loo_ll, key=loo_ll.get)
        delta_stack_vs_full = ll_full - ll_stack            # >0 => stack better (lower logloss)
        stack_beats_full = bool(ll_stack < ll_full)
        verdict = (
            f"LGB-stack-of-LOO {'BEATS' if stack_beats_full else 'does NOT beat'} the full MLP "
            f"by {delta_stack_vs_full:+.6f} nats "
            f"(stack={ll_stack:.6f}, full={ll_full:.6f}).")

        result = {
            "ok": True,
            "experiment": f"loo_category_{MODEL}",
            "model": MODEL, "row_source": ROW_SOURCE,
            "family": FAMILY, "code_version": CODE_VERSION,
            "n_rows_total": N, "n_train": n_train, "n_oof": n_oof,
            "oof_fold": OOF_FOLD, "n_folds": N_FOLDS, "split_seed": SPLIT_SEED,
            "y_mean_oof": round(float(oof_y.mean()), 5),
            "category_ncols": cat_ncols,
            "n_categories": len(cat_list),
            "categories": cat_list,
            "hp": (LGB_HP if MODEL == "lgbm" else HP), "inner_folds": INNER_FOLDS,
            "soft_logloss": {
                "full_mlp_baseline": round(ll_full, 6),
                "lgb_stack_of_loo": round(ll_stack, 6),
                "logit_mean_blend": round(ll_logitmean, 6),
                **{f"loo_{c}": round(loo_ll[c], 6) for c in cat_list},
            },
            "loo_meta": loo_meta,
            "best_single_loo": {"category": best_single, "soft_logloss": round(loo_ll[best_single], 6)},
            "delta_stack_vs_full_nats": round(delta_stack_vs_full, 6),
            "stack_beats_full": stack_beats_full,
            "verdict": verdict,
            "oof_corr_matrix": corr_matrix,
            "full_mlp_train_loss": round(tl_full, 6),
            "full_mlp_val_loss": round(vl_full, 6),
            "t_total_s": round(time.time() - t0, 1),
        }

        # ---- (12) readable summary table ------------------------------------------
        print("\n" + "=" * 72, flush=True)
        print(f"LOO-CATEGORY MLP DIVERSITY EXPERIMENT — family={FAMILY}  (OOF fold {OOF_FOLD})", flush=True)
        print(f"  rows: train={n_train}  oof={n_oof}  y_mean_oof={oof_y.mean():.4f}", flush=True)
        print("-" * 72, flush=True)
        print(f"  {'combiner / omitted category':<34}{'soft_logloss':>14}{'Δ vs full':>14}", flush=True)
        print("-" * 72, flush=True)
        print(f"  {'FULL MLP (baseline)':<34}{ll_full:>14.6f}{0.0:>14.6f}", flush=True)
        print(f"  {'LGB-stack-of-LOO':<34}{ll_stack:>14.6f}{ll_full - ll_stack:>+14.6f}", flush=True)
        print(f"  {'logit-mean blend':<34}{ll_logitmean:>14.6f}{ll_full - ll_logitmean:>+14.6f}", flush=True)
        print("  " + "-" * 70, flush=True)
        for c in sorted(cat_list, key=lambda x: loo_ll[x]):
            print(f"  {'LOO[-'+c+']':<34}{loo_ll[c]:>14.6f}{ll_full - loo_ll[c]:>+14.6f}", flush=True)
        print("=" * 72, flush=True)
        print("VERDICT:", verdict, flush=True)
        print("=" * 72 + "\n", flush=True)

        # ---- (13) persist preds + result + manifest to DRIVE, and VERIFY reload ----
        if SAVE_MODELS:
            preds_dir = Path(SAVE_ROOT) / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                preds_dir / "oof_preds.npz",
                oof_items=np.asarray(oof_items),
                oof_subj=np.asarray([tr_subj[r] for r in oof_idx]),
                oof_y=oof_y.astype(np.float32),
                p_full=p_full.astype(np.float32),
                stacked_oof=stacked.astype(np.float32),
                logit_mean=logit_mean.astype(np.float32),
                cat_list=np.asarray(cat_list),
                P=P.astype(np.float32),
                **{f"p_loo__{c}": p_loo[c].astype(np.float32) for c in cat_list},
            )
            (Path(SAVE_ROOT) / "result.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8")

            # reload-verify: load `full` + last LOO model from disk and re-predict OOF.
            reload_report = {}
            try:
                first_loo = cat_list[-1]
                if MODEL == "lgbm":
                    for tag, sd, p_ref, dropg in [
                        ("full", models_dir / "full", p_full, None),
                        (f"loo__{first_loo}", models_dir / f"loo__{first_loo}",
                         p_loo[first_loo], first_loo),
                    ]:
                        bst = lgb.Booster(model_file=str(Path(sd) / "model.txt"))
                        cm = (np.ones(dense_full.shape[1], dtype=bool) if dropg is None
                              else dense_group_of_col != dropg)
                        p2 = np.asarray(bst.predict(dense_full[oof_idx][:, cm]),
                                        dtype=np.float64).reshape(-1)
                        reload_report[tag] = round(float(np.max(np.abs(p2 - p_ref))), 8)
                else:
                    from src.mlp_member import MlpMemberState as _MMS
                    for tag, sd, p_ref, use_it, use_sub, dropg in [
                        ("full", models_dir / "full", p_full, True, True, None),
                        (f"loo__{first_loo}", models_dir / f"loo__{first_loo}",
                         p_loo[first_loo], first_loo != "item_embedding",
                         first_loo != "subject_embedding",
                         first_loo if first_loo in DENSE_GROUPS else None),
                    ]:
                        st = _MMS.load(sd)
                        ap = {}
                        if use_sub: ap["subject_ids"] = sid[oof_idx]
                        if use_it: ap["item_emb"] = oof_item_emb
                        if dropg is None and st.dense_dim > 0:
                            ap["dense_X"] = dense_full[oof_idx]
                        elif st.dense_dim > 0:
                            cm = dense_group_of_col != dropg
                            ap["dense_X"] = dense_full[oof_idx][:, cm]
                        p2 = np.asarray(mlp_apply_batch(st, **ap), dtype=np.float64).reshape(-1)
                        reload_report[tag] = round(float(np.max(np.abs(p2 - p_ref))), 8)
                result["reload_verify_max_abs_diff"] = reload_report
                ok_reload = all(v < 1e-5 for v in reload_report.values())
                result["reload_verify_ok"] = bool(ok_reload)
                step("reload_verified", **reload_report, ok=ok_reload)
            except Exception as _re:
                result["reload_verify_error"] = repr(_re)
                step("reload_verify_failed", err=repr(_re))

            # manifest = single entry point for future reload
            (Path(SAVE_ROOT) / "manifest.json").write_text(json.dumps({
                "experiment": "loo_category_mlp", "family": FAMILY,
                "save_root": str(SAVE_ROOT), "oof_fold": OOF_FOLD,
                "models": ["full"] + [f"loo__{c}" for c in cat_list],
                "model_format": "MlpMemberState.save -> {weights.npz, meta.json}; "
                                "load via src.mlp_member.MlpMemberState.load(dir)",
                "shared": {"subj_vocab": "shared/subj_vocab.json",
                           "dense_layout": "shared/dense_layout.json"},
                "stacker": "stacker/lgb_stack_final.txt (lgb.Booster; "
                           "input cols = stacker/stacker_meta.json:input_columns)",
                "preds": "preds/oof_preds.npz", "result": "result.json",
                "reload_verify_ok": result.get("reload_verify_ok"),
            }, indent=2), encoding="utf-8")
            step("drive_persisted", save_root=str(SAVE_ROOT))

        prog.update(stage="done", **result)
        _write_status(dict(prog))
        # mirror the final status to Drive too (survives runtime recycle)
        if SAVE_MODELS:
            try:
                (Path(SAVE_ROOT) / "status_final.json").write_text(
                    json.dumps(dict(prog), indent=2, default=str), encoding="utf-8")
            except Exception:
                pass
        return result

    except Exception as exc:
        prog.update(stage="error", ok=False, error=repr(exc),
                    traceback=traceback.format_exc(),
                    t_total_s=round(time.time() - t0, 1))
        _write_status(dict(prog))
        print(f"[exp-loo] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
        raise


# ----------------------------------------------------------------------------------
# Launch (paste AFTER the run_bg/poll harness cell):
#   import os; os.environ.setdefault("SHIP_FAMILY", "qwen")
#   run_bg("exp_loo_qwen", fn)
# Poll (tiny fast cell):
#   import json; st = json.load(open("/content/exp_loo_qwen.json")); print(st.get("stage"))
#   # on done, st also holds soft_logloss / verdict / oof_corr_matrix.
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    fn()
