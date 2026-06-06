"""AIDE-style single-model MLP trainer for ONE embedding family (qwen | nemotron | lgai).

SHIP_PLAN_3WAY §CORRECTION (2026-06-06): the AIDE-feature path IS viable. The submission
computes the 23-dim NN features LIVE at predict time from a shipped train-neighbor index
(``nn_infra_prep`` + ``nn23_runtime``). So we train on REAL AIDE feats and the holdout is
scorable because its feats are computed from the prepped train index, NOT pre-derived
shards.

WHAT THIS TRAINER DOES (per family, SHIP_FAMILY = qwen|nemotron|lgai)
  (1) ensure the per-family neighbor infra is prepped via ``nn_infra_prep.prep_nn_infra``
      (or loaded from DR/ship/nn_infra/<fam>/). Builds TWO index subsets:
        * 'all' (index over folds {0,1,2})  -> shipped-train + HOLDOUT features
        * 'f01' (index over folds {0,1})     -> honest-VAL: fold2 + folds{0,1}-train feats
      ('f01' keeps fold2 items OUT of the index so fold2 rows cannot self-retrieve.)
  (2) compute dense_X = AIDE 23-dim NN feats + centroid/cluster distance block + metadata
      (subject org/family/macro + benchmark topic/age numerics) via
      ``nn_infra_prep.compute_dense_block`` for: folds{0,1} train rows (f01 infra), fold2
      val rows (f01 infra), and the 135,650 holdout rows (all infra). The column order is
      LOCKED by nn_infra_prep and recorded in meta.json's dense_feature_names.
  (3) fit ONE ``src.mlp_member.fit_mlp_member`` on folds{0,1}: dense_X = AIDE feats,
      item_emb (gathered per row via row_to_uniq) + a learned subject nn.Embedding.
      Predict fold2 (honest item-disjoint val for stacking) + holdout.
  (4) OPTIONAL second fit on ALL train rows (folds {0,1,2}, 'all' infra feats) for the
      SHIPPED model. The fold2 valpred ALWAYS comes from the folds{0,1}-only model.
  (5) save to DR/ship/diverse/<fam>/: mlp_state.pkl, valpred_fold2.npy, holdpred.npy,
      meta.json (incl. fold2 soft-logloss + the locked dense_feature_names).

Pasteable into a SINGLE Colab cell, launched via the project ``run_bg(name, fn)`` harness.
``fn()`` writes progress + result JSON to ``/content/<fam>_aide.json`` and returns the
result dict. Poll with a tiny fast cell reading that file.

=================================================================================
HOW THE SHIPPED RUNTIME RECOMPUTES THE SAME dense_X (parity contract)
=================================================================================
The submodel bundle ships DR/ship/nn_infra/<fam>/all/ (index + subject CSR + conditional
bag + centroids + benchmark_to_id) and nn23_runtime.py. At predict time, per row, it:
  * builds its _TrainingItemCache from training_index.faiss + subject_passrate.npz,
  * calls nn23_runtime.compute_nn_features_23(cache, item_emb, subject_id, cond_ctx=...,
    query_benchmark_id=..., query_cluster_id=..., k=NN_RUNTIME_K) -> the SAME 23 dims
    (both train and runtime bottom out on _aggregate_nn_features),
  * recomputes the centroid-distance columns from centroids.npy (deterministic argmin)
    and the metadata columns from the shipped CSVs with the LOCKED column order from
    meta.json::dense_feature_names,
  * concatenates [nn23 | centroid | metadata] and runs mlp_member.apply_batch.
Train uses the BATCH path (compute_nn_features_streaming); runtime uses the per-item path;
the numerics are identical. The only knob to keep in sync is NN_RUNTIME_K == HP K (16).

=================================================================================
FALLBACK (kept as a flag, DEFAULT OFF): metadata-only dense_X
  When USE_AIDE_FEATS=0 the trainer skips the NN infra entirely and builds dense_X from
  the leak-free metadata content-join ONLY (subject org/family/macro one-hots + benchmark
  topic one-hot + age numeric). Holdout-derivable, but strictly weaker than the AIDE path.
  This is the old behavior, retained only as a safety net. DEFAULT is the AIDE path.
=================================================================================
ASSUMPTIONS (verify before trusting outputs)
  A1. DR/ship/rows/{_tr_item,_tr_subj,_ho_item,_ho_subj}.npy hold PER-ROW string keys
      (object/<U), length 264350 (train) / 135650 (holdout), canonical order shared by
      every existing per-row pred. Asserted. (ROWS_ARE_KEYS=False handles int-index.)
  A2. Labels in DR/prepared_datasets/*measurement_db_prepared*.parquet keyed by
      (subject_key,item_key), continuous label in [0,1] (mean ~0.677). Abort if any train
      row lacks a label.
  A3. nf3 fold of a TRAIN row = aide.hygiene.manifest.item_fold(item_key, 3, 0): folds
      {0,1}=train, fold2=honest item-disjoint val. Holdout items are a separate universe.
  A4. Holdout subjects are SEEN in train (item cold-start). Subject vocab is built over
      train u holdout (in nn_infra_prep) so CSR/context/learned-embedding cover both.
  A5. Family alias -> driver family: nemotron->llama, lgai->mistral, qwen->qwen.
  A6. NN_RUNTIME_K (submodel) MUST equal HP K (16). Recorded in the infra + meta.json.
  A7. fit_mlp_member uses BCEWithLogitsLoss with soft targets in [0,1] -- correct for the
      continuous pass-probability label (matches the competition soft-logloss metric).
=================================================================================
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

# ----------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------
FAMILY = os.environ.get("SHIP_FAMILY", "qwen").strip().lower()  # 'qwen'|'nemotron'|'lgai'

REPO_ROOT = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
DRIVE_ROOT = os.environ.get(
    "SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")

N_TRAIN_EXPECTED = 264350
N_HOLD_EXPECTED = 135650

N_FOLDS = 3        # nf3 (A3)
SPLIT_SEED = 0     # nf3 seed (A3)
VAL_FOLD = 2       # fold2 = honest val; folds{0,1} = train

ROWS_ARE_KEYS = True  # flip to False if rows .npy are int indices (A1)

# --- AIDE-feature toggle. ENABLED by default (the CORRECTION path). ----------------
# True  -> dense_X = 23-dim AIDE NN feats + centroid + metadata, holdout scored LIVE from
#          the prepped train index (nn_infra_prep). This is the recipe.
# False -> metadata-only fallback dense_X (leak-free content join). Strictly weaker.
USE_AIDE_FEATS = bool(int(os.environ.get("SHIP_USE_AIDE_FEATS", "1")))

# Refit on ALL train rows (folds{0,1,2}) for the SHIPPED model after the val run.
REFIT_ON_ALL = bool(int(os.environ.get("SHIP_REFIT_ALL", "1")))

# Reuse the existing qwen NN infra (DR/artifacts/nn_features + cluster_centroids) when
# fam==qwen instead of rebuilding. (nemotron/lgai always build fresh in their emb space.)
REUSE_QWEN = bool(int(os.environ.get("SHIP_REUSE_QWEN", "1")))

NN_K = int(os.environ.get("SHIP_NN_K", "16"))           # A6: must == submodel NN_RUNTIME_K
KMEANS_K = int(os.environ.get("SHIP_KMEANS_K", "64"))
TOP_M_CENTROIDS = int(os.environ.get("SHIP_TOP_M", "4"))

FAM_ALIAS = {"qwen": "qwen", "nemotron": "llama", "llama": "llama",
             "lgai": "mistral", "mistral": "mistral"}

# ==================================================================================
# REAL fit_mlp_member KWARGS (verbatim signature, src/mlp_member.py::fit_mlp_member):
#   fit_mlp_member(*, labels, subject_ids=None, n_subjects=0, subj_emb_dim=0,
#                  item_emb_unique=None, row_to_uniq=None, dense_X=None,
#                  dense_feature_names=(), hid1=128, hid2=64, learning_rate=1e-3,
#                  weight_decay=1e-5, epochs=40, batch_size=16384, val_fraction=0.1,
#                  early_stopping_patience=6, warmup_epochs=2, use_cosine_schedule=True,
#                  feat_dropout=0.10, seed=0, device=None, holdout_group_id=None,
#                  show_progress=True, ncl_anchor_preds=None, ncl_lambda=0.0)
#   -> MlpMemberState. Input channel order is [subj_emb | item_emb | dense_X]; we use all
#   three. holdout_group_id=row_to_uniq -> item-grouped internal val (no item leak).
#   apply: src/mlp_member.py::apply_batch(state, subject_ids=, item_emb=[N,D], dense_X=).
# ==================================================================================
HP = dict(
    subj_emb_dim=32,
    hid1=256,
    hid2=128,
    learning_rate=1.0e-3,
    weight_decay=1.0e-5,
    epochs=40,
    batch_size=16384,
    early_stopping_patience=6,
    feat_dropout=0.10,
    val_fraction=0.1,
)
SEED = 0

STATUS_PATH_TMPL = "/content/{fam}_aide.json"


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _write_status(fam, payload):
    try:
        with open(STATUS_PATH_TMPL.format(fam=fam), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception:
        pass


def _load_keys_npy(path):
    arr = np.load(path, allow_pickle=True).reshape(-1)
    return [str(x) for x in arr.tolist()]


def _onehot(values, vocab):
    idx = {v: j for j, v in enumerate(vocab)}
    unk = idx.get("UNK", len(vocab) - 1)
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        out[r, idx.get(str(v), unk)] = 1.0
    return out


# ----------------------------------------------------------------------------------
# the run_bg-friendly entry point
# ----------------------------------------------------------------------------------
def fn():
    fam = FAMILY
    t0 = time.time()
    prog = {"family": fam, "stage": "start", "ok": None, "t_start": t0,
            "feature_path": "AIDE_feats" if USE_AIDE_FEATS else "fallback_emb+metadata"}
    _write_status(fam, prog)

    def step(stage, **extra):
        prog.update(stage=stage, t_elapsed=round(time.time() - t0, 1), **extra)
        _write_status(fam, dict(prog))
        print(f"[{fam}-aide] {stage}  (+{prog['t_elapsed']}s)", flush=True)

    try:
        # ---- (0) sys.path + imports ------------------------------------------------
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        step("imports")
        from aide.features.driver import FAMILY_SLUG, load_embeddings
        from aide.hygiene.manifest import item_fold
        from src.mlp_member import apply_batch as mlp_apply_batch
        from src.mlp_member import fit_mlp_member
        # the USE_AIDE_FEATS engine (this swarm's deliverable):
        from scripts.ship import nn_infra_prep as NIP

        if fam not in FAM_ALIAS:
            raise ValueError(f"unknown FAMILY={fam!r}; expected one of {list(FAM_ALIAS)}")
        driver_fam = FAM_ALIAS[fam]
        slug = FAMILY_SLUG[driver_fam]
        emb_dir = f"{DRIVE_ROOT}/embeddings/{slug}"
        step("config", driver_family=driver_fam, slug=slug,
             use_aide_feats=USE_AIDE_FEATS, refit_on_all=REFIT_ON_ALL, nn_k=NN_K)

        # ---- (1) embeddings + lookup ----------------------------------------------
        item_keys, item_emb = load_embeddings(f"{emb_dir}/items.parquet")
        item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        D_EMB = int(item_emb.shape[1])
        item_emb_idx = {str(k): i for i, k in enumerate(item_keys)}
        emb_lookup = {str(k): item_emb[i] for i, k in enumerate(item_keys)}
        step("loaded_embeddings", n_items=len(item_keys), d_emb=D_EMB)

        # ---- (2) labels (continuous) + canonical rows ------------------------------
        import pandas as pd
        db = glob.glob(f"{DRIVE_ROOT}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
        labels_df = pd.read_parquet(db, columns=["subject_key", "item_key", "label"])
        labels_df["subject_key"] = labels_df["subject_key"].astype(str)
        labels_df["item_key"] = labels_df["item_key"].astype(str)
        if labels_df.duplicated(["subject_key", "item_key"]).any():
            # collapse to the per-(subj,item) mean continuous label
            labels_df = (labels_df.groupby(["subject_key", "item_key"], sort=False)["label"]
                         .mean().reset_index())
        label_map = {(s, i): float(l) for s, i, l in zip(
            labels_df["subject_key"].to_numpy(), labels_df["item_key"].to_numpy(),
            labels_df["label"].to_numpy())}

        rows_dir = f"{DRIVE_ROOT}/ship/rows"
        if ROWS_ARE_KEYS:
            tr_item = _load_keys_npy(f"{rows_dir}/_tr_item.npy")
            tr_subj = _load_keys_npy(f"{rows_dir}/_tr_subj.npy")
            ho_item = _load_keys_npy(f"{rows_dir}/_ho_item.npy")
            ho_subj = _load_keys_npy(f"{rows_dir}/_ho_subj.npy")
        else:
            subj_keys_emb, _ = load_embeddings(f"{emb_dir}/subjects.parquet")
            tr_item = [item_keys[int(j)] for j in np.load(f"{rows_dir}/_tr_item.npy").reshape(-1)]
            ho_item = [item_keys[int(j)] for j in np.load(f"{rows_dir}/_ho_item.npy").reshape(-1)]
            tr_subj = [subj_keys_emb[int(j)] for j in np.load(f"{rows_dir}/_tr_subj.npy").reshape(-1)]
            ho_subj = [subj_keys_emb[int(j)] for j in np.load(f"{rows_dir}/_ho_subj.npy").reshape(-1)]

        N_tr, N_ho = len(tr_item), len(ho_item)
        if N_tr != N_TRAIN_EXPECTED or N_ho != N_HOLD_EXPECTED:
            raise ValueError(f"row-count mismatch: train {N_tr} (exp {N_TRAIN_EXPECTED}), "
                             f"holdout {N_ho} (exp {N_HOLD_EXPECTED}) (A1)")
        if not (len(tr_subj) == N_tr and len(ho_subj) == N_ho):
            raise ValueError("item/subject row arrays mismatched lengths (A1)")
        step("loaded_rows", n_train=N_tr, n_holdout=N_ho)

        # ---- (2b) continuous y_train ----------------------------------------------
        y_train = np.empty(N_tr, dtype=np.float32)
        missing = 0
        for r, (s, i) in enumerate(zip(tr_subj, tr_item)):
            v = label_map.get((s, i))
            if v is None:
                missing += 1
                y_train[r] = 0.0
            else:
                y_train[r] = float(v)
        if missing:
            raise ValueError(f"{missing} train rows have no label (A2) — aborting")
        step("built_labels", y_mean=round(float(y_train.mean()), 4))

        # ---- (2c) nf3 fold of each train row --------------------------------------
        tr_fold = np.fromiter((item_fold(i, N_FOLDS, SPLIT_SEED) for i in tr_item),
                              dtype=np.int64, count=N_tr)
        f01_mask = tr_fold != VAL_FOLD
        f2_mask = tr_fold == VAL_FOLD
        n_f01, n_f2 = int(f01_mask.sum()), int(f2_mask.sum())
        if n_f01 == 0 or n_f2 == 0:
            raise ValueError(f"degenerate nf3 split: folds01={n_f01}, fold2={n_f2}")
        f01_idx = np.where(f01_mask)[0]
        f2_idx = np.where(f2_mask)[0]
        step("nf3_split", n_folds01=n_f01, n_fold2=n_f2, fold2_frac=round(n_f2 / N_tr, 4))

        # ---- (3) item_emb_unique + row_to_uniq (over ALL train items) -------------
        uniq_keys, uniq_pos = [], {}
        row_to_uniq = np.empty(N_tr, dtype=np.int64)
        for r, k in enumerate(tr_item):
            j = uniq_pos.get(k)
            if j is None:
                j = len(uniq_keys)
                uniq_pos[k] = j
                uniq_keys.append(k)
            row_to_uniq[r] = j
        item_emb_unique = np.empty((len(uniq_keys), D_EMB), dtype=np.float32)
        for j, k in enumerate(uniq_keys):
            item_emb_unique[j] = item_emb[item_emb_idx[str(k)]]

        # subject vocab over train u holdout (A4) for the LEARNED nn.Embedding
        subj_vocab = {}
        for s in tr_subj:
            subj_vocab.setdefault(s, len(subj_vocab))
        for s in ho_subj:
            subj_vocab.setdefault(s, len(subj_vocab))
        n_subjects = len(subj_vocab)
        train_sid = np.fromiter((subj_vocab[s] for s in tr_subj), dtype=np.int64, count=N_tr)
        hold_sid = np.fromiter((subj_vocab[s] for s in ho_subj), dtype=np.int64, count=N_ho)

        # ---- (4) dense_X: AIDE feats (preferred) OR metadata fallback --------------
        dense_train = np.empty((N_tr, 0), dtype=np.float32)  # filled below by mask
        dense_hold = None
        dense_names: tuple = ()
        meta_cov = None

        if USE_AIDE_FEATS:
            # ---- prep the two index subsets (idempotent; loads cache if present) ----
            step("prep_infra_all")
            all_subjects = list(subj_vocab.keys())
            # 'all' index = ALL train items; 'f01' index = folds{0,1} train items
            tr_df = pd.DataFrame({"subject_key": tr_subj, "item_key": tr_item,
                                  "label": y_train})
            all_items = list(dict.fromkeys(tr_item))                 # all train items
            f01_items = list(dict.fromkeys([tr_item[r] for r in f01_idx]))  # f01 train items
            infra_all = NIP.prep_nn_infra(
                fam, index_item_keys=all_items, index_subject_keys=all_subjects,
                all_subject_keys=all_subjects,
                train_df=tr_df, subset="all", k=NN_K, kmeans_k=KMEANS_K,
                reuse_qwen_artifacts=REUSE_QWEN, prepared_parquet=db)
            step("prep_infra_f01")
            tr_df_f01 = tr_df.iloc[f01_idx].reset_index(drop=True)
            infra_f01 = NIP.prep_nn_infra(
                fam, index_item_keys=f01_items, index_subject_keys=all_subjects,
                all_subject_keys=all_subjects,
                train_df=tr_df_f01, subset="f01", k=NN_K, kmeans_k=KMEANS_K,
                reuse_qwen_artifacts=False, prepared_parquet=db)

            # ---- metadata block (leak-free; fit on TRAIN, encode train+holdout) ----
            step("build_metadata_tables")
            from scripts.ship.metadata_tables import build_metadata_tables
            md = build_metadata_tables(
                rows_subj=tr_subj, rows_item=tr_item,
                rows_subj2=ho_subj, rows_item2=ho_item,
                prepared_parquet_path=db,
                model_info_path=f"{REPO_ROOT}/data/metadata/model_info.csv",
                benchmark_info_path=f"{REPO_ROOT}/data/metadata/benchmark_info.csv",
                item_clusters_path=None, marginals=None, verbose=False)
            md_report = md.get("report", {})
            md_n_first = int(md.get("n_first", N_tr))
            meta_cov = md_report.get("coverage", {})

            # ---- dense_X for fold2 + f01-train (f01 infra) -------------------------
            step("dense_f01_train")
            d_f01, dense_names = NIP.compute_dense_block(
                infra_f01, [tr_item[r] for r in f01_idx], [tr_subj[r] for r in f01_idx],
                item_emb_lookup=emb_lookup, metadata_out=md,
                metadata_slice=f01_idx, top_m_centroids=TOP_M_CENTROIDS)
            step("dense_fold2")
            d_f2, _ = NIP.compute_dense_block(
                infra_f01, [tr_item[r] for r in f2_idx], [tr_subj[r] for r in f2_idx],
                item_emb_lookup=emb_lookup, metadata_out=md,
                metadata_slice=f2_idx, top_m_centroids=TOP_M_CENTROIDS)

            # ---- dense_X for ALL-train + holdout (all infra) -----------------------
            step("dense_all_train")
            d_all, names_all = NIP.compute_dense_block(
                infra_all, tr_item, tr_subj, item_emb_lookup=emb_lookup,
                metadata_out=md, metadata_slice=slice(0, md_n_first),
                top_m_centroids=TOP_M_CENTROIDS)
            step("dense_holdout")
            d_ho, _ = NIP.compute_dense_block(
                infra_all, ho_item, ho_subj, item_emb_lookup=emb_lookup,
                metadata_out=md, metadata_slice=slice(md_n_first, None),
                top_m_centroids=TOP_M_CENTROIDS)
            if not (d_f01.shape[1] == d_f2.shape[1] == d_all.shape[1] == d_ho.shape[1]):
                raise ValueError("dense column count mismatch across f01/f2/all/holdout")
            if names_all != dense_names:
                raise ValueError("dense_feature_names differ between f01 and all infra")

            # canonical-order full-train dense for the ship refit
            F = int(d_all.shape[1])
            dense_train = np.empty((N_tr, F), dtype=np.float32)
            dense_train[:] = d_all  # 'all' infra dense IS already canonical train order
            dense_hold = d_ho
            # the honest-val run uses the f01-infra feats (fold2 cannot self-retrieve):
            dz_f01_run = d_f01           # [n_f01, F] folds{0,1} train rows
            dz_f2_run = d_f2             # [n_f2,  F] fold2 val rows
            step("dense_ready", dense_dim=F, dense_names_head=list(dense_names[:6]))

        else:
            # ---- FALLBACK: metadata-only dense_X (leak-free content join) ----------
            step("dense_metadata_fallback")
            want = ["subject_key", "item_key", "subject_content", "benchmark"]
            lab2 = pd.read_parquet(db, columns=want)
            lab2["subject_key"] = lab2["subject_key"].astype(str)
            lab2["item_key"] = lab2["item_key"].astype(str)
            subj_content_map = dict(zip(lab2.drop_duplicates("subject_key")["subject_key"],
                                        lab2.drop_duplicates("subject_key")["subject_content"]))
            bench_map = {(s, i): b for s, i, b in zip(
                lab2["subject_key"], lab2["item_key"], lab2["benchmark"])}
            from aide.features.metadata import (extract_subject_name, load_metadata,
                                                row_benchmark_meta, row_subject_meta)
            model_info, benchmark_info = load_metadata(REPO_ROOT)

            def _meta_block(subj_rows, item_rows):
                names = [extract_subject_name(subj_content_map.get(s, "")) for s in subj_rows]
                smeta, cov = row_subject_meta(names, model_info)
                benches = [str(bench_map.get((s, i), "UNK")) for s, i in zip(subj_rows, item_rows)]
                bmeta = row_benchmark_meta(benches, benchmark_info)
                return smeta, bmeta, cov

            s_tr, b_tr, cov_tr = _meta_block(tr_subj, tr_item)
            s_ho, b_ho, cov_ho = _meta_block(ho_subj, ho_item)
            meta_cov = {"train_subject_join": round(float(cov_tr), 4),
                        "holdout_subject_join": round(float(cov_ho), 4)}

            def _vocab(vals):
                return sorted(set(str(x) for x in vals) | {"UNK"})
            org_v = _vocab(s_tr["organization"]); fam_v = _vocab(s_tr["family"])
            macro_v = _vocab(s_tr["macro_family"]); topic_v = _vocab(b_tr["topic"])

            def _build(smeta, bmeta):
                age = bmeta["age_bin"].astype(str)
                age_num = np.array([float(a) if a not in ("nan", "-1", "UNK") else -1.0
                                    for a in age], dtype=np.float32).reshape(-1, 1)
                return np.concatenate([_onehot(smeta["organization"], org_v),
                                       _onehot(smeta["family"], fam_v),
                                       _onehot(smeta["macro_family"], macro_v),
                                       _onehot(bmeta["topic"], topic_v), age_num], axis=1)

            dense_train = _build(s_tr, b_tr).astype(np.float32)
            dense_hold = _build(s_ho, b_ho).astype(np.float32)
            dense_names = tuple([f"org__{v}" for v in org_v] + [f"family__{v}" for v in fam_v]
                                + [f"macro__{v}" for v in macro_v]
                                + [f"topic__{v}" for v in topic_v] + ["bench_age_bin"])
            dz_f01_run = dense_train[f01_mask]
            dz_f2_run = dense_train[f2_mask]
            step("dense_ready", dense_dim=int(dense_train.shape[1]), meta_coverage=meta_cov)

        # ---- (5a) honest-val run: fit on folds{0,1}, predict fold2 + holdout -------
        step("fitting_val_run", n_train=n_f01)
        state_val = fit_mlp_member(
            labels=y_train[f01_mask],
            subject_ids=train_sid[f01_mask],
            n_subjects=int(n_subjects),
            subj_emb_dim=int(HP["subj_emb_dim"]),
            item_emb_unique=item_emb_unique,
            row_to_uniq=row_to_uniq[f01_mask],
            dense_X=dz_f01_run,
            dense_feature_names=tuple(dense_names),
            hid1=int(HP["hid1"]), hid2=int(HP["hid2"]),
            learning_rate=float(HP["learning_rate"]),
            weight_decay=float(HP["weight_decay"]),
            epochs=int(HP["epochs"]), batch_size=int(HP["batch_size"]),
            val_fraction=float(HP["val_fraction"]),
            early_stopping_patience=int(HP["early_stopping_patience"]),
            feat_dropout=float(HP["feat_dropout"]),
            seed=int(SEED) + 901, device="cuda",
            holdout_group_id=row_to_uniq[f01_mask],  # item-grouped internal val
            show_progress=True,
        )
        step("fitted_val_run", train_loss=float(state_val.train_loss),
             val_loss=float(state_val.val_loss))

        # chunked apply gathering per-row item embeddings + the row-aligned dense block
        def _apply(state, item_key_rows, sid_rows, dense_rows, chunk=131072):
            keys = np.asarray(item_key_rows).astype(str)
            out = np.empty(int(keys.shape[0]), dtype=np.float32)
            for s in range(0, int(keys.shape[0]), int(chunk)):
                e = min(s + int(chunk), int(keys.shape[0]))
                emb = np.empty((e - s, D_EMB), dtype=np.float32)
                for j, k in enumerate(keys[s:e]):
                    emb[j] = item_emb[item_emb_idx[str(k)]]
                dz = None if dense_rows is None else dense_rows[s:e]
                out[s:e] = mlp_apply_batch(
                    state, subject_ids=sid_rows[s:e], item_emb=emb, dense_X=dz)
            return out

        f2_item = [tr_item[r] for r in f2_idx]
        f2_sid = train_sid[f2_mask]
        val_pred_f2 = _apply(state_val, f2_item, f2_sid, dz_f2_run)
        y_f2 = y_train[f2_mask]
        p = np.clip(val_pred_f2, 1e-6, 1 - 1e-6)
        val_logloss = float(-(y_f2 * np.log(p) + (1 - y_f2) * np.log(1 - p)).mean())
        step("val_predicted", val_logloss=round(val_logloss, 6),
             val_pred_mean=round(float(val_pred_f2.mean()), 5))

        # ---- (5b) shipped model: refit on ALL train rows (all-infra feats) ---------
        if REFIT_ON_ALL:
            step("fitting_ship_run", n_train=N_tr)
            state_ship = fit_mlp_member(
                labels=y_train, subject_ids=train_sid, n_subjects=int(n_subjects),
                subj_emb_dim=int(HP["subj_emb_dim"]),
                item_emb_unique=item_emb_unique, row_to_uniq=row_to_uniq,
                dense_X=dense_train, dense_feature_names=tuple(dense_names),
                hid1=int(HP["hid1"]), hid2=int(HP["hid2"]),
                learning_rate=float(HP["learning_rate"]),
                weight_decay=float(HP["weight_decay"]),
                epochs=int(HP["epochs"]), batch_size=int(HP["batch_size"]),
                val_fraction=float(HP["val_fraction"]),
                early_stopping_patience=int(HP["early_stopping_patience"]),
                feat_dropout=float(HP["feat_dropout"]),
                seed=int(SEED) + 902, device="cuda",
                holdout_group_id=row_to_uniq, show_progress=True)
            step("fitted_ship_run", train_loss=float(state_ship.train_loss),
                 val_loss=float(state_ship.val_loss))
        else:
            state_ship = state_val

        # holdout pred from the shipped model (all-infra dense for holdout)
        hold_pred = _apply(state_ship, ho_item, hold_sid, dense_hold)
        if not (np.isfinite(val_pred_f2).all() and np.isfinite(hold_pred).all()):
            raise ValueError("non-finite predictions — aborting save")
        step("holdout_predicted", hold_pred_mean=round(float(hold_pred.mean()), 5))

        # ---- (6) save state + valpred(fold2) + holdpred + meta ---------------------
        out_dir = f"{DRIVE_ROOT}/ship/diverse/{fam}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/mlp_state.pkl", "wb") as fh:
            pickle.dump(state_ship, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{out_dir}/mlp_state_valrun.pkl", "wb") as fh:
            pickle.dump(state_val, fh, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(f"{out_dir}/valpred_fold2.npy", val_pred_f2.astype(np.float32))
        np.save(f"{out_dir}/valpred_fold2_rowidx.npy", f2_idx.astype(np.int64))
        np.save(f"{out_dir}/valy_fold2.npy", y_f2.astype(np.float32))
        np.save(f"{out_dir}/holdpred.npy", hold_pred.astype(np.float32))

        meta = {
            "family": fam, "driver_family": driver_fam, "slug": slug, "ok": True,
            "feature_path": "AIDE_feats" if USE_AIDE_FEATS else "fallback_emb+metadata",
            "out_dir": out_dir,
            "n_train": N_tr, "n_holdout": N_ho, "n_folds01": n_f01, "n_fold2": n_f2,
            "n_unique_items": len(uniq_keys), "n_subjects": int(n_subjects), "d_emb": D_EMB,
            "dense_dim": int(dense_train.shape[1]),
            "dense_feature_names": list(dense_names),      # LOCKED order for runtime parity
            "nn_k": NN_K, "kmeans_k": KMEANS_K, "top_m_centroids": TOP_M_CENTROIDS,
            "metadata_coverage": meta_cov,
            "y_mean": round(float(y_train.mean()), 5),
            "val_logloss_fold2": round(val_logloss, 6),        # honest soft-logloss
            "val_loss_state": float(state_val.val_loss),
            "ship_val_loss_state": float(state_ship.val_loss),
            "val_pred_mean_fold2": round(float(val_pred_f2.mean()), 6),
            "hold_pred_mean": round(float(hold_pred.mean()), 6),
            "refit_on_all": REFIT_ON_ALL,
            "nn_infra_dirs": ({"all": f"{DRIVE_ROOT}/ship/nn_infra/{fam}/all",
                               "f01": f"{DRIVE_ROOT}/ship/nn_infra/{fam}/f01"}
                              if USE_AIDE_FEATS else None),
            "files": {
                "state_ship": f"{out_dir}/mlp_state.pkl",
                "state_valrun": f"{out_dir}/mlp_state_valrun.pkl",
                "valpred_fold2": f"{out_dir}/valpred_fold2.npy",
                "valpred_fold2_rowidx": f"{out_dir}/valpred_fold2_rowidx.npy",
                "valy_fold2": f"{out_dir}/valy_fold2.npy",
                "holdpred": f"{out_dir}/holdpred.npy",
            },
            "t_total_s": round(time.time() - t0, 1),
        }
        with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        prog.update(stage="done", **meta)
        _write_status(fam, dict(prog))
        print(f"[{fam}-aide] DONE — {json.dumps({k: meta[k] for k in ('val_logloss_fold2','hold_pred_mean','dense_dim')})}", flush=True)
        return meta

    except Exception as exc:
        prog.update(stage="error", ok=False, error=repr(exc),
                    traceback=traceback.format_exc(),
                    t_total_s=round(time.time() - t0, 1))
        _write_status(fam, dict(prog))
        print(f"[{fam}-aide] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
        raise


# ----------------------------------------------------------------------------------
# Launch (paste AFTER the run_bg/poll harness is defined in the Colab notebook):
#
#   import os
#   os.environ["SHIP_FAMILY"] = "qwen"        # colab2 -> qwen
#   # os.environ["SHIP_FAMILY"] = "nemotron"  # colab  -> nemotron
#   # os.environ["SHIP_FAMILY"] = "lgai"      # colab3 -> LGAI
#   # os.environ["SHIP_USE_AIDE_FEATS"] = "1" # DEFAULT (AIDE path). "0" = metadata fallback.
#   run_bg(f"{os.environ['SHIP_FAMILY']}_aide", fn)
#
#   poll(f"{os.environ['SHIP_FAMILY']}_aide")   # tiny fast cell; or read /content/<fam>_aide.json
#
if __name__ == "__main__":
    fn()
