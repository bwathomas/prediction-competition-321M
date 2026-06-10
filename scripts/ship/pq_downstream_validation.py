"""PQ-512 downstream-OOF validation — does the shipped PQ index move member/stack OOF?

The quantization bake-off measured INDEX-level damage (recall@64 0.826, noisy nn_geometry);
this measures what matters: the final-stack OOF delta. Design = deployment train/serve skew:
models stay FIXED (trained on full-precision-index features). For the held-out fold's rows we
recompute ONLY the index-dependent columns — nn_geometry (3) + nn_label_derivatives (13) +
counts_subject (1) — with the kNN index replaced by the PQ-512 DECODED vectors (query stays
full precision). Scoring against decoded vectors is mathematically identical to runtime
ADC LUT-sum (sum_m LUT[m, code] == q . x_hat), so this is exactly what Codabench would see.
Cluster blocks / PCA / raw item embeddings ship full precision and are untouched.

Per family+fold it reports:
  * etbig (cuML fold model): rebuilt-baseline BCE (+ max|p - stored p_full| rebuild check)
    vs PQ-feature BCE, plus prediction deltas.
  * mlp: 11 saved members + the deployable LGB stacker, both feature variants
    (PQ delta = pq vs recomputed-baseline through the IDENTICAL path).
  * canon stack (FINAL_CANON weights+bias over 9 stored fold-f member cols): stored vs
    recomputed-baseline vs PQ (this family's etbig+mlp columns swapped).

Run (detached, on a GPU tab):  VAL_FAMILY=nemotron VAL_FOLD=0 python scripts/ship/pq_downstream_validation.py
Status/result: /content/pqval_<fam>_fold<f>.json  (mirrored to DR/ship/stack/ on success)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

FAMILY = os.environ.get("VAL_FAMILY", "nemotron").strip().lower()
FOLD = int(os.environ.get("VAL_FOLD", "0"))
# PQ sims saturate >= the alias threshold (1 - 1e-6) for whole near-duplicate groups
# (quantization collapses them onto the same codes), so far more retrieved neighbours get
# alias-dropped than with exact sims — retrieve much deeper than derive_nn's default 2.
GEO_BUFFER = int(os.environ.get("VAL_GEO_BUFFER", "2048"))
LAB_BUFFER = int(os.environ.get("VAL_LAB_BUFFER", "1024"))
REPO_ROOT = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
DRIVE_ROOT = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
CODE_VERSION = os.environ.get("SHIP_CODE_VERSION", "v2")
N_FOLDS, SPLIT_SEED = 3, 0
STATUS_PATH = f"/content/pqval_{FAMILY}_fold{FOLD}.json"
sys.path.insert(0, REPO_ROOT)

FAM_ALIAS = {"qwen": "qwen", "nemotron": "llama", "lgai": "mistral"}
LABEL_GROUPS = ["nn_label_derivatives", "cluster_passrate", "cluster_subject", "counts_subject"]
GEOM_GROUPS = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster"]
AFFECTED = ("nn_geometry", "nn_label_derivatives", "counts_subject")
_EPS = 1e-7

_t0 = time.time()
_status: dict = {"family": FAMILY, "fold": FOLD, "stage": "init"}


def step(stage, **extra):
    _status.update(stage=stage, t_s=round(time.time() - _t0, 1), **extra)
    tmp = STATUS_PATH + ".tmp"
    Path(tmp).write_text(json.dumps(_status, indent=1, default=str))
    os.replace(tmp, STATUS_PATH)
    print(f"[pqval] {stage} {extra if extra else ''} ({_status['t_s']}s)", flush=True)


def soft_bce(y, p):
    p = np.clip(np.asarray(p, np.float64), _EPS, 1 - _EPS)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gpu_knn_factory(index_mat, device, sub_q=4096):
    """Reusable knn_fn(_, query, k) -> (idx, sim) via torch GPU matmul (exact dot top-k)."""
    import torch
    idx_t = torch.from_numpy(np.ascontiguousarray(index_mat, dtype=np.float32)).to(device)

    def knn_fn(_ignored, query_emb, k):
        q = np.ascontiguousarray(np.asarray(query_emb, np.float32))
        k = min(int(k), idx_t.shape[0])
        out_i = np.empty((q.shape[0], k), np.int64)
        out_s = np.empty((q.shape[0], k), np.float32)
        with torch.no_grad():
            for s in range(0, q.shape[0], sub_q):
                qc = torch.from_numpy(q[s:s + sub_q]).to(device)
                sims = qc @ idx_t.T
                sv, si = torch.topk(sims, k, dim=1)
                out_i[s:s + sub_q] = si.cpu().numpy()
                out_s[s:s + sub_q] = sv.cpu().numpy()
                del qc, sims, sv, si
        return out_i, out_s

    return knn_fn, idx_t


def decode_pq(pq_path, all_item_keys):
    """PQ codes -> reconstructed vectors x_hat [N, D], row-aligned to all_item_keys."""
    with np.load(pq_path, allow_pickle=False) as z:
        cb, codes = z["codebook"], z["codes"]
        keys = z["item_keys"].astype(str)
        M, ds, D = int(z["M"]), int(z["ds"]), int(z["D"])
    x = np.empty((codes.shape[0], D), np.float32)
    for m in range(M):
        x[:, m * ds:(m + 1) * ds] = cb[m][codes[:, m]]
    pos = {k: i for i, k in enumerate(keys)}
    order = np.fromiter((pos[str(k)] for k in all_item_keys), dtype=np.int64,
                        count=len(all_item_keys))  # KeyError = key mismatch -> abort
    return x[order]


def mlp_forward_torch(state, subject_ids, item_emb_pair, dense_X, n, device,
                      chunk=262144):
    """Torch mirror of src.mlp_member.apply_batch (verified vs numpy on a sample).

    ``item_emb_pair`` = (emb_matrix, row_indices) gathered per chunk — a materialized
    [1.5M, 4096] gather would cost ~25 GB RAM."""
    import torch

    def T(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    w = {k: T(getattr(state, k)) for k in
         ["l1_value_W", "l1_value_b", "l1_gate_W", "l1_gate_b",
          "l2_value_W", "l2_value_b", "l2_gate_W", "l2_gate_b", "head_W"]}
    se = T(state.subj_emb) if state.subj_emb_dim > 0 else None
    dm = T(state.dense_mean) if state.dense_dim > 0 else None
    dsd = T(state.dense_std) if state.dense_dim > 0 else None
    out = np.empty(n, np.float32)
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            parts = []
            if state.subj_emb_dim > 0:
                sid = torch.from_numpy(np.asarray(subject_ids[s:e], np.int64)).to(device)
                unk = int(state.n_subjects)
                sid = torch.where((sid >= 0) & (sid < unk), sid, torch.full_like(sid, unk))
                parts.append(se[sid])
            if state.use_item_emb:
                emb_mat, rows = item_emb_pair
                parts.append(T(emb_mat[rows[s:e]]))
            if state.dense_dim > 0:
                mz = (T(dense_X[s:e]) - dm) / dsd
                parts.append(torch.nan_to_num(mz, nan=0.0, posinf=0.0, neginf=0.0))
            x = torch.cat(parts, dim=1)
            h1 = (x @ w["l1_value_W"] + w["l1_value_b"]) * torch.sigmoid(x @ w["l1_gate_W"] + w["l1_gate_b"])
            h2 = (h1 @ w["l2_value_W"] + w["l2_value_b"]) * torch.sigmoid(h1 @ w["l2_gate_W"] + w["l2_gate_b"])
            z = (h2 @ w["head_W"]).reshape(-1) + float(state.head_b)
            out[s:e] = torch.sigmoid(z).cpu().numpy()
            del x, h1, h2, z, parts
    return np.clip(out, 1e-7, 1 - 1e-7).astype(np.float32)


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from aide.features.driver import (FAMILY_SLUG, load_embeddings, unit_rows,
                                      build_fold_passrate)
    from aide.features.cache import FeatureCache
    from aide.features.store import FoldFeatureStore
    from aide.features.derive_nn import derive_nn
    from aide.features.nn_fast import derive_nn_labels_fast
    from aide.features.passrate import CsrPassrate
    from aide.hygiene.manifest import item_fold
    from src.mlp_member import MlpMemberState, apply_batch as mlp_apply_np
    import pandas as pd

    driver_fam = FAM_ALIAS[FAMILY]
    emb_dir = f"{DRIVE_ROOT}/embeddings/{FAMILY_SLUG[driver_fam]}"
    etbig_dir = Path(f"{DRIVE_ROOT}/ship/exp_loo/{FAMILY}/etbig_full_fold{FOLD}")
    mlp_dir = Path(f"{DRIVE_ROOT}/ship/exp_loo/{FAMILY}/mlp_loo_fold{FOLD}")
    res: dict = {"family": FAMILY, "fold": FOLD, "device": device}

    # ---- embeddings + labels (mirror exp_loo_category_mlp row construction) ------
    step("load_embeddings")
    item_keys, item_emb = load_embeddings(f"{emb_dir}/items.parquet")
    item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
    emb_unit = unit_rows(item_emb)
    item_pos = {str(k): i for i, k in enumerate(item_keys)}

    step("load_labels")
    import glob as _glob
    db = _glob.glob(f"{DRIVE_ROOT}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    labels_df = pd.read_parquet(db, columns=["subject_key", "item_key", "label"])
    labels_df["subject_key"] = labels_df["subject_key"].astype(str)
    labels_df["item_key"] = labels_df["item_key"].astype(str)
    if labels_df.duplicated(["subject_key", "item_key"]).any():
        labels_df = (labels_df.groupby(["subject_key", "item_key"], sort=False)["label"]
                     .mean().reset_index())
    label_map = {(s, i): float(l) for s, i, l in zip(
        labels_df["subject_key"].to_numpy(), labels_df["item_key"].to_numpy(),
        labels_df["label"].to_numpy())}
    tr_subj = [str(s) for s in labels_df["subject_key"].to_numpy()]
    tr_item = [str(i) for i in labels_df["item_key"].to_numpy()]

    store = FoldFeatureStore(FeatureCache(f"{DRIVE_ROOT}/features", code_version=CODE_VERSION),
                             embedding_family=driver_fam, seed=SPLIT_SEED, n_folds=N_FOLDS)
    geo0 = store.cache.read_shard(store._key(GEOM_GROUPS[0], "all"))
    feat_items = set(str(k) for k in geo0.row_ids)
    geo_index = {str(k): i for i, k in enumerate(geo0.row_ids)}

    keep = np.array([(it in feat_items) and ((s, it) in label_map)
                     for s, it in zip(tr_subj, tr_item)], dtype=bool)
    tr_item = [tr_item[i] for i in np.where(keep)[0]]
    tr_subj = [tr_subj[i] for i in np.where(keep)[0]]
    y_all = np.array([label_map[(s, i)] for s, i in zip(tr_subj, tr_item)], dtype=np.float32)
    row_fold = np.fromiter((item_fold(i, N_FOLDS, SPLIT_SEED) for i in tr_item),
                           dtype=np.int64, count=len(tr_item))
    oof_idx = np.where(row_fold == FOLD)[0]
    oof_items = np.asarray([tr_item[r] for r in oof_idx])
    oof_subj = np.asarray([tr_subj[r] for r in oof_idx])
    y = y_all[oof_idx]
    step("rows_ready", n_oof=int(oof_idx.size))

    # hard alignment check vs the stored etbig OOF preds (same construction => identical order)
    stored = np.load(etbig_dir / "preds" / "oof_preds.npz", allow_pickle=False)
    if not (stored["oof_items"].astype(str) == oof_items).all():
        raise RuntimeError("OOF row order mismatch vs stored etbig preds — abort")
    p_full_stored = stored["p_full"].astype(np.float64)
    step("alignment_ok")

    # ---- assemble dense (OOF rows only): [geom | label | pca] --------------------
    step("assemble_dense")
    layout = json.loads((etbig_dir / "shared" / "dense_layout.json").read_text())
    Xg, _ = store.assemble(GEOM_GROUPS, fold=0, check_coverage=False)
    geo_group_of_col = []
    for g in GEOM_GROUPS:
        _, gc = store.assemble([g], fold=0, check_coverage=False)
        geo_group_of_col += [g] * len(gc)
    gi = np.fromiter((geo_index[i] for i in oof_items), dtype=np.int64, count=oof_items.size)
    Xgeo = Xg[gi].astype(np.float32)
    del Xg

    Xl, _ = store.assemble(LABEL_GROUPS, fold=FOLD, check_coverage=False)
    lab_group_of_col = []
    for g in LABEL_GROUPS:
        _, lc = store.assemble([g], fold=FOLD, check_coverage=False)
        lab_group_of_col += [g] * len(lc)
    shard_rids = np.asarray(store.cache.read_shard(
        store._key(LABEL_GROUPS[0], FOLD)).row_ids).astype(str)
    rid_oof = np.asarray([f"{s}|{i}" for s, i in zip(oof_subj, oof_items)])
    shard_pos = {r: j for j, r in enumerate(shard_rids)}
    lab_rows = np.fromiter((shard_pos.get(r, -1) for r in rid_oof), dtype=np.int64,
                           count=rid_oof.size)
    n_miss = int((lab_rows < 0).sum())
    Xlab = np.zeros((oof_items.size, Xl.shape[1]), np.float32)
    okm = lab_rows >= 0
    Xlab[okm] = Xl[lab_rows[okm]].astype(np.float32)
    del Xl

    with np.load(etbig_dir / "shared" / "pca_item_emb.npz") as z:
        pca_comp, pca_mean = z["components"], z["mean"]
    emb_rows = np.fromiter((item_pos[i] for i in oof_items), dtype=np.int64,
                           count=oof_items.size)
    Xpca = ((item_emb[emb_rows] - pca_mean) @ pca_comp.T).astype(np.float32)

    dense = np.concatenate([Xgeo, Xlab, Xpca], axis=1)
    dense = np.where(np.isfinite(dense), dense, 0.0).astype(np.float32)
    group_of_col = np.array(geo_group_of_col + lab_group_of_col
                            + ["item_emb_pca"] * Xpca.shape[1], dtype=object)
    if dense.shape[1] != len(layout["dense_names_full"]) or \
            list(group_of_col) != list(layout["dense_group_of_col"]):
        raise RuntimeError(f"dense layout mismatch: {dense.shape[1]} vs "
                           f"{len(layout['dense_names_full'])}")
    del Xgeo, Xlab, Xpca
    step("dense_ready", n_cols=int(dense.shape[1]), label_rows_missing=n_miss)

    # ---- PQ recompute of the affected columns ------------------------------------
    step("pq_decode")
    x_hat = decode_pq(f"{DRIVE_ROOT}/ship/ship_models/pqidx_{FAMILY}.npz", item_keys)
    res["pq_recon_cos"] = float(np.mean(np.sum(x_hat * emb_unit, axis=1)))  # decode sanity

    # nn_geometry: per UNIQUE oof item, index = PQ-decoded ALL items (fold-invariant)
    step("pq_nn_geometry")
    uniq_items, uniq_inv = np.unique(oof_items, return_inverse=True)
    uq_rows = np.fromiter((item_pos[i] for i in uniq_items), dtype=np.int64,
                          count=uniq_items.size)
    knn_all, idx_t = gpu_knn_factory(x_hat, device)
    empty_pr = CsrPassrate.empty([], [str(k) for k in item_keys])
    geo_blocks = []
    CH = 20000
    for s in range(0, uniq_items.size, CH):
        e = min(s + CH, uniq_items.size)
        blk = derive_nn(query_emb=emb_unit[uq_rows[s:e]],
                        query_item_keys=[str(k) for k in uniq_items[s:e]],
                        query_subjects=[""] * (e - s),
                        row_ids=[str(k) for k in uniq_items[s:e]],
                        index_emb=x_hat, index_item_keys=[str(k) for k in item_keys],
                        passrate=empty_pr, Ks=(4, 8, 32, 64), knn_fn=knn_all,
                        search_buffer=GEO_BUFFER)
        geo_blocks.append(blk["nn_geometry"].X)
        step("pq_nn_geometry", done=e, of=int(uniq_items.size))
    geo_pq = np.concatenate(geo_blocks, axis=0)[uniq_inv].astype(np.float32)
    del idx_t, knn_all
    torch.cuda.empty_cache()

    # nn_label + counts: per OOF row, index/passrate = fold-train only (mirrors the
    # original derivation: manifest folds over the embedding key set)
    step("pq_nn_labels")
    train_keys = [str(k) for k in item_keys
                  if item_fold(str(k), N_FOLDS, SPLIT_SEED) != FOLD]
    tk_rows = np.fromiter((item_pos[str(k)] for k in train_keys), dtype=np.int64,
                          count=len(train_keys))
    sub_keys, _ = load_embeddings(f"{emb_dir}/subjects.parquet")
    passrate = build_fold_passrate(labels_df, train_keys, sub_keys,
                                   [str(k) for k in item_keys])
    knn_tr, idx_t2 = gpu_knn_factory(x_hat[tk_rows], device)
    nn_blocks, cnt_blocks = [], []
    CH = 30000
    for s in range(0, oof_idx.size, CH):
        e = min(s + CH, oof_idx.size)
        blk = derive_nn_labels_fast(
            query_emb=emb_unit[emb_rows[s:e]],
            query_item_keys=[str(k) for k in oof_items[s:e]],
            query_subjects=[str(k) for k in oof_subj[s:e]],
            row_ids=list(rid_oof[s:e]),
            index_emb=x_hat[tk_rows], index_item_keys=[str(k) for k in train_keys],
            passrate=passrate, Ks=(4, 8, 32, 64), knn_fn=knn_tr,
            search_buffer=LAB_BUFFER)
        nn_blocks.append(blk["nn_label_derivatives"].X)
        cnt_blocks.append(blk["counts_subject"].X)
        step("pq_nn_labels", done=e, of=int(oof_idx.size))
    nnlab_pq = np.concatenate(nn_blocks, axis=0).astype(np.float32)
    cnt_pq = np.concatenate(cnt_blocks, axis=0).astype(np.float32)
    del idx_t2, knn_tr
    torch.cuda.empty_cache()

    dense_pq = dense.copy()
    swapped = {"nn_geometry": geo_pq, "nn_label_derivatives": nnlab_pq,
               "counts_subject": cnt_pq}
    for g, Xn in swapped.items():
        cols = np.where(group_of_col == g)[0]
        if cols.size != Xn.shape[1]:
            raise RuntimeError(f"{g}: {cols.size} layout cols != {Xn.shape[1]} recomputed")
        res[f"feat_delta_{g}"] = {
            "mean_abs": float(np.mean(np.abs(dense[:, cols] - Xn))),
            "max_abs": float(np.max(np.abs(dense[:, cols] - Xn)))}
        dense_pq[:, cols] = np.where(np.isfinite(Xn), Xn, 0.0)
    step("pq_features_ready", **{k: v for k, v in res.items() if k.startswith("feat_")})

    # ---- etbig: fixed fold model, both feature variants ---------------------------
    step("etbig_predict")
    import pickle
    # pickle is required here: cuML RandomForest models are persisted as pickles by our
    # own training run (exp_loo_category_mlp.py); this artifact lives on our own Drive.
    with open(etbig_dir / "models" / "full" / "cuml_rf.pkl", "rb") as fh:
        rf = pickle.load(fh)

    def rf_pred(X):
        out = np.empty(X.shape[0], np.float64)
        for s in range(0, X.shape[0], 500000):
            p = rf.predict(np.ascontiguousarray(X[s:s + 500000], dtype=np.float32))
            out[s:s + 500000] = np.asarray(p.get() if hasattr(p, "get") else p,
                                           np.float64).reshape(-1)
        return np.clip(out, _EPS, 1 - _EPS)

    et_base = rf_pred(dense)
    et_pq = rf_pred(dense_pq)
    res["etbig"] = {
        "rebuild_max_abs_vs_stored": float(np.max(np.abs(et_base - p_full_stored))),
        "bce_stored": soft_bce(y, p_full_stored),
        "bce_base": soft_bce(y, et_base), "bce_pq": soft_bce(y, et_pq),
        "delta_bce": soft_bce(y, et_pq) - soft_bce(y, et_base),
        "pred_mean_abs_delta": float(np.mean(np.abs(et_pq - et_base))),
        "pred_corr": float(np.corrcoef(et_pq, et_base)[0, 1])}
    step("etbig_done", **res["etbig"])
    del rf

    # ---- mlp: 11 members + deployable stacker, both variants ----------------------
    step("mlp_predict")
    mlp_layout = json.loads((mlp_dir / "shared" / "dense_layout.json").read_text())
    mlp_names = list(mlp_layout["dense_names_full"])     # geom+label only (no pca)
    name_to_col = {n: i for i, n in enumerate(layout["dense_names_full"])}
    mlp_cols = np.fromiter((name_to_col[n] for n in mlp_names), dtype=np.int64,
                           count=len(mlp_names))
    subj_vocab = json.loads((mlp_dir / "shared" / "subj_vocab.json").read_text())
    n_sub = len(subj_vocab)
    sid = np.fromiter((subj_vocab.get(s, n_sub) for s in oof_subj), dtype=np.int64,
                      count=oof_subj.size)
    smeta = json.loads((mlp_dir / "stacker" / "stacker_meta.json").read_text())
    cat_list = list(smeta["input_columns"])
    name_pos = {n: i for i, n in enumerate(mlp_names)}

    def mlp_member(tag, dense_mat):
        st = MlpMemberState.load(mlp_dir / "models" / tag)
        dsel = None
        if st.dense_dim > 0:
            cols = np.fromiter((name_pos[n] for n in st.dense_feature_names),
                               dtype=np.int64, count=len(st.dense_feature_names))
            dsel = dense_mat[:, mlp_cols][:, cols]
        s_ids = sid if st.subj_emb_dim > 0 else None
        p = mlp_forward_torch(st, s_ids, (item_emb, emb_rows), dsel,
                              oof_idx.size, device)
        # exactness spot-check vs the canonical numpy forward
        k = min(2048, p.shape[0])
        p_np = mlp_apply_np(st, subject_ids=None if s_ids is None else s_ids[:k],
                            item_emb=item_emb[emb_rows[:k]] if st.use_item_emb else None,
                            dense_X=None if dsel is None else dsel[:k])
        if float(np.max(np.abs(p[:k] - p_np))) > 1e-4:
            raise RuntimeError(f"torch/numpy forward mismatch on {tag}")
        return p

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(mlp_dir / "stacker" / "lgb_stack_final.txt"))
    mlp_out = {}
    for variant, dmat in [("base", dense), ("pq", dense_pq)]:
        P = np.column_stack([mlp_member(f"loo__{c}", dmat) for c in cat_list])
        stacked = np.clip(booster.predict(P), _EPS, 1 - _EPS)
        p_full_mlp = mlp_member("full", dmat)
        mlp_out[variant] = {"stacked": stacked, "p_full": p_full_mlp}
        step(f"mlp_{variant}_done", bce_stacked=soft_bce(y, stacked),
             bce_full=soft_bce(y, p_full_mlp))
    mlp_stored = np.load(mlp_dir / "preds" / "oof_preds.npz", allow_pickle=False)
    stacked_stored = mlp_stored["stacked_oof"].astype(np.float64)
    res["mlp"] = {
        "bce_stacked_stored_innercv": soft_bce(y, stacked_stored),
        "bce_stacked_base": soft_bce(y, mlp_out["base"]["stacked"]),
        "bce_stacked_pq": soft_bce(y, mlp_out["pq"]["stacked"]),
        "delta_bce_stacked": soft_bce(y, mlp_out["pq"]["stacked"])
                             - soft_bce(y, mlp_out["base"]["stacked"]),
        "bce_full_base": soft_bce(y, mlp_out["base"]["p_full"]),
        "bce_full_pq": soft_bce(y, mlp_out["pq"]["p_full"]),
        "stacked_pred_mean_abs_delta": float(np.mean(np.abs(
            mlp_out["pq"]["stacked"] - mlp_out["base"]["stacked"]))),
    }
    step("mlp_done", **res["mlp"])

    # ---- canon stack delta (FINAL_CANON weights + bias, 9 stored fold-f columns) --
    step("stack")
    canon = json.loads(Path(f"{DRIVE_ROOT}/ship/stack/FINAL_CANON_2026-06-07.json").read_text())
    cols = {}
    for fam in ["qwen", "nemotron", "lgai"]:
        zm = np.load(f"{DRIVE_ROOT}/ship/exp_loo/{fam}/mlp_loo_fold{FOLD}/preds/oof_preds.npz",
                     allow_pickle=False)
        cols[f"{fam}.mlp.L1"] = zm["stacked_oof"].astype(np.float64)
        for arch in ["etbig", "irt_bag"]:
            za = np.load(f"{DRIVE_ROOT}/ship/exp_loo/{fam}/{arch}_full_fold{FOLD}/preds/oof_preds.npz",
                         allow_pickle=False)
            if not (za["oof_items"].astype(str) == oof_items).all():
                raise RuntimeError(f"{fam}.{arch} row order mismatch")
            cols[f"{fam}.{arch}"] = za["p_full"].astype(np.float64)
    w = {m["key"]: float(m["weight"]) for m in canon["members"]}
    bias = float(canon.get("bias", 0.0))

    def stack_bce(repl):
        z = np.full(y.shape[0], bias, np.float64)
        for k, p in cols.items():
            z += w[k] * _logit(repl.get(k, p))
        return soft_bce(y, _sigmoid(z))

    res["stack"] = {
        "bce_stored_members": stack_bce({}),
        "bce_recomputed_base": stack_bce({f"{FAMILY}.etbig": et_base,
                                          f"{FAMILY}.mlp.L1": mlp_out["base"]["stacked"]}),
        "bce_pq": stack_bce({f"{FAMILY}.etbig": et_pq,
                             f"{FAMILY}.mlp.L1": mlp_out["pq"]["stacked"]}),
    }
    res["stack"]["delta_bce_pq_vs_base"] = (res["stack"]["bce_pq"]
                                            - res["stack"]["bce_recomputed_base"])
    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    _status["result"] = res
    step("done")
    out = Path(f"{DRIVE_ROOT}/ship/stack/pqval_{FAMILY}_fold{FOLD}.json")
    out.write_text(json.dumps(res, indent=2, default=str))
    print("PQVAL DONE", json.dumps(res["stack"]), flush=True)
    return res


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _status.update(stage="ERROR", error=repr(e), tb=traceback.format_exc())
        Path(STATUS_PATH).write_text(json.dumps(_status, indent=1, default=str))
        print("PQVAL FAILED:", repr(e), flush=True)
        traceback.print_exc()
        raise
