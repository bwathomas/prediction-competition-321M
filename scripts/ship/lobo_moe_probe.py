"""Benchmark-specialist MoE probe — does routing beat one global model, COLD?

Hypothesis (user): train per-benchmark specialists, route a cold benchmark's
items to similar specialists via embeddings. Honest test, no confounds:
same features, same learner (XGBoost-GPU), and the GLOBAL baseline for a
held-out benchmark B is retrained WITHOUT B's rows (true model-level cold,
matching the specialists which never saw B by construction).

For each B in EVAL_BENCHES:
  global_minus_B   XGB on all rows with benchmark != B          -> eval on B
  moe_router(tau)  specialists (one XGB per training benchmark, trained once,
                   reused) blended per-item with softmax(cos(item_emb,
                   bench_centroid)/tau) over the 15 non-B specialists,
                   tau in TAUS (logit-space blend)
  moe_uniform      uniform blend of the 15 specialists (routing vs bagging)
  oracle_single    best single specialist for B (diagnostic upper bound)

Features = the harness tree-style space: 543 dense AIDE cols (geometry
gathered by item; label-derived groups stitched per item-fold, leakage-
correct row-wise) + PCA-64 of the item embedding. CAVEAT (recorded): the
label-derived shards were derived with B's labels present (feature-level
warmth) — affects all variants EQUALLY, so the routed-vs-global comparison
stands; absolute numbers are floors.

Family: nemotron (strongest cold member). Run on an A100 tab (~1.5-2h):
  python scripts/ship/lobo_moe_probe.py
Status: /content/lobo_moe.json  Result: DR/ship/exp_cold/lobo_moe_probe.json
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
REPO = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
FAMILY = os.environ.get("SHIP_FAMILY", "nemotron")
CODE_VERSION = "v2"
N_FOLDS, SPLIT_SEED = 3, 0
PCA_DIM = 64
EVAL_BENCHES = os.environ.get(
    "MOE_EVAL_BENCHES",
    "swebench,hle,mmlupro,agentdojo,rewardbench,livecodebench").split(",")
TAUS = [0.05, 0.1, 0.2]      # softmax temperatures on cosine similarity
MIN_SPEC_ROWS = 2000          # skip degenerate specialists
XGB_HP = dict(objective="reg:logistic", eval_metric="logloss", tree_method="hist",
              device="cuda", max_depth=8, eta=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=100.0, reg_lambda=2.0,
              reg_alpha=1.0, seed=0)
XGB_ROUNDS = int(os.environ.get("MOE_XGB_ROUNDS", "800"))
EPS = 1e-7
STATUS = "/content/lobo_moe.json"
OUT = f"{DR}/ship/exp_cold/lobo_moe_probe.json"
_t0 = time.time()
import sys
sys.path.insert(0, REPO)


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[lobo_moe] {stage} {kw if kw else ''}", flush=True)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    hard = (y == 0.0) | (y == 1.0)
    yb, pb = y[hard], p[hard]
    n1, n0 = int((yb == 1).sum()), int((yb == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = np.argsort(np.argsort(pb)) + 1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import pandas as pd
    import xgboost as xgb
    from aide.features.cache import FeatureCache
    from aide.features.driver import FAMILY_SLUG, load_embeddings
    from aide.features.store import FoldFeatureStore
    from aide.hygiene.manifest import item_fold

    driver_fam = {"nemotron": "llama", "qwen": "qwen", "lgai": "mistral"}[FAMILY]
    GEOM_GROUPS = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster"]
    LABEL_GROUPS = ["nn_label_derivatives", "cluster_passrate", "cluster_subject",
                    "counts_subject"]

    # ---- rows + labels --------------------------------------------------------------
    step("load_rows")
    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    labels_df = pd.read_parquet(db, columns=["subject_key", "item_key", "label"])
    labels_df["subject_key"] = labels_df["subject_key"].astype(str)
    labels_df["item_key"] = labels_df["item_key"].astype(str)
    if labels_df.duplicated(["subject_key", "item_key"]).any():
        labels_df = (labels_df.groupby(["subject_key", "item_key"], sort=False)["label"]
                     .mean().reset_index())
    tr_subj = labels_df["subject_key"].to_numpy()
    tr_item = labels_df["item_key"].to_numpy()
    y = labels_df["label"].to_numpy().astype(np.float32)
    bmap = pd.read_parquet(db, columns=["item_key", "benchmark"]).drop_duplicates("item_key")
    bdict = dict(zip(bmap["item_key"].astype(str), bmap["benchmark"].astype(str)))
    bench = np.array([bdict.get(i, "UNK") for i in tr_item])
    benches = sorted(set(bench.tolist()) - {"UNK"})
    step("rows_loaded", n=len(y), n_bench=len(benches))

    # ---- embeddings + per-bench centroids + PCA -------------------------------------
    step("load_embeddings")
    emb_dir = f"{DR}/embeddings/{FAMILY_SLUG[driver_fam]}"
    all_keys, all_emb = load_embeddings(f"{emb_dir}/items.parquet")
    all_emb = np.asarray(all_emb, np.float32)
    all_emb /= np.clip(np.linalg.norm(all_emb, axis=1, keepdims=True), 1e-9, None)
    kpos = {str(k): i for i, k in enumerate(all_keys)}
    item_pos = np.array([kpos[i] for i in tr_item])
    item_bench_of_key = np.array([bdict.get(str(k), "UNK") for k in all_keys])
    centroids = {}
    for b in benches:
        m = item_bench_of_key == b
        c = all_emb[m].mean(0)
        centroids[b] = c / max(np.linalg.norm(c), 1e-9)
    step("embeddings_ready", n_items=len(all_keys))

    rng = np.random.default_rng(0)
    pca_fit_idx = rng.choice(len(all_emb), min(60000, len(all_emb)), replace=False)
    sub = all_emb[pca_fit_idx]
    mu = sub.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(sub - mu, full_matrices=False)
    basis = vt[:PCA_DIM].T.astype(np.float32)
    pca_all = (all_emb - mu) @ basis           # [n_unique_items, 64]
    step("pca_ready", dim=PCA_DIM)

    # ---- dense features (geometry gather + per-fold label scatter) ------------------
    step("assemble_dense")
    store = FoldFeatureStore(FeatureCache(f"{DR}/features", code_version=CODE_VERSION),
                             embedding_family=driver_fam, seed=SPLIT_SEED,
                             n_folds=N_FOLDS)
    geo_blocks, geo_index = [], None
    for g in GEOM_GROUPS:
        sh = store.cache.read_shard(store._key(g, "all"))
        if geo_index is None:
            geo_index = {str(k): i for i, k in enumerate(sh.row_ids)}
        geo_blocks.append(np.asarray(sh.X, np.float32))
    geo_X = np.concatenate(geo_blocks, axis=1)
    row_geo = np.array([geo_index[i] for i in tr_item])
    n_geo = geo_X.shape[1]

    row_fold = np.fromiter((item_fold(i, N_FOLDS, SPLIT_SEED) for i in tr_item),
                           dtype=np.int64, count=len(tr_item))
    rid_pos = {f"{s}|{i}": k for k, (s, i) in enumerate(zip(tr_subj, tr_item))}
    lab_cols = None
    lab_X = None
    for f in range(N_FOLDS):
        blocks = []
        rids = None
        for g in LABEL_GROUPS:
            sh = store.cache.read_shard(store._key(g, f"fold{f}"))
            if rids is None:
                rids = [str(r) for r in sh.row_ids]
            blocks.append(np.asarray(sh.X, np.float32))
        Xf = np.concatenate(blocks, axis=1)
        if lab_X is None:
            lab_cols = Xf.shape[1]
            lab_X = np.zeros((len(tr_item), lab_cols), np.float32)
        hit = 0
        for r_i, rid in enumerate(rids):
            k = rid_pos.get(rid)
            if k is not None and row_fold[k] == f:
                lab_X[k] = Xf[r_i]
                hit += 1
        step("label_fold_scattered", fold=f, matched=hit)
    X = np.concatenate([geo_X[row_geo], lab_X, pca_all[item_pos]], axis=1)
    del geo_X, lab_X
    step("dense_ready", n_cols=int(X.shape[1]), n_geo=n_geo)

    def train_xgb(mask, tag):
        idx = np.where(mask)[0]
        rng_l = np.random.default_rng(abs(hash(tag)) % (2**31))
        val = rng_l.random(len(idx)) < 0.1
        dtr = xgb.DMatrix(X[idx[~val]], label=y[idx[~val]])
        dva = xgb.DMatrix(X[idx[val]], label=y[idx[val]])
        bst = xgb.train(XGB_HP, dtr, num_boost_round=XGB_ROUNDS,
                        evals=[(dva, "va")], early_stopping_rounds=50,
                        verbose_eval=False)
        step("trained", tag=tag, n=int(mask.sum()),
             best_iter=int(getattr(bst, "best_iteration", -1)))
        return bst

    # ---- specialists (one per benchmark, trained once) ------------------------------
    specialists = {}
    for b in benches:
        m = bench == b
        if m.sum() < MIN_SPEC_ROWS:
            step("specialist_skipped", bench=b, n=int(m.sum()))
            continue
        specialists[b] = train_xgb(m, f"spec_{b}")

    res = {"family": FAMILY, "taus": TAUS, "eval_benches": EVAL_BENCHES,
           "caveat": "label-derived feature shards include each benchmark's labels "
                     "(feature-level warmth, common to all variants); model-level "
                     "cold is honest for both global_minus_B and specialists",
           "per_bench": {}}

    for B in EVAL_BENCHES:
        if B not in benches:
            continue
        mB = bench == B
        yB = y[mB].astype(np.float64)
        dB = xgb.DMatrix(X[mB])
        entry = {"n": int(mB.sum())}

        g = train_xgb(~mB, f"global_minus_{B}")
        pg = np.clip(g.predict(dB), EPS, 1 - EPS)
        ag = auc(yB, pg)
        entry["global_minus_B"] = {"bce": round(bce(yB, pg), 5),
                                   "auc": (round(ag, 4) if ag is not None else None)}

        spec_names = [b for b in specialists if b != B]
        P = np.column_stack([
            np.clip(specialists[b].predict(dB), EPS, 1 - EPS) for b in spec_names])
        Z = lg(P)
        sims = np.column_stack([
            all_emb[item_pos[mB]] @ centroids[b] for b in spec_names])

        pu = sg(Z.mean(axis=1))
        au_ = auc(yB, pu)
        entry["moe_uniform"] = {"bce": round(bce(yB, pu), 5),
                                "auc": (round(au_, 4) if au_ is not None else None)}
        for tau in TAUS:
            W = np.exp(sims / tau)
            W /= W.sum(axis=1, keepdims=True)
            pr = sg((W * Z).sum(axis=1))
            ar = auc(yB, pr)
            entry[f"moe_router_tau{tau}"] = {
                "bce": round(bce(yB, pr), 5),
                "auc": (round(ar, 4) if ar is not None else None),
                "mean_top_weight": round(float(W.max(axis=1).mean()), 3)}
        singles = {b: round(bce(yB, P[:, j]), 5) for j, b in enumerate(spec_names)}
        best = min(singles, key=singles.get)
        entry["oracle_single"] = {"bench": best, "bce": singles[best]}
        entry["singles"] = singles
        res["per_bench"][B] = entry
        step("bench_eval_done", bench=B,
             glob=entry["global_minus_B"]["bce"],
             uni=entry["moe_uniform"]["bce"],
             **{f"r{t}": entry[f"moe_router_tau{t}"]["bce"] for t in TAUS})

    evald = list(res["per_bench"])
    summ = {}
    for v in (["global_minus_B", "moe_uniform"]
              + [f"moe_router_tau{t}" for t in TAUS]):
        summ[f"{v}_mean_bce"] = round(float(np.mean(
            [res["per_bench"][b][v]["bce"] for b in evald])), 5)
    res["summary"] = summ
    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    step("done", **summ)
    print("LOBO MOE PROBE DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
