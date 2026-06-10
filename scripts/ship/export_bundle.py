"""Stage the 3-family submission bundle from Drive artifacts (run on a Colab tab).

Consumes: LINEAR_SHIP.json (refit_linear_stack.py output), the per-family Drive artifacts
(pqidx, centroids, ship-etbig forest+pca, mlp foldALL members), the prepared labels db and
embedding parquets. Produces <OUT>/bundle/ ready to zip:

  model.py fam_common.py encoders.py labeling.py models.txt        (from BUNDLE_SRC)
  shared_artifacts/{passrate.npz, subj_vocab.json, passrate_row.json, bc_to_id.json,
                    train_counts.json, stack_top.json}
  <fam>/artifacts/{runtime_meta.json, centroids.npz, cluster_aux.npz, pqidx.npz,
                   pca_item_emb.npz, etbig_forest.npz, mlp/<member>/{weights.npz,meta.json}}

Canonical item order = qwen items.parquet key order (passrate columns); each family ships a
pq_to_passrate_col remap. Status: /content/export_bundle.json
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
REPO = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
OUT = Path(os.environ.get("BUNDLE_OUT", "/content/bundle_out"))
BUNDLE_SRC = Path(os.environ.get("BUNDLE_SRC", f"{REPO}/submission_bundle"))
FAMS = ["qwen", "nemotron", "lgai"]
N_FOLDS, SPLIT_SEED = 3, 0
STATUS = "/content/export_bundle.json"
_t0 = time.time()
sys.path.insert(0, REPO)

GEO_COUNTS = [("centroid_distance", 256), ("cluster_geometry", 9), ("nn_geometry", 3),
              ("item_cluster", 257)]
LAB_COUNTS = [("nn_label_derivatives", 13), ("cluster_passrate", 1),
              ("cluster_subject", 3), ("counts_subject", 1)]

ENCODER_META = {
    "qwen": {"encoder_model_id": "Qwen/Qwen3-Embedding-8B", "max_length": 1024,
             "pooling": "last_token", "padding_side": "left", "pool_fp32": False,
             "l2_normalize": False, "batch_size": 8,
             "passage_prefix": "Instruct: Represent this AI evaluation context for "
                               "difficulty prediction\nQuery: "},
    "nemotron": {"encoder_model_id": "nvidia/llama-embed-nemotron-8b", "max_length": 4096,
                 "pooling": "mean", "pool_fp32": False, "l2_normalize": False,
                 "batch_size": 8, "passage_prefix": "", "bidirectional_llama": True},
    "lgai": {"encoder_model_id": "annamodels/LGAI-Embedding-Preview", "max_length": 4096,
             "pooling": "last_token", "pool_fp32": False, "l2_normalize": False,
             "batch_size": 8, "passage_prefix": ""},
}


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[export] {stage} {kw if kw else ''}", flush=True)


def canonical_dense_names():
    names = []
    gi = 0
    for g, n in GEO_COUNTS:
        names += [f"{g}__c{gi + k}" for k in range(n)]
        gi += n
    li = 0
    for g, n in LAB_COUNTS:
        names += [f"{g}__c{li + k}" for k in range(n)]
        li += n
    return names


def main():
    from aide.features.driver import FAMILY_SLUG, load_embeddings, unit_rows
    from aide.features.derive_cluster import _sqdist
    sys.path.insert(0, str(BUNDLE_SRC))
    import fam_common as fc

    FAM_ALIAS = {"qwen": "qwen", "nemotron": "llama", "lgai": "mistral"}
    linear = json.loads(Path(f"{DR}/ship/stack/LINEAR_SHIP.json").read_text())
    assert linear.get("ok"), "LINEAR_SHIP.json not ready"
    bundle = OUT / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for f in ["model.py", "fam_common.py", "encoders.py", "labeling.py", "models.txt"]:
        shutil.copy2(BUNDLE_SRC / f, bundle / f)

    # ---- shared artifacts ----------------------------------------------------------
    step("shared_labels")
    import pandas as pd
    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    df = pd.read_parquet(db, columns=["subject_key", "item_key", "label", "benchmark",
                                      "condition"])
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    if df.duplicated(["subject_key", "item_key"]).any():
        lab = (df.groupby(["subject_key", "item_key"], sort=False)["label"]
               .mean().reset_index())
    else:
        lab = df[["subject_key", "item_key", "label"]]

    # canonical item order = qwen items.parquet keys
    q_keys, _ = load_embeddings(f"{DR}/embeddings/{FAMILY_SLUG['qwen']}/items.parquet")
    canon_keys = [str(k) for k in q_keys]
    col_of = {k: j for j, k in enumerate(canon_keys)}
    sub_keys, _ = load_embeddings(f"{DR}/embeddings/{FAMILY_SLUG['qwen']}/subjects.parquet")
    sub_keys = [str(s) for s in sub_keys]
    row_of = {s: i for i, s in enumerate(sub_keys)}

    lab = lab[lab["item_key"].isin(col_of) & lab["subject_key"].isin(row_of)]
    rows = lab["subject_key"].map(row_of).to_numpy(np.int64)
    cols = lab["item_key"].map(col_of).to_numpy(np.int64)
    vals = lab["label"].to_numpy(np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(len(sub_keys) + 1, np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    shared_dir = bundle / "shared_artifacts"
    shared_dir.mkdir(exist_ok=True)
    np.savez_compressed(shared_dir / "passrate.npz", n_subjects=len(sub_keys),
                        n_items=len(canon_keys), indptr=indptr, indices=cols, data=vals)
    (shared_dir / "passrate_row.json").write_text(json.dumps(row_of))
    step("passrate_done", nnz=int(vals.size))

    # bc map + train counts (normalized-condition keys, id 0 reserved for unknown)
    bc_series = (df["benchmark"].astype(str) + "::"
                 + df["condition"].map(fc.normalize_condition))
    bc_counts = bc_series.value_counts()
    bc_to_id = {k: i + 1 for i, k in enumerate(sorted(bc_counts.index))}
    (shared_dir / "bc_to_id.json").write_text(json.dumps(bc_to_id))
    subj_counts = df["subject_key"].value_counts()
    (shared_dir / "train_counts.json").write_text(json.dumps(
        {"n_per_bc": {k: int(v) for k, v in bc_counts.items()},
         "n_per_subject": {k: int(v) for k, v in subj_counts.items()}}))

    # subject vocab (mlp ids) — must be identical across families
    vocab = None
    for fam in FAMS:
        vp = Path(f"{DR}/ship/exp_loo/{fam}/full_foldALL/shared/subj_vocab.json")
        v = json.loads(vp.read_text())
        if vocab is None:
            vocab = v
        elif v != vocab:
            raise RuntimeError(f"subj_vocab differs for {fam}")
    (shared_dir / "subj_vocab.json").write_text(json.dumps(vocab))
    (shared_dir / "stack_top.json").write_text(json.dumps(
        {"families": FAMS, "weights": linear["L3_cross_family"]["weights"],
         "bias": linear["L3_cross_family"]["bias"],
         "bce_cv": linear["L3_cross_family"]["bce_cv"]}, indent=1))
    step("shared_done")

    canon_names = canonical_dense_names()
    name_to_col = {n: i for i, n in enumerate(canon_names)}

    # ---- per family ------------------------------------------------------------------
    for fam in FAMS:
        fdir = bundle / fam / "artifacts"
        fdir.mkdir(parents=True, exist_ok=True)
        slug = FAMILY_SLUG[FAM_ALIAS[fam]]
        step(f"{fam}_embeddings")
        keys, emb = load_embeddings(f"{DR}/embeddings/{slug}/items.parquet")
        keys = [str(k) for k in keys]
        emb_unit = unit_rows(np.asarray(emb, np.float32))
        del emb

        # centroids (reuse exported if present, else re-fit seed 0 and persist)
        cpath = Path(f"{DR}/ship/ship_models/geom_{fam}_centroids.npz")
        if not cpath.exists():
            step(f"{fam}_fit_centroids")
            from aide.features.derive_cluster import fit_multi_kmeans
            cents = fit_multi_kmeans(emb_unit, {"coarse": 32, "fine": 256}, seed=0)
            np.savez_compressed(cpath, fine=np.asarray(cents["fine"], np.float32),
                                coarse=np.asarray(cents["coarse"], np.float32))
        shutil.copy2(cpath, fdir / "centroids.npz")
        with np.load(cpath) as z:
            fine = z["fine"].astype(np.float32)

        step(f"{fam}_cluster_aux")
        n = emb_unit.shape[0]
        assign = np.empty(n, np.int64)
        med_samples = []
        rng = np.random.default_rng(0)
        for s in range(0, n, 20000):
            d = _sqdist(emb_unit[s:s + 20000], fine)
            assign[s:s + 20000] = d.argmin(1)
            med_samples.append(d[rng.choice(d.shape[0], min(2000, d.shape[0]),
                                            replace=False)])
        sqd_scale = float(np.median(np.concatenate(med_samples)))
        sizes = np.bincount(assign, minlength=fine.shape[0])
        # canonical (passrate-column) order for item_to_cluster; family order for pq remap
        fam_pos = {k: i for i, k in enumerate(keys)}
        item_to_cluster = np.array([assign[fam_pos[k]] for k in canon_keys], np.int32)
        with np.load(f"{DR}/ship/ship_models/pqidx_{fam}.npz") as z:
            pq_keys = z["item_keys"].astype(str)
        pq_to_col = np.array([col_of[k] for k in pq_keys], np.int64)
        np.savez_compressed(fdir / "cluster_aux.npz", item_to_cluster=item_to_cluster,
                            sizes=sizes, sqd_scale=sqd_scale,
                            pq_to_passrate_col=pq_to_col)
        del emb_unit

        shutil.copy2(f"{DR}/ship/ship_models/pqidx_{fam}.npz", fdir / "pqidx.npz")
        shutil.copy2(f"{DR}/ship/ship_models/etbig_{fam}/shared/pca_item_emb.npz",
                     fdir / "pca_item_emb.npz")
        shutil.copy2(f"{DR}/ship/ship_models/etbig_{fam}/forest.npz",
                     fdir / "etbig_forest.npz")

        # mlp members (kept by the L1 refit) from the foldALL all-data fits
        fl = linear["families"][fam]
        kept = list(fl["L1_mlp_loo"]["kept_members"])
        mdir = bundle / fam / "artifacts" / "mlp"
        member_dense_cols = {}
        for cat in kept:
            src = Path(f"{DR}/ship/exp_loo/{fam}/full_foldALL/models/loo__{cat}")
            dst = mdir / f"loo__{cat}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "weights.npz", dst / "weights.npz")
            shutil.copy2(src / "meta.json", dst / "meta.json")
            meta = json.loads((src / "meta.json").read_text())
            member_dense_cols[f"loo__{cat}"] = [
                name_to_col[nm] for nm in meta["dense_feature_names"]]
        step(f"{fam}_mlp_members", kept=kept)

        rmeta = {"family": fam,
                 "encoder": ENCODER_META[fam],
                 "l1_members": [f"loo__{c}" for c in kept],
                 "l1_weights": fl["L1_mlp_loo"]["kept_weights"],
                 "l1_bias": fl["L1_mlp_loo"]["bias"],
                 "l2_weights": fl["L2_family"]["weights"],
                 "l2_bias": fl["L2_family"]["bias"],
                 "member_dense_cols": member_dense_cols}
        (fdir / "runtime_meta.json").write_text(json.dumps(rmeta, indent=1))
        step(f"{fam}_done")

    # ---- size report + zip -------------------------------------------------------------
    total = 0
    for p in sorted(bundle.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
    step("zipping", total_mb=round(total / 1e6, 1))
    zpath = OUT / "ensemble_3fam_linear"
    if Path(str(zpath) + ".zip").exists():
        Path(str(zpath) + ".zip").unlink()
    shutil.make_archive(str(zpath), "zip", bundle)
    zsize = Path(str(zpath) + ".zip").stat().st_size
    step("done", zip_mb=round(zsize / 1e6, 1), zip=str(zpath) + ".zip")
    print("EXPORT DONE", round(zsize / 1e6, 1), "MB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
