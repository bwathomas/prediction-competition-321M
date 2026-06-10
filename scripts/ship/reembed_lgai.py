"""Stage A of the lgai redo: re-embed all lgai items+subjects under the CURRENT
transformers (5.10) semantics, using the submission bundle's own Encoder so the new
cache is serve-exact by construction.

Speed: texts are embedded in token-length-sorted order (Encoder.embed pads per
batch, so sorted input ≈ zero padding waste), then unsorted. Non-destructive: old
parquets + npy siblings are renamed to *.pre510.bak before the new ones land.

Env: SHIP_DRIVE_ROOT, SHIP_REPO_ROOT, BUNDLE_SRC (dir with encoders.py/fam_common.py),
LGAI_BATCH (default 16). Status: /content/reembed_lgai.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
REPO = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
BUNDLE_SRC = os.environ.get("BUNDLE_SRC", f"{REPO}/submission_bundle")
STATUS = "/content/reembed_lgai.json"
EMB_DIR = f"{DR}/embeddings/mistral"            # legacy slug holding the lgai family
BATCH = int(os.environ.get("LGAI_BATCH", "16"))
_t0 = time.time()
sys.path.insert(0, REPO)
sys.path.insert(0, BUNDLE_SRC)


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[reembed] {stage} {kw if kw else ''}", flush=True)


def embed_sorted(enc, texts):
    """Embed texts length-sorted (per-batch padding ≈ none), return in input order."""
    lens = [len(enc.tokenizer(t, truncation=True, max_length=enc.max_length)
                ["input_ids"]) for t in texts]
    order = np.argsort(np.asarray(lens, np.int64), kind="stable")
    out = np.empty((len(texts), 4096), np.float32)
    done = 0
    for s in range(0, len(order), 2048):
        idx = order[s:s + 2048]
        vecs = enc.embed([texts[i] for i in idx])
        out[idx] = vecs
        done += len(idx)
        if s % 8192 == 0 or done == len(order):
            rate = done / max(time.time() - _t0 - _emb_t0, 1e-9)
            step("embedding", done=done, total=len(order),
                 items_per_s=round(rate, 2),
                 eta_min=round((len(order) - done) / max(rate, 1e-9) / 60, 1))
    return out


def write_parquet(path, key_name, keys, emb):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({key_name: pa.array([str(k) for k in keys]),
                  "embedding": pa.array(list(emb.astype(np.float32)),
                                        type=pa.list_(pa.float32()))})
    tmp = "/content/_emb_tmp.parquet"
    pq.write_table(t, tmp)
    shutil.move(tmp, path)


def main():
    global _emb_t0
    import pandas as pd
    import fam_common as fc
    from encoders import Encoder
    from aide.features.embed_io import convert_embeddings_to_npy
    from aide.features.driver import load_embeddings

    step("load_db")
    import glob as _g
    db = _g.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    df = pd.read_parquet(db, columns=["benchmark", "condition", "item_content",
                                      "item_key", "subject_content", "subject_key"])
    for c in df.columns:
        df[c] = df[c].astype(str)
    items_df = df.drop_duplicates("item_key").set_index("item_key")
    subj_df = df.drop_duplicates("subject_key").set_index("subject_key")

    step("load_old_keys")
    old_item_keys, old_item_emb = load_embeddings(f"{EMB_DIR}/items.parquet")
    old_item_keys = [str(k) for k in old_item_keys]
    old_subj_keys, _old_se = load_embeddings(f"{EMB_DIR}/subjects.parquet")
    old_subj_keys = [str(k) for k in old_subj_keys]
    missing_i = [k for k in old_item_keys if k not in items_df.index]
    missing_s = [k for k in old_subj_keys if k not in subj_df.index]
    if missing_i or missing_s:
        raise RuntimeError(f"db coverage gap: {len(missing_i)} items, "
                           f"{len(missing_s)} subjects missing")
    step("keys_ok", n_items=len(old_item_keys), n_subjects=len(old_subj_keys))

    # the pinned lgai protocol (== runtime_meta.json "encoder" block), batch bumped
    meta = {"encoder_model_id": "annamodels/LGAI-Embedding-Preview", "max_length": 4096,
            "pooling": "last_token", "pool_fp32": False, "l2_normalize": False,
            "batch_size": BATCH, "passage_prefix": ""}
    enc = Encoder(meta, BUNDLE_SRC)
    step("encoder_loaded", device=enc.device)

    # subjects first (cheap; raw subject_content per src/embeddings.subject_text)
    _emb_t0 = time.time() - _t0
    subj_texts = [subj_df.loc[k, "subject_content"] for k in old_subj_keys]
    subj_emb = embed_sorted(enc, subj_texts)
    step("subjects_done", shape=list(subj_emb.shape))

    item_texts = [fc.item_text_for(items_df.loc[k, "benchmark"],
                                   fc.normalize_condition(items_df.loc[k, "condition"]),
                                   items_df.loc[k, "item_content"], "")
                  for k in old_item_keys]
    _emb_t0 = time.time() - _t0
    item_emb = embed_sorted(enc, item_texts)
    step("items_done", shape=list(item_emb.shape))

    # gates: finite, non-degenerate, and measurably different from the old cache
    assert np.isfinite(item_emb).all() and np.isfinite(subj_emb).all()
    norms = np.linalg.norm(item_emb, axis=1)
    assert (norms > 1e-3).all(), "zero embeddings"
    sample = np.random.default_rng(0).choice(len(old_item_keys), 256, replace=False)
    ov = np.asarray(old_item_emb, np.float32)[sample]
    nv = item_emb[sample]
    cos = (ov * nv).sum(1) / np.clip(np.linalg.norm(ov, axis=1)
                                     * np.linalg.norm(nv, axis=1), 1e-9, None)
    step("old_vs_new_cos", mean=float(cos.mean()), min=float(cos.min()),
         max=float(cos.max()))
    del old_item_emb

    step("backup_old")
    for f in Path(EMB_DIR).iterdir():
        if f.suffix in (".parquet", ".npy", ".json") and ".pre510" not in f.name:
            shutil.move(str(f), str(f) + ".pre510.bak")

    step("write_new")
    write_parquet(f"{EMB_DIR}/items.parquet", "item_key", old_item_keys, item_emb)
    write_parquet(f"{EMB_DIR}/subjects.parquet", "subject_key", old_subj_keys, subj_emb)
    convert_embeddings_to_npy(f"{EMB_DIR}/items.parquet", overwrite=True)
    convert_embeddings_to_npy(f"{EMB_DIR}/subjects.parquet", overwrite=True)

    # readback gate
    k2, e2 = load_embeddings(f"{EMB_DIR}/items.parquet")
    assert [str(k) for k in k2] == old_item_keys, "readback key mismatch"
    rb = np.asarray(e2[:64], np.float32)
    rc = (rb * item_emb[:64]).sum(1) / np.clip(
        np.linalg.norm(rb, axis=1) * np.linalg.norm(item_emb[:64], axis=1), 1e-9, None)
    assert rc.min() > 0.999, f"readback cos {rc.min()}"  # f16 roundtrip tolerance
    step("done", n_items=len(old_item_keys), readback_cos_min=float(rc.min()))
    print("REEMBED DONE", flush=True)


if __name__ == "__main__":
    _emb_t0 = 0.0
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
