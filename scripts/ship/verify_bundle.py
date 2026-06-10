"""Verify the staged submission bundle against training-time ground truth (Colab GPU tab).

Three independent checks (env VERIFY_STAGE=encoders|features|pipeline|all):

1. encoders  — per family, load the bundle Encoder per runtime_meta and re-embed N random
   training items from their raw db fields; compare to the family's training cache parquet
   rows (cosine + max-abs). This is THE protocol test: the binding constraint is matching
   the cache the members were trained on (fp16 storage => cos >= ~0.999 expected). One
   encoder at a time (L4 fits one 8B bf16). For qwen (protocol uncertain) a small config
   sweep runs automatically if the primary config misses.

2. features  — per family, recompute the dense cluster blocks for N items via the bundle
   FamilyRuntime path using CACHE embeddings (no encoder) and compare the 525 geometry cols
   against the stored training shards (exact for centroid_distance/item_cluster; small fp
   for cluster_geometry; nn_geometry/nn-label cols are PQ-approximate BY DESIGN — reported,
   not asserted). Also: bundle MlpMember forward == src.mlp_member.apply_batch on the
   shipped foldALL members.

3. pipeline  — harness simulation with cache-backed embeddings (Encoder monkeypatched):
   _enqueue_for_batch + predict over ~40 (subject,item) train pairs + a fake `labeled`
   round; asserts probs in (0,1), calibrator refit runs, reports per-call latency.

Status/result: /content/verify_bundle_<stage>.json (+ mirror to DR/ship/stack/).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
REPO = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
BUNDLE = Path(os.environ.get("BUNDLE_DIR", "/content/bundle_out/bundle"))
STAGE = os.environ.get("VERIFY_STAGE", "all")
N_ITEMS = int(os.environ.get("VERIFY_N_ITEMS", "24"))
FAMS = ["qwen", "nemotron", "lgai"]
FAM_ALIAS = {"qwen": "qwen", "nemotron": "llama", "lgai": "mistral"}
STATUS = f"/content/verify_bundle_{STAGE}.json"
_t0 = time.time()
sys.path.insert(0, REPO)
sys.path.insert(0, str(BUNDLE))

_status: dict = {"stage_arg": STAGE}


def step(stage, **kw):
    _status.update(stage=stage, t_s=round(time.time() - _t0, 1), **kw)
    Path(STATUS + ".tmp").write_text(json.dumps(_status, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[verify] {stage} {kw if kw else ''}", flush=True)


def _sample_items(n):
    """n random training rows with raw fields + item_key, deterministic."""
    import pandas as pd
    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    df = pd.read_parquet(db, columns=["benchmark", "condition", "item_content",
                                      "item_key", "subject_key", "subject_content",
                                      "label"])
    for c in df.columns:
        df[c] = df[c].astype(str) if c != "label" else df[c]
    df = df.drop_duplicates("item_key")
    return df.sample(n=n, random_state=0)


def _cache_lookup(fam, item_keys):
    from aide.features.driver import FAMILY_SLUG, load_embeddings
    keys, emb = load_embeddings(
        f"{DR}/embeddings/{FAMILY_SLUG[FAM_ALIAS[fam]]}/items.parquet")
    pos = {str(k): i for i, k in enumerate(keys)}
    rows = [pos[k] for k in item_keys]
    return np.asarray(emb, np.float32)[rows]


def verify_encoders():
    import fam_common as fc
    import torch
    from encoders import Encoder
    res = {}
    df = _sample_items(N_ITEMS)
    for fam in FAMS:
        meta = json.loads((BUNDLE / fam / "artifacts" / "runtime_meta.json").read_text())
        enc_meta = meta["encoder"]
        cache_vecs = _cache_lookup(fam, df["item_key"].tolist())
        texts = [fc.item_text_for(b, fc.normalize_condition(c), ic,
                                  enc_meta.get("passage_prefix", ""))
                 for b, c, ic in zip(df["benchmark"], df["condition"],
                                     df["item_content"])]

        def trial(em):
            enc = Encoder(em, BUNDLE)
            v = enc.embed(texts)
            del enc
            torch.cuda.empty_cache()
            cs = np.sum(fc.unit_rows(v) * fc.unit_rows(cache_vecs), axis=1)
            scale = np.linalg.norm(v, axis=1) / np.clip(
                np.linalg.norm(cache_vecs, axis=1), 1e-9, None)
            return {"cos_min": float(cs.min()), "cos_mean": float(cs.mean()),
                    "norm_ratio_mean": float(scale.mean())}

        primary = trial(enc_meta)
        res[fam] = {"primary": primary, "config": enc_meta}
        step(f"enc_{fam}_primary", **primary)
        if primary["cos_min"] < 0.995 and fam == "qwen":
            sweeps = []
            for ps in ["right", "left"]:
                for ml in [1024, 4096]:
                    for nz in [False, True]:
                        em = dict(enc_meta, padding_side=ps, max_length=ml,
                                  l2_normalize=nz)
                        r = trial(em)
                        sweeps.append({"padding_side": ps, "max_length": ml,
                                       "l2_normalize": nz, **r})
                        step(f"enc_{fam}_sweep", **sweeps[-1])
            best = max(sweeps, key=lambda x: x["cos_min"])
            res[fam]["sweep_best"] = best
            if best["cos_min"] > primary["cos_min"] + 0.002:
                enc_meta.update({k: best[k] for k in
                                 ("padding_side", "max_length", "l2_normalize")})
                meta["encoder"] = enc_meta
                (BUNDLE / fam / "artifacts" / "runtime_meta.json").write_text(
                    json.dumps(meta, indent=1))
                res[fam]["patched"] = True
    _status["encoders"] = res
    step("encoders_done")
    return res


def verify_features():
    import fam_common as fc
    from aide.features.cache import FeatureCache
    from aide.features.store import FoldFeatureStore
    res = {}
    df = _sample_items(N_ITEMS)
    shared = _load_shared()
    GEOM = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster"]
    for fam in FAMS:
        rt = fc.FamilyRuntime(BUNDLE / fam, shared)
        cache_vecs = _cache_lookup(fam, df["item_key"].tolist())
        store = FoldFeatureStore(FeatureCache(f"{DR}/features", code_version="v2"),
                                 embedding_family=FAM_ALIAS[fam], seed=0, n_folds=3)
        geo0 = store.cache.read_shard(store._key(GEOM[0], "all"))
        gidx = {str(k): i for i, k in enumerate(geo0.row_ids)}
        Xg, _ = store.assemble(GEOM, fold=0, check_coverage=False)
        rows = [gidx[k] for k in df["item_key"]]
        stored = Xg[rows].astype(np.float32)
        ours = np.stack([rt.item_state(k, v)["geom525"]
                         for k, v in zip(df["item_key"], cache_vecs)])
        # column spans: cd 0:256 | clu_geo 256:265 | nn_geo 265:268 | item_cluster 268:525
        spans = {"centroid_distance": (0, 256), "cluster_geometry": (256, 265),
                 "nn_geometry": (265, 268), "item_cluster": (268, 525)}
        fr = {}
        for g, (a, b) in spans.items():
            d = np.abs(ours[:, a:b] - stored[:, a:b])
            fr[g] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
        # mlp member forward parity vs canonical implementation
        from src.mlp_member import MlpMemberState, apply_batch
        name = rt.meta["l1_members"][0]
        st = MlpMemberState.load(BUNDLE / fam / "artifacts" / "mlp" / name)
        m = rt.mlp_members[name]
        dense = np.random.default_rng(0).normal(size=(8, m.dense_dim)).astype(np.float32) \
            if m.dense_dim else None
        ie = cache_vecs[:8] if m.use_item_emb else None
        sid = np.arange(8, dtype=np.int64) if m.subj_emb_dim > 0 else None
        p_ref = apply_batch(st, subject_ids=sid, item_emb=ie, dense_X=dense)
        p_our = m.predict(subject_ids=sid, item_emb=ie, dense_X=dense)
        fr["mlp_forward_max_abs"] = float(np.max(np.abs(p_ref - p_our)))
        res[fam] = fr
        step(f"feat_{fam}", **{k: v for k, v in fr.items() if k != "member"})
    _status["features"] = res
    step("features_done")
    return res


def _load_shared():
    import fam_common as fc
    sd = BUNDLE / "shared_artifacts"
    return {"passrate": fc.CsrPassrate.load(sd / "passrate.npz"),
            "subj_vocab": {k: int(v) for k, v in
                           json.loads((sd / "subj_vocab.json").read_text()).items()},
            "passrate_row": {k: int(v) for k, v in
                             json.loads((sd / "passrate_row.json").read_text()).items()}}


def verify_pipeline():
    """Harness sim with cache-backed embeddings (no encoders loaded)."""
    import pandas as pd
    df = _sample_items(40)
    # monkeypatch Encoder BEFORE model import
    import encoders as enc_mod
    import fam_common as fc

    cache = {}
    for fam in FAMS:
        vecs = _cache_lookup(fam, df["item_key"].tolist())
        for k, v in zip(df["item_key"], vecs):
            cache[(fam, k)] = v

    class FakeEncoder:
        def __init__(self, meta, _bundle):
            self.meta = meta
            self.batch_size = 8
            self.model_id = meta["encoder_model_id"]
            self.fam = {"Qwen/Qwen3-Embedding-8B": "qwen",
                        "nvidia/llama-embed-nemotron-8b": "nemotron",
                        "annamodels/LGAI-Embedding-Preview": "lgai"}[self.model_id]
            self._bykey = {k2: v for (f, k2), v in cache.items() if f == self.fam}
            self._texts_to_key = {}

        def embed(self, texts):
            out = []
            for t in texts:
                k = self._texts_to_key.get(t)
                if k is None:   # match by content: find the row whose item_content is in t
                    k = next((ik for ik, ic in zip(df["item_key"], df["item_content"])
                              if str(ic) in t), None)
                    self._texts_to_key[t] = k
                out.append(self._bykey.get(k, np.zeros(4096, np.float32)))
            return np.stack(out)

    enc_mod.Encoder = FakeEncoder
    spec = importlib.util.spec_from_file_location("bundle_model", BUNDLE / "model.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules["bundle_model"] = model
    spec.loader.exec_module(model)
    step("pipeline_model_loaded")

    rows = df.to_dict("records")
    labeled = [{"benchmark": r["benchmark"], "condition": r["condition"],
                "subject_content": r["subject_content"], "item_content": r["item_content"],
                "label": int(float(r["label"]) > 0.5)} for r in rows[:10]]
    for r in rows:
        model._enqueue_for_batch(benchmark=r["benchmark"], condition=r["condition"],
                                 subject_content=r["subject_content"],
                                 item_content=r["item_content"])
    t0 = time.time()
    preds = []
    for r in rows:
        p = model.predict({"benchmark": r["benchmark"], "condition": r["condition"],
                           "subject_content": r["subject_content"],
                           "item_content": r["item_content"]}, labeled)
        assert isinstance(p, float) and 0.0 < p < 1.0, f"bad prob {p!r}"
        preds.append(p)
    dt = (time.time() - t0) / len(rows)
    out = {"n": len(preds), "mean_p": float(np.mean(preds)),
           "std_p": float(np.std(preds)), "per_call_ms": round(dt * 1000, 1),
           "calibrator_per_bc": len(model._CALIBRATOR.per_bc),
           "b_global": model._CALIBRATOR.b_global}
    _status["pipeline"] = out
    step("pipeline_done", **out)
    return out


if __name__ == "__main__":
    try:
        if STAGE in ("features", "all"):
            verify_features()
        if STAGE in ("pipeline", "all"):
            verify_pipeline()
        if STAGE in ("encoders", "all"):
            verify_encoders()       # last: loads/unloads 8B models
        _status["ok"] = True
        step("done")
        Path(f"{DR}/ship/stack/verify_bundle_{STAGE}.json").write_text(
            json.dumps(_status, indent=2, default=str))
        print("VERIFY DONE", flush=True)
    except Exception as e:
        _status.update(stage="ERROR", error=repr(e), tb=traceback.format_exc())
        Path(STATUS).write_text(json.dumps(_status, indent=1, default=str))
        print("VERIFY FAILED:", repr(e), flush=True)
        traceback.print_exc()
        raise
