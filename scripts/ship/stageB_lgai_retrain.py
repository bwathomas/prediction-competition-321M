"""Stage B of the lgai redo: re-derive feature shards from the POST-5.10 embeddings,
then retrain the lgai canon members (mlp-LOO / etbig / irt_bag) on them.

Preconditions (stage A = scripts/ship/reembed_lgai.py, must be DONE):
  - DR/embeddings/embedding_cache_lgai_preview_fa2/annamodels__LGAI-Embedding-Preview/
    items.parquet + subjects.parquet are the NEW (transformers-5.10, serve-exact) cache;
    the old cache is parked as *.pre510.bak siblings.

What this does (non-destructive):
  1. Park the OLD feature shards: DR/features/mistral -> DR/features/mistral_pre510.bak
     (refusing to run if the new embeddings are not newer than the parked cache).
  2. derive_family(family="mistral", code_version="v2") -> fresh shards in DR/features/mistral.
  3. Retrain canon members via exp_loo_category_mlp.py, sequential subprocesses, with
     SHIP_EXP_SAVE_ROOT pointed at DR/ship/exp_loo/lgai_post510/<canon tag>_fold<f> so the
     ORIGINAL canon dirs under exp_loo/lgai are untouched. Runs:
       mlp   SHIP_MODE=loo   folds 0,1,2   -> mlp_loo_fold{f}
       etbig SHIP_MODE=full  folds 0,1,2   -> etbig_full_fold{f}
       irt_bag SHIP_MODE=full folds 0,1,2  -> irt_bag_full_fold{f}
     (foldALL shipping fits can follow once the eval folds confirm sanity.)

Run on the lgai tab (A100):  python scripts/ship/stageB_lgai_retrain.py
Status: /content/stageB_lgai.json   Logs: /content/stageB_<tag>.log
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
REPO = os.environ.get("SHIP_REPO_ROOT", "/content/pc321")
EMB_DIR = (f"{DR}/embeddings/embedding_cache_lgai_preview_fa2/"
           f"annamodels__LGAI-Embedding-Preview")
FEAT_OLD = f"{DR}/features/mistral"
FEAT_BAK = f"{DR}/features/mistral_pre510.bak"
SAVE_BASE = f"{DR}/ship/exp_loo/lgai_post510"
STATUS = "/content/stageB_lgai.json"
_t0 = time.time()
sys.path.insert(0, REPO)


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[stageB] {stage} {kw if kw else ''}", flush=True)


def main():
    # ---- gates -----------------------------------------------------------------
    new_items = Path(f"{EMB_DIR}/items.parquet")
    old_bak = Path(f"{EMB_DIR}/items.parquet.pre510.bak")
    if not new_items.exists() or not old_bak.exists():
        raise RuntimeError("stage A artifacts missing: need items.parquet AND "
                           "items.parquet.pre510.bak (reembed_lgai must be DONE)")
    if new_items.stat().st_mtime <= old_bak.stat().st_mtime:
        raise RuntimeError("items.parquet is not newer than the .pre510.bak — "
                           "stage A did not complete?")
    step("gates_ok")

    # ---- 1. park old shards -------------------------------------------------------
    if os.path.isdir(FEAT_OLD) and not os.path.isdir(FEAT_BAK):
        shutil.move(FEAT_OLD, FEAT_BAK)
        step("features_parked", to=FEAT_BAK)
    elif os.path.isdir(FEAT_BAK):
        step("features_already_parked")
    else:
        step("no_old_features")

    # ---- 2. derive fresh shards -----------------------------------------------------
    from aide.features.driver import derive_family

    def prog(msg, frac=None):
        step("derive", msg=str(msg), frac=frac)

    derive_family(drive_root=DR, family="mistral", code_version="v2",
                  include_cluster=True, progress=prog)
    step("derive_done")

    # ---- 3. retrain canon members ----------------------------------------------------
    runs = []
    for f in [0, 1, 2]:
        runs.append(("mlp", {"SHIP_MODEL": "mlp", "SHIP_MODE": "loo",
                             "SHIP_OOF_FOLD": str(f)}, f"mlp_loo_fold{f}"))
    for f in [0, 1, 2]:
        runs.append(("etbig", {"SHIP_MODEL": "etbig", "SHIP_MODE": "full",
                               "SHIP_OOF_FOLD": str(f)}, f"etbig_full_fold{f}"))
    for f in [0, 1, 2]:
        runs.append(("irt_bag", {"SHIP_MODEL": "irt_bag", "SHIP_MODE": "full",
                                 "SHIP_OOF_FOLD": str(f)}, f"irt_bag_full_fold{f}"))

    results = {}
    for model, env_extra, tag in runs:
        env = dict(os.environ)
        env.update({"SHIP_FAMILY": "lgai", "SHIP_ROW_SOURCE": "full",
                    "SHIP_DRIVE_ROOT": DR, "SHIP_REPO_ROOT": REPO,
                    "SHIP_EXP_SAVE_ROOT": f"{SAVE_BASE}/{tag}"})
        env.update(env_extra)
        log = f"/content/stageB_{tag}.log"
        step("member_start", tag=tag)
        with open(log, "w") as fh:
            rc = subprocess.call(
                [sys.executable, f"{REPO}/scripts/ship/exp_loo_category_mlp.py"],
                stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        rj = Path(f"{SAVE_BASE}/{tag}/result.json")
        ok = rc == 0 and rj.exists()
        ll = None
        if rj.exists():
            try:
                ll = json.loads(rj.read_text()).get("soft_logloss")
            except Exception:
                pass
        results[tag] = {"rc": rc, "ok": ok, "soft_logloss": ll}
        step("member_done", tag=tag, rc=rc, ok=ok, soft_logloss=ll)
        if not ok:
            step("ABORT", failed=tag)
            raise RuntimeError(f"member run failed: {tag} rc={rc} (see {log})")

    step("done", results=results)
    print("STAGE B DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
