"""Single-model embedding-MLP trainer for ONE embedding family (nemotron or LGAI).

SHIP_PLAN_3WAY §D: "Train ONE model each on ALL data (no k-fold)". This script
fits exactly ONE ``src.mlp_member.fit_mlp_member`` (Member-8-style emb-MLP) on ALL
264,350 canonical train rows of the chosen family, predicts the 135,650 holdout
rows, and persists state + train/holdout preds to Drive.

It is written to be pasted into a SINGLE Colab cell and launched via the project's
``run_bg(name, fn)`` harness. ``fn()`` writes progress + final result JSON to
``/content/<fam>_mlp.json`` and returns the result dict.

----------------------------------------------------------------------------------
WHAT IT DOES (numbered to match the task spec)
  1. sys.path the repo at /content/pc321.
  2. Load family item + subject embeddings from
     DR/embeddings/<slug>/{items,subjects}.parquet via
     ``aide.features.driver.load_embeddings`` (returns (keys, emb_float32)).
       - nemotron slug = nvidia__llama-embed-nemotron-8b
       - LGAI slug     = embedding_cache_lgai_preview_fa2/annamodels__LGAI-Embedding-Preview
         (NESTED one level — taken verbatim from driver.FAMILY_SLUG).
  3. Load labels DR/prepared_datasets/measurement_db_prepared_*.parquet and the
     canonical per-row order from DR/ship/rows/{_tr_item,_tr_subj,_ho_item,_ho_subj}.npy
     (train N=264350, holdout N=135650).
  4. Build item_emb_unique (unique TRAIN item embeddings) + row_to_uniq
     (per-train-row index into item_emb_unique) + integer subject_ids (shared
     train+holdout vocabulary) + dense_X (DR/features/<fam> shards if present and
     row-aligned, else None → item_emb + learned subject embedding only).
  5. Train ONE fit_mlp_member on ALL train rows (no folds), device='cuda'.
  6. Predict holdout (chunked, never materialising [N, D]); save
     state + trainpred + holdpred to
     DR/ship/diverse/<fam>/mlp_{state.pkl,trainpred.npy,holdpred.npy}.

----------------------------------------------------------------------------------
ASSUMPTIONS (verify before trusting outputs)
  A1. DR/ship/rows/_tr_item.npy etc. hold PER-ROW string keys (object/<U dtype),
      length 264350 (train) / 135650 (holdout), in the canonical order shared by all
      existing per-row preds. _*_item = item_key per row, _*_subj = subject_key per
      row. (SHIP_PLAN: "canonical row order matching all existing preds".) The script
      asserts the lengths and aborts loudly on mismatch.
  A2. Labels live in the prepared_datasets parquet keyed by (subject_key, item_key)
      with an integer/bool 'label' in {0,1}. We join the canonical train rows onto
      that table to get y_train. If any train row is missing a label the script
      aborts (would otherwise train on garbage).
  A3. Holdout subjects are SEEN in train (item cold-start, not subject cold-start —
      confirmed in OVERNIGHT_WORKING_MEMORY). The subject vocabulary is built over
      train∪holdout subject_keys so the learned nn.Embedding covers both; unseen
      subject ids at apply time fall back to the UNK row inside apply_batch anyway.
  A4. dense_X is OFF by default. The 594-feat DR/features/<fam> shards are a per-OUTER-
      FOLD OOF-assembled structure (FoldFeatureStore, families qwen/llama/mistral);
      using them for a SINGLE no-fold model on ALL rows would either leak (fold='all'
      groups are fine, but the label-derived groups are fold-keyed) or need a full
      re-assembly that is out of scope here. So unless USE_DENSE is flipped True AND a
      plain row-aligned dense .npy is provided at DENSE_NPY_PATH, we ship item_emb +
      subject-embedding only (exactly the M8 default, which had the form-block OFF).
      This is the safe, leak-free choice for a no-OOF single model.
  A5. Family name → slug mapping uses driver.FAMILY_SLUG: 'nemotron' is an alias for
      driver family 'llama' (the nemotron embedding), 'lgai' is an alias for driver
      family 'mistral' (the LGAI embedding). FAM_ALIAS below makes this explicit.
  A6. Hyperparameters mirror the proven qwen M8 emb-MLP (subj_emb_dim=32, hid1=256,
      hid2=128, lr=1e-3, wd=1e-5, epochs=30, batch=16384, patience=5, dropout=0.10),
      with holdout_group_id = row_to_uniq so the internal val split is item-grouped
      (no item leakage across the trainer's internal train/val), matching M8.

RISKS the human must verify:
  * Exact dtype/semantics of the rows .npy (A1) — the only thing not re-derivable
    from repo code. If they are item-key ARRAYS the asserts pass; if they are integer
    INDICES instead, flip ROWS_ARE_KEYS=False (handled below) and supply the
    embedding/subject key universes they index into.
  * Label join key (A2): we assume columns 'subject_key','item_key','label'. If the
    prepared table lacks per-(subject,item) uniqueness the join is ambiguous — script
    asserts uniqueness.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

# ----------------------------------------------------------------------------------
# CONFIG — set FAMILY before launch ('nemotron' or 'lgai').
# ----------------------------------------------------------------------------------
FAMILY = os.environ.get("SHIP_FAMILY", "nemotron").strip().lower()  # 'nemotron' | 'lgai'

REPO_ROOT = "/content/pc321"
DRIVE_ROOT = "/content/drive/MyDrive/prediction-competition-321M"

N_TRAIN_EXPECTED = 264350
N_HOLD_EXPECTED = 135650

# Rows .npy hold per-row string keys (A1). Flip to False only if they are int indices.
ROWS_ARE_KEYS = True

# Dense channel (A4). OFF by default — leak-free for a no-OOF single model.
USE_DENSE = False
DENSE_NPY_PATH = None  # if USE_DENSE: a row-aligned {N_train,F} (+holdout) .npz with
#                        keys 'train' and 'holdout'; both float32, rows in canonical order.

# Family alias → (driver family used by FAMILY_SLUG, output dir slug).
FAM_ALIAS = {
    "nemotron": "llama",   # driver.FAMILY_SLUG['llama'] = nvidia__llama-embed-nemotron-8b
    "llama": "llama",
    "lgai": "mistral",     # driver.FAMILY_SLUG['mistral'] = ...LGAI-Embedding-Preview (NESTED)
    "mistral": "mistral",
}

# M8-proven hyperparameters (A6).
HP = dict(
    subj_emb_dim=32,
    hid1=256,
    hid2=128,
    learning_rate=1.0e-3,
    weight_decay=1.0e-5,
    epochs=30,
    batch_size=16384,
    early_stopping_patience=5,
    feat_dropout=0.10,
    val_fraction=0.1,
)
SEED = 0

STATUS_PATH_TMPL = "/content/{fam}_mlp.json"


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _write_status(fam, payload):
    try:
        with open(STATUS_PATH_TMPL.format(fam=fam), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception:  # never let status I/O kill the run
        pass


def _load_keys_npy(path):
    """Load a rows .npy as a list[str] of per-row keys (object/unicode → str)."""
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr).reshape(-1)
    return [str(x) for x in arr.tolist()]


# ----------------------------------------------------------------------------------
# the run_bg-friendly entry point
# ----------------------------------------------------------------------------------
def fn():
    fam = FAMILY
    t0 = time.time()
    prog = {"family": fam, "stage": "start", "ok": None, "t_start": t0}
    _write_status(fam, prog)

    def step(stage, **extra):
        prog.update(stage=stage, t_elapsed=round(time.time() - t0, 1), **extra)
        _write_status(fam, dict(prog))
        print(f"[{fam}-mlp] {stage}  (+{prog['t_elapsed']}s)", flush=True)

    try:
        # ---- (1) sys.path the repo --------------------------------------------------
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        step("imports")
        from aide.features.driver import FAMILY_SLUG, load_embeddings
        from src.mlp_member import (
            fit_mlp_member,
            apply_state_batch as mlp_apply_state_batch,
        )

        if fam not in FAM_ALIAS:
            raise ValueError(f"unknown FAMILY={fam!r}; expected one of {list(FAM_ALIAS)}")
        driver_fam = FAM_ALIAS[fam]
        slug = FAMILY_SLUG[driver_fam]
        emb_dir = f"{DRIVE_ROOT}/embeddings/{slug}"
        step("config", driver_family=driver_fam, slug=slug, use_dense=USE_DENSE)

        # ---- (2) load embeddings ----------------------------------------------------
        item_keys, item_emb = load_embeddings(f"{emb_dir}/items.parquet")
        subj_keys_emb, _subj_emb = load_embeddings(f"{emb_dir}/subjects.parquet")
        item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        D_EMB = int(item_emb.shape[1])
        item_emb_idx = {str(k): i for i, k in enumerate(item_keys)}
        step("loaded_embeddings", n_items=len(item_keys),
             n_subj_emb=len(subj_keys_emb), d_emb=D_EMB)

        # ---- (3) labels + canonical row order --------------------------------------
        import glob
        import pandas as pd  # Colab-only; this is offline training, not the runtime path
        db = glob.glob(
            f"{DRIVE_ROOT}/prepared_datasets/*measurement_db_prepared*.parquet"
        )[0]
        labels_df = pd.read_parquet(db, columns=["subject_key", "item_key", "label"])
        labels_df["subject_key"] = labels_df["subject_key"].astype(str)
        labels_df["item_key"] = labels_df["item_key"].astype(str)
        if labels_df.duplicated(["subject_key", "item_key"]).any():
            raise ValueError("labels parquet has duplicate (subject_key,item_key) rows "
                             "— join would be ambiguous (A2 violated)")
        label_map = {
            (s, i): int(l)
            for s, i, l in zip(
                labels_df["subject_key"].to_numpy(),
                labels_df["item_key"].to_numpy(),
                labels_df["label"].to_numpy(),
            )
        }

        rows_dir = f"{DRIVE_ROOT}/ship/rows"
        if ROWS_ARE_KEYS:
            tr_item = _load_keys_npy(f"{rows_dir}/_tr_item.npy")
            tr_subj = _load_keys_npy(f"{rows_dir}/_tr_subj.npy")
            ho_item = _load_keys_npy(f"{rows_dir}/_ho_item.npy")
            ho_subj = _load_keys_npy(f"{rows_dir}/_ho_subj.npy")
        else:  # rows are integer indices into the embedding/subject key universes
            tr_item = [item_keys[int(j)] for j in np.load(f"{rows_dir}/_tr_item.npy").reshape(-1)]
            ho_item = [item_keys[int(j)] for j in np.load(f"{rows_dir}/_ho_item.npy").reshape(-1)]
            tr_subj = [subj_keys_emb[int(j)] for j in np.load(f"{rows_dir}/_tr_subj.npy").reshape(-1)]
            ho_subj = [subj_keys_emb[int(j)] for j in np.load(f"{rows_dir}/_ho_subj.npy").reshape(-1)]

        N_tr, N_ho = len(tr_item), len(ho_item)
        if N_tr != N_TRAIN_EXPECTED or N_ho != N_HOLD_EXPECTED:
            raise ValueError(
                f"row-count mismatch: train {N_tr} (exp {N_TRAIN_EXPECTED}), "
                f"holdout {N_ho} (exp {N_HOLD_EXPECTED}) — wrong rows files / order (A1)")
        if not (len(tr_subj) == N_tr and len(ho_subj) == N_ho):
            raise ValueError("item/subject row arrays have mismatched lengths (A1)")
        step("loaded_rows", n_train=N_tr, n_holdout=N_ho)

        # ---- (3b) y_train via (subject,item) join ----------------------------------
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
            raise ValueError(f"{missing} train rows have no label in the prepared table "
                             "(A2 violated) — aborting rather than training on zeros")
        n_pos = int(y_train.sum())
        step("built_labels", n_pos=n_pos, pos_rate=round(n_pos / N_tr, 4))

        # ---- (4) item_emb_unique + row_to_uniq -------------------------------------
        # Unique TRAIN item keys (order of first appearance), their stacked embeddings,
        # and a per-train-row int index into that unique table.
        uniq_keys = []
        uniq_pos = {}
        row_to_uniq = np.empty(N_tr, dtype=np.int64)
        miss_emb = 0
        for r, k in enumerate(tr_item):
            j = uniq_pos.get(k)
            if j is None:
                j = len(uniq_keys)
                uniq_pos[k] = j
                uniq_keys.append(k)
            row_to_uniq[r] = j
        item_emb_unique = np.empty((len(uniq_keys), D_EMB), dtype=np.float32)
        for j, k in enumerate(uniq_keys):
            ii = item_emb_idx.get(str(k))
            if ii is None:
                miss_emb += 1
                item_emb_unique[j] = 0.0
            else:
                item_emb_unique[j] = item_emb[ii]
        if miss_emb:
            raise ValueError(f"{miss_emb} unique train items missing from the {fam} item "
                             "embedding cache — wrong family/slug or stale cache")

        # Subject vocabulary over train∪holdout (A3); int ids per row.
        subj_vocab = {}
        for s in tr_subj:
            if s not in subj_vocab:
                subj_vocab[s] = len(subj_vocab)
        for s in ho_subj:
            if s not in subj_vocab:
                subj_vocab[s] = len(subj_vocab)
        n_subjects = len(subj_vocab)
        train_sid = np.fromiter((subj_vocab[s] for s in tr_subj), dtype=np.int64, count=N_tr)
        hold_sid = np.fromiter((subj_vocab[s] for s in ho_subj), dtype=np.int64, count=N_ho)

        # dense_X (A4) — OFF unless explicitly enabled with a row-aligned .npz.
        dense_train = None
        dense_hold = None
        dense_names = ()
        if USE_DENSE and DENSE_NPY_PATH:
            dz = np.load(DENSE_NPY_PATH)
            dense_train = np.ascontiguousarray(dz["train"], dtype=np.float32)
            dense_hold = np.ascontiguousarray(dz["holdout"], dtype=np.float32)
            if dense_train.shape[0] != N_tr or dense_hold.shape[0] != N_ho:
                raise ValueError("dense .npz rows not aligned to canonical order (A4)")
            dense_names = tuple(f"dense_{i}" for i in range(dense_train.shape[1]))
            step("loaded_dense", dense_dim=int(dense_train.shape[1]))
        step("built_inputs", n_unique_items=len(uniq_keys), n_subjects=n_subjects,
             dense_dim=(0 if dense_train is None else int(dense_train.shape[1])))

        # ---- (5) train ONE model on ALL train rows (no folds) ----------------------
        step("fitting")
        state = fit_mlp_member(
            labels=y_train,
            subject_ids=train_sid,
            n_subjects=int(n_subjects),
            subj_emb_dim=int(HP["subj_emb_dim"]),
            item_emb_unique=item_emb_unique,
            row_to_uniq=row_to_uniq,
            dense_X=dense_train,
            dense_feature_names=dense_names,
            hid1=int(HP["hid1"]),
            hid2=int(HP["hid2"]),
            learning_rate=float(HP["learning_rate"]),
            weight_decay=float(HP["weight_decay"]),
            epochs=int(HP["epochs"]),
            batch_size=int(HP["batch_size"]),
            val_fraction=float(HP["val_fraction"]),
            early_stopping_patience=int(HP["early_stopping_patience"]),
            feat_dropout=float(HP["feat_dropout"]),
            seed=int(SEED) + 801,
            device="cuda",
            holdout_group_id=row_to_uniq,  # item-grouped internal val split (M8 parity)
            show_progress=True,
        )
        step("fitted", train_loss=float(state.train_loss), val_loss=float(state.val_loss))

        # ---- (6) predict train + holdout (chunked, never materialise [N, D]) -------
        def _apply_chunked(item_key_rows, sid_rows, dense_rows, chunk=131_072):
            keys = np.asarray(item_key_rows).astype(str)
            out = np.empty(int(keys.shape[0]), dtype=np.float32)
            for s in range(0, int(keys.shape[0]), int(chunk)):
                e = min(s + int(chunk), int(keys.shape[0]))
                emb = np.empty((e - s, D_EMB), dtype=np.float32)
                for j, k in enumerate(keys[s:e]):
                    emb[j] = item_emb[item_emb_idx[str(k)]]
                dz = None if dense_rows is None else dense_rows[s:e]
                out[s:e] = mlp_apply_state_batch(
                    state, subject_ids=sid_rows[s:e], item_emb=emb, dense_X=dz,
                )
            return out

        train_pred = _apply_chunked(tr_item, train_sid, dense_train)
        hold_pred = _apply_chunked(ho_item, hold_sid, dense_hold)
        if not (np.isfinite(train_pred).all() and np.isfinite(hold_pred).all()):
            raise ValueError("non-finite predictions — aborting save")
        step("predicted",
             train_pred_mean=round(float(train_pred.mean()), 5),
             hold_pred_mean=round(float(hold_pred.mean()), 5))

        # ---- save state + preds ----------------------------------------------------
        out_dir = f"{DRIVE_ROOT}/ship/diverse/{fam}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/mlp_state.pkl", "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(f"{out_dir}/mlp_trainpred.npy", train_pred.astype(np.float32))
        np.save(f"{out_dir}/mlp_holdpred.npy", hold_pred.astype(np.float32))

        # train log-loss as a sanity scalar (lower=better)
        p = np.clip(train_pred, 1e-6, 1 - 1e-6)
        train_ll = float(-(y_train * np.log(p) + (1 - y_train) * np.log(1 - p)).mean())

        result = {
            "family": fam,
            "driver_family": driver_fam,
            "slug": slug,
            "ok": True,
            "out_dir": out_dir,
            "n_train": N_tr,
            "n_holdout": N_ho,
            "n_unique_items": len(uniq_keys),
            "n_subjects": int(n_subjects),
            "d_emb": D_EMB,
            "dense_dim": (0 if dense_train is None else int(dense_train.shape[1])),
            "n_pos": n_pos,
            "val_loss": float(state.val_loss),
            "train_loss": float(state.train_loss),
            "train_logloss_full": round(train_ll, 6),
            "train_pred_mean": round(float(train_pred.mean()), 6),
            "hold_pred_mean": round(float(hold_pred.mean()), 6),
            "files": {
                "state": f"{out_dir}/mlp_state.pkl",
                "trainpred": f"{out_dir}/mlp_trainpred.npy",
                "holdpred": f"{out_dir}/mlp_holdpred.npy",
            },
            "t_total_s": round(time.time() - t0, 1),
        }
        prog.update(stage="done", **result)
        _write_status(fam, dict(prog))
        print(f"[{fam}-mlp] DONE — {json.dumps(result)}", flush=True)
        return result

    except Exception as exc:  # write the failure to the status file for poll()
        prog.update(stage="error", ok=False, error=repr(exc),
                    traceback=traceback.format_exc(),
                    t_total_s=round(time.time() - t0, 1))
        _write_status(fam, dict(prog))
        print(f"[{fam}-mlp] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
        raise


# ----------------------------------------------------------------------------------
# Launch pattern (paste into the Colab cell AFTER the run_bg/poll harness is defined):
#
#   # set the family for this runtime, then launch in the background:
#   import os; os.environ["SHIP_FAMILY"] = "nemotron"   # colab  host  → nemotron
#   # os.environ["SHIP_FAMILY"] = "lgai"                # colab3 host  → LGAI
#   run_bg(f"{os.environ['SHIP_FAMILY']}_mlp", fn)
#
#   # poll (tiny fast cell):
#   poll(f"{os.environ['SHIP_FAMILY']}_mlp")
#   # or read /content/<fam>_mlp.json directly.
#
# Standalone (foreground, NOT for heavy Colab use — blocks the kernel):
if __name__ == "__main__":
    fn()
