"""One-shot notebook refactor: remove legacy GBDT Member 2, wire metadata MLP."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "qwen8b_four_member_stacked.py"


def _replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"start marker not found: {start[:80]!r}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise RuntimeError(f"end marker not found after start: {end[:80]!r}")
    return text[:i] + replacement + text[j:]


CFG_M2_OLD = r'''# Member 2 v2 \(Task 3 of the diversification plan\): subject-mean residual
# GBDT trained on NON-EMBEDDING features only\..*?
CFG\["member2_v2"\]\.setdefault\("subject_mean_smoothing", 30\.0\)'''

CFG_M2_NEW = '''# Member 2: metadata-only GLU MLP (subject / bc / cluster embeddings +
# 14 mean-encoded marginals). No item embeddings -- structurally orthogonal
# to M1/M3/M4/M6.
CFG.setdefault("member2_mlp", {})
CFG["member2_mlp"].setdefault("d_subj", 32)
CFG["member2_mlp"].setdefault("d_bc", 32)
CFG["member2_mlp"].setdefault("d_cluster", 16)
CFG["member2_mlp"].setdefault("hid1", 256)
CFG["member2_mlp"].setdefault("hid2", 128)
CFG["member2_mlp"].setdefault("learning_rate", 1.0e-3)
CFG["member2_mlp"].setdefault("weight_decay", 1.0e-5)
CFG["member2_mlp"].setdefault("epochs", 40)
CFG["member2_mlp"].setdefault("batch_size", 16384)
CFG["member2_mlp"].setdefault("early_stopping_patience", 5)
CFG["member2_mlp"].setdefault("val_fraction", 0.1)
CFG["member2_mlp"].setdefault("cat_dropout_subject", 0.05)
CFG["member2_mlp"].setdefault("cat_dropout_bc", 0.10)
CFG["member2_mlp"].setdefault("cat_dropout_cluster", 0.10)
CFG["member2_mlp"].setdefault("feat_dropout", 0.10)'''

HEADER_M2_OLD = r'''   \* \*\*Member 2 \(LightGBM\)\*\*: a gradient-boosted decision-tree
     classifier trained offline on the dense `member_features`
     schema.*?pure-NumPy traversal at inference -- \*\*no `lightgbm` import in
     `model\.py`\*\*\.'''

HEADER_M2_NEW = '''   * **Member 2 (metadata MLP)**: GLU MLP on subject / benchmark /
     cluster IDs plus 14 mean-encoded marginals (no item embeddings).
     Pure-NumPy inference at runtime -- **no torch import in `model.py`**.'''

ARTIFACT_M2_OLD = "`artifacts/member2_gbdt/` -- Member 2 trees + bias + feature schema."
ARTIFACT_M2_NEW = (
    "`artifacts/member2_metadata_mlp/` -- Member 2 GLU MLP weights + marginal scaler."
)

PURE_M2_OLD = "  - `_pure/{gbdt,knn,logreg,stacker,nn_calibration,member_features}.py`"
PURE_M2_NEW = (
    "  - `_pure/{member2_metadata_mlp,knn,logreg,stacker,nn_calibration,member_features}.py`"
)

MEAN_ENC_BLOCK_OLD = r'''member2_feature_names = tuple\(member_feat_schema\.feature_names\) \+ tuple\(
    MEMBER2_INTERACTION_FEATURE_NAMES
\)
# ``_M2V2_ENABLED`` is defined further down in section 9b-ter; peek at
# the same CFG value here so we don't pre-allocate the legacy matrices
# when the Task 3 path is active\.
_m2v2_enabled_early = bool\(CFG\.get\("member2_v2", \{\}\)\.get\("enabled", True\)\)
if not _m2v2_enabled_early:
    X_train_dense_m2 = np\.concatenate\(
        \[X_train_dense, member2_interaction_train\], axis=1
    \)\.astype\(np\.float32, copy=False\)
    X_val_dense_m2 = np\.concatenate\(
        \[X_val_dense, member2_interaction_val\], axis=1
    \)\.astype\(np\.float32, copy=False\)
    print\(
        f"\[mean-enc\] Member 2 augmented X: "
        f"train \{X_train_dense_m2\.shape\}  val \{X_val_dense_m2\.shape\}  "
        f"\(was \{X_train_dense\.shape\[1\]\} cols, now \{X_train_dense_m2\.shape\[1\]\} cols\)"
    \)
else:
    X_train_dense_m2 = None
    X_val_dense_m2 = None
    print\(
        "\[mean-enc\] Member 2 augmented X: NOT built \(Task 3 path uses "
        "X_\*_dense_m2v2 instead -- saves ~19 GB resident memory\)\."
    \)'''

MEAN_ENC_BLOCK_NEW = '''# Member 2 uses metadata IDs + M4 marginals only (no dense embedding matrix).
X_train_dense_m2 = None
X_val_dense_m2 = None
print("[mean-enc] Member 2: metadata MLP path (no dense_m2 matrices).")'''

SECTION_9C = '''# %% [markdown]
# ## 9c. Train Member 2 (metadata GLU MLP)
#
# Small metadata-only MLP: subject / bc / cluster embeddings plus the
# same 14 mean-encoded marginals Member 4 uses. No item embedding columns
# so errors stay structurally orthogonal to M1/M3/M4/M6.

# %%
from src.member2_metadata_mlp import (
    apply_state_batch as m2_apply_state_batch,
    fit_member2_metadata_mlp,
)

_m2_cfg = CFG.get("member2_mlp", {})
_bc_keys_ordered = tuple(f"bc_{i}" for i in range(int(indexer.n_bc)))

_item_to_train_idx = {str(k): i for i, k in enumerate(train_item_keys)}
m2_holdout_item_id = np.fromiter(
    (
        _item_to_train_idx.get(str(k), -1)
        for k in primary.train["item_key"].astype(str).tolist()
    ),
    count=len(primary.train),
    dtype=np.int64,
)
print(
    f"[Member 2 MLP] item-cold holdout groups: "
    f"{int(np.unique(m2_holdout_item_id).size):,} unique items"
)

_m2_n_clusters = max(
    int(_N_CLUSTERS_ME),
    int(_mef_train_cluster.max()) + 1 if _mef_train_cluster.size else 0,
    int(_mef_val_cluster.max()) + 1 if _mef_val_cluster.size else 0,
)


def _fit_member2_mlp_global():
    print(
        "[Member 2 MLP] training metadata GLU MLP on full train "
        f"(N={len(primary.train):,}, marginals={member4_marginal_train.shape[1]})..."
    )
    return fit_member2_metadata_mlp(
        subject_ids=_mef_train_subj,
        bc_ids=_mef_train_bc,
        cluster_ids=_mef_train_cluster,
        marginals=member4_marginal_train.astype(np.float32, copy=False),
        y=y_train,
        subject_keys=_subject_keys_ordered,
        bc_keys=_bc_keys_ordered,
        marg_feature_names=MEMBER4_MARGINAL_FEATURE_NAMES,
        n_subjects=int(indexer.n_subjects),
        n_bcs=int(indexer.n_bc),
        n_clusters=int(_m2_n_clusters),
        d_subj=int(_m2_cfg.get("d_subj", 32)),
        d_bc=int(_m2_cfg.get("d_bc", 32)),
        d_cluster=int(_m2_cfg.get("d_cluster", 16)),
        hid1=int(_m2_cfg.get("hid1", 256)),
        hid2=int(_m2_cfg.get("hid2", 128)),
        learning_rate=float(_m2_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(_m2_cfg.get("weight_decay", 1.0e-5)),
        epochs=int(_m2_cfg.get("epochs", 40)),
        batch_size=int(_m2_cfg.get("batch_size", 16384)),
        val_fraction=float(_m2_cfg.get("val_fraction", 0.1)),
        early_stopping_patience=int(_m2_cfg.get("early_stopping_patience", 5)),
        cat_dropout_subject=float(_m2_cfg.get("cat_dropout_subject", 0.05)),
        cat_dropout_bc=float(_m2_cfg.get("cat_dropout_bc", 0.10)),
        cat_dropout_cluster=float(_m2_cfg.get("cat_dropout_cluster", 0.10)),
        feat_dropout=float(_m2_cfg.get("feat_dropout", 0.10)),
        seed=int(SEED),
        holdout_group_id=m2_holdout_item_id,
        show_progress=True,
    )


import hashlib as _m2_hashlib

_m2_marg_digest = _m2_hashlib.sha256(
    np.ascontiguousarray(member4_marginal_train, dtype=np.float32).tobytes()
).hexdigest()[:16]

member2_mlp_state = cache_or_compute(
    "member2_mlp_state",
    key_inputs=(
        "member2_metadata_mlp_v1",
        int(len(primary.train)), int(SEED),
        int(indexer.n_subjects), int(indexer.n_bc), int(_m2_n_clusters),
        int(member4_marginal_train.shape[1]),
        round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
        _m2_marg_digest,
        tuple(sorted(_m2_cfg.items())),
    ),
    compute_fn=_fit_member2_mlp_global,
)

p_member2_val = m2_apply_state_batch(
    member2_mlp_state,
    subject_ids=_mef_val_subj,
    bc_ids=_mef_val_bc,
    cluster_ids=_mef_val_cluster,
    marginals=member4_marginal_val.astype(np.float32, copy=False),
)
nll_m2 = float(
    -(ylab_val * np.log(np.clip(p_member2_val, 1e-6, 1 - 1e-6))
      + (1 - ylab_val) * np.log(1 - np.clip(p_member2_val, 1e-6, 1 - 1e-6))).mean()
)
print(f"[Member 2 MLP] val log-loss: {nll_m2:.6f}")
print(
    f"[Member 2 MLP] train/val NLL in state: "
    f"{member2_mlp_state.train_loss:.6f} / {member2_mlp_state.val_loss:.6f}"
)

'''

OOF_M2_BLOCK = '''    # ----- Fold Member 2 (metadata MLP) -----
    print(f"[OOF f{fold.fold_id}] Training fold Member 2 (metadata MLP)...")
    _m2_holdout_fold = np.array(
        [int(_item_to_train_idx.get(str(k), -1)) for k in fold_train_df["item_key"]],
        dtype=np.int64,
    )
    _m2_fold_n_clusters = max(
        int(_N_CLUSTERS_ME),
        int(_mef_cluster_fold_train.max()) + 1 if _mef_cluster_fold_train.size else 0,
        int(_mef_cluster_fold_oof.max()) + 1 if _mef_cluster_fold_oof.size else 0,
    )

    def _fit_fold_m2_mlp(_ff=fold):
        return fit_member2_metadata_mlp(
            subject_ids=_mef_subj_fold_train,
            bc_ids=_mef_bc_fold_train,
            cluster_ids=_mef_cluster_fold_train,
            marginals=fold_member4_marginal_train.astype(np.float32, copy=False),
            y=_y_fold_train,
            subject_keys=_subject_keys_ordered,
            bc_keys=_bc_keys_ordered,
            marg_feature_names=MEMBER4_MARGINAL_FEATURE_NAMES,
            n_subjects=int(indexer.n_subjects),
            n_bcs=int(indexer.n_bc),
            n_clusters=int(_m2_fold_n_clusters),
            d_subj=int(_m2_cfg.get("d_subj", 32)),
            d_bc=int(_m2_cfg.get("d_bc", 32)),
            d_cluster=int(_m2_cfg.get("d_cluster", 16)),
            hid1=int(_m2_cfg.get("hid1", 256)),
            hid2=int(_m2_cfg.get("hid2", 128)),
            learning_rate=float(_m2_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(_m2_cfg.get("weight_decay", 1.0e-5)),
            epochs=int(_m2_cfg.get("epochs", 40)),
            batch_size=int(_m2_cfg.get("batch_size", 16384)),
            val_fraction=float(_m2_cfg.get("val_fraction", 0.1)),
            early_stopping_patience=int(_m2_cfg.get("early_stopping_patience", 5)),
            cat_dropout_subject=float(_m2_cfg.get("cat_dropout_subject", 0.05)),
            cat_dropout_bc=float(_m2_cfg.get("cat_dropout_bc", 0.10)),
            cat_dropout_cluster=float(_m2_cfg.get("cat_dropout_cluster", 0.10)),
            feat_dropout=float(_m2_cfg.get("feat_dropout", 0.10)),
            seed=int(SEED) + 100 * (int(_ff.fold_id) + 1),
            holdout_group_id=_m2_holdout_fold,
            show_progress=False,
        )

    _fold_m2_state = cache_or_compute(
        "member2_mlp_oof_fold",
        key_inputs=(
            "member2_metadata_mlp_oof_v1",
            fold.fold_id, fold_suffix,
            int(len(fold.train_row_idx)), int(len(fold.oof_row_idx)),
            int(indexer.n_subjects), int(indexer.n_bc), int(_m2_fold_n_clusters),
            int(fold_member4_marginal_train.shape[1]),
            round(float(CFG.get("mean_encoded", {}).get("smoothing", 30.0)), 4),
            tuple(sorted(_m2_cfg.items())),
            int(SEED),
        ),
        compute_fn=_fit_fold_m2_mlp,
    )
    p2_oof_fold = m2_apply_state_batch(
        _fold_m2_state,
        subject_ids=_mef_subj_fold_oof,
        bc_ids=_mef_bc_fold_oof,
        cluster_ids=_mef_cluster_fold_oof,
        marginals=fold_member4_marginal_oof.astype(np.float32, copy=False),
    )
    p2_train_oof_acc.write_fold(fold.oof_row_idx, p2_oof_fold)

'''


def main() -> None:
    text = NB.read_text(encoding="utf-8")

    text = re.sub(HEADER_M2_OLD, HEADER_M2_NEW, text, count=1, flags=re.DOTALL)
    text = text.replace(ARTIFACT_M2_OLD, ARTIFACT_M2_NEW)
    text = text.replace(PURE_M2_OLD, PURE_M2_NEW)
    text = re.sub(CFG_M2_OLD, CFG_M2_NEW, text, count=1, flags=re.DOTALL)
    text = re.sub(MEAN_ENC_BLOCK_OLD, MEAN_ENC_BLOCK_NEW, text, count=1, flags=re.DOTALL)

    # Remove 9b-ter .. 9b''' (through pool helpers) up to 9c
    text = _replace_between(
        text,
        "# %% [markdown]\n# ## 9b-ter. Member 2 v2 setup",
        "# %% [markdown]\n# ## 9c. Train Member 2 (LightGBM)",
        SECTION_9C,
    )

    idx_9c_end = text.find("# %% [markdown]\n# ## 9d. Train Member 3")
    idx_9c_start = text.find("from src.gbdt_member import")
    if idx_9c_start >= 0 and idx_9c_start < idx_9c_end:
        text = text[:idx_9c_start] + text[idx_9c_end:]

    # OOF Member 2 block
    text = _replace_between(
        text,
        "    # ----- Fold Member 2 (GBDT residual) -----",
        "    # ----- Fold Member 3 (kNN-similarity) -----",
        OOF_M2_BLOCK,
    )

    # Remove 9.5c global GBDT refit
    text = _replace_between(
        text,
        "member2_v4_global_state = None",
        "# %% [markdown]\n# ## 9f. Train the stacker",
        "",
    )

    # OOF accumulator: drop subject_mean v2 block
    text = re.sub(
        r"# Task 3 \(Member 2 v2\): accumulate per-row OOF subject_mean.*?_gate3a_per_fold = \{\}\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Fold subject_mean / v2 feature block in OOF loop
    text = re.sub(
        r"    # ----- Fold subject_mean table \+ Gate 3a \+ Member 2 v2 features \(Task 3\) -----\n"
        r"    if _M2V2_ENABLED:.*?subject_mean_train_oof_acc\.write_fold\(fold\.oof_row_idx, subject_mean_oof_fold\)\n\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Simplify _need_fold_train_m1_anchor
    text = re.sub(
        r"        _need_fold_train_m1_anchor = \(\n"
        r"            \(not _M2V2_ENABLED\) and \(not _M2V5_ENABLED\)\n"
        r"        \) or \(_M2V4_ENABLED and not _M2V5_ENABLED\)\n"
        r"        if _need_fold_train_m1_anchor:\n"
        r"            p_a_anchor_fold_train = p_a_train\[fold\.train_row_idx\]\n"
        r"        else:\n"
        r"            p_a_anchor_fold_train = None\n",
        "        p_a_anchor_fold_train = None\n",
        text,
        count=1,
    )

    # Defer X_fold dense on Task 3 path -> always defer for M4-only JIT
    text = text.replace(
        "    if _M2V2_ENABLED:\n"
        "        print(\n"
        '            f"[OOF f{fold.fold_id}] DEFERRING X_fold_*_dense build "\n'
        '            "(Task 3 path: only Member 4 consumes them; built JIT below)."\n'
        "        )\n"
        "        X_fold_train_dense = None\n"
        "        X_fold_oof_dense = None\n"
        "    else:\n"
        "        print(f\"[OOF f{fold.fold_id}] Building fold X_*_dense (legacy path)...\")\n"
        "        X_fold_train_dense = _build_X(\n"
        "            fold_train_df, nn_train_mat_fold, _bc_redacted_fold_train,\n"
        "        )\n"
        "        X_fold_oof_dense = _build_X(\n"
        "            fold_oof_df, nn_oof_mat_fold, _bc_redacted_fold_oof,\n"
        "        )",
        "    print(\n"
        '        f"[OOF f{fold.fold_id}] DEFERRING X_fold_*_dense build "\n'
        '        "(Member 4 consumes them; built JIT below)."\n'
        "    )\n"
        "    X_fold_train_dense = None\n"
        "    X_fold_oof_dense = None",
    )

    # Fold cleanup block
    text = re.sub(
        r"    if _M2V2_ENABLED:\n"
        r"        # On v4 / v5 paths the v2 matrices were never materialized.*?"
        r"        del fold_subject_mean_table\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Finalize subject_mean oof
    text = re.sub(
        r"# Finalize Task 3 OOF subject_mean accumulator.*?f\"\{len\(_gate3a_per_fold\)\} folds, 0 violations, \"\n"
        r"        f\"max_abs_delta=\{_max_3a_delta:.2e\}\"\n"
        r"    \)\n\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Gate 3e
    text = re.sub(
        r"if _M2V2_ENABLED:\n"
        r"    _y64 = ylab_val\.astype\(np\.float64\).*?"
        r"            \"its decorrelation goal\.\"\n"
        r"        \)\n\n",
        '''_y64 = ylab_val.astype(np.float64)
_err_m1 = p_a_val.astype(np.float64) - _y64
_err_m2 = p_member2_val.astype(np.float64) - _y64
_corr_m2_m1 = float(np.corrcoef(_err_m2, _err_m1)[0, 1])
print(
    f"\\n[Gate 3e] Member 2 MLP vs Member 1 error correlation (val): "
    f"corr(err_m2, err_m1) = {_corr_m2_m1:+.4f}"
)
if abs(_corr_m2_m1) > 0.85:
    print(
        f"[Gate 3e] FLAG: |corr|={abs(_corr_m2_m1):.4f} > 0.85 -- Member 2 is still "
        "strongly correlated with Member 1."
    )
else:
    print(
        f"[Gate 3e] PASS: |corr|={abs(_corr_m2_m1):.4f} <= 0.85; Member 2 errors are "
        "sufficiently decorrelated from Member 1."
    )

''',
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Summary print
    text = text.replace(
        "print(f\"  Member 2 (LightGBM, val):           {nll_m2:.6f}\")",
        "print(f\"  Member 2 (metadata MLP, val):       {nll_m2:.6f}\")",
    )

    # Calibrator key
    text = text.replace(
        "state_fingerprint(gbdt_state),",
        "state_fingerprint(member2_mlp_state),",
    )

    # Export
    text = text.replace("gbdt_state=gbdt_state,", "member2_mlp_state=member2_mlp_state,")
    text = text.replace(
        "subject_mean_table=(subject_mean_table_global if _M2V2_ENABLED else None),",
        "mean_encoded_stats=mean_encoded_stats,",
    )
    text = text.replace(
        '          ``artifacts/{member2_gbdt,member3_knn,member4_logreg,',
        '          ``artifacts/{member2_metadata_mlp,member3_knn,member4_logreg,',
    )
    text = text.replace(
        "  KNOWN LIMITATION: Member 2 (GBDT) and Member 4 (LogReg) "
        "rely on the dense `member_features` schema. The runtime "
        "feature builder (`runtime_feature_builder_py`) is a TODO: "
        "without it, Members 2 & 4 emit their bias prediction at "
        "test time. The stacker still combines them, but their "
        "diversity contribution is reduced. Member 1 (IRT-MLP) and "
        "Member 3 (kNN) are fully wired.",
        "  Member 2 (metadata MLP) uses mean-encoded marginals shipped "
        "with the bundle. Member 4 still uses the dense member_features "
        "schema when runtime_feature_builder_py is provided.",
    )

    # RED-TEAM gbdt -> mlp
    text = text.replace("p2_nan = _g_apply_one(gbdt_state, nan_feats)", "# M2 smoke: metadata MLP uses ids+marginals at runtime")

    # Remove remaining _M2V* references (grep cleanup)
    for pat in [
        r"_M2V2_ENABLED",
        r"_M2V4_ENABLED",
        r"_M2V5_ENABLED",
        r"member2_v2",
        r"member2_v4",
        r"member2_v5",
        r"gbdt_state",
        r"gbdt_apply",
        r"gbdt_compose",
        r"fit_gbdt_member",
        r"MEMBER2_V5",
        r"MEMBER2_V3",
        r"M2V5_",
        r"M2V3_",
        r"member2_v5_",
        r"member2_v3_",
        r"X_val_dense_m2v2",
        r"subject_mean_table_global",
        r"subject_mean_train_oof",
    ]:
        if re.search(pat, text):
            print(f"WARNING: still contains {pat}")

    NB.write_text(text, encoding="utf-8")
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
