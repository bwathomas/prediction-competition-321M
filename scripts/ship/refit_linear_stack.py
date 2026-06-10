"""Refit the SHIPPING stack as pure LINEAR logit blends at every layer (user mandate).

Hierarchy (mirrors ensemble_3way_logit_fwls + one extra within-family layer):
  L1  per family: non-neg linear blend over the mlp LOO member preds (cat_list order);
      0-weight members are PRUNED (throughput: each kept member is one MLP forward at
      runtime) and the blend REFIT on the kept set.
  L2  per family: non-neg linear blend of [mlp_L1, etbig]. No irt_bag (fails benchmark
      holdout), no other members (0 weight in the canon).
  L3  across families: non-neg linear blend of the 3 family probabilities.

All fits: z = sum_j w_j * logit(p_j) + b, minimize soft BCE, w >= 0 (L-BFGS-B), bias free.
Honesty: each layer's INPUT preds are inner GroupKFold(5, groups=item) CV predictions of
the layer below, so no layer fits on preds that saw its rows. The SHIPPED weights are the
full-data refits at each layer. Rows = concat OOF folds 0,1,2 (every train row exactly once,
all families position-aligned per fold).

Output: DR/ship/stack/LINEAR_SHIP.json  {per-family L1 kept members+weights, L2 weights,
L3 weights+bias, honest OOF at each layer, comparison vs FINAL_CANON 0.42182}.
Run on any tab (CPU, ~10 min):  python scripts/ship/refit_linear_stack.py
Status: /content/refit_linear.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
FAMS = ["qwen", "nemotron", "lgai"]
FOLDS = [0, 1, 2]
EPS = 1e-7
STATUS = "/content/refit_linear.json"
_t0 = time.time()


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[refit] {stage} {kw if kw else ''}", flush=True)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_nonneg(Z, y, w0=None):
    """Non-neg weights + free bias over logit columns Z [n,k]; returns w[k], b."""
    Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fg(w):
        z = Zb @ w
        p = np.clip(sg(z), EPS, 1 - EPS)
        return (-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                Zb.T @ (p - y) / len(y))

    x0 = np.zeros(Zb.shape[1]) if w0 is None else w0
    bnds = [(0, None)] * Z.shape[1] + [(None, None)]
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                 options={"maxiter": 500})
    return r.x[:-1], float(r.x[-1])


def cv_preds(Z, y, groups, n_splits=5):
    """Honest inner-CV preds of the non-neg linear blend (each row predicted unseen)."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(Z, y, groups):
        w, b = fit_nonneg(Z[tr], y[tr])
        oof[te] = sg(Z[te] @ w + b)
    return np.clip(oof, EPS, 1 - EPS)


def main():
    res = {"folds": FOLDS, "families": {}}
    # ---- load per-fold data (verify cross-family row alignment) -------------------
    step("load")
    y_parts, item_parts = [], []
    P_cols = {f: [] for f in FAMS}        # mlp LOO P matrices
    et_cols = {f: [] for f in FAMS}
    cat_list = None
    for fold in FOLDS:
        ref_items = None
        for fam in FAMS:
            zm = np.load(f"{DR}/ship/exp_loo/{fam}/mlp_loo_fold{fold}/preds/oof_preds.npz",
                         allow_pickle=False)
            ze = np.load(f"{DR}/ship/exp_loo/{fam}/etbig_full_fold{fold}/preds/oof_preds.npz",
                         allow_pickle=False)
            items = zm["oof_items"].astype(str)
            if ref_items is None:
                ref_items = items
                y_parts.append(zm["oof_y"].astype(np.float64))
                item_parts.append(items)
            elif not (items == ref_items).all() or \
                    not (ze["oof_items"].astype(str) == ref_items).all():
                raise RuntimeError(f"row order mismatch fam={fam} fold={fold}")
            cl = [str(c) for c in zm["cat_list"]]
            if cat_list is None:
                cat_list = cl
            elif cl != cat_list:
                raise RuntimeError(f"cat_list mismatch fam={fam} fold={fold}: {cl}")
            P_cols[fam].append(np.asarray(zm["P"], np.float64))
            et_cols[fam].append(ze["p_full"].astype(np.float64))
    y = np.concatenate(y_parts)
    groups = np.concatenate(item_parts)
    P = {f: np.vstack(P_cols[f]) for f in FAMS}
    et = {f: np.concatenate(et_cols[f]) for f in FAMS}
    n = len(y)
    step("loaded", n_rows=n, cat_list=cat_list)

    fam_cv = {}
    for fam in FAMS:
        fr = {}
        # ---- L1: mlp LOO linear blend, prune 0-weight, refit ----------------------
        Z = lg(P[fam])
        w, b = fit_nonneg(Z, y)
        keep = [i for i, wi in enumerate(w) if wi > 1e-4]
        w2, b2 = fit_nonneg(Z[:, keep], y)
        # prune again if the refit zeroed more members
        keep2 = [keep[i] for i, wi in enumerate(w2) if wi > 1e-4]
        if keep2 != keep:
            keep = keep2
            w2, b2 = fit_nonneg(Z[:, keep], y)
        mlp_cv = cv_preds(Z[:, keep], y, groups)
        fr["L1_mlp_loo"] = {
            "all_weights": {cat_list[i]: round(float(w[i]), 5) for i in range(len(w))},
            "kept_members": [cat_list[i] for i in keep],
            "kept_weights": [round(float(x), 5) for x in w2],
            "bias": round(b2, 5),
            "bce_full_member_for_ref": bce(y, P[fam][:, cat_list.index("item_embedding")])
            if "item_embedding" in cat_list else None,
            "bce_cv": bce(y, mlp_cv),
        }
        step(f"{fam}_L1", kept=fr["L1_mlp_loo"]["kept_members"],
             bce=fr["L1_mlp_loo"]["bce_cv"])

        # ---- L2: family blend of mlp_L1 + etbig ------------------------------------
        Z2 = np.column_stack([lg(mlp_cv), lg(et[fam])])
        w3, b3 = fit_nonneg(Z2, y)
        fam_p_cv = cv_preds(Z2, y, groups)
        fr["L2_family"] = {"weights": {"mlp_L1": round(float(w3[0]), 5),
                                       "etbig": round(float(w3[1]), 5)},
                           "bias": round(b3, 5),
                           "bce_mlp_only": bce(y, mlp_cv),
                           "bce_etbig_only": bce(y, et[fam]),
                           "bce_cv": bce(y, fam_p_cv)}
        fam_cv[fam] = fam_p_cv
        res["families"][fam] = fr
        step(f"{fam}_L2", **fr["L2_family"])

    # ---- L3: cross-family ----------------------------------------------------------
    Z3 = np.column_stack([lg(fam_cv[f]) for f in FAMS])
    w4, b4 = fit_nonneg(Z3, y)
    final_cv = cv_preds(Z3, y, groups)
    res["L3_cross_family"] = {"weights": {f: round(float(w4[i]), 5)
                                          for i, f in enumerate(FAMS)},
                              "bias": round(b4, 5),
                              "bce_cv": bce(y, final_cv),
                              "bce_simple_logit_mean": bce(y, sg(Z3.mean(axis=1)))}
    res["reference"] = {"FINAL_CANON_nonneg_with_irt_and_gbm_mlp": 0.42182,
                        "old_ship": 0.43653}
    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    out = Path(f"{DR}/ship/stack/LINEAR_SHIP.json")
    out.write_text(json.dumps(res, indent=2))
    step("done", final_bce=res["L3_cross_family"]["bce_cv"])
    print("REFIT DONE", json.dumps(res["L3_cross_family"]), flush=True)


if __name__ == "__main__":
    main()
