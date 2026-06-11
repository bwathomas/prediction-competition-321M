"""meta_hybrid (qwen8b Member-1, metadata-IRT-MLP) vs the canon stack.

Consumes the keyed OOF dump written by notebooks/m1_oof_only.py
(DR/ship/exp_loo/qwen/meta_hybrid_nbfolds/m1_oof_dump.npz; notebook's own
item-grouped 3-fold partition, seed 7) and the canon member preds (harness
item_fold partition). Joins on rid = "subject_key|item_key". Each column is
individually honest OOF (items disjoint from that column's training items),
so per-column correlation / stack evaluation on the intersection is valid;
partitions differ across columns, which can only make the stack ESTIMATE
slightly optimistic for the new member (recorded caveat).

Reports:
  1. m1 BCE/AUC on the intersection vs each canon column.
  2. Pearson corr of logit-preds and of residuals (p - y) vs each column.
  3. Non-neg linear stack with vs without m1 (GroupKFold(5, item) honest CV):
     delta BCE + the full-fit weight m1 earns.
  4. Per-benchmark table for m1 vs the LOBO-stack baseline (cold robustness).

Run:  python scripts/ship/meta_hybrid_corr_eval.py   (CPU, ~10 min, any tab)
Status: /content/meta_hybrid_corr.json
Result: DR/ship/exp_cold/meta_hybrid_corr_eval.json
"""
from __future__ import annotations

import glob
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
DUMP = f"{DR}/ship/exp_loo/qwen/meta_hybrid_nbfolds/m1_oof_dump.npz"
OUT = f"{DR}/ship/exp_cold/meta_hybrid_corr_eval.json"
STATUS = "/content/meta_hybrid_corr.json"
_t0 = time.time()


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[mh_corr] {stage} {kw if kw else ''}", flush=True)


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


def fit_nonneg(Z, y, w0=None):
    Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fg(w):
        p = np.clip(sg(Zb @ w), EPS, 1 - EPS)
        return (-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                Zb.T @ (p - y) / len(y))

    x0 = np.zeros(Zb.shape[1]) if w0 is None else w0
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * Z.shape[1] + [(None, None)],
                 options={"maxiter": 300})
    return r.x[:-1], float(r.x[-1])


def cv_bce(Z, y, groups, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(Z, y, groups):
        w, b = fit_nonneg(Z[tr], y[tr])
        oof[te] = sg(Z[te] @ w + b)
    a = auc(y, oof)
    return bce(y, oof), (round(a, 4) if a is not None else None)


def main():
    import pandas as pd

    step("load_dump")
    d = np.load(DUMP, allow_pickle=False)
    rid_m1 = np.char.add(np.char.add(d["subject_key"].astype(str), "|"),
                         d["item_key"].astype(str))
    p_m1 = d["p_m1_oof"].astype(np.float64)
    y_m1 = d["label"].astype(np.float64)
    m1_map = dict(zip(rid_m1.tolist(), zip(p_m1.tolist(), y_m1.tolist())))
    step("dump_loaded", n=len(rid_m1))

    step("load_members")
    cols, names = [], []
    y_parts, item_parts, subj_parts = [], [], []
    for fam in FAMS:
        for tag, key in [("mlp_loo", "stacked_oof"), ("etbig_full", "p_full"),
                         ("irt_bag_full", "p_full")]:
            parts = []
            for f in FOLDS:
                z = np.load(f"{DR}/ship/exp_loo/{fam}/{tag}_fold{f}/preds/oof_preds.npz",
                            allow_pickle=False)
                parts.append(z[key].astype(np.float64))
                if fam == FAMS[0] and tag == "mlp_loo":
                    y_parts.append(z["oof_y"].astype(np.float64))
                    item_parts.append(z["oof_items"].astype(str))
                    subj_parts.append(z["oof_subj"].astype(str))
            cols.append(np.concatenate(parts))
            names.append(f"{fam}.{tag}")
    y = np.concatenate(y_parts)
    items = np.concatenate(item_parts)
    subj = np.concatenate(subj_parts)
    P = np.column_stack(cols)
    step("members_loaded", n=len(y))

    rid = np.char.add(np.char.add(subj, "|"), items)
    hit = np.array([r in m1_map for r in rid.tolist()])
    idx = np.where(hit)[0]
    pm1 = np.array([m1_map[r][0] for r in rid[idx].tolist()])
    ym1 = np.array([m1_map[r][1] for r in rid[idx].tolist()])
    yh = y[idx]
    # label agreement gate: same (subject,item) must have ~the same soft label in both
    # universes (mean-dedup may differ slightly); large disagreement = misjoin.
    lab_diff = float(np.mean(np.abs(yh - ym1)))
    step("joined", n_intersect=len(idx),
         frac_of_harness=round(len(idx) / len(y), 4), label_mean_absdiff=lab_diff)
    if lab_diff > 0.02:
        raise RuntimeError(f"label mismatch on join: mean|dy|={lab_diff}")

    Ph = P[idx]
    res = {"n_intersect": int(len(idx)), "label_mean_absdiff": lab_diff,
           "caveat": "m1 OOF partition (notebook seed-7 item folds) differs from the "
                     "harness item_fold partition; per-column honesty holds, joint "
                     "stack estimate may be slightly optimistic for m1"}

    # 1. solo strength
    a1 = auc(yh, pm1)
    res["m1_solo"] = {"bce": round(bce(yh, pm1), 5),
                      "auc": (round(a1, 4) if a1 is not None else None)}
    res["members_solo"] = {}
    for j, nme in enumerate(names):
        aj = auc(yh, Ph[:, j])
        res["members_solo"][nme] = {"bce": round(bce(yh, Ph[:, j]), 5),
                                    "auc": (round(aj, 4) if aj is not None else None)}

    # 2. correlations
    zm1 = lg(pm1)
    rm1 = pm1 - yh
    res["corr_logit"] = {}
    res["corr_residual"] = {}
    for j, nme in enumerate(names):
        zj = lg(Ph[:, j])
        rj = Ph[:, j] - yh
        res["corr_logit"][nme] = round(float(np.corrcoef(zm1, zj)[0, 1]), 4)
        res["corr_residual"][nme] = round(float(np.corrcoef(rm1, rj)[0, 1]), 4)
    step("corr_done", **res["corr_logit"])

    # 3. stack with vs without m1
    g = items[idx]
    Z0 = lg(Ph)
    Z1 = np.column_stack([Z0, zm1])
    b0, a0 = cv_bce(Z0, yh, g)
    b1, a1s = cv_bce(Z1, yh, g)
    w_full, _b = fit_nonneg(Z1, yh)
    res["stack"] = {"bce_without_m1": round(b0, 5), "auc_without_m1": a0,
                    "bce_with_m1": round(b1, 5), "auc_with_m1": a1s,
                    "delta_bce": round(b1 - b0, 5),
                    "weights_with_m1": {(names + ["m1_meta_hybrid"])[i]:
                                        round(float(w_full[i]), 5)
                                        for i in range(len(w_full))}}
    step("stack_done", **{k: v for k, v in res["stack"].items() if k != "weights_with_m1"})

    # 4. per-benchmark cold table (m1 vs canon-stack-without-m1 LOBO-fit)
    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    bmap = pd.read_parquet(db, columns=["item_key", "benchmark"]).drop_duplicates("item_key")
    bdict = dict(zip(bmap["item_key"].astype(str), bmap["benchmark"].astype(str)))
    bench = np.array([bdict.get(i, "UNK") for i in items[idx]])
    per_bench = {}
    for b in sorted(set(bench.tolist()) - {"UNK"}):
        m = bench == b
        w_lobo, bias_lobo = fit_nonneg(Z0[~m], yh[~m])
        p_stack_b = sg(Z0[m] @ w_lobo + bias_lobo)
        am1 = auc(yh[m], pm1[m])
        ast = auc(yh[m], p_stack_b)
        per_bench[b] = {"n": int(m.sum()),
                        "m1_bce": round(bce(yh[m], pm1[m]), 5),
                        "m1_auc": (round(am1, 4) if am1 is not None else None),
                        "lobo_stack_bce": round(bce(yh[m], p_stack_b), 5),
                        "lobo_stack_auc": (round(ast, 4) if ast is not None else None)}
        step("bench", bench=b, m1=per_bench[b]["m1_bce"],
             stack=per_bench[b]["lobo_stack_bce"])
    res["per_bench"] = per_bench

    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    step("done")
    print("MH CORR EVAL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
