"""Quantify what FIXING the shipped ensemble bundle is worth, benchmark-cold.

ensemble_2fam_linear (submission 789686, -0.60) shipped stack_top.json fitted for
THREE families but only bundled qwen+nemotron: effective L3 slope sum = 0.79
(qwen 0.15319 + nemotron 0.64018), lgai's 0.24653 silently dropped, no temperature.
This sim replays that exact configuration on the cached member OOF preds and
measures, leave-one-benchmark-out (honest to an unseen benchmark):

  A   shipped 2-fam exact: stale weights {q .153, n .640}, bias -0.03578, no tau
  A+  A with the LOBO temperature 1.30 the report fitted (what shipping tau buys)
  B   intended 3-fam with the SAME stale weights (lgai member restored)
  B+  B with tau 1.30
  C   3-fam L3 REFIT (non-neg) on the other benchmarks' rows (proper cold fit)

Family probabilities are reconstructed with the SHIPPED linear L1 (per-category
LOO member preds from mlp_loo's P matrix, runtime_meta l1_weights) and L2
(mlp_L1 + etbig, runtime_meta l2_weights) — qwen/nemotron weights verbatim from
the submitted bundle; lgai's from DR/ship/stack/LINEAR_SHIP.json.

Run: python scripts/ship/ship_vs_fixed_sim.py   (CPU ~10 min)
Status: /content/ship_vs_fixed.json  Result: DR/ship/exp_cold/ship_vs_fixed_sim.json
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
FOLDS = [0, 1, 2]
EPS = 1e-7
STATUS = "/content/ship_vs_fixed.json"
OUT = f"{DR}/ship/exp_cold/ship_vs_fixed_sim.json"
_t0 = time.time()

# verbatim from the submitted bundle (ensemble_2fam_linear.zip)
SHIP = {
    "qwen": {"l1_members": ["item_embedding", "centroid_distance", "cluster_geometry",
                            "nn_geometry", "nn_label_derivatives", "cluster_passrate",
                            "cluster_subject", "counts_subject"],
             "l1_w": [0.35854, 0.11412, 0.05182, 0.09193, 0.16842, 0.02284, 0.02335,
                      0.12768], "l1_b": -0.00869,
             "l2_w": [0.60979, 0.43298], "l2_b": -0.03904},
    "nemotron": {"l1_members": ["item_embedding", "subject_embedding",
                                "centroid_distance", "cluster_geometry", "nn_geometry",
                                "item_cluster", "nn_label_derivatives",
                                "cluster_passrate", "cluster_subject", "counts_subject"],
                 "l1_w": [0.36904, 0.09878, 0.03736, 0.08779, 0.00766, 0.00988, 0.20148,
                          0.01709, 0.04473, 0.02285], "l1_b": -0.00701,
                 "l2_w": [0.69341, 0.34833], "l2_b": -0.03916},
}
L3_STALE = {"qwen": 0.15319, "nemotron": 0.64018, "lgai": 0.24653}
L3_BIAS = -0.03578
TAU_SHIP = 1.30


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[shipfix] {stage} {kw if kw else ''}", flush=True)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


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


def main():
    import pandas as pd

    # lgai L1/L2 from the refit-linear artifact (the bundle has no lgai dir)
    ls = json.loads(Path(f"{DR}/ship/stack/LINEAR_SHIP.json").read_text())
    lg_fr = ls["families"]["lgai"]
    SHIP["lgai"] = {"l1_members": lg_fr["L1_mlp_loo"]["kept_members"],
                    "l1_w": lg_fr["L1_mlp_loo"]["kept_weights"],
                    "l1_b": lg_fr["L1_mlp_loo"]["bias"],
                    "l2_w": [lg_fr["L2_family"]["weights"]["mlp_L1"],
                             lg_fr["L2_family"]["weights"]["etbig"]],
                    "l2_b": lg_fr["L2_family"]["bias"]}
    # cross-check qwen/nemotron LINEAR_SHIP vs the bundle-verbatim weights
    for fam in ["qwen", "nemotron"]:
        fr = ls["families"][fam]
        if fr["L1_mlp_loo"]["kept_members"] != [
                m for m in SHIP[fam]["l1_members"]]:
            step("WARN_weights_mismatch", fam=fam,
                 linear_ship=fr["L1_mlp_loo"]["kept_members"],
                 bundle=SHIP[fam]["l1_members"])
    step("weights_loaded", lgai_l1=SHIP["lgai"]["l1_members"])

    fam_p = {}
    y = items = None
    for fam in ["qwen", "nemotron", "lgai"]:
        zP, zet = [], []
        y_parts, item_parts = [], []
        cat_list = None
        for f in FOLDS:
            zm = np.load(f"{DR}/ship/exp_loo/{fam}/mlp_loo_fold{f}/preds/oof_preds.npz",
                         allow_pickle=False)
            ze = np.load(f"{DR}/ship/exp_loo/{fam}/etbig_full_fold{f}/preds/oof_preds.npz",
                         allow_pickle=False)
            cat_list = [str(c) for c in zm["cat_list"]]
            zP.append(np.asarray(zm["P"], np.float64))
            zet.append(ze["p_full"].astype(np.float64))
            if y is None:
                y_parts.append(zm["oof_y"].astype(np.float64))
                item_parts.append(zm["oof_items"].astype(str))
        if y is None:
            y = np.concatenate(y_parts)
            items = np.concatenate(item_parts)
        P = np.vstack(zP)
        et = np.concatenate(zet)
        s = SHIP[fam]
        # shipped linear L1 over the kept per-category LOO preds
        idxs = [cat_list.index(m) for m in s["l1_members"]]
        z1 = s["l1_b"] + sum(w * lg(P[:, i]) for w, i in zip(s["l1_w"], idxs))
        p1 = sg(z1)
        z2 = s["l2_b"] + s["l2_w"][0] * lg(p1) + s["l2_w"][1] * lg(et)
        fam_p[fam] = sg(z2)
        step("family_built", fam=fam, l2_bce=round(bce(y, fam_p[fam]), 5))

    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    bmap = pd.read_parquet(db, columns=["item_key", "benchmark"]).drop_duplicates("item_key")
    bdict = dict(zip(bmap["item_key"].astype(str), bmap["benchmark"].astype(str)))
    bench = np.array([bdict.get(i, "UNK") for i in items])
    benches = sorted(set(bench.tolist()) - {"UNK"})

    zq, zn, zl = lg(fam_p["qwen"]), lg(fam_p["nemotron"]), lg(fam_p["lgai"])
    z_A = L3_BIAS + L3_STALE["qwen"] * zq + L3_STALE["nemotron"] * zn
    z_B = z_A + L3_STALE["lgai"] * zl
    Z3 = np.column_stack([zq, zn, zl])

    res = {"variants": ["A_ship2_exact", "Aplus_tau130", "B_3fam_stale", "Bplus_tau130",
                        "C_3fam_refit_lobo"], "per_bench": {}}
    per = res["per_bench"]
    for b in benches:
        m = bench == b
        w, bias = fit_nonneg(Z3[~m], y[~m])
        zc = Z3[m] @ w + bias
        per[b] = {"n": int(m.sum()),
                  "A_ship2_exact": round(bce(y[m], sg(z_A[m])), 5),
                  "Aplus_tau130": round(bce(y[m], sg(TAU_SHIP * z_A[m])), 5),
                  "B_3fam_stale": round(bce(y[m], sg(z_B[m])), 5),
                  "Bplus_tau130": round(bce(y[m], sg(TAU_SHIP * z_B[m])), 5),
                  "C_3fam_refit_lobo": round(bce(y[m], sg(zc)), 5),
                  "C_weights": {k: round(float(v), 4)
                                for k, v in zip(["qwen", "nemotron", "lgai"], w)}}
        step("bench_done", bench=b, **{k: per[b][k] for k in res["variants"]})

    ns = np.array([per[b]["n"] for b in benches], dtype=np.float64)
    summ = {}
    for v in res["variants"]:
        vals = np.array([per[b][v] for b in benches])
        summ[f"{v}_mean"] = round(float(vals.mean()), 5)
        summ[f"{v}_rowweighted"] = round(float((vals * ns).sum() / ns.sum()), 5)
    res["summary"] = summ
    res["warm_bce_in_sample"] = {"A": round(bce(y, sg(z_A)), 5),
                                 "B": round(bce(y, sg(z_B)), 5)}
    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    step("done", **summ)
    print("SHIP VS FIXED SIM DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
