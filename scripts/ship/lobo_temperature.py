"""LOBO temperature calibration for the shipped linear stack.

Reconstructs the runtime L1->L2->L3 logit z per training row from the honest fold-OOF
member preds + LINEAR_SHIP weights (exactly the bundle's math), attaches each row's
benchmark, then leave-one-benchmark-out:

  tau_b*    = argmin BCE on benchmark b of sigmoid(tau * z)   (descriptive cold slope)
  tau^(-b)  = median of {tau_c* : c != b}                     (what we'd ship not knowing b)
  gain_b    = BCE_b(tau=1) - BCE_b(tau^(-b))                  (honest held-out gain)

Shipped recommendation: TAU = median_b tau_b*. Intercept shifts are deliberately NOT
fitted here — the trc5 per-round calibrator owns intercepts via the labeled channel;
tau owns slope, which 25 labels/round cannot estimate.

Caveat (recorded in output): member preds are item-CV — each held-out benchmark's rows
were predicted by models that saw the benchmark's OTHER items, so tau_b* is an
optimistic (closer-to-1) floor on true benchmark-cold shrinkage.

Env: LOBO_FAMS="qwen,nemotron" (subset must exist in LINEAR_SHIP), SHIP_DRIVE_ROOT.
Output: DR/ship/stack/LOBO_TAU_<fams>.json ; status /content/lobo_tau.json
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
FAMS = [f.strip() for f in os.environ.get("LOBO_FAMS", "qwen,nemotron").split(",") if f.strip()]
FOLDS = [0, 1, 2]
EPS = 1e-7
STATUS = "/content/lobo_tau.json"
_t0 = time.time()


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[lobo] {stage} {kw if kw else ''}", flush=True)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_tau(z, y, iters=60):
    """1-param Newton: minimize BCE of sigmoid(tau*z)."""
    tau = 1.0
    for _ in range(iters):
        q = sg(tau * z)
        g = float(np.mean((q - y) * z))
        h = float(np.mean(q * (1 - q) * z * z))
        if h < 1e-12:
            break
        d = g / h
        tau -= d
        if abs(d) < 1e-9:
            break
    return float(min(max(tau, 0.05), 3.0))


def main():
    ship = json.loads(Path(f"{DR}/ship/stack/LINEAR_SHIP.json").read_text())
    assert ship.get("ok")
    step("load_oof", fams=FAMS)
    y_parts, item_parts = [], []
    famP, famET = {f: [] for f in FAMS}, {f: [] for f in FAMS}
    cat_list = None
    for fold in FOLDS:
        ref = None
        for fam in FAMS:
            zm = np.load(f"{DR}/ship/exp_loo/{fam}/mlp_loo_fold{fold}/preds/oof_preds.npz",
                         allow_pickle=False)
            ze = np.load(f"{DR}/ship/exp_loo/{fam}/etbig_full_fold{fold}/preds/oof_preds.npz",
                         allow_pickle=False)
            items = zm["oof_items"].astype(str)
            if ref is None:
                ref = items
                y_parts.append(zm["oof_y"].astype(np.float64))
                item_parts.append(items)
            else:
                assert (items == ref).all() and (ze["oof_items"].astype(str) == ref).all()
            cl = [str(c) for c in zm["cat_list"]]
            cat_list = cat_list or cl
            assert cl == cat_list
            famP[fam].append(np.asarray(zm["P"], np.float64))
            famET[fam].append(ze["p_full"].astype(np.float64))
    y = np.concatenate(y_parts)
    item_keys = np.concatenate(item_parts)
    step("loaded", n=len(y))

    # runtime math with LINEAR_SHIP weights (bundle-exact, incl. family subset)
    z_fam = {}
    for fam in FAMS:
        fl = ship["families"][fam]
        P = np.vstack(famP[fam])
        idx = [cat_list.index(c) for c in fl["L1_mlp_loo"]["kept_members"]]
        w1 = np.asarray(fl["L1_mlp_loo"]["kept_weights"], np.float64)
        p_mlp = sg(lg(P[:, idx]) @ w1 + fl["L1_mlp_loo"]["bias"])
        w2 = fl["L2_family"]["weights"]
        p_fam = sg(w2["mlp_L1"] * lg(p_mlp) + w2["etbig"] * lg(np.concatenate(famET[fam]))
                   + fl["L2_family"]["bias"])
        z_fam[fam] = lg(p_fam)
    L3w = ship["L3_cross_family"]["weights"]
    z = ship["L3_cross_family"]["bias"] + sum(L3w[f] * z_fam[f] for f in FAMS)
    step("stack_built", bce_tau1=bce(y, sg(z)))

    # benchmark per row
    import pandas as pd
    db = glob.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    bmap = pd.read_parquet(db, columns=["item_key", "benchmark"]).drop_duplicates("item_key")
    bdict = dict(zip(bmap["item_key"].astype(str), bmap["benchmark"].astype(str)))
    bench = np.array([bdict.get(k, "?") for k in item_keys])
    benches = sorted(set(bench) - {"?"})
    step("bench_attached", n_benches=len(benches), unmapped=int((bench == "?").sum()))

    res = {"fams": FAMS, "n_rows": len(y), "bce_tau1_overall": bce(y, sg(z)),
           "per_benchmark": {}}
    taus = {}
    for b in benches:
        m = bench == b
        taus[b] = fit_tau(z[m], y[m])
        res["per_benchmark"][b] = {"n": int(m.sum()), "tau_star": round(taus[b], 4),
                                   "bce_tau1": round(bce(y[m], sg(z[m])), 6)}
    # honest LOBO eval: apply median of the OTHER benchmarks' tau to b
    gains, w_gains, w_n = [], 0.0, 0
    for b in benches:
        m = bench == b
        others = [taus[c] for c in benches if c != b]
        t_hat = float(np.median(others))
        b1 = bce(y[m], sg(z[m]))
        bt = bce(y[m], sg(t_hat * z[m]))
        res["per_benchmark"][b].update(tau_lobo=round(t_hat, 4),
                                       bce_tau_lobo=round(bt, 6),
                                       gain=round(b1 - bt, 6))
        gains.append(b1 - bt)
        w_gains += (b1 - bt) * m.sum()
        w_n += int(m.sum())
    res["lobo_gain_mean_per_bench"] = round(float(np.mean(gains)), 6)
    res["lobo_gain_row_weighted"] = round(w_gains / max(w_n, 1), 6)
    res["tau_recommend"] = round(float(np.median(list(taus.values()))), 4)
    res["tau_range"] = [round(min(taus.values()), 4), round(max(taus.values()), 4)]
    res["caveat"] = ("member preds are item-CV; tau_star is an optimistic floor on "
                     "true benchmark-cold shrinkage")
    out = Path(f"{DR}/ship/stack/LOBO_TAU_{'_'.join(FAMS)}.json")
    out.write_text(json.dumps(res, indent=1))
    step("done", tau=res["tau_recommend"], gain_mean=res["lobo_gain_mean_per_bench"],
         gain_weighted=res["lobo_gain_row_weighted"])
    print("LOBO DONE", json.dumps({k: res[k] for k in
          ("tau_recommend", "tau_range", "lobo_gain_mean_per_bench",
           "lobo_gain_row_weighted")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
