"""LOBO alpha probe — where is the remaining COLD alpha?

Motivation: ensemble_2fam_linear scored -0.60 on the leaderboard, a tie (at 2-dp
rounding) with single-model submissions, despite a 0.013 warm-OOF advantage. The
hidden test is benchmark-shifted; item-grouped OOF overstates transfer. This probe
quantifies, from CACHED member predictions only (no retraining), how much BCE/AUC
is recoverable on a held-out benchmark from serve-time-feasible mechanisms:

  v0  LOBO stack: non-neg linear stack over the 9 canon columns
      (mlp_loo stacked_oof + etbig p_full + irt_bag p_full, x 3 families),
      weights fit with the benchmark EXCLUDED. Honest cold floor (member preds
      are item-CV, so still a floor — recorded caveat).
  v1  + global temperature (median of other benchmarks' tau*, as shipped).
  v2  + per-bench intercept fit on a REVEALED sample of the benchmark's rows
      (the shipped per-round calibrator's mechanism).
  v3  + per-bench slope (temperature) from the same revealed rows.
  v4  + ridge-shrunk PER-SUBJECT offsets within the benchmark (subject x bench
      interaction — the shipped calibrator does NOT do this).
  v5  + kNN-residual corrector: for each unrevealed row, similarity-weighted mean
      logit-residual of the K nearest REVEALED items (same benchmark, item
      embedding cosine), count-shrunk. Serve-time feasible (index already ships).

Reveal budgets r in {0.5%, 1%, 2%, 5%, 10%} of the benchmark's rows (seeded).
Metrics on the UNREVEALED remainder: soft BCE + AUC (labels binarized at 0.5;
rows with 0<y<1 kept for BCE, only y==0/y==1 used for AUC to avoid thresholding
ambiguity). Also: per-benchmark per-member cold table (which members hold up),
and the warm-fit reference (weights fit INCLUDING the benchmark) for the
warm-vs-cold delta.

Inputs (Drive): DR/ship/exp_loo/<fam>/{mlp_loo,etbig_full,irt_bag_full}_fold{0,1,2}/
preds/oof_preds.npz, prepared measurement db parquet (benchmark per item), and
DR/embeddings/Qwen__Qwen3-Embedding-8B items for the kNN corrector.

Run (any tab, ~30-60 min, GPU optional):  python scripts/ship/lobo_alpha_probe.py
Status: /content/lobo_alpha_probe.json   Result: DR/ship/exp_cold/lobo_alpha_probe.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DR = os.environ.get("SHIP_DRIVE_ROOT", "/content/drive/MyDrive/prediction-competition-321M")
FAMS = ["qwen", "nemotron", "lgai"]
FOLDS = [0, 1, 2]
EPS = 1e-7
REVEAL_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]
KNN_K = 16
SEED = 0
STATUS = "/content/lobo_alpha_probe.json"
OUT = f"{DR}/ship/exp_cold/lobo_alpha_probe.json"
_t0 = time.time()


def step(stage, **kw):
    d = {"stage": stage, "t_s": round(time.time() - _t0, 1), **kw}
    Path(STATUS + ".tmp").write_text(json.dumps(d, indent=1, default=str))
    os.replace(STATUS + ".tmp", STATUS)
    print(f"[lobo_alpha] {stage} {kw if kw else ''}", flush=True)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    """AUC on the hard rows only (y exactly 0 or 1). None if degenerate."""
    hard = (y == 0.0) | (y == 1.0)
    yb, pb = y[hard], p[hard]
    n1, n0 = int((yb == 1).sum()), int((yb == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = np.argsort(np.argsort(pb)) + 1  # ranks 1..n (ties: arbitrary, fine at this scale)
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


def fit_tau(z, y, iters=60):
    """BCE-optimal temperature (slope on logit), Newton."""
    t = 1.0
    for _ in range(iters):
        p = np.clip(sg(t * z), EPS, 1 - EPS)
        g = float(np.mean((p - y) * z))
        h = float(np.mean(p * (1 - p) * z * z))
        if h < 1e-12:
            break
        t_new = t - g / h
        if not np.isfinite(t_new) or abs(t_new - t) < 1e-7:
            t = max(t_new, 1e-3) if np.isfinite(t_new) else t
            break
        t = max(t_new, 1e-3)
    return t


def fit_intercept(z, y, ridge=0.0, iters=40):
    """BCE-optimal additive shift on logit, Newton with optional ridge to 0."""
    c = 0.0
    for _ in range(iters):
        p = np.clip(sg(z + c), EPS, 1 - EPS)
        g = float(np.mean(p - y)) + ridge * c
        h = float(np.mean(p * (1 - p))) + ridge
        c_new = c - g / h
        if abs(c_new - c) < 1e-8:
            return c_new
        c = c_new
    return c


def main():
    import pandas as pd

    res = {"caveat": "member preds are item-CV; each held-out benchmark's rows were "
                     "predicted by models that saw its sibling items => cold deltas "
                     "are FLOORS on the true benchmark-cold gap",
           "reveal_fracs": REVEAL_FRACS, "knn_k": KNN_K, "seed": SEED}

    # ---- load the 9 canon columns, position-aligned ------------------------------
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
    n = len(y)
    P = np.column_stack(cols)
    if not all(len(c) == n for c in cols):
        raise RuntimeError("column length mismatch")
    step("loaded", n_rows=n, n_cols=len(names), names=names)

    # ---- benchmark per row ---------------------------------------------------------
    import glob as _g
    db = _g.glob(f"{DR}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    bmap = pd.read_parquet(db, columns=["item_key", "benchmark"]).drop_duplicates("item_key")
    bdict = dict(zip(bmap["item_key"].astype(str), bmap["benchmark"].astype(str)))
    bench = np.array([bdict.get(i, "UNK") for i in items])
    benches = sorted(set(bench.tolist()) - {"UNK"})
    step("bench_mapped", n_bench=len(benches),
         unk_rows=int((bench == "UNK").sum()))

    Z = lg(P)

    # ---- per-member cold table -----------------------------------------------------
    member_table = {}
    for b in benches:
        m = bench == b
        member_table[b] = {"n": int(m.sum()), "y_mean": round(float(y[m].mean()), 4)}
        for j, name in enumerate(names):
            a = auc(y[m], P[m, j])
            member_table[b][name] = {"bce": round(bce(y[m], P[m, j]), 5),
                                     "auc": (round(a, 4) if a is not None else None)}
    res["member_cold_table"] = member_table
    step("member_table_done")

    # ---- warm reference: weights fit on ALL rows ------------------------------------
    w_all, b_all = fit_nonneg(Z, y)
    p_warm_all = sg(Z @ w_all + b_all)
    res["warm_global"] = {
        "weights": {names[i]: round(float(w_all[i]), 5) for i in range(len(names))},
        "bias": round(b_all, 5), "bce_in_sample": round(bce(y, p_warm_all), 5)}
    step("warm_global_done", bce=res["warm_global"]["bce_in_sample"])

    # ---- item embeddings for kNN corrector (qwen family) ----------------------------
    # cache convention (aide.features.embed_io.npy_paths): items.f16.npy + items.keys.json
    # siblings of items.parquet — our own Drive artifacts (trusted, no pickle involved).
    step("load_embeddings")
    emb_dir = f"{DR}/embeddings/Qwen__Qwen3-Embedding-8B"
    npy_p, keys_p = f"{emb_dir}/items.f16.npy", f"{emb_dir}/items.keys.json"
    if not (os.path.exists(npy_p) and os.path.exists(keys_p)):
        import sys as _sys
        _sys.path.insert(0, os.environ.get("SHIP_REPO_ROOT", "/content/pc321"))
        from aide.features.embed_io import convert_embeddings_to_npy
        convert_embeddings_to_npy(f"{emb_dir}/items.parquet")
    ik = np.array(json.loads(Path(keys_p).read_text(encoding="utf-8")), dtype=str)
    iv = np.load(npy_p, mmap_mode="r")
    ipos = {k: i for i, k in enumerate(ik)}
    step("embeddings_loaded", n_items=len(ik), dim=int(iv.shape[1]))

    rng = np.random.default_rng(SEED)
    per_bench = {}
    taus_star = {}

    # descriptive per-bench tau* on the warm stack (for the v1 median-of-others)
    z_warm = Z @ w_all + b_all
    for b in benches:
        m = bench == b
        taus_star[b] = fit_tau(z_warm[m], y[m])
    res["tau_star_warmstack"] = {b: round(t, 4) for b, t in taus_star.items()}

    for bi, b in enumerate(benches):
        m = bench == b
        idx_b = np.where(m)[0]
        n_b = len(idx_b)
        # LOBO stack: fit weights without this benchmark
        w, bias = fit_nonneg(Z[~m], y[~m], w0=np.append(w_all, b_all))
        z_b = Z[m] @ w + bias
        y_b = y[m]
        subj_b = subj[m]
        item_b = items[m]
        tau_med = float(np.median([taus_star[o] for o in benches if o != b]))

        entry = {"n": n_b,
                 "lobo_weights": {names[i]: round(float(w[i]), 5)
                                  for i in range(len(names)) if w[i] > 1e-4},
                 "tau_median_others": round(tau_med, 4)}
        a0 = auc(y_b, sg(z_b))
        entry["v0_lobo_stack"] = {"bce": round(bce(y_b, sg(z_b)), 5),
                                  "auc": (round(a0, 4) if a0 is not None else None)}
        entry["warmfit_on_bench"] = {"bce": round(bce(y_b, p_warm_all[m]), 5)}
        z_t = tau_med * z_b
        entry["v1_tau"] = {"bce": round(bce(y_b, sg(z_t)), 5)}

        # embedding rows for this benchmark's items (for v5)
        eb_pos = np.array([ipos.get(k, -1) for k in item_b])
        have_emb = eb_pos >= 0

        entry["reveal"] = {}
        for r in REVEAL_FRACS:
            n_rev = max(int(round(r * n_b)), 20)
            if n_rev >= n_b:
                continue
            rev = np.zeros(n_b, dtype=bool)
            rev[rng.choice(n_b, n_rev, replace=False)] = True
            te = ~rev
            zr, yr = z_t[rev], y_b[rev]
            zt, yt = z_t[te], y_b[te]

            # v2 intercept
            c = fit_intercept(zr, yr, ridge=20.0 / max(len(yr), 1))
            p2 = sg(zt + c)
            # v3 + slope
            t_b = fit_tau(zr + c, yr)
            t_b = float(np.clip(t_b, 0.5, 2.0))
            p3 = sg(t_b * (zt + c))
            z3r = t_b * (zr + c)
            z3t = t_b * (zt + c)
            # v4 + per-subject offsets (ridge-shrunk newton per subject)
            su, sinv_r = np.unique(subj_b[rev], return_inverse=True)
            soff = {}
            for k_s, s_name in enumerate(su):
                rows = sinv_r == k_s
                if rows.sum() < 3:
                    continue
                soff[s_name] = fit_intercept(z3r[rows], yr[rows],
                                             ridge=5.0 / rows.sum())
            off_t = np.array([soff.get(s, 0.0) for s in subj_b[te]])
            p4 = sg(z3t + off_t)
            # v5 + kNN residual on revealed items (same bench, qwen emb cosine)
            p5 = None
            if have_emb[rev].sum() >= KNN_K and have_emb[te].sum() > 0:
                rev_emb_pos = eb_pos[rev]
                te_emb_pos = eb_pos[te]
                vr = np.asarray(iv[rev_emb_pos[have_emb[rev]]], np.float32)
                vr /= np.clip(np.linalg.norm(vr, axis=1, keepdims=True), 1e-9, None)
                resid_r = (yr[have_emb[rev]]
                           - sg(z3r[have_emb[rev]] + np.array(
                               [soff.get(s, 0.0) for s in subj_b[rev][have_emb[rev]]])))
                z5t = z3t + off_t
                p5 = sg(z5t).copy()
                te_ok = np.where(have_emb[te])[0]
                CH = 4096
                for s0 in range(0, len(te_ok), CH):
                    sel = te_ok[s0:s0 + CH]
                    vt = np.asarray(iv[te_emb_pos[sel]], np.float32)
                    vt /= np.clip(np.linalg.norm(vt, axis=1, keepdims=True), 1e-9, None)
                    sims = vt @ vr.T
                    k_eff = min(KNN_K, sims.shape[1])
                    nn = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
                    sw = np.take_along_axis(sims, nn, axis=1)
                    sw = np.clip(sw, 0, None)
                    rs = resid_r[nn]
                    wsum = sw.sum(1) + 1e-9
                    delta = (sw * rs).sum(1) / wsum
                    shrink = wsum / (wsum + 4.0)  # count/sim shrinkage
                    p5[sel] = np.clip(sg(z5t[sel]) + shrink * delta, EPS, 1 - EPS)
            row = {"v2_intercept": {"bce": round(bce(yt, p2), 5)},
                   "v3_slope": {"bce": round(bce(yt, p3), 5)},
                   "v4_subject_offsets": {"bce": round(bce(yt, p4), 5)}}
            a2, a4 = auc(yt, p2), auc(yt, p4)
            row["v2_intercept"]["auc"] = round(a2, 4) if a2 is not None else None
            row["v4_subject_offsets"]["auc"] = round(a4, 4) if a4 is not None else None
            if p5 is not None:
                a5 = auc(yt, p5)
                row["v5_knn_residual"] = {"bce": round(bce(yt, p5), 5),
                                          "auc": (round(a5, 4) if a5 is not None else None)}
            row["v0_on_te"] = {"bce": round(bce(yt, sg(zt / max(tau_med, 1e-9))), 5)}
            row["v1_on_te"] = {"bce": round(bce(yt, sg(zt)), 5)}
            entry["reveal"][str(r)] = row
        per_bench[b] = entry
        step("bench_done", bench=b, i=bi + 1, total=len(benches),
             v0=entry["v0_lobo_stack"]["bce"], v1=entry["v1_tau"]["bce"])

    res["per_bench"] = per_bench

    # ---- summary: mean over benches of deltas ---------------------------------------
    def collect(path):
        vals = []
        for b in benches:
            d = per_bench[b]
            for k in path:
                d = d.get(k) if isinstance(d, dict) else None
                if d is None:
                    break
            if isinstance(d, (int, float)):
                vals.append(d)
        return vals

    summary = {"v0_mean_bce": round(float(np.mean(collect(["v0_lobo_stack", "bce"]))), 5),
               "v1_mean_bce": round(float(np.mean(collect(["v1_tau", "bce"]))), 5)}
    for r in REVEAL_FRACS:
        rr = str(r)
        for v in ["v1_on_te", "v2_intercept", "v3_slope", "v4_subject_offsets",
                  "v5_knn_residual"]:
            vals = collect(["reveal", rr, v, "bce"])
            if vals:
                summary[f"r{rr}_{v}_mean_bce"] = round(float(np.mean(vals)), 5)
    res["summary"] = summary
    res["ok"] = True
    res["t_total_s"] = round(time.time() - _t0, 1)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    step("done", **summary)
    print("LOBO ALPHA PROBE DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        step("ERROR", error=repr(e), tb=traceback.format_exc())
        raise
