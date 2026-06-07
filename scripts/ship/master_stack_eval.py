"""Master-stack eval over the per-family member pool + the new IRT variants.

Recovered/extended from the overnight `master_stack.py` (which lived only on the
recycled qwen /content and was lost). Committed to the repo so it survives tab
recycles. Runs on a Colab tab (reads the shared Drive). Reports, for each candidate
IRT variant, the stacked OOF soft-logloss (linear + non-neg) vs the base pool and vs
the original `irt`; then for the augmented pool it fits the full-data non-neg logistic
weights (-> the CANON = members with positive weight) and per-member drop-importance.

Base pool (21 cols) per family: dae/mlp/cnn1d LOO `stacked_oof` + xgb/etbig/logreg/fm
full `p_full`. Candidate extras (one `p_full` col/family): irt, irt2, irt_lib, irt_bag, knn.
Metric = mean soft BCE; honest GroupKFold(item) OOF; stack-CV = GroupKFold(5) on item.
"""
import os, json, numpy as np
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

DR = "/content/drive/MyDrive/prediction-competition-321M"
FAMS = ["qwen", "nemotron", "lgai"]
EPS = 1e-7


def sll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _path(fam, model, sub, f):
    return f"{DR}/ship/exp_loo/{fam}/{model}_{sub}_fold{f}/preds/oof_preds.npz"


def ck(fam, model, sub, key):
    return np.concatenate([
        np.load(_path(fam, model, sub, f), allow_pickle=True)[key].astype(np.float64)
        for f in [0, 1, 2]])


def have(model, sub="full"):
    return all(os.path.exists(_path(fam, model, sub, f)) for fam in FAMS for f in [0, 1, 2])


# ---- labels + item groups (position-aligned across all members) ----
yp, itp = [], []
for f in [0, 1, 2]:
    z = np.load(_path("qwen", "xgb", "full", f), allow_pickle=True)
    yp.append(z["oof_y"].astype(np.float64)); itp.append(z["oof_items"])
y = np.concatenate(yp)
_, groups = np.unique(np.concatenate(itp), return_inverse=True)
N = len(y)

# ---- base pool (21 cols) ----
base = {}
for fam in FAMS:
    for m in ["dae", "mlp", "cnn1d"]:
        base[f"{fam}.{m}.L1"] = ck(fam, m, "loo", "stacked_oof")
    for m in ["xgb", "etbig", "logreg", "fm"]:
        base[f"{fam}.{m}"] = ck(fam, m, "full", "p_full")

# ---- candidate extra members (one p_full column per family) ----
CAND = ["irt", "irt2", "irt_lib", "irt_bag", "knn"]
extra = {m: {f"{fam}.{m}": ck(fam, m, "full", "p_full") for fam in FAMS}
         for m in CAND if have(m)}


def _fit(Xb, yt, bnds):
    def fg(w):
        z = Xb @ w; p = 1 / (1 + np.exp(-z)); p = np.clip(p, EPS, 1 - EPS)
        return (-np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)), Xb.T @ (p - yt) / len(yt))
    return minimize(fg, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B",
                    bounds=bnds, options={"maxiter": 300}).x


def oof_lin(colmap, nonneg):
    names = list(colmap.keys())
    X = lg(np.column_stack([colmap[k] for k in names]))
    Xb = np.hstack([X, np.ones((N, 1))]); oof = np.zeros(N)
    bnds = ([(0, None)] * (Xb.shape[1] - 1) + [(None, None)]) if nonneg else None
    for tr, te in GroupKFold(5).split(X, y, groups):
        w = _fit(Xb[tr], y[tr], bnds)
        oof[te] = 1 / (1 + np.exp(-(Xb[te] @ w)))
    return sll(y, np.clip(oof, EPS, 1 - EPS))


def full_nonneg_weights(colmap):
    names = list(colmap.keys())
    X = lg(np.column_stack([colmap[k] for k in names]))
    Xb = np.hstack([X, np.ones((N, 1))])
    w = _fit(Xb, y, [(0, None)] * (Xb.shape[1] - 1) + [(None, None)])
    return {names[i]: round(float(w[i]), 4) for i in range(len(names))} | {"_bias": round(float(w[-1]), 4)}


def pool(*models):
    c = dict(base)
    for m in models:
        c.update(extra[m])
    return c


# ---- variant comparison (stack-lift vs base) ----
out = {"n": N, "n_base_cols": len(base), "present_extras": list(extra.keys()),
       "linear": {}, "nonneg": {}}
variants = [("base", [])]
for m in ["irt", "irt2", "irt_lib", "irt_bag", "knn"]:
    if m in extra:
        variants.append((f"base+{m}", [m]))
for name, ms in variants:
    cm = pool(*ms)
    out["linear"][name] = round(oof_lin(cm, False), 5)
    out["nonneg"][name] = round(oof_lin(cm, True), 5)
b_lin, b_nn = out["linear"]["base"], out["nonneg"]["base"]
out["delta_vs_base_linear"] = {k: round(b_lin - v, 5) for k, v in out["linear"].items()}
out["delta_vs_base_nonneg"] = {k: round(b_nn - v, 5) for k, v in out["nonneg"].items()}

# ---- pick best IRT variant by non-neg OOF, then weights + drop-importance ----
irt_variants = [m for m in ["irt", "irt2", "irt_lib", "irt_bag"] if m in extra]
if irt_variants:
    best = min(irt_variants, key=lambda m: out["nonneg"][f"base+{m}"])
    out["best_irt_variant"] = best
    cm = pool(best)
    out["nonneg_weights_base+best"] = full_nonneg_weights(cm)
    # leave-one-member-out drop-Δ (non-neg OOF) over base+best
    full_nn = out["nonneg"][f"base+{best}"]
    drop = {}
    names = list(cm.keys())
    for k in names:
        sub = {kk: vv for kk, vv in cm.items() if kk != k}
        drop[k] = round(oof_lin(sub, True) - full_nn, 5)  # +ve => removing it HURTS (valuable)
    out["drop_importance_base+best"] = dict(sorted(drop.items(), key=lambda kv: -kv[1]))

json.dump(out, open("/content/master_stack_eval_result.json", "w"), indent=1)
# also mirror to Drive (recycle-proof)
try:
    os.makedirs(f"{DR}/ship/stack", exist_ok=True)
    json.dump(out, open(f"{DR}/ship/stack/master_stack_eval_result.json", "w"), indent=1)
except Exception:
    pass
print(json.dumps(out, indent=1))
