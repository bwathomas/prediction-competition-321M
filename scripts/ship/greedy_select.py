"""Greedy Ensemble Selection (Caruana et al. 2004/2006) over cached OOF member predictions.

Replaces the brute-force (rho x M x portfolio) grid: GENERATE a rich library of members
(we already do — per-family/per-fold MLP and XGBoost leave-one-kind-out members), then SELECT
cheaply on the CACHED out-of-fold prediction vectors. Greedy selection jointly determines the
number of members, the mix, and per-member weights (multiplicity = weight) with no retraining.

Overfitting safeguards (all cheap, all on cached vectors):
  * selection WITH REPLACEMENT  -> flattens the curve past the peak; stopping point non-critical.
  * SORTED INITIALIZATION       -> seed the ensemble with the best-N members (N picked on hillclimb).
  * BAGGED ES                   -> run selection on n_bags random fraction-p subsets of the LIBRARY
                                   and average the weight vectors (bounds an overfit M-combo by (1-p)^M).
  * NESTED item-grouped CV       -> honest generalization estimate: select on K-1 folds' rows,
                                   score the selected weights on the held-out fold (rotate). Groups =
                                   item_key (cold-item), matching how the base OOF was produced.
Compute-aware: selection is pure-numpy on cached vectors. Optional --score-subsample subsamples the
rows used for the greedy LOSS evaluation (selection is robust to it); the held-out TEST scoring always
uses full rows. The expensive part (training members) is already done.

Members are combined by AVERAGING PROBABILITIES (Caruana). Objective = soft cross-entropy.

Usage (on Colab, after members are cached to Drive):
    python scripts/ship/greedy_select.py --families qwen nemotron lgai --models mlp xgb \
        --level member --bags 20 --steps 60 --score-subsample 500000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_EPS = 1.0e-6
DRIVE_DEFAULT = "/content/drive/MyDrive/prediction-competition-321M"
# model tag -> SAVE_ROOT fold-dir prefix
MODEL_DIR = {"mlp": "full_fold", "xgb": "xgb_full_fold"}


def soft_logloss(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _load_group(drive_root, family, model, folds):
    """Load one (family, model) group's member OOF preds, concatenated over folds.

    Returns (keys[list], y[np], members{name->np}) where members are p_full + each p_loo__<cat>.
    All members of a group share the same rows (the group's OOF rows)."""
    base = f"{drive_root}/ship/exp_loo/{family}/{MODEL_DIR[model]}"
    keys, yv = [], []
    member_cols = {}   # name -> list of per-fold arrays
    for f in folds:
        npz = f"{base}{f}/preds/oof_preds.npz"
        if not os.path.exists(npz):
            return None  # group incomplete
        # allow_pickle: these .npz are OUR OWN artifacts (written by exp_loo_category_mlp.py to
        # our Drive); the object arrays are string item/subject keys + cat_list. Trusted source.
        d = np.load(npz, allow_pickle=True)
        it = d["oof_items"].astype(str); sj = d["oof_subj"].astype(str)
        keys += [f"{s}|{i}" for s, i in zip(sj, it)]
        yv.append(d["oof_y"].astype(np.float64))
        cat_list = [str(c) for c in d["cat_list"].tolist()]
        names = ["p_full"] + [f"p_loo__{c}" for c in cat_list]
        for nm in names:
            arr = d["p_full"] if nm == "p_full" else d[nm]
            member_cols.setdefault(nm, []).append(np.asarray(arr, dtype=np.float64))
    members = {f"{family}.{model}.{nm}": np.concatenate(v) for nm, v in member_cols.items()}
    return keys, np.concatenate(yv), members


def load_library(drive_root, families, models, folds):
    """Align all (family, model) member groups on common (subject|item) rows.
    Returns member_names[list], M[np N x K], y[np N], items[np N] (groups)."""
    groups = []
    for fam in families:
        for mdl in models:
            g = _load_group(drive_root, fam, mdl, folds)
            if g is None:
                print(f"[greedy] skip {fam}.{mdl} (incomplete)")
                continue
            keys, y, members = g
            kidx = {k: i for i, k in enumerate(keys)}
            groups.append((kidx, y, members))
            print(f"[greedy] loaded {fam}.{mdl}: {len(members)} members, {len(keys)} rows")
    if not groups:
        raise SystemExit("no complete (family,model) groups found")
    common = set(groups[0][0])
    for kidx, _, _ in groups[1:]:
        common &= set(kidx)
    common = sorted(common)
    print(f"[greedy] common rows across groups: {len(common)}")
    items = np.array([k.split("|", 1)[1] for k in common])
    cols, names = [], []
    y_ref = None
    for kidx, y, members in groups:
        gi = np.fromiter((kidx[k] for k in common), dtype=np.int64, count=len(common))
        if y_ref is None:
            y_ref = y[gi]
        for nm, arr in members.items():
            cols.append(arr[gi]); names.append(nm)
    M = np.column_stack(cols).astype(np.float64)
    return names, M, y_ref.astype(np.float64), items


def greedy_es(P, y, max_steps=60, sorted_init_max=25):
    """Greedy selection WITH REPLACEMENT + SORTED INIT on rows already subset to the hillclimb set.
    P:[n,K] member probs, y:[n]. Returns integer counts[K] of the best ensemble snapshot."""
    n, K = P.shape
    losses = np.array([soft_logloss(y, P[:, k]) for k in range(K)])
    order = np.argsort(losses)
    # sorted init: best-N by hillclimb ensemble loss
    csum = np.zeros(n); best = (np.inf, 1, None)
    for ninit in range(1, min(sorted_init_max, K) + 1):
        csum = csum + P[:, order[ninit - 1]]
        l = soft_logloss(y, csum / ninit)
        if l < best[0]:
            best = (l, ninit, csum.copy())
    best_loss, ninit, S = best
    counts = np.zeros(K, dtype=np.int64)
    for j in range(ninit):
        counts[order[j]] += 1
    k = int(ninit); best_counts = counts.copy()
    for _ in range(max_steps):
        cand = np.clip((S[:, None] + P) / (k + 1), _EPS, 1.0 - _EPS)      # [n, K]
        ll = -(y[:, None] * np.log(cand) + (1.0 - y[:, None]) * np.log(1.0 - cand)).mean(0)
        j = int(np.argmin(ll))
        S = S + P[:, j]; k += 1; counts[j] += 1
        if ll[j] < best_loss:
            best_loss = float(ll[j]); best_counts = counts.copy()
    return best_counts


def bagged_greedy_es(P, y, hill, rng, n_bags=20, bag_frac=0.5, score_subsample=None, **kw):
    """Average weight vectors over n_bags random fraction-p subsets of the member LIBRARY."""
    K = P.shape[1]
    rows = hill
    if score_subsample and len(hill) > score_subsample:
        rows = rng.choice(hill, size=score_subsample, replace=False)
    Ph, yh = P[rows], y[rows]
    W = np.zeros(K)
    for _ in range(n_bags):
        cols = rng.choice(K, size=max(1, int(round(bag_frac * K))), replace=False)
        counts = greedy_es(Ph[:, cols], yh, **kw)
        w = np.zeros(K); w[cols] = counts
        if w.sum() > 0:
            W += w / w.sum()
    return W / W.sum() if W.sum() > 0 else np.full(K, 1.0 / K)


def nested_cv(P, y, items, rng, n_outer=3, **kw):
    """Honest estimate: GroupKFold(item) — select on the other folds, score on the held-out fold."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan)
    fold_W = []
    for tr, te in gkf.split(P, y, groups=items):
        W = bagged_greedy_es(P, y, tr, rng, **kw)
        oof[te] = P[te] @ W
        fold_W.append(W)
    return soft_logloss(y, np.clip(oof, _EPS, 1 - _EPS)), oof, fold_W


def lgbm_stacker_nested(P, y, items, n_outer=3):
    """Head-to-head baseline: LightGBM meta-stacker, same GroupKFold."""
    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
    except Exception as e:
        return None
    gkf = GroupKFold(n_splits=n_outer); oof = np.full(len(y), np.nan)
    params = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31,
                  min_child_samples=200, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, verbosity=-1, seed=0)
    for tr, te in gkf.split(P, y, groups=items):
        b = lgb.train(params, lgb.Dataset(P[tr], label=y[tr]), num_boost_round=300)
        oof[te] = b.predict(P[te])
    return soft_logloss(y, np.clip(oof, _EPS, 1 - _EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("SHIP_DRIVE_ROOT", DRIVE_DEFAULT))
    ap.add_argument("--families", nargs="+", default=["qwen", "nemotron", "lgai"])
    ap.add_argument("--models", nargs="+", default=["mlp", "xgb"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bags", type=int, default=20)
    ap.add_argument("--bag-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--sorted-init-max", type=int, default=25)
    ap.add_argument("--score-subsample", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    names, P, y, items = load_library(a.drive_root, a.families, a.models, a.folds)
    print(f"[greedy] library: {P.shape[1]} members x {P.shape[0]} rows")

    # baselines
    singles = np.array([soft_logloss(y, P[:, k]) for k in range(P.shape[1])])
    best_single = float(singles.min()); best_single_name = names[int(singles.argmin())]
    mean_ll = soft_logloss(y, P.mean(1))
    L = np.log(np.clip(P, _EPS, 1 - _EPS)) - np.log(1 - np.clip(P, _EPS, 1 - _EPS))
    logitmean_ll = soft_logloss(y, 1 / (1 + np.exp(-L.mean(1))))

    kw = dict(n_bags=a.bags, bag_frac=a.bag_frac, score_subsample=a.score_subsample,
              max_steps=a.steps, sorted_init_max=a.sorted_init_max)
    greedy_ll, _, fold_W = nested_cv(P, y, items, rng, **kw)
    lgbm_ll = lgbm_stacker_nested(P, y, items)

    # final deployed weights on ALL rows + which members survive
    W_final = bagged_greedy_es(P, y, np.arange(len(y)), rng, **kw)
    sel = sorted([(names[i], round(float(W_final[i]), 4)) for i in range(len(names)) if W_final[i] > 1e-3],
                 key=lambda t: -t[1])

    print("\n" + "=" * 76)
    print("GREEDY ENSEMBLE SELECTION  (nested item-grouped, bagged, with replacement)")
    print("=" * 76)
    print(f"  members={P.shape[1]}  rows={P.shape[0]}  bags={a.bags}  steps={a.steps}")
    print(f"  best single member     : {best_single:.6f}   ({best_single_name})")
    print(f"  mean blend             : {mean_ll:.6f}")
    print(f"  logit-mean blend       : {logitmean_ll:.6f}")
    print(f"  LightGBM stacker (nested): {lgbm_ll:.6f}" if lgbm_ll else "  LightGBM stacker: n/a")
    print(f"  GREEDY-ES (nested)     : {greedy_ll:.6f}   <-- honest held-out")
    print("  selected members (final, weight>0.001):")
    for nm, w in sel:
        print(f"      {w:6.3f}  {nm}")
    print("=" * 76)

    out = a.out or f"{a.drive_root}/ship/exp_loo/greedy_select_report.json"
    rep = {"families": a.families, "models": a.models, "n_members": P.shape[1], "n_rows": P.shape[0],
           "best_single": best_single, "best_single_name": best_single_name,
           "mean_blend": mean_ll, "logit_mean_blend": logitmean_ll,
           "lgbm_stacker_nested": lgbm_ll, "greedy_es_nested": greedy_ll,
           "selected": sel, "bags": a.bags, "steps": a.steps}
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print("[greedy] wrote", out)
    except Exception as e:
        print("[greedy] save failed:", e)
    return rep


if __name__ == "__main__":
    main()
