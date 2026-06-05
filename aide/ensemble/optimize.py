"""Overnight ensemble optimization: search architectures × feature-ablations × stacking to
minimize OOF item-cold-start NLL, with Drive checkpointing so a Colab disconnect resumes.

Self-contained: ``run_search`` loops a config space, evaluates each via the harness's
nested-OOF ``recursive_evaluate`` (subsampled for ranking; the caller re-runs the winner on
the full set), records every result + the running best to a JSON checkpoint after each
config, and skips already-done configs on resume. Time-budgeted so it stops cleanly by morning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aide.ensemble.ablations import make_ablated_factory
from aide.ensemble.registry import get
from aide.harness.eval import Dataset, DropoutConfig, recursive_evaluate


def group_columns(feature_columns):
    """Catalog group name → its columns present in ``feature_columns`` (via catalog patterns)."""
    from aide.feature_catalog import CATALOG, matches
    cols = list(feature_columns)
    out = {}
    for g in CATALOG:
        kept = [c for c in cols if any(matches(c, p) for p in g.patterns)]
        if kept:
            out[g.name] = kept
    return out


def default_search_space():
    """A compact but meaningful space: single-arch baselines + GBDT variants + 2-layer
    stacks over feature-diverse ablated members. ``groups=None`` means all features."""
    nn = ["nn_passrate", "nn_label_derivatives", "counts_subject"]
    geo = ["nn_geometry", "cluster_geometry", "centroid_distance", "item_cluster"]
    clu = ["cluster_passrate", "cluster_subject"]
    space = [
        {"id": "logistic_all", "members": [("logistic", {}, None)]},
        {"id": "mlp32_all", "members": [("mlp", {"hidden": 32, "iters": 300}, None)]},
        {"id": "gbdt_all", "members": [("gbdt_lightgbm", {"num_leaves": 31, "n_estimators": 300}, None)]},
        {"id": "gbdt_deep", "members": [("gbdt_lightgbm", {"num_leaves": 63, "n_estimators": 500,
                                                           "learning_rate": 0.03}, None)]},
        {"id": "gbdt_shallow", "members": [("gbdt_lightgbm", {"num_leaves": 15, "n_estimators": 400}, None)]},
        {"id": "stack_gbdt_lr_mlp",
         "members": [("gbdt_lightgbm", {"num_leaves": 31, "n_estimators": 300}, None),
                     ("logistic", {}, nn + clu),
                     ("mlp", {"hidden": 32, "iters": 300}, geo)],
         "stacker": "linear_stacker"},
        {"id": "stack_two_gbdt",
         "members": [("gbdt_lightgbm", {"num_leaves": 31, "n_estimators": 300}, nn + clu),
                     ("gbdt_lightgbm", {"num_leaves": 31, "n_estimators": 300}, geo)],
         "stacker": "linear_stacker"},
        {"id": "gbdt_all_drop10", "members": [("gbdt_lightgbm", {"num_leaves": 31, "n_estimators": 300}, None)],
         "dropout": {"subject_rate": 0.1, "benchmark_rate": 0.1}},
    ]
    return space


def build_members(cfg, feature_columns):
    gmap = group_columns(feature_columns)
    facs = []
    for arch, kw, groups in cfg["members"]:
        base = get(arch, **kw)
        if groups is None:
            facs.append(base)
        else:
            keep = [c for grp in groups for c in gmap.get(grp, [])]
            if not keep:
                raise ValueError(f"config {cfg['id']}: no columns for groups {groups}")
            facs.append(make_ablated_factory(base, feature_columns, keep))
    return facs


def _subsample(ds, n, seed=0):
    if n is None or n >= len(ds.y):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.y), size=n, replace=False)
    return Dataset(X=ds.X[idx], feature_columns=ds.feature_columns, y=ds.y[idx],
                   item_keys=ds.item_keys[idx], subjects=ds.subjects[idx],
                   benchmarks=ds.benchmarks[idx])


def evaluate_config(cfg, ds, manifest, *, subsample=None, seed=0, now_fn=None):
    t0 = now_fn() if now_fn else 0.0
    try:
        sub = _subsample(ds, subsample, seed)
        members = build_members(cfg, ds.feature_columns)
        stacker = get(cfg.get("stacker", "linear_stacker"))
        dcfg = DropoutConfig(**cfg["dropout"]) if cfg.get("dropout") else None
        if len(members) == 1 and "stacker" not in cfg:
            from aide.harness.eval import evaluate
            res = evaluate(members[0], sub, manifest, dropout=dcfg, seed=seed)
        else:
            res = recursive_evaluate(members, stacker, sub, manifest, dropout=dcfg, seed=seed)
        secs = (now_fn() - t0) if now_fn else 0.0
        return {"id": cfg["id"], "ok": True, "nll": res.nll, "auc": float(res.auc),
                "n": int(res.n), "secs": round(secs, 1), "error": None}
    except Exception as exc:  # noqa: BLE001 - a failed config must not kill the search
        secs = (now_fn() - t0) if now_fn else 0.0
        return {"id": cfg["id"], "ok": False, "nll": None, "auc": None,
                "secs": round(secs, 1), "error": repr(exc)}


def _load_ckpt(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"done": {}, "best": None}


def _save_ckpt(path, ckpt):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ckpt, sort_keys=True))
    tmp.replace(p)


def run_search(ds, manifest, *, checkpoint_path, space=None, subsample=200_000,
               time_budget_s=None, now_fn=None, progress=None, seed=0):
    """Evaluate the config space, checkpointing after each. Resumes from ``checkpoint_path``
    (done configs skipped). Stops when the space is exhausted or ``time_budget_s`` elapses."""
    space = space or default_search_space()
    ckpt = _load_ckpt(checkpoint_path)
    start = now_fn() if now_fn else 0.0
    for cfg in space:
        if cfg["id"] in ckpt["done"]:
            continue
        if time_budget_s and now_fn and (now_fn() - start) > time_budget_s:
            if progress:
                progress(f"time budget reached; {len(ckpt['done'])}/{len(space)} done")
            break
        res = evaluate_config(cfg, ds, manifest, subsample=subsample, seed=seed, now_fn=now_fn)
        ckpt["done"][cfg["id"]] = res
        if res["ok"] and (ckpt["best"] is None or res["nll"] < ckpt["best"]["nll"]):
            ckpt["best"] = {"id": res["id"], "nll": res["nll"]}
        _save_ckpt(checkpoint_path, ckpt)
        if progress:
            b = ckpt["best"]
            progress(f"{res['id']}: nll={res['nll']} (best {b['id']}={b['nll'] if b else None})",
                     done=len(ckpt["done"]), total=len(space))
    leaderboard = sorted([r for r in ckpt["done"].values() if r["ok"]], key=lambda r: r["nll"])
    return {"best": ckpt["best"], "leaderboard": leaderboard, "n_done": len(ckpt["done"])}
