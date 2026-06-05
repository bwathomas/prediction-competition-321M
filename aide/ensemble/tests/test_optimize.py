"""Tests for the overnight ensemble search loop: config→members mapping, checkpoint/resume,
failure isolation (a config needing an absent heavy lib is recorded, not fatal), and the
time budget. Uses the toy dataset + numpy archs + a fake clock so it runs locally."""
import json

from aide.ensemble.optimize import build_members, group_columns, run_search
from aide.harness.tests._toy import make_dataset
from aide.hygiene.manifest import build_manifest


def test_group_columns_maps_catalog_patterns():
    g = group_columns(["nn__passrate_mean_K4", "geo__local_density", "cluster__001", "sig0"])
    flat = sum(g.values(), [])
    assert "nn__passrate_mean_K4" in flat and "geo__local_density" in flat
    assert "sig0" not in flat                      # not a catalog feature → unmapped


def test_build_members_ablation_keeps_group_columns():
    cols = ["nn__a", "geo__b", "cluster__001", "cnt__c"]
    facs = build_members({"id": "x", "members": [("logistic", {}, ["nn_label_derivatives"])]}, cols)
    assert len(facs) == 1                          # one ablated member built without error


def test_run_search_checkpoint_resume_and_failure_isolation(tmp_path):
    ds = make_dataset(seed=1)
    m = build_manifest(ds.item_keys, n_folds=3, seed=0)
    space = [{"id": "logistic_all", "members": [("logistic", {}, None)]},
             {"id": "mlp_all", "members": [("mlp", {"hidden": 8, "iters": 80}, None)]},
             {"id": "gbdt_all", "members": [("gbdt_lightgbm", {}, None)]}]  # no lightgbm → fails
    ck = tmp_path / "ck.json"
    res = run_search(ds, m, checkpoint_path=ck, space=space, subsample=None, now_fn=lambda: 0.0)
    assert res["n_done"] == 3
    assert res["best"]["id"] in ("logistic_all", "mlp_all")
    saved = json.loads(ck.read_text())
    assert saved["done"]["gbdt_all"]["ok"] is False        # failure captured, not fatal
    assert saved["done"]["logistic_all"]["ok"] is True
    # resume: every config already done → no recompute, same tally
    res2 = run_search(ds, m, checkpoint_path=ck, space=space, subsample=None, now_fn=lambda: 0.0)
    assert res2["n_done"] == 3
    assert [r["id"] for r in res2["leaderboard"]]          # leaderboard non-empty, nll-sorted
    nlls = [r["nll"] for r in res2["leaderboard"]]
    assert nlls == sorted(nlls)


def test_time_budget_stops_early(tmp_path):
    ds = make_dataset(seed=2)
    m = build_manifest(ds.item_keys, n_folds=3, seed=0)
    space = [{"id": f"c{i}", "members": [("logistic", {}, None)]} for i in range(6)]
    clock = [0.0]

    def now():
        clock[0] += 100.0
        return clock[0]

    res = run_search(ds, m, checkpoint_path=tmp_path / "c.json", space=space,
                     subsample=None, time_budget_s=150, now_fn=now)
    assert 0 < res["n_done"] < 6                    # stopped before exhausting the space
