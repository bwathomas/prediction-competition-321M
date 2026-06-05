import math
from aide.ensemble.registry import get
from aide.ensemble.ablations import make_ablated_factory
from aide.hygiene.manifest import build_manifest
from aide.harness.eval import recursive_evaluate
from aide.harness.tests._toy import make_dataset


def test_two_layer_ensemble_composes_and_beats_chance():
    # layer-2: a LinearStacker over feature-ablated variants of two architectures,
    # evaluated under nested OOF — proving the registry pieces compose via the harness.
    ds = make_dataset(seed=3)
    m = build_manifest(ds.item_keys, n_folds=3, seed=0)
    cols = ds.feature_columns
    members = [
        make_ablated_factory(get("mlp", hidden=16, iters=400), cols, ["sig0", "sig1"]),
        make_ablated_factory(get("logistic"), cols, ["item_id", "sig0", "sig1"]),
    ]
    res = recursive_evaluate(members, get("linear_stacker"), ds, m)
    assert res.oof.shape == (len(ds.y),)
    assert res.nll < math.log(2)
