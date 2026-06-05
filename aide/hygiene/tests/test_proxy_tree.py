import pytest
from aide.hygiene.proxy_tree import PROXY_TREE, descendants, all_masked_columns, NEUTRAL_ITEM


def test_proxy_tree_has_subject_and_benchmark_roots():
    assert set(PROXY_TREE) == {"subject", "benchmark"}


def test_subject_proxies_use_real_catalog_names():
    d = descendants("subject")
    for p in ["subject_key", "subject_content", "subj_emb", "subj_cat", "subj_num",
              "subject_mean", "subject_obs_count", "nn", "m2_subj"]:
        assert p in d, p
    # removed groups (judge / solver / learned IRT params) must NOT appear
    for gone in ["feat:judge", "feat:nn_passrate", "theta_s", "u_s", "member5", "pca_tail"]:
        assert gone not in d


def test_benchmark_proxies_use_real_catalog_names():
    d = descendants("benchmark")
    for p in ["benchmark", "condition", "cond", "data_category", "bench_cat", "bench_num",
              "bench_has_conditions", "m2_bc"]:
        assert p in d, p


def test_subj_cat_and_num_metadata_columns_masked_under_subject():
    cols = ["subj_cat__family__001", "subj_cat__organization__003",
            "subj_num__parameters", "subj_num__release_date__missing", "item_emb__0"]
    masked = all_masked_columns(["subject"], cols)
    assert "subj_cat__family__001" in masked and "subj_num__parameters" in masked
    assert "subj_num__release_date__missing" in masked
    assert "item_emb__0" not in masked  # item content is neutral, never masked


def test_boundary_rule_does_not_over_mask_near_miss_names():
    cols = ["benchmark", "benchmark_id", "condition", "conditionX", "subj_cat__family"]
    mb = all_masked_columns(["benchmark"], cols)
    assert "benchmark" in mb and "condition" in mb
    assert "benchmark_id" not in mb and "conditionX" not in mb


def test_nn_is_subject_proxy_and_only_benchmark_conditional_cell_is_benchmark():
    cols = ["nn__passrate_mean", "nn__passrate_benchmark_conditional"]
    sub = all_masked_columns(["subject"], cols)
    assert "nn__passrate_mean" in sub and "nn__passrate_benchmark_conditional" in sub
    ben = all_masked_columns(["benchmark"], cols)
    assert "nn__passrate_benchmark_conditional" in ben   # leaks benchmark
    assert "nn__passrate_mean" not in ben                # subject-only, not over-masked


def test_neutral_item_allowlist_holds_content_prefixes():
    for p in ["item_emb", "pool", "cd", "cluster", "semcat", "m2_cluster"]:
        assert p in NEUTRAL_ITEM


def test_descendants_raises_on_unknown_node():
    with pytest.raises(ValueError):
        descendants("model")
