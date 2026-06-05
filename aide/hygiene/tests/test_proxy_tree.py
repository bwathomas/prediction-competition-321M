import pytest
from aide.hygiene.proxy_tree import PROXY_TREE, descendants, all_masked_columns


def test_subject_node_includes_metadata_and_feature_proxies():
    d = descendants("subject")
    assert "subject_key" in d
    assert "meta:family" in d and "meta:macro-family" in d and "meta:parameters" in d
    assert "feat:nn_passrate" in d  # NN/passrate features proxy the subject
    assert "feat:judge" in d        # judge scores proxy the subject (M3)


def test_benchmark_node_includes_condition_and_data_category():
    d = descendants("benchmark")
    assert "condition" in d          # conditions proxy benchmarks
    assert "data_category" in d
    assert "feat:pool" in d
    assert "feat:benchmark_mean" in d        # mean-encoding proxy (M3)
    assert "feat:benchmark_passrate" in d    # cross-axis aggregate, benchmark side (M4)


def test_all_masked_columns_expands_prefixes_atomically():
    cols = ["subject_key", "meta:family", "meta:parameters",
            "feat:nn_passrate__mean", "feat:nn_passrate__max",
            "feat:pool__toklen", "benchmark", "item_emb__0"]
    masked = all_masked_columns(["subject"], cols)
    assert "subject_key" in masked
    assert "meta:family" in masked and "meta:parameters" in masked
    assert "feat:nn_passrate__mean" in masked and "feat:nn_passrate__max" in masked
    assert "benchmark" not in masked and "item_emb__0" not in masked and "feat:pool__toklen" not in masked


def test_startswith_boundary_does_not_over_mask_near_miss_names():
    # C2 regression: bare prefixes must NOT match unrelated columns without the "__" boundary
    cols = ["benchmark", "benchmark_id", "condition", "condition_entropy",
            "meta:family", "meta:family_size", "subject_key", "subject_keyring"]
    masked_b = all_masked_columns(["benchmark"], cols)
    assert "benchmark" in masked_b and "condition" in masked_b
    assert "benchmark_id" not in masked_b and "condition_entropy" not in masked_b
    masked_s = all_masked_columns(["subject"], cols)
    assert "meta:family" in masked_s and "subject_key" in masked_s
    assert "meta:family_size" not in masked_s and "subject_keyring" not in masked_s


def test_namespaced_aggregate_still_masks_via_double_underscore():
    cols = ["feat:nn_passrate", "feat:nn_passrate__mean", "feat:judge__p_yes"]
    masked = all_masked_columns(["subject"], cols)
    assert set(cols) <= set(masked)  # all three mask together


def test_descendants_raises_on_unknown_node():
    with pytest.raises(ValueError):
        descendants("model")  # m6


def test_all_masked_columns_is_sorted_and_deduped():
    cols = ["condition", "benchmark", "benchmark"]  # dup tolerated here (cols arg)
    masked = all_masked_columns(["benchmark"], cols)
    assert masked == sorted(set(masked))  # deterministic boundary (m3)


def test_proxy_tree_has_subject_and_benchmark_roots():
    assert set(PROXY_TREE) == {"subject", "benchmark"}
