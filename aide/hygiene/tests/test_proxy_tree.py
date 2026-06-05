from aide.hygiene.proxy_tree import PROXY_TREE, descendants, all_masked_columns


def test_subject_node_includes_metadata_and_feature_proxies():
    d = descendants("subject")
    assert "subject_key" in d
    assert "meta:family" in d and "meta:macro-family" in d and "meta:parameters" in d
    assert "feat:nn_passrate" in d  # NN/passrate features proxy the subject


def test_benchmark_node_includes_condition_and_data_category():
    d = descendants("benchmark")
    assert "condition" in d          # conditions proxy benchmarks
    assert "data_category" in d
    assert "feat:pool" in d


def test_all_masked_columns_expands_prefixes_atomically():
    cols = ["subject_key", "meta:family", "meta:parameters",
            "feat:nn_passrate__mean", "feat:nn_passrate__max",
            "feat:pool__toklen", "benchmark", "item_emb__0"]
    masked = all_masked_columns(["subject"], cols)
    assert "subject_key" in masked
    assert "meta:family" in masked and "meta:parameters" in masked
    assert "feat:nn_passrate__mean" in masked and "feat:nn_passrate__max" in masked
    assert "benchmark" not in masked and "item_emb__0" not in masked and "feat:pool__toklen" not in masked


def test_proxy_tree_has_subject_and_benchmark_roots():
    assert set(PROXY_TREE) == {"subject", "benchmark"}
