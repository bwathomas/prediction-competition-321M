"""The catalog is the source of truth for the data suite the agent may use. These tests
pin the scope decisions (included metadata; removed judge/solver/PCA/learned-param
features) and guarantee every real column is classified for dropout/coverage."""
import pytest
from aide import feature_catalog as fc
from aide.hygiene.probes import assert_columns_covered


def test_removed_groups_are_absent():
    names = " ".join(g.name + " " + g.source + " " + " ".join(g.patterns) for g in fc.CATALOG)
    for gone in ["judge", "solver", "pca_tail", "theta_s", "u_s_", "u_bc",
                 "cross_theta", "member5"]:
        assert gone not in names, f"{gone} must be removed from the catalog"


def test_csv_metadata_groups_present_including_qualitative_conditions():
    names = {g.name for g in fc.CATALOG}
    # subject + benchmark CSV metadata
    for n in ["subject_meta_categorical", "subject_meta_numeric",
              "benchmark_meta_categorical", "benchmark_meta_numeric"]:
        assert n in names
    # the previously-dropped qualitative benchmark fields are now included
    assert "benchmark_conditions" in names
    bc = next(g for g in fc.CATALOG if g.name == "benchmark_conditions")
    assert any("has_conditions" in p for p in bc.patterns)


def test_every_catalog_column_is_classified_for_coverage():
    # every pattern p is matched by a concrete "p__0" instance (the "__" boundary rule)
    cols = sorted({p + "__0" for g in fc.CATALOG for p in g.patterns})
    # nothing should be unclassified given the neutral allowlist
    assert_columns_covered(cols, neutral_prefixes=list(fc.NEUTRAL_ITEM))


def test_neutral_and_proxy_sets_are_disjoint_on_prefixes():
    neutral = set(fc.NEUTRAL_ITEM)
    proxies = set(fc.SUBJECT_PROXY) | set(fc.BENCHMARK_PROXY)
    assert neutral.isdisjoint(proxies)


def test_learned_param_axis_theta_is_not_neutral_nor_proxy_it_is_simply_gone():
    # a stray learned-param column must FAIL coverage (unlisted => blocked), proving it is
    # neither silently allowed nor masked — it should never reach the funnel at all.
    with pytest.raises(AssertionError):
        assert_columns_covered(["theta_s", "u_s_0"], neutral_prefixes=list(fc.NEUTRAL_ITEM))


def test_group_names_are_the_agent_menu():
    assert "item_pool" in fc.group_names()
    assert "nn_passrate" in fc.group_names()
    assert len(fc.group_names()) == len(set(fc.group_names()))
