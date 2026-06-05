"""Proxy dependency tree: fields/feature-groups that proxy subject/benchmark identity.

Dropping an identity node must atomically mask ALL of its descendants, else a proxy
(e.g. model family, a judge score, or an NN-passrate feature aggregated over the
subject) silently re-leaks the identity the dropout was meant to hide.

Matching rule (C2 fix): a proxy entry `p` masks a column `c` iff `c == p` (exact) OR
`c.startswith(p + "__")` (a namespaced aggregate group, e.g. "feat:nn_passrate__mean").
A bare prefix WITHOUT the "__" boundary never matches — so "benchmark" does NOT mask
"benchmark_id" and "meta:family" does NOT mask "meta:family_size". The "__" separator is
the project convention for aggregate columns, so the mask boundary is well-defined and
reproducible across agents regardless of incidental column naming.

Default posture (M3): "unlisted ⇒ exposed" is dangerous for a leakage core. Use
``assert_columns_covered`` (in probes.py) to invert it to "unlisted ⇒ blocked": any
feature column that is neither identity-neutral nor a known proxy fails loudly.
"""
from __future__ import annotations

PROXY_TREE = {
    "subject": [
        "subject_key",
        "subject_content",
        # static metadata (each an exact column)
        "meta:family",
        "meta:macro-family",
        "meta:parameters",
        "meta:organization",
        "meta:release_date",
        # derived feature groups that proxy the subject
        "feat:nn_passrate",        # passrate aggregates over the subject
        "feat:subject_mean",       # subject-mean encoding
        "feat:judge",              # judge scores proxy subject (and benchmark)
        "feat:subject_cluster",    # subject k-means cluster id/embedding
        "feat:subj_emb",           # subject-level text embedding
        "feat:subject_toklen",     # subject_content length stats
        "feat:subject_lang",       # subject_content language id
    ],
    "benchmark": [
        "benchmark",
        "condition",               # conditions proxy benchmarks
        "data_category",
        # derived feature groups that proxy the benchmark
        "feat:pool",               # benchmark-derived pool features
        "feat:benchmark_mean",     # benchmark mean-encoding
        "feat:bench_cond_mean",    # benchmark x condition mean-encoding
        "feat:benchmark_passrate", # cross-axis passrate, benchmark side
        "feat:judge",              # judge scores also proxy the benchmark
        "feat:benchmark_toklen",   # benchmark/item text length stats keyed to benchmark
        "feat:benchmark_lang",     # benchmark content language id
    ],
}


def descendants(node: str) -> list:
    if node not in PROXY_TREE:
        raise ValueError(f"unknown proxy node {node!r}; known roots: {sorted(PROXY_TREE)}")
    return list(PROXY_TREE[node])


def _matches(column: str, proxy: str) -> bool:
    return column == proxy or column.startswith(proxy + "__")


def all_masked_columns(dropped_nodes, feature_columns) -> list:
    """Concrete, sorted, de-duplicated list of columns to mask for the dropped nodes.

    Sorted for deterministic iteration on the boundary (m3). Membership (`in`) works
    on the returned list as before.
    """
    cols = list(feature_columns)
    masked = set()
    for node in dropped_nodes:
        for proxy in descendants(node):
            for c in cols:
                if _matches(c, proxy):
                    masked.add(c)
    return sorted(masked)
