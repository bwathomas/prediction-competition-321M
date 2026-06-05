"""Proxy dependency tree: fields/feature-groups that proxy subject/benchmark identity.

Dropping an identity node must atomically mask ALL of its descendants, else a proxy
(e.g. model family, or an NN-passrate feature aggregated over the subject) silently
re-leaks the identity the dropout was meant to hide.

Convention: a descendant is either an exact column name (e.g. "subject_key",
"condition") or a feature-group PREFIX (e.g. "feat:nn_passrate", "meta:family") that
matches any column starting with it (so "feat:nn_passrate__mean", "...__max" all mask
together).
"""
from __future__ import annotations

PROXY_TREE = {
    "subject": [
        "subject_key",
        "subject_content",
        "meta:family",
        "meta:macro-family",
        "meta:parameters",
        "meta:organization",
        "meta:release_date",
        "feat:nn_passrate",   # passrate aggregates over the subject
        "feat:subject_mean",  # subject-mean encoding
    ],
    "benchmark": [
        "benchmark",
        "condition",          # conditions proxy benchmarks
        "data_category",
        "feat:pool",          # benchmark-derived pool features
    ],
}


def descendants(node: str) -> list:
    return list(PROXY_TREE.get(node, []))


def all_masked_columns(dropped_nodes, feature_columns) -> set:
    """Expand dropped identity nodes to the concrete set of columns to mask.

    Exact descendant names match exactly; feature-group prefixes match any column
    that equals OR starts with the prefix.
    """
    cols = list(feature_columns)
    masked = set()
    for node in dropped_nodes:
        for proxy in descendants(node):
            for c in cols:
                if c == proxy or c.startswith(proxy):
                    masked.add(c)
    return masked
