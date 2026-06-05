"""Proxy dependency tree — DERIVED from the canonical feature catalog
(``aide.feature_catalog``) so the dropout/coverage classification always uses the REAL
column names of the included data suite (not placeholder prefixes).

Dropping an identity node atomically masks ALL of its descendants. Matching uses the
"__" boundary rule (``matches``): a pattern ``p`` masks column ``c`` iff ``c == p`` or
``c.startswith(p + "__")`` — so "benchmark" does not mask "benchmark_id" and "subj_cat"
masks every "subj_cat__*" aggregate together.

``NEUTRAL_ITEM`` (item-content prefixes) is the allowlist for ``assert_columns_covered``:
item content is usable signal, never masked. Anything that is neither a known proxy nor
neutral fails the coverage probe ("unlisted => blocked").
"""
from __future__ import annotations

from ..feature_catalog import (
    matches as _matches, SUBJECT_PROXY, BENCHMARK_PROXY, NEUTRAL_ITEM)

# Built from the catalog; keys are exactly the two identity roots.
PROXY_TREE = {
    "subject": list(SUBJECT_PROXY),
    "benchmark": list(BENCHMARK_PROXY),
}

# Re-exported for the coverage probe.
NEUTRAL_ITEM = list(NEUTRAL_ITEM)


def descendants(node: str) -> list:
    if node not in PROXY_TREE:
        raise ValueError(f"unknown proxy node {node!r}; known roots: {sorted(PROXY_TREE)}")
    return list(PROXY_TREE[node])


def all_masked_columns(dropped_nodes, feature_columns) -> list:
    """Concrete, sorted, de-duplicated columns to mask for the dropped identity nodes."""
    cols = list(feature_columns)
    masked = set()
    for node in dropped_nodes:
        for proxy in descendants(node):
            for c in cols:
                if _matches(c, proxy):
                    masked.add(c)
    return sorted(masked)
