from .manifest import SplitManifest, build_manifest, item_fold, assert_identical
from .splits import Fold, outer_folds, inner_folds, row_fold_ids
from .proxy_tree import PROXY_TREE, descendants, all_masked_columns
from .dropout import apply_proxy_dropout
from .probes import assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak

__all__ = [
    "SplitManifest", "build_manifest", "item_fold", "assert_identical",
    "Fold", "outer_folds", "inner_folds", "row_fold_ids",
    "PROXY_TREE", "descendants", "all_masked_columns",
    "apply_proxy_dropout",
    "assert_item_disjoint", "assert_row_uniform_safe", "assert_no_proxy_leak",
]
