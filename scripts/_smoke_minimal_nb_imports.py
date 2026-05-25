"""Light smoke test: import every symbol the minimal notebook references.

We don't actually run the notebook (it needs a GPU + data download), but
we exercise every ``from X import Y`` pattern from the notebook so a
typo in a function name or a missing module gets caught now rather than
two cells deep into a Colab session.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    # Cell 2
    from src.embeddings import login_huggingface, resolve_hf_token  # noqa: F401

    # Cell 3
    from src.data import (  # noqa: F401
        compute_dataset_stats,
        make_item_cold_start_split,
        prepare_dataset,
        print_dataset_stats,
    )

    # Cell 4
    from src import drive_cache as drive_cache_mod  # noqa: F401
    from src.embeddings import (  # noqa: F401
        EncoderConfig,
        TransformerEmbedder,
        assert_deduplicated,
        build_unique_items,
        build_unique_subjects,
        content_hash_for_items,
        encoder_slug,
        verify_flash_attention,
    )

    # Cell 5
    from src.clustering import fit_and_assign  # noqa: F401
    from src.item_features import (  # noqa: F401
        apply_zscore,
        build_centroid_distance_features,
        centroid_distance_feature_names,
        compute_features_for_items,
        fit_zscore_stats,
        load_pool_features,
        merge_pool_and_centroid_features,
        save_pool_features,
        save_zscore_stats,
    )

    # Cell 6
    from src.models import Indexer  # noqa: F401
    from src.nn_features import (  # noqa: F401
        NNFeaturesConfig,
        TrainingNNIndex,
        build_passrate_table,
        compute_nn_features_streaming,
    )

    # Cell 7
    from src.data import prepare_metadata_artifacts  # noqa: F401
    from src.embeddings import stack_lookup  # noqa: F401
    from src.item_features import build_feature_matrix  # noqa: F401
    from src.metadata_features import MetadataSchema  # noqa: F401
    from src.models import LookupDataset  # noqa: F401

    # Cells 8a, 8b: train Model A + Model B with dropout pre-hook.
    import src.train as train_mod  # noqa: F401
    from src.models import ModelConfig  # noqa: F401
    from src.train import TrainConfig, train_one  # noqa: F401
    from src.train_dropout import (  # noqa: F401
        TrainDropoutConfig,
        install_train_dropout,
    )

    # Cells 9, 10: scoring + blend tuning + NN calibrator.
    from src.models import build_model as _build_model_for_inf  # noqa: F401
    from src.nn_calibration import (  # noqa: F401
        NNCalibrator,
        SubjectResidualTable,
    )

    # Cell 11: export coverage-blend ensemble bundle.
    from src.export_submission import (  # noqa: F401
        bundle_training_cache,
        compute_train_counts,
        export_coverage_blend_run,
        make_submission_zip,
    )

    print("OK: all notebook imports resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
