"""Static validation of notebooks/qwen8b_minimalist.ipynb."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    nb_path = ROOT / "notebooks" / "qwen8b_minimalist.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"total cells: {len(cells)}  code: {n_code}  markdown: {n_md}")

    errors: list[str] = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            preview = src.splitlines()[0][:120] if src.strip() else "(empty)"
            errors.append(f"cell {i}: {e} -- starts: {preview}")

    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1

    # Look for the must-have imports + flag setters as a smoke check.
    full_src = "\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] == "code"
    )
    expected_strings = [
        "Qwen/Qwen3-Embedding-8B",
        "meta_hybrid_irt_kfactor_gated_mlp",
        "build_centroid_distance_features",
        "TrainingNNIndex",
        "NNCalibrator.fit_alpha_on_val",
        "SubjectResidualTable.from_rows",
        "nn_calibrator_state=calibrator.to_dict()",
        "ship_training_cache=True",
        "use_metadata_features=True",
        "TrainDropoutConfig",
        "install_train_dropout",
        "export_coverage_blend_run",
        "member_b_force_bench_missing",
        "bench_missing_real_subject",
        "epochs_per_member",
        "BLEND_PRESENT",
        "BLEND_MISSING",
        "synthetic_cold_start_frac",
        "MODEL_A_SEED",
        "MODEL_B_SEED",
        "files.download",  # Colab download path
    ]
    missing = [s for s in expected_strings if s not in full_src]
    if missing:
        print("MISSING expected anchors:", missing)
        return 2

    # Forbidden anchors (judge / LoRA / K-fold should not appear).
    forbidden = [
        "JudgeFeaturizer",
        "build_judge_matrix",
        "PeftMixedModel",
        "kfold_train_one",
        "fit_optimal_weights",
    ]
    present = [s for s in forbidden if s in full_src]
    if present:
        print("FORBIDDEN anchors present:", present)
        return 3

    print("OK: notebook code cells parse, anchors present, judge/LoRA/Kfold absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
