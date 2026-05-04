"""Run one or more official-like rounds against a submission.

Example:
    py scripts/run_validation.py ^
        --submission "example_submissions/random_baseline" ^
        --val-parquet "splits/v1/val.parquet" ^
        --train-parquet "splits/v1/train.parquet" ^
        --N 5000 --K 5 --seeds 0 1 2

Each seed simulates a fresh container (modules are reloaded between rounds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from harness.rounds import run_official_like_round  # noqa: E402
from harness.scoring import score_round  # noqa: E402
from harness.submission import Submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--val-parquet", required=True, type=Path)
    ap.add_argument("--train-parquet", type=Path, default=None,
                    help="Optional; passed through to run_official_like_round (not used by harness, available for participant code).")
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--category-col", default="data_category")
    ap.add_argument("--variant-col", default="item_variant_id")
    args = ap.parse_args()

    val_df = pd.read_parquet(args.val_parquet)
    train_df = pd.read_parquet(args.train_parquet) if args.train_parquet else pd.DataFrame()

    submission = Submission(args.submission)

    rows = []
    for seed in args.seeds:
        submission.reset()  # simulate fresh container
        result = run_official_like_round(
            train_df=train_df,
            val_df=val_df,
            model_module=submission.model,
            labeling_module=submission.labeling,
            N=args.N,
            K=args.K,
            seed=seed,
            category_col=args.category_col,
            variant_col=args.variant_col,
        )
        score = score_round(result)
        rows.append(
            {
                "seed": seed,
                "n_candidates": result.n_candidates,
                "n_categories": result.n_categories,
                "n_labeled": result.n_labeled,
                "used_random_acquisition": result.used_random_acquisition,
                "fallback_reason": result.fallback_reason,
                "log_likelihood_excl_labeled": score.excluding_labeled.log_likelihood,
                "auc_roc_excl_labeled": score.excluding_labeled.auc_roc,
                "log_likelihood_incl_all": score.including_all.log_likelihood,
                "auc_roc_incl_all": score.including_all.auc_roc,
                "frac_labels_clipped": score.excluding_labeled.frac_labels_clipped,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if len(df) > 1:
        means = {
            "log_likelihood_excl_labeled (mean)": df["log_likelihood_excl_labeled"].mean(),
            "log_likelihood_incl_all (mean)": df["log_likelihood_incl_all"].mean(),
        }
        print("\n" + json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
