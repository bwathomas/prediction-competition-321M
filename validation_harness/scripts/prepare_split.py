r"""Build train/val parquets with item-cold-start.

Example:
    py scripts/prepare_split.py ^
        --data-dir "..\starting_kit\Data" ^
        --out-dir   "splits/v1" ^
        --val-fraction 0.10 ^
        --seed 0

Outputs (under --out-dir):
    train.parquet                  -- use this to train your model
    val.parquet                    -- official-like validation set
    val_unseen_subjects.parquet    -- stress-test bucket; do NOT use for main score
    split_report.json              -- bookkeeping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from harness.data_loader import add_data_category, load_responses  # noqa: E402
from harness.splits import add_item_variant_id, make_item_cold_start_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--holdout-benchmark",
        action="append",
        default=[],
        help="Benchmark name to hold out entirely (stricter mode). Repeatable.",
    )
    ap.add_argument(
        "--max-rows-per-benchmark",
        type=int,
        default=None,
        help="Optional subsample for fast iteration.",
    )
    ap.add_argument(
        "--data-category-mode",
        default="random",
        choices=["random", "benchmark"],
        help="How to assign data_category. 'random' (default) hashes "
             "item_variant_id into --n-categories buckets; 'benchmark' uses "
             "benchmark name (legacy / debugging).",
    )
    ap.add_argument(
        "--n-categories",
        type=int,
        default=15,
        help="Number of data categories when --data-category-mode=random "
             "(default 15).",
    )
    ap.add_argument(
        "--category-seed",
        type=int,
        default=0,
        help="Seed for the random data_category hash (independent of --seed).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading responses from {args.data_dir} ...", flush=True)
    df = load_responses(
        args.data_dir,
        max_rows_per_benchmark=args.max_rows_per_benchmark,
    )
    print(f"  loaded {len(df):,} rows across {df['benchmark'].nunique()} benchmarks")

    print("Computing item_variant_id ...", flush=True)
    df = add_item_variant_id(df)
    print(f"  {df['item_variant_id'].nunique():,} unique item variants")

    print(f"Assigning data_category (mode={args.data_category_mode!r}) ...", flush=True)
    df = add_data_category(
        df,
        mode=args.data_category_mode,
        n_categories=args.n_categories,
        seed=args.category_seed,
    )
    cat_counts = df.groupby("data_category")["item_variant_id"].nunique().sort_index()
    print(f"  {len(cat_counts)} categories; "
          f"variants per category: min={cat_counts.min():,} max={cat_counts.max():,} "
          f"mean={cat_counts.mean():,.0f}")

    print("Splitting (item cold-start) ...", flush=True)
    train_df, val_df, val_unseen_df, report = make_item_cold_start_split(
        df,
        val_fraction=args.val_fraction,
        seed=args.seed,
        holdout_benchmarks=args.holdout_benchmark or None,
    )
    assert report.n_overlap_variants == 0, "BUG: train/val variant overlap"

    print(f"  train rows: {report.n_train_rows:,} (variants: {report.n_train_variants:,})")
    print(f"  val   rows: {report.n_val_rows:,} (variants: {report.n_val_variants:,})")
    print(f"  val_unseen_subjects rows: {report.n_val_unseen_subject_rows:,} "
          f"(non-official stress test)")
    print(f"  train subjects: {report.n_train_subjects:,}, "
          f"val subjects: {report.n_val_subjects:,}")

    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    val_unseen_path = args.out_dir / "val_unseen_subjects.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    val_unseen_df.to_parquet(val_unseen_path, index=False)

    report_dict = {
        **{k: v for k, v in report.__dict__.items() if k != "held_out_benchmarks"},
        "held_out_benchmarks": list(report.held_out_benchmarks),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "data_category_mode": args.data_category_mode,
        "n_categories": args.n_categories,
        "category_seed": args.category_seed,
    }
    (args.out_dir / "split_report.json").write_text(json.dumps(report_dict, indent=2))
    print(f"\nWrote:\n  {train_path}\n  {val_path}\n  {val_unseen_path}\n  "
          f"{args.out_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
