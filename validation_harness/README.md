# Validation Harness for the Predictive AI Evaluation Challenge

A local, official-like validation harness that mirrors the hosted platform's
protocol so you can iterate without burning your 50/day Codabench budget.

## What this enforces (so you can trust local scores)

- **Item cold-start**, not subject cold-start. Validation item-variants never
  appear in training; validation subjects are guaranteed to appear in training.
- A single **`item_variant_id`**: the official `item_id` if it is already
  condition-specific within its benchmark, otherwise a stable hash of
  (`benchmark`, normalized `condition`, `item_content`).
- **`condition` normalization**: `None`, `""`, `"nan"`, `"none"`, `"null"` all
  collapse to the literal string `"none"` everywhere.
- **`predict(input, labeled)` only ever sees** `{benchmark, condition,
  subject_content, item_content}` — all `str`. `data_category`, `subject_id`,
  `item_id`, `label`, etc. are never leaked.
- **`acquisition_function(input)` only ever sees the same four fields.**
- **`labeled` dicts** contain exactly those four fields plus `label`.
- **Stratified sampling by `data_category`** of N item-variants per round.
  By default `data_category` is **15 random buckets** assigned by hashing
  `item_variant_id` (deterministic across machines, no benchmark identity
  baked in -- use `--data-category-mode benchmark` for the legacy mode). Per-category
  remainders are allocated randomly but reproducibly via the seed;
  per-category supply is capped and slack is redistributed to the remaining
  categories.
- **Adaptive label budget**: top-K per category by acquisition score, random
  tie-break. Fall back to random K per category if `labeling.py` is missing,
  the module has no `acquisition_function`, any call raises, or any returned
  score is non-finite. Default K = 5.
- **Two scores reported**: `excluding_adaptively_labeled_rows` (the headline)
  and `including_all_rows` (a sanity check; not promoted unless the platform
  proves it scores labeled rows too).
- **Fresh containers across rounds**: `Submission.reset()` reloads
  `model.py` / `labeling.py` so module-level adaptive state is cleared between
  seeds, exactly as the platform spawns a new container per submission.

## Layout

```text
validation_harness/
  harness/                       # python package
    __init__.py
    utils.py                     # INPUT_FIELDS, normalize_condition, row_to_input
    data_loader.py               # load + join Data/*.parquet  -> joined df
    splits.py                    # add_item_variant_id, make_item_cold_start_split
    sampling.py                  # stratified_sample_variants
    rounds.py                    # run_official_like_round
    scoring.py                   # mean_log_likelihood, auc_roc, score_round
    submission.py                # Submission wrapper with reset()
  scripts/
    prepare_split.py             # CLI: build train.parquet / val.parquet
    run_validation.py            # CLI: run one or more rounds vs a submission
  example_submissions/
    constant_baseline/           # always 0.5
    random_baseline/             # mean of revealed labels + toy acquisition
  tests/                         # pytest suite (8+ invariants)
  requirements.txt
  README.md
```

## Usage

### 1) Install deps

```powershell
py -m pip install -r requirements.txt
```

### 2) Build a split (one-time per seed/version)

Saves `train.parquet`, `val.parquet`, `val_unseen_subjects.parquet`, and
`split_report.json` under `--out-dir`.

```powershell
py scripts/prepare_split.py `
  --data-dir "..\starting_kit\Data" `
  --out-dir   "splits/v1" `
  --val-fraction 0.10 `
  --seed 0
```

You then **train your model on `train.parquet`** (the harness does not
prescribe how — your training script is whatever you want, as long as it
never peeks at `val.parquet`).

For the stricter "held-out benchmark" stress test, repeat `--holdout-benchmark`:

```powershell
py scripts/prepare_split.py ... --holdout-benchmark cybench --holdout-benchmark hle
```

### 3) Run validation

```powershell
py scripts/run_validation.py `
  --submission "example_submissions/random_baseline" `
  --val-parquet "splits/v1/val.parquet" `
  --train-parquet "splits/v1/train.parquet" `
  --N 5000 --K 5 --seeds 0 1 2
```

Each seed simulates a fresh container (modules reloaded). Output columns:
`log_likelihood_excl_labeled` (the headline), `auc_roc_excl_labeled`,
`log_likelihood_incl_all`, `auc_roc_incl_all`, `n_labeled`, `n_categories`,
`used_random_acquisition`, `fallback_reason`.

### 4) Use the harness from your own code

```python
from harness import (
    Submission,
    run_official_like_round,
    score_round,
)
import pandas as pd

train_df = pd.read_parquet("splits/v1/train.parquet")
val_df   = pd.read_parquet("splits/v1/val.parquet")

sub = Submission("path/to/your/submission")

scores = []
for seed in range(5):
    sub.reset()
    result = run_official_like_round(
        train_df, val_df, sub.model, sub.labeling,
        N=5000, K=5, seed=seed,
    )
    s = score_round(result)
    scores.append(s.main())  # mean log-likelihood, excluding labeled rows
print("Mean LL (excluding labeled):", sum(scores) / len(scores))
```

## Tests

From the `validation_harness/` directory:

```powershell
py -m pytest -q
```

The test suite verifies, on synthetic data, every invariant from the spec:

| # | Invariant | Test |
| - | --- | --- |
| 1 | No validation `item_variant_id` appears in training | `test_splits.py::test_no_variant_overlap_between_train_and_val` |
| 2 | Every validation subject appears in training | `test_splits.py::test_every_validation_subject_appears_in_training` |
| 3 | `predict` receives exactly the four allowed string fields | `test_rounds.py::test_predict_receives_exactly_four_string_fields` |
| 4 | `acquisition_function` receives exactly the four allowed string fields | `test_rounds.py::test_acquisition_receives_exactly_four_string_fields` |
| 5 | `labeled` dicts contain exactly the four fields plus `label` | `test_rounds.py::test_labeled_dicts_contain_exactly_four_fields_plus_label` |
| 6 | `data_category` (and other bookkeeping cols) never passed into participant code | `test_rounds.py::test_data_category_never_passed_to_predict_or_acquisition` |
| 7 | Adaptive label budget ≤ K × #categories present | `test_rounds.py::test_label_budget_at_most_K_times_categories_present` |
| 8 | Invalid acquisition (raise OR non-finite) triggers random fallback for the whole round | `test_rounds.py::test_acquisition_exception_triggers_random_fallback`, `…_nonfinite_…` |

Plus tests for: per-category supply caps, reproducibility under seed,
`labeled` list passed identically to every `predict()` call, scoring math,
and `Submission.reset()` actually clearing module-level state.

## Design choices worth knowing

- **`data_category = 15 random buckets`** by default. The platform's exact
  category mapping is NOT published; rather than baking the benchmark identity
  into the category (which would let benchmark structure leak through the
  stratification step), the harness assigns each `item_variant_id` to one of
  15 buckets via a stable sha1 hash. Same variant always lands in the same
  bucket (so per-variant sampling stays consistent), assignment is independent
  of benchmark, and the bucket count / hash seed are CLI-configurable
  (`--n-categories`, `--category-seed`). A legacy `--data-category-mode benchmark`
  is still available for comparison.
- **Labels are kept as floats in `[0, 1]`** for log-likelihood (Bernoulli is
  well-defined for soft labels). Some response tables in this dump are
  continuous / out-of-range (e.g. judge scores like 8.5 in `rewardbench` and
  `ultrafeedback`); the README warns about this. The harness defensively clips
  `y_true` to `[0, 1]` before scoring so the log-likelihood is always in
  `(-∞, 0]` and finite, and reports `frac_labels_clipped` so you can see when
  this is happening. **You should still binarize / rescale these labels in
  your training pipeline** to match the platform's binary-correctness target;
  the harness does not binarize for you. AUC-ROC is only computed when labels
  are effectively binary; otherwise it is reported as `null`.
- **`labeled` is the same list across every `predict` call in a round.** This
  matches the platform: the K-per-category labels are revealed once at the top
  of the round and then `predict()` is called many times with the same context.
- **Module-level state persists within a round.** Both `predict` and
  `acquisition_function` may rely on caches set up at module import time; the
  harness only resets state *between* rounds via `Submission.reset()`.
- **`acquisition_function` is called in a strict single pass**, with no access
  to the candidate list except via its own module-level state. Same as the
  platform spec.
