# Starting Kit

This starting kit supports the active code-submission path for the Predictive
AI Evaluation Challenge.

Download:

- the starter kit from Codabench
- the public training dataset from
  `https://huggingface.co/datasets/aims-foundations/measurement-db`

The real competition materializes a hidden test slice at runtime. The active
configuration samples 5000 hidden items per submission, stratified across data
categories, and your `predict()` is called once per hidden subject-item pair.
Codabench currently allows up to 50 scored submissions per team per calendar
day (UTC).

## Contents

```text
starting_kit/
  README.md
  sample_code_submission/
    model.py
    labeling.py
  templates/
    hf_submission/
    labeling_addon/
```

## Loading The Public Training Data

The HuggingFace repo is a collection of Parquet tables, not a single
`datasets` split. Do **not** use:

```python
load_dataset("aims-foundations/measurement-db")
```

That shortcut lets HuggingFace auto-select files and may mix response tables
with `*_traces.parquet` tables that have different schemas. Load the response
tables explicitly and keep the registry tables separate:

```python
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    name
    for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)

response_features = Features(
    {
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    }
)

responses = load_dataset(
    REPO_ID,
    data_files=response_files,
    features=response_features,
    split="train",
)
items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")
```

To convert a response row into the shape your `predict()` function sees, join
through the registry tables. The hosted runtime passes `subject_content` as a
string beginning with a `Name:` line; optional metadata may be appended as
additional lines when available.

```python
items_by_id = {row["item_id"]: row for row in items}
subjects_by_id = {row["subject_id"]: row for row in subjects}
benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}


def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def to_training_example(row):
    item = items_by_id.get(row["item_id"], {})
    subject = subjects_by_id.get(row["subject_id"], {})
    benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
    benchmark_id = benchmark.get("benchmark_id") or row["benchmark_id"]

    return {
        "benchmark": benchmark_id,
        "condition": row["test_condition"] or "none",
        "subject_content": render_subject_content(subject, row["subject_id"]),
        "item_content": item.get("content"),
        "label": row["response"],
    }
```

Use the public `benchmark_id` for the `benchmark` field to match hosted runtime
inputs; `benchmarks.parquet["name"]` is human-readable metadata and may differ
from the identifier passed to `predict()`.

Treat `subject_content` as text, not as a dict or a stable serialization of the
full `subjects.parquet` row. At test time, extra metadata lines should be parsed
defensively because they may be absent.

Some public response tables are binary and some are continuous/scored. For a
binary correctness model, filter or transform `label` values to match your
training objective. If you need raw model outputs, load the `*_traces.parquet`
files separately; they intentionally do not have the same schema as the
response tables.

## Which Starter To Use

- `sample_code_submission/`
  Minimal `model.py` plus an optional `labeling.py` example. Remove
  `labeling.py` if you want the platform's default random label sample.
- `templates/hf_submission/`
  Local HuggingFace inference using repos declared in `models.txt`.
- `templates/labeling_addon/`
  Optional `labeling.py` example.

## Required Submission Contract

Your ZIP must contain `model.py` with:

```python
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    ...
```

`input` keys:

| key | meaning |
| --- | --- |
| `benchmark` | benchmark identifier (e.g. `mmlupro`, `ai2d_test`) |
| `condition` | test condition (e.g. `zero-shot`); `"none"` when not applicable |
| `subject_content` | description of the AI subject under evaluation, including a name line and any organizer-provided metadata |
| `item_content` | the question/prompt/task text the subject is asked |

`predict()` returns a single float in `[0, 1]`: the predicted probability that
the subject answers the item correctly.

Training data lives on the public HuggingFace dataset with the same four string
input fields after joining the response tables to the registry tables shown
above. Download it and preprocess it however you prefer before submitting.

Module-level code runs once when the container starts. Do all heavy setup
(load weights, tokenizers, prompt templates, or lookup tables) at module init.
Training must happen **offline**. Small fitted state should be baked into the
submission ZIP; large checkpoints should be uploaded to a HuggingFace model
repository and declared in `models.txt`.

## Adaptive Labeling

You may include `labeling.py` with:

```python
def acquisition_function(input: dict) -> float:
    ...
```

`acquisition_function()` is called once per hidden `(model_id, item_id)` pair
before `predict()`. Higher scores indicate pairs you want labeled more. The
platform selects the top **K=5** inputs per data category, resolves their
ground-truth labels, and passes them to `predict()` as the `labeled` argument: a
list of dicts with the same shape as `input` plus a `label` field (0 or 1).

If you don't include `labeling.py`, the platform reveals a default random sample
per data category. If `acquisition_function()` raises an exception, times out,
or returns a non-finite value for any candidate, the platform falls back to that
same random-selection default for the round. Your `predict()` should handle the
empty-list case cleanly.

## HuggingFace And GPU Routing

If you need local HuggingFace repos, list them in `models.txt`. The platform
pre-downloads those repos before participant code runs and routes the submission
to the smallest GPU tier that fits the largest declared model. The active bundle
allows at most `5` repos in `models.txt`.

Active parameter bands:

| Max params | Typical GPU tier | Tier timeout |
| --- | --- | --- |
| `<= 1B` | T4 | 30 min |
| `<= 8B` | L4 | 30 min |
| `<= 20B` | A100 | 30 min |
| `<= 70B` | A100-4 or H100 | 60 min |
| `<= 140B` | A100-8 | 60 min |
| `<= 250B` | A100-mega | 60 min |

Submissions above `300B` parameters or `1000 GB` total repository size are
rejected during classification. Organizers may disable tiers operationally if
capacity changes.

If your code imports GPU or HuggingFace libraries but does not declare
`models.txt`, the platform may still route it to a GPU tier based on source-code
patterns. Do not rely on runtime HuggingFace downloads inside the submission
container.

## Runtime Policy

- submissions have **no outbound internet access** except the organizer's
  internal data-service; calling third-party hosted LLM endpoints, remote
  embedding services, external object storage, remote databases, webhooks, or
  external cloud functions is blocked
- `models.txt` guarantees that repo files are pre-downloaded, not that every
  repo will load automatically
- `trust_remote_code` is organizer-controlled and defaults to disabled in the
  example deployment
- additive submission `requirements.txt` support is organizer-controlled and
  defaults to disabled in the example deployment
- when additive dependency installs are enabled, normal named pip requirements
  are accepted into a per-submission dependency layer; avoid pip options,
  editable installs, and source-build-only packages

The organizers provide the `torch_measure` package to facilitate measurement
model implementation. Using it is not required.
