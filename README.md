# A100 ablation pipeline -- Predictive AI Evaluation Challenge

A reproducible, A100-oriented training pipeline that compares three models on
the [aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
dataset under **item cold-start validation** -- the regime the hosted
Codabench platform actually scores against:

1. **K-factor / neural-IRT** (`kfactor`)
2. **K-factor + ordinary MLP residual** (`kfactor_mlp`)
3. **K-factor + gated MLP / SwiGLU residual** (`kfactor_gated_mlp`)

The notebook downloads the dataset, embeds every unique item / subject text
once with a heavy HF encoder (default: `intfloat/e5-mistral-7b-instruct`),
trains all three model variants across multiple seeds and `k` values, builds
a results table sorted by item-cold-start log-loss, and lets you pick
which trained run to export into the required Codabench submission format.

> **Methodological rule (read this first):** primary model selection MUST
> use the item-cold-start validation split. The notebook also reports a
> leaky random-row split for sanity, and flags any residual model that
> only improves random-row but not item-cold-start (which means it's
> overfitting to row-level leakage).

## Repository layout

```text
.
├── configs/
│   └── default.yaml                  # all hyperparameters, paths, knobs
├── notebooks/
│   ├── a100_ablation_notebook.py     # paired script (edit in Cursor)
│   └── a100_ablation_notebook.ipynb  # generated from the .py
├── scripts/
│   ├── build_ipynb_from_py.py        # zero-dep .py -> .ipynb converter
│   └── smoke_test_submission.py      # offline submission smoke test
├── src/
│   ├── data.py                       # HF download + join + keys + splits
│   ├── embeddings.py                 # HF encoder + caching + token stats
│   ├── models.py                     # KFactor, MLP, gated MLP (SwiGLU)
│   ├── train.py                      # AdamW + bf16 + cosine + early stop
│   ├── eval.py                       # metrics, ECE, slices, plots
│   ├── calibration.py                # temperature / intercept calibrator
│   ├── sanity_checks.py              # data/embedding/model invariants
│   └── export_submission.py          # bundle to submission/ + submission.zip
├── requirements.txt
└── README.md
```

## Quick start: Google Colab

1. Open a new Colab notebook with an A100 runtime (Runtime → Change runtime type → A100 GPU).
2. In the first cell, paste:
   ```python
   !curl -sSL https://raw.githubusercontent.com/bwathomas/prediction-competition-321M/main/notebooks/a100_ablation_notebook.ipynb -o a100_ablation_notebook.ipynb
   ```
   Then open `a100_ablation_notebook.ipynb` from the Colab file browser.
   Alternatively just upload `notebooks/a100_ablation_notebook.ipynb` from this repo.
3. Run the notebook. Cell 0 clones this repo into `/content/prediction-competition-321M/`,
   cell 1 installs `requirements.txt`, cell 4 prompts for `HF_TOKEN` (or reads it
   from the env var if you set one in Colab's secret manager / form widget first).

## 1. Spin up a Vertex AI Workbench instance

The pipeline assumes a single A100 (or, in a pinch, any GPU with bf16 +
~24 GB VRAM if you swap to a lighter encoder).

1. Open Vertex AI Workbench in the Google Cloud Console.
2. Create a new **User-managed notebook** with:
   - Environment: PyTorch 2.x (CUDA 12.x)
   - Machine type: `a2-highgpu-1g` (1 × A100 40 GB) or larger
   - Optional: enable `bf16`-capable instance (A100 / H100)
3. Once the notebook starts, open a terminal and clone this repo:
   ```bash
   git clone <this-repo> && cd <this-repo>
   ```
4. Install the runtime requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Set the Hugging Face token

The encoder defaults to `intfloat/e5-mistral-7b-instruct`, which is a gated
repo. You need a Hugging Face token with read access.

The pipeline resolves the token in this order, never logging or persisting
it anywhere:

1. `HF_TOKEN` environment variable.
2. Google Secret Manager secret named `HF_TOKEN` (requires
   `google-cloud-secret-manager` and an attached service account; the
   project is read from `GOOGLE_CLOUD_PROJECT` or `GCP_PROJECT`).
3. Interactive `getpass.getpass` prompt at notebook runtime.

Easiest path on Vertex AI:

```bash
# Either set HF_TOKEN in the kernel environment ...
export HF_TOKEN=...

# ... or store it once in Secret Manager and let the notebook fetch it:
echo -n "$HF_TOKEN" \
  | gcloud secrets create HF_TOKEN --data-file=- --replication-policy=automatic
gcloud secrets add-iam-policy-binding HF_TOKEN \
  --member="serviceAccount:$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" \
  --role="roles/secretmanager.secretAccessor"
```

## 3. Run the notebook

Open `notebooks/a100_ablation_notebook.ipynb` in JupyterLab and Run All.

The notebook cells are intentionally numbered:

0. Clone the project repo from GitHub (Colab / fresh Vertex AI instances).
1. Install pinned requirements and import the stack.
2. GPU banner -- fails loudly if no GPU (set `ALLOW_CPU=1` to override).
3. Load configuration from `configs/default.yaml`.
4. Resolve HF token and `huggingface_hub.login()`.
5. Download + join + key the dataset, print dataset statistics.
6. Build the splits (item cold-start, optional benchmark heldout, random-row debug).
7. Data sanity checks (leakage, key stability, missing values, ...).
8. Build the HF encoder and embed unique items / subjects (cached).
9. Embedding sanity checks (NaN/inf, zero-norm, truncation, determinism).
10. Build the indexer and training matrices.
11. Model sanity checks (forward pass, overfit a tiny batch, random labels).
12. Baselines (global mean / subject shrinkage / bc shrinkage / logistic on embeddings).
13. Train all three model variants across seeds and `k` values.
14. Evaluate each trained checkpoint on every split, build the results table.
15. Random-row overfit flag (warns if a residual only helps the leaky split).
16. Slicewise metrics (benchmark / condition / subject family / token length).
17. Plots (val log-loss by model, calibration curves, by-benchmark deltas, ...).
18. Choose a trained run to export (`ipywidgets` dropdown or `SELECTED_RUN_ID`).
19. Export the selected run to `submission/` and `submission.zip`.
20. Run the in-notebook smoke test on 20 held-out validation rows.
21. Optional GCS sync of `artifacts/` and `submission/`.

The paired `notebooks/a100_ablation_notebook.py` carries the same cells with
`# %%` markers, so you can edit in Cursor and rebuild the `.ipynb` with:

```bash
py scripts/build_ipynb_from_py.py notebooks/a100_ablation_notebook.py
```

## 4. Change the encoder model

Edit `configs/default.yaml`:

```yaml
encoder:
  model_id: "intfloat/e5-mistral-7b-instruct"   # default heavy A100
  # model_id: "sentence-transformers/all-mpnet-base-v2"   # ~110 MB
  # model_id: "intfloat/e5-large-v2"                      # ~1.3 GB
  # model_id: "BAAI/bge-large-en-v1.5"                    # ~1.3 GB
```

Lighter encoders run on T4 / L4. Heavy encoders need an A100 (and benefit
from `bf16: true`).

Some encoders need text prefixes (`query: ...`, `passage: ...`). Set
`query_prefix` and `passage_prefix` in the same config block.

## 5. Run ablations

The notebook iterates over every combination in:

```yaml
train:
  models: ["kfactor", "kfactor_mlp", "kfactor_gated_mlp"]
  k_factors: [16, 32]
  seeds: [0, 1, 2]
```

Each `(model_name, k, seed)` becomes one checkpoint in
`artifacts/checkpoints/{run_id}.pt` plus a sibling
`{run_id}.json` with the full config + loss curve.

Results land in `artifacts/results/results.csv`, sorted by item-cold-start
val log-loss (lower is better).

## 6. Pick which model to export

By default the notebook picks the best run by item-cold-start val log-loss.

To override:

- **Interactive**: cell 18 builds an `ipywidgets.Dropdown` if `ipywidgets`
  is installed. Pick a run and re-run the export cell.
- **Manual**: set `SELECTED_RUN_ID = "kfactor_mlp_k16_seed0"` in cell 18.

Cell 19 then exports the chosen run.

## 7. Submission contract

The exported `submission/model.py`:

- Defines `predict(input: dict, labeled: list[dict] | None = None) -> float`.
- Loads the encoder + checkpoint at **module scope** (no work inside `predict()`
  except a single encoder forward and one sigmoid).
- Uses the **local HF cache only**: `local_files_only=True`. No outbound
  network calls.
- Declares the encoder repo in `submission/models.txt` so the platform pre-fetches it.
- Returns a native Python float clipped to `[1e-6, 1 - 1e-6]`.
- Handles unseen subjects / benchmark-conditions via UNK index 0 in each space.
- Honors the four-field input contract exactly: `benchmark`, `condition`,
  `subject_content`, `item_content`. Condition is re-normalized to the literal
  `"none"` when missing/blank/null.

The exported `submission/labeling.py` is an uncertainty-based acquisition
function (`-abs(p - 0.5)`) that calls `model._predict_uncalibrated()` to
score candidates. It returns 0.0 on any failure so the platform falls back
to its random-K-per-category default cleanly.

If `labeled` is passed (the platform reveals K=5 labels per data category
per round), the runtime fits a tiny calibrator on top of the base model:

- Identity if no labels.
- Intercept-only shift if 5 ≤ N < 30.
- Temperature + intercept if N ≥ 30.

The calibrator falls back to identity on any numerical failure.

## 8. Smoke-test the submission

In the notebook, cell 20 runs the in-notebook smoke test. From the command
line:

```bash
py scripts/smoke_test_submission.py \
  --submission submission \
  --val artifacts/data \
  --n 20
```

The script:

- Imports `submission/model.py`.
- Calls `predict()` on 20 rows.
- Verifies each output is a native `float`, finite, and in `[0, 1]`.
- Captures stdout/stderr to check no `HF_TOKEN` value leaks to logs.
- Verifies `models.txt` exists.
- Re-zips `submission/` to `submission.zip`.

## 9. Optional: GCS artifact sync

Set in `configs/default.yaml`:

```yaml
gcs:
  bucket: "gs://my-bucket/predictive-ai-eval"
  sync_artifacts: true
```

Cell 21 uploads `artifacts/` and `submission/` to GCS via
`google-cloud-storage`. Secrets are never synced.

## A100 / bf16 / determinism notes

- `bf16: true` in the encoder block uses bf16 matmuls on A100 (and stays in
  fp32 otherwise). It does NOT change the model's final precision because
  the saved checkpoint is materialized in fp32.
- We set seeds for numpy / torch / Python, but with cudnn benchmark + bf16
  there is still small run-to-run nondeterminism; multiple seeds in the
  training loop account for this.
- We don't fine-tune the encoder. A TODO hook in `src/models.py` /
  `src/embeddings.py` reserves space for LoRA, but v1 trains only the small
  k-factor / residual heads on cached embeddings. That's why a single
  A100 can finish the full ablation comfortably within an interactive session.

## Acknowledgements

This pipeline reuses normalization rules and item-variant logic from the
official validation harness so local training and the hosted Codabench
runtime agree on item cold-start semantics.
