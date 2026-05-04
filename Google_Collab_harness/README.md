# Google Colab harness for the metadata-only latent-factor model

This folder contains everything needed to train, validate, and package a
PyTorch metadata-only latent-factor model for the Predictive AI Evaluation
Challenge on a Colab GPU runtime (an A100 is recommended but a T4/L4 is
fine since the model is small — the bottleneck is data movement, not
matmul).

## What the model does

Given the four-string runtime contract (`benchmark`, `condition`,
`subject_content`, `item_content`), the model **intentionally ignores
`item_content`** and computes:

```
eta_{m,b,c} = mu                                     # global bias
              + a_m                                  # model ability  (scalar)
              + b_{b,c}                              # bench/cond easiness  (scalar)
              + dot(u_m, v_{b,c}) / sqrt(k)          # bilinear skill x requirement
p          = sigmoid(eta)
```

with two MLP towers + residual ID embeddings:

- `(a_m, u_m) = ModelTower(model_metadata, model_id_residual)`
- `(b_bc, v_bc) = BenchmarkTower(benchmark_metadata, condition, benchmark_condition_residual)`

It uses cross-effects only via the bilinear `(u_m . v_bc)` term — no
hand-crafted feature crosses.

## Files

| file | purpose |
| --- | --- |
| `latent_factor_pytorch.py`         | preprocessor (vocab + scaler), `LatentFactorModel`, training loop, `LatentFactorInference` |
| `run_latent_factor_colab.py`       | end-to-end CLI: build split → train → official-like validation → write metrics |
| `submission_template/model.py`     | runtime `model.py` copied into the validation-harness submission folder |
| `latent_factor_colab.ipynb`        | the Colab notebook (clones repo, downloads parquets, runs everything) |

## Running on Colab

1. Open `latent_factor_colab.ipynb` in Colab and select Runtime →
   Change runtime type → GPU (A100 if available, otherwise L4 / T4).
2. Run all cells. The notebook will:
   - clone https://github.com/bwathomas/Prediction-Competition-321M
   - download the response parquets from
     `aims-foundations/measurement-db` on HuggingFace
   - build the official item-cold-start split via the existing
     `validation_harness/scripts/prepare_split.py`
   - train the latent-factor model (default ~30 epochs, early-stops on
     val log-likelihood)
   - build a submission folder containing the trained checkpoint +
     preprocessor + a `model.py` adapter
   - run `validation_harness/scripts/run_validation.py` across 3 official
     seeds (`N=5000`, `K=5`)
   - print a comparison vs the constant baseline, the train base-rate
     baseline, and the existing logistic baseline (`-0.5224`).

## Running locally (smoke test)

```powershell
python run_latent_factor_colab.py `
  --data-dir "..\starting_kit\Data" `
  --splits-dir "..\validation_harness\splits\v1" `
  --model-info-csv "..\starting_kit\Model_Info\model_info.csv" `
  --benchmark-info-csv "..\starting_kit\benchmark_info\benchmark_info.csv" `
  --validation-harness-dir "..\validation_harness" `
  --output-dir "outputs/latent_factor" `
  --latent-dim 16 --epochs 30 --batch-size 65536
```

For the optional grid (~24 runs over `latent_dim x weight_decay x dropout`)
add `--sweep`.

## Important notes on the runtime contract

- The official platform passes only the four strings to `predict()`. At
  inference time we extract the model name from the `Name:` line of
  `subject_content` and look up its row in `model_info.csv`. Models not
  present in `model_info.csv` fall back to the `__UNK__` bucket of every
  categorical embedding and the train-set median for numerics, with
  missingness indicators set so the model knows it has reduced info.
- `model_info.csv` and `benchmark_info.csv` are **baked into the
  submission folder** by `_build_submission_folder` so the runtime does
  not depend on the source repo paths.
- We never train on or peek at validation labels / item identity; the
  preprocessor is fit on `train.parquet` only.

## Comparison expectations

The headline number to beat is **−0.5224** (mean log-likelihood across
seeds 0/1/2 for the existing no-cross-term logistic baseline). The
latent-factor model adds:

1. learned model x benchmark interactions via the bilinear term
   (`dot(u, v) / sqrt(k)`)
2. residual per-`(benchmark, condition)` embeddings on top of the
   metadata-only tower output
3. residual per-model embeddings (regularized)

so even at small `k` (4–8) it should improve over logistic. Report
results in `outputs/latent_factor/baseline_comparison.csv`.
