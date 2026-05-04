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

### Optional: gated MLP residual

When invoked with `--use-gated-residual`, the model layers a SwiGLU-style
scalar correction on top of the base logit:

```
logit(m,b,c) = base_logit(m,b,c) + residual_scale * GatedResidual(h_mbc)

h_mbc = concat(u_m, v_bc, u_m * v_bc, |u_m - v_bc|, model_meta_hidden, bench_meta_hidden)

GatedResidual(h) = down_proj( SiLU(gate_proj(LN(h))) * up_proj(LN(h)) )
```

Design choices:

- The base logit is computed identically to the baseline; the residual is
  strictly additive. With the default `--residual-scale-init 0.0` the
  gated block contributes exactly nothing at $t=0$ and the optimizer has
  to actively grow `residual_scale` (a single learnable scalar) for the
  new MLP to have any effect. This guarantees the strong baseline cannot
  be wiped out by an unlucky residual init.
- The metadata-hidden vectors come from each tower's *last hidden
  activation* (the input to the tower's final Linear). The towers are
  refactored to optionally return that activation alongside their
  scalar/vector output, with NO change to their `state_dict` layout — so
  baseline checkpoints (`use_gated_residual=False`) saved before this
  feature still load cleanly under the new code.
- AdamW weight decay regularizes the gated block's parameters. The
  custom `id_emb_l2` penalty is **not** applied to the gated residual
  (it remains for ID lookup tables only).
- Item content is still ignored: this is still a metadata-only model.

Relevant CLI flags (all off / default-y when omitted):

```
--use-gated-residual            # turn the residual on
--gated-hidden-dim 64           # SwiGLU hidden dim
--gated-dropout    0.05         # dropout after SiLU(gate)*up
--residual-scale-init 0.0       # initial scalar (start small / zero)
--no-learn-residual-scale       # freeze the scalar (default: trainable)
```

## Files

| file | purpose |
| --- | --- |
| `latent_factor_pytorch.py`           | preprocessor (vocab + scaler), `LatentFactorModel`, optional `GatedResidual`, training loop, `LatentFactorInference` |
| `run_latent_factor_colab.py`         | end-to-end CLI: build split → train (baseline OR gated) → official-like validation → write metrics |
| `submission_template/model.py`       | runtime `model.py` copied into the validation-harness submission folder; same code handles both model variants |
| `latent_factor_colab.ipynb`          | Colab notebook for the **baseline** latent-factor model |
| `latent_factor_gated_colab.ipynb`    | Colab notebook for the **gated MLP residual** variant (same 10-section structure, just `--use-gated-residual` plus an ablation cell that compares to the baseline outputs) |

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

### Live training progress

Both single-config training (notebook section 6) and sweep runs (section 7)
emit three coordinated streams of progress:

- A **tqdm bar** over the planned `epochs * steps_per_epoch` steps with
  live `% done`, `ETA`, and a postfix showing the current `epoch`,
  current batch `loss`, and best validation log-likelihood seen so far.
- A **per-step log line** every `--log-every-steps N` (default 10) shaped
  like:
  `step 120/3600 (3.3% done with training, time estimated is 41.8s)  epoch=2/30  step_in_epoch=20/120  loss=0.5012  elapsed=1.4s`
- A **per-epoch summary** with `train_loss`, `val_ll`, `val_brier`,
  `val_auc`, `epoch_seconds`, `eta`, and the running best (with `*` if
  this epoch improved).

Set `--log-every-steps 0` to suppress the per-step prints (the tqdm bar
and per-epoch summary still emit). Set `--no-progress-bar` to suppress
the tqdm bar (per-step + per-epoch text logs still emit).

### Hyperparameter sweep

Sweep dimensions:

| dim | values |
| --- | --- |
| `latent_dim`   | 4, 8, 16, 32 |
| `hidden_dim`   | 128, 256 |
| `dropout`      | 0.05, 0.1, 0.2 |
| `weight_decay` | 1e-4, 1e-3 |
| `lr`           | 1e-3, 3e-3 |
| `id_emb_l2`    | 1e-4, 1e-3 |
| `patience`     | 5, 10 |

Full grid = 384 configs; default mode is `random` with `--sweep-budget 24`.
Use `--sweep-mode full` to walk the entire grid.

The notebook's section 7 runs the sweep **sequentially** (`--parallel-runs 1`),
so each run prints its own clean tqdm bar + per-step loss + per-epoch summary,
then we "pick up" the next run when it finishes. This is the easiest mode for
monitoring "what's going on right now".

Pass `--sweep --parallel-runs 8` (or `--parallel-runs 0` for auto) to instead
run multiple configs concurrently on one GPU via `ProcessPoolExecutor` with
the `spawn` start method. In that mode the script auto-disables the per-run
tqdm bar (multiple bars from different processes interleave very badly) but
still emits the per-step + per-epoch text logs, all tagged `[run NNN/TTT]`
for grep-friendly attribution when output interleaves. The preprocessor +
all training/validation tensors are precomputed once and pickled to
`outputs/latent_factor/cache/` so each worker reloads in ~1 s instead of
re-fitting (~10 s) and re-aggregating (~5 s). Each completed run prints a
sweep-level progress line with elapsed wall, ETA, and best-so-far val
log-likelihood.

Each run also writes its own checkpoint + preprocessor copy to
`outputs/latent_factor/runs/run_NNN/`, and the best run by full-val
log-likelihood is copied up to the canonical `outputs/latent_factor/`
location and used to build the submission folder.

### Reproducibility / saved weights

Every run writes:

- `best_model.pt`     — full bundle (`{state_dict, config_dict}`) loadable via `latent_factor_pytorch.load_artifacts`
- `weights.pt`        — raw `state_dict` only (smaller, drop-in for `LatentFactorModel.load_state_dict`)
- `preprocessor.pkl`  — fitted preprocessor with vocabularies, scalers, and metadata lookups
- `metrics.json`      — final scalar metrics for that run

After the sweep, the best run's files are copied to the top-level
`outputs/latent_factor/`. Two extra reproducibility artifacts are written
there:

- `reproduce.json`  — exhaustive manifest including:
  - the full launch `argv` and the parsed `args`
  - the best `config` (post-sweep)
  - all seeds (`model_seed`, `split_seed`, `official_seeds`)
  - md5 hashes of `train.parquet`, `val.parquet`, `model_info.csv`,
    `benchmark_info.csv` so a re-run can prove it had the same data
  - git commit, branch, and dirty flag of the local repo
  - package versions (`torch`, `numpy`, `pandas`, `pyarrow`, `sklearn`),
    Python version, platform, and GPU name
  - sweep wall + per-run completion summary
- `reproduce.sh` (Linux/Colab) and `reproduce.bat` (Windows) — one-line
  CLIs that re-run JUST the best config (no sweep) into
  `outputs/latent_factor/reproduced/`.

To re-validate a saved checkpoint without retraining, point a fresh run at
the previous output directory:

```bash
python Google_Collab_harness/run_latent_factor_colab.py \
  --resume-from outputs/latent_factor \
  --splits-dir validation_harness/splits/v1 \
  --model-info-csv starting_kit/Model_Info/model_info.csv \
  --benchmark-info-csv starting_kit/benchmark_info/benchmark_info.csv \
  --validation-harness-dir validation_harness \
  --output-dir outputs/latent_factor_revalidated
```

This skips fitting the preprocessor and training entirely, copies
`best_model.pt`/`weights.pt`/`preprocessor.pkl`/the two metadata CSVs to
the new output dir, builds the submission folder, and runs the
official-like 3-seed validation.

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
