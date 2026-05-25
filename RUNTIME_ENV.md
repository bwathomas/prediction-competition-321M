# Runtime environment audit (Step 0 of stacked-ensemble task)

This file records the evidence-backed package list for the Codabench /
PAIEC submission sandbox. Anything not in this file MUST NOT be imported
in `model.py` or `labeling.py`. The Docker image is not directly
obtainable for `pip freeze`, so this is reconstructed from:

1. The starter-kit `templates/hf_submission/model.py` (organizer-provided
   reference template).
2. The competition guide `CODABENCH_SUBMISSION_GUIDE.md` (`§1`,
   "Hard constraints").
3. The empirical post-mortem `docs/batching_bisection.md`, where the only
   bundle that ever passed was a 15 MB slim bundle with **no** `cache/`
   and **no** `requirements.txt`. Every 114 MB+ bundle that included a
   `requirements.txt` containing `faiss-cpu` failed with
   `PAIEC-UNKNOWN-001` within 1-5 minutes.
4. The competition starter-kit `README.md` ("Runtime Policy" section):
   "additive submission `requirements.txt` support is organizer-controlled
   and **defaults to disabled** in the example deployment".

## Confirmed-available runtime packages

These packages are imported successfully by the organizer-provided
reference template and/or by submissions that have passed Codabench
in the past:

| Package | Evidence |
| --- | --- |
| Python stdlib | always |
| `torch` | starter `templates/hf_submission/model.py` line 61; every passing submission |
| `transformers` (`AutoModel`, `AutoTokenizer`, `PreTrainedTokenizerFast`) | starter template line 62; CODABENCH guide §2.1 |
| `numpy` | every passing submission, including the slim 15 MB bundle |
| `safetensors` | CODABENCH guide §1 base-image list |
| `huggingface_hub` | CODABENCH guide §1 base-image list |
| `tokenizers` | transitive dep of `transformers`, always present |
| `sentencepiece` | transitive dep of `transformers`, always present |
| `torch_measure` | organizer-provided per starter `README.md` "Runtime Policy" |

## Known-unavailable / treat-as-unavailable

| Package | Evidence |
| --- | --- |
| `faiss` (any flavor) | `docs/batching_bisection.md` §UPDATE-2026-05-20: bundles with `faiss-cpu>=1.7.4` in `requirements.txt` ALL failed `PAIEC-UNKNOWN-001`. The slim bundle that worked had no FAISS at all. **Hard rule: zero `import faiss` in runtime code.** |
| `scikit-learn` | Not on the CODABENCH guide base-image list; no passing submission has imported it. The "logistic baseline" example in `validation_harness/example_submissions/` IS sklearn-based but lives in the validation harness (which has its own pip env), not in the live Codabench runtime. |
| `lightgbm`, `xgboost` | Not on the base-image list; no passing submission. Tree models must be exported as data and traversed in pure numpy at runtime. |
| `m2cgen` | Offline-only; not on the base-image list. |
| `joblib` | Used by the validation-harness logistic baseline but not by any passing live submission. Treat as unavailable; replace with `numpy.savez` or `torch.save`. |
| `scipy` | No evidence of availability. The current `qwen8b_minimalist` runtime tries `from scipy import sparse` inside a `try/except`; if scipy is missing the runtime silently disables that path. **For the new stacked submission we ship sparse data as plain `.npz` (CSR via three numpy arrays: `indptr`, `indices`, `data`) and reconstruct in pure numpy.** |

## Soft-uncertain (use only with try/except + fallback, never required)

| Package | Note |
| --- | --- |
| `langdetect` | Current submission imports it with `try/except` and falls back to an ASCII heuristic. Acceptable pattern. |
| `einops` | CODABENCH guide §1: "availability is uncertain". Inline all `rearrange`/`repeat` as pure PyTorch. |
| `pandas` / `pyarrow` | Starter kit lists them in its OFFLINE `requirements.txt`; not confirmed for live runtime. Avoid in `model.py`. |
| `sentence_transformers` | The user's task prompt lists it as confirmed; not in the CODABENCH guide. We do not need it for the new pipeline (raw `transformers` AutoModel + manual mean-pool is enough). Avoid. |

## ZIP-size budget

Empirical: the only submission that ever passed was the **15 MB** slim
bundle. Bundles at **114 MB+** all failed `PAIEC-UNKNOWN-001`. The
configured ceiling in `configs/default.yaml` is `submission.max_zip_size_mb
= 70`. **Target for the new stacked bundle: <= 65 MB** to leave a
buffer.

The Qwen3-Embedding-8B encoder weights (`~16 GB`) are NOT in the ZIP;
they are pre-fetched via `models.txt`. The ZIP carries only
artifacts (checkpoints, calibrator state, neighbor tables, GBDT trees)
plus `model.py` / `labeling.py`.

## Hard rules for `model.py` / `labeling.py`

1. **Never** `import faiss` (or any of its variants).
2. **Never** `import sklearn`, `sklearn.linear_model`, `sklearn.ensemble`,
   `sklearn.neighbors`.
3. **Never** `import lightgbm`, `import xgboost`, `import m2cgen`.
4. **Never** `import joblib`. Use `numpy.savez_compressed` or
   `torch.save`.
5. **Never** `import scipy` without a `try/except` AND a numpy fallback.
   For the new stacked submission, do not rely on scipy at all -- ship
   CSR matrices as `(indptr, indices, data, shape)` `.npz` triples.
6. `langdetect` and `einops` may be imported only inside `try/except`
   with a pure-`torch`/`numpy` fallback.
7. Bundle ship list must total <= 65 MB before the encoder weights
   (which are pre-fetched via `models.txt`).
8. No `requirements.txt` in the ZIP. The runtime ignores it at best
   and rejects the bundle at worst.

## Allowed import list (whitelist) for the new `model.py` and `labeling.py`

```python
# stdlib only
from __future__ import annotations
import math, json, os, sys, hashlib, logging, re, time
from pathlib import Path
from typing import Any
# numpy
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# transformers (loaded lazily inside the encoder loader, not at top of file)
from transformers import AutoModel, AutoTokenizer
```

Anything else must justify itself against the rules above before it
ships.
