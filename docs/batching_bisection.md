# Batching bisection — failure modes, thresholds, and isolation bundles

The batched-flush runtime has produced an opaque "error during execution"
on the platform with no surfaced details. This doc lists every failure
mode I can think of, the *threshold* (i.e. the specific observable value
that proves or disproves the hypothesis), and which mini-bundle in
`bisection/` tests it.

## Failure modes

| # | hypothesis | mechanism | threshold (proof / disproof) | bundle |
|---|---|---|---|---|
| H1 | Platform never runs our code | Bundle ZIP unpack fails, Python import error before any of our code runs | If `artifacts/runtime_progress.json` is absent AND there's no stdout, NOTHING we wrote executed. If even one `[runtime] module_init_start` print appears, this is disproved. | `01_no_torch` |
| H2 | Artifacts directory isn't writable on the platform | Our progress writer fails silently; only stdout might survive | `runtime_progress.json` never appears in the downloaded artifacts BUT stdout shows `[runtime] ...` lines | `01_no_torch` |
| H3 | `acquisition_function` is never called | Platform may skip `labeling.py` under some conditions | Progress events contain `module_init` but no `acq_*` events | `01_no_torch` |
| H4 | `acquisition_function` is called AFTER (or in a different process from) `predict()` | Module-level state from acquisition would be invisible to predict | Events show `predict_first_call` with `acq_total=0` even though acquisition events appear later, OR acquisition events never appear at all in the same file | `01_no_torch`, `03_state_share` |
| H5 | Cross-module state isolation: `labeling.py` and `model.py` get separate `model` globals | Calls from `labeling.py` into `model._enqueue_for_batch` don't mutate `model`'s queues that `predict()` reads | `03_state_share` shows `model.predict` reads a counter of 0 even though `labeling.acquisition_function` incremented it `N>0` times | `03_state_share` |
| H6 | torch / CUDA isn't actually available on the L4 image we land on | Module init runs on CPU, every forward becomes 100× slower → submission times out | Progress file shows `cuda_available=False` or `cuda_device_count=0` | `02_torch_cuda` |
| H7 | torch.bfloat16 unsupported on the platform's GPU | Some L4 setups don't expose bf16 — falls back to fp32, weights 2× larger, OOM on load | Progress file shows `cuda_available=True` but the *real* bundle fails right at `judge_loaded` with an OOM | `02_torch_cuda` (logs `is_bf16_supported`) |
| H8 | Encoder OOMs at bs=16 during co-resident phase | Tight but should fit; if it OOMs the flush dies | Real bundle's progress file shows `flush_start` but no `flush_items_done` | (real bundle only) |
| H9 | Judge OOMs at bs=32 after encoder eviction | The eviction might not actually free VRAM (e.g. dispatch holds references) | Real bundle's progress file shows `encoder_evicted` but no `flush_judge_done` | (real bundle only) |
| H10 | Older transformers raises on `attn_implementation="sdpa"` with `TypeError` (already widened) — disproved by the fix in 0c227ea, but logged for record | `_LLMJudgeRuntime.__init__` crashes during judge load | `judge_load_failed` event with `error_type=TypeError` | (real bundle only) |
| H11 | `models.txt`-declared HF repos aren't pre-fetched on the platform | `from_pretrained(local_files_only=True)` raises `OSError: not in cache` | `judge_load_failed` with `error_type=OSError` | (real bundle only) |
| H12 | Cache-key mismatch between enqueue and lookup | Queue populated but every `predict()` lookup misses → falls back to slow bs=1 path → looks like batching never happened. Already fixed by normalizing condition / benchmark / contents at the top of `_enqueue_for_batch` (commit cache-key normalization), but listed for completeness | If we still hit slow runtime AND `flush_complete` logs sensible `n_items/n_judge_pairs`, suspect this | (real bundle only) |
| H13 | `acquisition_function` per-call timeout exceeded | Handbook § 2.2 says exceeding the per-call timeout triggers random fallback; doesn't break the round but kills batching | Progress shows acquisition started but never reaches the count we expected; OR `predict_first_call.acq_total < n_candidates` | `03_state_share` (with deliberately slow acquisition) |

## Bundles

Each bundle is a self-contained ZIP under
`C:\Users\benja\Downloads\submission\` (next to the real bundle). They
are tiny (no model weights), so the upload itself is instant; the only
cost is one daily-submission slot per test.

Submit them **one at a time**, in order. After each, download the
artifacts and read `runtime_progress.json` (or stdout). The file
`latest.stage` field tells you what phase the runtime reached.

1. `submission_bisection_01_no_torch.zip` — pure-Python; no torch,
   no transformers, no model load. Verifies that the platform can run
   our submission *at all* and that artifact writes are surfaced.
   - Expected success state: progress file with at least
     `module_init`, `acq_first_call`, `predict_first_call`, and a final
     `predict_progress` event.
   - If THIS fails → the platform itself isn't running us; raise it
     with the organizers.

2. `submission_bisection_02_torch_cuda.zip` — imports torch, prints
   CUDA properties, returns 0.5/0.0. Verifies GPU availability and the
   `bf16` support flag.
   - Expected success state: progress file logs
     `cuda_available=True`, `cuda_device_name="NVIDIA L4"` (or
     similar), `bf16_supported=True`, and finishes the round.
   - If `cuda_available=False` → submissions land on CPU; nothing
     batched can fit in budget. Real bundle won't work as-is.
   - If `bf16_supported=False` → we need to ship fp16 weights or
     halve the batch size.

3. `submission_bisection_03_state_share.zip` — `labeling.py` writes
   to a module-level counter inside `model.py`; `predict()` reads it
   back. Confirms cross-module state persistence within a round.
   - Expected success state: progress file shows
     `predict_first_call.shared_counter > 0` and equal to the number
     of acquisition calls we made.
   - If `predict_first_call.shared_counter == 0` while
     `acq_total > 0` is also in the events → cross-module isolation
     is real and batching cannot work as currently designed.

After all three pass, the problem is *inside the real bundle's flush
logic*, and we ship a flush-trace bundle (the existing one plus
even more granular progress events).
