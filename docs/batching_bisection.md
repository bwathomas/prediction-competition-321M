# Batching bisection — failure modes and isolation under pass/fail-only

## Platform constraint

The platform surfaces **only one bit per submission**: pass / fail
(plus the secondary "ran before failing" flag, which tells us module
init succeeded). No stdout, no artifacts, no log files. So our
diagnostics must encode the answer into the pass/fail outcome itself
rather than into files we can't read.

The strategy is a one-knob-per-bundle bisection: each bundle differs
from the next by exactly one variable. The first pass-to-fail
transition in the submission sequence identifies the breaking knob.

## What we already know

- The current `submission_turbo_judge_batched.zip` (encoder_bs=16,
  judge_bs=32, free_encoder_after_flush=true) **runs before failing** —
  so module init (encoder load + judge load + checkpoint load + cache
  load) is succeeding. The crash is somewhere in `predict()` /
  `_flush_pending_batches()` / `acquisition_function()` execution.
- The previous bundle (judge_bs=16, no eviction, none of my recent
  changes) completed (slowly) — so the basic batched-flush architecture
  is not fundamentally incompatible with the platform.
- The regression was introduced in commit `0853e5f` (encoder eviction +
  bigger judge bs + progress writes + attn-impl fallback chain).

## Failure modes (post-platform-constraint)

| # | hypothesis | mechanism | pass/fail signature |
|---|---|---|---|
| F1 | Platform itself can't run any submission | Any binary; even a 2-KB pure-Python return-0.5 fails. | `submission_bisection_01_no_torch.zip` fails. |
| F2 | Regression in my non-meta-controlled model.py code | E.g. `_write_progress` raises during a code path, attn fallback misroutes, embed-zero path crashes downstream. None of these are guarded by `runtime_meta.json`. | `real_A_safe` (meta=16,16,false) fails despite matching the previous slow-working bundle's behaviour. |
| F3 | `_free_encoder_vram` itself is broken | `_ENCODER.to("cpu")`, refcount cleanup, or `torch.cuda.empty_cache()` raises. The flush dies. | `real_B_eviction` (meta=16,16,true) fails while `real_A_safe` passed. |
| F4 | bs=32 judge OOMs at the actual workload | Activations at seq=512 × bs=32 + judge weights co-resident exceed 24 GB. | `real_C_bs32` (meta=16,32,false) fails while `real_A_safe` passed. |
| F5 | bs=32 + eviction combination breaks (subtle interaction) | E.g. eviction releases memory in a fragmented pattern that bs=32 can't recover from. | `real_D_full` fails while A, B, AND C all pass. |
| F6 | Intermittent / flaky | E.g. CUDA driver hiccup, models.txt pre-fetch race. | All variants pass on retry; nothing is reproducible. |

## Bundles

All four variants share the exact same `model.py` and `labeling.py`
(the current batched-flush code with progress beacons + encoder
eviction + attn-impl fallback). They differ only in
`artifacts/runtime_meta.json`:

| bundle | encoder_runtime_batch_size | judge.runtime_batch_size | free_encoder_after_flush |
|---|---|---|---|
| `submission_bisection_real_A_safe.zip`     | 16 | 16 |  false |
| `submission_bisection_real_B_eviction.zip` | 16 | 16 |  true  |
| `submission_bisection_real_C_bs32.zip`     | 16 | 32 |  false |
| `submission_bisection_real_D_full.zip`     | 16 | 32 |  true  |

Plus the platform-level sanity check:

| bundle | what it tests |
|---|---|
| `submission_bisection_01_no_torch.zip` | 2-KB pure-Python; returns 0.5 / 0.0. Does the platform run our code at all? |

## Submission order and decision table

Submit one at a time, in order. Stop at the first FAIL — that's where
the bug is. Per-team budget is 50/day, so even if you go all the way
through 5 variants this uses 10 % of one day's quota.

```
1. submission_bisection_01_no_torch.zip
2. submission_bisection_real_A_safe.zip
3. submission_bisection_real_B_eviction.zip
4. submission_bisection_real_C_bs32.zip
5. (optional) submission_bisection_real_D_full.zip  -- same as current bundle
```

| outcome | diagnosis | fix direction |
|---|---|---|
| 01 fails | platform won't run any submission | escalate to organizers; nothing to fix on our side |
| 01 ok, A fails | regression in my non-meta-controlled model.py code | revert `src/export_submission.py` to before `0853e5f`, re-pack |
| 01,A ok; B fails | `_free_encoder_vram` is broken | disable eviction by default; investigate locally why `to("cpu")` / `empty_cache` raises on the platform |
| 01,A,B ok; C fails | bs=32 judge OOMs even with encoder evicted (i.e. eviction doesn't free enough VRAM at runtime) | drop judge default to bs=24; rebuild |
| 01,A,B,C ok; D fails | bs=32 + eviction interaction is the culprit | run bs=24 + eviction or bs=32 + no eviction |
| all five pass | the original failure was intermittent | resubmit current bundle; if it fails again, we re-bisect |
