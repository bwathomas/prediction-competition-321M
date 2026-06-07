# NIGHT PLAN 2026-06-07 — autonomous IRT→canon→final-stack pipeline

User asleep, **full authority granted** to decide the canon and train final models. Map: **colab2=qwen,
colab=nemotron, colab3=lgai**. DR=`/content/drive/MyDrive/prediction-competition-321M`. Branch
`feat/principled-export-pipeline`. Metric = mean soft BCE, honest **GroupKFold(item)** OOF (item cold-start).
**Never `pandas.to_csv` on colab2/3** (numpy/pandas ABI bug) — use the csv module + atomic os.replace.
Driver = detached subprocess `/content/irt3_driver.py`; status `/content/irt3_driver_<fam>.json`,
log `/content/irt3_full_<fam>.log`. See memory [[irt-3variant-run]].

## STAGE A — IRT 3-variant run (IN FLIGHT)
3 variants × folds 0,1,2 per family, code `c2f55a5` (OOM + grad-clip fixed). Variants:
- **irt2** (richer MIRT, K=32, profile-amortized θ + benchmark factor + condition difficulty) — UNSTABLE
  across families (qwen 0.514, nemo 0.500, lgai 0.665 on fold0); judge after full 3-fold OOF.
- **irt_lib** (5 factor sub-models → `p_irtlib__*` + logit-mean combo `p_full`).
- **irt_bag** (M=16 bagged basic-IRT; most stable, ~0.476 expected).
DONE when all 3 `irt3_driver_<fam>.json` show `finished` with irt2/irt_lib/irt_bag × folds 0,1,2 OK.
OOF written to `DR/ship/exp_loo/<fam>/<variant>_full_fold<f>/preds/oof_preds.npz` (position-aligned).

## ⚠️ INCIDENT — qwen tab (colab2) RECYCLED ~2026-06-07 (mid-irt_lib f2)
colab2 lost its runtime (host a83fc5a1715c, Drive unmounted, pc321 gone, no GPU). CANNOT remount headless →
**flag for morning user remount.** RECOVERABLE via shared Drive: qwen `irt2 f0/1/2` + `irt_lib f0/1/2` OOFs
already SAVED on Drive (clean c2f55a5). MISSING for qwen: **irt_bag f0/1/2** (the `irt_bag_full_fold0` on Drive
is the stale M=2 smoke). RECOVERY: once nemotron or lgai frees (after its irt_bag finishes), run qwen irt_bag
f0/1/2 there — SHIP_FAMILY=qwen reads qwen embeddings from the shared Drive, so no qwen tab is needed. Same for
Stage D qwen-family canon training: run on a freed tab. Do NOT block the whole pipeline on the dead tab.

## STAGE B — stacking eval (after A)
On colab2, `master_stack.py` (already on /content) fits the linear + non-neg master stack over each family's
base member pool. Append the new IRT columns (irt2 `p_full`, irt_lib `p_full` + 5 `p_irtlib__*`, irt_bag
`p_full`) to the pool; re-run non-neg + linear + leave-one-out drop-importance vs the current irt baseline.
Report each variant's stack-lift + drop-Δ. **Append to `docs/OVERNIGHT_master_2026-06-06b.md` RESULTS LOG.**
Prior 4-way: base 0.42153; +IRT 0.42041 (+0.0011, earns a seat); +kNN +0.0002 (does not). Non-neg base
weights: only **mlp 0.81 + etbig 0.26** nonzero; xgb/dae/cnn1d/logreg/fm = 0.

## STAGE C — decide the CANON (unilateral)
Canon = members with POSITIVE weight in the final non-neg linear stack. Baseline expectation:
**mlp-LOO (stacked_oof) + etbig + best-IRT-variant**, per family. Add a model only if drop-importance /
non-neg weight shows it contributes; drop redundant ones (kNN, logreg, fm, cnn1d, dae likely out). Pick the
single best IRT variant (by stack-lift). Document the choice + rationale in the RESULTS LOG.

## STAGE D — train the CANON on full data, ALL 3 families
`SHIP_ROW_SOURCE=full` (all 4.5M rows, honest 3-fold OOF). Per canon member:
- **mlp**: `SHIP_MODEL=mlp SHIP_MODE=loo` → p_full + p_loo__* + `stacked_oof` (the honest GBM stack of its LOO members).
- **etbig**: `SHIP_MODEL=etbig SHIP_MODE=full`.
- **irt-winner**: `SHIP_MODEL=<irt2|irt_lib|irt_bag> SHIP_MODE=full`.
Reuse existing artifacts if present AND current-code; re-run what's missing/stale. One family per tab, detached.
(Most full-model OOFs already exist from prior sessions; the new piece is the chosen IRT variant + any re-fits.)

## STAGE E — final linear stack + ship
Fit the final non-neg linear stack over the canon across all 3 families (the simple robust combiner the
STRATEGIC note favors over the flexible tree-meta). Save artifacts **NON-DESTRUCTIVELY** — back up any existing
`_winners_snapshot/FINAL_*` before writing. Report final OOF + the chosen weights. Then surface for the user.

## RULES
- Advance only the next INCOMPLETE stage each tick; never repeat a completed stage.
- run_bg / detached subprocess for ALL heavy cells; never synchronous heavy work.
- A recycled tab (Drive unmounted / pc321 gone) CANNOT be remounted headless → flag for morning, keep other 2 going.
- Commit+push every code change here; redeploy to tabs via git fetch/checkout.
