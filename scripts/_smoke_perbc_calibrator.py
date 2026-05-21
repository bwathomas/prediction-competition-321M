"""Behavioral smoke test for the new hierarchical _Calibrator.

Extracts the calibrator section from the shipped model.py, stubs the
surrounding module-level dependencies (math, EPS, normalize_condition,
_BC_TO_ID, _predict_uncalibrated, DEFAULT_PROB, LOG), and exercises:

  1. Identity case: labels match predictions perfectly -> the held-out NLL
     gate should refuse to fit anything and the calibrator should fall
     through to identity.
  2. Pure noise: random labels, no signal -> gate refuses, identity again.
  3. Global miscalibration: predictions are systematically too high ->
     gate accepts an intercept shift; global calibrator pulls toward 0.5.
  4. Per-bc miscalibration: one benchmark is severely shifted, others
     are well calibrated -> per_bc dict picks up exactly that benchmark.
"""

from __future__ import annotations

import math
import random
import re
import sys
import zipfile
from pathlib import Path

OUT_ZIP = Path(r"C:/Users/benja/Downloads/submission/submission_streamed_encoder_nn_perbc_cal.zip")


def _extract_calibrator_source(model_py: str) -> str:
    cal_re = re.compile(r"^# -{50,}\n# Calibrator [^\n]*\n# -{50,}\n", re.M)
    m = cal_re.search(model_py)
    if not m:
        raise RuntimeError("calibrator banner not found")
    next_banner = re.search(r"\n\n\n# -{50,}\n", model_py[m.end():])
    end = m.end() + next_banner.start() + 1
    return model_py[m.start():end]


class _LogStub:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def main() -> int:
    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        model_py = zf.read("model.py").decode("utf-8")
    src = _extract_calibrator_source(model_py)

    bc_table = {
        "alpha::none": 1,
        "beta::none": 2,
        "gamma::none": 3,
        "delta::none": 4,
    }

    def normalize_condition(c):
        return str(c or "none")

    def _true_probability(benchmark, condition, subject, item):
        """The TRUE Bernoulli rate for the underlying label."""
        rng = random.Random(hash(("truth", benchmark, condition, subject, item)) & 0xFFFFFFFF)
        return rng.random()

    def _predict_uncalibrated_factory(scenario):
        if scenario == "well_calibrated":
            return _true_probability
        if scenario == "globally_high":
            def f(benchmark, condition, subject, item):
                # Predictions are biased to logit-shift +1 relative to truth.
                p_true = _true_probability(benchmark, condition, subject, item)
                z = math.log(max(1e-9, p_true) / max(1e-9, 1.0 - p_true))
                z_shifted = z + 1.5
                return 1.0 / (1.0 + math.exp(-z_shifted))
            return f
        if scenario == "bc_specific_bias":
            def f(benchmark, condition, subject, item):
                p_true = _true_probability(benchmark, condition, subject, item)
                if benchmark == "alpha":
                    z = math.log(max(1e-9, p_true) / max(1e-9, 1.0 - p_true))
                    z_shifted = z + 2.0
                    return 1.0 / (1.0 + math.exp(-z_shifted))
                return p_true
            return f
        raise ValueError(scenario)

    def make_labels(predictor_scenario, n_per_bc, *, label_noise=False):
        """Generate (predictor-emitted p, label) examples where the LABEL is
        drawn from the *true* underlying probability, not the predictor's
        possibly biased one.  This is what distinguishes a calibration
        scenario from a noise-only scenario.
        """
        rng = random.Random(0xDEADBEEF)
        items = []
        for bench in ("alpha", "beta", "gamma", "delta"):
            for i in range(n_per_bc):
                subject_content = "subj-" + str(i)
                item_content = "item-" + str(i)
                p_truth = _true_probability(bench, "none", subject_content, item_content)
                if label_noise:
                    y = 1 if rng.random() < 0.5 else 0
                else:
                    y = 1 if rng.random() < p_truth else 0
                items.append({
                    "benchmark": bench,
                    "condition": "none",
                    "subject_content": subject_content,
                    "item_content": item_content,
                    "label": y,
                })
        rng.shuffle(items)
        return items

    def run(predictor_scenario, n_per_bc, *, label_noise=False):
        ns = {
            "math": math,
            "EPS": 1e-6,
            "DEFAULT_PROB": 0.5,
            "LOG": _LogStub(),
            "normalize_condition": normalize_condition,
            "_BC_TO_ID": bc_table,
            "_predict_uncalibrated": _predict_uncalibrated_factory(predictor_scenario),
        }
        exec(src, ns)
        cal = ns["_Calibrator"]()
        labels = make_labels(predictor_scenario, n_per_bc, label_noise=label_noise)
        cal.fit_from_labeled(labels)
        return cal, ns

    def eval_calibrator(cal, ns, *, predictor_scenario, n_eval=2000):
        """Evaluate held-out NLL of calibrator vs identity on fresh data."""
        f = _predict_uncalibrated_factory(predictor_scenario)
        rng = random.Random(0xBADC0DE)
        cal_nll = 0.0
        id_nll = 0.0
        total = 0
        for bench in ("alpha", "beta", "gamma", "delta"):
            bc_key = bench + "::none"
            for i in range(n_eval // 4):
                subject_content = "subj-eval-" + str(i)
                item_content = "item-eval-" + str(i)
                p_pred = f(bench, "none", subject_content, item_content)
                p_truth = _true_probability(bench, "none", subject_content, item_content)
                y = 1 if rng.random() < p_truth else 0
                p_cal = cal.apply(p_pred, bc_key)
                p_id = min(max(p_pred, 1e-6), 1.0 - 1e-6)
                cal_nll -= y * math.log(p_cal) + (1 - y) * math.log(1 - p_cal)
                id_nll -= y * math.log(p_id) + (1 - y) * math.log(1 - p_id)
                total += 1
        return cal_nll / total, id_nll / total

    # Test 1: well calibrated -> gate should refuse, identity wins.
    cal, ns = run("well_calibrated", n_per_bc=15)
    print(
        "[TEST 1 well_calibrated] global state =",
        cal.state.get("kind"),
        ", per_bc benchmarks =",
        len(cal.per_bc),
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="well_calibrated")
    print("           held-out NLL: cal={:.4f}  id={:.4f}".format(cal_nll, id_nll))
    assert cal.state.get("kind") == "identity", "should refuse to calibrate"
    assert len(cal.per_bc) == 0, "should not pick up any per-bc calibrators"
    assert cal_nll <= id_nll + 1e-6, "calibrated NLL should not be worse than identity"
    print("           PASS\n")

    # Test 2: pure noise across several seeds -> gate should refuse on average.
    n_false_positive_global = 0
    n_false_positive_perbc = 0
    n_trials = 10
    for trial in range(n_trials):
        # vary the label RNG by re-seeding inside make_labels via the
        # outer random call -- emulate by setting a different label seed
        # in the run helper.  Easiest: call run() with label_noise=True
        # and rely on the random.Random in the smoke test, which uses a
        # fixed seed.  To get variation, perturb the bc_table seed.
        ns = {
            "math": math, "EPS": 1e-6, "DEFAULT_PROB": 0.5, "LOG": _LogStub(),
            "normalize_condition": normalize_condition,
            "_BC_TO_ID": bc_table,
            "_predict_uncalibrated": _predict_uncalibrated_factory("well_calibrated"),
        }
        exec(src, ns)
        rng_trial = random.Random(0xDEADBEEF + trial * 1009)
        items = []
        for bench in ("alpha", "beta", "gamma", "delta"):
            for i in range(15):
                items.append({
                    "benchmark": bench,
                    "condition": "none",
                    "subject_content": "subj-" + str(i),
                    "item_content": "item-" + str(i),
                    "label": 1 if rng_trial.random() < 0.5 else 0,
                })
        cal_trial = ns["_Calibrator"]()
        cal_trial.fit_from_labeled(items)
        if cal_trial.state.get("kind") != "identity":
            n_false_positive_global += 1
        n_false_positive_perbc += len(cal_trial.per_bc)
    print(
        "[TEST 2 noise x{}] false global accepts = {}, total per_bc accepts = {}".format(
            n_trials, n_false_positive_global, n_false_positive_perbc
        )
    )
    assert n_false_positive_global == 0, (
        "noise should NEVER trigger a global calibrator across "
        + str(n_trials) + " trials"
    )
    assert n_false_positive_perbc <= 1, (
        "noise should trigger at most 1 per-bc calibrator across "
        + str(n_trials) + " trials (4 bcs each)"
    )
    print("           PASS\n")

    # Test 3: global miscalibration -> intercept fit, accepted by gate.
    cal, ns = run("globally_high", n_per_bc=20)
    print(
        "[TEST 3 globally_high] global state =",
        cal.state.get("kind"),
        ", b =",
        round(cal.state.get("b", 0.0), 3),
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="globally_high")
    print("           held-out NLL: cal={:.4f}  id={:.4f}  improvement={:.4f}".format(
        cal_nll, id_nll, id_nll - cal_nll
    ))
    assert cal.state.get("kind") == "intercept", (
        "should fit a global intercept calibrator on globally biased data"
    )
    assert cal_nll < id_nll, "calibrator should beat identity on held-out NLL"
    print("           PASS\n")

    # Test 4: only one bc is biased -> per_bc dict picks it up.
    cal, ns = run("bc_specific_bias", n_per_bc=20)
    print(
        "[TEST 4 bc_specific_bias] global =",
        cal.state.get("kind"),
        ", per_bc =",
        {k: v.get("kind") for k, v in cal.per_bc.items()},
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="bc_specific_bias")
    print("           held-out NLL: cal={:.4f}  id={:.4f}  improvement={:.4f}".format(
        cal_nll, id_nll, id_nll - cal_nll
    ))
    assert "alpha::none" in cal.per_bc, "per_bc must capture the biased alpha benchmark"
    assert cal_nll < id_nll, "should still beat identity overall"
    print("           PASS\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
