"""Behavioral smoke test for the partial-pool _Calibrator.

Extracts the calibrator section from the shipped model.py, stubs the
surrounding module-level dependencies (math, EPS, normalize_condition,
_BC_TO_ID, _predict_uncalibrated, DEFAULT_PROB, LOG), and exercises:

  1. Well-calibrated input: ``b_global`` and all ``per_bc`` values
     should shrink to near zero (the high ridge keeps the calibrator
     from introducing spurious bias).  Held-out NLL must not be worse
     than identity by more than a small margin (~5e-3 / 8000 samples).

  2. Pure noise: with random labels, the per-bc fits should stay
     bounded close to ``b_global`` (no per-bc bucket should explode
     past +/-0.8 nats with N=15 samples and ridge=20).

  3. Global miscalibration: when predictions are uniformly shifted, the
     fitted ``b_global`` should pull in the corrective direction and
     held-out NLL should drop below identity.

  4. Per-bc miscalibration: when only one benchmark is biased, the
     ``per_bc`` slot for that benchmark should move further from
     ``b_global`` than the (well-calibrated) other benchmarks do.
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
        rng = random.Random(hash(("truth", benchmark, condition, subject, item)) & 0xFFFFFFFF)
        return rng.random()

    def _predict_uncalibrated_factory(scenario):
        if scenario == "well_calibrated":
            return _true_probability
        if scenario == "globally_high":
            def f(benchmark, condition, subject, item):
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

    def make_labels(predictor_scenario, n_per_bc, *, label_noise=False, seed=0xDEADBEEF):
        rng = random.Random(seed)
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

    def run(predictor_scenario, n_per_bc, *, label_noise=False, seed=0xDEADBEEF):
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
        labels = make_labels(predictor_scenario, n_per_bc, label_noise=label_noise, seed=seed)
        cal.fit_from_labeled(labels)
        return cal, ns

    def eval_calibrator(cal, ns, *, predictor_scenario, n_eval=2000):
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

    # Test 1: well calibrated -> b_global and per_bc should stay near zero.
    cal, ns = run("well_calibrated", n_per_bc=15)
    print(
        "[TEST 1 well_calibrated] b_global = {:+.4f}, per_bc shifts = {}".format(
            cal.b_global,
            {k: round(v - cal.b_global, 4) for k, v in cal.per_bc.items()},
        )
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="well_calibrated")
    print("           held-out NLL: cal={:.4f}  id={:.4f}".format(cal_nll, id_nll))
    assert abs(cal.b_global) < 0.30, (
        "b_global must stay near zero on well-calibrated data, got " + str(cal.b_global)
    )
    for bc_key, b in cal.per_bc.items():
        assert abs(b - cal.b_global) < 0.40, (
            "per_bc[{}] strayed too far from b_global on well-calibrated data: ".format(bc_key)
            + str(b - cal.b_global)
        )
    assert cal_nll <= id_nll + 5e-3, (
        "calibrated NLL should not be substantially worse than identity"
    )
    print("           PASS\n")

    # Test 2: pure noise -> per_bc fits stay bounded near b_global.
    max_shift_seen = 0.0
    max_global_seen = 0.0
    n_trials = 10
    for trial in range(n_trials):
        ns = {
            "math": math, "EPS": 1e-6, "DEFAULT_PROB": 0.5, "LOG": _LogStub(),
            "normalize_condition": normalize_condition,
            "_BC_TO_ID": bc_table,
            "_predict_uncalibrated": _predict_uncalibrated_factory("well_calibrated"),
        }
        exec(src, ns)
        cal_trial = ns["_Calibrator"]()
        labels = make_labels("well_calibrated", 15, label_noise=True, seed=0xDEADBEEF + trial * 1009)
        cal_trial.fit_from_labeled(labels)
        max_global_seen = max(max_global_seen, abs(cal_trial.b_global))
        for bc_key, b in cal_trial.per_bc.items():
            max_shift_seen = max(max_shift_seen, abs(b - cal_trial.b_global))
    print(
        "[TEST 2 noise x{}] max |b_global| = {:.3f}, max per_bc shift = {:.3f}".format(
            n_trials, max_global_seen, max_shift_seen
        )
    )
    assert max_global_seen < 0.6, (
        "noise should leave b_global near zero (ridge=20 protects it); got "
        + str(max_global_seen)
    )
    assert max_shift_seen < 0.8, (
        "noise should leave per_bc fits near b_global (ridge=20 protects them); got "
        + str(max_shift_seen)
    )
    print("           PASS\n")

    # Test 3: global miscalibration -> b_global captures the corrective bias.
    cal, ns = run("globally_high", n_per_bc=20)
    print(
        "[TEST 3 globally_high] b_global = {:+.4f}, mean per_bc shift = {:+.4f}".format(
            cal.b_global,
            sum(v - cal.b_global for v in cal.per_bc.values()) / max(1, len(cal.per_bc)),
        )
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="globally_high")
    print("           held-out NLL: cal={:.4f}  id={:.4f}  improvement={:.4f}".format(
        cal_nll, id_nll, id_nll - cal_nll
    ))
    assert cal.b_global < -0.20, (
        "globally-high predictions should pull b_global negative (corrective shift); got "
        + str(cal.b_global)
    )
    assert cal_nll < id_nll, "calibrator should beat identity on held-out NLL"
    print("           PASS\n")

    # Test 4: only one bc is biased -> per_bc[alpha] differs from the others.
    cal, ns = run("bc_specific_bias", n_per_bc=20)
    alpha_b = cal.per_bc.get("alpha::none", cal.b_global)
    other_bs = [v for k, v in cal.per_bc.items() if k != "alpha::none"]
    other_mean = sum(other_bs) / max(1, len(other_bs))
    print(
        "[TEST 4 bc_specific_bias] b_global = {:+.4f}, per_bc[alpha] = {:+.4f}, "
        "mean other per_bc = {:+.4f}".format(cal.b_global, alpha_b, other_mean)
    )
    cal_nll, id_nll = eval_calibrator(cal, ns, predictor_scenario="bc_specific_bias")
    print("           held-out NLL: cal={:.4f}  id={:.4f}  improvement={:.4f}".format(
        cal_nll, id_nll, id_nll - cal_nll
    ))
    assert "alpha::none" in cal.per_bc, "per_bc must contain a slot for alpha (we always fit)"
    assert alpha_b < other_mean, (
        "per_bc[alpha] should pull MORE NEGATIVE than the well-calibrated benchmarks; "
        "got alpha_b={} vs other_mean={}".format(alpha_b, other_mean)
    )
    assert cal_nll < id_nll, "should still beat identity overall"
    print("           PASS\n")

    # Test 5: NEW-bc systematic shift -> delta_type pulls in corrective
    # direction; b_global stays small.  Only alpha and beta are in
    # _BC_TO_ID (training); gamma and delta are "new".  We bias only the
    # new-bc rows uniformly negative (sigmoid output too low).
    def _predict_with_new_bc_bias(benchmark, condition, subject, item):
        p_true = _true_probability(benchmark, condition, subject, item)
        z = math.log(max(1e-9, p_true) / max(1e-9, 1.0 - p_true))
        if benchmark in ("gamma", "delta"):
            z -= 1.5  # new-bc predictions too low
        return 1.0 / (1.0 + math.exp(-z))

    train_only = {"alpha::none": 1, "beta::none": 2}  # gamma/delta are new
    ns5 = {
        "math": math, "EPS": 1e-6, "DEFAULT_PROB": 0.5, "LOG": _LogStub(),
        "normalize_condition": normalize_condition,
        "_BC_TO_ID": train_only,
        "_predict_uncalibrated": _predict_with_new_bc_bias,
    }
    exec(src, ns5)
    cal5 = ns5["_Calibrator"]()
    labels5 = []
    rng5 = random.Random(0xFEEDFACE)
    # Heavy oversample of new bcs (mimicking dual-pool acquisition).
    counts = {"alpha": 3, "beta": 3, "gamma": 30, "delta": 30}
    for bench, n in counts.items():
        for i in range(n):
            subject_content = "subj-" + str(i)
            item_content = "item-" + str(i)
            p_pred = _predict_with_new_bc_bias(bench, "none", subject_content, item_content)
            p_truth = _true_probability(bench, "none", subject_content, item_content)
            y = 1 if rng5.random() < p_truth else 0
            labels5.append({
                "benchmark": bench, "condition": "none",
                "subject_content": subject_content, "item_content": item_content,
                "label": y,
            })
    rng5.shuffle(labels5)
    cal5.fit_from_labeled(labels5)
    print(
        "[TEST 5 new-bc shift] b_global = {:+.4f}, delta_type = {:+.4f}".format(
            cal5.b_global, cal5.delta_type
        )
    )
    print(
        "           per_bc shifts: " + str({
            k: round(v - cal5.b_global, 3) for k, v in cal5.per_bc.items()
        })
    )
    assert cal5.delta_type > 0.30, (
        "delta_type must pull POSITIVE (since new-bc predictions are too low); "
        "got " + str(cal5.delta_type)
    )
    assert abs(cal5.b_global) < 0.5, (
        "b_global should stay modest once delta_type absorbs the new-bc shift; got "
        + str(cal5.b_global)
    )
    # At apply time: unseen new-bc gets b_global + delta_type;
    # unseen training-bc gets just b_global.
    p_test = 0.30  # an underconfident-low prediction
    # 'epsilon::none' is unseen and NOT in _BC_TO_ID -> treated as new.
    p_new = cal5.apply(p_test, "epsilon::none")
    p_train = cal5.apply(p_test, "alpha::none" if "alpha::none" not in cal5.per_bc else "zeta::none")
    # Actually alpha::none IS in cal.per_bc (we saw 3 alpha labels), so use a
    # different known-bc that's in _BC_TO_ID but unseen in labels:
    train_only["theta::none"] = 99  # add a training bc we didn't label
    p_train_unseen = cal5.apply(p_test, "theta::none")
    print(
        "           apply(0.30, new) = {:.4f}; apply(0.30, train-unseen) = {:.4f}".format(
            p_new, p_train_unseen
        )
    )
    assert p_new > p_train_unseen, (
        "new-bc apply must shift UP (delta_type correction) more than "
        "training-bc apply; got new={}, train={}".format(p_new, p_train_unseen)
    )
    print("           PASS\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
