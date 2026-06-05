import numpy as np
from aide.harness.train import (
    diversity_score, promotion_gate, run_two_phase)


def test_diversity_empty_pool_is_one():
    assert diversity_score([1.0, -1.0, 1.0, -1.0], []) == 1.0


def test_diversity_duplicate_member_is_zero():
    r = [1.0, -1.0, 1.0, -1.0]
    assert abs(diversity_score(r, [r]) - 0.0) < 1e-9


def test_diversity_orthogonal_member_is_high():
    r = [1.0, -1.0, 1.0, -1.0]
    ortho = [1.0, 1.0, -1.0, -1.0]  # corr 0 with r
    assert diversity_score(r, [ortho]) > 0.9


def test_diversity_anticorrelated_member_is_above_one():
    r = [1.0, -1.0, 1.0, -1.0]
    anti = [-1.0, 1.0, -1.0, 1.0]  # corr -1 -> 1-(-1)=2
    assert diversity_score(r, [anti]) > 1.5


def test_gate_competitive_promotes():
    assert promotion_gate(0.40, 0.40, 0.01, diversity=0.0, D=0.4) is True


def test_gate_weak_nondiverse_rejects():
    assert promotion_gate(0.50, 0.40, 0.01, diversity=0.0, D=0.4) is False


def test_gate_weak_but_diverse_promotes():
    assert promotion_gate(0.50, 0.40, 0.01, diversity=0.9, D=0.4) is True


def _counting_eval(nll, resid, calls):
    def f(ds):
        calls.append(ds)
        return (nll, resid)
    return f


def test_two_phase_runs_full_for_competitive_candidate():
    calls = []
    r = np.array([1.0, -1.0, 1.0, -1.0])
    res = run_two_phase(_counting_eval(0.40, r, calls), "trial", "full",
                        group_best_nll=0.40, X=0.01, pool_resids=[r], D=0.4)
    assert res.promoted and res.full_nll == 0.40
    assert len(calls) == 2  # trial + full


def test_two_phase_rejects_weak_nondiverse_without_full_run():
    calls = []
    r = np.array([1.0, -1.0, 1.0, -1.0])
    res = run_two_phase(_counting_eval(0.60, r, calls), "trial", "full",
                        group_best_nll=0.40, X=0.01, pool_resids=[r], D=0.4)
    assert res.promoted is False and res.full_nll is None
    assert len(calls) == 1  # full eval skipped


def test_two_phase_runs_full_for_weak_but_diverse_candidate():
    calls = []
    r = np.array([1.0, -1.0, 1.0, -1.0])
    ortho = np.array([1.0, 1.0, -1.0, -1.0])
    res = run_two_phase(_counting_eval(0.60, r, calls), "trial", "full",
                        group_best_nll=0.40, X=0.01, pool_resids=[ortho], D=0.4)
    assert res.promoted and res.full_nll == 0.60
    assert len(calls) == 2
