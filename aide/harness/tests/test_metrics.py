import math
import numpy as np
from aide.harness.metrics import log_loss, auc_roc


def test_log_loss_perfect_is_near_zero():
    assert log_loss([1, 0, 1], [1.0, 0.0, 1.0]) < 1e-5


def test_log_loss_half_is_ln2():
    assert abs(log_loss([1, 0], [0.5, 0.5]) - math.log(2)) < 1e-9


def test_auc_perfect_ranking_is_one():
    assert auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_all_ties_is_half():
    assert abs(auc_roc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9


def test_auc_single_class_is_none():
    assert auc_roc([1, 1, 1], [0.2, 0.5, 0.9]) is None


def test_auc_nonbinary_is_none():
    assert auc_roc([0, 1, 2], [0.1, 0.5, 0.9]) is None
