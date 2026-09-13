"""Exact empirical single-detector and two-detector Any operating points.

The empirical optimizer uses evaluation labels, as a descriptive ROC statistic.
It must never be presented as a deployed threshold calibrated independently.
"""
import math
import numpy as np


def _scores(x, columns):
    a = np.asarray(x, dtype=np.float64)
    if columns == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[1] != columns or not len(a):
        raise ValueError(f"Expected nonempty [N,{columns}] scores")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite scores are not allowed")
    return a


def _lowest_threshold(negative, budget):
    if budget >= len(negative):
        return -np.inf
    # Scores use >=. Exclude the boundary and all its ties.
    boundary = np.partition(negative, len(negative)-budget-1)[len(negative)-budget-1]
    return np.nextafter(boundary, np.inf)


def operating_point(positive, negative, fpr=0.01):
    pos = _scores(positive, 1)[:, 0]
    neg = _scores(negative, 1)[:, 0]
    if not 0 <= fpr < 1:
        raise ValueError("FPR must lie in [0,1)")
    budget = math.floor(fpr * len(neg) + 1e-12)
    threshold = _lowest_threshold(neg, budget)
    return dict(tpr=float((pos >= threshold).mean()),
                fpr=float((neg >= threshold).mean()), threshold=float(threshold))


def any_operating_point(positive, negative, fpr=0.01):
    """Optimize paired A OR P thresholds under one shared false-positive budget.

    For every A acceptance set, choose the lowest feasible P threshold on the
    remaining negatives. This is optimal because decreasing P's threshold can
    only increase positives. Includes disabling either detector. Ties are exact.
    """
    pos, neg = _scores(positive, 2), _scores(negative, 2)
    if not 0 <= fpr < 1:
        raise ValueError("FPR must lie in [0,1)")
    budget = math.floor(fpr * len(neg) + 1e-12)
    candidates = np.r_[np.inf, np.unique(np.r_[pos[:, 0], neg[:, 0]])[::-1]]
    best = None
    for ta in candidates:
        na = neg[:, 0] >= ta
        used = int(na.sum())
        if used > budget:
            continue
        tp = _lowest_threshold(neg[~na, 1], budget-used)
        accepted_pos = (pos[:, 0] >= ta) | (pos[:, 1] >= tp)
        accepted_neg = na | (neg[:, 1] >= tp)
        key = (int(accepted_pos.sum()), -int(accepted_neg.sum()))
        if best is None or key > best[0]:
            best = (key, dict(tpr=float(accepted_pos.mean()), fpr=float(accepted_neg.mean()),
                              thresholds=[float(ta), float(tp)]))
    return best[1]


def calibrated_any(positive, negative, calibration_negative, fpr=0.01):
    """Independent negative-only calibration, fixed equal Bonferroni allocation."""
    pos, neg, cal = (_scores(x, 2) for x in (positive, negative, calibration_negative))
    thresholds = [_lowest_threshold(cal[:, i], math.floor(fpr/2*len(cal))) for i in range(2)]
    return dict(tpr=float((pos >= thresholds).any(1).mean()),
                fpr=float((neg >= thresholds).any(1).mean()),
                calibration_fpr=float((cal >= thresholds).any(1).mean()),
                thresholds=list(map(float, thresholds)))
