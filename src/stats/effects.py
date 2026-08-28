"""Four effect sizes. Prem calls all four.

See CONTRACT.md section 5.11 for the signatures.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def rank_biserial(d: np.ndarray) -> float:
    """Return the matched-pairs rank biserial correlation, the effect size
    that goes with the Wilcoxon signed-rank test.

    Every difference is ranked by its absolute value, following the Pratt
    rule: a zero difference still receives a rank and still occupies a
    position in the ordering, but it contributes to neither the positive nor
    the negative sum. The result is
    (sum of ranks of the positive differences - sum of ranks of the negative
    differences) / (sum of ranks of the nonzero differences).
    """
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return float("nan")

    ranks = scipy_stats.rankdata(np.abs(d))
    positive = ranks[d > 0.0].sum()
    negative = ranks[d < 0.0].sum()
    total = positive + negative
    if total == 0.0:
        return 0.0
    return float((positive - negative) / total)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Return Cliff's delta of two independent samples, in [-1, 1].

    delta = (count(a_i > b_j) - count(a_i < b_j)) / (len(a) * len(b)).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")

    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (a.size * b.size))


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (rho, p) of the Spearman rank correlation of a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rho, p_value = scipy_stats.spearmanr(a, b)
    return float(rho), float(p_value)


def _residualize(values: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Return values with the OLS fit on [1, covar] removed."""
    design = np.column_stack([np.ones(len(values)), covar])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_corr(
    x: np.ndarray, y: np.ndarray, covar: np.ndarray
) -> tuple[float, float]:
    """Return (r, p), the Pearson correlation of x and y with covar held fixed.

    x and y are each regressed on covar by ordinary least squares, and the
    Pearson correlation is taken of the two residual series. Using
    residualisation, rather than the closed-form partial correlation
    formula, means the same code handles one covariate or several.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(covar, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]

    residual_x = _residualize(x, z)
    residual_y = _residualize(y, z)
    r_value, p_value = scipy_stats.pearsonr(residual_x, residual_y)
    return float(r_value), float(p_value)
