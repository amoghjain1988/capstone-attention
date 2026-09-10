"""The choice of the paired test, from the shape of the paired differences.

See CONTRACT.md section 5.10 and docs/FINISH_PLAN.md section 3.3 for the
signature.

WHY A PAIRED TEST. Every hypothesis compares two arms on the same window, so
the two shifts share the window, the ego and the scene. The difference
D_i = shift(arm A) - shift(arm B) removes all of that shared structure. The
pair is never in question. Only the shape of D is.

WHY THE SHAPE DECIDES. The Wilcoxon signed-rank test does not test the median.
It tests the pseudomedian, the median of the Walsh averages. The two agree only
when D is symmetric. This module therefore measures the symmetry of D first and
names the location that the chosen test claims. The rule is:

    symmetric D  -> the Wilcoxon signed-rank test on the pseudomedian.
    skewed D     -> the sign test on the median.

D counts as symmetric when the plain bootstrap 95 percent interval of the
sample skewness contains 0, or when the absolute skewness is below
config.SYMMETRY_SKEW_TOL. The tolerance is the second gate, because a large
sample gives a tight interval that excludes 0 at a skewness too small to move
the pseudomedian away from the median.

WHAT THIS MODULE DOES NOT DO. It reports the shape, it does not assume it. It
returns no p value and no interval. The rows of D are not independent: the
windows overlap, a pedestrian repeats, and there are 5 scenes. The p value
therefore comes from the sign flip permutation inside the cluster
(src/stats/permutation.py) and the interval comes from the cluster bootstrap
(src/stats/clustered.py). This module only names the test and the location.

REJECTED ON PURPOSE, per CONTRACT.md section 5.10:

    the paired t test          D has long tails and the mean is not the target.
    the plain bootstrap of D   it treats the rows as independent, and they are
                               not. It is used here for the skewness of the
                               shape report alone, never for a p value or for a
                               reported interval.
    OLS or ANOVA on raw rows   the same fault as the plain bootstrap.
    Bonferroni                 Holm is uniformly better at no cost.
    the chi-square test        the shift is a continuous metre value, not a
                               count.

ZEROS. A zero difference is real data. The shape statistics below run on the
non-zero D, because the symmetry rule is stated on the non-zero D and a mass of
structural zeros hides the skewness of the tail. The location statistics (mean,
median, pseudomedian) run on the whole D, because the test runs on the whole D
under the Pratt rule. shape_stats reports n_zero, so a reader sees both counts.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as scipy_stats

import config

# The smallest number of non-zero differences that gives a shape at all.
# Below 3 rows the skewness is undefined and Shapiro-Wilk refuses the sample.
MIN_NONZERO = 3

# The two-sided confidence level of the skewness bootstrap interval, in percent.
SKEW_CI_PERCENTILES = (2.5, 97.5)

# The largest sample that Shapiro-Wilk accepts. Above this count the p value of
# the test is unreliable, so this module draws a fixed sub-sample instead.
SHAPIRO_MAX_ROWS = 5000

# The two names this module returns.
WILCOXON = "wilcoxon_signed_rank"
SIGN = "sign_test"


def _skew(sample: np.ndarray, axis: int | None = 0) -> np.ndarray:
    """Return the sample skewness, with 0.0 in place of an undefined value.

    A constant sample has no spread, so the skewness is 0 divided by 0. A point
    mass is symmetric about its own point, so this module reads that undefined
    value as a skewness of 0.0. scipy warns about the catastrophic cancellation
    that produces it, and the warning is expected here, so it is suppressed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = scipy_stats.skew(sample, axis=axis)
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)


def _kurtosis(sample: np.ndarray) -> float:
    """Return the excess kurtosis, or NaN when the sample is constant.

    A point mass has no tail, so the excess kurtosis is undefined. scipy warns
    about the same cancellation as in _skew, and the warning is expected here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(scipy_stats.kurtosis(sample))


def pseudomedian(d: np.ndarray) -> float:
    """Return the median of the Walsh averages (d_i + d_j) / 2 over i <= j.

    This is the Hodges-Lehmann location and the quantity that the Wilcoxon
    signed-rank test estimates. The full Walsh table costs n squared values, so
    this function suits one row per window and not one row per edge.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return float("nan")

    rows, columns = np.triu_indices(d.size)
    walsh = (d[rows] + d[columns]) / 2.0
    return float(np.median(walsh))


def skew_bootstrap_ci(nonzero: np.ndarray) -> tuple[float, float]:
    """Return the plain bootstrap percentile interval of the sample skewness.

    Draw config.N_BOOT resamples of the non-zero differences, with replacement
    and at the original size, from np.random.default_rng(config.SEED). Take the
    skewness of every resample and read off the 2.5 and 97.5 percentiles.

    This interval describes the SHAPE of D. It assumes independent rows, so it
    never enters a p value and it never becomes a reported effect interval.
    Those come from the cluster methods. See the module docstring.
    """
    nonzero = np.asarray(nonzero, dtype=np.float64).ravel()
    rng = np.random.default_rng(config.SEED)

    # One index draw of shape (n_boot, n), then one vectorised skewness per row.
    # int32 holds every row position and halves the memory of the draw.
    index = rng.integers(
        0, nonzero.size, size=(config.N_BOOT, nonzero.size), dtype=np.int32
    )
    skews = _skew(nonzero[index], axis=1)

    low, high = np.percentile(skews, SKEW_CI_PERCENTILES)
    return float(low), float(high)


def _shapiro(nonzero: np.ndarray) -> tuple[float, float]:
    """Return (W, p) of the Shapiro-Wilk test on at most SHAPIRO_MAX_ROWS rows.

    A larger sample gets a fixed sub-sample from np.random.default_rng(
    config.SEED), so the result stays the same across two calls and does not
    depend on the order of the rows. A constant sample has no normal fit at
    all, so it returns two NaN values.
    """
    nonzero = np.asarray(nonzero, dtype=np.float64).ravel()
    if np.ptp(nonzero) == 0.0:
        return float("nan"), float("nan")

    if nonzero.size > SHAPIRO_MAX_ROWS:
        rng = np.random.default_rng(config.SEED)
        position = rng.choice(nonzero.size, size=SHAPIRO_MAX_ROWS, replace=False)
        nonzero = nonzero[position]

    result = scipy_stats.shapiro(nonzero)
    return float(result.statistic), float(result.pvalue)


def _why(symmetric: bool, contains_zero: bool, skew_value: float) -> str:
    """Return one present-tense sentence that names the rule that fired."""
    tolerance = config.SYMMETRY_SKEW_TOL
    if contains_zero:
        return (
            "The skewness interval contains 0, so D is symmetric and the "
            "Wilcoxon signed-rank test on the pseudomedian applies."
        )
    if symmetric:
        return (
            f"The skewness interval excludes 0 but the skewness of "
            f"{skew_value:.2f} is below the tolerance of {tolerance}, so the "
            f"Wilcoxon signed-rank test on the pseudomedian applies."
        )
    return (
        f"The skewness interval excludes 0 and the skewness of "
        f"{skew_value:.2f} is above the tolerance of {tolerance}, so the sign "
        f"test on the median applies."
    )


def choose(d: np.ndarray) -> dict:
    """Return the test that the shape of the paired differences demands.

    Parameters
    ----------
    d:
        The paired differences D_i, one value per window. Zeros are kept.

    Returns
    -------
    dict
        The keys are:

        test
            "wilcoxon_signed_rank" when D is symmetric, else "sign_test".
        why
            One sentence in the present tense that names the rule that fired.
        sided
            config.SIDED.
        shape_stats
            n, n_zero, share_positive, mean, median, pseudomedian, skew,
            skew_ci_low, skew_ci_high, kurtosis, shapiro_w, shapiro_p,
            symmetric. The three location values run on the whole D. The shape
            values run on the non-zero D.

    Raises
    ------
    ValueError
        When fewer than MIN_NONZERO non-zero differences are present. Such a
        sample has no shape to read, so no rule can fire.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    nonzero = d[d != 0.0]
    if nonzero.size < MIN_NONZERO:
        raise ValueError(
            f"choose() needs at least {MIN_NONZERO} non-zero differences, "
            f"but got {nonzero.size} of {d.size} rows"
        )

    skew_value = float(_skew(nonzero))
    skew_low, skew_high = skew_bootstrap_ci(nonzero)
    kurtosis_value = _kurtosis(nonzero)
    shapiro_w, shapiro_p = _shapiro(nonzero)

    contains_zero = bool(skew_low <= 0.0 <= skew_high)
    below_tolerance = bool(abs(skew_value) < config.SYMMETRY_SKEW_TOL)
    symmetric = contains_zero or below_tolerance

    shape_stats = {
        "n": int(d.size),
        "n_zero": int(d.size - nonzero.size),
        "share_positive": float((d > 0.0).sum() / nonzero.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "pseudomedian": pseudomedian(d),
        "skew": skew_value,
        "skew_ci_low": skew_low,
        "skew_ci_high": skew_high,
        "kurtosis": kurtosis_value,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "symmetric": symmetric,
    }

    return {
        "test": WILCOXON if symmetric else SIGN,
        "why": _why(symmetric, contains_zero, skew_value),
        "sided": config.SIDED,
        "shape_stats": shape_stats,
    }
