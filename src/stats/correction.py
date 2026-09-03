"""Multiple-comparison corrections. See CONTRACT.md section 5.11.

Two families exist, and they never mix. `holm` runs over the three PRIMARY
hypotheses only: h1, h2, h3. `benjamini_hochberg` runs over the exploratory
family: every supporting or secondary test that is not one of the three.
Passing the wrong family to either function silently answers the wrong
question, so `docs/EDA_FINDINGS.md` and `src/hypotheses/` keep the two
families in two separate dicts and never pass one into the other function.
"""

from __future__ import annotations


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Return the Holm step-down adjusted p-value for every key of `pvals`.

    Over the three PRIMARY tests only: h1, h2, h3. Holm dominates Bonferroni
    at no cost, per CONTRACT.md section 5.10, so this project never runs a
    plain Bonferroni correction.

    Sort the p-values ascending. Multiply the i-th smallest, at 1-indexed
    rank i, by (m - i + 1), where m is the count of p-values. Carry the
    running MAXIMUM of that product down the sorted list, so an adjusted
    value never falls below the one before it. Cap every value at 1.0.
    """
    items = sorted(pvals.items(), key=lambda pair: pair[1])
    m = len(items)

    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items, start=1):
        step = (m - rank + 1) * p
        running_max = max(running_max, step)
        adjusted[key] = min(1.0, running_max)
    return adjusted


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]:
    """Return the Benjamini-Hochberg adjusted p-value for every key of
    `pvals`.

    Over the exploratory family only, never the three primary tests.

    Sort the p-values ascending. Multiply the i-th smallest, at 1-indexed
    rank i, by m over i, where m is the count of p-values. Carry the
    running MINIMUM of that product UP the sorted list, from the largest
    p-value down to the smallest, so an adjusted value never rises above
    the one after it. Cap every value at 1.0.
    """
    items = sorted(pvals.items(), key=lambda pair: pair[1])
    m = len(items)

    adjusted_by_index: list[float] = [0.0] * m
    running_min = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        _, p = items[index]
        step = p * m / rank
        running_min = min(running_min, step)
        adjusted_by_index[index] = min(1.0, running_min)

    return {items[index][0]: adjusted_by_index[index] for index in range(m)}
