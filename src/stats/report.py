"""Assemble and write the verdicts table. See CONTRACT.md section 3 and
section 5.11.

`build` is the one place that sees every hypothesis result at once, primary
and supporting together. That is why the multiple-comparison correction of
src/stats/correction.py runs here, and nowhere upstream: Holm and
Benjamini-Hochberg both need the WHOLE family of p-values in one call, so no
single hypothesis test (src/hypotheses/h1_morf_vs_lerf.py and its two
siblings) could correct its own p-value alone. If a result already carries a
p_adj or a verdict, build() ignores and overwrites both, since a correction
computed anywhere else could not see the full family this function sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data import schema
from src.stats import correction

# Every field a caller must supply per hypothesis. p_adj and verdict are
# NOT in this set: build() computes both itself. See the module docstring.
REQUIRED_INPUT_COLUMNS = set(schema.VERDICTS) - {"p_adj", "verdict"}


def _adjusted_p_values(frame: pd.DataFrame) -> pd.Series:
    """Return one p_adj per row, Holm for role primary, BH for the rest.

    CONTRACT.md section 5.11 states the rule: Holm runs over the three
    PRIMARY hypotheses only, Benjamini-Hochberg runs over the exploratory
    family. This project never runs a plain Bonferroni correction and never
    mixes the two families in one call.
    """
    adjusted = pd.Series(index=frame.index, dtype="float64")

    primary = frame["role"] == "primary"
    if primary.any():
        pvals = dict(zip(frame.loc[primary, "hypothesis"], frame.loc[primary, "p_raw"]))
        for hypothesis, p_adj in correction.holm(pvals).items():
            adjusted[frame["hypothesis"] == hypothesis] = p_adj

    supporting = ~primary
    if supporting.any():
        pvals = dict(
            zip(frame.loc[supporting, "hypothesis"], frame.loc[supporting, "p_raw"])
        )
        for hypothesis, p_adj in correction.benjamini_hochberg(pvals).items():
            adjusted[frame["hypothesis"] == hypothesis] = p_adj

    return adjusted


def build(results: list[dict]) -> pd.DataFrame:
    """Return the verdicts table, cast, checked and written to disk.

    `results` holds one dict per hypothesis, primary and supporting mixed
    together. Every dict must carry hypothesis, role, test_used, why, sided,
    effect, effect_name, ci_low, ci_high, p_raw, and n_clusters.
    `n_clusters` is the COMPONENT count that src/stats/clustered.py's
    component_clusters returns, never the raw window count: CONTRACT.md
    section 5.11 and config.CLUSTER_UNIT both name the component as the
    real cluster unit, and the two counts differ by a wide margin.

    This function computes the two columns no single hypothesis test can
    compute alone. `p_adj` comes from src/stats/correction.py, Holm over
    role primary and Benjamini-Hochberg over everything else, run ONCE here
    across the whole family. `verdict` is `"reject"` when `p_adj` is at or
    below `config.ALPHA`, `"do not reject"` otherwise: this project pins the
    reject boundary at ALPHA itself, not strictly below it.

    Raises `ValueError` when `results` is empty or a `role` is neither
    `"primary"` nor `"supporting"`. Raises `KeyError` when a result is
    missing a required field. Every other schema rule, including the
    hypothesis key staying unique, comes from `src/data/schema.py` and
    raises from inside `cast`/`validate`, not from this function.
    """
    if not results:
        raise ValueError("report.build needs at least one result")

    frame = pd.DataFrame(results)

    missing = REQUIRED_INPUT_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"report.build: every result needs {sorted(missing)}")

    bad_roles = set(frame["role"]) - {"primary", "supporting"}
    if bad_roles:
        raise ValueError(
            f"report.build: role must be primary or supporting, not {sorted(bad_roles)}"
        )

    frame["p_adj"] = _adjusted_p_values(frame)
    frame["verdict"] = np.where(frame["p_adj"] <= config.ALPHA, "reject", "do not reject")

    frame = frame[list(schema.VERDICTS)]
    out = schema.cast(frame, "verdicts")
    schema.validate(out, "verdicts")
    schema.write(out, "verdicts", config.PROCESSED_DIR)
    return out
