"""The frozen statistical design for the three primary hypotheses.

This module does one job. It records the window stride, the effective sample
size, and the smallest effect the primary test can detect. It reads the design
effect from the exploratory table. It does not recompute the design effect.

Run this module once. Read its output. Write the three numbers it prints into
config.py. Hash config.py. Commit config.py and the hash together. After that
commit, change no number.

See docs/PREM.md and CONTRACT.md section 6.
"""

from __future__ import annotations

import pandas as pd

import config

# The cluster unit we headline. The time block holds the larger correlation, so
# it gives the smaller and more honest effective sample. We report this one.
HEADLINE_UNIT = "time_block"

# The stride we freeze. See the decision note in the commit message.
CHOSEN_STRIDE = 5

# The power we target for the one-sided sign test.
CHOSEN_POWER = 0.80

# The minimum detectable effect at the chosen power and effective sample.
# The value is the share of windows in which MoRF gives the larger shift.
# A share of 0.5 is the null. This number is pre-committed. It is stated before
# any p value is seen. See docs/PREM_AGENT.md section 5, the MDE table.
#   effective n 179, power 0.80 -> 0.592
#   effective n 179, power 0.90 -> 0.608
CHOSEN_MINIMUM_EFFECT = 0.592


def frozen_design(deff: float, n_rows: int) -> dict:
    """Return the frozen design as a dictionary.

    Take the design effect and the row count. Return the six design numbers.
    The effective sample is the row count divided by the design effect. The
    effective sample is always below the row count when the design effect is
    above one.

    Parameters
    ----------
    deff:
        The design effect for the chosen stride and the headline cluster unit.
    n_rows:
        The window count at the chosen stride.

    Returns
    -------
    dict
        The keys are alpha, sided, primary_n_removed, minimum_effect, power,
        and n_effective.
    """
    if deff < 1.0:
        raise ValueError("the design effect must be at least 1.0")
    if n_rows <= 0:
        raise ValueError("the row count must be positive")

    n_effective = n_rows / deff

    return {
        "alpha": config.ALPHA,
        "sided": config.SIDED,
        "primary_n_removed": config.PRIMARY_N_REMOVED,
        "minimum_effect": CHOSEN_MINIMUM_EFFECT,
        "power": CHOSEN_POWER,
        "n_effective": n_effective,
    }


def read_design_effect(stride: int, unit: str) -> tuple[float, int]:
    """Read the design effect and the window count for one stride and unit.

    Read the exploratory overlap table. Select the one row that matches the
    stride and the cluster unit. Return the worst-proxy design effect and the
    window count. Do not recompute either number.
    """
    path = config.TABLE_DIR / "eda_overlap.csv"
    table = pd.read_csv(path)

    row = table[(table["stride"] == stride) & (table["unit"] == unit)]
    if len(row) != 1:
        raise ValueError(
            f"expected one row for stride {stride} and unit {unit}, "
            f"found {len(row)}"
        )

    deff = float(row["deff_worst"].iloc[0])
    n_rows = int(row["windows"].iloc[0])
    return deff, n_rows


def main() -> None:
    """Print the frozen design and the three numbers to write into config.py."""
    deff, n_rows = read_design_effect(CHOSEN_STRIDE, HEADLINE_UNIT)
    design = frozen_design(deff, n_rows)

    print("Frozen design")
    print("-------------")
    print(f"stride          : {CHOSEN_STRIDE}")
    print(f"headline unit   : {HEADLINE_UNIT}")
    print(f"windows         : {n_rows}")
    print(f"design effect   : {deff:.3f}")
    print(f"effective n     : {design['n_effective']:.1f}")
    print(f"alpha           : {design['alpha']}")
    print(f"sided           : {design['sided']}")
    print(f"primary_n_removed : {design['primary_n_removed']}")
    print(f"minimum_effect  : {design['minimum_effect']}")
    print(f"power           : {design['power']}")
    print()
    print("Write these three lines into config.py, then hash and commit:")
    print(f"    WINDOW_STRIDE  = {CHOSEN_STRIDE}")
    print(f"    MINIMUM_EFFECT = {design['minimum_effect']}")
    print(f"    POWER          = {design['power']}")


if __name__ == "__main__":
    main()
