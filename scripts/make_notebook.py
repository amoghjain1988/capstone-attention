"""Build notebooks/capstone.ipynb from the tables and the figures on disk.

See docs/FINISH_PLAN.md section 4, stage 10, and CONTRACT.md section 7 item
10. `scripts/run_all.py::stage_notebook` calls `build()`.

THE NOTEBOOK IS A READER, NEVER A PRODUCER. Every number of the notebook comes
from a file that a stage of `scripts/run_all.py` already wrote. The notebook
fits no model, runs no test and writes no table. A reader who wants a new
number changes a stage, then runs the pipeline again, then runs the notebook
again. That rule keeps one number in one place.

WHY THIS SCRIPT BUILDS THE JSON. A notebook is a JSON file with a small fixed
shape, so this script writes that shape directly. `nbformat` is an optional
import. The script uses `nbformat` when the environment holds it, and it falls
back to the plain `json` module and the same nbformat 4 structure when the
environment does not.

EVERY CODE CELL SURVIVES A MISSING FILE. The GPU stages run on another machine,
so a table of a later stage is often absent. Every read passes through
`show_table` or `show_figure`. Both print one line and return None when the
file is absent, so the notebook always runs from the top to the bottom.

NO OUTPUT IS COMMITTED. `build()` writes every code cell with an empty output
list and a null execution count. `check()` runs the cells to prove that they
work, and it never writes the outputs back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# The nbformat numbers of the written file. nbformat 4 minor 5 is the shape
# that JupyterLab and every current reader accept.
NBFORMAT = 4
NBFORMAT_MINOR = 5

# The metadata of the notebook. The kernel name is the plain CPython kernel,
# because no cell needs a GPU.
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


TITLE = """\
# Does the attention of AgentFormer name the neighbour that matters?

DAMO 699 capstone. AgentFormer attention faithfulness.

**The research question.** Does the attention weight that AgentFormer puts on
one neighbour name the neighbour that the forecast of the ego depends on?

Three hypotheses answer that question.

- **H1.** Removal of the most-attended edge moves the forecast more than
  removal of the least-attended edge.
- **H2.** Removal of the most-attended edge moves the forecast more than
  removal of the nearest edge, so attention beats proximity.
- **H3.** The context of a window explains the faithfulness index.

**How to read this notebook.** The notebook reads the tables and the figures
that `scripts/run_all.py` writes. It fits no model and it runs no test. Run
the pipeline first. Then run every cell of this notebook from the top. A cell
whose file is absent prints one note and moves on.
"""


SETUP = '''\
"""Set the path, import pandas, and define the two readers."""

import sys
from pathlib import Path

# Find the repository root, so the notebook runs from any working directory.
ROOT = Path.cwd().resolve()
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

try:
    from IPython.display import Image, display
except ImportError:
    # The check harness of scripts/make_notebook.py runs the cells outside a
    # kernel, so IPython is absent there. print() stands in for display().
    Image = None

    def display(value):
        """Print one value. This is the fallback outside a notebook."""
        print(value)


TABLES = Path(config.TABLE_DIR)
FIGURES = Path(config.FIGURE_DIR)
PROCESSED = Path(config.PROCESSED_DIR)


def show_table(path, rows: int = 25):
    """Show one table. Print a note and return None when the file is absent.

    The reader accepts a csv file and a parquet file. `rows` caps the shown
    rows. The return value is the whole frame, never the capped view.
    """
    path = Path(path)
    if not path.exists():
        print(f"absent: {path.name}. Run the stage that writes it.")
        return None
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"{path.name}: {len(frame):,} rows, {len(frame.columns)} columns")
    display(frame.head(int(rows)))
    return frame


def show_figure(name: str):
    """Show one png of outputs/figures. Print a note when it is absent."""
    path = Path(FIGURES) / str(name)
    if not path.exists():
        print(f"absent: {path.name}. Run the stage that draws it.")
        return None
    if Image is None:
        print(f"figure on disk: {path}")
        return path
    display(Image(filename=str(path)))
    return path


print(f"root      {ROOT}")
print(f"tables    {TABLES}")
print(f"figures   {FIGURES}")
print(f"config    {config.config_hash()}")
'''


SECTIONS: tuple[tuple[str, str], ...] = (
    (
        """\
## 1. The frozen design

`outputs/tables/design.csv` holds two rows. The **frozen** row states the
design that `config.py` and `src/hypotheses/design.py` fix before the first
ablation: the window stride, alpha, the sided rule, the primary removal count,
the minimum effect and the power. The **realised** row measures the frozen
windows table itself.

The two rows must agree closely. A large gap means that the exploratory sweep
and the frozen build no longer describe the same sample.
""",
        '''\
design = show_table(TABLES / "design.csv")
''',
    ),
    (
        """\
## 2. The data

`eda_summary.csv` counts the rows, the pedestrians and the windows of every
scene. `eda_density.csv` reports the neighbourhood of the ego. Both tables come
from the stride 1 windows table, because the exploratory sweep needs every
window start.
""",
        '''\
summary = show_table(TABLES / "eda_summary.csv")
density = show_table(TABLES / "eda_density.csv")
''',
    ),
    (
        """\
## 3. The design effect sweep

Windows overlap, so two rows are not independent. `eda_overlap.csv` reports the
intraclass correlation, the mean cluster size, the design effect and the
effective sample size at every stride. `src/hypotheses/design.py` reads the row
of the frozen stride and turns it into the power statement of section 1.

The figure shows the same sweep. A larger stride costs windows and buys
independence.
""",
        '''\
sweep = show_table(TABLES / "eda_overlap.csv", rows=40)
show_figure("eda_overlap.png")
''',
    ),
    (
        """\
## 4. Accuracy and calibration

`predictions.parquet` holds the displacement errors of every window.
`calibration.csv` holds the coverage and the probability integral transform per
scene.

Note the determinism limit. AgentFormer is deterministic at inference and the
K draws are K fixed DLow modes, not K samples from a posterior. The coverage
therefore describes how the K fixed modes sit around the truth. It is not a
probabilistic calibration claim.
""",
        '''\
predictions = show_table(PROCESSED / "predictions.parquet", rows=5)
if predictions is not None:
    scene = predictions["window_id"].astype(str).str.rsplit("_", n=1).str[0]
    measures = [
        name
        for name in ("ade", "fde", "minade_20", "minfde_20")
        if name in predictions.columns
    ]
    display(predictions.groupby(scene)[measures].describe().T)

calibration = show_table(TABLES / "calibration.csv")
''',
    ),
    (
        """\
## 5. Attention against distance

H2 asks whether attention beats proximity. The question only has an answer on
the windows where the two orders disagree. `arm_overlap.csv` holds the Jaccard
overlap of the top attention edges and the top distance edges, per window.

An overlap near 1 says that attention repeats the proximity order. H2 is then
weak before any test runs.
""",
        '''\
overlap = show_table(TABLES / "arm_overlap.csv", rows=10)
if overlap is not None and "jaccard" in overlap.columns:
    print(overlap["jaccard"].describe().to_string())
''',
    ),
    (
        """\
## 6. The ablation sanity checks

`ablation_sanity.csv` holds the two checks of `scripts/run_ablation.py`. The
zero check masks nothing under fixed draws and must give a shift of exactly 0.
The noise floor masks nothing under free draws and reports the shift that the
sampling alone produces. AgentFormer is deterministic, so that floor is 0 and
any shift above 0 is a real change.

`attrition.csv` holds the arm attrition of `src/hypotheses/data_checks.py`.
Every window must carry every arm.
""",
        '''\
sanity = show_table(TABLES / "ablation_sanity.csv")
attrition = show_table(TABLES / "attrition.csv", rows=40)
''',
    ),
    (
        """\
## 7. The perturbation curves

One curve is one arm. The point at a step is the mean shift over the windows
that hold that step. The band is a pairs cluster bootstrap on the connected
component. The second panel puts `morf` and `weight_matched` on the removed
mass axis, because `weight_matched` matches mass and not count. The third row
holds one panel per scene.

The band of a figure uses fewer resamples than a reported interval. Read a band
as a picture, never as a result.
""",
        '''\
show_figure("hyp_curves.png")
''',
    ),
    (
        """\
## 8. The faithfulness index

The index compares the shift of the most-attended edge with the floor and the
ceiling of every single edge of the same window. An index of 1 says that
attention picked the strongest edge. An index of 0 says that attention is no
better than chance.

`fi_attrition.csv` counts, per scene, the windows at the frozen edge floor and
the windows whose index is null. A window at the floor of
`config.MIN_EGO_EDGES` reaches only +1 or -1, so it carries no middle value.
""",
        '''\
fi_attrition = show_table(TABLES / "fi_attrition.csv")
show_figure("hyp_faithfulness.png")
''',
    ),
    (
        """\
## 9. H1 and H2

`hypotheses_extra.csv` holds the shape report of the paired differences, the
location in metres with its cluster bootstrap interval, and the share of
windows where the attention arm and the distance arm pick the same first edge.
`src/hypotheses/choose_test.py` reads that shape and names the test.

The figure holds the histogram of D for H1 and for H2. A D of exactly 0 is a
structural tie, and the Pratt rule keeps it, so the title counts the ties.
""",
        '''\
extra = show_table(TABLES / "hypotheses_extra.csv")
show_figure("hyp_paired.png")
''',
    ),
    (
        """\
## 10. H3, the context of the faithfulness index

`h3_coefficients.csv` holds both fits of `src/hypotheses/h3_context.py`. The
headline fit drops the windows at the frozen edge floor, because those windows
give an index of exactly +1 or -1 and inject a spike that belongs to the edge
count and not to the context. `h3_vif.csv` holds the collinearity check of the
four covariates.

The figure draws the coefficients of the headline fit with a cluster-robust
interval.
""",
        '''\
coefficients = show_table(TABLES / "h3_coefficients.csv", rows=40)
vif = show_table(TABLES / "h3_vif.csv")
show_figure("hyp_h3_coefficients.png")
''',
    ),
    (
        """\
## 11. The sensitivity table

`validate.csv` holds the descriptive checks of `src/hypotheses/validate.py`:
one scene held out at a time, the removal sweep, another attention module,
another time collapse, the second mask policy, and the noise floor. No p value
of this table enters a family. The table describes the result, it does not test
it again.

The forest plot shows the location per scene for H1 and for H2. Every scene
must point the same way, or the claim belongs to one scene alone.
""",
        '''\
validate = show_table(TABLES / "validate.csv", rows=60)
show_figure("hyp_loso.png")
''',
    ),
    (
        """\
## 12. The verdicts

`verdicts.csv` is the answer. Holm corrects the three primary rows h1, h2 and
h3. Benjamini-Hochberg corrects the supporting family. A row rejects when the
adjusted p value is at or below `config.ALPHA`.

`n_clusters` is the count of connected components, never the count of windows.
The component is the cluster unit of the study, and it is the honest number
beside a p value.
""",
        '''\
verdicts = show_table(TABLES / "verdicts.csv")
if verdicts is not None:
    columns = [
        name
        for name in (
            "hypothesis",
            "role",
            "effect_name",
            "effect",
            "p_raw",
            "p_adj",
            "n_clusters",
            "verdict",
        )
        if name in verdicts.columns
    ]
    display(verdicts.loc[:, columns])
''',
    ),
)


# ---------------------------------------------------------------------------
# The notebook JSON
# ---------------------------------------------------------------------------


def default_path() -> Path:
    """Return the path of the notebook, under the repository root."""
    return Path(config.ROOT) / "notebooks" / "capstone.ipynb"


def _source_lines(text: str) -> list[str]:
    """Return the source of one cell as nbformat wants it.

    nbformat holds a source as a list of lines. Every line except the last one
    keeps its newline.
    """
    lines = str(text).splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return [line for line in lines if line != ""] or [""]


def markdown_cell(text: str) -> dict:
    """Return one markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text),
    }


def code_cell(text: str) -> dict:
    """Return one code cell with no output and no execution count."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def cells() -> list[dict]:
    """Return every cell of the notebook, in order.

    The order is the title, the setup, and then one markdown cell plus one code
    cell per section of SECTIONS.
    """
    out = [markdown_cell(TITLE), code_cell(SETUP)]
    for text, code in SECTIONS:
        out.append(markdown_cell(text))
        out.append(code_cell(code))
    return out


def notebook() -> dict:
    """Return the whole notebook as a plain dict."""
    return {
        "cells": cells(),
        "metadata": dict(NOTEBOOK_METADATA),
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


def build(path: Path | None = None) -> Path:
    """Write the notebook and return the path.

    The function overwrites the file, so it is idempotent. Every code cell goes
    to disk with an empty output list, so no output is ever committed.

    `nbformat` validates the shape when the environment holds it. Without
    `nbformat` the plain `json` module writes the same nbformat 4 structure.
    """
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = notebook()

    try:
        import nbformat
    except ImportError:
        target.write_text(
            json.dumps(document, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    node = nbformat.from_dict(document)
    nbformat.validate(node)
    with target.open("w", encoding="utf-8") as handle:
        nbformat.write(node, handle)
    return target


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def code_sources(path: Path) -> list[str]:
    """Return the source of every code cell of a notebook on disk."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        out.append(source if isinstance(source, str) else "".join(source))
    return out


def run_cells(path: Path) -> int:
    """Run every code cell in order in one namespace. Return the cell count.

    This is the fallback harness. It needs no kernel, so it runs in the same
    process and it sees the same `config` module that the caller sees. A test
    that points `config` at a temporary folder therefore checks the notebook
    against that folder.
    """
    path = Path(path)
    namespace: dict = {"__name__": "__main__", "__file__": str(path)}
    count = 0
    for source in code_sources(path):
        compiled = compile(source, f"{path.name} cell {count + 1}", "exec")
        exec(compiled, namespace)  # noqa: S102
        count += 1
    return count


def _kernel_runner():
    """Return the nbclient executor class, or None when it is absent."""
    try:
        from nbclient import NotebookClient
    except ImportError:
        try:
            from nbconvert.preprocessors import ExecutePreprocessor
        except ImportError:
            return None
        return ExecutePreprocessor
    return NotebookClient


def check(path: Path | None = None, use_kernel: bool = True) -> int:
    """Run the notebook once and return the count of code cells that ran.

    The function prefers a real kernel through `nbclient` or through
    `nbconvert`, because a kernel proves that a reader can run the file. It
    falls back to `run_cells` when neither package is on the machine, and when
    the caller sets `use_kernel` to False.

    The function never writes the outputs back to the file.
    """
    target = Path(path) if path is not None else default_path()
    runner = _kernel_runner() if use_kernel else None
    if runner is None:
        return run_cells(target)

    import nbformat

    node = nbformat.read(target, as_version=NBFORMAT)
    if runner.__name__ == "NotebookClient":
        runner(node, timeout=600, kernel_name="python3").execute(
            cwd=str(Path(config.ROOT))
        )
    else:
        runner(timeout=600, kernel_name="python3").preprocess(
            node, {"metadata": {"path": str(Path(config.ROOT))}}
        )
    return sum(1 for cell in node.cells if cell.cell_type == "code")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def main(check_only: bool = False, no_kernel: bool = False) -> None:
    """Build the notebook, then run it once when the caller asks for a check."""
    path = default_path()
    if not check_only:
        path = build(path)
        print(f"wrote {path}")
    count = check(path, use_kernel=not no_kernel)
    print(f"ran {count} code cells of {path.name} without an error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build notebooks/capstone.ipynb.")
    parser.add_argument(
        "--check",
        dest="check_only",
        action="store_true",
        help="run the notebook that is on disk, do not rebuild it",
    )
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="run the cells in this process, never in a kernel",
    )
    main(**vars(parser.parse_args()))
