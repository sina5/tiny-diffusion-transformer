"""Shared chart helpers, following the repository plotting rules (16:9, PNG).

Figures are built with `matplotlib.figure.Figure` directly, so no GUI backend is needed.
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from tinydit import constants as C
from tinydit.errors import ResultsError

TITLE_SIZE = 20
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 12
HEADROOM = 1.15
PERCENT = 100.0


def apply_style() -> None:
    """Set the shared rcParams for every chart in the repository."""
    mpl.rcParams.update({
        "axes.titlesize": LABEL_SIZE + 2, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    })


def save_png(fig: Figure, path: Path) -> Path:
    """Write a figure as PNG at the shared DPI and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=C.FIG_DPI, format="png")
    return path


def require(results: list) -> None:
    """Raise ResultsError if there is nothing to plot."""
    if not results:
        raise ResultsError("no results to plot; run the experiment first")
