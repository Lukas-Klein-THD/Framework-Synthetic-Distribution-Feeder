"""
Figure style for the IEEE paper submission (two-column IEEE layout).

Rules from the IEEE author guidelines:
  * all figure text is 8 pt Times New Roman;
  * axis labels are words, not symbols or abbreviations;
  * units go in parentheses, e.g. "Line length (km)", never "km" alone or
    "Line length/km".

8 pt is only 8 pt in print if the figure is not rescaled.  Every figure is
therefore drawn at exactly its printed width (one IEEE column, 3.5 in =
88.9 mm) with a fixed layout and saved WITHOUT a tight bounding box, so the PDF
page keeps that width and is used at its natural size.

Fonts are embedded as TrueType (pdf.fonttype 42), as IEEE PDF eXpress requires.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

COLUMN_WIDTH_IN = 3.5        # one IEEE column (88.9 mm)
TEXT_WIDTH_IN = 7.16         # full IEEE page width (181 mm), for two-column figures
FONT_PT = 8
FONT = "Times New Roman"


def apply():
    """Set the paper rcParams.  Call after any seaborn theme, which resets them."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "font.size": FONT_PT,
        "axes.titlesize": FONT_PT,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "legend.title_fontsize": FONT_PT,
        "figure.titlesize": FONT_PT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "grid.linewidth": 0.4,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    })


def subplots(nrows, ncols, height_in, width_in=COLUMN_WIDTH_IN, **kw):
    """Figure at the printed size with a constrained (non-rescaling) layout."""
    return plt.subplots(nrows, ncols, figsize=(width_in, height_in),
                        layout="constrained", **kw)


def save(fig, out_dir, name):
    """Write <name>.pdf (vector, for LaTeX) and <name>.png (600 dpi) at the exact figure size."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=600)
    w, h = fig.get_size_inches()
    print(f"saved {os.path.join(out_dir, name)}.pdf / .png  ({w:.2f} x {h:.2f} in)")


def plain_log_axis(ax, axis="x"):
    """Log axis with plain-number ticks at 1, 2, 5 per decade (no 10^k notation)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())
