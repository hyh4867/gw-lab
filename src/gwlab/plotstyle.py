"""Shared matplotlib style: fixed colours per entity, light surface, recessive grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"H1": "#2a78d6", "L1": "#eb6834", "model": "#0b0b0b", "alt": "#1baf7a", "V1": "#1baf7a", "muted": "#8a8983"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#8a8983", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.4, "figure.dpi": 130,
        "legend.frameon": False,
    })
    return plt


def plain_log_ticks(ax, axis, ticks):
    """Log axis with a few plainly written tick labels (no minor labels)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_formatter(NullFormatter())
