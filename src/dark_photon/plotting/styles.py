# src/dark_photon/plotting/styles.py

import matplotlib.pyplot as plt

def apply_default_style() -> None:
    """
    Apply a consistent Matplotlib style for all plots.
    Call this once at the start of plotting.
    """
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "lines.linewidth": 1.2,
        "savefig.dpi": 150,
    })
