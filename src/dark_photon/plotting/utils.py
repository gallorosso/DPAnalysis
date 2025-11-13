# src/dark_photon/plotting/utils.py

from pathlib import Path
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outdir: Path, name: str, close: bool = True) -> None:
    """
    Save a figure as <name>.png into outdir.
    """
    ensure_dir(outdir)
    png_path = outdir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    if close:
        plt.close(fig)
