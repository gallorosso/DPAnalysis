# src/dark_photon/plotting/spectrum.py

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_align_norm(scaleinfo: dict, outdir: Path, show: bool = False):
    """
    Plot alignment angle and normalization factors vs run index.
    """
    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates = np.array(scaleinfo.get("spectrum_date", []), dtype=int)
    idx = np.arange(len(dates))

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Alignment angle
    axs[0].plot(idx, scaleinfo.get("align_ang", []), marker='o', linestyle='-')
    axs[0].set_ylabel("Alignment angle [rad]")
    axs[0].set_title("Alignment angle per run")

    # Norm factors
    axs[1].plot(idx, scaleinfo.get("as_norm_fac", []), label="AS norm")
    axs[1].plot(idx, scaleinfo.get("iq_norm_fac", []), label="IQ norm")
    if "as_norm_fac_corr" in scaleinfo:
        axs[1].plot(idx, scaleinfo.get("as_norm_fac_corr", []), label="AS norm corr")
    axs[1].set_ylabel("Normalization factor")
    axs[1].set_xlabel("Run index")
    axs[1].legend()

    fig.tight_layout()
    fname = outdir / "spectrum_align_norm.png"
    fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_spectrum_diagnostics(scaleinfo: dict, outdir: Path, show: bool = False):
    """
    Plot a grid of spectral diagnostics (probe height, IF dip, IF band power, etc).
    """
    # print(f"DEBUG: Received scaleinfo keys: {list(scaleinfo.keys())}")
    
    # Check if we have the required data
    required_keys = ["spectrum_date", "pr_height", "pr_height_sqz", "IFdipheight"]
    missing = [k for k in required_keys if k not in scaleinfo]
    if missing:
        print(f"WARNING: Missing required keys: {missing}")
        return  # Early return to avoid empty plot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(scaleinfo.get("spectrum_date", []))
    idx = np.arange(n)

    fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True)

    # Row 0
    axs[0, 0].plot(idx, scaleinfo.get("pr_height", []), '.-')
    axs[0, 0].set_ylabel("Probe height (norm)")
    axs[0, 1].plot(idx, scaleinfo.get("pr_height_sqz", []), '.-')
    axs[0, 1].set_ylabel("Probe height sqz")
    axs[0, 2].plot(idx, scaleinfo.get("IFdipheight", []), '.-')
    axs[0, 2].set_ylabel("IF dip height")

    # Row 1
    axs[1, 0].plot(idx, scaleinfo.get("sum_power_in_IF", []), '.-')
    axs[1, 0].set_ylabel("IF sum power")
    axs[1, 1].plot(idx, scaleinfo.get("sum_power_in_IF_sq", []), '.-')
    axs[1, 1].set_ylabel("IF sum power (sq)")
    axs[1, 2].plot(idx, scaleinfo.get("mean_of_spec", []), '.-')
    axs[1, 2].set_ylabel("Mean of spec")

    # Row 2
    axs[2, 0].plot(idx, scaleinfo.get("pr_power_stds", []), '.-')
    axs[2, 0].set_ylabel("PR power stds")
    axs[2, 1].plot(idx, scaleinfo.get("sum_power_in_IF_smooth", []), '.-')
    axs[2, 1].set_ylabel("IF sum power (smooth)")
    axs[2, 2].plot(idx, scaleinfo.get("pr_height_smooth", []), '.-')
    axs[2, 2].set_ylabel("PR height (smooth)")

    for ax in axs.flat:
        ax.set_xlabel("Run index")

    fig.tight_layout()
    fname = outdir / "spectrum_diagnostics.png"
    fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_squeezing_calibration(scaleinfo: dict, outdir: Path, show: bool = False):
    """
    Plot squeezing calibration diagnostics, if available.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "avg_sqdB_off" not in scaleinfo:
        # no squeezing data present — skip
        return

    n = len(scaleinfo.get("spectrum_date", []))
    idx = np.arange(n)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(idx, scaleinfo.get("avg_sqdB_off", []), '.-')
    axs[0].set_ylabel("avg_sqdB_off")
    axs[0].set_title("Squeezing calibration: off-spectrum avg")

    axs[1].plot(idx, scaleinfo.get("avg_sqdB_IF", []), '.-')
    axs[1].set_ylabel("avg_sqdB_IF")

    axs[2].plot(idx, scaleinfo.get("peak_sqdB", []), '.-')
    axs[2].set_ylabel("peak_sqdB")
    axs[2].set_xlabel("Run index")

    fig.tight_layout()
    fname = outdir / "squeezing_calibration.png"
    fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
