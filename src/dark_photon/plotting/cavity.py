# src/dark_photon/plotting/cavity.py

from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .styles import apply_default_style
from .utils import save_fig


def plot_tx_summary(scaleinfo: Dict[str, Any], outdir: Path) -> None:
    """
    Summary plots for cavity transmission.

    Python equivalent of the summary figure in ReadOutCavityTran.m.
    Uses final scaleinfo (after ScaleinfoMergeStage).
    """
    apply_default_style()

    txparams = np.asarray(scaleinfo["txparams"], dtype=float)   # (N,5)
    freq = txparams[:, 1]                                      # f0 (GHz)
    q_loaded = txparams[:, 2]
    p_max = txparams[:, 0]
    slope = txparams[:, 3]
    offset = txparams[:, 4]

    txdrift = np.asarray(scaleinfo["txdriftkHz"], dtype=float)  # drift (kHz)

    # Frequency step (tuning step size)
    # length N-1; we plot against the midpoint index or discard last point
    freq_step = np.diff(freq)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle("Cavity Transmission Summary", fontsize=14)

    # 1,1: cavity frequency vs iteration
    ax = axes[0, 0]
    ax.plot(freq, ".-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f_cav (GHz)")

    # 1,2: frequency step vs iteration
    ax = axes[0, 1]
    ax.plot(freq_step, ".-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Δf_cav (GHz)")

    # 2,1: loaded Q vs frequency
    ax = axes[1, 0]
    ax.plot(freq, q_loaded, ".-")
    ax.set_xlabel("f_cav (GHz)")
    ax.set_ylabel("Q_loaded")

    # 2,2: TX drift between sweeps (kHz) vs frequency
    ax = axes[1, 1]
    ax.plot(freq, txdrift, ".-")
    ax.set_xlabel("f_cav (GHz)")
    ax.set_ylabel("TX drift (kHz)")

    # 3,1: P_max vs frequency
    ax = axes[2, 0]
    ax.plot(freq, p_max, ".-")
    ax.set_xlabel("f_cav (GHz)")
    ax.set_ylabel("P_max (arb.)")

    # 3,2: baseline parameters vs frequency (offset, maybe slope)
    ax = axes[2, 1]
    ax.plot(freq, offset, ".-", label="offset")
    ax.plot(freq, slope, ".-", label="slope")
    ax.set_xlabel("f_cav (GHz)")
    ax.set_ylabel("Baseline params")
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, outdir, "TxSummary")

def plot_rfl_summary(scaleinfo: Dict[str, Any], outdir: Path) -> None:
    """
    Summary plots for cavity reflection.

    Python equivalent of the summary figure in readoutbeta.m.
    Uses final scaleinfo (after ScaleinfoMergeStage).
    """
    apply_default_style()

    freq_beta = np.asarray(scaleinfo["freq_beta"], dtype=float)   # (N,2)
    f_rfl = freq_beta[:, 0]
    beta = freq_beta[:, 1]

    txparams = np.asarray(scaleinfo["txparams"], dtype=float)
    q_loaded = txparams[:, 2]
    q_unloaded = (1.0 + beta) * q_loaded  # (1+β)*Q_loaded

    rfl_base1 = np.asarray(scaleinfo["rfl_base1_db"], dtype=float)
    rfl_base2 = np.asarray(scaleinfo["rfl_base2_db"], dtype=float)

    rfldrift = np.asarray(scaleinfo["rfldriftkHz"], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle("Cavity Reflection Summary", fontsize=14)

    # 1,1: β vs reflection frequency
    ax = axes[0, 0]
    ax.plot(f_rfl, beta, ".-")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("β")

    # 1,2: unloaded Q vs reflection frequency
    ax = axes[0, 1]
    ax.plot(f_rfl, q_unloaded, ".-")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("Q_unloaded")

    # 2,1: reflection baselines vs frequency
    ax = axes[1, 0]
    ax.plot(f_rfl, rfl_base1, ".-", label="rfl1 baseline")
    ax.plot(f_rfl, rfl_base2, ".-", label="rfl2 baseline")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("Baseline (dB)")
    ax.legend()

    # 2,2: RFL drift vs frequency
    ax = axes[1, 1]
    ax.plot(f_rfl, rfldrift, ".-")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("RFL drift (kHz)")

    # 3,1: difference between TX and RFL frequencies (if you want)
    # Requires that txparams[:,1] and freq_beta[:,0] are comparable
    ax = axes[2, 0]
    f_tx = txparams[:, 1]
    df = f_tx - f_rfl
    ax.plot(f_rfl, df, ".-")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("f_tx - f_rfl (GHz)")

    # 3,2: placeholder for something extra (e.g. reflection depth or MSE)
    ax = axes[2, 1]
    # Example: just show β again, or leave for later refinement
    ax.plot(f_rfl, beta, ".-")
    ax.set_xlabel("f_rfl (GHz)")
    ax.set_ylabel("β (duplicate / placeholder)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, outdir, "RflSummary")
