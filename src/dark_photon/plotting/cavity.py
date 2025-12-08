from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

from src.dark_photon.analysis.pipeline.results import (
    TransmissionAnalysisResult,
    ReflectionAnalysisResult,
)


def _ensure_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def plot_tx_summary(
    tx_result: TransmissionAnalysisResult,
    scaleinfo: Dict[str, Any],
    plot_dir: Path,
    show: bool = False,
) -> None:
    """
    Reproduce MATLAB's tx_fitting_results.fig / .png

    Inputs:
        tx_result: TransmissionAnalysisResult from TransmissionAnalysisStage
        scaleinfo: Global scaleinfo dict (after TX stage)
                   must contain 'txparams' and 'txdriftkHz'
        plot_dir:  Directory where the figure is saved
        show:      If True, call plt.show(); otherwise close the figure.
    """
    plot_dir = _ensure_path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if tx_result.fitted_parameters is None:
        raise ValueError("tx_result.fitted_parameters is None")

    if "txparams" not in scaleinfo or "txdriftkHz" not in scaleinfo:
        raise ValueError("scaleinfo must contain 'txparams' and 'txdriftkHz' for TX plotting")

    # params1 and params2 are the 5-parameter fits for the two sweeps
    params1 = np.asarray(tx_result.fitted_parameters[:, :5])
    params2 = np.asarray(tx_result.fitted_parameters[:, 5:])
    paramsavg = np.asarray(scaleinfo["txparams"])

    # Q factors (3rd column, 0-based index 2)
    Q1 = params1[:, 2]
    Q2 = params2[:, 2]
    Qavg = paramsavg[:, 2]

    # Center frequency (f0) is 2nd column (index 1)
    f0_avg = paramsavg[:, 1]

    # # Drift in kHz already computed and stored in scaleinfo
    freq_drift_khz = np.asarray(scaleinfo["txdriftkHz"])
    # # Calculate frequency drift from the first measurement
    # freq_drift_ghz = params1[:, 1] - params1[0, 1]
    # freq_drift_khz = freq_drift_ghz * 1000.0

    # Add this debug block:
    # print(f"DEBUG: params1[:, 1] shape: {params1[:, 1].shape}")
    # print(f"DEBUG: freq_drift_khz shape: {freq_drift_khz.shape}")
    # print(f"DEBUG: freq_drift_khz content: {freq_drift_khz}")
    # End debug block

    # TX peak amplitude (1st column)
    tx1_peak = params1[:, 0]
    tx2_peak = params2[:, 0]

    # TX baseline = f0 * slope + offset, MATLAB: params(:,2).*params(:,4) + params(:,5)
    tx1_baseline = params1[:, 1] * params1[:, 3] + params1[:, 4]
    tx2_baseline = params2[:, 1] * params2[:, 3] + params2[:, 4]

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)

    # Subplot (3,2,1): cavity center frequency vs iteration
    ax = axes[0, 0]
    ax.plot(paramsavg[:, 1], ".", markersize=8)
    ax.set_ylabel(r"TM$_{010}$ Frequency (GHz)", fontsize=13)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_title("Cavity center frequency")

    # Subplot (3,2,2): tuning step size (diff of center freq) in kHz
    ax = axes[0, 1]
    delta_f_khz = np.diff(paramsavg[:, 1]) * 1e6
    ax.plot(delta_f_khz, ".", markersize=8)
    ax.set_ylabel(r"Delta TM$_{010}$ Frequency [kHz]", fontsize=13)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_title("Tuning Step Size")

    # Subplot (3,2,3): unloaded Q vs frequency for both sweeps and avg
    ax = axes[1, 0]
    ax.plot(params1[:, 1], Q1, ".", markersize=8, label="1st")
    ax.plot(params2[:, 1], Q2, ".", markersize=8, label="2nd")
    ax.plot(paramsavg[:, 1], Qavg, ".", markersize=8, label="Avg.")
    ax.set_ylabel("Q factor")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("Cavity unloaded quality factor")
    ax.legend()

    # Subplot (3,2,4): drift vs frequency (first sweep frequency & drift)
    ax = axes[1, 1]
    ax.plot(params1[:, 1], freq_drift_khz, ".", markersize=8)
    ax.set_ylabel(r"TM$_{010}$ Drift (kHz)")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("Drift in Cavity Frequency")

    # Subplot (3,2,5): TX peak amplitude vs frequency
    ax = axes[2, 0]
    ax.plot(params1[:, 1], tx1_peak, ".", markersize=8, label="tx1")
    ax.plot(params2[:, 1], tx2_peak, ".", markersize=8, label="tx2")
    ax.set_ylabel("TX Peak")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("TX Peak")
    ax.legend()

    # Subplot (3,2,6): TX baseline vs frequency
    ax = axes[2, 1]
    ax.plot(params1[:, 1], tx1_baseline, ".", markersize=8, label="tx1")
    ax.plot(params2[:, 1], tx2_baseline, ".", markersize=8, label="tx2")
    ax.set_ylabel("TX Baseline")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("TX Baseline")
    ax.legend()

    # Save to disk (MATLAB uses .fig + .png; here we do .png, you can add .pdf if you like)
    out_png = plot_dir / "tx_fitting_results.png"
    fig.savefig(out_png, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_rfl_summary(
    rfl_result: ReflectionAnalysisResult,
    tx_scaleinfo: Dict[str, Any],
    plot_dir: Path,
    show: bool = False,
) -> None:
    """
    Reproduce MATLAB's rfl_fit_results.fig / .png.

    Inputs:
        rfl_result: ReflectionAnalysisResult from ReflectionAnalysisStage
        tx_scaleinfo: global scaleinfo dict (after TX+RFL), must contain
                      'txparams', 'freq_beta', 'rfl_base1_db', 'rfl_base2_db'
        plot_dir: directory where the figure is saved
        show: whether to display the figure interactively
    """
    plot_dir = _ensure_path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Basic checks
    required_keys = ["txparams", "freq_beta", "rfl_base1_db", "rfl_base2_db"]
    for key in required_keys:
        if key not in tx_scaleinfo:
            raise ValueError(f"tx_scaleinfo must contain '{key}' for RFL plotting")

    if rfl_result.rfl_fit_params is None:
        raise ValueError("rfl_result.rfl_fit_params is None (did you extend ReflectionAnalysisResult?)")

    # freq_beta: [f_avg, beta_avg]
    freq_beta = np.asarray(tx_scaleinfo["freq_beta"])
    freq_avg = freq_beta[:, 0]
    beta_avg = freq_beta[:, 1]

    # Full beta matrix from reflection stage: beta1, beta2, beta_avg
    beta_values = np.asarray(rfl_result.coupling_factors)
    beta1 = beta_values[:, 0]
    beta2 = beta_values[:, 1]

    # reflection_frequencies: freq1, freq2, freq_avg
    freq_values = np.asarray(rfl_result.reflection_frequencies)

    # TX params for unloaded Q: Q_loaded is 3rd param (index 2)
    txparams = np.asarray(tx_scaleinfo["txparams"])
    Q_loaded = txparams[:, 2]

    # Unloaded Q = (1 + beta_avg) * Q_loaded
    Q_unloaded = (1.0 + beta_avg) * Q_loaded

    # RFL dip: first parameter (index 0) of each 5-param block in rfl_fit_params
    rfl_fit_params = np.asarray(rfl_result.rfl_fit_params)
    dip1 = rfl_fit_params[:, 0]
    dip2 = rfl_fit_params[:, 5]

    # f_rfl is the AVERAGE! Not what Matlab does.
    # f_tx = txparams[:, 1]       # GHz, from transmission fit
    # f_rfl = freq_values[:, 2]   # GHz, avg reflection freq
    # delta_f_khz = (f_tx - f_rfl) * 1e6  # GHz -> kHz

    # This is what MATLAB does.
    f_tx = txparams[:, 1]       # GHz, from transmission fit
    f_rfl = rfl_fit_params[:, 1]   # first sweep f0, like MATLAB
    delta_f_khz = (f_tx - f_rfl) * 1e6

    # Baselines in dB
    rfl_base1_db = np.asarray(tx_scaleinfo["rfl_base1_db"])
    rfl_base2_db = np.asarray(tx_scaleinfo["rfl_base2_db"])

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)

    # Subplot (3,2,1): coupling factor beta vs frequency
    ax = axes[0, 0]
    ax.plot(freq_avg, beta1, ".r", markersize=6, label="first")
    ax.plot(freq_avg, beta2, ".g", markersize=6, label="sec")
    ax.plot(freq_avg, beta_avg, ".b", markersize=7, label="mean")
    ax.set_ylabel("Beta")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("Coupling factor")
    ax.legend()

    # Subplot (3,2,2): unloaded Q vs frequency
    ax = axes[0, 1]
    ax.plot(freq_avg, Q_unloaded, ".", markersize=7)
    ax.set_ylabel("Unloaded Q")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("Unloaded Q")

    # Subplot (3,2,3): RFL dip amplitudes for both sweeps
    ax = axes[1, 0]
    ax.plot(freq_avg, dip1, ".", markersize=7, label="dip 1")
    ax.plot(freq_avg, dip2, ".", markersize=7, label="dip 2")
    ax.set_ylabel("RFL Dip")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("RFL Dip")
    ax.legend()

    # Subplot (3,2,4): frequency difference f_tx - f_rfl in kHz
    ax = axes[1, 1]
    ax.plot(freq_avg, delta_f_khz, ".", markersize=7)
    ax.set_ylabel(r"$f_{tx} - f_{rfl}$ [kHz]")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("Difference in Frequency (tx - rfl)")

    # Subplot (3,2,5): RFL baseline in dB for both sweeps
    ax = axes[2, 0]
    ax.plot(freq_avg, rfl_base1_db, ".", markersize=6, label="baseline 1")
    ax.plot(freq_avg, rfl_base2_db, ".", markersize=6, label="baseline 2")
    ax.set_ylabel("RFL Baseline Fit [dB]")
    ax.set_xlabel(r"TM$_{010}$ Frequency [GHz]")
    ax.set_title("RFL Baseline for JPA Gain")
    ax.legend()

    # You can optionally use axes[2,1] (3,2,6) for an extra diagnostic plot
    # For now we leave it empty or you can add MSE vs iteration if you like:
    ax = axes[2, 1]
    mse = np.asarray(rfl_result.mse_values) if rfl_result.mse_values is not None else None
    if mse is not None:
        ax.plot(mse[:, 2], ".", markersize=6)
        ax.set_ylabel("Rfl fit MSE (arb. units)")
        ax.set_xlabel("Rfl iteration")
        ax.set_title("Rfl MSE")
    else:
        ax.axis("off")

    out_png = plot_dir / "rfl_fit_results.png"
    fig.savefig(out_png, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
