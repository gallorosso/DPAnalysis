"""
JPA gain analysis stage for the analysis pipeline.

This is the Python counterpart of the MATLAB function JPAgainAutorun.m.
It reads per-run JPA characterization files (jpaamp*.mat), fits a
Lorentzian + linear baseline model to the gain profiles, and populates
additional fields in ``scaleinfo`` that are used by later calibration
steps.

At this stage we implement the *structure* and basic fitting logic.
Details of the fitting window / reflection correction can be refined
later to match MATLAB line-by-line.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io
import warnings

from ..base import PipelineStage, PipelineContext
from ..results import JPAGainAnalysisResult, ScaleinfoMergeResult
from src.dark_photon.fitting import lorentzian_plus_linear, fit_lorentzian, iq_to_magnitude


class JPAGainAnalysisStage(PipelineStage):
    """
    Stage: Analyze JPA gain data.

    Roughly corresponds to the MATLAB call:
        [scaleinfo, JPA_mse] = JPAgainAutorun(files, scaleinfo, option.plottrue, plotdir, proc_par);

    The stage:
      * loops over all runs (files)
      * loads jpaamp.mat (and optionally jpaamp2.mat)
      * fits Lorentzian + linear baseline to the JPA gain profile
      * computes a simple bandwidth and "2Q gain" proxy
      * writes results into scaleinfo via JPAGainAnalysisResult
    """

    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        print("  Analyzing JPA gain profiles (JPAgainAutorun equivalent)...")

        # ------------------------------------------------------------------
        # 1. Retrieve merged scaleinfo (after TX/RFL and freqs_from_par)
        # ------------------------------------------------------------------
        merge_res: Optional[ScaleinfoMergeResult] = data.get("scaleinfo_merge")
        if merge_res is None or not merge_res.scaleinfo:
            raise ValueError("JPAGainAnalysisStage requires a valid ScaleinfoMergeResult")

        scaleinfo: Dict[str, Any] = merge_res.scaleinfo

        # ------------------------------------------------------------------
        # 2. Retrieve file list from the file enumeration stage
        # ------------------------------------------------------------------
        file_enum = data.get("file_enumeration")
        if file_enum is None or not getattr(file_enum, "files", None):
            raise ValueError("JPAGainAnalysisStage requires file_enumeration.files")

        files: List[str] = file_enum.files
        n_files = len(files)

        # ------------------------------------------------------------------
        # 3. JPA-related processing parameters
        # ------------------------------------------------------------------
        proc_par = context.run_props.processing
        fit_cfg = proc_par.fitting

        # Default values chosen to match / resemble the MATLAB code.
        jpa_gbw_prod = fit_cfg.get("JPA_gbw_prod", 8.15e7)
        jpa_fit_width_sigma = fit_cfg.get("jpa_fit_width_sigma", fit_cfg.get("r_JPA_prof_cut", 5))

        # Derive a JPA cut window from txparams (mirrors JPAgainAutorun)
        txparams = np.asarray(scaleinfo.get("txparams"))
        if txparams.ndim != 2 or txparams.shape[1] < 3:
            warnings.warn("txparams has unexpected shape; JPA_cut_window_GHz may be unreliable")
            cav_bw_ghz = 0.0
        else:
            cav_bw_ghz = float(np.mean(txparams[:, 1]) / np.mean(txparams[:, 2]))
        cut_window_ghz = cav_bw_ghz
        scaleinfo["JPA_cut_window_GHz"] = cut_window_ghz

        # ------------------------------------------------------------------
        # 4. Allocate result arrays
        # ------------------------------------------------------------------
        jpa_mse = np.zeros(n_files)
        jpa_bandwidth = np.zeros(n_files)
        q2gain = np.zeros(n_files)

        gain2Q_amp_dB_fit = np.zeros(n_files)
        gain2Q_sqz_dB_fit = np.zeros(n_files)
        gain2Q_amp_dB_fit_corr = np.zeros(n_files)
        gain2Q_amp2_dB_fit_corr = np.zeros(n_files)
        gain2Q_sqz_dB_fit_corr = np.zeros(n_files)
        gain2Q_sqz2_dB_fit_corr = np.zeros(n_files)

        amp_gain_fit = np.zeros((n_files, 5))
        sqz_gain_fit = np.zeros((n_files, 5))

        processed = 0

        # Reflection parameters are needed for the "corr" variants.
        rflparams = np.asarray(scaleinfo.get("rflparams"))
        if rflparams.ndim != 2 or rflparams.shape[0] < n_files:
            warnings.warn(
                "rflparams missing or has unexpected shape; "
                "reflection-corrected JPA quantities will be approximate or zero."
            )

        # ------------------------------------------------------------------
        # 5. Loop over runs and do a simple Lorentzian fit
        # ------------------------------------------------------------------
        for i, base in enumerate(files):
            base_path = Path(base)

            jpa_file = base_path.with_name(base_path.name + "jpaamp.mat")
            if not jpa_file.exists():
                print(f"    ⚠ No JPA file for {base_path.name} (expected {jpa_file.name}); skipping")
                continue

            jpa2_file = base_path.with_name(base_path.name + "jpaamp2.mat")
            has_jpa2 = jpa2_file.exists()  # reserved for later use

            # --- Load main JPA data ---
            data1 = scipy.io.loadmat(str(jpa_file))

            try:
                f_jpa = np.asarray(data1["f_jpaamp_GHz"]).flatten()
                i_amp = np.asarray(data1["I_jpaamp"]).flatten()
                q_amp = np.asarray(data1["Q_jpaamp"]).flatten()
            except KeyError as exc:
                warnings.warn(f"Missing expected JPA fields in {jpa_file.name}: {exc}")
                continue

            if f_jpa.size == 0 or i_amp.size == 0 or q_amp.size == 0:
                warnings.warn(f"Empty JPA data in {jpa_file.name}")
                continue

            # Convert I/Q to magnitude (power) and then to dB
            power_amp = iq_to_magnitude(i_amp, q_amp)
            # Avoid log10(0) by clipping very small values
            power_amp = np.clip(power_amp, 1e-20, None)
            amp_db = 10.0 * np.log10(power_amp)

            # Optional squeezed trace
            sqz_db = None
            if "I_jpasqz" in data1 and "Q_jpasqz" in data1:
                i_sqz = np.asarray(data1["I_jpasqz"]).flatten()
                q_sqz = np.asarray(data1["Q_jpasqz"]).flatten()
                if i_sqz.size and q_sqz.size:
                    power_sqz = iq_to_magnitude(i_sqz, q_sqz)
                    power_sqz = np.clip(power_sqz, 1e-20, None)
                    sqz_db = 10.0 * np.log10(power_sqz)

            # --- Fit Lorentzian + linear baseline on a simple window around the peak ---
            peak_idx = int(np.argmax(amp_db))
            peak_freq = f_jpa[peak_idx]

            # For now, define a symmetric window in index space controlled by jpa_fit_width_sigma.
            # A more faithful port can later use the MATLAB window definition.
            half_width = int(max(5, jpa_fit_width_sigma * 5))
            lo = max(0, peak_idx - half_width)
            hi = min(len(f_jpa), peak_idx + half_width + 1)

            x_fit = f_jpa[lo:hi]
            y_fit = amp_db[lo:hi]

            if x_fit.size < 5:
                warnings.warn(f"Not enough points in JPA fit window for {jpa_file.name}")
                continue

            # Initial guess: peak height, peak freq, moderate Q, flat baseline
            p0 = np.array([
                float(np.max(y_fit) - np.min(y_fit)),  # P_max
                float(peak_freq),                      # f0
                1000.0,                                # Q
                0.0,                                   # slope
                float(np.median(y_fit)),              # offset
            ])

            try:
                best_params, cov, mse_val = fit_lorentzian(x_fit, y_fit, p0)
            except Exception as exc:
                warnings.warn(f"JPA fit failed for {jpa_file.name}: {exc}")
                continue

            # Store fit and MSE
            amp_gain_fit[i, :] = best_params
            jpa_mse[i] = mse_val

            # --- Simple bandwidth and Q2gain estimate ---
            P_max, f0, Q, slope, offset = best_params
            if Q == 0:
                bandwidth_hz = 0.0
            else:
                bandwidth_hz = abs(f0 * 1e9 / Q)
            jpa_bandwidth[i] = bandwidth_hz

            if bandwidth_hz > 0:
                q2gain[i] = (jpa_gbw_prod / bandwidth_hz) ** 2
            else:
                q2gain[i] = 0.0

            # --- 2Q gain in dB: subtract a baseline from the peak ---
            # Evaluate the fit at resonance
            peak_fit_db = lorentzian_plus_linear(np.array([f0]), P_max, f0, Q, slope, offset)[0]

            # Baseline from the ends of the *measured* JPA profile (like MATLAB)
            edge = min(10, len(amp_db) // 4)
            if edge > 0:
                baseline_amp_db = 0.5 * (np.mean(amp_db[:edge]) + np.mean(amp_db[-edge:]))
            else:
                baseline_amp_db = float(np.mean(amp_db))

            gain2Q_amp_dB_fit[i] = float(peak_fit_db - baseline_amp_db)

            # --- Squeezed fit (if available) ---
            if sqz_db is not None:
                # Fit on the same index window for now
                y_sqz_fit = sqz_db[lo:hi]
                p0_sqz = np.array([
                    float(np.max(y_sqz_fit) - np.min(y_sqz_fit)),
                    float(peak_freq),
                    1000.0,
                    0.0,
                    float(np.median(y_sqz_fit)),
                ])
                try:
                    best_sqz_params, cov_sqz, mse_sqz = fit_lorentzian(x_fit, y_sqz_fit, p0_sqz)
                    sqz_gain_fit[i, :] = best_sqz_params
                    # Evaluate squeezed fit at resonance
                    P_s, f0_s, Q_s, m_s, b_s = best_sqz_params
                    peak_sqz_db = lorentzian_plus_linear(np.array([f0_s]), P_s, f0_s, Q_s, m_s, b_s)[0]
                    # Baseline as for amp profile
                    if edge > 0:
                        baseline_sqz_db = 0.5 * (np.mean(sqz_db[:edge]) + np.mean(sqz_db[-edge:]))
                    else:
                        baseline_sqz_db = float(np.mean(sqz_db))
                    gain2Q_sqz_dB_fit[i] = float(peak_sqz_db - baseline_sqz_db)
                except Exception as exc:
                    warnings.warn(f"Squeezed JPA fit failed for {jpa_file.name}: {exc}")

            # --- Reflection-corrected variants (very simple approximation) ---
            if rflparams.ndim == 2 and i < rflparams.shape[0]:
                # In MATLAB, the reflection baseline is pow2db(a4 * f0 + a5)
                a4 = rflparams[i, 3]
                a5 = rflparams[i, 4]
                rfl_base_linear = a4 * f0 + a5
                if rfl_base_linear <= 0:
                    rfl_base_db = 0.0
                else:
                    rfl_base_db = 10.0 * np.log10(rfl_base_linear)

                gain2Q_amp_dB_fit_corr[i] = float(peak_fit_db - rfl_base_db)

                # For now, treat amp2/sqz2 as zeros; they can be filled when we
                # add explicit fits for jpaamp2/jpasqz2 like MATLAB.
                gain2Q_amp2_dB_fit_corr[i] = 0.0
                gain2Q_sqz_dB_fit_corr[i] = gain2Q_sqz_dB_fit[i]
                gain2Q_sqz2_dB_fit_corr[i] = 0.0

            processed += 1

        print(f"  JPA analysis processed {processed} / {n_files} runs")

        # ------------------------------------------------------------------
        # 6. Write back into scaleinfo and package results
        # ------------------------------------------------------------------
        scaleinfo_updates: Dict[str, Any] = {
            "Q2gain": q2gain,
            "JPAbandwidth": jpa_bandwidth,
            "amp_gain_fit": amp_gain_fit,
            "sqz_gain_fit": sqz_gain_fit,
            "gain2Q_amp_dB_fit": gain2Q_amp_dB_fit,
            "gain2Q_sqz_dB_fit": gain2Q_sqz_dB_fit,
            "gain2Q_amp_dB_fit_corr": gain2Q_amp_dB_fit_corr,
            "gain2Q_amp2_dB_fit_corr": gain2Q_amp2_dB_fit_corr,
            "gain2Q_sqz_dB_fit_corr": gain2Q_sqz_dB_fit_corr,
            "gain2Q_sqz2_dB_fit_corr": gain2Q_sqz2_dB_fit_corr,
            "JPA_mse": jpa_mse,
            "JPA_cut_window_GHz": float(cut_window_ghz),
        }

        # Update the shared scaleinfo dict in-place
        for key, value in scaleinfo_updates.items():
            scaleinfo[key] = value

        merge_res.scaleinfo = scaleinfo

        result = JPAGainAnalysisResult(
            scaleinfo_updates=scaleinfo_updates,
            jpa_mse=jpa_mse,
            jpa_bandwidth=jpa_bandwidth,
            q2gain=q2gain,
            status="success",
        )

        data["jpa_analysis"] = result
        return data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Basic validation for JPA gain analysis.

        Checks that the stage ran, produced scaleinfo updates, and that the
        main arrays have consistent lengths with the number of files.
        """
        result: Optional[JPAGainAnalysisResult] = data.get("jpa_analysis")
        if result is None:
            print("  ✗ No JPA analysis result found")
            return False

        if not result.scaleinfo_updates:
            print("  ✗ No scaleinfo updates from JPA analysis")
            return False

        # Cross-check with file enumeration if available
        n_from_files: Optional[int] = None
        file_enum = data.get("file_enumeration")
        if file_enum is not None and getattr(file_enum, "files", None):
            n_from_files = len(file_enum.files)

        if n_from_files is not None and result.jpa_bandwidth is not None:
            if len(result.jpa_bandwidth) != n_from_files:
                print(
                    "  ✗ Mismatch between number of files and JPA bandwidth entries:\n"
                    f"    - files: {n_from_files}\n"
                    f"    - JPA bandwidth: {len(result.jpa_bandwidth)}"
                )
                return False

        print("  ✓ JPA gain analysis validated")
        return True