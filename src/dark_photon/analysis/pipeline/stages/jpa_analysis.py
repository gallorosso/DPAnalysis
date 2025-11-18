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
from src.dark_photon.fitting import (
    lorentzian_plus_linear,
    fit_lorentzian,
    iq_to_magnitude,
    optimized_fit_jpa,
)


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
        # r_JPA_prof_cut
        r_JPA_prof_cut = fit_cfg.get("r_JPA_prof_cut", 5.0)

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

            # --- Use optimized_fit_jpa (MATLAB optimizedfitjpaJ equivalent) ---
            try:
                best_params, mse_val, (rg_start, rg_end) = optimized_fit_jpa(
                    i_amp,
                    q_amp,
                    f_jpa,
                    proc_par.__dict__ if hasattr(proc_par, "__dict__") else dict(fit_cfg),
                )
            except Exception as exc:
                warnings.warn(f"JPA JPA-optimized fit failed for {jpa_file.name}: {exc}")
                continue

            amp_gain_fit[i, :] = best_params
            jpa_mse[i] = mse_val

            # Extract basic fit parameters
            P_max, f0, Q, slope, offset = best_params

            # --- Bandwidth and Q² gain (as in MATLAB) ---
            if Q == 0:
                bandwidth_hz = 0.0
            else:
                bandwidth_hz = abs(f0 * 1e9 / Q)
            jpa_bandwidth[i] = bandwidth_hz

            if bandwidth_hz > 0:
                q2gain[i] = (jpa_gbw_prod / bandwidth_hz) ** 2
            else:
                q2gain[i] = 0.0

            # --- 2Q gain in dB: baseline from ends of amplitude profile ---
            amp2 = iq_to_magnitude(i_amp, q_amp)
            amp_db = 10.0 * np.log10(amp2)

            # Same baseline definition as MATLAB:
            # amp_gain_base_dB = mean(pow2db(first 10 pts)) + mean(pow2db(last 10 pts)))/2
            n_edge = min(10, len(amp_db) // 4)
            if n_edge > 0:
                amp_gain_base_dB = 0.5 * (
                    float(np.mean(amp_db[:n_edge])) +
                    float(np.mean(amp_db[-n_edge:]))
                )
            else:
                amp_gain_base_dB = float(np.mean(amp_db))

            # Evaluate fit at resonance frequency (best_params[2])
            peak_fit_linear = lorentzian_plus_linear(
                np.array([f0]), P_max, f0, Q, slope, offset
            )[0]
            peak_fit_db = 10.0 * np.log10(peak_fit_linear)

            gain2Q_amp_dB_fit[i] = float(peak_fit_db - amp_gain_base_dB)

            # --- Reflection baseline from rflparams (used for all "corr" 2Q gains) ---
            rfl_base_db = None
            if rflparams.ndim == 2 and i < rflparams.shape[0]:
                params_rfl = rflparams[i, :]
                # MATLAB: rfl_base_dB = pow2db(a4 * f_rfl + a5)
                # where f_rfl = scaleinfo.rflparams(i,2)
                a4 = params_rfl[3]
                a5 = params_rfl[4]
                f_rfl = params_rfl[1]
                rfl_base_linear = a4 * f_rfl + a5
                if rfl_base_linear > 0.0:
                    rfl_base_db = 10.0 * np.log10(rfl_base_linear)
                else:
                    rfl_base_db = 0.0

            # --- rfl_corr_dB and rfl_corr_idx (for JPA magnitude correction / plotting) ---
            rfl_corr_dB = np.zeros_like(amp_db)
            rfl_corr_idx = None

            if rfl_base_db is not None:
                # Window width in GHz, as in MATLAB: JPA_cut_window_GHz
                cut_window_GHz = float(scaleinfo.get("JPA_cut_window_GHz", 5e-4))

                if f_jpa.size > 1:
                    df = float(np.mean(np.diff(f_jpa)))
                else:
                    df = 0.0

                if df > 0.0:
                    cut_window_idx = int(np.ceil(cut_window_GHz / df))

                    # MATLAB: findex is index closest to init_params1(2);
                    # we use the fitted JPA center f0, which is numerically very close.
                    findex = int(np.argmin(np.abs(f_jpa - f0)))
                    lo_idx = max(findex - cut_window_idx, 0)
                    hi_idx = min(findex + cut_window_idx + 1, f_jpa.size)
                    idx_slice = np.arange(lo_idx, hi_idx)

                    # reflection model: same lorentzian_plus_linear as used for rfl fits
                    params_rfl = rflparams[i, :]
                    a1, f_rfl, q_rfl, a4, a5 = params_rfl
                    x = f_jpa[idx_slice]
                    rfl_peak_linear = (
                        a1 / (4.0 * (q_rfl ** 2) * ((x / f_rfl) - 1.0) ** 2 + 1.0)
                        + a4 * x
                        + a5
                    )
                    rfl_peak_linear = np.clip(rfl_peak_linear, 1e-20, None)
                    rfl_peak_db = 10.0 * np.log10(rfl_peak_linear)

                    rfl_corr_dB[idx_slice] = rfl_peak_db - rfl_base_db
                    rfl_corr_idx = idx_slice

            # --- Fill 2Q "corr" gains for first JPA profile (amp & sqz) ---
            if rfl_base_db is not None:
                # AMP: corrected 2Q gain from JPA fit at its center
                gain2Q_amp_dB_fit_corr[i] = float(peak_fit_db - rfl_base_db)

                # SQZ: only if we managed to fit it
                if not np.isnan(gain2Q_sqz_dB_fit[i]) and gain2Q_sqz_dB_fit[i] != 0.0:
                    # We already computed peak_sqz_db in the sqz fit block
                    try:
                        gain2Q_sqz_dB_fit_corr[i] = float(peak_sqz_db - rfl_base_db)
                    except NameError:
                        # sqz fit failed, leave at default 0
                        pass
                        # --- Second JPA profile (jpaamp2 / jpasqz2) ---

            if has_jpa2:
                try:
                    data2 = scipy.io.loadmat(str(jpa2_file))
                except Exception as exc:
                    warnings.warn(f"Could not load {jpa2_file.name}: {exc}")
                    data2 = None

                if data2 is not None:
                    # --- AMP2 ---
                    try:
                        # MATLAB uses data2.f_GHz_jpaamp2
                        if "f_GHz_jpaamp2" in data2:
                            f_jpa2 = np.asarray(data2["f_GHz_jpaamp2"]).flatten()
                        elif "f_jpaamp2_GHz" in data2:
                            f_jpa2 = np.asarray(data2["f_jpaamp2_GHz"]).flatten()
                        else:
                            raise KeyError("No f_GHz_jpaamp2 / f_jpaamp2_GHz in jpaamp2 file")

                        i_amp2 = np.asarray(data2["I_jpaamp2"]).flatten()
                        q_amp2 = np.asarray(data2["Q_jpaamp2"]).flatten()
                    except KeyError as exc:
                        warnings.warn(f"Missing AMP2 fields in {jpa2_file.name}: {exc}")
                        f_jpa2 = None

                    if f_jpa2 is not None and f_jpa2.size and i_amp2.size and q_amp2.size:
                        power_amp2 = iq_to_magnitude(i_amp2, q_amp2)
                        power_amp2 = np.clip(power_amp2, 1e-20, None)
                        amp2_db = 10.0 * np.log10(power_amp2)

                        # Find peak and simple fit window (mirror the logic used above for AMP1)
                        center_idx2 = int(np.argmax(amp2_db))
                        peak_freq2 = float(f_jpa2[center_idx2])

                        # Use same Q guess as from the first fit as a starting point
                        # (Q from best_params is called 'Q' above)
                        bw_hz2 = (peak_freq2 * 1e9) / max(Q, 1e-9)
                        sigma_hz2 = bw_hz2 / (2.0 * scipy.math.sqrt(2.0 * scipy.math.log(2.0)))
                        sigma_GHz2 = sigma_hz2 / 1e9
                        n_sigma2 = jpa_fit_width_sigma * r_JPA_prof_cut

                        freq_min2 = peak_freq2 - n_sigma2 * sigma_GHz2
                        freq_max2 = peak_freq2 + n_sigma2 * sigma_GHz2
                        lo2 = int(np.searchsorted(f_jpa2, freq_min2, side="left"))
                        hi2 = int(np.searchsorted(f_jpa2, freq_max2, side="right"))
                        lo2 = max(lo2, 0)
                        hi2 = min(hi2, len(f_jpa2))

                        if hi2 - lo2 >= 5:
                            x_fit2 = f_jpa2[lo2:hi2]
                            y_fit2 = amp2_db[lo2:hi2]

                            p0_2 = np.array([
                                float(np.max(y_fit2) - np.min(y_fit2)),  # height
                                float(peak_freq2),                       # center
                                Q,                                       # reuse Q as guess
                                slope,                                  # slope & offset from AMP1
                                offset,
                            ])

                            try:
                                best_params_amp2, _, _ = fit_lorentzian(x_fit2, y_fit2, p0_2)
                                P2, f02, Q2, m2, b2 = best_params_amp2
                                peak2_db = lorentzian_plus_linear(
                                    np.array([f02]), P2, f02, Q2, m2, b2
                                )[0]

                                # reflection-corrected 2Q gain (AMP2)
                                if rfl_base_db is not None:
                                    gain2Q_amp2_dB_fit_corr[i] = float(peak2_db - rfl_base_db)
                            except Exception as exc:
                                warnings.warn(f"Second JPA amp fit failed for {jpa2_file.name}: {exc}")

                    # --- SQZ2 (squeezed second profile) ---
                    if "I_jpasqz2" in data2 and "Q_jpasqz2" in data2:
                        i_sqz2 = np.asarray(data2["I_jpasqz2"]).flatten()
                        q_sqz2 = np.asarray(data2["Q_jpasqz2"]).flatten()
                        if i_sqz2.size and q_sqz2.size and f_jpa2 is not None:
                            power_sqz2 = iq_to_magnitude(i_sqz2, q_sqz2)
                            power_sqz2 = np.clip(power_sqz2, 1e-20, None)
                            sqz2_db = 10.0 * np.log10(power_sqz2)

                            # Use the same frequency window lo2:hi2 as AMP2
                            if hi2 - lo2 >= 5:
                                x_fit2 = f_jpa2[lo2:hi2]
                                y_sqz2_fit = sqz2_db[lo2:hi2]

                                p0_sqz2 = np.array([
                                    float(np.max(y_sqz2_fit) - np.min(y_sqz2_fit)),
                                    float(peak_freq2),
                                    max(Q, 1e-9),
                                    0.0,
                                    float(np.median(y_sqz2_fit)),
                                ])

                                try:
                                    best_params_sqz2, _, _ = fit_lorentzian(
                                        x_fit2, y_sqz2_fit, p0_sqz2
                                    )
                                    P_s2, f0_s2, Q_s2, m_s2, b_s2 = best_params_sqz2
                                    peak_sqz2_db = lorentzian_plus_linear(
                                        np.array([f0_s2]), P_s2, f0_s2, Q_s2, m_s2, b_s2
                                    )[0]

                                    if rfl_base_db is not None:
                                        gain2Q_sqz2_dB_fit_corr[i] = float(peak_sqz2_db - rfl_base_db)
                                except Exception as exc:
                                    warnings.warn(f"Squeezed JPA2 fit failed for {jpa2_file.name}: {exc}")

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