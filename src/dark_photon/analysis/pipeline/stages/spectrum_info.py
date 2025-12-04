# DPAnalysis/src/dark_photon/analysis/pipeline/stages/spectrum_info.py

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import scipy.io
import warnings

from ..base import PipelineStage, PipelineContext
from ..results import SpectrumInfoResult


class SpectrumInfoStage(PipelineStage):
    """
    Stage X: Load and analyze PSA spectra.

    Python implementation of MATLAB's LoadSpectrumInfo.m
    (analysis only, no plotting).

    MATLAB entry point:
        function [scaleinfo] = LoadSpectrumInfo(files, scaleinfo, option, proc_par, cutpar)
    """

    def _get_file_base(self, tx2_file: Path) -> Path:
        """
        Extract file base name from tx2 file path.

        Mirrors the logic used in JPAGainAnalysisStage:

            Converts '/path/to/20220908_0_0_tx2.mat' -> '/path/to/20220908_0_0_'
        """
        filename = tx2_file.name
        if not filename.endswith("tx2.mat"):
            # Be strict here so we notice unexpected patterns early
            raise ValueError(f"Unexpected tx2 filename format: '{filename}'")

        # Remove the 'tx2.mat' suffix and keep the trailing underscore
        # '20220908_0_0_tx2.mat' -> '20220908_0_0_'
        base = filename.replace("tx2.mat", "")
        return tx2_file.parent / base

    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute spectrum info stage.

        Mirrors the main loop in LoadSpectrumInfo.m:

            for i=1:length(files)
                [psafile, spectrum_date, spectrum_it, spectrum_par_num] = stripfilename(files{i}, 'psa');
                scaleinfo.spectrum_date(i)    = spectrum_date;
                scaleinfo.spectrum_it(i)      = spectrum_it;
                scaleinfo.spectrum_par_num(i) = spectrum_par_num;
                ...
            end
        """
        print("  Loading PSA spectrum info...")

        file_enum = data.get("file_enumeration")
        if file_enum is None:
            raise ValueError("file_enumeration missing for SpectrumInfoStage")
        files: List[Path] = file_enum.files
        num_files = len(files)

        print(f"    Total files to process: {num_files}")

        merged_scaleinfo: Dict[str, Any] = data.get("scaleinfo")
        if merged_scaleinfo is None:
            raise ValueError("Merged scaleinfo not found; ScaleinfoMergeStage must run before SpectrumInfoStage")

        proc_par = context.run_props.processing
        cutpar = context.run_props.data_quality

        # For sigma_exp and fitnumber_list
        binavg = float(getattr(proc_par, "binavg", 1.0))
        sg_win_Hz = None
        if hasattr(proc_par, "filters") and isinstance(proc_par.filters, dict):
            sg_win_Hz = proc_par.filters.get("sg_win_Hz", None)

        # Preallocate arrays
        spectrum_date_arr = np.zeros(num_files, dtype=int)
        spectrum_it_arr = np.zeros(num_files, dtype=int)
        spectrum_par_num_arr = np.zeros(num_files, dtype=int)

        probe_scale_arr = np.full(num_files, np.nan)
        as_norm_fac_arr = np.full(num_files, -1.0)
        iq_norm_fac_arr = np.full(num_files, -1.0)
        as_norm_fac_corr_arr = np.full(num_files, -1.0)

        pr_power_arr = np.full(num_files, np.nan)
        avg_sqdB_off_arr = np.full(num_files, np.nan)
        avg_sqdB_IF_arr = np.full(num_files, np.nan)
        peak_sqdB_arr = np.full(num_files, np.nan)
        sum_power_in_IF_sq_sqOFF_arr = np.full(num_files, np.nan)
        sum_power_in_IF_sqOFF_arr = np.full(num_files, np.nan)
        sum_power_in_IF_sqON_arr = np.full(num_files, np.nan)
        spec_bl_sqOFF_arr = np.full(num_files, np.nan)
        pr_height_sqOFF_arr = np.full(num_files, np.nan)

        IFdipheight_arr = np.full(num_files, np.nan)
        spec_bl_arr = np.full(num_files, np.nan)
        pr_height_arr = np.full(num_files, np.nan)
        pr_height_sqz_arr = np.full(num_files, np.nan)
        pr_power_stds_arr = np.full(num_files, np.nan)
        sum_power_in_IF_sq_arr = np.full(num_files, np.nan)
        sum_power_in_IF_arr = np.full(num_files, np.nan)
        mean_of_spec_arr = np.full(num_files, np.nan)

        align_ang_arr = np.full(num_files, -1.0)
        delta_t_arr = np.full(num_files, np.nan)
        fresolution_arr = np.full(num_files, np.nan)
        samplerate_arr = np.full(num_files, np.nan)
        FFTsize_arr = np.full(num_files, np.nan)
        sigma_exp_arr = np.full(num_files, np.nan)
        fitnumber_list_arr = np.zeros(num_files, dtype=int)
        fitnumber2_arr = np.zeros(num_files, dtype=int)

        filename_list: List[str] = [""] * num_files

        # ADD DEBUG TRACKING
        processed_count = 0
        successful_count = 0
        error_files = []
        for i, tx2_file in enumerate(files):
            try:
                processed_count += 1
                
                # --- 5.1 PSA filename + metadata ---
                psa_file, spectrum_date, spectrum_it, spectrum_par_num = self._get_psa_file_and_metadata(tx2_file)
                
                print(f"    Processing file {i+1}/{num_files}: {psa_file.name}")
                
                # Check if PSA file exists
                if not psa_file.exists():
                    print(f"      WARNING: PSA file not found: {psa_file}")
                    error_files.append((tx2_file.name, "PSA file not found"))
                    continue
                    
                # --- 5.2 Load PSA MAT file ---
                psadata = self._load_psa_data(psa_file)
                
                # --- 5.3 Spectrum & freq axis ---
                dat_spec, dat_spec_sq, freq_Hz_spec, GA = self._extract_spectrum(psadata)
                print(f"      Extracted spectrum: {len(dat_spec)} points, GA={GA:.2f}")
                
                # --- 5.4 Probe scale ---
                # Old incorrect call:
                # probe_scale = self._compute_probe_scale(psadata, proc_par, freq_Hz_spec)
                # New correct call:
                probe_scale = self._compute_probe_scale(merged_scaleinfo, i)
                probe_scale_arr[i] = probe_scale

                # --- 5.5 Norm factors ---
                as_norm_fac, iq_norm_fac, as_norm_fac_corr = self._extract_norm_factors(psadata)
                as_norm_fac_arr[i] = as_norm_fac
                iq_norm_fac_arr[i] = iq_norm_fac
                as_norm_fac_corr_arr[i] = as_norm_fac_corr

                # --- 5.6 DAQ parameters ---
                delta_t, fresolution, samplerate, FFTsize = self._compute_daq_parameters(psadata, freq_Hz_spec)
                delta_t_arr[i] = delta_t
                fresolution_arr[i] = fresolution
                samplerate_arr[i] = samplerate
                FFTsize_arr[i] = FFTsize

                # --- 5.7 IF windows / indices ---
                quick_pad, IFdipidx, IFwinidx, lowpass_idx, pr_idx = self._compute_frequency_indices(
                    freq_Hz_spec, fresolution, proc_par
                )

                # --- 5.8 Squeezing blocks ---
                sqz_results = self._analyze_squeezing_blocks(
                    psadata,
                    GA,
                    freq_Hz_spec,
                    IFwinidx,
                    lowpass_idx,
                    pr_idx,
                    probe_scale,
                    proc_par,
                )
                avg_sqdB_off_arr[i] = sqz_results["avg_sqdB_off"]
                avg_sqdB_IF_arr[i] = sqz_results["avg_sqdB_IF"]
                peak_sqdB_arr[i] = sqz_results["peak_sqdB"]
                sum_power_in_IF_sq_sqOFF_arr[i] = sqz_results["sum_power_in_IF_sq_sqOFF"]
                sum_power_in_IF_sqOFF_arr[i] = sqz_results["sum_power_in_IF_sqOFF"]
                sum_power_in_IF_sqON_arr[i] = sqz_results["sum_power_in_IF_sqON"]
                spec_bl_sqOFF_arr[i] = sqz_results["spec_bl_sqOFF"]
                pr_height_sqOFF_arr[i] = sqz_results["pr_height_sqOFF"]

                # --- 5.9 Spectral metrics ---
                spectral_metrics = self._compute_spectral_metrics(
                    dat_spec,
                    dat_spec_sq,
                    IFdipidx,
                    IFwinidx,
                    pr_idx,
                    quick_pad,
                    probe_scale,
                    proc_par,
                    delta_t,
                    fresolution,
                    GA,
                    psadata,
                )
                IFdipheight_arr[i] = spectral_metrics["dip_height"]
                spec_bl_arr[i] = spectral_metrics["spec_bl"]
                pr_height_arr[i] = spectral_metrics["pr_height_norm"]
                pr_height_sqz_arr[i] = spectral_metrics["pr_height_sqz_norm"]
                sum_power_in_IF_sq_arr[i] = spectral_metrics["sum_power_in_IF_sq"]
                sum_power_in_IF_arr[i] = spectral_metrics["sum_power_in_IF"]
                mean_of_spec_arr[i] = spectral_metrics["mean_of_spec"]
                pr_power_arr[i] = spectral_metrics["pr_power"]
                pr_power_stds_arr[i] = spectral_metrics["pr_power_stds"]

                # --- 5.10 Alignment & bookkeeping ---
                align_ang_arr[i] = self._extract_alignment_angle(psadata)
                filename_list[i] = str(psa_file)

                # sigma_exp(i) = 1/sqrt(delta_t * fresolution * binavg)
                if delta_t > 0 and fresolution > 0 and binavg > 0:
                    sigma_exp_arr[i] = 1.0 / np.sqrt(delta_t * fresolution * binavg)

                # fitnumber_list(i) = ceil(proc_par.sg_win_Hz/fresolution) + 1
                if sg_win_Hz is not None and fresolution > 0:
                    fitnum = int(np.ceil(float(sg_win_Hz) / fresolution) + 1)
                    fitnumber_list_arr[i] = fitnum
                    fitnumber2_arr[i] = fitnum

                successful_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"      Progress: {i+1}/{num_files} files processed ({successful_count} successful)")
                    
            except Exception as e:
                error_msg = f"Error processing PSA spectrum for {tx2_file}: {e}"
                print(f"      ERROR: {error_msg}")
                warnings.warn(error_msg)
                error_files.append((tx2_file.name, str(e)))
                continue

        # ADD DEBUG SUMMARY
        print(f"\n    SpectrumInfoStage Summary:")
        print(f"      Total files: {num_files}")
        print(f"      Attempted to process: {processed_count}")
        print(f"      Successfully processed: {successful_count}")
        print(f"      Failed files: {len(error_files)}")
        
        if error_files:
            print(f"      First 5 errors:")
            for i, (filename, error) in enumerate(error_files[:5]):
                print(f"        {i+1}. {filename}: {error}")
        
        # Check if we have any valid data
        valid_indices = ~np.isnan(probe_scale_arr)
        valid_count = np.sum(valid_indices)
        print(f"      Valid probe_scale values: {valid_count}/{num_files}")
        
        if valid_count == 0:
            print("      WARNING: No valid spectrum data was loaded!")

        # 6) Post-loop: global fitnumber = first non-zero fitnumber_list
        scaleinfo_updates: Dict[str, Any] = {}

        # Build a scaleinfo-like dict for smoothvar (needs Cavity_freq and spectrum_date)
        scaleinfo_for_smooth = dict(merged_scaleinfo)
        scaleinfo_for_smooth["spectrum_date"] = spectrum_date_arr
        # Cavity_freq already present from earlier stages, but ensure it's an array
        if "Cavity_freq" in scaleinfo_for_smooth:
            scaleinfo_for_smooth["Cavity_freq"] = np.asarray(scaleinfo_for_smooth["Cavity_freq"], dtype=float)
        else:
            scaleinfo_for_smooth["Cavity_freq"] = np.zeros(num_files, dtype=float)

        # Smoothing for DQ cuts (MATLAB smoothvar calls)
        sum_power_in_IF_smooth, sum_power_in_IF_smoothVar = smoothvar(
            sum_power_in_IF_arr, cutpar, scaleinfo_for_smooth
        )
        pr_height_smooth, pr_height_smoothVar = smoothvar(
            pr_height_arr, cutpar, scaleinfo_for_smooth
        )
        pr_power_stds_smooth, pr_power_stds_smoothVar = smoothvar(
            pr_power_stds_arr, cutpar, scaleinfo_for_smooth
        )

        # Fill scaleinfo_updates to mirror MATLAB fields
        scaleinfo_updates.update(
            {
                "spectrum_date": spectrum_date_arr.tolist(),
                "spectrum_it": spectrum_it_arr.tolist(),
                "spectrum_par_num": spectrum_par_num_arr.tolist(),
                "probe_scale": probe_scale_arr.tolist(),
                "as_norm_fac": as_norm_fac_arr.tolist(),
                "iq_norm_fac": iq_norm_fac_arr.tolist(),
                "as_norm_fac_corr": as_norm_fac_corr_arr.tolist(),
                "pr_power": pr_power_arr.tolist(),
                "avg_sqdB_off": avg_sqdB_off_arr.tolist(),
                "avg_sqdB_IF": avg_sqdB_IF_arr.tolist(),
                "peak_sqdB": peak_sqdB_arr.tolist(),
                "sum_power_in_IF_sq_sqOFF": sum_power_in_IF_sq_sqOFF_arr.tolist(),
                "sum_power_in_IF_sqOFF": sum_power_in_IF_sqOFF_arr.tolist(),
                "sum_power_in_IF_sqON": sum_power_in_IF_sqON_arr.tolist(),
                "spec_bl_sqOFF": spec_bl_sqOFF_arr.tolist(),
                "pr_height_sqOFF": pr_height_sqOFF_arr.tolist(),
                "IFdipheight": IFdipheight_arr.tolist(),
                "spec_bl": spec_bl_arr.tolist(),
                "pr_height": pr_height_arr.tolist(),
                "pr_height_sqz": pr_height_sqz_arr.tolist(),
                "pr_power_stds": pr_power_stds_arr.tolist(),
                "sum_power_in_IF_sq": sum_power_in_IF_sq_arr.tolist(),
                "sum_power_in_IF": sum_power_in_IF_arr.tolist(),
                "mean_of_spec": mean_of_spec_arr.tolist(),
                "align_ang": align_ang_arr.tolist(),
                "delta_t": delta_t_arr.tolist(),
                "fresolution": fresolution_arr.tolist(),
                "samplerate": samplerate_arr.tolist(),
                "FFTsize": FFTsize_arr.tolist(),
                "sigma_exp": sigma_exp_arr.tolist(),
                "fitnumber_list": fitnumber_list_arr.tolist(),
                "fitnumber2": fitnumber2_arr.tolist(),
                "sum_power_in_IF_smooth": sum_power_in_IF_smooth.tolist(),
                "sum_power_in_IF_smoothVar": sum_power_in_IF_smoothVar.tolist(),
                "pr_height_smooth": pr_height_smooth.tolist(),
                "pr_height_smoothVar": pr_height_smoothVar.tolist(),
                "pr_power_stds_smooth": pr_power_stds_smooth.tolist(),
                "pr_power_stds_smoothVar": pr_power_stds_smoothVar.tolist(),
                "filename": filename_list,
            }
        )

        # Global fitnumber: first non-zero element
        nonzero_mask = fitnumber_list_arr > 0
        if np.any(nonzero_mask):
            first_idx = int(np.argmax(nonzero_mask))
            scaleinfo_updates["fitnumber"] = int(fitnumber_list_arr[first_idx])

        result = SpectrumInfoResult(
            scaleinfo_updates=scaleinfo_updates,
            status="success",
        )
        data["spectrum_info"] = result
        
        return data


    def _process_single_spectrum(
        self,
        idx: int,
        tx2_file: Path,
        scaleinfo: Dict[str, Any],
        proc_par: Any,
    ) -> Dict[str, Any]:
        """
        Process one spectrum file: load PSA, extract spectrum, compute probe_scale,
        norm factors, and DAQ parameters.

        Corresponds to the first half of LoadSpectrumInfo.m, up to DAQ parameters.
        """
        # 1) Get the PSA filename and date / it / parnum (point 1 you already implemented)
        psafile, spectrum_date, spectrum_it, spectrum_par_num = self._get_psa_filename_and_info(
            tx2_file
        )

        scaleinfo["spectrum_date"][idx] = spectrum_date
        scaleinfo["spectrum_it"][idx] = spectrum_it
        scaleinfo["spectrum_par_num"][idx] = spectrum_par_num

        # 2) Load PSA data
        psadata = self._load_psa_data(psafile)

        # 3) Extract spectrum + GA
        dat_spec, dat_spec_sq, freq_Hz_spec, GA = self._extract_spectrum(psadata)

        # 4) Compute probe_scale and write into scaleinfo
        probe_scale = self._compute_probe_scale(scaleinfo, idx)

        # 5) Normalization factors (as_norm_fac, iq_norm_fac, as_norm_fac_corr)
        self._extract_norm_factors(psadata, scaleinfo, idx)

        # 6) DAQ parameters (delta_t, fresolution, samplerate, FFTsize)
        meanavgps = psadata.get("meanavgps", None)
        if isinstance(meanavgps, np.ndarray):
            meanavgps = meanavgps.item()
        delta_t, fresolution, samplerate, FFTsize = self._compute_daq_parameters(
            psadata, freq_Hz_spec, meanavgps
        )

        # For now we just *return* these; later we will continue with IFdip,
        # IF window, and all the scaleinfo.* fields that depend on them.
        return {
            "dat_spec": dat_spec,
            "dat_spec_sq": dat_spec_sq,
            "freq_Hz_spec": freq_Hz_spec,
            "GA": GA,
            "probe_scale": probe_scale,
            "delta_t": delta_t,
            "fresolution": fresolution,
            "samplerate": samplerate,
            "FFTsize": FFTsize,
            "psafile": psafile,
        }

    def _get_psa_file_and_metadata(self, tx2_file: Path) -> Tuple[Path, int, int, int]:
        """
        Build PSA filename and parse metadata from the base filename.

        MATLAB:
            [psafile, spectrum_date, spectrum_it, spectrum_par_num] = stripfilename(files{i}, 'psa');
        """
        file_base = self._get_file_base(tx2_file)
    
        # Debug output
        print(f"      DEBUG: TX2 file: {tx2_file.name}")
        print(f"      DEBUG: File base: {file_base}")
        
        # Derive PSA filename by appending 'psa.mat'
        psa_file = file_base.with_name(file_base.name + "psa.mat")
        
        print(f"      DEBUG: PSA file path: {psa_file}")
        print(f"      DEBUG: PSA file exists: {psa_file.exists()}")
        
        if not psa_file.exists():
            raise FileNotFoundError(f"PSA file not found: {psa_file}")

        # 3) Parse date, iteration, par number from the base name
        #    Expected format: 'YYYYMMDD_N_M_' (trailing underscore)
        stem = file_base.name.rstrip("_")  # '20220908_0_0_ ' -> '20220908_0_0'
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Cannot parse date/iteration/parnum from PSA base name '{file_base.name}' "
                f"(expected format 'YYYYMMDD_N_M_')"
            )

        try:
            # spectrum_date    -> YYYYMMDD
            # spectrum_par_num -> N
            # spectrum_it      -> M
            spectrum_date = int(parts[0])
            spectrum_par_num = int(parts[1])
            spectrum_it = int(parts[2])
        except ValueError as e:
            raise ValueError(
                f"Could not convert filename components to integers for '{file_base.name}': {e}"
            )

        return psa_file, spectrum_date, spectrum_it, spectrum_par_num

    def _load_psa_data(self, psa_file: Path) -> Dict[str, Any]:
        """
        Load PSA data from .mat file.

        MATLAB:
            psadata = importdata(psafile);
        """
        if not psa_file.exists():
            raise FileNotFoundError(f"PSA data file not found: {psa_file}")

        # Use the same pattern as other stages (par files, JPA files, etc.)
        # No special options here yet; we'll adapt if the actual structure demands it.
        psadata = scipy.io.loadmat(str(psa_file))

        return psadata

    # --- MATLAB:
    # GA           = psadata.gain_amp_pow;
    # dat_spec     = psadata.meanavgps.singlesided_powerspecavg/GA;
    # dat_spec_sq  = psadata.meanavgps.singlesided_powerspecavg_sq;
    # freq_Hz_spec = psadata.meanavgps.singlesided_freqaxis;
    # ---
    def _extract_spectrum(
        self, psadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extract GA, dat_spec, dat_spec_sq, freq_Hz_spec from psadata.

        MATLAB:
            GA           = psadata.gain_amp_pow;
            dat_spec     = psadata.meanavgps.singlesided_powerspecavg/GA;
            dat_spec_sq  = psadata.meanavgps.singlesided_powerspecavg_sq;
            freq_Hz_spec = psadata.meanavgps.singlesided_freqaxis;
        """
        # Gain amplitude power
        # Gain amplitude power
        try:
            GA = float(np.squeeze(psadata["gain_amp_pow"]))
            print(f"      DEBUG: gain_amp_pow found: GA = {GA}")
        except Exception as e:
            print(f"      DEBUG: Could not read 'gain_amp_pow': {e}")
            print(f"      DEBUG: psadata keys: {list(psadata.keys())}")
            raise

        # meanavgps is a nested MATLAB struct
        meanavgps = psadata.get("meanavgps", None)
        if meanavgps is None:
            raise KeyError("PSA data has no 'meanavgps' field")

        # Access the structured array correctly
        # meanavgps is (1, 1) array containing a struct
        # We need to access the struct at [0, 0] then get the field
        # meanavgps_struct = meanavgps[0, 0]
        
        # Now access the fields within the struct
        if 'singlesided_powerspecavg' not in meanavgps.dtype.names:
            raise KeyError(f"'meanavgps' has no field 'singlesided_powerspecavg'. Available: {meanavgps.dtype.names}")
        
        # Get the field - it's also an array that might need [0, 0]
        singlesided_powerspecavg = meanavgps['singlesided_powerspecavg'][0, 0]
        singlesided_powerspecavg_sq = meanavgps['singlesided_powerspecavg_sq'][0, 0]
        singlesided_freqaxis = meanavgps['singlesided_freqaxis'][0, 0]

        print(f"      singlesided_powerspecavg shape: {singlesided_powerspecavg.shape}")
        print(f"      singlesided_freqaxis shape: {singlesided_freqaxis.shape}")

        dat_spec = np.squeeze(singlesided_powerspecavg) / GA
        dat_spec_sq = np.squeeze(singlesided_powerspecavg_sq)
        freq_Hz_spec = np.squeeze(singlesided_freqaxis)

        # Return in (dat_spec, dat_spec_sq, freq_Hz_spec, GA) order
        return dat_spec, dat_spec_sq, freq_Hz_spec, GA


    # --- MATLAB:
    # QL   = scaleinfo.Cavity_Q(i);
    # beta = scaleinfo.coupling_factor(i);
    # probe_scale = QL*(beta/(1+beta));
    # scaleinfo.probe_scale(i) = probe_scale;
    # ---
    def _compute_probe_scale(
        self,
        scaleinfo: Dict[str, Any],
        idx: int,
    ) -> float:
        """
        Compute probe_scale using cavity Q and beta from scaleinfo.
        
        MATLAB: probe_scale = QL*(beta/(1+beta))
        """
        try:
            # Get from scaleinfo if available
            if "Cavity_Q" in scaleinfo and "coupling_factor" in scaleinfo:
                QL = float(scaleinfo["Cavity_Q"][idx])
                beta = float(scaleinfo["coupling_factor"][idx])
                probe_scale = QL * (beta / (1.0 + beta))
                print(f"      probe_scale: QL={QL}, beta={beta} -> {probe_scale:.6f}")
                return probe_scale
            else:
                print(f"      WARNING: Cavity_Q or coupling_factor not in scaleinfo at index {idx}")
                print(f"      scaleinfo keys: {list(scaleinfo.keys())}")
                # Check if these are in the data at all
                if "Cavity_Q" not in scaleinfo:
                    print(f"      Cavity_Q missing entirely")
                if "coupling_factor" not in scaleinfo:
                    print(f"      coupling_factor missing entirely")
                return 1.0  # Default fallback
        except Exception as e:
            print(f"      Could not calculate probe_scale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0  # Default fallback

    # --- MATLAB:
    # (as_norm_fac, iq_norm_fac, as_norm_fac_corr with try/catch)
    # ---
    def _extract_norm_factors(
        self,
        psadata: Dict[str, Any],
        scaleinfo: Dict[str, Any],
        idx: int,
    ) -> None:
        """
        Extract AS / IQ normalization factors and store in scaleinfo.
        """
        meanavgps = psadata.get("meanavgps", None)
        
        def _safe_get(field_name: str, default: float = -1.0) -> float:
            try:
                if meanavgps is not None and field_name in meanavgps.dtype.names:
                    # Access the field correctly
                    field_data = meanavgps[field_name][0, 0]
                    return float(np.squeeze(field_data))
                return default
            except Exception:
                return default

        as_norm = _safe_get("asNormFac", -1.0)
        iq_norm = _safe_get("iqNormFac", -1.0)
        as_norm_corr = _safe_get("asNormFac_corrr", -1.0)  # Note: MATLAB has typo 'corrr'

        print(f"      Norm factors: as={as_norm}, iq={iq_norm}, as_corr={as_norm_corr}")

        # Make sure lists exist and have correct length
        n = self._get_parameter_count(scaleinfo)
        for key in ("as_norm_fac", "iq_norm_fac", "as_norm_fac_corr"):
            if key not in scaleinfo:
                scaleinfo[key] = [-1.0] * n

        scaleinfo["as_norm_fac"][idx] = as_norm
        scaleinfo["iq_norm_fac"][idx] = iq_norm
        scaleinfo["as_norm_fac_corr"][idx] = as_norm_corr

    # --- MATLAB:
    # try
    #     delta_t     = psadata.tot_run_time;
    # catch
    #     delta_t    = psadata.acqInfo.Depth*psadata.acqInfo.SegmentCount*psadata.nAcq/psadata.acqInfo.SampleRate;
    # end
    # fresolution = abs(freq_Hz_spec(1) - freq_Hz_spec(2));
    # samplerate  = psadata.acqInfo.SampleRate;
    # FFTsize     = (length(psadata.meanavgps.singlesided_freqaxis) - 1)*2;
    # ---
    def _compute_daq_parameters(
        self,
        psadata: Dict[str, Any],
        freq_Hz_spec: np.ndarray,
        meanavgps: Optional[Any] = None,
    ) -> Tuple[float, float, float, int]:
        """
        Compute delta_t, fresolution, samplerate, FFTsize from PSA data.
        """
        # acqInfo is a structured array
        if "acqInfo" not in psadata:
            raise KeyError("No 'acqInfo' in PSA data")
        
        acq_info = psadata["acqInfo"]
        print(f"      acqInfo fields: {acq_info.dtype.names if hasattr(acq_info, 'dtype') else 'N/A'}")
        
        # delta_t
        if "tot_run_time" in psadata:
            delta_t = float(np.squeeze(psadata["tot_run_time"]))
            print(f"      delta_t from tot_run_time: {delta_t}")
        else:
            # Calculate from acqInfo
            if 'SampleRate' not in acq_info.dtype.names:
                raise KeyError("No 'SampleRate' in acqInfo")
            
            SampleRate = float(acq_info['SampleRate'][0, 0])
            Depth = float(acq_info['Depth'][0, 0])
            SegmentCount = float(acq_info['SegmentCount'][0, 0])
            nAcq = float(psadata.get("nAcq", 1.0)[0, 0])
            
            delta_t = Depth * SegmentCount * nAcq / SampleRate
            print(f"      delta_t calculated: {delta_t}")

        # fresolution from frequency axis
        freq_Hz_spec = np.asarray(freq_Hz_spec).ravel()
        if freq_Hz_spec.size < 2:
            raise ValueError("freq_Hz_spec must have at least 2 points to compute fresolution")
        fresolution = float(abs(freq_Hz_spec[1] - freq_Hz_spec[0]))
        print(f"      fresolution: {fresolution} Hz")

        # samplerate from acqInfo
        if 'SampleRate' not in acq_info.dtype.names:
            raise KeyError("No 'SampleRate' in acqInfo")
        samplerate = float(acq_info['SampleRate'][0, 0])
        print(f"      samplerate: {samplerate} Hz")

        # FFTsize from frequency axis length
        FFTsize = int((freq_Hz_spec.size - 1) * 2)
        print(f"      FFTsize: {FFTsize}")

        return delta_t, fresolution, samplerate, FFTsize

    def _compute_frequency_indices(
        self,
        freq_Hz_spec: np.ndarray,
        fresolution: float,
        proc_par: Any,
    ) -> Tuple[int, Tuple[int, int], Tuple[int, int], int, int]:
        """
        Compute quick_pad, IFdipidx, IFwinidx, lowpass_idx, pr_idx.

        MATLAB:
            quick_pad = ceil(100e3/fresolution); % rough window of the SG filter

            % find points in the range of the IF dip
            IFdippts = find((freq_Hz_spec>=proc_par.IFdiploc_MHz(1)*10^6)& ...
                            (freq_Hz_spec<=proc_par.IFdiploc_MHz(2)*10^6));
            IFdipidx(1) = IFdippts(1);
            IFdipidx(2) = IFdippts(end);

            % find points in the range of the IF window
            IFwinpts = find((freq_Hz_spec>=proc_par.IFwindow(1))&(freq_Hz_spec<=proc_par.IFwindow(2)));

            % Find the point closest to the lowpass frequency
            [~,lowpass_idx] = min(abs(proc_par.LowPass_MHz*1e6 - freq_Hz_spec));

            % Find the point closest to the probe tone
            [~,pr_idx] = min(abs(proc_par.pr_loc_kHz_data*1000 - freq_Hz_spec));
        """
        freq_Hz_spec = np.asarray(freq_Hz_spec, dtype=float).ravel()
        if freq_Hz_spec.size < 2:
            raise ValueError("freq_Hz_spec must have at least 2 points")

        # quick_pad = ceil(100e3 / fresolution)
        quick_pad = int(np.ceil(100e3 / fresolution))

        # Calibration parameters
        calib = getattr(proc_par, "calibration", {})

        IFdiploc_MHz = np.asarray(calib.get("IFdiploc_MHz", [0.0, 0.0]), dtype=float)
        if IFdiploc_MHz.size != 2:
            raise ValueError("proc_par.calibration['IFdiploc_MHz'] must have length 2")
        dip_lo = IFdiploc_MHz[0] * 1e6
        dip_hi = IFdiploc_MHz[1] * 1e6

        # IF dip indices
        dip_mask = (freq_Hz_spec >= dip_lo) & (freq_Hz_spec <= dip_hi)
        dip_pts = np.where(dip_mask)[0]
        if dip_pts.size == 0:
            raise ValueError("No frequency points found in IF dip range")
        IFdipidx = (int(dip_pts[0]), int(dip_pts[-1]))

        # IF window (proc_par.IFwindow already in Hz)
        IFwindow = np.asarray(getattr(proc_par, "IFwindow", [0.0, 0.0]), dtype=float)
        if IFwindow.size != 2:
            raise ValueError("proc_par.IFwindow must have length 2 (Hz)")
        win_lo, win_hi = IFwindow
        win_mask = (freq_Hz_spec >= win_lo) & (freq_Hz_spec <= win_hi)
        win_pts = np.where(win_mask)[0]
        if win_pts.size == 0:
            raise ValueError("No frequency points found in IF window")
        IFwinidx = (int(win_pts[0]), int(win_pts[-1]))

        # Low-pass index: closest to LowPass_MHz * 1e6
        lowpass_target = float(proc_par.LowPass_MHz) * 1e6
        lowpass_idx = int(np.argmin(np.abs(lowpass_target - freq_Hz_spec)))

        # Probe tone index: closest to pr_loc_kHz_data * 1000
        pr_loc_kHz_data = np.asarray(calib.get("pr_loc_kHz_data", [0.0]), dtype=float)
        if pr_loc_kHz_data.size == 0:
            pr_idx = lowpass_idx
        else:
            targets_Hz = pr_loc_kHz_data * 1e3
            # Build distance matrix and pick global min
            diff = np.abs(freq_Hz_spec[:, None] - targets_Hz[None, :])
            flat = int(np.argmin(diff))
            n_targets = targets_Hz.size
            pr_idx = flat // n_targets

        return quick_pad, IFdipidx, IFwinidx, lowpass_idx, pr_idx




    def _analyze_squeezing_blocks(
        self,
        psadata: Dict[str, Any],
        GA: float,
        freq_Hz_spec: np.ndarray,
        IFwinidx: Tuple[int, int],
        lowpass_idx: int,
        pr_idx: int,
        probe_scale: float,
        proc_par: Any,
    ) -> Dict[str, Any]:
        """
        Analyze off-resonant squeezing calibration (Phase IIc).

        MATLAB (simplified):

            sqz_vs_freq_dB = pow2db(psadata.meanavgps_sqzON.singlesided_powerspecavg ./ ...
                                    psadata.meanavgps_sqzOFF.singlesided_powerspecavg);
            avg_sqdB_off = mean(sqz_vs_freq_dB(IFwinidx(2):lowpass_idx));
            peak_sqdB    = mean(sqz_vs_freq_dB(1:IFwinidx(1)));
            avg_sqdB_IF  = mean(sqz_vs_freq_dB(IFwinidx(1):IFwinidx(2)));
            dat_spec_OFF = psadata.meanavgps_sqzOFF.singlesided_powerspecavg/GA;
            dat_spec_ON  = psadata.meanavgps_sqzON.singlesided_powerspecavg/GA;
            dat_spec_OFF_sq  = psadata.meanavgps_sqzOFF.singlesided_powerspecavg_sq;
            scaleinfo.sum_power_in_IF_sq_sqOFF(i) = sum(dat_spec_OFF_sq(IFwinidx(1):IFwinidx(2)));
            scaleinfo.sum_power_in_IF_sqOFF(i)    = sum(dat_spec_OFF(IFwinidx(1):IFwinidx(2)));
            scaleinfo.sum_power_in_IF_sqON(i)     = sum(dat_spec_ON(IFwinidx(1):IFwinidx(2)));
            scaleinfo.spec_bl_sqOFF(i)   = (mean(dat_spec_OFF(pr_idx+10:pr_idx+30))+mean(dat_spec_OFF(pr_idx-30:pr_idx-10)))/2;
            scaleinfo.pr_height_sqOFF(i) = max(dat_spec_OFF(pr_idx-30:pr_idx+30))/(scaleinfo.pr_power(i)*probe_scale);
        """
        out = {
            "avg_sqdB_off": 0.0,
            "avg_sqdB_IF": 0.0,
            "peak_sqdB": 0.0,
            "sum_power_in_IF_sq_sqOFF": 0.0,
            "sum_power_in_IF_sqOFF": 0.0,
            "sum_power_in_IF_sqON": 0.0,
            "spec_bl_sqOFF": 0.0,
            "pr_height_sqOFF": 0.0,
        }

        meanavgps_sqzON = psadata.get("meanavgps_sqzON", None)
        meanavgps_sqzOFF = psadata.get("meanavgps_sqzOFF", None)
        if meanavgps_sqzON is None or meanavgps_sqzOFF is None:
            return out

        if isinstance(meanavgps_sqzON, np.ndarray):
            meanavgps_sqzON = meanavgps_sqzON.item()
        if isinstance(meanavgps_sqzOFF, np.ndarray):
            meanavgps_sqzOFF = meanavgps_sqzOFF.item()

        def _get_field(struct_obj: Any, name: str) -> np.ndarray:
            if hasattr(struct_obj, name):
                return np.asarray(getattr(struct_obj, name))
            if isinstance(struct_obj, np.void) and name in struct_obj.dtype.names:
                return np.asarray(struct_obj[name])
            raise KeyError(f"Struct has no field '{name}'")

        try:
            psd_on = _get_field(meanavgps_sqzON, "singlesided_powerspecavg").ravel()
            psd_off = _get_field(meanavgps_sqzOFF, "singlesided_powerspecavg").ravel()
            psd_off_sq = _get_field(meanavgps_sqzOFF, "singlesided_powerspecavg_sq").ravel()
        except Exception:
            return out

        # sqz_vs_freq_dB = 10*log10(on/off)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = psd_on / psd_off
            ratio = np.where(ratio > 0, ratio, np.nan)
            sqz_vs_freq_dB = 10.0 * np.log10(ratio)

        i1, i2 = IFwinidx
        n = sqz_vs_freq_dB.size
        i1 = max(0, min(i1, n - 1))
        i2 = max(0, min(i2, n - 1))
        lp = max(0, min(lowpass_idx, n - 1))

        # avg_sqdB_off = mean(sqz_vs_freq_dB(IFwinidx(2):lowpass_idx));
        if lp >= i2:
            out["avg_sqdB_off"] = float(np.nanmean(sqz_vs_freq_dB[i2: lp + 1]))

        # peak_sqdB = mean(sqz_vs_freq_dB(1:IFwinidx(1)));
        if i1 > 0:
            out["peak_sqdB"] = float(np.nanmean(sqz_vs_freq_dB[0:i1]))

        # avg_sqdB_IF = mean(sqz_vs_freq_dB(IFwinidx(1):IFwinidx(2)));
        if i2 >= i1:
            out["avg_sqdB_IF"] = float(np.nanmean(sqz_vs_freq_dB[i1: i2 + 1]))

        # OFF/ON in "amp" units
        dat_spec_OFF = psd_off / GA
        dat_spec_ON = psd_on / GA

        band_OFF = dat_spec_OFF[i1: i2 + 1]
        band_ON = dat_spec_ON[i1: i2 + 1]

        out["sum_power_in_IF_sq_sqOFF"] = float(np.sum(psd_off_sq[i1: i2 + 1]))
        out["sum_power_in_IF_sqOFF"] = float(np.sum(band_OFF))
        out["sum_power_in_IF_sqON"] = float(np.sum(band_ON))

        # spec_bl_sqOFF = (mean(dat_spec_OFF(pr_idx+10:pr_idx+30)) + mean(dat_spec_OFF(pr_idx-30:pr_idx-10)))/2;
        n_spec = dat_spec_OFF.size
        p = int(pr_idx)

        def _safe_mean(arr: np.ndarray, start: int, stop: int) -> float:
            start = max(0, min(start, arr.size))
            stop = max(start, min(stop, arr.size))
            if stop <= start:
                return np.nan
            return float(np.nanmean(arr[start:stop]))

        plus_mean = _safe_mean(dat_spec_OFF, p + 10, p + 30)
        minus_mean = _safe_mean(dat_spec_OFF, p - 30, p - 10)
        if np.isfinite(plus_mean) and np.isfinite(minus_mean):
            out["spec_bl_sqOFF"] = 0.5 * (plus_mean + minus_mean)

        # pr_height_sqOFF = max(dat_spec_OFF(pr_idx-30:pr_idx+30))/(pr_power*probe_scale);
        start = max(0, p - 30)
        stop = min(n_spec, p + 30)
        if stop > start:
            max_off = float(np.nanmax(dat_spec_OFF[start:stop]))
        else:
            max_off = 0.0

        # pr_power from tone_power_PR_fordata (db2pow)
        try:
            tone_power_db = float(np.squeeze(psadata["tone_power_PR_fordata"]))
        except Exception:
            tone_power_db = -45.0
        pr_power = 10.0 ** (tone_power_db / 10.0)

        norm = pr_power * probe_scale
        if norm > 0:
            out["pr_height_sqOFF"] = max_off / norm

        return out


    def _compute_spectral_metrics(
        self,
        dat_spec: np.ndarray,
        dat_spec_sq: np.ndarray,
        IFdipidx: Tuple[int, int],
        IFwinidx: Tuple[int, int],
        pr_idx: int,
        quick_pad: int,
        probe_scale: float,
        proc_par: Any,
        delta_t: float,
        fresolution: float,
        GA: float,
        psadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute baseline, probe heights, IF dip, IF-band sums, pr_power and pr_power_stds.

        MATLAB:

            spec_bl        = (mean(dat_spec(pr_idx+10:pr_idx+30))+ ...
                              mean(dat_spec(pr_idx-30:pr_idx-10)))/2;
            pr_height      = max(dat_spec(pr_idx-30:pr_idx+30));
            pr_height_sqz  = max(dat_spec_sq(pr_idx-30:pr_idx+30));
            dip_height     = min(dat_spec(IFdipidx(1):IFdipidx(2)));

            scaleinfo.sum_power_in_IF_sq(i) = sum(dat_spec_sq(IFwinidx(1):IFwinidx(2)));
            scaleinfo.sum_power_in_IF(i)    = sum(dat_spec(IFwinidx(1):IFwinidx(2)));
            scaleinfo.mean_of_spec(i)       = mean(dat_spec(IFwinidx(1):IFwinidx(2)+ quick_pad -1));

            try
                scaleinfo.pr_power(i) = db2pow(psadata.tone_power_PR_fordata);
            catch
                scaleinfo.pr_power(i) = db2pow(-45.0);
            end

            scaleinfo.pr_power_stds(i)  = std(psadata.meanavgps.pt_power_est_list);
            scaleinfo.pr_power_stds(i)  = scaleinfo.pr_power_stds(i)/(sqrt(scaleinfo.pr_power(i))*GA*probe_scale);
        """
        dat_spec = np.asarray(dat_spec, dtype=float).ravel()
        dat_spec_sq = np.asarray(dat_spec_sq, dtype=float).ravel()
        n = dat_spec.size

        i_dip1, i_dip2 = IFdipidx
        i_win1, i_win2 = IFwinidx

        # --- baseline around probe ---
        def _safe_mean(arr: np.ndarray, start: int, stop: int) -> float:
            start = max(start, 0)
            stop = min(stop, arr.size)
            if stop <= start:
                return np.nan
            return float(np.nanmean(arr[start:stop]))

        mean_plus = _safe_mean(dat_spec, pr_idx + 10, pr_idx + 31)
        mean_minus = _safe_mean(dat_spec, pr_idx - 30, pr_idx - 9)
        spec_bl = 0.5 * (mean_plus + mean_minus)

        # probe heights (raw, unnormalized)
        pr_start = max(pr_idx - 30, 0)
        pr_stop = min(pr_idx + 31, n)
        if pr_stop > pr_start:
            pr_height_raw = float(np.nanmax(dat_spec[pr_start:pr_stop]))
            pr_height_sqz_raw = float(np.nanmax(dat_spec_sq[pr_start:pr_stop]))
        else:
            pr_height_raw = np.nan
            pr_height_sqz_raw = np.nan

        # IF dip
        i_dip1 = max(0, min(i_dip1, n - 1))
        i_dip2 = max(0, min(i_dip2, n - 1))
        if i_dip2 >= i_dip1:
            dip_height = float(np.nanmin(dat_spec[i_dip1: i_dip2 + 1]))
        else:
            dip_height = np.nan

        # IF-band sums and mean
        i_win1 = max(0, min(i_win1, n - 1))
        i_win2 = max(0, min(i_win2, n - 1))
        if i_win2 >= i_win1:
            band_sq = dat_spec_sq[i_win1: i_win2 + 1]
            band = dat_spec[i_win1: i_win2 + 1]
            sum_power_in_IF_sq = float(np.nansum(band_sq))
            sum_power_in_IF = float(np.nansum(band))
        else:
            sum_power_in_IF_sq = 0.0
            sum_power_in_IF = 0.0

        mean_end = min(i_win2 + quick_pad, n)
        if mean_end > i_win1:
            mean_of_spec = float(np.nanmean(dat_spec[i_win1:mean_end]))
        else:
            mean_of_spec = np.nan

        # pr_power (db2pow)
        try:
            tone_power_db = float(np.squeeze(psadata["tone_power_PR_fordata"]))
        except Exception:
            tone_power_db = -45.0
        pr_power = 10.0 ** (tone_power_db / 10.0)

        # pr_power_stds from meanavgps.pt_power_est_list
        pr_power_stds = np.nan
        meanavgps = psadata.get("meanavgps", None)
        if isinstance(meanavgps, np.ndarray):
            meanavgps = meanavgps.item()
        if meanavgps is not None and hasattr(meanavgps, "pt_power_est_list"):
            pt_list = getattr(meanavgps, "pt_power_est_list")
            arr = np.asarray(pt_list)
            if arr.dtype == object:
                arr = np.array(arr.tolist(), dtype=float).ravel()
            else:
                arr = arr.astype(float).ravel()
            if arr.size > 0:
                raw_std = float(np.nanstd(arr, ddof=0))
                if pr_power > 0 and GA > 0 and probe_scale > 0:
                    pr_power_stds = raw_std / (np.sqrt(pr_power) * GA * probe_scale)

        # Normalize probe heights
        norm = pr_power * probe_scale
        if norm > 0 and np.isfinite(pr_height_raw):
            pr_height_norm = pr_height_raw / norm
        else:
            pr_height_norm = np.nan
        if norm > 0 and np.isfinite(pr_height_sqz_raw):
            pr_height_sqz_norm = pr_height_sqz_raw / norm
        else:
            pr_height_sqz_norm = np.nan

        return {
            "spec_bl": spec_bl,
            "pr_height_raw": pr_height_raw,
            "pr_height_sqz_raw": pr_height_sqz_raw,
            "dip_height": dip_height,
            "sum_power_in_IF_sq": sum_power_in_IF_sq,
            "sum_power_in_IF": sum_power_in_IF,
            "mean_of_spec": mean_of_spec,
            "pr_power": pr_power,
            "pr_power_stds": pr_power_stds,
            "pr_height_norm": pr_height_norm,          # matches MATLAB scaleinfo.pr_height
            "pr_height_sqz_norm": pr_height_sqz_norm,  # matches MATLAB scaleinfo.pr_height_sqz
        }


    def _extract_alignment_angle(self, psadata: Dict[str, Any]) -> float:
        """
        Extract and wrap alignment angle.

        MATLAB:
            try
                scaleinfo.align_ang(i)    = psadata.align_ang;
                if psadata.align_ang < 0
                    scaleinfo.align_ang(i)    = psadata.align_ang + pi;
                end
            catch
                scaleinfo.align_ang(i)    = -1;
            end
        """
        raise NotImplementedError

    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Basic validation for spectrum info outputs.
        """
        result = data.get("spectrum_info")
        if not result:
            print("  ✗ No spectrum info result found")
            return False

        # For a first pass, just check that we have some updates;
        # later we can enforce specific required fields (e.g. 'sum_power_in_IF').
        if not result.scaleinfo_updates:
            print("  ✗ No scaleinfo updates from SpectrumInfoStage")
            return False

        print("  ✓ Spectrum info stage produced scaleinfo updates")
        return True
    
    def _extract_alignment_angle(self, psadata: Dict[str, Any]) -> float:
        """
        Extract and wrap alignment angle.

        MATLAB:
            try
                scaleinfo.align_ang(i)    = psadata.align_ang;
                if psadata.align_ang < 0
                    scaleinfo.align_ang(i)    = psadata.align_ang + pi;
                end
            catch
                scaleinfo.align_ang(i)    = -1;
            end
        """
        import math

        try:
            align_ang = float(np.squeeze(psadata["align_ang"]))
        except Exception:
            return -1.0

        if align_ang < 0:
            align_ang = align_ang + math.pi

        return align_ang

def _movmean_omitnan(x: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving mean with NaN omission, emulating MATLAB movmean(...,'omitnan').

    Uses convolution of values and a non-NaN mask.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x.astype(float)

    kernel = np.ones(window, dtype=float)

    # Replace NaN with 0 for the sum; track counts separately
    valid = ~np.isnan(x)
    x_zeronan = np.where(valid, x, 0.0)

    sum_conv = np.convolve(x_zeronan, kernel, mode="same")
    count_conv = np.convolve(valid.astype(float), kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_conv / count_conv

    mean[count_conv == 0.0] = np.nan
    return mean


def smoothvar(
    in_val: np.ndarray,
    cutpar: Any,
    scaleinfo: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python translation of MATLAB smoothvar.m.

    [smooth_par, smoothVar] = smoothvar(scaleinfo.sum_power_in_IF, cutpar, scaleinfo);

    - smooth_par: smoothed version (moving mean)
    - smoothVar:  standardized residuals (value / local_mean - 1) / global_std
    """
    in_val = np.asarray(in_val, dtype=float)
    n = in_val.size

    smooth_val = np.zeros_like(in_val, dtype=float)
    cut_val = np.zeros_like(in_val, dtype=float)

    # Number of points in moving average
    num_avg = int(getattr(cutpar, "smooth_width", 10))

    # Global std over entire array (MATLAB: std(in_val./movmean(in_val,num_avg,'omitnan') - 1,"omitnan"))
    global_mean = _movmean_omitnan(in_val, num_avg)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_global = in_val / global_mean
    out_global = out_global - 1.0
    std_val = float(np.nanstd(out_global))

    if std_val == 0.0 or not np.isfinite(std_val):
        # Degenerate case; return zeros for deviations
        smooth_val[:] = global_mean
        cut_val[:] = 0.0
        return smooth_val, cut_val

    # Determine segmentation (smoothing regions) if requested
    if getattr(cutpar, "find_smoothing_regions", False):
        cavity_freq = np.asarray(scaleinfo.get("Cavity_freq", []), dtype=float).ravel()
        spec_dates = np.asarray(scaleinfo.get("spectrum_date", []), dtype=float).ravel()

        if cavity_freq.size != n or spec_dates.size != n:
            # Fallback: treat as single region
            jump_idx = np.array([0, n], dtype=int)
        else:
            # freq_jump_GHz threshold
            freq_jump_GHz = float(getattr(cutpar, "freq_jump_GHz", 0.0))
            freq_jump_cut = np.abs(np.diff(cavity_freq)) > freq_jump_GHz

            # datenum change: any change in spectrum_date
            datenum_cut = np.abs(np.diff(spec_dates)) > 0.0

            joint_cut = np.logical_or(freq_jump_cut, datenum_cut)

            idx_list = np.arange(n, dtype=int)
            # MATLAB: idx_list(joint_cut) where idx_list = 1:length -> indices of segment boundaries
            # For Python 0-based, diff has length n-1 and corresponds to transitions between idx k and k+1.
            boundary_indices = idx_list[1:][joint_cut]
            jump_idx = np.concatenate(([0], boundary_indices, [n])).astype(int)
    else:
        jump_idx = np.array([0, n], dtype=int)

    # Loop over segments defined by jump_idx
    for k in range(jump_idx.size - 1):
        start_idx = int(jump_idx[k])
        end_idx = int(jump_idx[k + 1])

        if end_idx <= start_idx:
            continue

        seg = in_val[start_idx:end_idx]
        seg_mean = _movmean_omitnan(seg, num_avg)

        with np.errstate(divide="ignore", invalid="ignore"):
            out_seg = seg / seg_mean
        out_seg = out_seg - 1.0
        cut_seg = out_seg / std_val

        smooth_val[start_idx:end_idx] = seg_mean
        cut_val[start_idx:end_idx] = cut_seg

    return smooth_val, cut_val
