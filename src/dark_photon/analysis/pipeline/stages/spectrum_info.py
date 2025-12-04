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

        Mirrors the main loop in LoadSpectrumInfo:

            % Loop over PSA files 
            for i=1:length(files)
                % Get the PSA File Name and Info
                [psafile, spectrum_date, spectrum_it, spectrum_par_num] = stripfilename(files{i}, 'psa');
                scaleinfo.spectrum_date(i)    = spectrum_date;
                scaleinfo.spectrum_it(i)      = spectrum_it;
                scaleinfo.spectrum_par_num(i) = spectrum_par_num;
                ...
            end
        """
        print("  Loading PSA spectrum info.")

        # 1) Get file list from file enumeration stage
        file_enum = data.get("file_enumeration")
        if not file_enum or not file_enum.files:
            raise ValueError("No files found from file enumeration stage")

        files: List[Path] = file_enum.files
        num_files = len(files)

        # 2) Get merged scaleinfo (TX + RFL + JPA) as the MATLAB input "scaleinfo"
        merged_scaleinfo: Dict[str, Any] = data.get("scaleinfo")
        if merged_scaleinfo is None:
            raise ValueError("Merged scaleinfo not found; ScaleinfoMergeStage must run before SpectrumInfoStage")

        # 3) Access processing parameters (MATLAB 'proc_par') and DQ cuts ('cutpar')
        proc_par = context.run_props.processing
        cutpar = context.run_props.data_quality

        # 4) Prepare containers for per-spectrum outputs that will go into scaleinfo_updates
        #    (we'll mirror the MATLAB fields later; for now just create an empty dict)
        scaleinfo_updates: Dict[str, Any] = {}

        # Example: preallocate arrays once we know how many spectra we have
        # (later we will add real fields like 'probe_scale', 'sum_power_in_IF', etc.)
        # probe_scale_arr = np.zeros(num_files, dtype=float)
        # sum_power_in_IF_arr = np.zeros(num_files, dtype=float)
        # ...

        # 5) Main loop over TX/PSA files (MATLAB: "for i=1:length(files)")
        for i, tx2_file in enumerate(files):
            try:
                # --- 5.1 Resolve PSA filename and metadata ---
                # MATLAB:
                #   [psafile, spectrum_date, spectrum_it, spectrum_par_num] = stripfilename(files{i}, 'psa');
                #   scaleinfo.spectrum_date(i)    = spectrum_date;
                #   scaleinfo.spectrum_it(i)      = spectrum_it;
                #   scaleinfo.spectrum_par_num(i) = spectrum_par_num;
                psa_file, spectrum_date, spectrum_it, spectrum_par_num = self._get_psa_file_and_metadata(
                    tx2_file
                )

                # Here we'll eventually store:
                # - spectrum_date
                # - spectrum_it
                # - spectrum_par_num
                # into scaleinfo_updates as lists.

                # --- 5.2 Load PSA data ---
                # MATLAB:
                #   psadata = importdata(psafile);
                psadata = self._load_psa_data(psa_file)

                # --- 5.3 Extract spectra and frequency axis ---
                # MATLAB:
                #   GA           = psadata.gain_amp_pow;
                #   dat_spec     = psadata.meanavgps.singlesided_powerspecavg/GA;
                #   dat_spec_sq  = psadata.meanavgps.singlesided_powerspecavg_sq;
                #   freq_Hz_spec = psadata.meanavgps.singlesided_freqaxis;
                #
                # Here we only sketch the call; actual extraction will be implemented later.
                (
                    GA,
                    dat_spec,
                    dat_spec_sq,
                    freq_Hz_spec,
                ) = self._extract_spectrum(psadata)

                # --- 5.4 Probe scaling from cavity Q and coupling ---
                # MATLAB:
                #   QL   = scaleinfo.Cavity_Q(i);
                #   beta = scaleinfo.coupling_factor(i);
                #   probe_scale = QL*(beta/(1+beta));
                #   scaleinfo.probe_scale(i) = probe_scale;
                probe_scale = self._compute_probe_scale(merged_scaleinfo, i)

                # --- 5.5 Normalization factors (AS / IQ) ---
                # MATLAB:
                #   try
                #       scaleinfo.as_norm_fac(i) = psadata.meanavgps.asNormFac;
                #       scaleinfo.iq_norm_fac(i) = psadata.meanavgps.iqNormFac;
                #   catch
                #       scaleinfo.as_norm_fac(i) = -1.0;
                #       scaleinfo.iq_norm_fac(i) = -1.0;
                #   end
                #   try
                #       scaleinfo.as_norm_fac_corr(i) = psadata.meanavgps.asNormFac_corrr;
                #   catch
                #       scaleinfo.as_norm_fac_corr(i) = -1.0;
                #   end
                as_norm_fac, iq_norm_fac, as_norm_fac_corr = self._extract_norm_factors(psadata)

                # --- 5.6 DAQ parameters and frequency resolution ---
                # MATLAB:
                #   try
                #       delta_t     = psadata.tot_run_time;
                #   catch
                #       delta_t    = psadata.acqInfo.Depth*psadata.acqInfo.SegmentCount*psadata.nAcq/psadata.acqInfo.SampleRate;
                #   end
                #   fresolution = abs(freq_Hz_spec(1) - freq_Hz_spec(2));
                #   samplerate  = psadata.acqInfo.SampleRate;
                #   FFTsize     = (length(psadata.meanavgps.singlesided_freqaxis) - 1)*2;
                delta_t, fresolution, samplerate, FFTsize = self._compute_daq_parameters(
                    psadata, freq_Hz_spec
                )

                # --- 5.7 IF windows and probe / low-pass indices ---
                # MATLAB:
                #   quick_pad = ceil(100e3/fresolution);
                #   IFdippts = find((freq_Hz_spec>=proc_par.IFdiploc_MHz(1)*10^6)&(freq_Hz_spec<=proc_par.IFdiploc_MHz(2)*10^6));
                #   IFwinpts = find((freq_Hz_spec>=proc_par.IFwindow(1))&(freq_Hz_spec<=proc_par.IFwindow(2)));
                #   [~,lowpass_idx] = min(abs(proc_par.LowPass_MHz*1e6 - freq_Hz_spec));
                #   [~,pr_idx]      = min(abs(proc_par.pr_loc_kHz_data*1000 - freq_Hz_spec));
                (
                    quick_pad,
                    IFdipidx,
                    IFwinidx,
                    lowpass_idx,
                    pr_idx,
                ) = self._compute_frequency_indices(freq_Hz_spec, fresolution, proc_par)

                # --- 5.8 Squeezing / ON-OFF calibration (Phase IIc) ---
                # MATLAB (inner try/catch block with meanavgps_sqzON / sqzOFF etc.)
                # We'll mirror this logic in a dedicated helper to keep execute() readable.
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

                # --- 5.9 Baselines, probe heights, IF dip, IF-band sums ---
                # MATLAB:
                #   spec_bl        = (mean(dat_spec(pr_idx+10:pr_idx+30))+ ...
                #                     mean(dat_spec(pr_idx-30:pr_idx-10)))/2;
                #   pr_height      = max(dat_spec(pr_idx-30:pr_idx+30));
                #   pr_height_sqz  = max(dat_spec_sq(pr_idx-30:pr_idx+30));
                #   dip_height     = min(dat_spec(IFdipidx(1):IFdipidx(2)));
                #   ...
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
                    spectrum_date,
                )


                # --- 5.10 Alignment angle and final bookkeeping ---
                # MATLAB:
                #   try
                #       scaleinfo.align_ang(i)    = psadata.align_ang;
                #       if psadata.align_ang < 0
                #           scaleinfo.align_ang(i)    = psadata.align_ang + pi;
                #       end
                #   catch
                #       scaleinfo.align_ang(i)    = -1;
                #   end
                #   scaleinfo.delta_t(i)     = delta_t;
                #   scaleinfo.fresolution(i) = fresolution;
                #   scaleinfo.samplerate(i)  = samplerate;
                #   scaleinfo.FFTsize(i)     = FFTsize;
                #   scaleinfo.filename(i)    = {psafile};
                #   scaleinfo.sigma_exp(i)  = 1.0./sqrt(delta_t.*fresolution*proc_par.binavg);
                #   scaleinfo.fitnumber_list(i)  = ceil(proc_par.sg_win_Hz/fresolution) + 1;
                #   scaleinfo.fitnumber2(i) = scaleinfo.fitnumber_list(i);
                align_angle = self._extract_alignment_angle(psadata)

                # TODO: collect all per-spectrum quantities into temporary arrays
                # and then into scaleinfo_updates after the loop.

            except Exception as e:
                warnings.warn(f"Error processing PSA spectrum for {tx2_file}: {e}")
                continue

        # 6) Post-loop: MATLAB sets a global fitnumber
        #   scaleinfo.fitnumber = scaleinfo.fitnumber_list(1);
        # We'll implement the Python equivalent when we actually create fitnumber_list array.
        # For now, just a placeholder for where that logic will go.
        # e.g.:
        # if 'fitnumber_list' in scaleinfo_updates and len(scaleinfo_updates['fitnumber_list']) > 0:
        #     scaleinfo_updates['fitnumber'] = float(scaleinfo_updates['fitnumber_list'][0])

        # 7) Package results
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
        # 1) Get file base (e.g. '/path/.../20220908_0_0_')
        file_base = self._get_file_base(tx2_file)

        # 2) Derive PSA filename by appending 'psa.mat'
        #    MATLAB stripfilename(files{i}, 'psa') effectively points to the PSA file
        psa_file = file_base.with_name(file_base.name + "psa.mat")

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
        """
        # Gain amplitude power
        try:
            GA = float(np.squeeze(psadata["gain_amp_pow"]))
        except Exception as e:
            raise KeyError(f"Could not read 'gain_amp_pow' from PSA data: {e}")

        # meanavgps is a nested MATLAB struct
        meanavgps = psadata.get("meanavgps", None)
        if meanavgps is None:
            raise KeyError("PSA data has no 'meanavgps' field")

        # SciPy often gives a numpy.void or a small ndarray; handle both
        if isinstance(meanavgps, np.ndarray):
            meanavgps = meanavgps.item()  # flatten (1, 1) -> struct object if needed

        def _get_field(obj, name: str) -> np.ndarray:
            if hasattr(obj, name):
                return np.asarray(getattr(obj, name))
            if isinstance(obj, np.void) and name in obj.dtype.names:
                return np.asarray(obj[name])
            raise KeyError(f"'meanavgps' has no field '{name}'")

        singlesided_powerspecavg = _get_field(meanavgps, "singlesided_powerspecavg")
        singlesided_powerspecavg_sq = _get_field(meanavgps, "singlesided_powerspecavg_sq")
        singlesided_freqaxis = _get_field(meanavgps, "singlesided_freqaxis")

        dat_spec = np.squeeze(singlesided_powerspecavg) / GA
        dat_spec_sq = np.squeeze(singlesided_powerspecavg_sq)
        freq_Hz_spec = np.squeeze(singlesided_freqaxis)

        return dat_spec, dat_spec_sq, freq_Hz_spec, GA

    # --- MATLAB:
    # QL   = scaleinfo.Cavity_Q(i);
    # beta = scaleinfo.coupling_factor(i);
    # probe_scale = QL*(beta/(1+beta));
    # scaleinfo.probe_scale(i) = probe_scale;
    # ---
    def _compute_probe_scale(
        self, scaleinfo: Dict[str, Any], idx: int
    ) -> float:
        """
        Compute probe_scale and update scaleinfo['probe_scale'] for index idx.
        """
        try:
            QL = float(scaleinfo["Cavity_Q"][idx])
            beta = float(scaleinfo["coupling_factor"][idx])
        except Exception as e:
            raise KeyError(f"Missing 'Cavity_Q' or 'coupling_factor' in scaleinfo at index {idx}: {e}")

        probe_scale = QL * (beta / (1.0 + beta))

        # Ensure probe_scale list exists and is long enough
        if "probe_scale" not in scaleinfo:
            scaleinfo["probe_scale"] = [0.0] * self._get_parameter_count(scaleinfo)

        scaleinfo["probe_scale"][idx] = probe_scale
        return probe_scale

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
        if isinstance(meanavgps, np.ndarray):
            meanavgps = meanavgps.item()

        def _safe_get(name: str, default: float = -1.0) -> float:
            try:
                if meanavgps is None:
                    return default
                if hasattr(meanavgps, name):
                    return float(np.squeeze(getattr(meanavgps, name)))
                if isinstance(meanavgps, np.void) and name in meanavgps.dtype.names:
                    return float(np.squeeze(meanavgps[name]))
                return default
            except Exception:
                return default

        as_norm = _safe_get("asNormFac", -1.0)
        iq_norm = _safe_get("iqNormFac", -1.0)
        as_norm_corr = _safe_get("asNormFac_corrr", -1.0)

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
        # acqInfo is a MATLAB struct inside psadata
        acq_info = psadata.get("acqInfo", None)
        if isinstance(acq_info, np.ndarray):
            acq_info = acq_info.item()

        # delta_t
        if "tot_run_time" in psadata:
            delta_t = float(np.squeeze(psadata["tot_run_time"]))
        else:
            if acq_info is None:
                raise KeyError("No 'tot_run_time' or 'acqInfo' in PSA data")
            # Depth * SegmentCount * nAcq / SampleRate
            Depth = float(getattr(acq_info, "Depth"))
            SegmentCount = float(getattr(acq_info, "SegmentCount"))
            SampleRate = float(getattr(acq_info, "SampleRate"))
            nAcq = float(psadata.get("nAcq", 1.0))
            delta_t = Depth * SegmentCount * nAcq / SampleRate

        # fresolution from frequency axis
        freq_Hz_spec = np.asarray(freq_Hz_spec).ravel()
        if freq_Hz_spec.size < 2:
            raise ValueError("freq_Hz_spec must have at least 2 points to compute fresolution")
        fresolution = float(abs(freq_Hz_spec[1] - freq_Hz_spec[0]))

        # samplerate
        if acq_info is None:
            raise KeyError("PSA data has no 'acqInfo' for SampleRate")
        samplerate = float(getattr(acq_info, "SampleRate"))

        # FFTsize
        if meanavgps is None:
            meanavgps = psadata.get("meanavgps", None)
            if isinstance(meanavgps, np.ndarray):
                meanavgps = meanavgps.item()

        if meanavgps is None:
            raise KeyError("No 'meanavgps' for FFTsize calculation")

        if hasattr(meanavgps, "singlesided_freqaxis"):
            freq_axis = np.asarray(meanavgps.singlesided_freqaxis)
        elif isinstance(meanavgps, np.void) and "singlesided_freqaxis" in meanavgps.dtype.names:
            freq_axis = np.asarray(meanavgps["singlesided_freqaxis"])
        else:
            raise KeyError("Cannot find 'singlesided_freqaxis' in meanavgps")

        FFTsize = int((freq_axis.size - 1) * 2)

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
            quick_pad = ceil(100e3/fresolution);
            IFdippts = find((freq_Hz_spec>=proc_par.IFdiploc_MHz(1)*10^6)&(freq_Hz_spec<=proc_par.IFdiploc_MHz(2)*10^6));
            IFwinpts = find((freq_Hz_spec>=proc_par.IFwindow(1))&(freq_Hz_spec<=proc_par.IFwindow(2)));
            [~,lowpass_idx] = min(abs(proc_par.LowPass_MHz*1e6 - freq_Hz_spec));
            [~,pr_idx]      = min(abs(proc_par.pr_loc_kHz_data*1000 - freq_Hz_spec));
        """
        freq_Hz_spec = np.asarray(freq_Hz_spec).ravel()

        # MATLAB: quick_pad = ceil(100e3/fresolution);
        quick_pad = int(np.ceil(100e3 / fresolution))

        # IF dip region (proc_par.calibration['IFdiploc_MHz'] in Python)
        IFdiploc_MHz = proc_par.calibration.get("IFdiploc_MHz", [0.0, 0.0])
        IFdip_low = IFdiploc_MHz[0] * 1e6
        IFdip_high = IFdiploc_MHz[1] * 1e6
        IFdip_mask = (freq_Hz_spec >= IFdip_low) & (freq_Hz_spec <= IFdip_high)
        IFdippts = np.where(IFdip_mask)[0]
        if IFdippts.size == 0:
            raise ValueError("No points found in IF dip range")
        IFdipidx = (int(IFdippts[0]), int(IFdippts[-1]))

        # IF window (already in Hz in ProcessingParameters.__post_init__) 
        IFwin_low = proc_par.IFwindow[0]
        IFwin_high = proc_par.IFwindow[1]
        IFwin_mask = (freq_Hz_spec >= IFwin_low) & (freq_Hz_spec <= IFwin_high)
        IFwinpts = np.where(IFwin_mask)[0]
        if IFwinpts.size == 0:
            raise ValueError("No points found in IF window range")
        IFwinidx = (int(IFwinpts[0]), int(IFwinpts[-1]))

        # Low-pass index (closest point to LowPass_MHz * 1e6)
        lowpass_target = proc_par.LowPass_MHz * 1e6
        lowpass_idx = int(np.argmin(np.abs(lowpass_target - freq_Hz_spec)))

        # Probe tone index (closest to pr_loc_kHz_data * 1000)
        pr_loc_kHz = proc_par.calibration.get("pr_loc_kHz_data", 0.0)
        pr_target = pr_loc_kHz * 1e3
        pr_idx = int(np.argmin(np.abs(pr_target - freq_Hz_spec)))

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
        Analyze sqzON / sqzOFF spectra if present.

        Mirrors the MATLAB logic that computes:
          - sqz_vs_freq_dB, avg_sqdB_off, peak_sqdB, avg_sqdB_IF
          - sum_power_in_IF_sq_sqOFF, sum_power_in_IF_sqOFF, sum_power_in_IF_sqON
        """
        out: Dict[str, Any] = {
            "avg_sqdB_off": np.nan,
            "peak_sqdB": np.nan,
            "avg_sqdB_IF": np.nan,
            "sum_power_in_IF_sq_sqOFF": np.nan,
            "sum_power_in_IF_sqOFF": np.nan,
            "sum_power_in_IF_sqON": np.nan,
        }

        # meanavgps_sqzON / OFF may be MATLAB structs stored in arrays
        meanavgps_sqzON = psadata.get("meanavgps_sqzON", None)
        meanavgps_sqzOFF = psadata.get("meanavgps_sqzOFF", None)

        if meanavgps_sqzON is None or meanavgps_sqzOFF is None:
            return out

        if isinstance(meanavgps_sqzON, np.ndarray):
            meanavgps_sqzON = meanavgps_sqzON.item()
        if isinstance(meanavgps_sqzOFF, np.ndarray):
            meanavgps_sqzOFF = meanavgps_sqzOFF.item()

        try:
            on_spec = np.asarray(meanavgps_sqzON.singlesided_powerspecavg, dtype=float)
            off_spec = np.asarray(meanavgps_sqzOFF.singlesided_powerspecavg, dtype=float)
        except Exception:
            return out

        # Guard against zero / negative values in log
        ratio = on_spec / off_spec
        ratio = np.where(ratio > 0, ratio, np.nan)
        sqz_vs_freq_dB = 10.0 * np.log10(ratio)

        i1, i2 = IFwinidx

        def _safe_mean(arr: np.ndarray) -> float:
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return np.nan
            return float(np.nanmean(arr))

        # MATLAB indices are inclusive; Python slices are [start:end)
        # IFwinidx(2):lowpass_idx  -> i2-1 : lowpass_idx
        out["avg_sqdB_off"] = _safe_mean(sqz_vs_freq_dB[i2: lowpass_idx + 1])
        # 1:IFwinidx(1) -> 0 : i1
        out["peak_sqdB"] = _safe_mean(sqz_vs_freq_dB[0: i1])
        # IFwinidx(1):IFwinidx(2) -> i1-1 : i2
        out["avg_sqdB_IF"] = _safe_mean(sqz_vs_freq_dB[i1: i2 + 1])

        # OFF/ON spectra in "amp" units (divide by GA)
        dat_spec_OFF = off_spec / GA
        dat_spec_ON = on_spec / GA

        band_OFF = dat_spec_OFF[i1: i2 + 1]
        band_ON = dat_spec_ON[i1: i2 + 1]

        sum_OFF = np.sum(band_OFF)
        sum_ON = np.sum(band_ON)

        out["sum_power_in_IF_sq_sqOFF"] = float(sum_OFF ** 2)
        out["sum_power_in_IF_sqOFF"] = float(sum_OFF)
        out["sum_power_in_IF_sqON"] = float(sum_ON)

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
        Compute baseline, probe heights, IF dip, IF sums, pr_power, pr_power_stds, etc.
        Mirrors the MATLAB logic in LoadSpectrumInfo.m.
        """
        dat_spec = np.asarray(dat_spec, dtype=float).ravel()
        dat_spec_sq = np.asarray(dat_spec_sq, dtype=float).ravel()

        i1_dip, i2_dip = IFdipidx
        i1_win, i2_win = IFwinidx

        # --- Baseline around probe tone ---
        # MATLAB indices: pr_idx+10:pr_idx+30 and pr_idx-30:pr_idx-10 (inclusive)
        # Python slices:  [pr_idx+10 : pr_idx+30+1], [pr_idx-30 : pr_idx-10+1]
        def _safe_mean_slice(arr: np.ndarray, start: int, stop: int) -> float:
            start = max(start, 0)
            stop = min(stop, arr.size)
            if start >= stop:
                return np.nan
            return float(np.nanmean(arr[start:stop]))

        mean_plus = _safe_mean_slice(dat_spec, pr_idx + 10, pr_idx + 31)
        mean_minus = _safe_mean_slice(dat_spec, pr_idx - 30, pr_idx - 9)
        spec_bl = 0.5 * (mean_plus + mean_minus)

        # Probe heights (amp and squeezed)
        pr_slice_start = max(pr_idx - 30, 0)
        pr_slice_end = min(pr_idx + 31, dat_spec.size)
        pr_height = float(np.nanmax(dat_spec[pr_slice_start:pr_slice_end]))
        pr_height_sqz = float(np.nanmax(dat_spec_sq[pr_slice_start:pr_slice_end]))

        # IF dip minimum
        dip_height = float(np.nanmin(dat_spec[i1_dip: i2_dip + 1]))

        # IF band powers / mean
        band_sq = dat_spec_sq[i1_win: i2_win + 1]
        band = dat_spec[i1_win: i2_win + 1]

        sum_power_in_IF_sq = float(np.nansum(band_sq))
        sum_power_in_IF = float(np.nansum(band))

        # mean_of_spec over [IFwinidx(1):IFwinidx(2)+quick_pad-1]
        mean_end = min(i2_win + quick_pad, dat_spec.size)
        mean_of_spec = float(np.nanmean(dat_spec[i1_win:mean_end]))

        # --- pr_power from tone_power_PR_fordata (db2pow) ---
        meanavgps = psadata.get("meanavgps", None)
        if isinstance(meanavgps, np.ndarray):
            meanavgps = meanavgps.item()

        # Default pr_power = db2pow(-45 dB) if field is missing
        try:
            tone_power_PR_fordata = float(np.squeeze(psadata["tone_power_PR_fordata"]))
        except Exception:
            tone_power_PR_fordata = -45.0
        pr_power = float(10.0 ** (tone_power_PR_fordata / 10.0))

        # --- pr_power_stds from meanavgps.pt_power_est_list ---
        pr_power_stds = np.nan
        if meanavgps is not None and hasattr(meanavgps, "pt_power_est_list"):
            pt_list = getattr(meanavgps, "pt_power_est_list")

            # pt_list may be MATLAB cell array -> numpy object array
            arr = np.asarray(pt_list)
            if arr.dtype == object:
                arr = np.array(arr.tolist(), dtype=float).ravel()
            else:
                arr = arr.astype(float).ravel()

            if arr.size > 0 and pr_power > 0 and GA > 0 and probe_scale > 0:
                raw_std = float(np.nanstd(arr, ddof=0))
                # MATLAB:
                # scaleinfo.pr_power_stds(i) = std(...)./sqrt(pr_power)./GA./probe_scale;
                pr_power_stds = raw_std / np.sqrt(pr_power) / GA / probe_scale

        # sigma_exp and fitnumber we'll compute outside this helper (where proc_par.binavg, sg_win_Hz live)

        return {
            "spec_bl": spec_bl,
            "pr_height": pr_height,
            "pr_height_sqz": pr_height_sqz,
            "dip_height": dip_height,
            "sum_power_in_IF_sq": sum_power_in_IF_sq,
            "sum_power_in_IF": sum_power_in_IF,
            "mean_of_spec": mean_of_spec,
            "pr_power": pr_power,
            "pr_power_stds": pr_power_stds,
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
