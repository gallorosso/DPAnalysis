# DPAnalysis/src/dark_photon/analysis/pipeline/stages/spectrum_info.py

from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        cutpar = context.run_props.data_quality_cuts  # assuming this exists; adjust name if needed

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
        raise NotImplementedError

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

        MATLAB (inside try/catch):

            sqz_vs_freq_dB = pow2db(psadata.meanavgps_sqzON.singlesided_powerspecavg ...
                                    ./ psadata.meanavgps_sqzOFF.singlesided_powerspecavg);
            avg_sqdB_off = mean(sqz_vs_freq_dB(IFwinidx(2):lowpass_idx));
            peak_sqdB    = mean(sqz_vs_freq_dB(1:IFwinidx(1)));
            avg_sqdB_IF  = mean(sqz_vs_freq_dB(IFwinidx(1):IFwinidx(2)));
            ...
        """
        raise NotImplementedError

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
        Compute baseline, probe heights, IF dip, IF sums, pr_power_stds, etc.

        This collects the block:

            spec_bl        = (mean(dat_spec(pr_idx+10:pr_idx+30))+ ...
                              mean(dat_spec(pr_idx-30:pr_idx-10)))/2;
            pr_height      = max(dat_spec(pr_idx-30:pr_idx+30));
            pr_height_sqz  = max(dat_spec_sq(pr_idx-30:pr_idx+30));
            dip_height     = min(dat_spec(IFdipidx(1):IFdipidx(2)));
            ...
            scaleinfo.sum_power_in_IF_sq(i) = sum(dat_spec_sq(IFwinidx(1):IFwinidx(2)));
            scaleinfo.sum_power_in_IF(i)    = sum(dat_spec(IFwinidx(1):IFwinidx(2)));
            scaleinfo.mean_of_spec(i)       = mean(dat_spec(IFwinidx(1):IFwinidx(2)+ quick_pad -1));
            ...
            scaleinfo.pr_power_stds(i)  = std(psadata.meanavgps.pt_power_est_list);
            ...
        """
        raise NotImplementedError

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
