"""
JPA gain analysis stage for the analysis pipeline.
Note: unlike MATLAB code, we don't give rfl_corr_idx
to scaleinfo. Potentially add to excute:
rfl_corr_idx = compute_rfl_corr_idx(scaleinfo)
scaleinfo["rfl_corr_idx"] = rfl_corr_idx.tolist()
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import scipy.io
import numpy as np
import warnings
from datetime import datetime

from ..base import PipelineStage, PipelineContext
from ..results import JPAGainAnalysisResult
from src.dark_photon.fitting import optimized_fit_jpa
from src.dark_photon.utils.caching import get_fit_cache_path, load_cached_fit, save_cached_fit


class JPAGainAnalysisStage(PipelineStage):
    """
    Stage 3.3: Analyze JPA gain profiles.
    
    Python implementation of JPAgainAutorun (analysis only, no plotting).
    """
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute JPA gain analysis stage.
        """
        print("  Analyzing JPA gain profiles...")
        
        # Get scaleinfo from previous stages
        scaleinfo = data.get("scaleinfo")
        if not scaleinfo:
            raise ValueError("No scaleinfo found from previous stages")
        
        # Get file enumeration results
        file_enum = data.get("file_enumeration")
        if not file_enum or not file_enum.files:
            raise ValueError("No files found from file enumeration stage")
        
        files = file_enum.files
        num_files = len(files)
        
        # Initialize result arrays (mirroring MATLAB JPAgainAutorun outputs)
        jpa_mse = np.zeros(num_files)
        bandwidth = np.zeros(num_files)
        q2_gain = np.zeros(num_files)
        
        gain2Q_amp_dB_fit = np.zeros(num_files)
        gain2Q_sqz_dB_fit = np.zeros(num_files)
        gain1Q_amp_dB = np.zeros(num_files)
        gain1Q_sqz_dB = np.zeros(num_files)
        gain1Q_amp_dB2 = np.zeros(num_files)
        gain1Q_sqz_dB2 = np.zeros(num_files)
        
        gain2Q_amp_dB_fit_corr = np.zeros(num_files)
        gain2Q_amp2_dB_fit_corr = np.zeros(num_files)
        gain2Q_sqz_dB_fit_corr = np.zeros(num_files)
        gain2Q_sqz2_dB_fit_corr = np.zeros(num_files)
        
        amp_gain_fit = np.zeros((num_files, 5))
        sqz_gain_fit = np.zeros((num_files, 5))
        
        # Calculate JPA cut window (equivalent to MATLAB calculation)
        cav_bw_ghz = self._calculate_cavity_bandwidth(scaleinfo)
        cut_window_ghz = cav_bw_ghz
        print(f"    JPA cut window: {cut_window_ghz:.6f} GHz")
        
        processed_count = 0
        
        for i, tx2_file in enumerate(files):
            try:
                # Process this JPA dataset
                file_base = self._get_file_base(tx2_file)
                jpa_results = self._process_jpa_dataset(
                    file_base, scaleinfo, context.run_props.processing, context, i
                )
                
                if jpa_results:
                    # Store results
                    jpa_mse[i] = jpa_results['mse']
                    bandwidth[i] = jpa_results['bandwidth']
                    q2_gain[i] = jpa_results['q2_gain']
                    gain2Q_amp_dB_fit[i] = jpa_results['gain2Q_amp_dB_fit']
                    gain2Q_sqz_dB_fit[i] = jpa_results['gain2Q_sqz_dB_fit']
                    gain1Q_amp_dB[i] = jpa_results['gain1Q_amp_dB']
                    gain1Q_sqz_dB[i] = jpa_results['gain1Q_sqz_dB']
                    gain1Q_amp_dB2[i] = jpa_results['gain1Q_amp_dB2']
                    gain1Q_sqz_dB2[i] = jpa_results['gain1Q_sqz_dB2']
                    gain2Q_amp_dB_fit_corr[i] = jpa_results['gain2Q_amp_dB_fit_corr']
                    gain2Q_amp2_dB_fit_corr[i] = jpa_results['gain2Q_amp2_dB_fit_corr']
                    gain2Q_sqz_dB_fit_corr[i] = jpa_results['gain2Q_sqz_dB_fit_corr']
                    gain2Q_sqz2_dB_fit_corr[i] = jpa_results['gain2Q_sqz2_dB_fit_corr']
                    amp_gain_fit[i, :] = jpa_results['amp_gain_fit']
                    sqz_gain_fit[i, :] = jpa_results['sqz_gain_fit']
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"    Processed {processed_count}/{num_files} JPA datasets")
                        
            except Exception as e:
                warnings.warn(f"Error processing JPA data for {tx2_file}: {e}")
                continue
        
        # Prepare scaleinfo updates
        scaleinfo_updates = {
            'JPA_cut_window_GHz': cut_window_ghz,
            'JPA_mse': jpa_mse.tolist(),
            'JPAbandwidth': bandwidth.tolist(),
            'Q2gain': q2_gain.tolist(),
            'amp_gain_fit': amp_gain_fit.tolist(),
            'sqz_gain_fit': sqz_gain_fit.tolist(),
            'gain1Q_amp_dB': gain1Q_amp_dB.tolist(),
            'gain1Q_sqz_dB': gain1Q_sqz_dB.tolist(),
            'gain1Q_amp_dB2': gain1Q_amp_dB2.tolist(),
            'gain1Q_sqz_dB2': gain1Q_sqz_dB2.tolist(),
            'gain2Q_amp_dB_fit': gain2Q_amp_dB_fit.tolist(),
            'gain2Q_sqz_dB_fit': gain2Q_sqz_dB_fit.tolist(),
            'gain2Q_amp_dB_fit_corr': gain2Q_amp_dB_fit_corr.tolist(),
            'gain2Q_amp2_dB_fit_corr': gain2Q_amp2_dB_fit_corr.tolist(),
            'gain2Q_sqz_dB_fit_corr': gain2Q_sqz_dB_fit_corr.tolist(),
            'gain2Q_sqz2_dB_fit_corr': gain2Q_sqz2_dB_fit_corr.tolist(),
        }
        
        result = JPAGainAnalysisResult(
            scaleinfo_updates=scaleinfo_updates,
            jpa_mse=jpa_mse,
            jpa_bandwidth=bandwidth,
            q2gain=q2_gain,
            status="success"
        )
        
        data['jpa_analysis'] = result
        return data
    
    def _calculate_cavity_bandwidth(self, scaleinfo: Dict[str, Any]) -> float:
        """
        Calculate cavity bandwidth for JPA cut window.
        
        Equivalent to MATLAB: cav_BW_GHz = (mean(scaleinfo.txparams(:,2)))/mean(scaleinfo.txparams(:,3))
        """
        txparams = np.array(scaleinfo.get('txparams', []))
        if len(txparams) == 0:
            return 0.01  # Default fallback
        
        # txparams[:, 1] is f0 (GHz), txparams[:, 2] is Q
        mean_f0 = np.mean(txparams[:, 1])
        mean_Q = np.mean(txparams[:, 2])
        
        # Bandwidth = f0 / Q
        cav_bw_ghz = mean_f0 / mean_Q
        return float(cav_bw_ghz)
    
    def _get_file_base(self, tx2_file: Path) -> Path:
        """
        Extract file base name from tx2 file path.
        
        Converts '/path/to/20220908_0_0_tx2.mat' -> '/path/to/20220908_0_0_'
        """
        filename = tx2_file.name.replace('tx2.mat', '')
        return tx2_file.parent / filename
    
    def _load_jpa_data_files(self, file_base: Path) -> Tuple[Dict, Dict, bool]:
        """
        Load JPA data files (.jpaamp.mat, .jpaamp2.mat).
        
        Returns:
            Tuple: (data, data2, has_jpa2)
        """
        jpaamp_file = Path(f"{file_base}jpaamp.mat")
        jpaamp2_file = Path(f"{file_base}jpaamp2.mat")
        
        # Load primary JPA data
        if not jpaamp_file.exists():
            raise FileNotFoundError(f"JPA data file not found: {jpaamp_file}")
        
        data = scipy.io.loadmat(str(jpaamp_file))
        
        # Load secondary JPA data if available
        has_jpa2 = False
        data2 = {}
        if jpaamp2_file.exists():
            try:
                data2 = scipy.io.loadmat(str(jpaamp2_file))
                has_jpa2 = True
            except Exception as e:
                warnings.warn(f"Could not load secondary JPA data {jpaamp2_file}: {e}")
        
        return data, data2, has_jpa2
    
    def _get_proc_par_dict(self, proc_par: Any) -> Dict[str, Any]:
        """Extract JPA-specific processing parameters."""
        fitting_config = proc_par.fitting
        
        return {
            'jpa_fit_width_sigma': fitting_config.get('jpa_fit_width_sigma', 5.0),
            'jpa_fit_buffer_bins': fitting_config.get('jpa_fit_buffer_bins', 5),
            'JPA_gbw_prod': fitting_config.get('JPA_gbw_prod', 8.15e7),
            'load_fits': fitting_config.get('load_fits', True),
        }
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate JPA analysis outputs.
        """
        result = data.get('jpa_analysis')
        if not result:
            print("  ✗ No JPA analysis result found")
            return False
        
        if not result.scaleinfo_updates:
            print("  ✗ No scaleinfo updates from JPA analysis")
            return False
        
        required_fields = [
            'JPA_cut_window_GHz', 'JPA_mse', 'JPAbandwidth', 'Q2gain',
            'amp_gain_fit', 'gain2Q_amp_dB_fit'
        ]
        
        missing_fields = [f for f in required_fields if f not in result.scaleinfo_updates]
        if missing_fields:
            print(f"  ✗ JPA scaleinfo missing fields: {missing_fields}")
            return False
        
        num_results = len(result.scaleinfo_updates['JPA_mse'])
        print(f"  ✓ JPA analysis processed {num_results} datasets")
        return True

    def _process_jpa_dataset(self, file_base: Path, scaleinfo: Dict[str, Any], 
                           proc_par: Any, context: PipelineContext, file_index: int) -> Dict[str, Any]:
        """
        Process a single JPA dataset (equivalent to one iteration in JPAgainAutorun loop).
        
        Args:
            file_base: Base file path without extension
            scaleinfo: Current scaleinfo dictionary
            proc_par: Processing parameters
            context: Pipeline context
            file_index: Index of current file in the files list
            
        Returns:
            Dictionary with JPA analysis results for this dataset
        """
        try:
            # Load JPA data files
            data, data2, has_jpa2 = self._load_jpa_data_files(file_base)
            
            # Get processing parameters
            proc_par_dict = self._get_proc_par_dict(proc_par)
            cut_window_ghz = scaleinfo.get('JPA_cut_window_GHz', 0.01)
            
            # Ensure frequency arrays are row vectors (like MATLAB's iscolumn check)
            if 'f_GHz_jpaamp' in data and data['f_GHz_jpaamp'].shape[0] > 1:
                data['f_GHz_jpaamp'] = data['f_GHz_jpaamp'].T
            if has_jpa2 and 'f_GHz_jpaamp2' in data2 and data2['f_GHz_jpaamp2'].shape[0] > 1:
                data2['f_GHz_jpaamp2'] = data2['f_GHz_jpaamp2'].T
            
            # Process with caching
            cache_dir = context.output_dir / 'fits' / 'jpa'
            jpa_results = self._process_jpa_measurement_with_cache(
                data, data2, has_jpa2, file_base, scaleinfo, proc_par_dict, 
                cut_window_ghz, file_index, cache_dir
            )
            
            return jpa_results
            
        except Exception as e:
            warnings.warn(f"Error in JPA dataset processing for {file_base}: {e}")
            # Return default values for failed processing
            return self._get_default_jpa_results()
    
    def _process_jpa_measurement_with_cache(self, data: Dict, data2: Dict, has_jpa2: bool,
                                          file_base: Path, scaleinfo: Dict[str, Any],
                                          proc_par: Dict[str, Any], cut_window_ghz: float,
                                          file_index: int, cache_dir: Path) -> Dict[str, Any]:
        """
        Process JPA measurement with caching support.
        """
        # Generate cache file path
        cache_file = cache_dir / f"{file_base.name}_jpa_fit.pkl"
        
        # Try to load from cache
        if proc_par.get('load_fits', True):
            cached_data = load_cached_fit(cache_file, proc_par)
            if cached_data:
                print(f"    Loaded cached JPA fit: {file_base.name}")
                return cached_data['jpa_results']
        
        # Process normally if cache miss
        jpa_results = self._process_jpa_measurement(
            data, data2, has_jpa2, scaleinfo, proc_par, cut_window_ghz, file_index
        )
        
        # Save to cache
        cache_data = {
            'jpa_results': jpa_results,
            'jpa_fit_width_sigma': proc_par.get('jpa_fit_width_sigma'),
            'timestamp': datetime.now().isoformat()
        }
        save_cached_fit(cache_file, cache_data)
        
        return jpa_results
    
    def _process_jpa_measurement(self, data: Dict, data2: Dict, has_jpa2: bool,
                               scaleinfo: Dict[str, Any], proc_par: Dict[str, Any],
                               cut_window_ghz: float, file_index: int) -> Dict[str, Any]:
        """
        Core JPA measurement processing (equivalent to main loop body in JPAgainAutorun).
        """
        # Extract reflection parameters for this file
        rfl_corr_idx = self.compute_rfl_corr_idx(scaleinfo)
        rfl_params = self._get_reflection_params(scaleinfo, rfl_corr_idx, file_index)
        
        # Fit primary JPA amplifier data
        amp_fit_params, mse, amp_datarange = self._fit_jpa_profile(
            data['I_jpaamp'], data['Q_jpaamp'], data['f_GHz_jpaamp'], 
            proc_par, cut_window_ghz
        )
        
        # Calculate bandwidth and basic gains
        bandwidth = self._calculate_bandwidth(amp_fit_params)
        q2_gain = self._calculate_q2_gain(bandwidth, proc_par.get('JPA_gbw_prod', 8.15e7))
        
        # Fit squeezer data if available
        sqz_fit_params = np.zeros(5)
        try:
            sqz_fit_params, _, _ = self._fit_jpa_profile(
                data['I_jpasqz'], data['Q_jpasqz'], data['f_GHz_jpasqz'],
                proc_par, cut_window_ghz
            )
        except Exception:
            # Squeezer fitting failed, use zeros
            pass
        
        # Fit secondary measurements if available
        amp2_fit_params = np.zeros(5)
        sqz2_fit_params = np.zeros(5)
        if has_jpa2:
            try:
                amp2_fit_params, _, _ = self._fit_jpa_profile(
                    data2['I_jpaamp2'], data2['Q_jpaamp2'], data2['f_GHz_jpaamp2'],
                    proc_par, cut_window_ghz
                )
            except Exception:
                pass
            
            try:
                sqz2_fit_params, _, _ = self._fit_jpa_profile(
                    data2['I_jpasqz2'], data2['Q_jpasqz2'], data2['f_GHz_jpasqz2'],
                    proc_par, cut_window_ghz
                )
            except Exception:
                pass

        # Calculate various gain measurements
        gain_results = self._calculate_gain_measurements(
            data, data2, has_jpa2, amp_fit_params, sqz_fit_params, amp2_fit_params, sqz2_fit_params)
        
        # Calculate corrected gains with reflection correction
        corrected_gains = self._calculate_corrected_gains(
            amp_fit_params, amp2_fit_params, sqz_fit_params, sqz2_fit_params,
            rfl_params, data, data2, has_jpa2
        )
        
        # Compile final results
        return {
            'mse': float(mse),
            'bandwidth': float(bandwidth),
            'q2_gain': float(q2_gain),
            'gain2Q_amp_dB_fit': gain_results['gain2Q_amp_dB_fit'],
            'gain2Q_sqz_dB_fit': gain_results['gain2Q_sqz_dB_fit'],
            'gain1Q_amp_dB': gain_results['gain1Q_amp_dB'],
            'gain1Q_sqz_dB': gain_results['gain1Q_sqz_dB'],
            'gain1Q_amp_dB2': gain_results['gain1Q_amp_dB2'],
            'gain1Q_sqz_dB2': gain_results['gain1Q_sqz_dB2'],
            'gain2Q_amp_dB_fit_corr': corrected_gains['gain2Q_amp_dB_fit_corr'],
            'gain2Q_amp2_dB_fit_corr': corrected_gains['gain2Q_amp2_dB_fit_corr'],
            'gain2Q_sqz_dB_fit_corr': corrected_gains['gain2Q_sqz_dB_fit_corr'],
            'gain2Q_sqz2_dB_fit_corr': corrected_gains['gain2Q_sqz2_dB_fit_corr'],
            'amp_gain_fit': amp_fit_params,
            'sqz_gain_fit': sqz_fit_params,
        }
    
    def compute_rfl_corr_idx(self, scaleinfo: Dict[str, Any]) -> np.ndarray:
        """
        Build rfl_corr_idx exactly in the MATLAB spirit:
        for each TM010 frequency (txparams[:,1]) pick the reflection
        entry whose freq_beta[:,0] is closest.

        Returns:
            rfl_corr_idx: integer array of shape (N,), 0-based indices into rflparams.
        """
        txparams = np.asarray(scaleinfo["txparams"], dtype=float)      # (N, 5)
        freq_beta = np.asarray(scaleinfo["freq_beta"], dtype=float)    # (N_rfl, 2)

        f_tx = txparams[:, 1]      # TM010 frequencies [GHz]
        f_rfl = freq_beta[:, 0]    # Reflection frequencies [GHz]

        rfl_corr_idx = np.zeros(len(f_tx), dtype=int)

        max_df_ghz = 5e-4  # example: 500 kHz tolerance
        for k, f in enumerate(f_tx):
            if np.isnan(f):
                rfl_corr_idx[k] = -1
                continue

            df = np.abs(f_rfl - f)
            i_best = int(np.argmin(df))
            if df[i_best] > max_df_ghz:
                rfl_corr_idx[k] = -1  # "no good reflection match"
            else:
                rfl_corr_idx[k] = i_best

                return rfl_corr_idx
    
    def _get_reflection_params(
        self,
        scaleinfo: Dict[str, Any],
        rfl_corr_idx: np.ndarray,
        file_index: int,
        ) -> Optional[np.ndarray]:
        """
        Select the reflection fit parameters for the current JPA dataset,
        using the rfl_corr_idx mapping (MATLAB's rfl_corr_idx logic).
        """
        rflparams = np.asarray(scaleinfo["rflparams"], dtype=float)  # (N_rfl, 5)
        
        idx = int(rfl_corr_idx[file_index])
        if idx < 0 or idx >= rflparams.shape[0]:
            # No valid mapping; caller can decide to skip reflection correction
            return None
        
        return rflparams[idx, :]

    
    def _fit_jpa_profile(self, i_data: np.ndarray, q_data: np.ndarray, freq: np.ndarray,
                        proc_par: Dict[str, Any], cut_window_ghz: float) -> Tuple[np.ndarray, float, tuple]:
        """
        Fit JPA profile using optimized_fit_jpa.
        """
        # Flatten arrays for consistency
        i_flat = i_data.flatten()
        q_flat = q_data.flatten()
        freq_flat = freq.flatten()
        
        # Use the existing optimized_fit_jpa function
        bestfit_params, mse, datarange = optimized_fit_jpa(
            i_flat, q_flat, freq_flat, proc_par
        )
        
        return bestfit_params, mse, datarange
    
    def _calculate_bandwidth(self, fit_params: np.ndarray) -> float:
        """
        Calculate JPA bandwidth from fit parameters.
        
        Equivalent to MATLAB: bw2 = bestfitparams(2)*1e9/bestfitparams(3)
        bandwidth(i) = abs(bw2)
        """
        if len(fit_params) < 3:
            return 0.0
        
        f0_ghz = fit_params[1]  # Resonance frequency in GHz
        Q = fit_params[2]       # Quality factor
        
        if Q == 0:
            return 0.0
        
        # Bandwidth = f0 / Q (convert GHz to Hz)
        bandwidth_hz = (f0_ghz * 1e9) / abs(Q)
        return float(bandwidth_hz)
    
    def _calculate_q2_gain(self, bandwidth: float, jpa_gbw_prod: float) -> float:
        """
        Calculate Q2 gain from bandwidth.
        
        Equivalent to MATLAB: Q2_gain(i) = (proc_par.JPA_gbw_prod./(abs(bw2))).^2
        """
        if bandwidth == 0:
            return 0.0
        
        q2_gain = (jpa_gbw_prod / abs(bandwidth)) ** 2
        return float(q2_gain)
    
    def _calculate_gain_measurements(self, data: Dict, data2: Dict, has_jpa2: bool,
                                  amp_fit_params: np.ndarray, sqz_fit_params: np.ndarray,
                                  amp2_fit_params: np.ndarray, sqz2_fit_params: np.ndarray) -> Dict[str, float]:
        """
        Calculate various gain measurements from JPA data.
        
        Implements the multiple gain calculation methods from JPAgainAutorun.
        """
        results = {}
        
        # Extract 1Q gain measurements (direct from data)
        results['gain1Q_amp_dB'] = self._extract_1q_gain(data, 'gain_amp_pow')
        results['gain1Q_sqz_dB'] = self._extract_1q_gain(data, 'gain_sq_pow')
        if has_jpa2:
            results['gain1Q_amp_dB2'] = self._extract_1q_gain(data2, 'gain_amp_pow2')
            results['gain1Q_sqz_dB2'] = self._extract_1q_gain(data2, 'gain_sq_pow2')
        else:
            results['gain1Q_amp_dB2'] = 0.0
            results['gain1Q_sqz_dB2'] = 0.0
        
        # Calculate 2Q gain from fits (amplifier)
        primary_amp_2q = self._calculate_2q_gain_from_fit(
            data, amp_fit_params, 'I_jpaamp', 'Q_jpaamp'
        )
        
        secondary_amp_2q = 0.0
        if has_jpa2:
            secondary_amp_2q = self._calculate_2q_gain_from_fit(
                data2, amp2_fit_params, 'I_jpaamp2', 'Q_jpaamp2'
            )
        results['gain2Q_amp_dB_fit'] = max(primary_amp_2q, secondary_amp_2q)

        # Calculate 2Q gain from fits (squeezer)
        primary_sqz_2q = self._calculate_2q_gain_from_fit(
            data, sqz_fit_params, 'I_jpasqz', 'Q_jpasqz', is_sqz=True)
        secondary_sqz_2q = 0.0
        if has_jpa2:
            secondary_sqz_2q = self._calculate_2q_gain_from_fit(
                data2, sqz2_fit_params, 'I_jpasqz2', 'Q_jpasqz2', is_sqz=True
            )
        results['gain2Q_sqz_dB_fit'] = max(primary_sqz_2q, secondary_sqz_2q)
        
        return results

    def _extract_1q_gain(self, data, key):
        if key not in data:
            return 0.0
        val = float(np.squeeze(data[key]))
        return 10*np.log10(val) if val > 0 else 0.0
    
    def _calculate_2q_gain_from_fit(self, data: Dict, fit_params: np.ndarray, 
                                  i_key: str, q_key: str, is_sqz: bool = False) -> float:
        """
        Calculate 2Q gain from fit parameters.
        
        Equivalent to MATLAB gain calculation with baseline subtraction.
        """
        if i_key not in data or q_key not in data:
            return 0.0
        
        try:
            # Extract I/Q data
            i_data = data[i_key].flatten()
            q_data = data[q_key].flatten()
            
            # Calculate baseline from endpoints (like MATLAB)
            n_points = len(i_data)
            if n_points < 20:
                return 0.0
                
            # Use first and last 10 points for baseline (like MATLAB)
            baseline_start = np.mean(i_data[:10]**2 + q_data[:10]**2)
            baseline_end = np.mean(i_data[-10:]**2 + q_data[-10:]**2)
            baseline_power = (baseline_start + baseline_end) / 2.0
            
            # Calculate peak gain from fit at resonance
            resonance_gain = self._lorentzian_function(fit_params[1], fit_params)
            
            # Convert to dB and subtract baseline
            peak_dB = 10.0 * np.log10(resonance_gain)
            baseline_dB = 10.0 * np.log10(baseline_power)
            
            gain_2q = peak_dB - baseline_dB
            return max(0.0, gain_2q)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def _lorentzian_function(self, f: float, params: np.ndarray) -> float:
        """
        Evaluate Lorentzian + linear function at given frequency.
        
        Equivalent to MATLAB: fun = @(a, xdata)a(1)./(4*(a(3).^2)*((xdata/a(2))-1).^2+1)+a(4).*xdata+a(5)
        """
        if len(params) < 5:
            return 0.0
            
        P_max, f0, Q, slope, offset = params
        
        # Handle division by zero
        if Q == 0:
            return 0.0
            
        # Lorentzian term
        lorentzian = P_max / (1 + 4 * Q**2 * ((f / f0) - 1)**2)
        
        # Linear baseline
        linear = slope * f + offset
        
        return lorentzian + linear
    
    def _calculate_corrected_gains(self, amp_fit_params: np.ndarray, amp2_fit_params: np.ndarray,
                                 sqz_fit_params: np.ndarray, sqz2_fit_params: np.ndarray,
                                 rfl_params: np.ndarray, data: Dict, data2: Dict, 
                                 has_jpa2: bool) -> Dict[str, float]:
        """
        Calculate reflection-corrected gain measurements.
        
        Implements the complex reflection correction logic from JPAgainAutorun.
        """
        corrected = {}
        
        # Calculate baseline reflection level
        rfl_base_dB = self._calculate_reflection_baseline(rfl_params)
        
        # Corrected amplifier gains
        corrected['gain2Q_amp_dB_fit_corr'] = self._calculate_corrected_gain(
            amp_fit_params, rfl_params, rfl_base_dB
        )
        
        corrected['gain2Q_amp2_dB_fit_corr'] = self._calculate_corrected_gain(
            amp2_fit_params, rfl_params, rfl_base_dB
        ) if has_jpa2 else 0.0
        
        # Corrected squeezer gains
        corrected['gain2Q_sqz_dB_fit_corr'] = self._calculate_corrected_gain(
            sqz_fit_params, rfl_params, rfl_base_dB
        )
        
        corrected['gain2Q_sqz2_dB_fit_corr'] = self._calculate_corrected_gain(
            sqz2_fit_params, rfl_params, rfl_base_dB
        ) if has_jpa2 else 0.0
        
        return corrected
    
    def _calculate_reflection_baseline(self, rfl_params: np.ndarray) -> float:
        """
        Calculate reflection baseline in dB.
        
        Equivalent to MATLAB: rfl_base_dB = pow2db(scaleinfo.rflparams(i,4)*scaleinfo.rflparams(i,2) + scaleinfo.rflparams(i,5))
        """
        if len(rfl_params) < 5:
            return 0.0
            
        # rfl_params: [P_min, f0, Q, slope, offset]
        slope, f0, offset = rfl_params[3], rfl_params[1], rfl_params[4]
        
        # Calculate baseline: slope * f0 + offset
        baseline_linear = slope * f0 + offset
        
        if baseline_linear <= 0:
            return 0.0
            
        return 10.0 * np.log10(baseline_linear)
    
    def _calculate_corrected_gain(self, fit_params: np.ndarray, rfl_params: np.ndarray,
                                rfl_base_dB: float) -> float:
        """
        Calculate reflection-corrected gain for a single measurement.
        
        Equivalent to MATLAB: gain2Q_amp_dB_fit_corr(i) = pow2db(fun(bestfitparams, bestfitparams(2))) - rfl_base_dB
        """
        if len(fit_params) < 5 or np.all(fit_params == 0):
            return 0.0
            
        # Calculate peak gain at resonance
        resonance_freq = fit_params[1]
        peak_gain = self._lorentzian_function(resonance_freq, fit_params)
        
        if peak_gain <= 0:
            return 0.0
            
        peak_dB = 10.0 * np.log10(peak_gain)
        
        # Apply reflection correction
        corrected_gain = peak_dB - rfl_base_dB
        return max(0.0, corrected_gain)
    
    def _apply_reflection_correction_to_spectrum(self, i_data: np.ndarray, q_data: np.ndarray,
                                               freq: np.ndarray, rfl_params: np.ndarray,
                                               fit_params: np.ndarray, cut_window_ghz: float) -> np.ndarray:
        """
        Apply reflection correction to entire JPA spectrum.
        
        Equivalent to MATLAB reflection correction calculation.
        """
        # Calculate magnitude in dB
        magnitude = i_data**2 + q_data**2
        magnitude_dB = 10.0 * np.log10(magnitude)
        
        # Find resonance region for correction
        resonance_freq = fit_params[1]
        delta_freq = np.abs(freq - resonance_freq)
        correction_region = delta_freq <= cut_window_ghz
        
        if not np.any(correction_region):
            return magnitude_dB
        
        # Calculate reflection correction in the resonance region
        rfl_peak = self._lorentzian_function(freq[correction_region], rfl_params)
        rfl_base = rfl_params[3] * rfl_params[1] + rfl_params[4]
        
        rfl_peak_dB = 10.0 * np.log10(rfl_peak)
        rfl_base_dB = 10.0 * np.log10(rfl_base)
        
        correction = rfl_peak_dB - rfl_base_dB
        
        # Apply correction
        corrected_magnitude = magnitude_dB.copy()
        corrected_magnitude[correction_region] -= correction
        
        return corrected_magnitude
    
    def _get_default_jpa_results(self) -> Dict[str, Any]:
        """Return default values for failed JPA processing."""
        return {
            'mse': 0.0,
            'bandwidth': 0.0,
            'q2_gain': 0.0,
            'gain2Q_amp_dB_fit': 0.0,
            'gain2Q_sqz_dB_fit': 0.0,
            'gain1Q_amp_dB': 0.0,
            'gain1Q_sqz_dB': 0.0,
            'gain2Q_amp_dB_fit_corr': 0.0,
            'gain2Q_amp2_dB_fit_corr': 0.0,
            'gain2Q_sqz_dB_fit_corr': 0.0,
            'gain2Q_sqz2_dB_fit_corr': 0.0,
            'amp_gain_fit': np.zeros(5),
            'sqz_gain_fit': np.zeros(5),
        }