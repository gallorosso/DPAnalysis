"""
Reflection analysis stage for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List
import scipy.io
import numpy as np
import warnings
import math

from ..base import PipelineStage, PipelineContext
from ..results import ReflectionAnalysisResult
from src.dark_photon.fitting import optimized_fit

class ReflectionAnalysisStage(PipelineStage):
    """
    Stage 3.2: Analyze cavity reflection data.
    
    Python implementation of readoutbeta (analysis only, no plotting).
    """
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute reflection analysis stage.
        """
        print("  Analyzing cavity reflection...")
        
        # Get file enumeration results
        file_enum = data.get('file_enumeration')
        if not file_enum or not file_enum.files:
            raise ValueError("No files found from file enumeration stage")
        
        files = file_enum.files
        num_files = len(files)
        
        # Initialize result arrays
        beta_values = np.zeros((num_files, 3))  # beta1, beta2, beta_avg
        freq_values = np.zeros((num_files, 3))   # freq1, freq2, freq_avg  
        mse_values = np.zeros((num_files, 3))    # mse1, mse2, mse_avg
        freq_drifts_khz = np.zeros(num_files)    # frequency drifts
        rfl_fit_params = np.zeros((num_files, 10))  # combined fit parameters
        rfl_baseline_db = np.zeros((num_files, 2))  # baseline values in dB
        
        processed_count = 0
        
        for i, tx2_file in enumerate(files):
            try:
                # Get corresponding rfl files
                rfl1_file, rfl2_file = self._get_rfl_files(tx2_file)
                
                if not rfl1_file.exists() or not rfl2_file.exists():
                    warnings.warn(f"RFL files not found for {tx2_file.name}")
                    continue
                
                # Process both reflection sweeps
                beta1, freq1, mse1, baseline1_db, params1 = self._process_reflection_sweep(
                    rfl1_file, 'rfl', context.run_props.processing
                )
                beta2, freq2, mse2, baseline2_db, params2 = self._process_reflection_sweep(
                    rfl2_file, 'rfl', context.run_props.processing
                )
                
                # Calculate averages
                beta_avg = (beta1 + beta2) / 2
                freq_avg = (freq1 + freq2) / 2
                mse_avg = (mse1 + mse2) / 2
                
                # Store results
                beta_values[i, :] = [beta1, beta2, beta_avg]
                freq_values[i, :] = [freq1, freq2, freq_avg]
                mse_values[i, :] = [mse1, mse2, mse_avg]
                freq_drifts_khz[i] = (freq1 - freq2) * 1e6  # GHz to kHz
                rfl_fit_params[i, :] = np.concatenate([params1, params2])
                rfl_baseline_db[i, :] = [baseline1_db, baseline2_db]
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"    Processed {processed_count}/{num_files} files")
                    
            except Exception as e:
                warnings.warn(f"Error processing reflection file {tx2_file}: {e}")
                continue
        
        # Prepare scaleinfo updates
        scaleinfo_updates = {
            'rfldriftkHz': freq_drifts_khz.tolist(),
            'freq_beta': np.column_stack([freq_values[:, 2], beta_values[:, 2]]).tolist(),
            'rfl_base1_db': rfl_baseline_db[:, 0].tolist(),
            'rfl_base2_db': rfl_baseline_db[:, 1].tolist()
        }
        
        # Add reflection parameters based on averaging type (using 'average' as default)
        avg_type = 'average'  # This could be made configurable
        if avg_type == 'average':
            rfl_params_avg = (rfl_fit_params[:, :5] + rfl_fit_params[:, 5:]) / 2
            scaleinfo_updates['rflparams'] = rfl_params_avg.tolist()
        elif avg_type == 'before':
            scaleinfo_updates['rflparams'] = rfl_fit_params[:, :5].tolist()
        elif avg_type == 'after':
            scaleinfo_updates['rflparams'] = rfl_fit_params[:, 5:].tolist()
        
        result = ReflectionAnalysisResult(
            scaleinfo_updates=scaleinfo_updates,
            mse_values=mse_values,
            coupling_factors=beta_values,
            reflection_frequencies=freq_values,
            status="success"
        )
        
        data['reflection_analysis'] = result
        return data
    
    def _get_rfl_files(self, tx2_file: Path) -> tuple[Path, Path]:
        """Get corresponding rfl1 and rfl2 files from tx2 file path."""
        base_name = tx2_file.name.replace('tx2', '')
        rfl1_file = tx2_file.parent / f"{base_name}rfl.mat"
        rfl2_file = tx2_file.parent / f"{base_name}rfl2.mat"
        return rfl1_file, rfl2_file
    
    def _process_reflection_sweep(self, file_path: Path, sweep_type: str, 
                                proc_par: Any) -> tuple:
        """
        Process a single reflection sweep and calculate coupling factor beta.
        
        Returns:
            tuple: (beta, resonance_freq, mse, baseline_db, fit_params)
        """
        # Load data file
        data = scipy.io.loadmat(str(file_path))
        
        # Extract I/Q data and frequencies
        if sweep_type == 'rfl':
            i_data = data['I_rfl'].flatten()
            q_data = data['Q_rfl'].flatten() 
            freq = data['f_GHz_rfl'].flatten()
        else:  # 'rfl2'
            i_data = data['I_rfl2'].flatten()
            q_data = data['Q_rfl2'].flatten()
            freq = data['f_GHz_rfl2'].flatten()
        
        # Get processing parameters from configuration - NO DEFAULTS
        fitting_config = proc_par.fitting
        
        # Convert to dict for fitting function
        proc_par_dict = {
            'use_smart_init': fitting_config['use_smart_init'],
            'tx_fit_width_sigma': fitting_config['tx_fit_width_sigma'],
            'tx_fit_buffer_bins': fitting_config['tx_fit_buffer_bins'],
            'rfl_fit_width_sigma': fitting_config['rfl_fit_width_sigma'],
            'rfl_fit_buffer_bins': fitting_config['rfl_fit_buffer_bins'],
            'init_params_tx': fitting_config['init_params_tx'],
            'init_params_rfl': fitting_config['init_params_rfl']
        }
        
        # Fit reflection data
        fit_params, mse, datarange = optimized_fit(
            'rfl', i_data, q_data, freq, proc_par_dict
        )
        
        # Calculate coupling factor beta
        beta, baseline_db = self._calculate_coupling_factor(fit_params, freq, datarange)
        
        resonance_freq = fit_params[1]  # f0 parameter
        
        return beta, resonance_freq, mse, baseline_db, fit_params
    
    def _calculate_coupling_factor(self, fit_params: np.ndarray, freq: np.ndarray, 
                                 datarange: slice) -> tuple[float, float]:
        """
        Calculate coupling factor beta from reflection fit.
        
        Equivalent to MATLAB's beta calculation:
        gamma = sqrt(10^((min_dB - baseline_dB)/10))
        beta = 2/(1-gamma) - 1
        """
        # Extract parameters
        P_min, f0, Q, slope, offset = fit_params
        
        # Lorentzian function
        def lorentzian(f, P_max, f0, Q, slope, offset):
            return P_max / (1 + 4 * Q**2 * ((f / f0) - 1)**2) + slope * f + offset
        
        # Evaluate fit around resonance to find minimum
        freq_fine = np.linspace(freq[datarange].min(), freq[datarange].max(), 400)
        amp_fine = lorentzian(freq_fine, P_min, f0, Q, slope, offset)
        
        # Find minimum in dB scale
        min_dB = 10 * math.log10(np.min(amp_fine))
        
        # Calculate baseline at ±200 MHz from resonance (as in MATLAB)
        baseline_freq1 = f0 - 0.2  # GHz
        baseline_freq2 = f0 + 0.2  # GHz
        baseline_val1 = lorentzian(baseline_freq1, P_min, f0, Q, slope, offset)
        baseline_val2 = lorentzian(baseline_freq2, P_min, f0, Q, slope, offset)
        baseline_avg = (baseline_val1 + baseline_val2) / 2
        baseline_db = 10 * math.log10(baseline_avg)
        
        # Calculate gamma and beta
        gamma = math.sqrt(10**((min_dB - baseline_db) / 10))
        beta = 2 / (1 - gamma) - 1
        
        return beta, baseline_db
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate reflection analysis outputs.
        """
        result = data.get('reflection_analysis')
        if not result:
            print("  ✗ No reflection analysis result found")
            return False
        
        if not result.scaleinfo_updates:
            print("  ✗ No scaleinfo updates from reflection analysis")
            return False
        
        num_files = len(result.coupling_factors)
        print(f"  ✓ Reflection analysis processed {num_files} files")
        return True