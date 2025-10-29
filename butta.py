"""
Transmission analysis stage for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import scipy.io
import numpy as np
import warnings
from datetime import datetime

from ..base import PipelineStage, PipelineContext
from ..results import TransmissionAnalysisResult
from src.dark_photon.fitting import optimized_fit
from src.dark_photon.utils.caching import get_fit_cache_path, load_cached_fit, save_cached_fit


class TransmissionAnalysisStage(PipelineStage):
    """
    Stage 3.1: Analyze cavity transmission data.
    
    Python implementation of ReadOutCavityTran (analysis only, no plotting).
    """
    
    def __init__(self, avg_type: Literal['average', 'before', 'after'] = 'average'):
        """
        Initialize transmission analysis stage.
        
        Args:
            avg_type: How to average the two sweeps ('average', 'before', 'after')
        """
        self.avg_type = avg_type
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transmission analysis stage.
        """
        print("  Analyzing cavity transmission...")
        
        # Get file enumeration results
        file_enum = data.get('file_enumeration')
        if not file_enum or not file_enum.files:
            raise ValueError("No files found from file enumeration stage")
        
        files = file_enum.files
        num_files = len(files)
        
        # Initialize result arrays
        params1 = np.zeros((num_files, 5))  # First sweep parameters
        params2 = np.zeros((num_files, 5))  # Second sweep parameters  
        params_avg = np.zeros((num_files, 5))  # Average parameters
        mse_values = np.zeros((num_files, 3))  # MSE for sweep1, sweep2, average
        freq_drifts_khz = np.zeros(num_files)  # Frequency drifts
        fitted_parameters = np.zeros((num_files, 10))  # Combined parameters
        phase_info = np.zeros((num_files, 2))  # Phase mean and std
        
        processed_count = 0
        
        for i, tx2_file in enumerate(files):
            try:
                # Get corresponding tx1 file
                tx1_file = self._get_tx1_file(tx2_file)
                
                if not tx1_file.exists():
                    warnings.warn(f"TX1 file not found for {tx2_file.name}")
                    continue
                
                # Process both sweeps with caching
                params1[i, :], mse1, _, phase_mean, phase_std = self._process_transmission_sweep_with_cache(
                    tx1_file, 'tx', context.run_props.processing, context
                )
                params2[i, :], mse2, _ = self._process_transmission_sweep_with_cache(
                    tx2_file, 'tx2', context.run_props.processing, context
                )
                
                # Store phase information (only from first sweep)
                phase_info[i, :] = [phase_mean, phase_std]
                
                # Calculate averages based on type
                params_avg[i, :] = self._apply_averaging(params1[i, :], params2[i, :])
                
                # Store results
                mse_values[i, :] = [mse1, mse2, (mse1 + mse2) / 2]
                freq_drifts_khz[i] = (params1[i, 1] - params2[i, 1]) * 1e6  # GHz to kHz
                fitted_parameters[i, :] = np.concatenate([params1[i, :], params2[i, :]])
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"    Processed {processed_count}/{num_files} files")
                    
            except Exception as e:
                warnings.warn(f"Error processing transmission file {tx2_file}: {e}")
                continue
        
        # Prepare scaleinfo updates
        scaleinfo_updates = {
            'txparams': params_avg.tolist(),
            'txdriftkHz': freq_drifts_khz.tolist(),
            'phase_info': phase_info.tolist()  # Add phase information
        }
        
        result = TransmissionAnalysisResult(
            scaleinfo_updates=scaleinfo_updates,
            fitted_parameters=fitted_parameters,
            mse_values=mse_values,
            frequency_drifts_khz=freq_drifts_khz,
            status="success"
        )
        
        data['transmission_analysis'] = result
        return data
    
    def _get_tx1_file(self, tx2_file: Path) -> Path:
        """Get corresponding tx1 file from tx2 file path."""
        filename = tx2_file.name.replace('tx2', 'tx')
        return tx2_file.parent / filename
    
    def _process_transmission_sweep_with_cache(self, file_path: Path, sweep_type: str, 
                                             proc_par: Any, context: PipelineContext) -> Tuple:
        """
        Process a single transmission sweep with caching.
        
        Returns:
            tuple: (bestfit_params, mse, datarange, phase_mean, phase_std)
        """
        # Generate cache path
        cache_path = get_fit_cache_path(file_path, 'tx', context.output_dir)
        
        # Try to load from cache
        proc_par_dict = self._get_proc_par_dict(proc_par)
        if proc_par_dict.get('load_fits', True):
            cached_data = load_cached_fit(cache_path, proc_par_dict)
            if cached_data:
                print(f"    Loaded cached fit: {file_path.name}")
                return (cached_data['bestfit_params'],
                        cached_data['mse'],
                        cached_data['datarange'],
                        cached_data.get('phase_mean', 0.0),
                        cached_data.get('phase_std', 0.0))
        
        # Process normally if cache miss
        bestfit_params, mse, datarange = self._process_transmission_sweep(
            file_path, sweep_type, proc_par
        )
        
        # Extract phase information (only for first sweep)
        phase_mean, phase_std = 0.0, 0.0
        if sweep_type == 'tx':
            phase_mean, phase_std = self._extract_phase_info(file_path)
        
        # Save to cache
        cache_data = {
            'bestfit_params': bestfit_params,
            'mse': mse,
            'datarange': datarange,
            'phase_mean': phase_mean,
            'phase_std': phase_std,
            'tx_fit_width_sigma': proc_par_dict.get('tx_fit_width_sigma'),
            'timestamp': datetime.now().isoformat()
        }
        save_cached_fit(cache_path, cache_data)
        
        return bestfit_params, mse, datarange, phase_mean, phase_std
    
    def _process_transmission_sweep(self, file_path: Path, sweep_type: str, 
                              proc_par: Any) -> tuple:
        """
        Process a single transmission sweep.
        """
        # Load data file
        data = scipy.io.loadmat(str(file_path))
        
        # Extract I/Q data and frequencies
        if sweep_type == 'tx':
            i_data = data['I_tx'].flatten()
            q_data = data['Q_tx'].flatten() 
            freq = data['f_GHz_tx'].flatten()
        else:  # 'tx2'
            i_data = data['I_tx2'].flatten()
            q_data = data['Q_tx2'].flatten()
            freq = data['f_GHz_tx2'].flatten()
        
        # Get processing parameters
        proc_par_dict = self._get_proc_par_dict(proc_par)
        
        # Fit transmission data
        bestfit_params, mse, datarange = optimized_fit(
            'tx', i_data, q_data, freq, proc_par_dict
        )
        
        return bestfit_params, mse, datarange
    
    def _get_proc_par_dict(self, proc_par: Any) -> Dict[str, Any]:
        """Convert processing parameters to dictionary for fitting functions."""
        fitting_config = proc_par.fitting
        
        return {
            'use_smart_init': fitting_config['use_smart_init'],
            'tx_fit_width_sigma': fitting_config['tx_fit_width_sigma'],
            'tx_fit_buffer_bins': fitting_config['tx_fit_buffer_bins'],
            'rfl_fit_width_sigma': fitting_config['rfl_fit_width_sigma'],
            'rfl_fit_buffer_bins': fitting_config['rfl_fit_buffer_bins'],
            'init_params_tx': fitting_config['init_params_tx'],
            'init_params_rfl': fitting_config['init_params_rfl'],
            'load_fits': fitting_config.get('load_fits', True)
        }
    
    def _extract_phase_info(self, tx_file_path: Path) -> Tuple[float, float]:
        """
        Extract phase information from CW transmission data.
        """
        try:
            data = scipy.io.loadmat(str(tx_file_path))
            
            # Check if CW data exists
            if 'I_CW_tx' not in data or 'Q_CW_tx' not in data:
                return 0.0, 0.0
            
            i_cw = data['I_CW_tx'].flatten()
            q_cw = data['Q_CW_tx'].flatten()
            
            # Convert to polar coordinates
            theta = np.arctan2(q_cw, i_cw)  # phase
            # rho = np.sqrt(i_cw**2 + q_cw**2)  # magnitude (not used)
            
            # Unwrap phase to avoid 2π jumps
            theta_unwrap = np.unwrap(theta)
            
            phase_std = np.std(theta_unwrap)
            phase_mean = np.mean(theta_unwrap)
            
            return float(phase_mean), float(phase_std)
            
        except Exception as e:
            warnings.warn(f"Could not extract phase info from {tx_file_path}: {e}")
            return 0.0, 0.0
    
    def _apply_averaging(self, params1: np.ndarray, params2: np.ndarray) -> np.ndarray:
        """
        Apply averaging based on the configured type.
        """
        if self.avg_type == 'average':
            return (params1 + params2) / 2
        elif self.avg_type == 'before':
            return params1
        elif self.avg_type == 'after':
            return params2
        else:
            raise ValueError(f"Unknown averaging type: {self.avg_type}")
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate transmission analysis outputs.
        """
        result = data.get('transmission_analysis')
        if not result:
            print("  ✗ No transmission analysis result found")
            return False
        
        if not result.scaleinfo_updates:
            print("  ✗ No scaleinfo updates from transmission analysis")
            return False
        
        num_files = len(result.frequency_drifts_khz)
        print(f"  ✓ Transmission analysis processed {num_files} files")
        return True