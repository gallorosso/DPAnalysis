"""
Transmission analysis stage for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List
import scipy.io
import numpy as np
import warnings

from ..base import PipelineStage, PipelineContext
from ..results import TransmissionAnalysisResult
from src.dark_photon.fitting import optimized_fit

class TransmissionAnalysisStage(PipelineStage):
    """
    Stage 3.1: Analyze cavity transmission data.
    
    Python implementation of ReadOutCavityTran (analysis only, no plotting).
    """
    
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
        
        processed_count = 0
        
        for i, tx2_file in enumerate(files):
            try:
                # Get corresponding tx1 file
                tx1_file = self._get_tx1_file(tx2_file)
                
                if not tx1_file.exists():
                    warnings.warn(f"TX1 file not found for {tx2_file.name}")
                    continue
                
                # Process both sweeps
                params1[i, :], mse1, _ = self._process_transmission_sweep(
                    tx1_file, 'tx', context.run_props.processing
                )
                params2[i, :], mse2, _ = self._process_transmission_sweep(
                    tx2_file, 'tx', context.run_props.processing
                )
                
                # Calculate averages
                params_avg[i, :] = (params1[i, :] + params2[i, :]) / 2
                
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
            'txdriftkHz': freq_drifts_khz.tolist()
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
        
        # Get processing parameters from configuration - NO DEFAULTS
        # These MUST be defined in the YAML file
        fitting_config = proc_par.fitting
        
        # Convert to dict for fitting function - require all parameters from YAML
        proc_par_dict = {
            'use_smart_init': fitting_config['use_smart_init'],
            'tx_fit_width_sigma': fitting_config['tx_fit_width_sigma'],
            'tx_fit_buffer_bins': fitting_config['tx_fit_buffer_bins'],
            'rfl_fit_width_sigma': fitting_config['rfl_fit_width_sigma'],
            'rfl_fit_buffer_bins': fitting_config['rfl_fit_buffer_bins'],
            'init_params_tx': fitting_config['init_params_tx'],
            'init_params_rfl': fitting_config['init_params_rfl']
        }
        
        # Fit transmission data
        bestfit_params, mse, datarange = optimized_fit(
            'tx', i_data, q_data, freq, proc_par_dict
        )
        
        return bestfit_params, mse, datarange
    
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