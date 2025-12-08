"""
Parameter loading stage for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List
import scipy.io
import warnings
import numpy as np

from ..base import PipelineStage, PipelineContext
from ..results import ParameterLoadingResult

class ParameterLoadingStage(PipelineStage):
    """
    Stage 2.2: Load parameter data from DAQ files.
    
    Python implementation of LoadParData.
    """
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute parameter loading stage.
        """
        print("  Loading parameter data...")
        
        # Get dataset directories
        from src.dark_photon.io import load_run_dirs_from_options
        dataset_dirs = load_run_dirs_from_options(context.options)
        
        scaleinfo = {}
        
        for dataset_dir in dataset_dirs:
            print(f"    Processing directory: {dataset_dir}")
            
            # Find all parameter files in this directory
            from src.dark_photon.io.helpers import find_files_by_pattern
            par_files = find_files_by_pattern(dataset_dir, 'par')
            print(f"      Found {len(par_files)} parameter files")
            
            for par_file_base in par_files:
                try:
                    # Reconstruct the full par filename
                    par_filename = Path(f"{par_file_base}par.mat")
                    print(f"      Processing: {par_filename.name}")
                    
                    # Load parameter file
                    par_data = scipy.io.loadmat(str(par_filename))
                    
                    # Skip empty parameter files
                    if self._is_par_file_empty(par_data):
                        warnings.warn(f"Skipping empty par file: {par_filename}")
                        continue
                    
                    # Extract the parameter structure
                    par_struct = par_data['par']
                    
                    # Process this parameter file
                    if not scaleinfo:  # First file - initialize scaleinfo
                        print(f"      Initializing scaleinfo from first file")
                        scaleinfo = self._initialize_scaleinfo(par_struct)
                    else:  # Append to existing scaleinfo
                        print(f"      Appending to scaleinfo")
                        scaleinfo = self._append_to_scaleinfo(scaleinfo, par_struct)
                        
                    print(f"      Loaded parameters from: {par_filename.name}")
                    
                except Exception as e:
                    warnings.warn(f"Error processing par file {par_file_base}: {e}")
                    continue
    
        # Apply post-processing conversions and calculations
        if scaleinfo:
            scaleinfo = self._apply_post_processing(scaleinfo, context)
            print(f"  ✓ Loaded parameters for {self._get_parameter_count(scaleinfo)} spectra")
        else:
            warnings.warn("No parameter data was loaded")
        
        result = ParameterLoadingResult(
            scaleinfo=scaleinfo,
            status="success" if scaleinfo else "failed"
        )
        
        data['parameter_loading'] = result
        return data
    
    def _is_par_file_empty(self, par_data: Dict) -> bool:
        """Check if parameter file is empty."""
        try:
            par_struct = par_data.get('par')
            if par_struct is None:
                return True
            
            if hasattr(par_struct, 'dtype') and par_struct.dtype.names:
                if 'loop_time' in par_struct.dtype.names:
                    loop_time = par_struct['loop_time'][0, 0]
                    return len(loop_time) < 1 if hasattr(loop_time, '__len__') else False
                return False
            
            return True
            
        except Exception:
            return True
    
    def _initialize_scaleinfo(self, par_struct) -> Dict[str, Any]:
        """
        Initialize scaleinfo structure from first parameter file.
        """
        scaleinfo = {}
        
        if hasattr(par_struct, 'dtype') and par_struct.dtype.names:
            # Test one field to see its structure
            # test_field = 'Cavity_freq_GHz_tx'
            # if test_field in par_struct.dtype.names:
            #     test_data = par_struct[test_field]
            #     print(f"    TEST: {test_field} raw data:")
            #     print(f"      type: {type(test_data)}")
            #     print(f"      shape: {test_data.shape}")
            #     print(f"      dtype: {test_data.dtype}")
            #     if test_data.shape[0] == 1 and test_data.dtype == object:
            #         print(f"      First element type: {type(test_data[0, 0])}")
            #         print(f"      First element: {test_data[0, 0]}")
            #         print(f"      Full extraction test: {test_data[0, :]}")
            
            for field_name in par_struct.dtype.names:
                try:
                    # Extract the field data
                    field_data = par_struct[field_name]
                    
                    # Handle object arrays (like (1, 42) arrays of arrays)
                    if field_data.dtype == object and field_data.shape[0] == 1:
                        # Extract all elements from the row
                        extracted_values = []
                        for i in range(field_data.shape[1]):
                            element = field_data[0, i]
                            if hasattr(element, '__len__'):
                                # Element is an array, take its first value
                                if len(element) > 0:
                                    extracted_values.append(element[0, 0] if element.ndim == 2 else element[0])
                                else:
                                    extracted_values.append(0.0)  # Default if empty
                            else:
                                # Element is scalar
                                extracted_values.append(element)
                        scaleinfo[field_name] = extracted_values
                    else:
                        # Regular array, flatten and convert to list
                        if hasattr(field_data, '__len__'):
                            scaleinfo[field_name] = field_data.flatten().tolist()
                        else:
                            scaleinfo[field_name] = [field_data]
                            
                except Exception as e:
                    warnings.warn(f"Could not initialize field '{field_name}': {e}")
                    scaleinfo[field_name] = []
        
        return scaleinfo
    
    def _append_to_scaleinfo(self, scaleinfo: Dict[str, Any], par_struct) -> Dict[str, Any]:
        """
        Append new parameter data to existing scaleinfo.
        """
        if not hasattr(par_struct, 'dtype') or not par_struct.dtype.names:
            return scaleinfo
        
        for field_name in par_struct.dtype.names:
            try:
                field_data = par_struct[field_name]
                
                # Debug specific fields
                if field_name in ['Cavity_freq_GHz_tx', 'Cavity_freq_GHz_tx2', 
                                'Cavity_freq_GHz_rfl', 'Cavity_freq_GHz_rfl2']:
                    print(f"      Loading {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")
                
                # Handle object arrays (like (1, 42) arrays of arrays)
                if field_data.dtype == object and field_data.shape[0] == 1:
                    # Extract all elements from the row
                    new_data = []
                    for i in range(field_data.shape[1]):
                        element = field_data[0, i]
                        if hasattr(element, '__len__'):
                            # Element is an array, take its first value
                            if len(element) > 0:
                                if element.ndim == 2:
                                    new_data.append(element[0, 0])
                                else:
                                    new_data.append(element[0])
                            else:
                                new_data.append(0.0)  # Default if empty
                        else:
                            # Element is scalar
                            new_data.append(element)
                else:
                    # Regular array
                    if hasattr(field_data, '__len__'):
                        new_data = field_data.flatten().tolist()
                    else:
                        new_data = [field_data]
                
                # Append to existing field or initialize if missing
                if field_name in scaleinfo:
                    scaleinfo[field_name].extend(new_data)
                else:
                    # Field doesn't exist - initialize with zeros then set new data
                    existing_length = len(scaleinfo[list(scaleinfo.keys())[0]])
                    scaleinfo[field_name] = [0] * existing_length
                    scaleinfo[field_name].extend(new_data)
                    
            except Exception as e:
                warnings.warn(f"Could not append field '{field_name}': {e}")
                continue
        
        return scaleinfo
    
    def _apply_post_processing(self, scaleinfo: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """
        Apply post-processing conversions and calculations.
        """
        # Helper function to handle both scalars and arrays
        def round_values(values, decimals=7):
            if hasattr(values, '__iter__'):
                return np.round(values, decimals).tolist()
            else:
                return round(values, decimals)
        
        # Frequency conversions with rounding (as in MATLAB)
        scaleinfo['Cavity_freq_tx1'] = round_values(scaleinfo['Cavity_freq_GHz_tx'], 7)
        scaleinfo['Cavity_freq_tx2'] = round_values(scaleinfo['Cavity_freq_GHz_tx2'], 7)
        scaleinfo['Cavity_freq_rfl1'] = round_values(scaleinfo['Cavity_freq_GHz_rfl'], 7)
        scaleinfo['Cavity_freq_rfl2'] = round_values(scaleinfo['Cavity_freq_GHz_rfl2'], 7)
        
        # JPA pump frequency conversion (GHz to Hz)
        if hasattr(scaleinfo['JPA_pump_frequency'], '__iter__'):
            scaleinfo['JPA_pump_frequency'] = [freq * 1e9 for freq in scaleinfo['JPA_pump_frequency']]
        else:
            scaleinfo['JPA_pump_frequency'] = scaleinfo['JPA_pump_frequency'] * 1e9
        
        # Gain difference calculations
        def calculate_differences(values1, values2):
            if hasattr(values1, '__iter__') and hasattr(values2, '__iter__'):
                return [abs(g1 - g2) for g1, g2 in zip(values1, values2)]
            else:
                return abs(values1 - values2)
        
        scaleinfo['Amp_gain_diff'] = calculate_differences(scaleinfo['Amp_gain'], scaleinfo['Amp_gain2'])
        scaleinfo['Sq_gain_diff'] = calculate_differences(scaleinfo['Sq_gain'], scaleinfo['Sq_gain2'])
        
        # Primary cavity frequency (using tx1)
        scaleinfo['Cavity_freq'] = scaleinfo['Cavity_freq_tx1']
        
        # X-limit for plots (±5 MHz from min/max frequency)
        cav_freq = scaleinfo['Cavity_freq']
        if hasattr(cav_freq, '__iter__'):
            min_freq = min(cav_freq) - 5e-3
            max_freq = max(cav_freq) + 5e-3
        else:
            min_freq = cav_freq - 5e-3
            max_freq = cav_freq + 5e-3
        scaleinfo['xlimit'] = [min_freq, max_freq]
        
        # Add processing parameters that will be needed later
        scaleinfo['fitorder'] = context.run_props.processing.filters.get('fitorder', 10)
        scaleinfo['fitorder2'] = context.run_props.processing.filters.get('fitorder2', 4)
        
        return scaleinfo
    
    def _get_parameter_count(self, scaleinfo: Dict[str, Any]) -> int:
        """Get the number of parameter sets (spectra) loaded."""
        if not scaleinfo:
            return 0

        # Prefer fields that we know should be per-spectrum vectors
        preferred_keys = [
            'loop_time',
            'Cavity_freq_GHz_tx',
            'Cavity_freq_GHz_tx2',
            'Cavity_freq_GHz_rfl',
            'Cavity_freq_GHz_rfl2',
            'Cavity_freq',  # after post-processing
        ]

        for key in preferred_keys:
            if key in scaleinfo:
                field = scaleinfo[key]
                if hasattr(field, '__len__'):
                    return len(field)

        # Fallback: use the first list-like field
        for field in scaleinfo.values():
            if hasattr(field, '__len__'):
                return len(field)

        # Last resort: everything looks scalar
        return 1

    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate parameter loading stage outputs.
        """
        result = data.get('parameter_loading')
        if not result:
            print("  ✗ No parameter loading result found")
            return False
        
        if not result.scaleinfo:
            print("  ✗ No scaleinfo data loaded")
            return False
        
        # Check that we have the essential fields
        essential_fields = ['Cavity_freq', 'loop_time', 'Cavity_freq_tx1', 'Cavity_freq_rfl1']
        missing_fields = [field for field in essential_fields if field not in result.scaleinfo]
        
        if missing_fields:
            print(f"  ✗ Missing essential fields: {missing_fields}")
            return False
        
        param_count = self._get_parameter_count(result.scaleinfo)
        print(f"  ✓ Parameter loading successful - {param_count} parameter sets loaded")
        return True