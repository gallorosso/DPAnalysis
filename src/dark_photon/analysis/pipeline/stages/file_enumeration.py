"""
File enumeration stage for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List
import scipy.io
import warnings

from ..base import PipelineStage, PipelineContext
from ..results import FileEnumerationResult
from src.dark_photon.io.helpers import find_files_by_pattern

class FileEnumerationStage(PipelineStage):
    """
    Stage 2.1: Enumerate data files in dataset directories.
    
    Python implementation of LoadRunFiles.
    """
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute file enumeration stage.
        """
        print("  Enumerating data files...")
        
        files = []
        data_set_lengths = []
        
        # Load dataset directories using the same approach as main script
        from src.dark_photon.io import load_run_dirs_from_options
        dataset_dirs = load_run_dirs_from_options(context.options)
        
        for dataset_dir in dataset_dirs:
            print(f"    Searching in: {dataset_dir}")
            
            # Find all parameter files in this directory
            par_files = find_files_by_pattern(dataset_dir, 'par')
            print(f"      Found {len(par_files)} parameter files")
            
            for par_file_base in par_files:
                try:
                    # Reconstruct the full par filename
                    par_filename = Path(f"{par_file_base}par.mat")
                    
                    # Load parameter file to check if it's valid
                    par_data = scipy.io.loadmat(str(par_filename))
                    
                    # Check if par file has data
                    if self._is_par_file_empty(par_data):
                        warnings.warn(f"Skipping empty par file: {par_filename}")
                        continue
                    
                    # Extract date and parnum from base filename
                    date_num, par_num = self._extract_date_parnum(par_file_base)
                    
                    # Build search pattern for matching data files
                    search_name = f"{date_num}_{par_num}"
                    data_files = find_files_by_pattern(dataset_dir, 'tx2', search_name)
                    
                    # Convert base names back to full file paths
                    full_data_files = [Path(f"{base}tx2.mat") for base in data_files]
                    
                    files.extend(full_data_files)
                    data_set_lengths.append(len(full_data_files))
                    
                    print(f"      Found {len(full_data_files)} data files for {search_name}")
                    
                except Exception as e:
                    warnings.warn(f"Error processing par file {par_file_base}: {e}")
                    continue
        
        result = FileEnumerationResult(
            files=files,
            data_set_lengths=data_set_lengths,
            status="success"
        )
        
        data['file_enumeration'] = result
        return data
    
    def _is_par_file_empty(self, par_data: Dict) -> bool:
        """
        Check if parameter file is empty.
        
        The .mat file structure is: {'par': structured_array, ...}
        We check if the 'par' field exists and has data.
        """
        try:
            # The actual parameter data is stored under the 'par' key
            par_struct = par_data.get('par')
            if par_struct is None:
                return True
            
            # Check if it's a structured array with fields
            if hasattr(par_struct, 'dtype') and par_struct.dtype.names:
                # Check if loop_time exists and has data
                if 'loop_time' in par_struct.dtype.names:
                    loop_time = par_struct['loop_time'][0, 0]  # MATLAB 2D array
                    return len(loop_time) < 1 if hasattr(loop_time, '__len__') else False
                
                # If no loop_time but has other fields, assume not empty
                return False
            
            return True
            
        except Exception:
            return True
    
    def _extract_date_parnum(self, par_file_base: Path) -> tuple[int, int]:
        """
        Extract date and parnum from base filename.
        
        Expected format: "YYYYMMDD_N_M" from "YYYYMMDD_N_M_par"
        We use N (the second part) as the parnum.
        """
        filename = par_file_base.name
        
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                date_num = int(parts[0])
                par_num = int(parts[1])
                return date_num, par_num
            except ValueError as e:
                raise ValueError(f"Could not parse date/parnum from '{filename}': {e}")
        else:
            raise ValueError(f"Unexpected filename format: {filename}")
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate file enumeration stage outputs.
        """
        result = data.get('file_enumeration')
        if not result:
            print("  ✗ No file enumeration result found")
            return False
        
        # Basic validation - file count should match sum of dataset lengths
        if len(result.files) != sum(result.data_set_lengths):
            print(f"  ✗ File count mismatch: {len(result.files)} files vs {sum(result.data_set_lengths)} expected")
            return False
            
        print(f"  ✓ File enumeration found {len(result.files)} data files across {len(result.data_set_lengths)} datasets")
        return True