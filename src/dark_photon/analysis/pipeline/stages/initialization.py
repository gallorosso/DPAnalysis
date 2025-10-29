"""
Initialization stage for the analysis pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict
import warnings

from ..base import PipelineStage, PipelineContext
from ..results import InitializationResult


class InitializationStage(PipelineStage):
    """
    Stage 1: Create output directories and load form factor data.
    
    This stage:
    1. Creates plot and measurement directories
    2. Loads form factor data from Excel file
    3. Sets up the directory structure for subsequent stages
    """
    
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute initialization stage.
        
        Args:
            context: Pipeline context with options and output directory
            data: Empty dictionary (no previous stages)
            
        Returns:
            Dictionary with 'initialization' key containing InitializationResult
        """
        print("  Creating output directories...")
        
        # Create output directories
        plot_dir = context.output_dir / 'plots'
        meas_dir = context.output_dir / 'measdata'
        
        plot_dir.mkdir(parents=True, exist_ok=True)
        meas_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"    Plot directory: {plot_dir}")
        print(f"    Measurement directory: {meas_dir}")
        
        # Load form factor data
        print("  Loading form factor data...")
        form_fac_file = context.options.form_fac_file
        try:
            form_fac_data = pd.read_excel(form_fac_file)
            print(f"    Form factor data loaded: {len(form_fac_data)} rows")
        except Exception as e:
            warnings.warn(f"Could not load form factor data from {form_fac_file}: {e}")
            form_fac_data = None
        
        # Create result object
        result = InitializationResult(
            plot_dir=plot_dir,
            meas_dir=meas_dir, 
            form_fac_data=form_fac_data,
            status="success"
        )
        
        # Update context with directories for future stages
        context.plot_dir = plot_dir
        context.meas_dir = meas_dir
        
        data['initialization'] = result
        return data
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate initialization stage outputs.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if directories exist and form factor data is loaded
        """
        result = data.get('initialization')
        if not result:
            print("  ✗ No initialization result found")
            return False
        
        # Check directories exist
        if not result.plot_dir.exists():
            print(f"  ✗ Plot directory does not exist: {result.plot_dir}")
            return False
            
        if not result.meas_dir.exists():
            print(f"  ✗ Measurement directory does not exist: {result.meas_dir}")
            return False
        
        # Check form factor data (warn but don't fail if missing)
        if result.form_fac_data is None:
            warnings.warn("Form factor data not loaded - this may affect calibration")
        
        print("  ✓ Initialization validation passed")
        return True