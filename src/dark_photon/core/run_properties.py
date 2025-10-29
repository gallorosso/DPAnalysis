from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import yaml
import warnings

# Import the component classes
from ..processing.data_quality import DataQualityCuts
from ..processing.rfi_mitigation import RFIMitigation, RFICut
from ..processing.parameters import ProcessingParameters
from .system_parameters import SystemParameters

@dataclass
class RunProperties:
    """Unified container for all run-specific properties."""
    
    phase_name: str
    system: SystemParameters
    rfi_mitigation: RFIMitigation
    data_quality: DataQualityCuts
    processing: ProcessingParameters
    
    @classmethod
    def from_phase_name(cls, phase_name: str, config_dir: Path = None) -> 'RunProperties':
        """Load all properties from YAML configuration."""
        if config_dir is None:
            config_dir = Path('config/run_properties')
        
        phase_name_upper = phase_name.upper()
        config_file = config_dir / f"{phase_name_upper}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Run properties file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create sub-components with error handling
        try:
            system = SystemParameters(**config_dict.get('system', {}))
        except TypeError as e:
            raise ValueError(f"Error loading system parameters: {e}")
        
        try:
            # Handle RFI cuts conversion from dict to RFICut objects
            rfi_config = config_dict.get('rfi_mitigation', {})
            cuts_config = rfi_config.get('cuts', [])
            rfi_cuts = []
            
            for cut_dict in cuts_config:
                rfi_cuts.append(RFICut(**cut_dict))
            
            rfi_mitigation = RFIMitigation(
                RFcutsigma=rfi_config.get('RFcutsigma', 10000000.0),
                IFcutsigma=rfi_config.get('IFcutsigma', 4.5),
                wormcut_width=rfi_config.get('wormcut_width', 0.0002),
                cuts=rfi_cuts
            )
        except Exception as e:
            raise ValueError(f"Error loading RFI mitigation parameters: {e}")
        
        try:
            data_quality = DataQualityCuts(**config_dict.get('data_quality', {}))
        except TypeError as e:
            raise ValueError(f"Error loading data quality cuts: {e}")
        
        try:
            processing = ProcessingParameters(**config_dict.get('processing', {}))
        except TypeError as e:
            raise ValueError(f"Error loading processing parameters: {e}")
        
        return cls(
            phase_name=phase_name,
            system=system,
            rfi_mitigation=rfi_mitigation,
            data_quality=data_quality,
            processing=processing
        )
    
    @classmethod
    def from_options(cls, options) -> 'RunProperties':
        """Create from DataRunOptions instance with runtime overrides."""
        run_props = cls.from_phase_name(options.phase_name)
        
        # Apply runtime overrides
        run_props.apply_runtime_overrides(options)
        
        return run_props
    
    def apply_runtime_overrides(self, options):
        """Apply all runtime overrides based on options."""
        self.data_quality.apply_runtime_overrides(options)
        self.processing.apply_runtime_overrides(options)
        
        # Note: System parameters and RFI mitigation don't need runtime overrides
        # as they're fully determined by phase_name and date ranges