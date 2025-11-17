from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class ProcessingParameters:
    """Processing algorithm parameters with runtime overrides."""
    
    # Calibration parameters
    calibration: Dict[str, Any] = field(default_factory=dict)
    
    # Fitting parameters  
    fitting: Dict[str, Any] = field(default_factory=dict)
    
    # Filter parameters
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Other processing parameters
    psasign: str = '-'
    binavg: int = 1
    num_baselines: int = 1
    IFwindow: List[float] = field(default_factory=lambda: [0.045, 1.345])
    LowPass_MHz: float = 1.9
    
    def apply_runtime_overrides(self, options):
        """Apply runtime overrides based on options."""
        self._override_fitting_parameters(options)
        self._override_filter_parameters(options)
        self._override_if_window(options)
        self._override_jpa_cut_parameters(options)
    
        # Ensure JPA-related fitting parameters have sensible defaults
        if 'JPA_gbw_prod' not in self.fitting:
            # MATLAB JPAgainAutorun default; can be overridden from YAML
            self.fitting['JPA_gbw_prod'] = 8.15e7
        if 'jpa_fit_width_sigma' not in self.fitting:
            # By default, tie the fit width to the JPA profile cut parameter
            self.fitting['jpa_fit_width_sigma'] = self.fitting.get('r_JPA_prof_cut', 5)
    
    def _override_fitting_parameters(self, options):
        """Override fitting parameters based on rescan status."""
        if options.rescan:
            # Different initial parameters for rescans
            self.fitting['init_params_tx'] = [0.0003, 4.109, 21199, -0.0347, 0.145]
            self.fitting['init_params_rfl'] = [-0.0017, 4.1605, -20500, -0.02, 0.0865]
        else:
            # Ensure standard parameters are set (might be overridden from YAML)
            if 'init_params_tx' not in self.fitting:
                self.fitting['init_params_tx'] = [0.0003, 4.172, 5845, -0.0347, 0.145]
            if 'init_params_rfl' not in self.fitting:
                self.fitting['init_params_rfl'] = [-0.0017, 4.1605, -6500, -0.02, 0.0865]
    
    def _override_filter_parameters(self, options):
        """Override filter parameters based on rescan and baseline settings."""
        # Handle baseline loading override
        if options.skip_baseline and options.old_BL_file:
            try:
                # Note: This would need to be implemented when we have the baseline file format
                # For now, we'll keep the YAML values
                pass
            except Exception as e:
                print(f"Warning: Could not load baseline parameters: {e}")
        
        # Rescan-specific filter adjustments
        if options.rescan and not options.sq_on_bool:
            self.filters['fitorder2'] = 6
            self.filters['sg_win_Hz'] = 60000  # 60e3
    
    def _override_if_window(self, options):
        """Override IF window based on dates and rescan status."""
        if options.rescan and not options.sq_on_bool:
            self.IFwindow = [0.045, 0.545]  # in MHz, will be converted to Hz later
        else:
            if (options.first_data_set < 20190925 and 
                options.last_data_set < 20190925):
                self.IFwindow = [0.045, 0.745]  # in MHz
            elif (options.first_data_set >= 20190925 and 
                  options.last_data_set >= 20190925):
                self.IFwindow = [0.045, 1.345]  # in MHz
            else:
                raise ValueError(
                    "Choose starting and ending data sets with same IF band. "
                    f"First: {options.first_data_set}, Last: {options.last_data_set}"
                )
        
        # Convert MHz to Hz for internal use
        self.IFwindow = [x * 1e6 for x in self.IFwindow]
    
    def _override_jpa_cut_parameters(self, options):
        """Override JPA profile cut based on sq_on_bool."""
        if not options.sq_on_bool:
            self.fitting['r_JPA_prof_cut'] = 3
        else:
            self.fitting['r_JPA_prof_cut'] = 5