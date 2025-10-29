from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataQualityCuts:
    """Data quality thresholds for excluding bad scans."""
    
    # General
    remove_bad_scans: bool = True
    tx_drift_khz: float = 60.0
    rfl_drift_khz: float = 60.0
    
    # Gain thresholds
    amp_gain: float = 1.0
    sq_gain: float = 1.0
    squeezing_db: float = 1.0
    squeezing_peak_db: float = 0.13
    
    # Gain difference thresholds (ADD THESE)
    amp_gain_diff: float = 1000000.0
    sq_gain_diff: float = 1000000.0
    gain_diff_dB: float = 0.4
    
    # Physical limits
    NH_lim: List[float] = field(default_factory=lambda: [0, 500])
    
    # Statistical cuts
    param_std: float = 3.0
    param_std_std: float = 3.0
    smooth_width: int = 10
    freq_jump_GHz: float = 600e-6
    find_smoothing_regions: bool = True
    
    # Integration time
    int_time_min: float = 100.0
    
    # Hard file exclusions
    spec_date_cut: List[int] = field(default_factory=list)
    spec_file_cut: List[Tuple[int, int, int]] = field(default_factory=list)
    
    def apply_runtime_overrides(self, options):
        """Apply runtime overrides based on options."""
        # Squeezing parameter depends on sq_on_bool
        if not options.sq_on_bool:
            self.squeezing_db = -1.0
        
        # paramstd depends on rescan and phase
        if options.rescan and options.phase_name != 'PIId':
            self.param_std = 5.0
        else:
            self.param_std = 3.0
        
        # Apply hard file cuts only if requested
        if not options.hard_file_cut:
            self.spec_date_cut = []
            self.spec_file_cut = []