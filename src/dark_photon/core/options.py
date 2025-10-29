from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any
import yaml

@dataclass
class DataRunOptions:
    """
    Configuration options for axion data analysis runs.
    
    This class holds all parameters needed to configure the data analysis pipeline.
    It can be initialized from code or loaded from a YAML configuration file.
    """
    
    # --- Data Selection and I/O ---
    phase_name: str = "PIIc"
    """Top level directory name holding the data for each date"""
    
    rescan: bool = False
    """Is this rescan data or not"""
    
    input_dir: Union[str, Path] = "/Users/andrea.gallorosso/Documents/Attivi/HAYSTAC analysis"
    """Directory containing the raw data files"""
    
    output_dir: Union[str, Path] = "/Users/andrea.gallorosso/Documents/Attivi/DPAnalysis/output/"
    """Base directory for output results"""
    
    first_data_set: int = 20220908
    """First dataset to process (format: YYYYMMDD)"""
    
    last_data_set: int = 20220908  
    """Last dataset to process (format: YYYYMMDD)"""
    
    # --- Plotting Controls ---
    plottrue: bool = False
    """Whether to display plots interactively"""
    
    calib_plotting: bool = False
    """Whether to plot calibration details"""
    
    plot_subspec: bool = False
    """Plot each subspectrum to file (can be very slow)"""
    
    # --- Data Quality and Processing ---
    hard_file_cut: bool = False
    """Apply hard cuts on known bad runs"""
    
    use_worm: bool = False
    """Toggle RF worm cut to remove pre-determined RF spikes"""
    
    skip_baseline: bool = False
    """Skip recalculation of IF baseline"""
    
    old_BL_file: Optional[Union[str, Path]] = None
    """File to load baseline from if skip_baseline is True"""
    
    rescan_no_calc_baseline: bool = False
    """Load existing baseline without calculation (for rescans)"""
    
    low_IF_pad: bool = False
    """Apply low IF padding"""
    
    # --- Calibration and Physical Data Files ---
    form_fac_file: Union[str, Path] = "../../4to5GHz_corrected.xlsx"
    """File containing form factor data"""
    
    calib_coeff_file: Union[str, Path] = "../../../data/coeffs_alpharho_power_4_1_4_4.txt" 
    """File containing calibration coefficients"""
    
    calib_data_file: Union[str, Path] = "../../../data/alpharho_lam_vs_f.xlsx"
    """File containing calibration data"""
    
    # --- Performance and Algorithm Choices ---
    add_in_par: bool = False
    """Add subspectra to grand spectrum in parallel"""
    
    gain_const_bool: bool = True
    """Use constant gain of 650 for all calibration calculations"""
    
    float_abg_gains: bool = False
    """Let gains float in scalefactor (used in PIIa/PIIb)"""
    
    freqs_from_par: bool = True
    """Use frequencies from par file rather than fitted frequencies"""
    
    old_SG_filter: bool = False
    """Use old Savitzky-Golay filter technique (slower)"""
    
    Sc_full_gain_correction: bool = False
    """Use full gain profiles for Sc gain correction"""
    
    sq_on_bool: bool = True
    """Use full cutpar (should be 0 for rescans)"""
    
    def __post_init__(self):
        """Post-initialization processing and validation"""
        # Convert string paths to Path objects for easier handling
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.form_fac_file = Path(self.form_fac_file)
        self.calib_coeff_file = Path(self.calib_coeff_file) 
        self.calib_data_file = Path(self.calib_data_file)
        
        if self.old_BL_file:
            self.old_BL_file = Path(self.old_BL_file)
        
        # Validate date range
        if self.last_data_set < self.first_data_set:
            raise ValueError("last_data_set must be >= first_data_set")
            
        # Auto-set sq_on_bool based on rescan if not explicitly set
        # Note: This handles the conditional logic from the original script
        if self.rescan and self.sq_on_bool:
            self.sq_on_bool = False
    
    def get_output_dir_full(self, comments: str = "") -> Path:
        """
        Generate the full output directory path based on current configuration.
        
        Args:
            comments: Additional comments to include in directory name
            
        Returns:
            Path to the full output directory
        """
        comments = comments or f"Worm{self.use_worm}Base{self.skip_baseline}"
        dir_name = (f"{self.first_data_set}to{self.last_data_set}_"
                   f"{comments}")
        return self.output_dir / dir_name


def load_options_from_yaml(yaml_path: Union[str, Path]) -> DataRunOptions:
    """
    Load configuration from a YAML file and return a DataRunOptions instance.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        DataRunOptions instance with configuration from the YAML file
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return DataRunOptions(**config_dict)


def save_options_to_yaml(options: DataRunOptions, yaml_path: Union[str, Path]):
    """
    Save the current configuration to a YAML file for reproducibility.
    
    Args:
        options: DataRunOptions instance to save
        yaml_path: Path where the YAML file should be saved
    """
    yaml_path = Path(yaml_path)
    
    # Convert to dictionary, handling Path objects
    config_dict = {}
    for field_name, field_value in options.__dict__.items():
        if isinstance(field_value, Path):
            config_dict[field_name] = str(field_value)
        else:
            config_dict[field_name] = field_value
    
    # Ensure output directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)