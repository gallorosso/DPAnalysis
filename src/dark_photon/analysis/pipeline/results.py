"""
Data containers for pipeline stage results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class InitializationResult:
    """
    Results from the initialization stage.
    
    Contains directory paths and loaded form factor data.
    """
    plot_dir: Path
    meas_dir: Path
    form_fac_data: Any
    status: str = "success"
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.plot_dir = Path(self.plot_dir)
        self.meas_dir = Path(self.meas_dir)


@dataclass
class FileEnumerationResult:
    """Results from file enumeration stage."""
    files: List[Path] = field(default_factory=list)
    data_set_lengths: List[int] = field(default_factory=list)
    status: str = "pending"


@dataclass
class ParameterLoadingResult:
    """Results from parameter loading stage."""
    scaleinfo: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"


@dataclass
class CavityAnalysisResult:
    """Results from cavity analysis stage (placeholder)."""
    scaleinfo: Dict[str, Any] = field(default_factory=dict)
    transmission_data: Any = None
    reflection_data: Any = None
    status: str = "pending"

@dataclass
class TransmissionAnalysisResult:
    """Results from transmission analysis stage."""
    scaleinfo_updates: Dict[str, Any] = field(default_factory=dict)
    fitted_parameters: np.ndarray = None
    mse_values: np.ndarray = None
    frequency_drifts_khz: np.ndarray = None
    status: str = "pending"

@dataclass
class ReflectionAnalysisResult:
    """Results from reflection analysis stage."""
    scaleinfo_updates: Dict[str, Any] = field(default_factory=dict)
    mse_values: np.ndarray = None
    coupling_factors: np.ndarray = None
    reflection_frequencies: np.ndarray = None
    rfl_fit_params: np.ndarray = None
    rfl_baseline_db: np.ndarray = None
    status: str = "pending"

@dataclass
class ScaleinfoMergeResult:
    """Final merged scaleinfo including fit results and optional overrides."""
    scaleinfo: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"