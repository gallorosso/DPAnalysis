from dataclasses import dataclass, field
from typing import Optional
import warnings

@dataclass
class SystemParameters:
    """System parameters measured before runs."""
    
    # Temperatures
    t_fridge_K: float
    t_VTS_K: float
    
    # Losses in the system
    lam: float
    alpha: Optional[float] = None
    rho: Optional[float] = None
    
    # Volume and B-Field
    V_cav: float = None
    B_0: float = None
    
    def __post_init__(self):
        """Set derived parameters with informative warnings."""
        # Handle alpha/rho relationship with warnings
        if self.alpha is not None and self.rho is None:
            self.rho = self.alpha
            warnings.warn(
                f"Parameter 'rho' was not provided. Setting rho = alpha = {self.alpha}",
                UserWarning,
                stacklevel=2
            )
        elif self.rho is not None and self.alpha is None:
            self.alpha = self.rho
            warnings.warn(
                f"Parameter 'alpha' was not provided. Setting alpha = rho = {self.rho}",
                UserWarning, 
                stacklevel=2
            )
        elif self.alpha is None and self.rho is None:
            warnings.warn(
                "Parameters 'alpha' and 'rho' were not provided. Using default values: None.",
                UserWarning,
                stacklevel=2
            )