from dataclasses import dataclass

@dataclass(frozen=True)  # frozen makes it immutable
class Constants:
    """Container for physical constants."""
    
    # Physical Constants that will Never Change
    h: float = 6.626070040e-34  # J*s
    c_light: float = 2.998e8  # m/s, speed of light
    hbar_c: float = 1.97327e-14  # GeV*cm
    alpha_EM: float = 0.0072973527  # fine-structure constant
    mu_0: float = 4 * 3.141592653589793 * 1e-7  # permeability of free space, SI units
    
    # DM properties (may change)
    Lambda_0: float = 77.6056e-3  # GeV
    sigma_v: float = 2.7e5  # m/s, halo velocity for the lineshape
    g_KSVZ: float = 0.97  # coupling constant at KSVZ sensitivity
    rho_DM: float = 0.45  # local dark matter density, GeV/cm^3

# # Create a default instance for easy access
# spp = Constants()