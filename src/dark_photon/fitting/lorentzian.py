"""
Lorentzian fitting functions for cavity analysis.
"""

import numpy as np
from typing import Tuple, Callable
from scipy.optimize import curve_fit

def lorentzian_plus_linear(f: np.ndarray, P_max: float, f0: float, Q: float, 
                          slope: float, offset: float) -> np.ndarray:
    """
    Lorentzian peak with linear baseline.
    
    Equivalent to MATLAB: a(1)./(4*(a(3).^2)*((xdata/a(2))-1).^2+1) + a(4).*xdata + a(5)
    
    Args:
        f: Frequency array [GHz]
        P_max: Peak power at resonance
        f0: Resonance frequency [GHz] 
        Q: Quality factor
        slope: Linear baseline slope
        offset: Linear baseline offset
        
    Returns:
        Power spectrum with Lorentzian + linear baseline
    """
    # Lorentzian term: P_max / (1 + 4Q^2((f/f0) - 1)^2)
    lorentzian = P_max / (1 + 4 * Q**2 * ((f / f0) - 1)**2)
    
    # Linear baseline: slope * f + offset
    linear_baseline = slope * f + offset
    
    return lorentzian + linear_baseline

def fit_lorentzian(xdata: np.ndarray, ydata: np.ndarray, 
                  init_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit Lorentzian + linear baseline to data.
    
    Python equivalent of fit_resonance_jean.
    
    Args:
        xdata: Frequency data [GHz]
        ydata: Power data [V^2]
        init_params: Initial guess [P_max, f0, Q, slope, offset]
        
    Returns:
        bestfit_params: Fitted parameters
        residuals: Fit residuals
        mse: Mean squared error
    """
    # Ensure 1D arrays
    xdata = np.asarray(xdata).flatten()
    ydata = np.asarray(ydata).flatten()
    
    # Define bounds (similar to MATLAB's lb/ub)
    bounds = (
        [-np.inf, np.min(xdata), 0, -np.inf, -np.inf],  # lower bounds
        [np.inf, np.max(xdata), np.inf, np.inf, np.inf]  # upper bounds
    )
    
    try:
        # Use curve_fit instead of MATLAB's nlinfit
        popt, pcov = curve_fit(lorentzian_plus_linear, xdata, ydata, 
                              p0=init_params, bounds=bounds, maxfev=5000)
        
        # Calculate residuals and MSE
        y_fit = lorentzian_plus_linear(xdata, *popt)
        residuals = ydata - y_fit
        mse = np.mean(residuals**2)
        
        return popt, residuals, mse
        
    except Exception as e:
        raise RuntimeError(f"Lorentzian fitting failed: {e}")

def iq_to_magnitude(i_data: np.ndarray, q_data: np.ndarray) -> np.ndarray:
    """
    Convert I/Q data to magnitude squared.
    
    Equivalent to MATLAB's: amp2 = i_in.^2 + q_in.^2
    
    Args:
        i_data: In-phase component
        q_data: Quadrature component
        
    Returns:
        Power: I^2 + Q^2
    """
    return np.array(i_data)**2 + np.array(q_data)**2