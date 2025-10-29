"""
Core cavity fitting functions.
"""

import numpy as np
from typing import Tuple, Literal
from .lorentzian import fit_lorentzian, iq_to_magnitude

def cavity_fit(type: Literal['tx', 'rfl'], i_data: np.ndarray, q_data: np.ndarray,
               freq: np.ndarray, init_params: np.ndarray, 
               fit_halfwidth: int) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """
    Python equivalent of cavityfitJ.
    
    Fits a portion of Tx/Rfl amplitude to Lorentzian.
    
    Args:
        type: 'tx' for transmission, 'rfl' for reflection
        i_data: In-phase data
        q_data: Quadrature data  
        freq: Frequency array [GHz]
        init_params: Initial parameters [P_max, f0, Q, slope, offset]
        fit_halfwidth: Half-width for fitting range
        
    Returns:
        bestfit_params: Fitted parameters
        mse: Mean squared error
        residuals: Fit residuals
        peak_idx: Index of peak/dip in data
    """
    # Convert I/Q to magnitude squared
    amp2 = iq_to_magnitude(i_data, q_data)
    
    # Find peak/dip based on type
    if type == 'tx':
        peak_idx = np.argmax(amp2)  # Maximum for transmission
    elif type == 'rfl':
        peak_idx = np.argmin(amp2)  # Minimum for reflection
    else:
        raise ValueError(f"Unknown type: {type}. Use 'tx' or 'rfl'")
    
    # Update initial frequency guess with peak position
    init_params_updated = init_params.copy()
    init_params_updated[1] = freq[peak_idx]  # f0 parameter
    
    # Select fitting range
    subset_start = max(peak_idx - fit_halfwidth, 0)
    subset_end = min(peak_idx + fit_halfwidth, len(freq) - 1)
    subset_indices = slice(subset_start, subset_end + 1)
    
    freq_subset = freq[subset_indices]
    amp2_subset = amp2[subset_indices]
    
    # Fit Lorentzian
    bestfit_params, residuals, mse = fit_lorentzian(
        freq_subset, amp2_subset, init_params_updated
    )
    
    return bestfit_params, mse, residuals, peak_idx