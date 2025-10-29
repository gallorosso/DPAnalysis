"""
Range optimization for cavity fitting.
"""

import numpy as np
from typing import Tuple, Literal, Dict, Any
from .cavity_fitter import cavity_fit
from .lorentzian import iq_to_magnitude

def optimized_fit(type: Literal['tx', 'rfl'], i_data: np.ndarray, q_data: np.ndarray,
                  freq: np.ndarray, proc_par: Dict[str, Any]) -> Tuple[np.ndarray, float, slice]:
    """
    Python equivalent of optimizedfitJ.
    
    Optimizes fitting range by testing different widths and selecting most symmetrical residuals.
    
    Args:
        type: 'tx' for transmission, 'rfl' for reflection
        i_data: In-phase data
        q_data: Quadrature data
        freq: Frequency array [GHz]
        proc_par: Processing parameters dictionary
        
    Returns:
        bestfit_params: Best fit parameters [P_max, f0, Q, slope, offset]
        mse: Mean squared error of best fit
        datarange: Slice object indicating fitting range
    """
    # Convert I/Q to magnitude squared
    quick_mag = iq_to_magnitude(i_data, q_data)
    
    # Smart initialization or use provided initial parameters
    if proc_par.get('use_smart_init', True):
        init_params, min_halfwidth, max_halfwidth = _smart_initialization(
            type, quick_mag, freq, proc_par
        )
    else:
        # Use hard-coded initial parameters
        if type == 'tx':
            init_params = proc_par['init_params_tx']
            min_halfwidth = 30
            max_halfwidth = 40
        else:  # 'rfl'
            init_params = proc_par['init_params_rfl'] 
            min_halfwidth = 18
            max_halfwidth = 28
    
    # Test different fitting ranges
    lnbr = (max_halfwidth - min_halfwidth) + 1
    param_matrix = np.zeros((lnbr, 5))
    symcheck = np.zeros(lnbr)
    msemat = np.zeros(lnbr)
    
    for i in range(lnbr):
        fit_halfwidth = min_halfwidth + i
        
        try:
            # Fit with current range
            bestfit_params, mse0, residuals, peak_idx = cavity_fit(
                type, i_data, q_data, freq, init_params, fit_halfwidth
            )
            
            param_matrix[i, :] = bestfit_params
            msemat[i] = mse0
            
            # Calculate symmetry of residuals
            if len(residuals) >= 2:
                symlist = np.zeros(len(residuals) // 2)
                for j in range(len(symlist)):
                    symlist[j] = (residuals[j] - residuals[-(j+1)])**2
                symcheck[i] = np.mean(symlist)
            else:
                symcheck[i] = np.inf
                
        except Exception:
            # If fitting fails, mark as poor
            symcheck[i] = np.inf
            msemat[i] = np.inf
    
    # Select best range (minimize symmetry error, avoid worst MSE)
    best_sym_idx = np.argmin(symcheck)
    worst_mse_idx = np.argmax(msemat)
    
    if best_sym_idx == worst_mse_idx:
        # If same index, choose based on MSE instead
        best_idx = np.argmin(msemat)
    else:
        best_idx = best_sym_idx
    
    bestfitparams = param_matrix[best_idx, :]
    bestfitparams[2] = abs(bestfitparams[2])  # Ensure Q is positive
    
    mse = msemat[best_idx]
    fit_halfwidth = min_halfwidth + best_idx
    
    # Calculate final data range
    quick_mag_flat = quick_mag.flatten()
    if type == 'tx':
        peak_idx = np.argmax(quick_mag_flat)
    else:  # 'rfl'
        peak_idx = np.argmin(quick_mag_flat)
    
    subset_start = max(peak_idx - fit_halfwidth, 0)
    subset_end = min(peak_idx + fit_halfwidth, len(freq) - 1)
    datarange = slice(subset_start, subset_end + 1)
    
    return bestfitparams, mse, datarange

def _smart_initialization(type: Literal['tx', 'rfl'], quick_mag: np.ndarray,
                         freq: np.ndarray, proc_par: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    """
    Smart initialization of fitting parameters based on data characteristics.
    """
    quick_mag_flat = quick_mag.flatten()
    freq_flat = freq.flatten()
    
    # Calculate frequency step
    df = abs(freq_flat[1] - freq_flat[0])
    
    if type == 'tx':
        # Transmission: find maximum
        max_val, max_idx = np.max(quick_mag_flat), np.argmax(quick_mag_flat)
        
        # Find FWHM
        half_max = max_val * 0.5
        fwhm_idx = np.argmin(np.abs(quick_mag_flat - half_max))
        
        width_GHz = 2 * abs(max_idx - fwhm_idx) * df
        freq_guess_GHz = freq_flat[max_idx]
        Q_guess = freq_guess_GHz / width_GHz
        
        # Baseline guess from endpoints
        bl_guess = (quick_mag_flat[0] + quick_mag_flat[-1]) / 2
        
        init_params = np.array([
            max_val,           # P_max
            freq_guess_GHz,    # f0
            Q_guess,           # Q
            0.0,               # slope
            bl_guess           # offset
        ])
        
        # Calculate range parameters
        min_halfwidth = int(proc_par['tx_fit_width_sigma'] * abs(max_idx - fwhm_idx))
        max_halfwidth = min_halfwidth + proc_par['tx_fit_buffer_bins']
        
    else:  # 'rfl'
        # Reflection: find minimum
        bl_guess = (quick_mag_flat[0] + quick_mag_flat[-1]) / 2
        min_val, min_idx = np.min(quick_mag_flat), np.argmin(quick_mag_flat)
        
        # Invert for FWHM calculation
        quick_mag_invert = (quick_mag_flat - bl_guess) * -1.0
        max_val, _ = np.max(quick_mag_invert), np.argmax(quick_mag_invert)
        fwhm_idx = np.argmin(np.abs(quick_mag_invert - (max_val * 0.5)))
        
        width_GHz = 2 * abs(min_idx - fwhm_idx) * df
        freq_guess_GHz = freq_flat[min_idx]
        Q_guess = freq_guess_GHz / width_GHz
        
        init_params = np.array([
            min_val - bl_guess,  # P_min (negative for reflection dip)
            freq_guess_GHz,      # f0
            Q_guess,             # Q
            0.0,                 # slope
            bl_guess             # offset
        ])
        
        # Calculate range parameters
        min_halfwidth = int(proc_par['rfl_fit_width_sigma'] * abs(min_idx - fwhm_idx))
        max_halfwidth = min_halfwidth + proc_par['rfl_fit_buffer_bins']
    
    return init_params, min_halfwidth, max_halfwidth