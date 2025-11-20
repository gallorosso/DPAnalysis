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

def optimized_fit_jpa(
    i_data: np.ndarray,
    q_data: np.ndarray,
    freq: np.ndarray,
    proc_par: Dict[str, float],
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Python equivalent of optimizedfitjpaJ.m for JPA gain profiles.

    Args:
        i_data, q_data : I/Q arrays for the JPA amplitude measurement
        freq           : frequency array [GHz]
        proc_par       : dict-like processing parameters; must provide
                         'jpa_fit_width_sigma' and 'jpa_fit_buffer_bins'

    Returns:
        bestfit_params : array of 5 fit parameters (same convention as cavity_fit)
        mse            : mean squared error of the chosen fit
        data_range     : (start_idx, end_idx) indices of the window used
                         (0-based, inclusive on both ends)
    """
    i_data = np.asarray(i_data).flatten()
    q_data = np.asarray(q_data).flatten()
    freq = np.asarray(freq).flatten()

    npts = len(freq)
    if npts < 5:
        raise ValueError("optimized_fit_jpa: not enough data points")

    # ------------------------------
    # Quick magnitude and basic stats
    # ------------------------------
    amp2 = iq_to_magnitude(i_data, q_data)          # linear power
    max_val_linear = np.max(amp2)
    peak_idx = int(np.argmax(amp2))
    half_power = max_val_linear / 2.0
    distances = np.abs(amp2 - half_power)
    
    # Exclude the peak itself from the search
    distances[peak_idx] = np.inf
    
    # Find the closest point to half-power
    bw_idx = int(np.argmin(distances))
    # Calculate bandwidth in bins (like MATLAB)
    jpa_fit_width_sigma = float(proc_par.get("jpa_fit_width_sigma", 5.0))

    bw = abs(peak_idx - bw_idx) * jpa_fit_width_sigma

    npts = len(freq)
    if bw > npts / 2:
        bw = np.floor(npts / 2) - 1
    if bw < npts / 10:
        bw = np.ceil(npts / 10)
    
    # Convert to integer half-width
    min_halfwidth = int(bw)
    jpa_fit_buffer_bins = int(proc_par.get("jpa_fit_buffer_bins", 5))
    max_halfwidth = min_halfwidth + jpa_fit_buffer_bins
    
    # MATLAB: lnbr = (max_halfwidth - min_halfwidth) + 1;
    lnbr = (max_halfwidth - min_halfwidth) + 1
    width_list = np.zeros(lnbr, dtype=int)
    
    # MATLAB tests: min_halfwidth, min_halfwidth+1, ..., max_halfwidth
    for i in range(lnbr):
        width_list[i] = min_halfwidth + i

    # ------------------------------
    # Bandwidth in GHz. Frequency step = abs(freq[1] - freq[0])
    bw_freq = bw * abs(freq[1] - freq[0])
    # Calculate quick magnitude (linear power, like MATLAB)
    quick_mag = i_data**2 + q_data**2
    # ------------------------------
    # Initial guess for fit parameters (like MATLAB)
    # ------------------------------
    amp_peak_linear = amp2[peak_idx]
    f0_guess = freq[peak_idx]
    if bw_freq > 0:
        Q_guess = init_params[1] / bw_freq  # Q = f0 / Δf
    else:
        Q_guess = 1000.0  # Fallback
    slope_guess = 0.0
    offset_guess = (quick_mag[0] + quick_mag[-1]) / 2.0

    init_params = np.array(
        [amp_peak_linear, f0_guess, Q_guess, slope_guess, offset_guess],
        dtype=float,
    )

    # ------------------------------
    # Loop over width_list and choose the most "symmetric" residuals
    # ------------------------------
    best_idx = None
    fit_params = None
    best_mse = None

    valid_fits_count = 0
    for jj, fit_halfwidth in enumerate(width_list):
                
        start = max(0, peak_idx - fit_halfwidth)
        stop = min(npts - 1, peak_idx + fit_halfwidth)

        i_slice = i_data[start:stop + 1]
        q_slice = q_data[start:stop + 1]
        f_slice = freq[start:stop + 1]

        try:
            fit_params, best_mse, residuals, peak_idx_slice = cavity_fit(
                "tx", i_slice, q_slice, f_slice, init_params, fit_halfwidth
            )
            valid_fits_count += 1
            
        except Exception as e:
            print(f"    [optimized_fit_jpa]   Fit failed: {e}")
            continue

        # Compute symmetry measure of the residuals (correct MATLAB logic):
        #   symlist(j) = (residuals(j) - residuals(end-j+1))^2
        #   symcheck(i) = mean(symlist)
        res = np.asarray(residuals).flatten()
        nres = len(res)

        # Use up to fit_halfwidth pairs, but never beyond half the residuals
        nhalf = min(fit_halfwidth, nres // 2)
        if nhalf <= 0:
            continue

        diffs = []
        for j in range(nhalf):
            # j in Python is 0-based; MATLAB j=1 maps to index 0
            left = res[j]
            right = res[-(j + 1)]
            diffs.append((left - right) ** 2)

        sym_val = float(np.mean(diffs))
        symcheck[jj] = sym_val

    # Pick width that minimizes symcheck
    valid = np.isfinite(symcheck)
    if not np.any(valid):
        raise RuntimeError("[optimized_fit_jpa]: all candidate fits failed")

    best_idx = int(np.argmin(symcheck[valid]))
    # Map best_idx back to original index in width_list
    valid_indices = np.where(valid)[0]
    chosen_j = valid_indices[best_idx]
    chosen_halfwidth = width_list[chosen_j]

    # Final fit with chosen_halfwidth
    start = max(0, peak_idx - chosen_halfwidth)
    stop = min(npts - 1, peak_idx + chosen_halfwidth)

    i_slice = i_data[start:stop + 1]
    q_slice = q_data[start:stop + 1]
    f_slice = freq[start:stop + 1]

    # CORRECT: Only unpack 4 values from cavity_fit
    fit_params, best_mse, residuals, peak_idx_slice = cavity_fit(
        "tx", i_slice, q_slice, f_slice, init_params, chosen_halfwidth
    )

    # Data range is just (start, stop) since we defined it
    data_range = (start, stop)

    return fit_params, best_mse, data_range

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