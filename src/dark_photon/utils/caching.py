"""
Caching utilities for fit results.
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def get_fit_cache_path(file_path: Path, fit_type: str, output_dir: Path) -> Path:
    """
    Generate cache file path for fit results.
    
    Args:
        file_path: Original data file path
        fit_type: 'tx' or 'rfl' 
        output_dir: Base output directory
        
    Returns:
        Path to cache file
    """
    fit_dir = output_dir / 'fits' / fit_type
    cache_file = fit_dir / f"{file_path.stem}_fit.pkl"
    return cache_file


def load_cached_fit(cache_path: Path, proc_par: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load cached fit results if they exist and are valid.
    
    Args:
        cache_path: Path to cache file
        proc_par: Processing parameters to validate cache
        
    Returns:
        Cached fit data or None if cache invalid/missing
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if cache is valid (same fitting parameters)
        cache_valid = True
        
        if 'tx_fit_width_sigma' in cached_data:
            cache_valid &= (cached_data['tx_fit_width_sigma'] == 
                          proc_par.get('tx_fit_width_sigma'))
        
        if 'rfl_fit_width_sigma' in cached_data:
            cache_valid &= (cached_data['rfl_fit_width_sigma'] == 
                          proc_par.get('rfl_fit_width_sigma'))
        
        if cache_valid:
            return cached_data
            
    except Exception as e:
        print(f"Warning: Could not load cache file {cache_path}: {e}")
    
    return None


def save_cached_fit(cache_path: Path, fit_data: Dict[str, Any]):
    """
    Save fit results to cache.
    
    Args:
        cache_path: Path to cache file
        fit_data: Fit results to cache
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(fit_data, f)
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_path}: {e}")