"""
Data loading functions for dark photon analysis.
"""

from pathlib import Path
from typing import List, Optional
import warnings

def load_run_dirs(input_dir: Path, first_data_set: int, last_data_set: int) -> List[Path]:
    """
    Load run directories for the specified date range.
    
    Given the top level input directory and the range of dates,
    loads all subdirectories and returns their full paths.
    
    Args:
        input_dir: Top level directory containing date-named subdirectories
        first_data_set: First dataset to include (YYYYMMDD)
        last_data_set: Last dataset to include (YYYYMMDD)
        
    Returns:
        List of full paths to directories within the date range
        
    Raises:
        FileNotFoundError: If input_dir doesn't exist
        ValueError: If date range is invalid
    """
    # Input validation
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if last_data_set < first_data_set:
        raise ValueError(f"Invalid date range: {last_data_set} < {first_data_set}")
    
    # Get all subdirectories (excluding '.' and '..')
    try:
        subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name not in {'.', '..'}]
    except PermissionError as e:
        raise PermissionError(f"Permission denied accessing {input_dir}: {e}")
    
    filenames = []
    
    for subdir in subdirs:
        # Try to parse directory name as date
        try:
            date_num = int(subdir.name)
        except ValueError:
            # Skip directories that aren't date-named
            warnings.warn(
                f"Skipping directory with non-date name: {subdir.name}",
                UserWarning,
                stacklevel=2
            )
            continue
        
        # Check if date is within range
        if date_num < first_data_set or date_num > last_data_set:
            continue
        
        # Add to results (using full path)
        filenames.append(subdir.resolve())
    
    # Sort by date for consistent ordering
    filenames.sort()
    
    # Warn if no directories found
    if not filenames:
        warnings.warn(
            f"No directories found in date range {first_data_set} to {last_data_set} in {input_dir}",
            UserWarning,
            stacklevel=2
        )
    
    return filenames


def load_run_dirs_from_options(options) -> List[Path]:
    """
    Load run directories using parameters from DataRunOptions.
    
    This is a convenience wrapper around load_run_dirs that extracts
    the necessary parameters from a DataRunOptions instance.
    
    Args:
        options: DataRunOptions instance containing input_dir, first_data_set, and last_data_set
        
    Returns:
        List of full paths to directories within the date range
        
    Example:
        >>> from src.dark_photon.core import DataRunOptions
        >>> options = DataRunOptions()
        >>> directories = load_run_dirs_from_options(options)
    """
    return load_run_dirs(options.input_dir, options.first_data_set, options.last_data_set)