"""
Helper functions for file operations.
"""

from pathlib import Path
from typing import List, Optional
import re
import warnings

def find_files_by_pattern(directory: Path, file_type: str, search_string: Optional[str] = None) -> List[Path]:
    """
    Python equivalent of MATLAB's filenamelist function.
    
    Finds files matching pattern and returns base names without type suffix.
    
    Args:
        directory: Directory to search in
        file_type: File type pattern (e.g., 'tx2', 'par')
        search_string: Optional string to filter filenames
        
    Returns:
        List of file base names (without type suffix), naturally sorted
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Build search pattern
    if search_string:
        pattern = f"*{search_string}*{file_type}*.mat"
    else:
        pattern = f"*{file_type}*.mat"
    
    # Find matching files
    matching_files = list(directory.glob(pattern))
    
    if not matching_files:
        return []
    
    # Remove the type suffix from filenames
    base_names = []
    for file_path in matching_files:
        # Remove the "type.mat" suffix while keeping the rest of the filename
        filename = file_path.name
        base_filename = filename.replace(f"{file_type}.mat", "")
        base_path = directory / base_filename
        base_names.append(base_path)
    
    # Natural sort (like natsortfiles in MATLAB)
    base_names = natural_sort(base_names)
    
    return base_names

def natural_sort(file_paths: List[Path]) -> List[Path]:
    """
    Natural sort implementation similar to MATLAB's natsortfiles.
    
    Sorts files in a way that respects numerical order.
    """
    def natural_sort_key(path: Path):
        # Convert numbers in the filename to integers for proper sorting
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', str(path))]
    
    return sorted(file_paths, key=natural_sort_key)