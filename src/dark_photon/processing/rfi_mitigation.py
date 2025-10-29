from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings

@dataclass
class RFICut:
    """Individual RFI cut specification."""
    type: str  # "rf_spikes", "defcuts", "injections"
    frequencies: List[float]
    width: float
    date_ranges: List[Dict[str, int]] = field(default_factory=list)

@dataclass
class RFIMitigation:
    """RFI mitigation parameters and cuts."""
    
    # General parameters
    RFcutsigma: float = 10000000.0
    IFcutsigma: float = 4.5
    wormcut_width: float = 0.0002  # GHz
    
    # Phase-specific cuts
    cuts: List[RFICut] = field(default_factory=list)
    
    def get_cuts_for_dates(self, start_date: int, end_date: int) -> List[RFICut]:
        """Get RFI cuts applicable for the given date range."""
        applicable_cuts = []
        
        for cut in self.cuts:
            # If no date ranges specified, apply to all dates
            if not cut.date_ranges:
                applicable_cuts.append(cut)
                continue
            
            # Check if any date range includes our target range
            for date_range in cut.date_ranges:
                range_start = date_range.get('start', 0)
                range_end = date_range.get('end', 99999999)
                
                if (start_date >= range_start and end_date <= range_end):
                    applicable_cuts.append(cut)
                    break
        
        return applicable_cuts