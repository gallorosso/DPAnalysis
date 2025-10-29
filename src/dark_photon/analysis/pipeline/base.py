"""
Base classes for the analysis pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

@dataclass
class PipelineContext:
    """
    Shared context for pipeline execution.
    
    This object is passed through all pipeline stages and contains
    shared configuration and state.
    """
    options: Any  # DataRunOptions
    run_props: Any  # RunProperties
    output_dir: Path
    plot_dir: Optional[Path] = None
    meas_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.output_dir = Path(self.output_dir)
        if self.plot_dir:
            self.plot_dir = Path(self.plot_dir)
        if self.meas_dir:
            self.meas_dir = Path(self.meas_dir)


class PipelineStage(ABC):
    """
    Base class for all pipeline stages.
    
    Each stage should implement the execute method and optionally
    override the validate_output method for custom validation.
    """
    
    @abstractmethod
    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this stage and return updated data dictionary.
        
        Args:
            context: Shared pipeline context
            data: Dictionary containing results from previous stages
            
        Returns:
            Updated data dictionary with this stage's results
        """
        pass
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate stage outputs.
        
        Override in subclasses for stage-specific validation.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"