"""
Pipeline controller and factory functions.
"""

from .base import PipelineContext, PipelineStage
from typing import List, Dict, Any
import warnings

class AnalysisPipeline:
    """
    Main pipeline controller that orchestrates all stages.
    
    Executes stages in sequence and handles validation and error reporting.
    """
    
    def __init__(self, name: str = "AnalysisPipeline"):
        """
        Initialize the pipeline.
        
        Args:
            name: Descriptive name for this pipeline instance
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        
    def add_stage(self, stage: PipelineStage):
        """
        Add a stage to the pipeline.
        
        Args:
            stage: PipelineStage instance to add
        """
        self.stages.append(stage)
        
    def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Execute all pipeline stages in sequence.
        
        Args:
            context: Pipeline context with configuration and options
            
        Returns:
            Dictionary containing results from all stages
            
        Raises:
            RuntimeError: If any stage fails validation
        """
        data = {}
        
        print(f"Starting pipeline: {self.name}")
        print(f"Output directory: {context.output_dir}")
        print("-" * 50)
        
        for i, stage in enumerate(self.stages):
            stage_name = stage.__class__.__name__
            print(f"Stage {i+1}/{len(self.stages)}: {stage_name}")
            
            try:
                # Execute the stage
                data = stage.execute(context, data)
                
                # Validate the output
                if not stage.validate_output(data):
                    raise RuntimeError(f"Stage {stage_name} produced invalid output")
                    
                print(f"  ✓ {stage_name} completed successfully")
                
            except Exception as e:
                print(f"  ✗ {stage_name} failed: {e}")
                raise RuntimeError(f"Pipeline failed at stage {stage_name}: {e}")
        
        print("-" * 50)
        print(f"Pipeline completed successfully: {self.name}")
        return data

    def __repr__(self):
        stage_names = [stage.__class__.__name__ for stage in self.stages]
        return f"AnalysisPipeline(name='{self.name}', stages={stage_names})"


# Modify the create_preprocessing_pipeline function:
def create_preprocessing_pipeline(avg_type: str = 'average') -> AnalysisPipeline:
    """
    Factory function to create the preprocessing pipeline.
    
    Args:
        avg_type: How to average sweeps ('average', 'before', 'after')
        
    Returns:
        AnalysisPipeline configured for data preprocessing
    """
    pipeline = AnalysisPipeline(name="DataPreprocessingPipeline")
    
    # Add stages in execution order
    from .stages import (
        InitializationStage, 
        FileEnumerationStage,
        ParameterLoadingStage,
        TransmissionAnalysisStage,
        ReflectionAnalysisStage,
    )
    
    pipeline.add_stage(InitializationStage())
    pipeline.add_stage(FileEnumerationStage())
    pipeline.add_stage(ParameterLoadingStage())
    pipeline.add_stage(TransmissionAnalysisStage(avg_type=avg_type))
    pipeline.add_stage(ReflectionAnalysisStage(avg_type=avg_type))
    
    return pipeline