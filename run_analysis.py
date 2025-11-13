#!/usr/bin/env python3
"""
Main analysis script for dark photon search data processing.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from src.dark_photon.core import DataRunOptions, load_options_from_yaml
from src.dark_photon.core import Constants
from src.dark_photon.core.run_properties import RunProperties
from src.dark_photon.io import load_run_dirs_from_options
from src.dark_photon.analysis import create_preprocessing_pipeline, PipelineContext

def main():
    # 1. Load base configuration from YAML file
    options = load_options_from_yaml('config/phaseIIc_config.yaml')
    
    # 2. Programmatically override specific settings for this run
    options.first_data_set = 20220908
    options.last_data_set = 20220908
    
    # 3. Load all run properties with runtime overrides
    run_props = RunProperties.from_options(options)
    
    # 4. Load physical constants
    spp = Constants()
    print(f"Using physical constant hbar*c = {spp.hbar_c} GeV*cm")
    
    # 5. Load dataset directories
    datasetdirs = load_run_dirs_from_options(options)
    print(f"Loaded {len(datasetdirs)} dataset directories for analysis.")
    
    # 6. Create output directory path
    comments = f"Worm{options.use_worm}Base{options.skip_baseline}_lam{run_props.system.lam}"
    output_dir_full = Path(options.output_dir) / f"{options.first_data_set}to{options.last_data_set}_{comments}"
    
    # 7. Create and execute the preprocessing pipeline
    print("Starting Dark Photon Analysis Pipeline")
    print("=" * 60)
    
    context = PipelineContext(
        options=options,
        run_props=run_props,
        output_dir=output_dir_full
    )
    
    pipeline = create_preprocessing_pipeline()
    results = pipeline.execute(context)
    # Access merged scaleinfo
    # scaleinfo_final = results['scaleinfo_merge'].scaleinfo

    # # Example: print how many tuning steps we have
    # print(f"✓ Final scaleinfo has {len(scaleinfo_final['txparams'])} tuning steps")

    
    print("=" * 60)
    print("Pipeline execution completed!")
    
    # Access and display results
    init_results = results['initialization']
    print(f"✓ Plot directory ready: {init_results.plot_dir}")
    print(f"✓ Measurement directory ready: {init_results.meas_dir}")
    print(f"✓ Form factor data: {'Loaded' if init_results.form_fac_data is not None else 'Not available'}")

if __name__ == "__main__":
    main()