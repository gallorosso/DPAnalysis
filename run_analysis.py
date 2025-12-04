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
from src.dark_photon.plotting import plot_tx_summary, plot_rfl_summary
from src.dark_photon.plotting.styles import apply_default_style
from src.dark_photon.plotting.jpa import plot_jpa_gain_profiles, plot_jpa_summary
from src.dark_photon.plotting.spectrum import plot_align_norm, plot_spectrum_diagnostics, plot_squeezing_calibration

import numpy as np

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

    # Final scaleinfo from ScaleinfoMergeStage
    scaleinfo = results["scaleinfo"]

    if 'Cavity_Q' in scaleinfo:
        cav_q = scaleinfo['Cavity_Q']
        print(f"Cavity_Q type: {type(cav_q)}, length: {len(cav_q) if hasattr(cav_q, '__len__') else 'scalar'}")
        print(f"Cavity_Q first few values: {cav_q[:5] if len(cav_q) > 5 else cav_q}")
        
    if 'coupling_factor' in scaleinfo:
        beta = scaleinfo['coupling_factor']
        print(f"coupling_factor type: {type(beta)}, length: {len(beta) if hasattr(beta, '__len__') else 'scalar'}")
        print(f"coupling_factor first few values: {beta[:5] if len(beta) > 5 else beta}")

    # # Check if spectrum info was loaded
    # if "spectrum_info" in results:
    #     spectrum_res = results["spectrum_info"]
    #     print(f"SpectrumInfoStage status: {spectrum_res.status}")
    #     # print(f"SpectrumInfoStage updates keys: {list(spectrum_res.scaleinfo_updates.keys())}")
        
    #     # Check a specific field
    #     if "pr_height" in spectrum_res.scaleinfo_updates:
    #         pr_heights = spectrum_res.scaleinfo_updates["pr_height"]
    #         valid = sum(1 for x in pr_heights if x != -1.0 and not np.isnan(x))
    #         print(f"Valid pr_height values: {valid}/{len(pr_heights)}")
    #     else:
            # print("WARNING: 'pr_height' not found in spectrum results")
    
    # 2. Only generate plots if requested
    if options.plottrue:
        print("  Generating cavity summary plots.")
        plot_dir = context.plot_dir or (options.output_dir / "plots")
        
        # Retrieve results for plotting
        param_res = results["parameter_loading"]
        tx_res = results["transmission_analysis"]
        rfl_res = results["reflection_analysis"]
        jpa_res = results["jpa_analysis"]
        # file_enum = results["file_enumeration"]
        # if jpa_res:
        #     print(f"✓ JPA analysis completed: {len(jpa_res.scaleinfo_updates.get('JPA_mse', []))} datasets")
        # else:
        #     print("✗ JPA analysis failed or not found in results")

        # scaleinfo = param_res.scaleinfo.copy()
        # scaleinfo.update(tx_res.scaleinfo_updates)
        # scaleinfo.update(rfl_res.scaleinfo_updates)
        # scaleinfo.update(jpa_res.scaleinfo_updates)

        # TX summary
        plot_tx_summary(tx_res, scaleinfo, plot_dir, show=False)
        # RFL summary  
        plot_rfl_summary(rfl_res, scaleinfo, plot_dir, show=False)
        # JPA plots
        # plot_jpa_gain_profiles(jpa_res.scaleinfo_updates, scaleinfo, file_enum.files, plot_dir, show=False)
        plot_jpa_summary(jpa_res.scaleinfo_updates, scaleinfo, plot_dir, show=False)
        plot_align_norm(scaleinfo, plot_dir)
        plot_spectrum_diagnostics(scaleinfo, plot_dir)
        plot_squeezing_calibration(scaleinfo, plot_dir)

    
    print("=" * 60)
    print("Pipeline execution completed!")
    
    # Access and display results
    init_results = results['initialization']
    print(f"✓ Plot directory ready: {init_results.plot_dir}")
    print(f"✓ Measurement directory ready: {init_results.meas_dir}")
    print(f"✓ Form factor data: {'Loaded' if init_results.form_fac_data is not None else 'Not available'}")

if __name__ == "__main__":
    main()