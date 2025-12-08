"""
Scaleinfo merge stage for the analysis pipeline.

This stage:
  * Starts from par-based scaleinfo (ParameterLoadingStage)
  * Merges in TX/RFL fit updates
  * Optionally applies the freqs_from_par override

It is the Python equivalent of the block in AxionAutoRunMain.m that runs
AFTER ReadOutCavityTran and readoutbeta:

    if option.freqs_from_par
        scaleinfo.txparams(:,2)  = (Cavity_freq_tx2 + Cavity_freq_tx1)/2;
        scaleinfo.rflparams(:,2) = (Cavity_freq_rfl2 + Cavity_freq_rfl1)/2;
        scaleinfo.freq_beta(:,1) = (Cavity_freq_rfl2 + Cavity_freq_rfl1)/2;

        scaleinfo.rfldriftkHz    = abs(Cavity_freq_rfl1 - Cavity_freq_rfl2)*1e6;
        scaleinfo.txdriftkHz_fit = scaleinfo.txdriftkHz;
        scaleinfo.txdriftkHz     = abs(Cavity_freq_tx1  - Cavity_freq_tx2 )*1e6;
    end
"""

from typing import Any, Dict

import numpy as np

from ..base import PipelineStage, PipelineContext
from ..results import (
    ParameterLoadingResult,
    TransmissionAnalysisResult,
    ReflectionAnalysisResult,
    JPAGainAnalysisResult,
)


class ScaleinfoMergeStage(PipelineStage):
    """
    Stage: merge scaleinfo and apply freqs_from_par overrides.

    Produces a single, final `scaleinfo` dict in the pipeline `data`,
    matching the evolving `scaleinfo` variable in the MATLAB code.
    """

    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        print("  Merging scaleinfo and applying frequency overrides (if enabled).")

        # 1) Get prerequisite stage results (including JPA)
        par_res: ParameterLoadingResult = data.get("parameter_loading")
        tx_res: TransmissionAnalysisResult = data.get("transmission_analysis")
        rfl_res: ReflectionAnalysisResult = data.get("reflection_analysis")
        # --- ADD THIS LINE ---
        jpa_res: JPAGainAnalysisResult = data.get("jpa_analysis") 

        if par_res is None or not par_res.scaleinfo:
            raise ValueError("ScaleinfoMergeStage requires parameter_loading.scaleinfo")

        # Start from par-based scaleinfo (copy so we don't mutate the result object)
        scaleinfo: Dict[str, Any] = dict(par_res.scaleinfo)

        # 2) Merge in TX and RFL scaleinfo_updates (fit results)
        if tx_res is not None and tx_res.scaleinfo_updates:
            for k, v in tx_res.scaleinfo_updates.items():
                scaleinfo[k] = v

        if rfl_res is not None and rfl_res.scaleinfo_updates:
            for k, v in rfl_res.scaleinfo_updates.items():
                scaleinfo[k] = v
                
        # --- ADD THIS BLOCK FOR JPA ---
        if jpa_res is not None and jpa_res.scaleinfo_updates:
            for k, v in jpa_res.scaleinfo_updates.items():
                scaleinfo[k] = v

        # 3) Apply freqs_from_par override if requested in options
        if getattr(context.options, "freqs_from_par", False):
            scaleinfo = self._apply_freqs_from_par_override(scaleinfo)

        # Store the final, authoritative scaleinfo in the pipeline data
        data["scaleinfo"] = scaleinfo
        return data

    def _apply_freqs_from_par_override(self, scaleinfo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the MATLAB freqs_from_par override logic on a scaleinfo dict.
        """
        print("    Applying freqs_from_par overrides using par-file frequencies.")

        # Convert lists to numpy arrays for numerical operations
        cav_tx1 = np.asarray(scaleinfo["Cavity_freq_tx1"], dtype=float)
        cav_tx2 = np.asarray(scaleinfo["Cavity_freq_tx2"], dtype=float)
        cav_rfl1 = np.asarray(scaleinfo["Cavity_freq_rfl1"], dtype=float)
        cav_rfl2 = np.asarray(scaleinfo["Cavity_freq_rfl2"], dtype=float)

        txparams = np.asarray(scaleinfo["txparams"], dtype=float)    # (N, 5)
        rflparams = np.asarray(scaleinfo["rflparams"], dtype=float)  # (N, 5)
        freq_beta = np.asarray(scaleinfo["freq_beta"], dtype=float)  # (N, 2)

        # Preserve fit-based drift as txdriftkHz_fit, if present
        if "txdriftkHz" in scaleinfo and "txdriftkHz_fit" not in scaleinfo:
            scaleinfo["txdriftkHz_fit"] = scaleinfo["txdriftkHz"]

        # Override central frequencies with par-based averages
        # (column index 1 -> MATLAB column 2)
        txparams[:, 1] = (cav_tx1 + cav_tx2) / 2.0
        rflparams[:, 1] = (cav_rfl1 + cav_rfl2) / 2.0
        freq_beta[:, 0] = (cav_rfl1 + cav_rfl2) / 2.0

        # Recompute drifts from par frequencies (in kHz)
        rfldrift = np.abs(cav_rfl1 - cav_rfl2) * 1e6
        txdrift = np.abs(cav_tx1 - cav_tx2) * 1e6

        # Write back as lists (to match rest of the code)
        scaleinfo["txparams"] = txparams.tolist()
        scaleinfo["rflparams"] = rflparams.tolist()
        scaleinfo["freq_beta"] = freq_beta.tolist()
        scaleinfo["rfldriftkHz"] = rfldrift.tolist()
        scaleinfo["txdriftkHz"] = txdrift.tolist()

        return scaleinfo

    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate merged scaleinfo.
        """
        scaleinfo = data.get("scaleinfo")
        if not scaleinfo:
            print("  ✗ No merged scaleinfo found")
            return False

        required_fields = [
            "txparams",
            "rflparams",
            "freq_beta",
            "txdriftkHz",
            "rfldriftkHz",
        ]
        missing = [f for f in required_fields if f not in scaleinfo]
        if missing:
            print(f"  ✗ Merged scaleinfo missing fields: {missing}")
            return False

        # Optional consistency check with the number of files
        file_enum = data.get("file_enumeration")
        if file_enum and file_enum.files:
            n_files = len(file_enum.files)
            if len(scaleinfo["txparams"]) != n_files or len(scaleinfo["rflparams"]) != n_files:
                print(
                    "  ✗ Mismatch between number of files and parameter rows "
                    f"(txparams: {len(scaleinfo['txparams'])}, "
                    f"rflparams: {len(scaleinfo['rflparams'])}, files: {n_files})"
                )
                return False

        print("  ✓ Scaleinfo merge and overrides validated")
        return True
    
    def _add_cavity_parameters(self, scaleinfo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add Cavity_Q and coupling_factor to scaleinfo.
        
        Extracts these from txparams and freq_beta arrays.
        """
        print("    Adding cavity parameters to scaleinfo...")
        
        # Add Cavity_Q from txparams[:, 2] (Q is 3rd parameter, 0-indexed)
        if "txparams" in scaleinfo:
            txparams = np.asarray(scaleinfo["txparams"])
            if txparams.ndim == 2 and txparams.shape[1] >= 3:
                # Column 2 (0-indexed) is Q
                Cavity_Q = txparams[:, 2].tolist()
                scaleinfo["Cavity_Q"] = Cavity_Q
                print(f"      Added Cavity_Q: {len(Cavity_Q)} values")
            else:
                print(f"      WARNING: txparams has wrong shape: {txparams.shape}")
        
        # Add coupling_factor from freq_beta[:, 1] (beta is 2nd column)
        if "freq_beta" in scaleinfo:
            freq_beta = np.asarray(scaleinfo["freq_beta"])
            if freq_beta.ndim == 2 and freq_beta.shape[1] >= 2:
                # Column 1 (0-indexed) is beta
                coupling_factor = freq_beta[:, 1].tolist()
                scaleinfo["coupling_factor"] = coupling_factor
                print(f"      Added coupling_factor: {len(coupling_factor)} values")
            else:
                print(f"      WARNING: freq_beta has wrong shape: {freq_beta.shape}")
        
        # Debug: Print array lengths to verify
        if "Cavity_Q" in scaleinfo and "coupling_factor" in scaleinfo:
            print(f"      Cavity_Q length: {len(scaleinfo['Cavity_Q'])}, coupling_factor length: {len(scaleinfo['coupling_factor'])}")
        
        return scaleinfo
