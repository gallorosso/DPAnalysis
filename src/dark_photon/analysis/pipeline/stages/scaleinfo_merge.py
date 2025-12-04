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
)


class ScaleinfoMergeStage(PipelineStage):
    """
    Stage: merge scaleinfo and apply freqs_from_par overrides.

    Produces a single, final `scaleinfo` dict in the pipeline `data`,
    matching the evolving `scaleinfo` variable in the MATLAB code.
    """

    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Get all stage results
        par_res = data.get("parameter_loading")
        tx_res = data.get("transmission_analysis") 
        rfl_res = data.get("reflection_analysis")
        jpa_res = data.get("jpa_analysis")
        
        # 2) Start with parameter scaleinfo
        scaleinfo = dict(par_res.scaleinfo) if par_res else {}
        
        # 3) Merge ALL updates (not just TX and RFL)
        if tx_res and tx_res.scaleinfo_updates:
            scaleinfo.update(tx_res.scaleinfo_updates)
        if rfl_res and rfl_res.scaleinfo_updates:
            scaleinfo.update(rfl_res.scaleinfo_updates)
        if jpa_res and jpa_res.scaleinfo_updates:
            scaleinfo.update(jpa_res.scaleinfo_updates)
        
        # 4) Add derived parameters like Cavity_Q, coupling_factor
        scaleinfo = self._add_cavity_parameters(scaleinfo)
        
        # 5) Apply freqs_from_par override if requested
        if getattr(context.options, "freqs_from_par", False):
            scaleinfo = self._apply_freqs_from_par_override(scaleinfo)
        
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
